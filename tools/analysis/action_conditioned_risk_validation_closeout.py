"""V32-only saved-output, full-model action and fixed-cohort audit.

Native motion, score algebra, ties and position accounting use independent
NumPy calculations. Action feature extraction is shared, and the four-attribute
report uses the existing evaluator; neither is independent physical truth.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation import action_conditioned_risk as action
from association.utils.runtime import write_json_gzip
from tools.analysis.action_conditioned_risk_closeout import native
from tools.analysis.action_conditioned_risk_validation import ARMS, METHODS, PRIMARY
from tools.analysis.audit_shared_control_arrays import position, stats
from tools.analysis.history_followup_closeout import account, center, distance, load, numpy_score, read


EXPECTED_POPULATION = dict(scenes=32, components=9, targets=1427, reference_points=72196)
EXPECTED_COHORTS = {
    'HARD': dict(main_points=71396, eligible_points=784, old_MC_harms=11,
                 old_MC_rescues=70, old_O_front_harms=128, old_MC_front_harms=138),
    'STRUCT': dict(main_points=70775, eligible_points=655, old_MC_harms=0,
                   old_MC_rescues=42, old_O_front_harms=107, old_MC_front_harms=92),
}
LABELS = ('front', 'O', 'MC', 'CF', 'RM', 'JM')
EXTRA_PAIRS = tuple(arms[i] + '_NN_vs_' + arms[1] + '_NN'
                    for arms in ARMS.values() for i in (7, 8))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_memory():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    if rss > 8 * 1024**3:
        raise MemoryError('RSS8GiB')


def validate_scope(run):
    r = read(run / 'run.json')
    assert r['status'] == 'COMPLETED' and r['exit_code'] == 0
    assert r['mode'] == 'action_conditioned_risk_validation'
    assert r['idea_id'] == 'IDEA-ACTION-RISK-V32-20261102'
    assert r['plan_version'] == '1.1'
    assert r['evaluation_splits'] == ['V_select'] and r['training_scene_ids'] == []
    assert r['completed_T_scenes'] == 0 and tuple(r['methods']) == METHODS
    assert len(r['scene_ids']) == len(set(r['scene_ids'])) == EXPECTED_POPULATION['scenes']
    assert r['completed_V_scenes'] == r['V_scenes'] == len(r['scene_ids'])
    assert tuple(r['primary_comparisons']) == PRIMARY
    for key in ('optimizer_updates', 'neural_head_fits', 'analytic_head_fits',
                'native_likelihood_fits', 'normalizer_fits', 'teacher_forward_calls',
                'frontend_optimizer_updates', 'T_reevaluation_calls'):
        assert r[key] == 0, key
    assert r['test_opened'] is False and r['reference_used_for_prediction'] is False
    parent, control = (read(Path(r[k]) / 'run.json') for k in ('parent_run', 'control_run'))
    assert parent['status'] == control['status'] == 'COMPLETED'
    assert parent['exit_code'] == control['exit_code'] == 0
    assert parent['scene_ids'] == control['scene_ids'] == r['scene_ids']
    assert control['evaluation_splits'] == ['V_select'] and control['training_scene_ids'] == []
    assert control['parent_run'] == r['parent_run']
    assert parent['native_run'] == r['native_run'] and parent['manifest'] == r['manifest']
    assert control['native_run'] == r['native_run'] and control['manifest'] == r['manifest']
    expected_model = Path(r['parent_run']) / 'private/model-full.pt'
    assert Path(r['model_path']).resolve() == expected_model.resolve()
    manifest = read(Path(r['manifest']))
    assert len(parent['fit_training_scene_ids']) == 128
    assert parent['fit_training_scene_ids'] == manifest['splits']['T_core'][:128]
    assert len({str(manifest['components'][sid]) for sid in r['scene_ids']}) == EXPECTED_POPULATION['components']
    assert not list((run / 'private').glob('*T_core*'))
    assert not (run / 'T-summary.json').exists()
    return r


def decision_weights(nodes, old_score, mc_score, sources, kind):
    """Reconstruct source -> node -> surviving action mass without producer calls."""
    nodes = np.asarray(nodes, int)
    original = nodes[old_score[nodes] == old_score[nodes].min()]
    pool = original if kind in ('O', 'CF') else nodes
    modes = (0,) if kind == 'O' else (1,) if kind == 'RM' else (0, 1)
    records = [(int(n), k, float(old_score[n] if k == 0 else mc_score[n]))
               for n in pool for k in modes]
    best = min(row[2] for row in records)
    chosen = [row for row in records if row[2] == best]
    classes = {str(sources[n]) for n, _, _ in chosen}
    weights = np.zeros((len(old_score), 2))
    for n, k, _ in chosen:
        own = {v for v, _, _ in chosen if sources[v] == sources[n]}
        surviving = sum(v == n for v, _, _ in chosen)
        weights[n, k] = 1 / len(classes) / len(own) / surviving
    return weights


def reconstruct(run, r, split, sid, front, m, events, groups, count, op, oraw, raw, policy):
    assert split == 'V_select'
    arms = ARMS[front]
    prior = load(Path(r['control_run']) / 'private' / f'raw-V_select-{sid}.npz')
    prior_q = load(Path(r['control_run']) / 'private' / f'partitions-V_select-{sid}.npz')
    q = load(run / 'private' / f'partitions-V_select-{sid}.npz')
    for key in ('node_ids', 'query_seconds'):
        np.testing.assert_array_equal(m[key], events[key])
    np.testing.assert_array_equal(m['native_count'], np.diff(events['event_offsets']))
    for key in ('pairs', 'scores'):
        np.testing.assert_array_equal(q[key], prior_q[key])
    for prefix in arms[:7]:
        for suffix in ('_MEAN', '_NN'):
            np.testing.assert_array_equal(raw[prefix + suffix], prior[prefix + suffix])
        for suffix in ('', '_scores'):
            np.testing.assert_array_equal(q[prefix + suffix], prior_q[prefix + suffix])
    mc, velocity, bounds, known = native(m, events)
    x, eligible = action.features(m, groups, count, op[front + '__features'],
                                  mc, velocity, bounds, known)
    np.testing.assert_allclose(policy[front + '__features'], x, rtol=0, atol=1e-10)
    for key, expected in (('eligible', eligible), ('known', known),
                          ('original_features', op[front + '__features']),
                          ('original_scores', op[front + '__scores'])):
        np.testing.assert_array_equal(policy[front + '__' + key], expected)
    np.testing.assert_array_equal(policy[arms[1] + '__weights'], op[arms[1] + '__weights'])
    model = torch.load(Path(r['model_path']), map_location='cpu', weights_only=False)
    original_scores = numpy_score(model, op[front + '__features'].reshape(-1, 12)).reshape(known.shape)
    original_gap = float(np.max(np.abs(original_scores - op[front + '__scores'])))
    expected = numpy_score(model, x.reshape(-1, 12)).reshape(known.shape)
    expected[~known] = op[front + '__scores'][~known]
    scores = policy[front + '__scores']
    gap = float(np.max(np.abs(expected - scores)))
    assert np.isfinite(scores).all() and max(gap, original_gap) < 1e-4
    raw_node_mask = np.zeros(known.shape, bool)
    pool_sizes = np.zeros(eligible.shape, np.int64)
    for gid, group in enumerate(groups):
        nodes = np.asarray(group, int)
        old = op[front + '__scores'][nodes]
        pool_sizes[gid, eligible[gid]] = (old == old.min(0)).sum(0)[eligible[gid]]
        raw_node_mask[np.ix_(nodes, np.flatnonzero(eligible[gid]))] = True
        for j in np.flatnonzero(eligible[gid]):
            wo = decision_weights(nodes, op[front + '__scores'][:, j], scores[:, j], m['sources'], 'O')[:, 0]
            np.testing.assert_array_equal(wo[nodes], op[arms[1] + '__weights'][nodes, j])
            np.testing.assert_allclose(center(m['states'][nodes, j, :2], wo[nodes]),
                                       raw[arms[1] + '_NN'][gid, j, :2], rtol=0, atol=1e-10)
            np.testing.assert_allclose(center(mc[nodes, j, :2], wo[nodes]),
                                       raw[arms[6] + '_NN'][gid, j, :2], rtol=0, atol=1e-10)
            for prefix, kind in zip(arms[7:], ('CF', 'RM', 'JM')):
                weights = decision_weights(nodes, op[front + '__scores'][:, j], scores[:, j], m['sources'], kind)
                np.testing.assert_array_equal(policy[prefix + '__weights'][nodes, j], weights[nodes])
                nn, mm = np.nonzero(weights)
                values = np.array([m['states'][n, j, :2] if k == 0 else mc[n, j, :2] for n, k in zip(nn, mm)])
                np.testing.assert_allclose(raw[prefix + '_NN'][gid, j, :2],
                                           center(values, weights[nn, mm]), rtol=0, atol=1e-10)
    np.testing.assert_array_equal(pool_sizes, policy[front + '__original_pool_size'])
    fixed_columns = [i for i in range(12) if i not in (6, 7, 8)]
    np.testing.assert_array_equal(x[..., fixed_columns], op[front + '__features'][..., fixed_columns])
    np.testing.assert_array_equal(x[~(raw_node_mask & known)], op[front + '__features'][~(raw_node_mask & known)])
    for prefix in arms[7:]:
        assert policy[prefix + '__weights'].shape == (*known.shape, 2)
        assert not policy[prefix + '__weights'][~raw_node_mask].any()
        assert np.isfinite(policy[prefix + '__weights']).all()
        np.testing.assert_array_equal(raw[prefix + '_NN'][~eligible], oraw[arms[1] + '_NN'][~eligible])
    return eligible, max(gap, original_gap)


def cohort_transition(original_delta, candidate_delta):
    old_harm, new_harm = original_delta > 1000, candidate_delta > 1000
    old_rescue, new_rescue = original_delta < -1000, candidate_delta < -1000
    out = dict(old_harms=int(old_harm.sum()), current_harms=int(new_harm.sum()),
               persistent_harms=int((old_harm & new_harm).sum()),
               resolved_harms=int((old_harm & ~new_harm).sum()),
               newly_introduced_harms=int((~old_harm & new_harm).sum()),
               old_rescues=int(old_rescue.sum()), retained_rescues=int((old_rescue & new_rescue).sum()),
               lost_rescue_threshold=int((old_rescue & ~new_rescue).sum()),
               old_rescues_reversed_to_harm=int((old_rescue & new_harm).sum()),
               old_rescues_now_worse_than_control=int((old_rescue & (candidate_delta > 0)).sum()))
    assert out['old_harms'] == out['persistent_harms'] + out['resolved_harms']
    assert out['current_harms'] == out['persistent_harms'] + out['newly_introduced_harms']
    assert out['old_rescues'] == out['retained_rescues'] + out['lost_rescue_threshold']
    return out


def clipping(original, changed, mask, model):
    a, b = original[mask], changed[mask]
    delta = np.abs(b - a)
    result = dict(rows=len(a), changed_feature_cells=int((delta != 0).sum()),
                  changed_feature_rows=int((delta != 0).any(1).sum()),
                  feature_max_abs_delta=(delta.max(0) if len(a) else np.zeros(12)).tolist())
    for name, normalizer in (('residual', model), ('quadratic', model['base'])):
        before = np.abs((a - normalizer['mean']) / normalizer['std']) > 8
        after = np.abs((b - normalizer['mean']) / normalizer['std']) > 8
        before_row, after_row = before.any(1), after.any(1)
        result[name] = dict(original_clipped_cells=int(before.sum()), original_clipped_rows=int(before_row.sum()),
                            action_clipped_cells=int(after.sum()), action_clipped_rows=int(after_row.sum()),
                            newly_clipped_cells=int((after & ~before).sum()), newly_clipped_rows=int((after_row & ~before_row).sum()),
                            shared_clipped_cells=int((after & before).sum()), shared_clipped_rows=int((after_row & before_row).sum()),
                            original_clipped_by_feature=before.sum(0).tolist(), action_clipped_by_feature=after.sum(0).tolist())
    return result


def row_scope(rows):
    result = dict(points=len(rows), groups=len({(x['scene_id'], x['group_index']) for x in rows}),
                  scenes=len({x['scene_id'] for x in rows}), components=len({x['component_id'] for x in rows}),
                  best_node_sets_disjoint=sum(x['best_node_sets_disjoint'] for x in rows))
    for key in ('O', 'MC', 'CF', 'RM', 'JM', 'oracle_O', 'oracle_MC', 'oracle_joint', 'oracle_fixed'):
        result[key + '_MAE'] = float(np.mean([x[key] for x in rows])) if rows else None
    for key in ('CF', 'RM', 'JM'):
        result[key] = {field: sum(x[key + '_' + field] for x in rows)
                       for field in ('node_changed', 'node_weight_changed', 'outside_original_pool', 'action_tie_mixture', 'fallback_mass_present')}
        result[key].update(MC_mass_mean=float(np.mean([x[key + '_MC_mass'] for x in rows])) if rows else None,
                           harms_vs_O=sum(x[key] - x['O'] > 1000 for x in rows),
                           rescues_vs_O=sum(x[key] - x['O'] < -1000 for x in rows))
    return result


def concentration(rows, candidate, control):
    changed = [x for x in rows if x[candidate] != x[control]]
    by_component = {}
    for row in changed:
        key = row['component_id']
        by_component[key] = by_component.get(key, 0.) + row[candidate] - row[control]
    absolute = sum(abs(x) for x in by_component.values())
    result = dict(changed_points=len(changed), changed_groups=len({(x['scene_id'], x['group_index']) for x in changed}),
                  changed_scenes=len({x['scene_id'] for x in changed}), changed_components=len(by_component),
                  largest_absolute_component_share=max(map(abs, by_component.values())) / absolute if absolute else None)
    for label, pred in (('improvement', lambda x: x[candidate] < x[control]),
                        ('worsening', lambda x: x[candidate] > x[control]),
                        ('harms_over_1km', lambda x: x[candidate] - x[control] > 1000),
                        ('rescues_over_1km', lambda x: x[candidate] - x[control] < -1000)):
        subset = [x for x in changed if pred(x)]
        result[label] = dict(points=len(subset), groups=len({(x['scene_id'], x['group_index']) for x in subset}),
                             scenes=len({x['scene_id'] for x in subset}), components=len({x['component_id'] for x in subset}))
    return result


def add_locations(store, label, mask, groups, sid, component):
    entry = store.setdefault(label, dict(points=0, groups=set(), scenes=set(), components=set()))
    entry['points'] += int(mask.sum())
    if mask.any():
        entry['groups'].update((sid, int(gid)) for gid in np.unique(groups[mask]))
        entry['scenes'].add(sid)
        entry['components'].add(component)


def opportunities(run, r):
    began = time.monotonic()
    manifest = read(Path(r['manifest']))
    model = torch.load(Path(r['model_path']), map_location='cpu', weights_only=False)
    pool = {front: [] for front in ARMS}
    errors = {front: {label: [] for label in LABELS} for front in ARMS}
    counts = {front: dict(raw_eligible=0, unclaimed_or_invalid=0) for front in ARMS}
    tail_locations = {front: {} for front in ARMS}
    population = dict(targets=0, reference_points=0)
    feature_rows = []
    for sid in r['scene_ids']:
        root = Path(r['native_run']) / 'private'
        m = load(root / 'native-motion/V_select' / f'scene-{sid}.npz')
        events = load(root / 'native-events/V_select' / f'scene-{sid}.npz')
        ev = load(root / 'native-evaluation/V_select' / f'scene-{sid}.npz')
        mc, _, _, known = native(m, events)
        q = load(run / 'private' / f'partitions-V_select-{sid}.npz')
        policy = load(run / 'private' / f'policy-V_select-{sid}.npz')
        op = load(Path(r['parent_run']) / 'private' / f'policy-V_select-{sid}.npz')
        raw = load(run / 'private' / f'raw-V_select-{sid}.npz')
        saved = load(run / 'private' / f'states-V_select-{sid}.npz')
        masks = load(run / 'private' / f'action-feature-population-V_select-{sid}.npz')
        registry = {str(t): i for i, t in enumerate(ev['target_ids'])}
        nt = len(registry)
        tv = ev['target_valid'][..., :2].all(-1)
        population['targets'] += nt
        population['reference_points'] += int(tv.sum())
        scene_rows = []
        for front, arms in ARMS.items():
            groups = [np.flatnonzero(q[front] == value) for value in sorted(set(q[front].tolist()))]
            eligible = policy[front + '__eligible']
            counts[front]['raw_eligible'] += int(eligible.sum())
            main_eligible = np.zeros_like(eligible)
            raw_node_mask = np.zeros(m['support'].shape, bool)
            main_node_mask = np.zeros_like(raw_node_mask)
            prefixes = (front, arms[1], arms[6], *arms[7:])
            common = None
            scene_errors = {}
            for label, prefix in zip(LABELS, prefixes):
                valid = saved[prefix + '_NN_aligned_prediction_valid'][:nt, :, :2].all(-1)
                use = tv & valid
                if common is not None:
                    np.testing.assert_array_equal(use, common)
                common = use
                err = position(saved[prefix + '_NN_aligned_prediction'][:nt], ev['target_states'], use)
                scene_errors[label] = err
                errors[front][label].append(err[use])
            gid_grid = np.broadcast_to(saved[arms[1] + '_NN_aligned_group_indices'][:nt, None], common.shape)
            original_MC_harm = (scene_errors['MC'] - scene_errors['O']) > 1000
            original_MC_rescue = (scene_errors['MC'] - scene_errors['O']) < -1000
            original_front_harm = (scene_errors['O'] - scene_errors['front']) > 1000
            for label in ('MC', 'CF', 'RM', 'JM'):
                front_harm = (scene_errors[label] - scene_errors['front']) > 1000
                for suffix, mask in (('old_harms', original_front_harm), ('current_harms', front_harm),
                                     ('persistent_harms', original_front_harm & front_harm),
                                     ('resolved_harms', original_front_harm & ~front_harm),
                                     ('newly_introduced_harms', ~original_front_harm & front_harm)):
                    add_locations(tail_locations[front], label + '_vs_front/' + suffix,
                                  common & mask, gid_grid, sid, str(manifest['components'][sid]))
                if label == 'MC':
                    continue
                new_harm = (scene_errors[label] - scene_errors['O']) > 1000
                new_rescue = (scene_errors[label] - scene_errors['O']) < -1000
                for suffix, mask in (('old_harms', original_MC_harm), ('current_harms', new_harm),
                                     ('persistent_harms', original_MC_harm & new_harm),
                                     ('resolved_harms', original_MC_harm & ~new_harm),
                                     ('newly_introduced_harms', ~original_MC_harm & new_harm),
                                     ('old_rescues', original_MC_rescue),
                                     ('retained_rescues', original_MC_rescue & new_rescue),
                                     ('lost_rescue_threshold', original_MC_rescue & ~new_rescue),
                                     ('old_rescues_reversed_to_harm', original_MC_rescue & new_harm)):
                    add_locations(tail_locations[front], label + '_MC_fixed/' + suffix,
                                  common & mask, gid_grid, sid, str(manifest['components'][sid]))
            claimed = set()
            before = len(pool[front])
            for gid, nodes in enumerate(groups):
                raw_node_mask[np.ix_(nodes, np.flatnonzero(eligible[gid]))] = True
                key = str(ev['identities'][nodes[0]])
                if key not in registry or key in claimed:
                    continue
                claimed.add(key)
                ti = registry[key]
                main_eligible[gid] = eligible[gid] & common[ti]
                main_node_mask[np.ix_(nodes, np.flatnonzero(main_eligible[gid]))] = True
                for j in np.flatnonzero(main_eligible[gid]):
                    target = ev['target_states'][ti, j]
                    eo = distance(m['states'][nodes, j, :2], target)
                    em = distance(mc[nodes, j, :2], target)
                    original_weights = op[arms[1] + '__weights'][nodes, j]
                    chosen = original_weights > 0
                    original = set(nodes[chosen].tolist())
                    assert len(original) == policy[front + '__original_pool_size'][gid, j]
                    row = dict(scene_id=sid, component_id=str(manifest['components'][sid]), front=front,
                               group_index=gid, query_index=int(j), nodes=len(nodes), original_pool_size=len(original),
                               pure=len(set(ev['identities'][nodes].tolist())) == 1,
                               oracle_O=float(eo.min()), oracle_MC=float(em.min()),
                               oracle_joint=float(min(eo.min(), em.min())),
                               oracle_fixed=float(min(eo[chosen].min(), em[chosen].min())),
                               best_node_sets_disjoint=not bool(set(nodes[eo == eo.min()]) & set(nodes[em == em.min()])))
                    for label, prefix in zip(LABELS, prefixes):
                        row[label] = float(distance(raw[prefix + '_NN'][gid, j, :2], target))
                    for label, prefix in zip(('CF', 'RM', 'JM'), arms[7:]):
                        weights = policy[prefix + '__weights'][nodes, j]
                        node_weights = weights.sum(1)
                        selected = set(nodes[node_weights > 0].tolist())
                        row[label + '_node_changed'] = selected != original
                        row[label + '_node_weight_changed'] = not np.array_equal(node_weights, original_weights)
                        row[label + '_outside_original_pool'] = bool(selected - original)
                        row[label + '_MC_mass'] = float(weights[:, 1].sum())
                        row[label + '_action_tie_mixture'] = bool((weights[:, 0] > 0).any() and (weights[:, 1] > 0).any())
                        row[label + '_fallback_mass_present'] = bool((weights[~known[nodes, j], 1] > 0).any())
                    pool[front].append(row)
                    scene_rows.append(row)
            counts[front]['unclaimed_or_invalid'] += int(eligible.sum()) - (len(pool[front]) - before)
            for suffix, expected in (('main_eligible', main_eligible), ('raw_node_mask', raw_node_mask), ('main_node_mask', main_node_mask)):
                np.testing.assert_array_equal(masks[front + '__' + suffix], expected)
            feature_rows.append(dict(scene_id=sid, front=front,
                                     raw_eligible=dict(group_queries=int(eligible.sum()), node_queries=int(raw_node_mask.sum()),
                                                       clipping=clipping(op[front + '__features'], policy[front + '__features'], raw_node_mask, model)),
                                     main_eligible=dict(group_queries=int(main_eligible.sum()), node_queries=int(main_node_mask.sum()),
                                                        clipping=clipping(op[front + '__features'], policy[front + '__features'], main_node_mask, model))))
        dest = run / 'private' / f'opportunity-V_select-{sid}.json.gz'
        if dest.exists():
            raise FileExistsError(dest)
        write_json_gzip(dest, scene_rows)
        check_memory()
    assert population == {key: EXPECTED_POPULATION[key] for key in population}
    assert feature_rows == read(run / 'action-feature-accounting.json')
    public, result_table = {}, []
    summary = read(run / 'summary.json')
    assert set(summary['primary_bootstrap']) == set(PRIMARY)
    for front, rows in pool.items():
        values = {label: np.concatenate(parts) for label, parts in errors[front].items()}
        expected = EXPECTED_COHORTS[front]
        assert len(values['O']) == expected['main_points'] and len(rows) == expected['eligible_points']
        old_delta = values['MC'] - values['O']
        assert int((old_delta > 1000).sum()) == expected['old_MC_harms']
        assert int((old_delta < -1000).sum()) == expected['old_MC_rescues']
        old_front = values['O'] - values['front']
        assert int((old_front > 1000).sum()) == expected['old_O_front_harms']
        assert int((values['MC'] - values['front'] > 1000).sum()) == expected['old_MC_front_harms']
        scopes = {}
        for name, keep in (('all', lambda x: True), ('pure', lambda x: x['pure']), ('mixed', lambda x: not x['pure']),
                           ('single_node', lambda x: x['nodes'] == 1), ('multiple_nodes', lambda x: x['nodes'] > 1),
                           ('single_node_pool', lambda x: x['original_pool_size'] == 1), ('tied_pool', lambda x: x['original_pool_size'] > 1),
                           ('MC_old_harm', lambda x: x['MC'] - x['O'] > 1000), ('MC_old_rescue', lambda x: x['MC'] - x['O'] < -1000)):
            scopes[name] = row_scope([row for row in rows if keep(row)])
        fixed = {label: cohort_transition(old_delta, values[label] - values['O']) for label in ('CF', 'RM', 'JM')}
        front_tail = {label: cohort_transition(old_front, values[label] - values['front']) for label in ('MC', 'CF', 'RM', 'JM')}
        locations = {key: dict(points=value['points'], **{field: len(value[field]) for field in ('groups', 'scenes', 'components')})
                     for key, value in tail_locations[front].items()}
        for label, outcome in fixed.items():
            for key in ('old_harms', 'current_harms', 'persistent_harms', 'resolved_harms', 'newly_introduced_harms',
                        'old_rescues', 'retained_rescues', 'lost_rescue_threshold', 'old_rescues_reversed_to_harm'):
                assert locations[label + '_MC_fixed/' + key]['points'] == outcome[key]
        for label, outcome in front_tail.items():
            for key in ('old_harms', 'current_harms', 'persistent_harms', 'resolved_harms', 'newly_introduced_harms'):
                assert locations[label + '_vs_front/' + key]['points'] == outcome[key]
        pairs = ((7, 6), (9, 7), (9, 1), (9, 6), (9, 8), (8, 6), (7, 1), (8, 1))
        label_by_index = {1: 'O', 6: 'MC', 7: 'CF', 8: 'RM', 9: 'JM'}
        pair_locations = {}
        for ci, bi in pairs:
            candidate, control = label_by_index[ci], label_by_index[bi]
            item = concentration(rows, candidate, control)
            delta = values[candidate] - values[control]
            assert item['harms_over_1km']['points'] == int((delta > 1000).sum())
            assert item['rescues_over_1km']['points'] == int((delta < -1000).sum())
            pair_locations[ARMS[front][ci] + '_NN_vs_' + ARMS[front][bi] + '_NN'] = item
        public[front] = dict(**counts[front], main_points=len(values['O']), scopes=scopes,
                             fixed_MC_vs_O_cohorts=fixed, original_front_harm_cohorts=front_tail,
                             tail_case_counts=locations, paired_case_counts=pair_locations)
        arms = ARMS[front]
        for candidate, control, ci, bi in (('CF', 'MC', 7, 6), ('JM', 'CF', 9, 7)):
            pair = arms[ci] + '_NN_vs_' + arms[bi] + '_NN'
            old_stats, new_stats = stats(values[control]), stats(values[candidate])
            delta = values[candidate] - values[control]
            result_table.append(dict(front=front, comparison=pair, main_points=len(delta),
                                     mean_delta_m=float(delta.mean()),
                                     metric_deltas_m={key: new_stats[key] - old_stats[key] for key in ('mae', 'rmse', 'p95', 'p99')},
                                     bootstrap=summary['primary_bootstrap'][pair],
                                     without_largest_component=summary['without_largest_component']['pairwise'][pair]['common_valid_position'],
                                     new_harms_over_1km=int((delta > 1000).sum()), rescues_over_1km=int((delta < -1000).sum()),
                                     maximum_harm_m=float(max(0., delta.max())), maximum_rescue_m=float(max(0., -delta.min())),
                                     concentration=concentration(rows, candidate, control),
                                     fixed_MC_cohort_outcome=fixed[candidate], front_harm_outcome=front_tail[candidate],
                                     interpretation='descriptive values only; no automatic acceptance or default replacement'))
    report = dict(status='COMPLETED', exit_code=0, population=population, heads=public,
                  elapsed_seconds=time.monotonic() - began,
                  scope='V32 full main population fixed harm/rescue cohorts; opportunity rows only claimed valid eligible points; per-node oracle excludes tied mixtures and is not a deployment lower bound; indexed rows remain private')
    write_new(run / 'action-opportunity-accounting.json', report)
    write_new(run / 'action-validation-results.json', dict(status='COMPLETED', rows=result_table,
              scope='front-specific CF-MC and JM-CF mean, tails and concentration; shared development V, not independent confirmation or calibrated risk'))


def audit(run):
    run = Path(run).resolve()
    started = time.monotonic()
    dest = run / 'action-validation-closeout.json'
    outputs = ('independent-output-audit.json', 'four-attribute-accounting.json', 'history-followup-accounting.json',
               'action-opportunity-accounting.json', 'action-validation-results.json', dest.name)
    for name in outputs:
        if (run / name).exists():
            raise FileExistsError(run / name)
    for path in (run / 'private').glob('opportunity-V_select-*.json.gz'):
        raise FileExistsError(path)
    result = dict(status='RUNNING', run=str(run))
    try:
        torch.set_num_threads(4)
        r = validate_scope(run)
        from tools.analysis.association_context_closeout import closeout
        closeout(run)
        check_memory()
        account(run, run / 'history-followup-accounting.json',
                (ARMS, PRIMARY + EXTRA_PAIRS, reconstruct,
                 dict(fits=0, updates=0, T_scenes=0, V_scenes=EXPECTED_POPULATION['scenes'])))
        check_memory()
        opportunities(run, r)
        result.update(status='COMPLETED', exit_code=0, scope='V-only full-model audit; no training, T reevaluation or test access')
    except BaseException as error:
        result.update(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
        write_new(dest, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    audit(args.run)
