"""C118: isolate attribute-level use of newly grouped observations at a frozen partition."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import time

import numpy as np

from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from tools.analysis import association_registered_holdout_fusion as c107
from tools.analysis import association_registered_supplier_exclusion as c104
from tools.analysis.offline_association_tail import read_motion, summary_state, write
from tools.analysis.same_source_swap_pilot import read

C107 = 'LEVEL1-OFFLINE-REGISTERED-HOLDOUT-FUSION-20261112-attempt-1'
C113 = 'LEVEL1-OFFLINE-TWO-ANCHOR-TIME-VALIDATION-20261112-attempt-1'
ARMS = ('MEAN', 'KEEP_ORIGINAL', 'ATTRIBUTE')
REGIONS = ('all', 'membership_changed', 'new_active', 'new_true', 'new_wrong',
           'new_true_only', 'new_only_support', 'position_screened', 'heading_screened')
POSITION_M, HEADING_DEG, EXTRA_AGE_S = 840.336, 45., 120.


def emit(membership, states):
    return [dict(node_indices=nodes.tolist(), states=value)
            for nodes, value in zip(c104.c98.group_nodes(membership), states, strict=True)]


def states_for(motion, partition, support):
    return np.asarray([row['states'] for row in c107.mean_groups(motion, partition, support)]).reshape(-1, 64, 4)


def predict(motion, old, current):
    support = motion['support']
    mean = states_for(motion, current, support)
    groups = c104.c98.group_nodes(current)
    active_new, no_old = np.zeros_like(support), np.zeros((len(groups), 64), bool)
    group_changed = np.zeros(len(groups), bool)
    for gi, group in enumerate(groups):
        is_old = old[group] == old[group[0]]
        active_new[group[~is_old]] = support[group[~is_old]]
        group_changed[gi] = not np.array_equal(group, np.flatnonzero(old == old[group[0]]))
    old_active = support & ~active_new
    if not active_new.any():
        return dict(MEAN=mean, KEEP_ORIGINAL=mean, ATTRIBUTE=mean,
                    active_new=active_new, old_active=old_active, position_used=active_new,
                    heading_used=active_new, no_old=no_old, group_changed=group_changed)
    keep = old_active.copy()
    for gi, group in enumerate(groups):
        no_old[gi] = ~old_active[group].any(axis=0) & active_new[group].any(axis=0)
        keep[group] |= active_new[group] & no_old[gi]
    old_mean = states_for(motion, current, keep)
    pos_mask, head_mask = keep.copy(), keep.copy()
    for gi, group in enumerate(groups):
        if not active_new[group].any():
            continue
        positions = motion['states'][group, :, :2]
        course = motion['states'][group, :, 3]
        distance = c107._distance(positions, old_mean[gi, :, :2])
        difference = np.abs((course - old_mean[gi, :, 3] + 180.) % 360. - 180.)
        nearest = motion['nearest_seconds'][group]
        old_age = np.where(old_active[group], nearest, np.inf).min(axis=0)
        timely = nearest <= old_age + EXTRA_AGE_S
        pos_mask[group] |= active_new[group] & timely & (distance <= POSITION_M)
        head_mask[group] |= active_new[group] & timely & (difference <= HEADING_DEG)
    attribute = mean.copy()
    attribute[..., :2] = states_for(motion, current, pos_mask)[..., :2]
    attribute[..., 3] = states_for(motion, current, head_mask)[..., 3]
    np.testing.assert_array_equal(attribute[..., 2], mean[..., 2])
    for name, mask in (('retained', keep), ('position', pos_mask), ('heading', head_mask)):
        for group in groups:
            np.testing.assert_array_equal(mask[group].any(axis=0), support[group].any(axis=0), err_msg=name)
    return dict(MEAN=mean, KEEP_ORIGINAL=old_mean, ATTRIBUTE=attribute,
                active_new=active_new, old_active=old_active, position_used=pos_mask & active_new,
                heading_used=head_mask & active_new, no_old=no_old, group_changed=group_changed)


def audit_regions(saved, partition, identities, matched):
    groups = c104.c98.group_nodes(partition)
    masks = {key: np.zeros((len(matched), 64), bool) for key in REGIONS}
    masks['all'][:] = True
    count = Counter()
    for ti, gi in enumerate(matched):
        if gi < 0:
            continue
        nodes = groups[gi]
        same = identities[nodes] == identities[nodes[0]]
        newly = saved['active_new'][nodes]
        new_true, new_wrong = newly & same[:, None], newly & ~same[:, None]
        chosen = {key: saved[f'{key}_used'][nodes] for key in ('position', 'heading')}
        masks['membership_changed'][ti] = saved['group_changed'][gi]
        masks['new_active'][ti] = newly.any(axis=0)
        masks['new_true'][ti] = new_true.any(axis=0)
        masks['new_wrong'][ti] = new_wrong.any(axis=0)
        masks['new_true_only'][ti] = masks['new_true'][ti] & ~masks['new_wrong'][ti]
        masks['new_only_support'][ti] = saved['no_old'][gi]
        for key, use in chosen.items():
            masks[f'{key}_screened'][ti] = (newly & ~use).any(axis=0)
            count[f'{key}_true_kept_slots'] += int((use & new_true).sum())
            count[f'{key}_wrong_kept_slots'] += int((use & new_wrong).sum())
            count[f'{key}_true_screened_slots'] += int((new_true & ~use).sum())
            count[f'{key}_wrong_screened_slots'] += int((new_wrong & ~use).sum())
        count['new_true_slots'] += int(new_true.sum())
        count['new_wrong_slots'] += int(new_wrong.sum())
    count.update({f'{key}_target_q': int(value.sum()) for key, value in masks.items()})
    count['new_true_targets'] = int(masks['new_true'].any(axis=1).sum())
    count['new_wrong_targets'] = int(masks['new_wrong'].any(axis=1).sum())
    return masks, dict(count)


def evaluate(motion, current, saved, ev):
    outputs = {arm: emit(current, saved[arm]) for arm in ARMS}
    ids, truth, valid = (ev[key] for key in ('identities', 'target_states', 'target_valid'))
    reports, details = evaluate_fusion_methods(outputs, prediction_speed_units=dict.fromkeys(ARMS, 'kn'),
        node_identities=ids, target_ids=ev['target_ids'], target_states=truth, target_valid=valid,
        support=motion['support'], nearest_seconds=motion['nearest_seconds'], target_speed_unit='mps', split='V_select')
    size = len(truth)
    aligned = {name: row['aligned_group_indices'][:size] for name, row in details.items()}
    for arm in ARMS[1:]:
        np.testing.assert_array_equal(aligned[arm], aligned['MEAN'])
        np.testing.assert_array_equal(details[arm]['aligned_prediction_valid'], details['MEAN']['aligned_prediction_valid'])
    np.testing.assert_array_equal(details['ATTRIBUTE']['aligned_prediction'][..., 2],
                                  details['MEAN']['aligned_prediction'][..., 2])
    masks, counts = audit_regions(saved, current, ids, aligned['MEAN'])
    errors = {arm: c107.feature_errors(row['aligned_prediction'][:size], truth,
                row['aligned_prediction_valid'][:size] & valid) for arm, row in details.items()}
    matrices = {region: {key: np.column_stack([errors[arm][key][mask] for arm in ARMS])
                         for key in c107.FEATURES} for region, mask in masks.items()}
    head = reports['MEAN']
    counts.update(assigned_targets=int((aligned['MEAN'] >= 0).sum()), targets=size,
                  groups=head['predicted_group_count'], duplicates=head['duplicate_prediction_group_count'])
    return dict(counts=counts, paper={arm: reports[arm]['paper_fusion_metrics'] for arm in ARMS}), matrices


def pooled(rows, matrices):
    stats = {}
    counts = Counter()
    for row in rows:
        counts.update(row['counts'])
    for region in REGIONS:
        stats[region] = {}
        for attr in c107.FEATURES:
            values = np.concatenate([row[region][attr] for row in matrices], axis=0)
            common = values[np.isfinite(values).all(axis=1)]
            out = dict(common=len(common), arms={name: c107.attribute_stats(common[:, index])
                       for index, name in enumerate(ARMS)})
            for i, arm in enumerate(ARMS[1:], 1):
                delta = common[:, i] - common[:, 0]
                out[f'{arm}_vs_mean'] = dict(worsened=int((delta > 1.e-8).sum()),
                    improved=int((delta < -1.e-8).sum()), max_harm=float(delta.max()) if len(delta) else None)
            stats[region][attr] = out
    return dict(scenes=len(rows), components=len({row['component'] for row in rows}), counts=dict(counts),
                errors=stats, paper={arm: c107.aggregate_paper_fusion([row['paper'][arm] for row in rows]) for arm in ARMS})


def run(args, start):
    manifest = read(args.manifest)
    recent, isolated, _ = c107.holdout(manifest)
    historical = manifest['splits']['V_select']
    assert len(historical) == 300 and not set(recent) & set(historical)
    base, replacement = args.runs / C107, args.runs / C113
    c98, reg = args.runs / c104.C98, args.runs / c107.C104
    for folder in (base, replacement, c98, reg):
        assert read(folder/'run.json')['status'] == 'COMPLETED'
    assert read(base/'roster.json')['new'] == recent
    index = read(args.runs/c104.c98.DATA/'private/index.json')['V_select']
    assert set(index) == set(historical)
    private = args.output/'private'
    private.mkdir()
    populations = dict(C113=recent, C104_V300=historical)
    for name, roster in populations.items():
        for sid in roster:
            if name == 'C113':
                with np.load(base/'private'/f'{sid}.npz', allow_pickle=False) as saved, \
                     np.load(replacement/'private'/f'{sid}.npz', allow_pickle=False) as changed:
                    motion = {key: saved[key] for key in ('states', 'support', 'nearest_seconds', 'sources',
                                                        'query_seconds', 'node_ids')}
                    old, current, baseline = saved['group_REG'], changed['partition'], changed['prediction']
            else:
                motion = read_motion(index[sid])
                with np.load(c98/'private/V_select'/f'{sid}.npz', allow_pickle=False) as before, \
                     np.load(reg/'private/V_select'/f'{sid}.npz', allow_pickle=False) as after:
                    old, current = before['WS'], after['REG_MIX']
                baseline = None
            predicted = predict(motion, old, current)
            if baseline is not None:
                np.testing.assert_array_equal(predicted['MEAN'], baseline)
            np.savez_compressed(private/f'{sid}.npz', current=current, query=motion['query_seconds'],
                                nodes=motion['node_ids'], **predicted)
    write(args.output/'freeze.json', dict(populations={name: len(rows) for name, rows in populations.items()},
        primary_isolated=len(isolated), raw_archive_reads=0, evaluations_before_prediction=0,
        position_gate_m=POSITION_M, heading_gate_degrees=HEADING_DEG, extra_age_s=EXTRA_AGE_S))
    summary_state(args.output, start, status='RUNNING', phase='evaluation')
    results = {}
    for name, roster in populations.items():
        rows, matrices = [], []
        for sid in roster:
            if name == 'C113':
                with np.load(base/'private'/f'{sid}.npz', allow_pickle=False) as saved:
                    motion = {key: saved[key] for key in ('support', 'nearest_seconds')}
                path, node_key, query_key = base/'private'/f'{sid}-evaluation.npz', 'nodes', 'query'
            else:
                with np.load(index[sid]['motion'], allow_pickle=False) as saved:
                    motion = {key: saved[key] for key in ('support', 'nearest_seconds')}
                path, node_key, query_key = index[sid]['evaluation'], 'node_ids', 'query_seconds'
            with np.load(private/f'{sid}.npz', allow_pickle=False) as saved, np.load(path, allow_pickle=False) as truth:
                np.testing.assert_array_equal(saved['nodes'], truth[node_key])
                np.testing.assert_array_equal(saved['query'], truth[query_key])
                ev = {key: truth[key] for key in ('identities', 'target_ids', 'target_states', 'target_valid')}
                row, metrics = evaluate(motion, saved['current'], saved, ev)
            rows.append(dict(scene=sid, component=manifest['components'][sid], **row))
            matrices.append(metrics)
        total = pooled(rows, matrices)
        by_component = defaultdict(list)
        for i, row in enumerate(rows):
            by_component[row['component']].append(i)
        affected = {comp: indexes for comp, indexes in by_component.items()
                    if any(rows[i]['counts']['new_active_target_q'] for i in indexes)}
        total['affected_components'] = {comp: pooled([rows[i] for i in indexes], [matrices[i] for i in indexes])
                                        for comp, indexes in affected.items()}
        if name == 'C104_V300':
            c113_components = set(results['C113']['affected_components'])
            keep = [i for i, row in enumerate(rows) if row['component'] not in c113_components]
            total['without_c113_affected_components'] = pooled([rows[i] for i in keep], [matrices[i] for i in keep])
        results[name] = total
        with gzip.open(args.output/f'{name}-scenes.json.gz', 'wt') as stream:
            json.dump(rows, stream, separators=(',', ':'), allow_nan=False)
    write(args.output/'summary.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args, start)
    except Exception as error:
        summary_state(args.output, start, status='FAILED', error=repr(error))
        raise
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, model_forwards=0, test_opened=False)


if __name__ == '__main__':
    main()
