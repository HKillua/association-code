"""Frozen O/MC action scoring on the original V32; no training or T evaluation."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation import action_conditioned_risk as action
from association.evaluation import history_candidates as history
from association.evaluation.local_source_evidence import predict_residual
from tools.analysis import association_context_pilot as p
from tools.analysis.action_conditioned_risk_pilot import ARMS, METHODS
from tools.analysis.drift_memory_pilot import METHODS as CONTROL_METHODS

PRIMARY = tuple(values[b] + '_NN_vs_' + values[a] + '_NN'
                for values in ARMS.values()
                for b, a in ((7, 6), (9, 7), (9, 1), (9, 6), (9, 8), (8, 6)))
EXPECTED_POPULATION = {
    'scene_count': 32, 'component_count': 9, 'target_count': 1427,
    'query_count': 64, 'reference_points': 72196,
    'HARD': {'points': 71396, 'covered_targets': 1401, 'duplicate_groups': 203,
             'eligible_points': 784},
    'STRUCT': {'points': 70775, 'covered_targets': 1380, 'duplicate_groups': 104,
               'eligible_points': 655},
}
SPLIT = 'V_select'
SCORE_ATOL = 1e-4
POSITION_ATOL = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_inputs(parent, control, manifest, parent_run):
    """Check immutable lineage using metadata only, without opening T arrays."""
    require(parent['status'] == control['status'] == 'COMPLETED'
            and parent['exit_code'] == control['exit_code'] == 0,
            'completed original 178 and 191 runs required')
    require(parent['idea_id'] == 'IDEA-HISTORY-CANDIDATE-ACTIONS-178'
            and control['idea_id'] == 'IDEA-MEAN-VELOCITY-VALIDATION-191',
            'original 178/191 controls required')
    require(Path(control['parent_run']).resolve() == Path(parent_run).resolve(),
            '191 parent binding')
    for key in ('native_run', 'manifest'):
        require(Path(parent[key]).resolve() == Path(control[key]).resolve(),
                '191/178 input lineage: ' + key)
    ids = parent['scene_ids']
    require(len(ids) == len(set(ids)) == EXPECTED_POPULATION['scene_count']
            and ids == control['scene_ids'] and set(ids).issubset(manifest['splits'][SPLIT]),
            'original ordered V32 scene population')
    require(control['evaluation_splits'] == [SPLIT]
            and control['training_scene_ids'] == []
            and control['completed_V_scenes'] == len(ids), '191 V-only completion')
    require(len(parent['training_scene_ids']) == 32
            and len(parent['fit_training_scene_ids']) == 128
            and parent['fit_training_scene_ids'] == manifest['splits']['T_core'][:128],
            '178 original T128 full-model metadata')
    require(len({str(manifest['components'][sid]) for sid in ids})
            == EXPECTED_POPULATION['component_count'], 'original V9 components')
    require(tuple(control['methods']) == CONTROL_METHODS, '191 exact 28-arm control')
    return list(ids)


def node_mask(groups, group_mask, shape):
    mask = np.zeros(shape, bool)
    for gid, nodes in enumerate(groups):
        mask[np.ix_(nodes, np.flatnonzero(group_mask[gid]))] = True
    return mask


def clipping_statistics(original, changed, mask, model):
    """Matched raw node-query rows; group/query denominators are reported separately."""
    before, after = original[mask], changed[mask]
    delta = after - before
    result = dict(rows=len(before), changed_feature_cells=int((delta != 0).sum()),
                  changed_feature_rows=int((delta != 0).any(1).sum()),
                  feature_max_abs_delta=(np.max(np.abs(delta), axis=0)
                                         if len(delta) else np.zeros(12)).tolist())
    for name, normalizer in (('residual', model), ('quadratic', model['base'])):
        old = np.abs((before - normalizer['mean']) / normalizer['std']) > 8
        new = np.abs((after - normalizer['mean']) / normalizer['std']) > 8
        old_rows, new_rows = old.any(1), new.any(1)
        result[name] = dict(
            original_clipped_cells=int(old.sum()), original_clipped_rows=int(old_rows.sum()),
            action_clipped_cells=int(new.sum()), action_clipped_rows=int(new_rows.sum()),
            newly_clipped_cells=int((new & ~old).sum()),
            newly_clipped_rows=int((new_rows & ~old_rows).sum()),
            shared_clipped_cells=int((new & old).sum()),
            shared_clipped_rows=int((new_rows & old_rows).sum()),
            original_clipped_by_feature=old.sum(0).tolist(),
            action_clipped_by_feature=new.sum(0).tolist())
    return result


def main_eligible_mask(eligible, artifact):
    """Reference-dependent diagnostic only; call after prediction files are frozen."""
    result = np.zeros_like(eligible)
    n = len(artifact['target_ids'])
    valid = (artifact['aligned_truth_valid'][:n, :, :2].all(-1)
             & artifact['aligned_prediction_valid'][:n, :, :2].all(-1))
    for target, gid in enumerate(artifact['aligned_group_indices'][:n]):
        if gid >= 0:
            result[int(gid)] = eligible[int(gid)] & valid[target]
    return result


def validate_population(result, rows, eligible_counts):
    expected = EXPECTED_POPULATION
    all_rows = result['all_components']
    require(len(rows) == all_rows['scene_count'] == expected['scene_count'], 'V scene count')
    require(len(result['by_component']) == expected['component_count'], 'V component count')
    require(set(all_rows['methods']) == set(METHODS), 'complete 40-arm summary')
    require(tuple(result['primary_bootstrap']) == PRIMARY, 'fixed 12 primary comparisons')
    for front, arms in ARMS.items():
        bound = expected[front]
        require(eligible_counts[front] == bound['eligible_points'], front + ' main eligible count')
        for prefix in arms:
            for suffix in ('_MEAN', '_NN'):
                item = all_rows['methods'][prefix + suffix]
                pop, pos = item['population'], item['position']
                require(pop['target_identity_count'] == expected['target_count']
                        and pos['available_target_count'] == expected['reference_points']
                        and pos['intersection_count'] == bound['points']
                        and pos['covered_target_trajectory_count'] == bound['covered_targets']
                        and pop['anchor_matched_target_count'] == bound['covered_targets']
                        and pop['duplicate_prediction_group_count'] == bound['duplicate_groups'],
                        prefix + suffix + ' frozen V population/coverage')


def run(a):
    require(a.time_policy == 'advisory', 'advisory time policy required')
    old = p.common.old.frozen
    out = Path(a.output)
    for frozen in (Path(a.parent_run).resolve(), Path(a.control_run).resolve()):
        require(out.resolve() != frozen and frozen not in out.resolve().parents,
                'new output must be outside frozen input runs')
    out.mkdir(exist_ok=False)
    (out / 'private').mkdir()
    (out / 'scenes').mkdir()
    started = time.monotonic()
    state = dict(
        status='RUNNING', idea_id='IDEA-ACTION-RISK-V32-20261102', plan_version='1.1',
        mode='action_conditioned_risk_validation', parent_run=str(a.parent_run),
        control_run=str(a.control_run), training_scene_ids=[], scene_ids=[],
        evaluation_splits=[SPLIT], methods=METHODS, primary_comparisons=PRIMARY,
        V_scenes=0, completed_T_scenes=0, completed_V_scenes=0, T_reevaluation_calls=0,
        optimizer_updates=0, neural_head_fits=0, analytic_head_fits=0,
        normalizer_fits=0, native_likelihood_fits=0, teacher_forward_calls=0,
        frontend_optimizer_updates=0, native_mean_computations=0,
        frozen_risk_forward_calls=0, original_score_binding_forward_calls=0,
        test_opened=False, reference_used_for_prediction=False,
        all_rules_frozen_before_V=True, time_policy='advisory',
        started_at=datetime.now(timezone.utc).isoformat())

    def check(*, enforce_memory=True, **kw):
        state.update(kw, elapsed_seconds=time.monotonic() - started,
                     peak_rss_bytes=old.peak_rss_bytes())
        old.write(out / 'run.json', state)
        if enforce_memory and state['peak_rss_bytes'] > 8 * 1024**3:
            raise MemoryError('RSS8GiB')

    try:
        check(phase='input_binding')
        torch.set_num_threads(4)
        parent = old.read(Path(a.parent_run) / 'run.json')
        control = old.read(Path(a.control_run) / 'run.json')
        manifest = old.read(Path(parent['manifest']))
        ids = validate_inputs(parent, control, manifest, a.parent_run)
        model_path = Path(a.parent_run) / 'private/model-full.pt'
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        require(model['dimension'] == 12 and len(model['base']['beta']) == 91,
                '178 full model architecture')
        for normalizer in (model, model['base']):
            require(np.shape(normalizer['mean']) == np.shape(normalizer['std']) == (12,)
                    and np.isfinite(normalizer['mean']).all()
                    and np.isfinite(normalizer['std']).all()
                    and (np.asarray(normalizer['std']) > 0).all(), 'finite original normalizer')
        check(native_run=parent['native_run'], manifest=parent['manifest'],
              model_path=str(model_path), scene_ids=ids, V_scenes=len(ids),
              risk_model_training_scene_count=128)
        rows, arrays, diagnostics = [], [], []
        eligible_counts = {front: 0 for front in ARMS}
        max_original_score_gap = 0.
        for sid in ids:
            native = Path(parent['native_run']) / 'private'
            m = p.common.load_npz(native / 'native-motion' / SPLIT / f'scene-{sid}.npz')
            events = p.common.load_npz(native / 'native-events' / SPLIT / f'scene-{sid}.npz')
            for key in ('node_ids', 'query_seconds'):
                p.common.old._same(m[key], events[key], 'native ' + key)
            offsets = np.asarray(events['event_offsets'])
            require(offsets.ndim == 1 and len(offsets) == len(m['states']) + 1
                    and np.issubdtype(offsets.dtype, np.integer) and offsets[0] == 0
                    and offsets[-1] == len(events['event_times_seconds'])
                    and (np.diff(offsets) >= 2).all(), 'complete native event offsets')
            p.common.old._same(m['native_count'], np.diff(offsets), 'native counts')
            require(len(m['query_seconds']) == EXPECTED_POPULATION['query_count'], 'Q64')
            for start, stop in zip(offsets[:-1], offsets[1:]):
                require((np.diff(events['event_times_seconds'][start:stop]) > 0).all(),
                        'strict native event times')
            op = p.common.load_npz(Path(a.parent_run) / 'private' / f'policy-{SPLIT}-{sid}.npz')
            q = p.common.load_npz(Path(a.parent_run) / 'private' / f'partitions-{SPLIT}-{sid}.npz')
            parts = p.common.load_npz(Path(a.control_run) / 'private' / f'partitions-{SPLIT}-{sid}.npz')
            raw = p.common.load_npz(Path(a.control_run) / 'private' / f'raw-{SPLIT}-{sid}.npz')
            require(set(raw) == set(CONTROL_METHODS), '191 exact saved raw arms')
            parent_keys = tuple(prefix + suffix for arms in ARMS.values()
                                for prefix in arms[:2] for suffix in ('_MEAN', '_NN'))
            original_raw = p.common.load_npz(Path(a.parent_run) / 'private' / f'raw-{SPLIT}-{sid}.npz', parent_keys)
            for key in parent_keys:
                p.common.old._same(raw[key], original_raw[key], '191/178 raw ' + key)
            for key in ('pairs', 'scores'):
                p.common.old._same(parts[key], q[key], '191/178 partition ' + key)
            mc, velocity, bounds, known, means = action.candidates(m, events)
            state['native_mean_computations'] += means
            groups, policy, masks = {}, {}, {}
            for front, arms in ARMS.items():
                g = p.common.old.groups_from_membership(q[front], len(m['states']))
                for prefix in arms[:7]:
                    p.common.old._same(parts[prefix], q[front], '191 membership ' + prefix)
                    p.common.old._same(parts[prefix + '_scores'], q[front + '_scores'], '191 scores ' + prefix)
                original_x, count = history.descriptors(m, g)
                np.testing.assert_allclose(original_x, op[front + '__features'], rtol=0, atol=POSITION_ATOL)
                p.common.old._same(count, op[front + '__count'], '178 support counts')
                old_score = op[front + '__scores']
                predicted = predict_residual(model, original_x.reshape(-1, 12), 'cpu').reshape(old_score.shape)
                gap = float(np.max(np.abs(predicted - old_score)))
                require(gap < SCORE_ATOL, '178 full-model original V score binding')
                max_original_score_gap = max(max_original_score_gap, gap)
                state['original_score_binding_forward_calls'] += 1
                rebuilt, old_weights = history.readouts(m, g, count, old_score, raw[front + '_NN'])
                np.testing.assert_allclose(rebuilt['O'], raw[arms[1] + '_NN'], rtol=0, atol=POSITION_ATOL)
                p.common.old._same(old_weights['O'], op[arms[1] + '__weights'], '178 O weights')
                x, eligible = action.features(m, g, count, op[front + '__features'], mc, velocity, bounds, known)
                score = predict_residual(model, x.reshape(-1, 12), 'cpu').reshape(old_score.shape)
                state['frozen_risk_forward_calls'] += 1
                score[~known] = old_score[~known]
                outputs, weights = action.readouts(m, g, old_score, score, mc, eligible, raw[arms[1] + '_NN'])
                pool_size = np.zeros(eligible.shape, np.int64)
                for gid, nodes in enumerate(g):
                    for j in np.flatnonzero(eligible[gid]):
                        pool_size[gid, j] = int((old_score[nodes, j] == old_score[nodes, j].min()).sum())
                        w = op[arms[1] + '__weights'][nodes, j]
                        np.testing.assert_allclose(action.spherical_position(mc[nodes, j, :2], w),
                                                   raw[arms[6] + '_NN'][gid, j, :2], rtol=0, atol=POSITION_ATOL)
                p.common.old._same(raw[arms[6] + '_NN'][~eligible], raw[arms[1] + '_NN'][~eligible], 'MC preserved domain')
                p.common.old._same(raw[arms[6] + '_NN'][..., 2:], raw[arms[1] + '_NN'][..., 2:], 'MC speed/heading')
                for prefix, key in zip(arms[7:], ('CF', 'RM', 'JM')):
                    raw[prefix + '_NN'] = outputs[key]
                    raw[prefix + '_MEAN'] = raw[front + '_MEAN'].copy()
                    p.common.old._same(outputs[key][~eligible], raw[arms[1] + '_NN'][~eligible], 'action preserved domain')
                    p.common.old._same(outputs[key][..., 2:], raw[arms[1] + '_NN'][..., 2:], 'action speed/heading')
                    policy[prefix + '__weights'] = weights[key]
                policy.update({front + '__' + k: v for k, v in dict(
                    features=x, scores=score, eligible=eligible, known=known,
                    original_features=op[front + '__features'], original_scores=old_score,
                    original_pool_size=pool_size).items()})
                policy[arms[1] + '__weights'] = op[arms[1] + '__weights']
                masks[front] = node_mask(g, eligible, m['support'].shape)
                for prefix in arms:
                    groups[prefix] = g
                    parts[prefix], parts[prefix + '_scores'] = q[front], q[front + '_scores']
            require(set(raw) == set(METHODS), '40 complete raw arms')
            # No reference-dependent file or diagnostic is read until these are saved.
            np.savez_compressed(out / 'private' / f'raw-{SPLIT}-{sid}.npz', **raw)
            np.savez_compressed(out / 'private' / f'partitions-{SPLIT}-{sid}.npz', **parts)
            np.savez_compressed(out / 'private' / f'policy-{SPLIT}-{sid}.npz', **policy)
            ev = p.common.evaluation(Path(parent['native_run']), SPLIT, sid, m)
            row, arr, artifact = p.evaluate(m, ev, groups, {}, raw, SPLIT, method_names=METHODS)
            original = old.read(Path(a.control_run) / 'scenes' / f'{SPLIT}-{sid}.json.gz')
            for name in CONTROL_METHODS:
                require(row['methods'][name] == original['methods'][name], '191 exact physical report ' + name)
            population_masks = {}
            for front, arms in ARMS.items():
                eligible = policy[front + '__eligible']
                main = main_eligible_mask(eligible, artifact[arms[1] + '_NN'])
                main_nodes = node_mask(groups[front], main, m['support'].shape)
                eligible_counts[front] += int(main.sum())
                population_masks.update({front + '__main_eligible': main,
                                         front + '__raw_node_mask': masks[front],
                                         front + '__main_node_mask': main_nodes})
                item = dict(scene_id=sid, front=front)
                for name, group_mask, mask in (('raw_eligible', eligible, masks[front]),
                                               ('main_eligible', main, main_nodes)):
                    item[name] = dict(group_queries=int(group_mask.sum()), node_queries=int(mask.sum()),
                        clipping=clipping_statistics(op[front + '__features'], policy[front + '__features'], mask, model))
                diagnostics.append(item)
                control_artifact = artifact[arms[1] + '_NN']
                for prefix in arms:
                    for suffix in ('_MEAN', '_NN'):
                        for key in ('aligned_prediction_valid', 'aligned_truth_valid', 'aligned_group_indices'):
                            p.common.old._same(artifact[prefix + suffix][key], control_artifact[key], 'fixed action population ' + key)
            np.savez_compressed(out / 'private' / f'action-feature-population-{SPLIT}-{sid}.npz', **population_masks)
            row.update(scene_id=sid, component_id=str(manifest['components'][sid]), search={},
                association={prefix + '_' + kind: original['association'][front + '_' + kind]
                             for front, arms in ARMS.items() for prefix in arms for kind in ('raw', 'decoded')})
            p.common.save_artifacts(out / 'private' / f'states-{SPLIT}-{sid}.npz', artifact)
            old.write(out / 'scenes' / f'{SPLIT}-{sid}.json.gz', row)
            rows.append(row)
            arrays.append(arr)
            check(phase=SPLIT, completed_V_scenes=len(rows), max_original_score_gap=max_original_score_gap)
        result = p.finish(rows, arrays, METHODS, PRIMARY)
        validate_population(result, rows, eligible_counts)
        result.update(status='COMPLETED', idea_id=state['idea_id'])
        old.write(out / 'summary.json', result)
        old.write(out / 'scene-summary.json.gz', rows)
        old.write(out / 'action-feature-accounting.json', diagnostics)
        check(status='COMPLETED', exit_code=0, eligible_main_points=eligible_counts,
              finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        check(enforce_memory=False, status='FAILED', exit_code=1,
              error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('parent-run', 'control-run', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    run(parser.parse_args())
