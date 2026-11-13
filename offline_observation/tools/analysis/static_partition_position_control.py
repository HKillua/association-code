"""115: truly recompute ordinary Mean for four frozen114 pilot partitions.

Position-only cached-reference diagnostic; no archive, model or score inference.
Private coordinates/identity/member arrays stay under private/ on the server.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.evaluation.fusion_controls import ordinary_fusion, attribute_errors
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.system_attribution import _same, assert_metrics
from tools.experiments.level1_system_evaluate import common_target_metrics
from tools.analysis import frozen_score_decision_diagnostic as frozen
from tools.analysis import static_partition_pilot as partition
from tools.analysis.compare_association_duration import _load_metadata, _load_arrays
from tools.analysis.learned_group_oracle import _bind_source

IDEA = 'IDEA-STATIC-PARTITION-POSITION-CONTROL-20261029'
PARTITION_RUN = 'LEVEL1-STATIC-PARTITION-PILOT-20261029-114-attempt-1'
METHODS = ('A0', 'WS', 'CL', 'CC')
BASE = 'learned_assoc_mean_supported'
COUNT_KEYS = ('available_target_count', 'available_prediction_count', 'intersection_count',
    'missing_prediction_count', 'prediction_without_target_count', 'available_target_trajectory_count',
    'covered_target_trajectory_count', 'missing_target_trajectory_count')
POPULATION_KEYS = ('target_identity_count', 'predicted_group_count', 'duplicate_prediction_group_count',
    'anchor_matched_target_count', 'reference_unavailable_target_count')


def groups_from_membership(value, n):
    value = np.asarray(value)
    canonical = frozen.canonical_membership(value, n)
    if not np.array_equal(canonical, value):
        raise ValueError('114 membership must retain canonical original-node ordering')
    return [np.flatnonzero(canonical == gid).tolist() for gid in range(int(canonical.max())+1)] if n else []


def construct_states(fragment, memberships):
    """Only physical inputs and saved memberships feed prediction operators."""
    states, support, nearest = (fragment[k] for k in ('native_states', 'support', 'nearest_seconds'))
    groups, predictions = {}, {}
    for name in METHODS:
        groups[name] = groups_from_membership(memberships[name], len(states))
        predictions[name] = ordinary_fusion(states, nearest, support, groups[name])['predictions']['mean_supported']
    return groups, predictions


def stats(values):
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError('nonfinite valid position error')
    return dict(point_count=len(values), mae=float(values.mean()) if len(values) else None,
        mse=float(np.mean(values*values)) if len(values) else None,
        rmse=float(np.sqrt(np.mean(values*values))) if len(values) else None,
        p95=float(np.quantile(values, .95)) if len(values) else None,
        p99=float(np.quantile(values, .99)) if len(values) else None, unit='metres')


def delta_stats(delta):
    delta = np.asarray(delta, np.float64)
    if not np.isfinite(delta).all():
        raise FloatingPointError('nonfinite valid paired error difference')
    better, worse = delta < -1e-9, delta > 1e-9
    return dict(common_point_count=len(delta), mean_error_change=float(delta.mean()) if len(delta) else None,
        signed_error_change_sum=float(delta.sum()), benefit_sum=float(-delta[better].sum()),
        harm_sum=float(delta[worse].sum()), better_point_count=int(better.sum()),
        worse_point_count=int(worse.sum()), tied_point_count=int((~better & ~worse).sum()), unit='metres')


def target_arrays(artifact):
    t = len(artifact['target_ids'])
    truth, valid = artifact['aligned_truth'][:t], artifact['aligned_truth_valid'][:t]
    prediction = artifact['aligned_prediction'][:t]
    predicted_valid = artifact['aligned_prediction_valid'][:t]
    errors = attribute_errors(prediction, truth)['position']
    use = valid[..., :2].all(-1) & predicted_valid[..., :2].all(-1)
    return truth, valid, errors, use


def fixed_regions(fragment, targets):
    registry = {str(target): i for i, target in enumerate(targets)}
    shape = (len(targets), len(fragment['query_seconds']))
    support, nearest = np.zeros(shape, bool), np.full(shape, np.inf)
    for node, identity in enumerate(fragment['identities']):
        row = registry[str(identity)]
        support[row] |= fragment['support'][node]
        nearest[row] = np.minimum(nearest[row], fragment['nearest_seconds'][node])
    return dict(all=np.ones(shape, bool), inside_support=support, outside_support=~support,
        far_gt_1800s=(nearest > 1800) & np.isfinite(nearest), no_input=~np.isfinite(nearest))


def representatives(artifact):
    t = len(artifact['target_ids'])
    first = artifact['group_node_indices'][artifact['group_offsets'][:-1]]
    selected = artifact['aligned_group_indices'][:t]
    result = np.full(t, -1, np.int64)
    use = selected >= 0
    result[use] = first[selected[use]]
    return result


def compare_pair(a, b, regions):
    # Deliberately two arms per call, never the intersection of all four arms.
    reports, common = common_target_metrics(dict(A0=a, candidate=b))
    truth, valid, ae, old = target_arrays(a)
    _, _, be, new = target_arrays(b)
    use = valid[..., :2].all(-1) & common[..., :2].all(-1)
    ra, rb = representatives(a), representatives(b)
    represented = (ra >= 0) & (rb >= 0)
    same, changed = represented & (ra == rb), represented & (ra != rb)
    masks = dict(regions, same_anchor=np.broadcast_to(same[:, None], use.shape),
                 changed_anchor=np.broadcast_to(changed[:, None], use.shape))
    strata, arrays = {}, {}
    for name, mask in masks.items():
        selected = use & mask
        av, bv = ae[selected], be[selected]
        delta = bv-av
        strata[name] = dict(A0=stats(av), candidate=stats(bv), candidate_minus_A0=delta_stats(delta))
        arrays[name] = dict(A0=av, candidate=bv, delta=delta)
    gained, lost = new & ~old, old & ~new
    arrays['_gained'], arrays['_lost'] = be[gained], ae[lost]
    row = dict(common_valid_position={name: reports[name]['full_target_metrics']['all']['position']
                                      for name in ('A0', 'candidate')},
        strata=strata, same_anchor_target_count=int(same.sum()), changed_anchor_target_count=int(changed.sum()),
        coverage_transitions=dict(gained_target_points=int(gained.sum()), lost_target_points=int(lost.sum()),
            rescued_target_count=int((new.any(1) & ~old.any(1)).sum()),
            lost_target_count=int((old.any(1) & ~new.any(1)).sum())),
        newly_covered_position=stats(be[gained]), lost_coverage_position=stats(ae[lost]))
    return row, arrays, dict(common_valid=use, gained=gained, lost=lost)


def analyze_scene(record, fragment, saved, memberships, public114, expected):
    """Bind frozen inputs/A0, recompute all states, then evaluate cached truth."""
    n = len(record['node_ids'])
    for key in ('node_ids', 'sources'):
        _same(fragment[key], record[key], '056 association/system '+key)
    if (fragment['native_states'].shape != (n, 64, 4)
            or fragment['query_seconds'].shape != (64,)
            or not np.isfinite(fragment['query_seconds']).all()
            or not (np.diff(fragment['query_seconds']) > 0).all()
            or fragment['exact_observation'].shape != (n, 64)
            or fragment['exact_observation'].dtype != bool):
        raise ValueError('original complete Q64 fragment inputs required')
    raw = frozen.evaluate_scenes([record], frozen.THRESHOLD)
    frozen.counts_match(raw, frozen.metric_rows(public114['raw']), '114 original raw')
    if (public114['scene_id'] != record['scene_id'] or public114['component_id'] != record['component_id']
            or not public114.get('raw_invariance_verified') or not public114.get('original_A0_membership_verified')):
        raise ValueError('114 scene/raw/A0 binding required')
    _same(memberships['A0'], fragment['learned_assoc_membership'], 'A0 original membership')
    _same(frozen.decode(n, record['pairs'], record['scores']), memberships['A0'], 'A0 raw-score reconstruction')
    groups, predictions = construct_states(fragment, memberships)
    _same(np.r_[0, np.cumsum([len(g) for g in groups['A0']])], fragment['learned_assoc_group_offsets'], 'A0 offsets')
    _same(np.asarray([node for group in groups['A0'] for node in group], np.int64),
          fragment['learned_assoc_group_node_indices'], 'A0 node partition')
    _same(predictions['A0'], fragment[BASE], 'A0 exact ordinary Mean prediction')

    # All four partitions and physical predictions are fixed before truth use.
    identities, targets = fragment['identities'], fragment['target_identity_ids']
    if (identities.shape != (n,) or targets.ndim != 1 or len(set(targets.tolist())) != len(targets)
            or any(not isinstance(x, str) or not x for x in [*identities.tolist(), *targets.tolist()])):
        raise ValueError('complete unique target registry and node identities required')
    _same(record['labels'], identities[record['pairs'][:, 0]] == identities[record['pairs'][:, 1]], 'identity labels')
    old = {key[len(BASE)+1:]: value for key, value in saved.items() if key.startswith(BASE+'_')}
    _same(old['target_ids'], targets, 'full target registry')
    t = len(targets)
    truth, valid = old['aligned_truth'][:t], old['aligned_truth_valid'][:t]
    if truth.shape != (t, 64, 4) or valid.shape != truth.shape or valid.dtype != bool:
        raise ValueError('full target reference rows and per-feature validity required')
    _same(saved['ols_anchor_target_ids'], targets, 'second cached method full target registry')
    _same(saved['ols_anchor_aligned_truth'][:t], truth, 'cached method full target reference')
    _same(saved['ols_anchor_aligned_truth_valid'][:t], valid, 'cached method full target validity')
    registry = {identity: row for row, identity in enumerate(targets)}
    for node, identity in enumerate(identities):
        if identity not in registry:
            raise ValueError('node identity absent from complete target registry')
        _same(fragment['target'][node], truth[registry[identity]], 'fragment/full target reference')
        _same(fragment['target_valid'][node], valid[registry[identity]], 'fragment/full target reference validity')
    artifacts, methods, error_arrays = {}, {}, {}
    regions = fixed_regions(fragment, targets)
    for name in METHODS:
        co = np.asarray(memberships[name])[record['pairs'][:, 0]] == np.asarray(memberships[name])[record['pairs'][:, 1]]
        decoded = frozen.evaluate_scenes([dict(record, scores=co.astype(np.float64))], .5)
        frozen.counts_match(decoded, frozen.metric_rows(public114['arms'][name]['decoded']), name+' saved114 decoded')
        outputs = [dict(node_indices=g, states=predictions[name][g[0]]) for g in groups[name]]
        report = evaluate_full_scene(outputs, node_identities=identities, target_identities=targets,
            target_states=truth, target_valid=valid, node_observed=fragment['support'], split='V_select',
            grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
        artifact = report.pop('artifact')
        if name == 'A0':
            if set(artifact) != set(old):
                raise ValueError('A0 full system artifact schema changed')
            for key in artifact:
                _same(artifact[key], old[key], 'A0 artifact/'+key)
            if {key: report[key] for key in POPULATION_KEYS} != expected['population']:
                raise ValueError('A0 original full-target population differs')
            assert_metrics(report['full_target_metrics'], expected['full_target_metrics'], 'A0 original full target')
        artifacts[name] = artifact
        _, _, errors, own = target_arrays(artifact)
        method = dict(population={key: report[key] for key in POPULATION_KEYS},
            unmatched_target_count=len(report['unmatched_target_ids']), position={}, fixed_target_strata={})
        method['actual_group_support_target_grid_counts'] = {
            str(label): int((artifact['aligned_input_support_kind'][:t] == code).sum())
            for code, label in enumerate(artifact['input_support_region_names'])}
        for region in ('all', 'observed', 'missing'):
            mask = np.ones_like(own) if region == 'all' else artifact['aligned_observed'][:t]
            if region == 'missing':
                mask = ~mask
            values = errors[own & mask]
            method['position'][region] = dict(report['full_target_metrics'][region]['position'],
                p95=stats(values)['p95'], p99=stats(values)['p99'])
        error_arrays[name] = {}
        for region, mask in regions.items():
            values = errors[own & mask]
            available = int((valid[..., :2].all(-1) & mask).sum())
            method['fixed_target_strata'][region] = dict(stats(values), available_target_points=available,
                missing_prediction_points=available-len(values), coverage=len(values)/available if available else None)
            error_arrays[name][region] = values
        methods[name] = method
    pairwise, pair_arrays, private_masks = {}, {}, {}
    for name in METHODS[1:]:
        pairwise[name], pair_arrays[name], private_masks[name] = compare_pair(artifacts['A0'], artifacts[name], regions)
    return dict(scene_id=record['scene_id'], component_id=record['component_id'], node_count=n,
        A0_exact_replay_verified=True, raw_and_membership_bindings_verified=True,
        methods=methods, pairwise=pairwise), dict(artifacts=artifacts, pair_masks=private_masks), dict(methods=error_arrays, pairs=pair_arrays)


def pooled_position(rows, errors):
    out = {key: sum(row[key] for row in rows) for key in COUNT_KEYS}
    joined = np.concatenate(errors)
    out.update({key: value for key, value in stats(joined).items() if key != 'point_count'})
    if len(joined) != out['intersection_count']:
        raise ValueError('pooled position error/count population differs')
    out['coverage'] = out['intersection_count']/out['available_target_count'] if out['available_target_count'] else None
    out['trajectory_coverage'] = out['covered_target_trajectory_count']/out['available_target_trajectory_count'] if out['available_target_trajectory_count'] else None
    defined = [row['mae'] for row in rows if row['mae'] is not None]
    out.update(scene_macro_mae=float(np.mean(defined)) if defined else None, defined_mae_scene_count=len(defined), scene_count=len(rows))
    return out


def aggregate(rows, arrays):
    result = dict(scene_count=len(rows), scene_ids=[r['scene_id'] for r in rows], methods={}, pairwise={})
    for name in METHODS:
        values = [row['methods'][name] for row in rows]
        result['methods'][name] = dict(population={key: sum(v['population'][key] for v in values) for key in POPULATION_KEYS},
            unmatched_target_count=sum(v['unmatched_target_count'] for v in values),
            actual_group_support_target_grid_counts={key: sum(v['actual_group_support_target_grid_counts'][key] for v in values)
                for key in values[0]['actual_group_support_target_grid_counts']},
            position=pooled_position([v['position']['all'] for v in values], [x['methods'][name]['all'] for x in arrays]),
            fixed_target_strata={region: dict(stats(np.concatenate([x['methods'][name][region] for x in arrays])),
                available_target_points=sum(v['fixed_target_strata'][region]['available_target_points'] for v in values),
                missing_prediction_points=sum(v['fixed_target_strata'][region]['missing_prediction_points'] for v in values))
                for region in arrays[0]['methods'][name]})
        for region in result['methods'][name]['fixed_target_strata'].values():
            region['coverage'] = region['point_count']/region['available_target_points'] if region['available_target_points'] else None
    for name in METHODS[1:]:
        values = [r['pairwise'][name] for r in rows]
        result['pairwise'][name] = dict(
            coverage_transitions={k: sum(v['coverage_transitions'][k] for v in values) for k in values[0]['coverage_transitions']},
            same_anchor_target_count=sum(v['same_anchor_target_count'] for v in values),
            changed_anchor_target_count=sum(v['changed_anchor_target_count'] for v in values),
            common_valid_position={side: pooled_position([v['common_valid_position'][side] for v in values],
                [x['pairs'][name]['all'][side] for x in arrays]) for side in ('A0', 'candidate')},
            newly_covered_position=stats(np.concatenate([x['pairs'][name]['_gained'] for x in arrays])),
            lost_coverage_position=stats(np.concatenate([x['pairs'][name]['_lost'] for x in arrays])),
            strata={region: dict(
                A0=stats(np.concatenate([x['pairs'][name][region]['A0'] for x in arrays])),
                candidate=stats(np.concatenate([x['pairs'][name][region]['candidate'] for x in arrays])),
                candidate_minus_A0=delta_stats(np.concatenate([x['pairs'][name][region]['delta'] for x in arrays])))
                for region in arrays[0]['pairs'][name] if not region.startswith('_')})
    return result


def metadata(source, previous, manifest):
    source, previous = Path(source).resolve(), Path(previous).resolve()
    association = source.parent/'association'
    _bind_source(source)
    physical = _load_metadata(source, manifest, 1000)
    ids, components, original, expected_raw = frozen.metadata(association, manifest)
    if previous.name != PARTITION_RUN:
        raise ValueError('original114 attempt-1 required')
    run, summary, pilot = (frozen.read(previous/name) for name in ('run.json', 'summary.json', 'pilot-summary.json'))
    selected = partition.pilot_selection(ids)
    if (run.get('status') != 'COMPLETED' or summary.get('status') != 'COMPLETED'
            or run.get('idea_id') != 'IDEA-STATIC-PARTITION-PILOT-20261029'
            or run.get('plan_version') != '1.0' or summary.get('plan_version') != '1.0'
            or run.get('threshold') != frozen.THRESHOLD or run.get('seed') != partition.SEED
            or run.get('methods') != list(partition.METHODS) or run.get('pilot_scene_ids') != selected
            or Path(run.get('source_association', '')).resolve() != association
            or Path(summary.get('source_cache', {}).get('path', '')).resolve() != association/'final-scores.npz'
            or summary.get('expanded') is not True or summary.get('completed_scene_ids') != ids
            or summary.get('analyzed_bindings_scene_count') != len(ids)
            or pilot.get('pilot') != summary.get('pilot') or pilot.get('expansion_gate') != summary.get('expansion_gate')
            or summary.get('pilot', {}).get('scene_ids') != selected
            or any(run.get(k) != 0 for k in ('model_forwards', 'models_constructed', 'optimizer_updates'))
            or any(run.get(k) is not False for k in ('gpu_used', 'scores_recomputed', 'threshold_reselected', 'normalizer_refitted', 'native_read', 'reference_read', 'test_opened'))):
        raise ValueError('completed original114 fixed pilot/source/no-model binding required')
    return dict(ids=selected, components=components, physical=physical, expected_raw=expected_raw,
                original=original, all_ids=ids, association=association)


def run(args):
    source, previous, output = (Path(getattr(args, k)).resolve() for k in ('source_system', 'partition_run', 'output'))
    for protected in (source.parent, previous):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('output must be separate from immutable056/114 inputs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be 1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0', source_system=str(source),
        started_at=datetime.now(timezone.utc).isoformat(), ram_budget_bytes=frozen.RAM_BUDGET,
        partition_run=str(previous), cpu_threads=args.cpu_threads, methods=list(METHODS), completed_scenes=0,
        model_forwards=0, models_constructed=0, optimizer_updates=0, gpu_used=False, csv_read=False,
        archive_read=False, reference_archive_read=False, cached_reference_read=False, normalizer_refitted=False,
        scores_recomputed=False, groups_recomputed=False, test_opened=False)
    frozen.write(output/'run.json', state)
    result = None
    def checkpoint(**extra):
        state.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=frozen.peak_rss_bytes(),
                     advisory_exceeded=time.monotonic()-started > 600, **extra)
        frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > frozen.RAM_BUDGET:
            raise MemoryError('4GiB resident-memory budget exceeded')
    try:
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([frozen.MAX_AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard)); state['address_space_limit_bytes'] = limit
            if hasattr(os, 'sched_getaffinity'):
                cpus = sorted(os.sched_getaffinity(0))[:args.cpu_threads]
                os.sched_setaffinity(0, cpus); state['cpu_affinity'] = cpus
        import torch
        torch.set_num_threads(args.cpu_threads)
        meta = metadata(source, previous, frozen.read(args.manifest))
        checkpoint(scene_ids=meta['ids'], total_scenes=len(meta['ids']))
        (output/'scenes').mkdir(); (output/'private').mkdir()
        rows, arrays = [], []
        index = {sid: i for i, sid in enumerate(meta['all_ids'])}
        cache = meta['association']/'final-scores.npz'
        before = cache.stat()
        with np.load(cache, allow_pickle=False) as saved:
            if saved['scene_ids'].tolist() != meta['all_ids']:
                raise ValueError('original056 score scene order differs')
            for sid in meta['ids']:
                record = frozen.load_record(saved, sid, str(meta['components'][sid]))
                frozen.counts_match(frozen.evaluate_scenes([record], frozen.THRESHOLD),
                    {scope: meta['expected_raw'][scope][index[sid]] for scope in frozen.SCOPES}, 'original056 raw')
                fragment, system = _load_arrays(source, sid)
                state['cached_reference_read'] = True
                public = frozen.read(previous/'scenes'/f'scene-{sid}.json.gz')
                private = frozen.read(previous/'private'/f'scene-{sid}.json.gz')
                memberships = {name: np.asarray(private['memberships'][name]) for name in METHODS}
                _same(memberships['A0'], frozen.canonical_membership(meta['original'][sid], len(record['sources'])), 'original056 A0')
                row, retained, errors = analyze_scene(record, fragment, system, memberships, public,
                    meta['physical']['expected'][sid][BASE])
                frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                payload = {name+'_'+key: value for name, artifact in retained['artifacts'].items() for key, value in artifact.items()}
                payload.update({name+'_'+key: value for name, masks in retained['pair_masks'].items() for key, value in masks.items()})
                np.savez_compressed(output/'private'/f'states-{sid}.npz', **payload)
                rows.append(row); arrays.append(errors)
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        after = cache.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('original056 score cache changed during read-only diagnostic')
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **aggregate(rows, arrays),
            components={cid: aggregate([r for r in rows if r['component_id'] == cid],
                [a for r, a in zip(rows, arrays) if r['component_id'] == cid]) for cid in sorted({r['component_id'] for r in rows})},
            all_A0_exact_replays_verified=True, no_expansion=True, model_forwards=0, optimizer_updates=0,
            position_only=True, cached_reference_read=True,
            limitations=['fixed repeatedly used development V32; no independent generalization',
                'position errors are conditional on own/common valid reference intersections; coverage separately retained',
                'ordinary Mean and first-anchor correspondence fixed; no learned fusion or conditional-state effect',
                'old speed units not interpreted; original reference generation/tail validity remains unknown'])
        frozen.write(output/'scene-summary.json.gz', rows)
        checkpoint(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        frozen.write(output/'summary.json', result)
        checkpoint(status='COMPLETED', exit_code=0)
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=frozen.peak_rss_bytes(), exit_code=1)
        frozen.write(output/'run.json', state)
        if result is not None and (output/'summary.json').exists():
            result.update(status=state['status'], error=state['error']); frozen.write(output/'summary.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-system', 'partition-run', 'manifest', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc(); return 1
    print(f'COMPLETED: {args.output}/summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
