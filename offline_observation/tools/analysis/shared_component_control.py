"""Cross-group heldout relative correction x local Huber-distance reweighting.

Frozen V32/WS population, zero training and no reference-dependent prediction.
The saved Q64 physical input is offline; this does not claim causal deployment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from association.evaluation.fusion_controls import attribute_errors, ordinary_fusion
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.query_contribution import query_contribution, recover_group_states
from tools.analysis import query_contribution_control as qc

old = qc.old
IDEA = 'IDEA-SHARED-COMPONENT-CONTROL-20261029'
RADIUS = 6371000.
HUBER_DELTA = 1000.
HUBER_STEPS = 5
METHODS = ('WS_MEAN', 'WS_HUBER', 'WS_FIXED_MEAN', 'WS_FIXED_HUBER',
           'WS_SYMMETRIC_MEAN', 'WS_SYMMETRIC_HUBER', 'WS_ANCHOR', 'WS_NEAREST_SUPPORTED')
COMPARISONS = tuple(('WS_MEAN', m) for m in METHODS[1:]) + (
    ('WS_FIXED_MEAN', 'WS_FIXED_HUBER'), ('WS_HUBER', 'WS_FIXED_HUBER'),
    ('WS_SYMMETRIC_MEAN', 'WS_SYMMETRIC_HUBER'),
    ('WS_FIXED_MEAN', 'WS_SYMMETRIC_MEAN'), ('WS_FIXED_HUBER', 'WS_SYMMETRIC_HUBER')) + tuple(
    (base, candidate) for base in ('WS_ANCHOR', 'WS_NEAREST_SUPPORTED')
    for candidate in METHODS[1:6])


def spherical_shift(states, offset_ne):
    """Apply a constant tangent displacement at each position; zero is exact."""
    result = np.asarray(states, np.float64).copy()
    offset = np.asarray(offset_ne, np.float64)
    norm = float(np.linalg.norm(offset))
    if norm == 0.:
        return result
    lat, lon = np.deg2rad(result[..., 0]), np.deg2rad(result[..., 1])
    p = np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)
    north = np.stack((-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)), axis=-1)
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    direction = (offset[0]*north+offset[1]*east)/norm
    moved = np.cos(norm/RADIUS)*p+np.sin(norm/RADIUS)*direction
    result[..., 0] = np.rad2deg(np.arctan2(moved[..., 2], np.hypot(moved[..., 0], moved[..., 1])))
    result[..., 1] = np.rad2deg(np.arctan2(moved[..., 1], moved[..., 0]))
    return result


def heldout_centers(states, support, sources, groups):
    """Use other group folds only, including their chart latitude, to fit centers."""
    order = sorted(set(sources.tolist()))
    if not 1 <= len(order) <= 2:
        raise ValueError('one or two visible sources required; no hidden subset selection')
    g = len(groups)
    degrees = np.full((g, 2), np.nan)
    observations = []
    for group in groups:
        pairs = []
        if len(order) == 2:
            left = [node for node in group if sources[node] == order[0]]
            right = [node for node in group if sources[node] == order[1]]
            for a in left:
                for b in right:
                    visible = support[a] & support[b]
                    delta = states[b, visible, :2]-states[a, visible, :2]
                    delta[:, 1] = (delta[:, 1]+180.) % 360.-180.
                    if len(delta):
                        pairs.append(delta)
        observations.append(np.concatenate(pairs) if pairs else np.empty((0, 2)))
    for gid, values in enumerate(observations):
        if len(values):
            degrees[gid] = values.mean(0)
    folds = np.arange(g) % 3
    centers, chart_latitudes, counts = np.zeros((3, 2)), np.zeros(3), np.zeros(3, np.int64)
    fit_eligible = np.isfinite(degrees).all(1)
    for fold in range(3):
        train = (folds != fold) & fit_eligible
        counts[fold] = int(train.sum())
        if counts[fold] < 2:
            continue
        train_nodes = np.concatenate([np.asarray(group, np.int64) for gid, group in enumerate(groups) if train[gid]])
        visible_lat = states[train_nodes, :, 0][support[train_nodes]]
        lat0 = float(visible_lat.mean())
        if abs(lat0) > 90. or abs(np.cos(np.deg2rad(lat0))) < 1e-8:
            raise ValueError('training chart latitude is outside the supported local chart')
        chart_latitudes[fold] = lat0
        scale = RADIUS*np.pi/180.*np.array([1., np.cos(np.deg2rad(lat0))])
        centers[fold] = np.median(degrees[train]*scale, axis=0)
    before, after = [], []
    undefined_groups = undefined_queries = defined_queries = 0
    for gid, values in enumerate(observations):
        if not len(values):
            continue
        fold = folds[gid]
        if counts[fold] < 2:
            # Zero correction is a fallback, not an observed chart latitude.
            undefined_groups += 1
            undefined_queries += len(values)
            continue
        scale = RADIUS*np.pi/180.*np.array([1., np.cos(np.deg2rad(chart_latitudes[fold]))])
        vector = values*scale
        defined_queries += len(values)
        before.append(float(np.linalg.norm(vector, axis=1).mean()))
        after.append(float(np.linalg.norm(vector-centers[fold], axis=1).mean()))
    audit = dict(fold_training_group_counts=counts.tolist(), fold_correction_enabled=(counts >= 2).tolist(),
        fold_center_norm_m=np.linalg.norm(centers, axis=1).tolist(),
        eligible_group_count=int(fit_eligible.sum()), no_shared_support_group_count=int((~fit_eligible).sum()),
        cross_source_supported_pair_queries=int(sum(map(len, observations))),
        relative_residual_defined_group_count=len(before),
        relative_residual_defined_pair_queries=int(defined_queries),
        relative_residual_undefined_group_count=int(undefined_groups),
        relative_residual_undefined_pair_queries=int(undefined_queries),
        input_only_group_equal_relative_residual_before=float(np.mean(before)) if before else None,
        input_only_group_equal_relative_residual_after=float(np.mean(after)) if after else None,
        relative_residual_definition='input equirectangular source delta minus heldout center; not post-shift state disagreement',
        same_source_members_are_not_independent_measurements=True)
    private = dict(fold_center_ne_m=centers, fold_chart_latitude=chart_latitudes,
                   group_mean_delta_degrees=degrees, group_fit_eligible=fit_eligible,
                   group_fold=folds, source_order=np.asarray(order))
    return centers, order, audit, private


def reweighted_mean(states, nearest, support, groups):
    first = np.asarray([group[0] for group in groups], np.int64)
    mean = ordinary_fusion(states, nearest, support, groups)['predictions']['mean_supported'][first]
    center = mean.copy()
    weights = support.astype(np.float64)
    for _ in range(HUBER_STEPS):
        for gid, group in enumerate(groups):
            members = np.asarray(group, np.int64)
            distance = attribute_errors(states[members], center[gid])['position']
            weights[members] = support[members]*np.minimum(1., HUBER_DELTA/np.maximum(distance, 1e-12))
        updated = recover_group_states(states, weights, groups)['group_states']
        for gid, group in enumerate(groups):
            members = np.asarray(group, np.int64)
            positive = support[members]
            maximum = weights[members].max(0)
            minimum = np.where(positive, weights[members], np.inf).min(0)
            equal = (~positive.any(0)) | (maximum-minimum <= 1e-12)
            # An equal factor changes no physical contribution and must not
            # manufacture a coordinate difference through numerical geometry.
            updated[gid, equal] = mean[gid, equal]
        updated[..., 2:] = mean[..., 2:]
        center = updated
    return center, weights


def construct_predictions(states, nearest, support, sources, groups):
    """Predict with explicit physical inputs only; no identity/reference accepted."""
    states = np.asarray(states, np.float64)
    nearest, support = np.asarray(nearest), np.asarray(support)
    sources = np.asarray(sources)
    if (states.ndim != 3 or states.shape[1:] != (64, 4) or not np.isfinite(states).all()
            or sources.shape != (len(states),) or any(not isinstance(v, str) or not v for v in sources)):
        raise ValueError('complete finite Q64 states and one source string per node required')
    # Historical operator validates support, nearest and the complete partition.
    ordinary = ordinary_fusion(states, nearest, support, groups)
    first = np.asarray([group[0] for group in groups], np.int64)
    predictions = dict(WS_MEAN=ordinary['predictions']['mean_supported'][first],
                       WS_ANCHOR=states[first].copy())
    predictions['WS_NEAREST_SUPPORTED'] = query_contribution(
        states, nearest, support, sources, groups, method='nearest_supported')['group_states']
    centers, order, audit, private = heldout_centers(states, support, sources, groups)
    weights = {}
    predictions['WS_HUBER'], weights['WS_HUBER'] = reweighted_mean(states, nearest, support, groups)
    for gauge in ('FIXED', 'SYMMETRIC'):
        corrected = states.copy()
        for gid, group in enumerate(groups):
            center = centers[gid % 3]
            for source in order:
                members = np.asarray([n for n in group if sources[n] == source], np.int64)
                factor = (-1. if source == order[-1] else 0.) if gauge == 'FIXED' else (
                    -.5 if source == order[-1] else .5)
                if len(members) and len(order) == 2:
                    corrected[members] = spherical_shift(states[members], center*factor)
        mean_name, huber_name = 'WS_'+gauge+'_MEAN', 'WS_'+gauge+'_HUBER'
        predictions[mean_name] = ordinary_fusion(corrected, nearest, support, groups)['predictions']['mean_supported'][first]
        predictions[huber_name], weights[huber_name] = reweighted_mean(corrected, nearest, support, groups)
    for name, mass in weights.items():
        audit[name+'_downweighted_node_queries'] = int(((mass < 1.-1e-12) & support).sum())
        audit[name+'_nonuniform_supported_group_queries'] = int(sum(
            np.sum(np.where(support[group], mass[group], np.inf).min(0) < mass[group].max(0)-1e-12)
            for group in groups))
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise FloatingPointError('nonfinite shared-component prediction')
    return predictions, audit, dict(private, **{name+'_query_weights': value for name, value in weights.items()})


def analyze_scene(fragment, memberships, retained, public, *, imported_artifacts=None, replay_source='115'):
    states, nearest, support = (fragment[k] for k in ('native_states', 'nearest_seconds', 'support'))
    groups = old.groups_from_membership(memberships['WS'], len(states))
    if imported_artifacts is None:
        predictions, audit, private = construct_predictions(states, nearest, support, fragment['sources'], groups)
    else:
        if set(imported_artifacts) != set(METHODS):
            raise ValueError('complete original shared-component prediction arms required')
        predictions = {name: imported_artifacts[name]['group_states'] for name in METHODS}
        audit, private = public['shared_component_audit'], {}
    old._same(predictions['WS_MEAN'], retained['WS']['group_states'], replay_source+' exact WS Mean states')
    old._same(retained['WS']['node_group_indices'], memberships['WS'], replay_source+' WS membership')
    # All prediction operators above have now finished. Truth is only bound below.
    targets, identities = fragment['target_identity_ids'], fragment['identities']
    saved = retained['WS']
    old._same(saved['target_ids'], targets, replay_source+' complete target registry')
    truth, valid = saved['aligned_truth'][:len(targets)], saved['aligned_truth_valid'][:len(targets)]
    registry = {identity: row for row, identity in enumerate(targets)}
    if len(registry) != len(targets):
        raise ValueError('duplicate target registry')
    for node, identity in enumerate(identities):
        old._same(fragment['target'][node], truth[registry[identity]], '056/'+replay_source+' full target')
        old._same(fragment['target_valid'][node], valid[registry[identity]], '056/'+replay_source+' full target valid')
    regions = old.fixed_regions(fragment, targets)
    artifacts, methods, arrays = {}, {}, {}
    for name in METHODS:
        outputs = emit_group_outputs(predictions[name], groups, node_count=len(states))
        report = old.evaluate_full_scene(outputs, node_identities=identities, target_identities=targets,
            target_states=truth, target_valid=valid, node_observed=support, split='V_select',
            grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=support, nearest_seconds=nearest))
        artifact = report.pop('artifact')
        if imported_artifacts is not None:
            if set(artifact) != set(imported_artifacts[name]):
                raise ValueError('imported shared-component artifact schema differs')
            for key in artifact:
                old._same(artifact[key], imported_artifacts[name][key], 'original shared artifact/'+name+'/'+key)
        if name == 'WS_MEAN':
            if set(artifact) != set(saved):
                raise ValueError(replay_source+' Mean artifact schema differs')
            for key in saved:
                old._same(artifact[key], saved[key], replay_source+' exact WS_MEAN/'+key)
        else:
            for key in ('aligned_prediction_valid', 'aligned_group_indices', 'node_group_indices',
                        'group_anchor_target_indices', 'group_valid', 'group_node_indices', 'group_offsets'):
                old._same(artifact[key], artifacts['WS_MEAN'][key], 'fixed WS output invariant '+name+'/'+key)
        methods[name], arrays[name] = qc.describe(report, artifact, fragment, regions)
        if name != 'WS_MEAN' and (methods[name]['population'] != methods['WS_MEAN']['population']
                or methods[name]['unmatched_target_count'] != methods['WS_MEAN']['unmatched_target_count']
                or any(methods[name]['position']['all'][k] != methods['WS_MEAN']['position']['all'][k]
                       for k in old.COUNT_KEYS)):
            raise ValueError('fixed WS population/coverage changed')
        artifacts[name] = artifact
    pairs, pair_arrays = {}, {}
    for base, candidate in COMPARISONS:
        key = candidate+'_vs_'+base
        pairs[key], pair_arrays[key], _ = old.compare_pair(artifacts[base], artifacts[candidate], regions)
        pairs[key].update(baseline_method=base, candidate_method=candidate,
                         side_labels=dict(A0=base, candidate=candidate))
    row = dict(scene_id=public['scene_id'], component_id=public['component_id'], node_count=len(states),
        fixed_R_population_invariants_verified=True,
        methods=methods, pairwise=pairs, shared_component_audit=audit)
    row['original'+replay_source+'_exact_replay_verified'] = True
    return row, dict(methods=arrays, pairs=pair_arrays), artifacts, private


def run(args):
    source, previous, control, output = (Path(getattr(args, key)).resolve()
        for key in ('source_system', 'partition_run', 'control_run', 'output'))
    for protected in (source.parent, previous, control):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new output must not overlap immutable inputs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be 1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), source_system=str(source), partition_run=str(previous),
        control_run=str(control), methods=list(METHODS), cpu_threads=args.cpu_threads, completed_scenes=0,
        model_forwards=0, models_constructed=0, optimizer_updates=0, gpu_used=False,
        scores_recomputed=False, groups_recomputed=False, threshold_reselected=False, normalizer_refitted=False,
        archive_read=False, csv_read=False, cached_reference_read=False, test_opened=False,
        correction_folds=3, minimum_training_groups=2, huber_delta_m=HUBER_DELTA, huber_steps=HUBER_STEPS)
    def checkpoint(**extra):
        state.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes(),
                     advisory_exceeded=time.monotonic()-started > 600, **extra)
        old.frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > old.frozen.RAM_BUDGET:
            raise MemoryError('4GiB resident-memory budget exceeded')
    result = None
    try:
        checkpoint()
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([old.frozen.MAX_AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
            state['address_space_limit_bytes'] = limit
            if hasattr(os, 'sched_getaffinity'):
                cpus = sorted(os.sched_getaffinity(0))[:args.cpu_threads]
                os.sched_setaffinity(0, cpus)
                state['cpu_affinity'] = cpus
        import torch
        torch.set_num_threads(args.cpu_threads)
        meta = qc.metadata(source, previous, control, old.frozen.read(args.manifest))
        checkpoint(scene_ids=meta['ids'], total_scenes=len(meta['ids']))
        (output/'scenes').mkdir()
        (output/'private').mkdir()
        rows, arrays = [], []
        with np.load(meta['association']/'final-scores.npz', allow_pickle=False) as scores:
            old._same(scores['scene_ids'], meta['all_ids'], '056 score scene order')
            for sid in meta['ids']:
                record = old.frozen.load_record(scores, sid, str(meta['components'][sid]))
                fragment, _ = old._load_arrays(source, sid)
                state['cached_reference_read'] = True
                for key in ('node_ids', 'sources'):
                    old._same(fragment[key], record[key], 'original056 '+key)
                private114 = old.frozen.read(previous/'private'/f'scene-{sid}.json.gz')
                memberships = {r: np.asarray(private114['memberships'][r]) for r in qc.PARTITIONS}
                old._same(memberships['A0'], fragment['learned_assoc_membership'], '056 A0 membership')
                with np.load(control/'private'/f'states-{sid}.npz', allow_pickle=False) as saved:
                    keys = {key[3:] for key in saved.files if key.startswith('A0_')}
                    retained = {r: {key: saved[r+'_'+key] for key in keys} for r in qc.PARTITIONS}
                public = old.frozen.read(control/'scenes'/f'scene-{sid}.json.gz')
                if (public['scene_id'] != sid or public['component_id'] != str(meta['components'][sid])
                        or public['node_count'] != len(record['node_ids']) or not public['A0_exact_replay_verified']):
                    raise ValueError('115 scene/source binding differs')
                row, errors, artifacts, private = analyze_scene(fragment, memberships, retained, public)
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                payload = {name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()}
                payload.update(private)
                np.savez_compressed(output/'private'/f'states-{sid}.npz', **payload)
                rows.append(row)
                arrays.append(errors)
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0',
            **qc.aggregate(rows, arrays, method_names=METHODS),
            components={cid: qc.aggregate([row for row in rows if row['component_id'] == cid],
                [a for row, a in zip(rows, arrays) if row['component_id'] == cid], method_names=METHODS)
                for cid in sorted({row['component_id'] for row in rows})},
            original115_exact_replays_verified=True, fixed_R_population_invariants_verified=True,
            no_expansion=True, position_only=True, cached_reference_read=True,
            limitations=['repeated development V32, not independent generalization or credible peer comparison',
                'relative source difference is not identified physical bias or absolute reference truth',
                'fixed-low-source and symmetric gauges are declared controls, never reference-selected',
                'five Huber-distance reweighting steps are not a claim of exact spherical Huber minimization',
                'fixed partition and first-anchor correspondence; state changes cannot remove duplicates',
                'cached Q64 interpolation and all-scene grouping are offline; no causal end-to-end claim'])
        old.frozen.write(output/'scene-summary.json.gz', rows)
        checkpoint(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        old.frozen.write(output/'summary.json', result)
        checkpoint(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=old.frozen.peak_rss_bytes(), finished_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'run.json', state)
        if result is not None and (output/'summary.json').exists():
            result.update(status=state['status'], error=state['error'])
            old.frozen.write(output/'summary.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-system', 'partition-run', 'control-run', 'manifest', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        return 1
    print(f'COMPLETED: {args.output}/summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
