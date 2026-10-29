"""122: held-out motion-dependent structure in input-only shared source deltas.

No state correction, reference/identity array fields, neural training or clock claim.
All signed coefficients, relative vectors and group residual arrays stay private.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
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

from association.evaluation.grouping import _partition
from tools.analysis import static_partition_position_control as old
from tools.analysis.shared_component_control import METHODS as SHARED_METHODS

IDEA = 'IDEA-SHARED-MOTION-TIME-RESIDUAL-20261029'
ORIGINAL_IDEA = 'IDEA-SHARED-COMPONENT-CONTROL-20261029'
SHARED_RUN = 'LEVEL1-SHARED-COMPONENT-20261029-118-attempt-1'
METHODS = ('STATIC_LS', 'TIME_LS', 'JOINT_LS', 'JOINT_PERMUTED_LS')
COMPARISONS = tuple(combinations(METHODS, 2))
PHYSICAL_KEYS = ('native_states', 'support', 'query_seconds', 'sources', 'node_ids')
RCOND, SEED, RADIUS = 1e-8, 20, 6371000.


def group_motion_rows(states, support, query, sources, groups):
    """Read complete physical inputs only; each eligible group contributes one row."""
    states = np.asarray(states, np.float64)
    support, query, sources = np.asarray(support), np.asarray(query, np.float64), np.asarray(sources)
    if (states.ndim != 3 or states.shape[1:] != (64, 4) or not np.isfinite(states).all()
            or (np.abs(states[..., 0]) > 90).any() or (np.abs(states[..., 1]) > 180).any()
            or support.shape != states.shape[:2] or support.dtype != bool
            or query.shape != (64,) or not np.isfinite(query).all() or (np.diff(query) <= 0).any()
            or sources.shape != (len(states),)
            or any(not isinstance(source, str) or not source for source in sources)):
        raise ValueError('finite complete Q64 physical inputs, true seconds and visible sources required')
    groups = _partition(groups, len(states))
    order = sorted(set(sources.tolist()))
    if len(order) > 2:
        raise ValueError('original one/two-source population required')
    g = len(groups)
    delta, velocity = np.full((g, 2), np.nan), np.full((g, 2), np.nan)
    shared_count, derivative_count = np.zeros(g, np.int64), np.zeros(g, np.int64)
    duration = query[2:]-query[:-2]
    for gid, group in enumerate(groups):
        if len(order) < 2:
            continue
        low = [node for node in group if sources[node] == order[0]]
        high = [node for node in group if sources[node] == order[1]]
        deltas, velocities = [], []
        for a in low:
            for b in high:
                joint = support[a] & support[b]
                shared_count[gid] += int(joint.sum())
                usable = joint[:-2] & joint[1:-1] & joint[2:]
                if not usable.any():
                    continue
                d = states[b, 1:-1, :2]-states[a, 1:-1, :2]
                d[:, 1] = (d[:, 1]+180.) % 360.-180.
                speeds = []
                for node in (a, b):
                    difference = states[node, 2:, :2]-states[node, :-2, :2]
                    difference[:, 1] = (difference[:, 1]+180.) % 360.-180.
                    speeds.append(difference/duration[:, None])
                deltas.append(d[usable])
                velocities.append((.5*(speeds[0]+speeds[1]))[usable])
                derivative_count[gid] += int(usable.sum())
        if deltas:
            delta[gid] = np.concatenate(deltas).mean(0)
            velocity[gid] = np.concatenate(velocities).mean(0)
    return dict(delta_degrees=delta, velocity_degrees_per_second=velocity,
                shared_pair_query_count=shared_count, derivative_pair_query_count=derivative_count,
                eligible=np.isfinite(delta).all(1) & np.isfinite(velocity).all(1),
                source_order=np.asarray(order))


def fit_models(delta, velocity, *, joint_only=False):
    """Equal group weights; d=b-v*tau. Rank-deficient time fits fall back to static."""
    d, v = np.asarray(delta, np.float64), np.asarray(velocity, np.float64)
    if d.ndim != 2 or d.shape[1:] != (2,) or v.shape != d.shape or not np.isfinite(d).all() or not np.isfinite(v).all():
        raise ValueError('finite [eligible_groups,2] input deltas and velocities required')
    n = len(d)
    base = d.mean(0) if n else None
    models = {}
    if not joint_only:
        models['STATIC_LS'] = dict(b=base, tau=0. if n else None,
            status='fitted' if n else 'unavailable', reason=None if n else 'no_training_group',
            rank=2 if n else 0, singular_values=[float(np.sqrt(n))]*2 if n else [],
            expected_rank=2, fit_attempted=bool(n))
    for name, joint in ((('JOINT_LS', True),) if joint_only else (('TIME_LS', False), ('JOINT_LS', True))):
        expected = 3 if joint else 1
        model = dict(b=base, tau=None, status='static_fallback' if n else 'unavailable',
                     reason='fewer_than_two_training_groups', rank=0, singular_values=[],
                     expected_rank=expected, fit_attempted=False)
        if n >= 2:
            time_column = -v.reshape(-1, 1)
            if joint:
                design = np.column_stack((np.tile(np.eye(2), (n, 1)), time_column))
            else:
                design = time_column
            coefficient, _, rank, singular = np.linalg.lstsq(design, d.reshape(-1), rcond=RCOND)
            model.update(rank=int(rank), singular_values=singular.tolist(), fit_attempted=True,
                         reason='numerical_rank_deficient')
            if rank == expected:
                model.update(b=coefficient[:2].copy() if joint else np.zeros(2),
                             tau=float(coefficient[-1]), status='fitted', reason=None)
        elif not n:
            model['reason'] = 'no_training_group'
        models[name] = model
    return models


def public_model(model):
    return dict(status=model['status'], reason=model['reason'], numerical_rank=model['rank'],
        expected_rank=model['expected_rank'], singular_values=model['singular_values'],
        spatial_coefficient_norm_m=float(np.linalg.norm(model['b'])) if model['b'] is not None else None,
        time_coefficient_norm_seconds=abs(model['tau']) if model['tau'] is not None else None,
        fit_attempted=model['fit_attempted'])


def residual_stats(value):
    value = np.asarray(value, np.float64)
    if value.ndim != 1 or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError('finite nonnegative group residual norms required')
    return dict(group_count=len(value), mean_m=float(value.mean()) if len(value) else None,
        rmse_m=float(np.sqrt(np.mean(value**2))) if len(value) else None,
        median_m=float(np.median(value)) if len(value) else None,
        p95_m=float(np.quantile(value, .95)) if len(value) else None,
        quantity='norm of input-only group-mean relative residual; not reference error')


def paired_stats(base, candidate):
    a, b = np.asarray(base), np.asarray(candidate)
    if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise ValueError('all models must retain the same held-out residual population')
    use = np.isfinite(a)
    delta = b[use]-a[use]
    return dict(defined_group_count=int(use.sum()),
        candidate_minus_baseline_mean_m=float(delta.mean()) if len(delta) else None,
        candidate_minus_baseline_rmse_m=(float(np.sqrt(np.mean(b[use]**2))-np.sqrt(np.mean(a[use]**2)))
                                       if len(delta) else None),
        improved_groups=int((delta < -1e-9).sum()), worsened_groups=int((delta > 1e-9).sum()),
        tied_groups=int((np.abs(delta) <= 1e-9).sum()))


def analyze_scene(physical, membership, calibration, *, scene_id, component_id):
    """No truth/identity access: explicitly select the physical allowlist."""
    states, support, query, sources = (np.asarray(physical[k]) for k in
                                      ('native_states', 'support', 'query_seconds', 'sources'))
    groups = old.groups_from_membership(membership, len(states))
    rows = group_motion_rows(states, support, query, sources, groups)
    old._same(calibration['source_order'], rows['source_order'], '118 visible source order')
    folds = np.asarray(calibration['group_fold'])
    if folds.dtype.kind not in 'iu' or not np.array_equal(folds, np.arange(len(groups)) % 3):
        raise ValueError('original118 group folds differ')
    errors = {name: np.full(len(groups), np.nan) for name in METHODS}
    private = dict(group_delta_degrees=rows['delta_degrees'],
        group_velocity_degrees_per_second=rows['velocity_degrees_per_second'],
        group_eligible=rows['eligible'], group_fold=folds, source_order=rows['source_order'],
        group_shared_pair_query_count=rows['shared_pair_query_count'],
        group_derivative_pair_query_count=rows['derivative_pair_query_count'],
        fold_chart_latitude=np.full(3, np.nan))
    for name in METHODS:
        private['fold_parameters_'+name] = np.full((3, 3), np.nan)
    rng, public_folds, time_parameters = np.random.default_rng(SEED), [], {name: [] for name in METHODS}
    for fold in range(3):
        train = np.flatnonzero((folds != fold) & rows['eligible'])
        test = np.flatnonzero((folds == fold) & rows['eligible'])
        # One predetermined shuffle per set. No training/test cross-assignment.
        train_assignment, test_assignment = train[rng.permutation(len(train))], test[rng.permutation(len(test))]
        private[f'fold_{fold}_training_velocity_groups'] = train_assignment
        private[f'fold_{fold}_heldout_velocity_groups'] = test_assignment
        scale, latitude = None, None
        if len(train):
            train_nodes = np.concatenate([np.asarray(groups[i], np.int64) for i in train])
            latitude = float(states[train_nodes, :, 0][support[train_nodes]].mean())
            if not np.isfinite(latitude) or abs(np.cos(np.deg2rad(latitude))) < 1e-8:
                raise ValueError('training-fold physical chart outside supported latitude')
            private['fold_chart_latitude'][fold] = latitude
            scale = RADIUS*np.pi/180.*np.array([1., np.cos(np.deg2rad(latitude))])
        train_d = rows['delta_degrees'][train]*scale if scale is not None else np.empty((0, 2))
        train_v = rows['velocity_degrees_per_second'][train]*scale if scale is not None else np.empty((0, 2))
        permuted_v = rows['velocity_degrees_per_second'][train_assignment]*scale if scale is not None else np.empty((0, 2))
        fitted = fit_models(train_d, train_v)
        fitted['JOINT_PERMUTED_LS'] = fit_models(train_d, permuted_v, joint_only=True)['JOINT_LS']
        for name, model in fitted.items():
            if model['b'] is None:
                continue
            tau = model['tau'] if model['tau'] is not None else 0.
            private['fold_parameters_'+name][fold] = [*model['b'], model['tau'] if model['tau'] is not None else np.nan]
            evaluation_v = rows['velocity_degrees_per_second'][
                test_assignment if name == 'JOINT_PERMUTED_LS' else test]*scale
            evaluation_d = rows['delta_degrees'][test]*scale
            errors[name][test] = np.linalg.norm(evaluation_d-model['b']+evaluation_v*tau, axis=1)
            if model['tau'] is not None and name != 'STATIC_LS':
                time_parameters[name].append(model['tau'])
        public_folds.append(dict(fold=fold, training_group_count=len(train), heldout_eligible_group_count=len(test),
            chart_defined=latitude is not None,
            centered_velocity_energy_m2_per_s2=float(np.mean(np.sum((train_v-train_v.mean(0))**2, axis=1))) if len(train) else None,
            training_velocity_assignment_changed_groups=int((train_assignment != train).sum()),
            heldout_velocity_assignment_changed_groups=int((test_assignment != test).sum()),
            models={name: public_model(model) for name, model in fitted.items()}))
    for name in METHODS:
        private['heldout_residual_norm_m_'+name] = errors[name]
    public_methods = {name: residual_stats(value[np.isfinite(value)]) for name, value in errors.items()}
    for name, values in time_parameters.items():
        v = np.asarray(values)
        public_methods[name]['time_coefficient_stability'] = dict(defined_folds=len(v),
            std_seconds=float(v.std()) if len(v) else None,
            range_seconds=float(np.ptp(v)) if len(v) else None,
            absolute_median_seconds=float(np.median(np.abs(v))) if len(v) else None,
            negative_count=int((v < 0).sum()), positive_count=int((v > 0).sum()), zero_count=int((v == 0).sum()))
    public = dict(scene_id=str(scene_id), component_id=str(component_id), node_count=len(states),
        group_count=len(groups), motion_eligible_group_count=int(rows['eligible'].sum()),
        no_shared_support_group_count=int((rows['shared_pair_query_count'] == 0).sum()),
        shared_without_derivative_group_count=int(((rows['shared_pair_query_count'] > 0) & ~rows['eligible']).sum()),
        cross_source_supported_pair_queries=int(rows['shared_pair_query_count'].sum()),
        derivative_supported_pair_queries=int(rows['derivative_pair_query_count'].sum()),
        evaluated_group_count=public_methods['STATIC_LS']['group_count'],
        undefined_group_count=len(groups)-public_methods['STATIC_LS']['group_count'],
        methods=public_methods, folds=public_folds,
        pairwise={candidate+'_vs_'+base: dict(paired_stats(errors[base], errors[candidate]),
            baseline_method=base, candidate_method=candidate) for base, candidate in COMPARISONS})
    return public, private, errors


COUNT_KEYS = ('node_count', 'group_count', 'motion_eligible_group_count', 'no_shared_support_group_count',
              'shared_without_derivative_group_count', 'cross_source_supported_pair_queries',
              'derivative_supported_pair_queries', 'evaluated_group_count', 'undefined_group_count')


def aggregate(rows, arrays):
    combined = {name: np.concatenate([value[name] for value in arrays]) for name in METHODS}
    out = dict(scene_count=len(rows), scene_ids=[row['scene_id'] for row in rows],
               **{key: sum(row[key] for row in rows) for key in COUNT_KEYS})
    out['methods'] = {}
    for name, values in combined.items():
        defined = [row['methods'][name]['mean_m'] for row in rows if row['methods'][name]['mean_m'] is not None]
        out['methods'][name] = dict(residual_stats(values[np.isfinite(values)]),
            scene_equal_mean_m=float(np.mean(defined)) if defined else None, defined_scenes=len(defined))
    out['pairwise'] = {candidate+'_vs_'+base: dict(paired_stats(combined[base], combined[candidate]),
        baseline_method=base, candidate_method=candidate,
        scene_mean_changes={row['scene_id']: row['pairwise'][candidate+'_vs_'+base]['candidate_minus_baseline_mean_m']
                            for row in rows}) for base, candidate in COMPARISONS}
    return out


def metadata(source, previous, shared, manifest):
    meta = old.metadata(source, previous, manifest)
    run, summary = (old.frozen.read(shared/name) for name in ('run.json', 'summary.json'))
    if (shared.name != SHARED_RUN or run.get('status') != 'COMPLETED' or summary.get('status') != 'COMPLETED'
            or run.get('idea_id') != ORIGINAL_IDEA or summary.get('idea_id') != ORIGINAL_IDEA
            or run.get('plan_version') != '1.0' or summary.get('plan_version') != '1.0'
            or Path(run.get('source_system', '')).resolve() != source
            or Path(run.get('partition_run', '')).resolve() != previous
            or run.get('scene_ids') != meta['ids'] or summary.get('scene_ids') != meta['ids']
            or run.get('completed_scenes') != len(meta['ids'])
            or run.get('methods') != list(SHARED_METHODS)
            or not summary.get('fixed_R_population_invariants_verified') or not summary.get('no_expansion')):
        raise ValueError('completed original118 fixed V32/source binding required')
    return meta


def load_physical(source, sid):
    with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in PHYSICAL_KEYS}


def run(args):
    source, previous, shared, output = (Path(getattr(args, key)).resolve()
        for key in ('source_system', 'partition_run', 'shared_run', 'output'))
    for protected in (source.parent, previous, shared):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new output must not overlap immutable inputs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be 1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), source_system=str(source), partition_run=str(previous),
        shared_run=str(shared), methods=list(METHODS), cpu_threads=args.cpu_threads, completed_scenes=0,
        model_forwards=0, models_constructed=0, optimizer_updates=0, gpu_used=False,
        predictions_changed=False, scores_recomputed=False, groups_recomputed=False,
        threshold_reselected=False, normalizer_refitted=False, original118_centers_read=False,
        archive_read=False, csv_read=False, reference_array_fields_loaded=False,
        identity_array_fields_loaded=False, association_label_array_fields_loaded=False,
        metadata_binding_uses_existing_evaluation_summaries=True,
        test_opened=False, seed=SEED, numerical_rank_rcond=RCOND)
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
        meta = metadata(source, previous, shared, old.frozen.read(args.manifest))
        checkpoint(scene_ids=meta['ids'], total_scenes=len(meta['ids']))
        (output/'scenes').mkdir()
        (output/'private').mkdir()
        reports, arrays = [], []
        with np.load(meta['association']/'final-scores.npz', allow_pickle=False) as scores:
            old._same(scores['scene_ids'], meta['all_ids'], '056 score scene order')
            for sid in meta['ids']:
                physical = load_physical(source, sid)
                for key in ('node_ids', 'sources'):
                    old._same(physical[key], scores[f'{sid}_{key}'], '056 '+key)
                previous_row = old.frozen.read(previous/'private'/f'scene-{sid}.json.gz')
                membership = np.asarray(previous_row['memberships']['WS'])
                with np.load(shared/'private'/f'states-{sid}.npz', allow_pickle=False) as saved:
                    old._same(saved['WS_MEAN_node_group_indices'], membership, '118 WS membership')
                    calibration = {key: saved[key].copy() for key in ('source_order', 'group_fold')}
                public = old.frozen.read(shared/'scenes'/f'scene-{sid}.json.gz')
                if (public['scene_id'] != sid or public['component_id'] != str(meta['components'][sid])
                        or public['node_count'] != len(physical['node_ids'])
                        or not public.get('fixed_R_population_invariants_verified')):
                    raise ValueError('118 scene/source binding differs')
                report, private, errors = analyze_scene(physical, membership, calibration,
                    scene_id=sid, component_id=meta['components'][sid])
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', report)
                np.savez_compressed(output/'private'/f'residuals-{sid}.npz', **private)
                reports.append(report)
                arrays.append(errors)
                checkpoint(completed_scenes=len(reports), last_scene_id=sid)
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **aggregate(reports, arrays),
            components={cid: aggregate([r for r in reports if r['component_id'] == cid],
                [a for r, a in zip(reports, arrays) if r['component_id'] == cid])
                for cid in sorted({r['component_id'] for r in reports})},
            no_expansion=True, predictions_changed=False, reference_array_fields_loaded=False,
            identity_array_fields_loaded=False, metadata_binding_uses_existing_evaluation_summaries=True,
            model_forwards=0, optimizer_updates=0,
            analytic_parameter_fit_count=sum(model['fit_attempted'] for row in reports
                for fold in row['folds'] for model in fold['models'].values()),
            limitations=['input-only relative group-mean residuals, not physical reference errors or clock evidence',
                'offline reconstructed Q64 derivatives and working chart; not native-time causal inference',
                'fixed repeated V32 development population/shared identities; no independent generalization',
                'wrong associations or velocity-dependent spatial errors can explain a time coefficient',
                'original118 source ordering fixes a relative direction, not a known physical absolute anchor'])
        largest = max(result['components'], key=lambda cid: result['components'][cid]['scene_count'])
        remaining = [(r, a) for r, a in zip(reports, arrays) if r['component_id'] != largest]
        result['without_largest_component'] = dict(excluded_component_id=largest,
            remaining=aggregate([r for r, _ in remaining], [a for _, a in remaining]) if remaining else None)
        old.frozen.write(output/'scene-summary.json.gz', reports)
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
    for key in ('source-system', 'partition-run', 'shared-run', 'manifest', 'output'):
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
