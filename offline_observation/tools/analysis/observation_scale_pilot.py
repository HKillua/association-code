"""137: T-only noisy-observation scale selection and fixed T32/V32 recovery.

No identity/reference is read until all nine T heldout criteria are frozen.
The original 127 CV/RTS kernel is reused with retained-only local geometry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.evaluation import observation_scale as scale
from association.evaluation import source_risk as risk
from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.query_contribution import recover_group_states
from tools.analysis import native_cv_state_control as cv
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source

old = common.old
IDEA = 'IDEA-OBSERVATION-SCALE-20261030'
EQUAL = 'WS_SOURCE_EQUAL'
NEW = ('WS_REFERENCE_SCALE_FORWARD_EQUAL', 'WS_REFERENCE_SCALE_RTS_EQUAL',
       'WS_OBS_SCALE_FORWARD_EQUAL', 'WS_OBS_SCALE_RTS_EQUAL')
T_METHODS = (EQUAL,)+NEW
V_METHODS = source.METHODS+cv.METHODS[2:]+NEW
PRIMARY = (NEW[3]+'_vs_'+EQUAL, NEW[3]+'_vs_'+NEW[1])
RAM, ADDRESS_SPACE = 6*1024**3, 12*1024**3


def heldout_scene(native, check=None):
    """Physical-only selection input and complete independently auditable rows."""
    fragments = []
    work = {key: 0 for key in scale.WORK_KEYS}
    fits = 0
    for index, record in enumerate(cv.native_records(native)):
        row = scale.heldout_rho_errors(times=record['times'], latlon=record['values'][:, :2])
        fragments.append(row)
        fits += row['analytic_filter_smoother_fits']
        for key in work:
            work[key] += row['analytic_work'][key]
        if check is not None and index % 16 == 0:
            check()
    if not fragments:
        raise ValueError('native scene has no fragments')
    offsets = np.r_[0, np.cumsum([r['heldout_event_count'] for r in fragments])].astype(np.int64)
    saved = dict(node_ids=native['node_ids'], query_seconds=native['query_seconds'],
        heldout_offsets=offsets,
        retained_count=np.asarray([r['retained_event_count'] for r in fragments], np.int64),
        eligible=np.asarray([r['eligible'] for r in fragments], bool),
        mse_by_rho=np.asarray([r['mse_m2'] for r in fragments]))
    for key in ('heldout_indices', 'heldout_times', 'heldout_latlon'):
        saved[key] = np.concatenate([r[key] for r in fragments], axis=0)
    for key in ('rts_positions', 'forward_positions'):
        saved[key] = np.concatenate([r[key] for r in fragments], axis=1)
    intervals = np.concatenate([np.diff(r['times']) for r in cv.native_records(native)])
    audit = dict(native_fragment_count=len(fragments),
        eligible_fragment_count=int(saved['eligible'].sum()),
        native_event_count=len(native['event_times_seconds']),
        heldout_event_count=int(offsets[-1]), retained_event_count=int(saved['retained_count'].sum()),
        native_gap_seconds=dict(count=len(intervals), min=float(intervals.min()),
            median=float(np.median(intervals)), p95=float(np.quantile(intervals, .95)), max=float(intervals.max())),
        analytic_filter_smoother_fits=fits, analytic_work=work)
    return saved, audit


def recover_equal(motion, groups, mean, pack, positions):
    """Change positions only; preserve the original state-specific fallback."""
    states = motion['states'].copy()
    if positions.shape != states[..., :2].shape or not np.isfinite(positions).all():
        raise ValueError('complete finite node positions required')
    states[..., :2] = positions
    first = np.asarray([g[0] for g in groups], np.int64)
    local_mean = ordinary_fusion(states, motion['nearest_seconds'], motion['support'], groups)[
        'predictions']['mean_supported'][first]
    recovered = risk.recover_source_policy(states=states, support=motion['support'], groups=groups,
        pack=pack, scores=np.zeros(pack['available'].shape), kind='equal', mean_states=local_mean)
    recovered['recovery_audit'] = recover_group_states(states, recovered['node_weights'], groups)['audit']
    recovered['group_states'][..., 2:] = mean[..., 2:]
    # This is used only to reconstruct the already completed 127 historical
    # controls, never to assign source priority in a new policy.
    local_mean[..., 2:] = mean[..., 2:]
    return recovered, states, local_mean


def predict(motion, groups, mean, pack, native, observation_rho, reference_rho, check=None):
    """Full-node recovery without any reference data; identical rhos deduplicate."""
    for key in ('node_ids', 'sources', 'query_seconds'):
        old._same(native[key], motion[key], 'native/motion '+key)
    unique = list(dict.fromkeys((float(reference_rho), float(observation_rho))))
    if any(rho not in scale.RHOS for rho in unique):
        raise ValueError('fixed original rho grid required')
    base = risk.recover_source_policy(states=motion['states'], support=motion['support'],
        groups=groups, pack=pack, scores=np.zeros(pack['available'].shape), kind='equal', mean_states=mean)
    predictions = {EQUAL: base['group_states']}
    stored = dict(unique_rhos=np.asarray(unique), reference_rho_index=np.asarray(unique.index(reference_rho)),
        obs_rho_index=np.asarray(unique.index(observation_rho)),
        **{EQUAL+'__'+key: base[key] for key in ('source_weights', 'node_weights', 'group_states')})
    audit = {EQUAL: source.contribution_summary(pack, base['source_weights'])}
    rebuilt = {}
    work = {key: 0 for key in scale.WORK_KEYS}
    for index, rho in enumerate(unique):
        node_positions, cost = cv.predict_scene(native, rho, check=check)
        for key in work:
            work[key] += cost[key]
        for direction, suffix in (('forward', 'FORWARD'), ('rts', 'RTS')):
            stored[f'rho_{index}__{direction}_positions'] = node_positions[direction]
            result, states, local_mean = recover_equal(motion, groups, mean, pack, node_positions[direction])
            if rho == reference_rho:
                low = recover_group_states(states, cv.low_weights(motion['support'], motion['sources'], groups), groups)
                low['group_states'][..., 2:] = mean[..., 2:]
                rebuilt['WS_CV_'+suffix+'_MEAN'] = local_mean
                rebuilt['WS_CV_'+suffix+'_LOW'] = low['group_states']
            for label, selected_rho in (('REFERENCE', reference_rho), ('OBS', observation_rho)):
                if rho == selected_rho:
                    name = f'WS_{label}_SCALE_{suffix}_EQUAL'
                    predictions[name] = result['group_states']
                    for key in ('source_weights', 'node_weights', 'group_states'):
                        stored[name+'__'+key] = result[key]
                    audit[name] = dict(source.contribution_summary(pack, result['source_weights']),
                        recovery=result['recovery_audit'])
    for name, states in predictions.items():
        old._same(states[..., 2:], mean[..., 2:], 'original speed/heading '+name)
    return predictions, rebuilt, stored, audit, work


def run(args):
    paths = {key: Path(getattr(args, key)).resolve() for key in
             ('native_run', 'value_run', 'partition_run', 'source_risk_run', 'output')}
    output = paths['output']
    for key, path in paths.items():
        if key != 'output' and (output == path or output in path.parents or path in output.parents):
            raise ValueError('new output overlaps frozen input')
    if not 1 <= args.cpu_threads <= 4 or getattr(args, 'time_policy', 'advisory') != 'advisory':
        raise ValueError('CPU1..4 and advisory time policy required')
    output.mkdir(parents=True, exist_ok=False)
    (output/'private').mkdir()
    for split in ('T_core', 'V_select'):
        (output/'scenes'/split).mkdir(parents=True)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        **{key: str(value) for key, value in paths.items() if key != 'output'},
        manifest=str(Path(args.manifest).resolve()), T_methods=list(T_METHODS), methods=list(V_METHODS),
        imported_methods=list(source.METHODS+cv.METHODS[2:]), rhos=list(scale.RHOS),
        completed_selection_scene_count=0, completed_training_scene_count=0, completed_scene_count=0,
        native_event_count=0, native_fragment_count=0, selection_native_event_count=0,
        selection_eligible_fragment_count=0, selection_heldout_event_count=0, selection_retained_event_count=0,
        analytic_filter_smoother_fits=0, selection_filter_smoother_fits=0, recovery_filter_smoother_fits=0,
        analytic_work={key: 0 for key in scale.WORK_KEYS},
        analytic_counter_scope='Successfully returned scene kernels; an interrupted current scene may contain additional uncounted partial work. Complete endpoints include every declared fit.',
        teacher_forward_calls=0, neural_models_constructed=0, optimizer_updates=0,
        target_reference_fit_count=0, analytic_head_fits=0, archive_read=False, csv_read=False,
        test_opened=False, source_id_feature=False, gpu_used=False, cpu_threads=args.cpu_threads,
        time_policy='advisory', predictions_frozen_before_reference=True,
        selection_frozen_before_reference_control=False, costs={})

    def check(**extra):
        state.update(extra, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes())
        state['advisory_exceeded'] = state['elapsed_seconds'] > 9000
        old.frozen.write(output/'run.json', state)
        if state['advisory_exceeded'] and not (output/'time_advisory.json').exists():
            old.frozen.write(output/'time_advisory.json', dict(elapsed_seconds=state['elapsed_seconds'],
                action='continue only fixed nine-rho T selection and original T32/V32 recovery; no retry'))
        if state['peak_rss_bytes'] > RAM:
            raise MemoryError('6GiB resident budget exceeded')

    def add_work(work):
        for key in state['analytic_work']:
            state['analytic_work'][key] += int(work[key])

    try:
        check(phase='binding')
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([ADDRESS_SPACE]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        manifest = old.frozen.read(args.manifest)
        tids, vids = common.metadata(paths['native_run'], paths['value_run'], paths['partition_run'],
                                      paths['source_risk_run'], manifest)
        check(training_scene_ids=tids, scene_ids=vids, phase='T_observation_selection')
        selection_inputs, selection_rows = [], []
        stage = time.monotonic()
        for sid in tids:
            check(last_scene_id=sid)
            native = common.load_npz(paths['native_run']/'private/native-events/T_core'/f'scene-{sid}.npz')
            saved, counts = heldout_scene(native, check=check)
            np.savez_compressed(output/'private'/f'heldout-T-{sid}.npz', **saved)
            selection_inputs.append(dict(scene_id=sid, mse_m2=saved['mse_by_rho'], eligible=saved['eligible']))
            selection_rows.append(dict(scene_id=sid, component_id=str(manifest['components'][sid]), **counts))
            state['selection_native_event_count'] += counts['native_event_count']
            state['selection_eligible_fragment_count'] += counts['eligible_fragment_count']
            state['selection_heldout_event_count'] += counts['heldout_event_count']
            state['selection_retained_event_count'] += counts['retained_event_count']
            state['selection_filter_smoother_fits'] += counts['analytic_filter_smoother_fits']
            state['analytic_filter_smoother_fits'] += counts['analytic_filter_smoother_fits']
            add_work(counts['analytic_work'])
            check(completed_selection_scene_count=len(selection_rows))
        selection = scale.select_observation_rho(selection_inputs)
        selection.update(idea_id=IDEA, plan_version='1.0', training_scene_ids=tids,
            observations=selection_rows, analytic_filter_smoother_fits=state['selection_filter_smoother_fits'],
            selected_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'scale-selection.json', selection)
        state['costs']['T_observation_selection_seconds'] = time.monotonic()-stage
        observation_rho = selection['selected_rho']
        # Both the file and phase are persisted before even reading the fixed
        # reference-chosen control scale, not merely before reference scoring.
        check(phase='selection_frozen', selection_frozen_before_reference_control=True,
              observation_selected_rho=observation_rho)
        control_selection = old.frozen.read(paths['native_run']/'scale-selection.json')
        reference_rho = float(control_selection['selected_rho'])
        if reference_rho not in scale.RHOS:
            raise ValueError('completed 127 reference scale is outside original grid')
        unique_rhos = list(dict.fromkeys((reference_rho, observation_rho)))
        check(reference_selected_rho=reference_rho, unique_recovery_rhos=unique_rhos,
              recovery_alias=reference_rho == observation_rho)
        summaries = {}
        for split, ids, methods in (('T_core', tids, T_METHODS), ('V_select', vids, V_METHODS)):
            rows, arrays = [], []
            stage = time.monotonic()
            for sid in ids:
                check(phase=split+'_physical_prediction', last_scene_id=sid)
                membership = common.membership(paths['value_run'], paths['partition_run'], split, sid)
                motion, groups, mean, pack, saved = common.physical(paths['native_run'], split, sid, membership)
                native = common.load_npz(paths['native_run']/'private/native-events'/split/f'scene-{sid}.npz')
                predictions, rebuilt, positions, weights, work = predict(motion, groups, mean, pack, native,
                    observation_rho, reference_rho, check=check)
                saved.update(positions, support=motion['support'])
                np.savez_compressed(output/'private'/f'native-state-{split}-{sid}.npz', **saved)
                n = len(motion['states'])
                fits = n*len(unique_rhos)
                state['recovery_filter_smoother_fits'] += fits
                state['analytic_filter_smoother_fits'] += fits
                state['native_fragment_count'] += n
                state['native_event_count'] += len(native['event_times_seconds'])
                add_work(work)
                retained = {}
                if split == 'V_select':
                    retained = common.load_artifacts(paths['source_risk_run']/'private'/f'states-{sid}.npz', source.METHODS)
                    historical = common.load_artifacts(paths['native_run']/'private'/f'states-{sid}.npz', cv.METHODS)
                    for name in cv.METHODS[:2]:
                        if set(retained[name]) != set(historical[name]):
                            raise ValueError('127/133 shared artifact fields differ')
                        for key in retained[name]:
                            old._same(retained[name][key], historical[name][key], '127/133 common '+name+'/'+key)
                    old._same(mean, retained['WS_MEAN']['group_states'], 'original Mean')
                    old._same(predictions[EQUAL], retained[EQUAL]['group_states'], '133 source-equal')
                    retained.update({name: historical[name] for name in cv.METHODS[2:]})
                    predictions.update({name: value['group_states'] for name, value in retained.items()})
                    predictions.update(rebuilt)
                # All new physical predictions, including full node positions,
                # are already saved before this identity/reference read.
                ev = common.evaluation(paths['native_run'], split, sid, motion)
                row, local_arrays, artifacts = common.evaluate_predictions(motion, groups, pack, ev,
                    predictions, retained, method_names=methods, split=split)
                common.save_artifacts(output/'private'/f'states-{split}-{sid}.npz', artifacts)
                row.update(scene_id=sid, component_id=str(manifest['components'][sid]), split=split,
                    node_count=n, source_support_scope=source.support_counts(pack), source_contributions=weights,
                    native_event_count=len(native['event_times_seconds']), unique_recovery_rhos=unique_rhos,
                    analytic_filter_smoother_fits=fits, analytic_work=work,
                    original133_exact_replays_verified=split == 'V_select',
                    original127_exact_replays_verified=split == 'V_select', fixed_population_invariants_verified=True)
                old.frozen.write(output/'scenes'/split/f'scene-{sid}.json.gz', row)
                rows.append(row)
                arrays.append(local_arrays)
                check(**{('completed_training_scene_count' if split == 'T_core' else 'completed_scene_count'): len(rows)})
            if len(rows) != 32 or any(len(row['methods']) != len(methods)
                or len(row['pairwise']) != len(methods)*(len(methods)-1)//2 for row in rows):
                raise ValueError('fixed scene/method/pair endpoint incomplete')
            state['costs'][split+'_recovery_evaluation_seconds'] = time.monotonic()-stage
            summaries[split] = common.summarize(rows, arrays, method_names=methods, primary_pairs=PRIMARY)
            old.frozen.write(output/('T-scene-summary.json.gz' if split == 'T_core' else 'scene-summary.json.gz'), rows)
            old.frozen.write(output/('T-summary.json' if split == 'T_core' else 'summary.json'), summaries[split])
        check(phase='finalizing')
        summary = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **summaries['V_select'],
            training_summary=summaries['T_core'], primary_comparisons=list(PRIMARY),
            observation_selected_rho=observation_rho, reference_selected_rho=reference_rho,
            unique_recovery_rhos=unique_rhos, recovery_alias=reference_rho == observation_rho,
            costs=state['costs'], elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'],
            analytic_filter_smoother_fits=state['analytic_filter_smoother_fits'],
            selection_filter_smoother_fits=state['selection_filter_smoother_fits'],
            recovery_filter_smoother_fits=state['recovery_filter_smoother_fits'], analytic_work=state['analytic_work'],
            limitations=['Offline retained-window chart and RTS; FWD is not a streaming demonstration',
                'Noisy observed holdout error is not latent-state reference risk or calibrated Q/R',
                'Interior holdout with retained endpoints does not directly tune extrapolation',
                'Position alone changes; original speed/heading and output population preserved',
                'Zero-support uses current model first-node fallback and can change its position',
                'Actual K2 and repeatedly developed V32 do not establish unseen-sensor generalization',
                'Reference scale and historical priority are controls only; new contribution is source-equal',
                'Teacher/grouping remain supervised; new observation-scale selection is reference-blind'])
        old.frozen.write(output/'summary.json', summary)
        check(status='COMPLETED', exit_code=0, phase='completed', finished_at=datetime.now(timezone.utc).isoformat())
        return summary
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started,
            failed_scene_partial_compute='Unknown; counters include only kernels that returned successfully for a complete scene',
            finished_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-run', 'value-run', 'partition-run', 'source-risk-run', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        return 1
    print(f'COMPLETED: {args.output}/summary.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
