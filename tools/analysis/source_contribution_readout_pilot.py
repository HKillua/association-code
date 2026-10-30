"""140 bounded T32/V32 chord-risk/readout pilot; no raw events or neural calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import resource
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.evaluation import source_contribution_readout as core
from association.evaluation import source_risk as risk
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source

old = common.old
IDEA = 'IDEA-SOURCE-CONTRIBUTION-READOUT-20261030'
NEW = ('WS_CHORD_RISK_HARD', 'WS_CHORD_RISK_INVERSE', 'WS_CHORD_RISK_GEOMETRY')
T_METHODS = ('WS_SOURCE_EQUAL', 'WS_SHARED_Q_SOFT', 'WS_SHARED_Q_HARD')+NEW
V_METHODS = source.METHODS+NEW
PRIMARY = (NEW[0]+'_vs_WS_SHARED_Q_HARD', NEW[1]+'_vs_'+NEW[0],
           NEW[2]+'_vs_'+NEW[0], NEW[2]+'_vs_'+NEW[1])
RAM, ADDRESS_SPACE = 6*1024**3, 12*1024**3


def bind(saved, original, *, name):
    for key, value in saved.items():
        if key not in original:
            raise ValueError('133 missing '+name+'/'+key)
        old._same(value, original[key], '133 physical binding '+name+'/'+key)


def _stat(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
        max=float(values.max()) if len(values) else None,
        min=float(values.min()) if len(values) else None)


def policy_diagnostic(pack, raw, weights):
    available = pack['available']
    values = raw[available]
    result = dict(available_source_queries=int(available.sum()),
        raw_negative_count=int((values < 0).sum()), raw_negative_values_km2=_stat(values[values < 0]),
        clipped_zero_count=int((values <= 0).sum()), methods={})
    for name, policy in weights.items():
        defined = policy['solver_defined']
        obj = policy['objective_km2'][defined]
        result['methods'][name] = dict(floor_hit_count=int(policy['floor_hit'].sum()),
            floor_denominator_source_queries=int(available.sum()),
            defined_group_queries=int(defined.sum()), negative_objective_count=int((obj < 0).sum()),
            objective_km2=_stat(obj), duality_gap_km2=_stat(policy['duality_gap_km2'][defined]),
            solver_iterations=_stat(policy['solver_iterations'][defined]),
            solver_not_converged_count=int((defined & ~policy['solver_converged']).sum()) if name == NEW[2] else None,
            contributions=source.contribution_summary(pack, policy['source_weights']))
    return result


def endpoint_diagnostic(pack, labels, scores):
    """Hypothetical endpoint selection, separate from actual physical mixing."""
    weights = risk.source_weights(scores, pack['available'], pack['group_offsets'], kind='hard')
    costs, regrets, credits = [], [], []
    for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:]):
        use = labels['labeled'][a:b].any(0)
        if not use.any():
            continue
        valid = labels['labeled'][a:b, use]
        error = np.where(valid, labels['errors_m'][a:b, use], np.inf)
        best = error.min(0)
        w = weights[a:b, use]
        selected = (w*np.where(valid, error, 0.)).sum(0)
        costs.extend(selected.tolist())
        regrets.extend((selected-best).tolist())
        credits.extend((w*(error == best)).sum(0).tolist())
    return dict(query_count=len(costs), expected_endpoint_error_m=_stat(costs),
        selection_regret_m=_stat(regrets), selection_credit=_stat(credits),
        regret_gt_1000m_count=int((np.asarray(regrets) > 1000.).sum()),
        regret_gt_1000m_sum=float(np.asarray(regrets)[np.asarray(regrets) > 1000.].sum()))


def reference_diagnostic(pack, groups, ev, raw, old_scores):
    labels = source.training_labels(pack, groups, ev)
    mask = labels['labeled']
    target = core.chord_targets(labels['errors_m'][mask])
    weight = labels['sample_weight'][mask]
    raw_residual = raw[mask]-target
    clipped_residual = np.maximum(raw[mask], 0.)-target
    return dict(label_audit=labels['audit'], labeled_source_queries=int(mask.sum()),
        raw_negative_count=int((raw[mask] < 0).sum()),
        inverse_floor_hit_labeled_count=int((np.maximum(raw[mask], 0.) < core.RISK_FLOOR_KM2).sum()),
        raw_squared_risk_residual_km4=_stat(raw_residual**2),
        clipped_squared_risk_residual_km4=_stat(clipped_residual**2),
        training_weight_sum=float(weight.sum()),
        weighted_raw_squared_residual_sum_km4=float(weight@(raw_residual**2)),
        weighted_clipped_squared_residual_sum_km4=float(weight@(clipped_residual**2)),
        old_Q_endpoint=endpoint_diagnostic(pack, labels, old_scores),
        chord_risk_endpoint=endpoint_diagnostic(pack, labels, np.maximum(raw, 0.))), labels


def run(args):
    names = ('native_run', 'value_run', 'partition_run', 'source_risk_run', 'output')
    paths = {key: Path(getattr(args, key)).resolve() for key in names}
    output = paths['output']
    for key, path in paths.items():
        if key != 'output' and (output == path or output in path.parents or path in output.parents):
            raise ValueError('new output overlaps frozen input')
    if not 1 <= args.cpu_threads <= 4 or args.time_policy != 'advisory':
        raise ValueError('CPU1..4 and advisory policy required')
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('private', 'scenes/T_core', 'scenes/V_select'):
        (output/folder).mkdir(parents=True)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(),
        **{k: str(v) for k, v in paths.items() if k != 'output'},
        manifest=str(Path(args.manifest).resolve()), T_methods=list(T_METHODS), V_methods=list(V_METHODS),
        primary_pairs=list(PRIMARY), analytic_head_fits=0, ridge_system_solves=0,
        no_label_fallback_heads=0, completed_training_cache_count=0,
        completed_training_scene_count=0, completed_scene_count=0,
        teacher_forward_calls=0, neural_models_constructed=0, optimizer_updates=0,
        native_event_files_read=0, grid_oracle_calls=0, archive_read=False, csv_read=False,
        test_opened=False, gpu_used=False, cpu_threads=args.cpu_threads, time_policy='advisory',
        T_old_Q_artifacts='new recovery from original133 OOF scores, not imported historical T artifacts',
        full_T_models_frozen_before_V=False, completed_counts_scope='completed stages/scenes only; interrupted current partial work unknown', costs={})

    def check(**extra):
        state.update(extra, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes())
        state['advisory_exceeded'] = state['elapsed_seconds'] > 1800
        old.frozen.write(output/'run.json', state)
        if state['advisory_exceeded'] and not (output/'time_advisory.json').exists():
            old.frozen.write(output/'time_advisory.json', dict(elapsed_seconds=state['elapsed_seconds'],
                action='continue fixed13 fits and originalT32/V32 only'))
        if state['peak_rss_bytes'] > RAM:
            raise MemoryError('6GiB RSS budget exceeded')

    try:
        check(phase='binding')
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([ADDRESS_SPACE]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        manifest = old.frozen.read(args.manifest)
        tids, vids = common.metadata(paths['native_run'], paths['value_run'], paths['partition_run'],
                                     paths['source_risk_run'], manifest)
        check(training_scene_ids=tids, scene_ids=vids, phase='T_cached_labels')
        training = []
        stage = time.monotonic()
        for sid in tids:
            member = common.membership(paths['value_run'], paths['partition_run'], 'T_core', sid)
            motion, groups, mean, pack, saved = common.physical(paths['native_run'], 'T_core', sid, member)
            original = common.load_npz(paths['source_risk_run']/'private'/f'training-{sid}.npz')
            bind(saved, original, name='T_core/'+sid)
            component = str(manifest['components'][sid])
            if str(original['component_id'].item()) != component:
                raise ValueError('133 training component differs')
            mask, error, weight = (original[k] for k in ('labeled', 'errors_m', 'sample_weight'))
            if (mask.shape != pack['available'].shape or mask.dtype != bool
                    or error.shape != mask.shape or weight.shape != mask.shape
                    or not np.isfinite(weight).all() or (weight[mask] <= 0).any()
                    or np.any(weight[~mask] != 0.) or np.any(mask & ~pack['available'])):
                raise ValueError('133 training label mask/weight invalid')
            target = np.full(mask.shape, np.nan)
            target[mask] = core.chord_targets(error[mask])
            old._same(original['y'][mask], np.log1p(error[mask]/1000.), '133 true radial/log label binding')
            saved.update(component_id=np.asarray(component), errors_m=error, labeled=mask,
                         sample_weight=weight, y_chord_km2=target)
            np.savez_compressed(output/'private'/f'training-{sid}.npz', **saved)
            training.append(dict(scene_id=sid, component_id=component, features=pack['features'],
                mask=mask, y=target, weight=weight))
            check(completed_training_cache_count=len(training), last_scene_id=sid)
        state['costs']['T_cache_binding_seconds'] = time.monotonic()-stage
        components = sorted({row['component_id'] for row in training})
        if len(components) != 12:
            raise ValueError('original12 T components required')
        models, folds = {}, []
        stage = time.monotonic()
        for heldout in components+[None]:
            fold = 'full' if heldout is None else 'leave_'+heldout
            selected = [row for row in training if row['component_id'] != heldout]
            x = np.concatenate([row['features'][row['mask']] for row in selected])
            y = np.concatenate([row['y'][row['mask']] for row in selected])
            w = np.concatenate([row['weight'][row['mask']] for row in selected])
            check(phase='T_fit', fold=fold)
            model = core.fit_risk_model(x, y, w)
            state['analytic_head_fits'] += 1
            state['ridge_system_solves'] += int(not model['no_labels_fallback'])
            state['no_label_fallback_heads'] += int(model['no_labels_fallback'])
            models.update({fold+'__'+key: np.asarray(value) for key, value in model.items()})
            np.savez_compressed(output/'private'/'models.npz', **models)
            folds.append(dict(fold=fold, heldout_component=heldout,
                training_components=sorted({row['component_id'] for row in selected}),
                training_scene_ids=[row['scene_id'] for row in selected],
                training_label_count=len(y), training_weight_sum=float(w.sum())))
            old.frozen.write(output/'T-fold-index.json', folds)
            if heldout is not None:
                for row in training:
                    if row['component_id'] == heldout:
                        raw = core.predict_risk(model, row['features'].reshape(-1, 10)).reshape(row['mask'].shape)
                        np.savez_compressed(output/'private'/f"oof-{row['scene_id']}.npz", risk_raw=raw)
            check()
        if state['analytic_head_fits'] != 13:
            raise ValueError('exactly13 single-head fits required')
        full_model = model
        state['costs']['T_fit_seconds'] = time.monotonic()-stage
        check(full_T_models_frozen_before_V=True, fold=None)
        summaries = {}
        for split, ids, methods in (('T_core', tids, T_METHODS), ('V_select', vids, V_METHODS)):
            rows, arrays = [], []
            stage = time.monotonic()
            check(phase=split+'_prediction_evaluation')
            for sid in ids:
                member = common.membership(paths['value_run'], paths['partition_run'], split, sid)
                motion, groups, mean, pack, saved = common.physical(paths['native_run'], split, sid, member)
                oldfile = f'training-{sid}.npz' if split == 'T_core' else f'policy-{sid}.npz'
                original = common.load_npz(paths['source_risk_run']/'private'/oldfile)
                bind(saved, original, name=split+'/'+sid)
                if split == 'T_core':
                    raw = common.load_npz(output/'private'/f'oof-{sid}.npz')['risk_raw']
                    old_q = common.load_npz(paths['source_risk_run']/'private'/f'oof-{sid}.npz')['QUADRATIC__scores']
                else:
                    raw = core.predict_risk(full_model, pack['features'].reshape(-1, 10)).reshape(pack['available'].shape)
                    old_q = original['WS_SHARED_Q_HARD__scores']
                if raw.shape != pack['available'].shape or old_q.shape != raw.shape or not np.isfinite(old_q).all():
                    raise ValueError('new risk or originalQ score shape/values differ')
                clipped = np.maximum(raw, 0.)
                saved.update(risk_raw=raw, risk_clipped=clipped, old_Q_scores=old_q)
                predictions, policies = {}, {}
                for name, kind in zip(NEW, ('hard', 'inverse', 'geometry')):
                    policy = core.contribution_weights(clipped, pack['available'], pack['group_offsets'],
                                                       pack['source_states'], kind=kind)
                    recovered = core.recover_contributions(states=motion['states'], support=motion['support'], groups=groups,
                        pack=pack, source_weights=policy['source_weights'], mean_states=mean)
                    policy.update(recovered)
                    policies[name] = policy
                    predictions[name] = recovered['group_states']
                    saved.update({name+'__'+key: value for key, value in policy.items()})
                if split == 'T_core':
                    for name, kind, score in (('WS_SOURCE_EQUAL', 'equal', np.zeros_like(old_q)),
                        ('WS_SHARED_Q_SOFT', 'soft', old_q), ('WS_SHARED_Q_HARD', 'hard', old_q)):
                        replay = risk.recover_source_policy(states=motion['states'], support=motion['support'], groups=groups,
                            pack=pack, scores=score, kind=kind, mean_states=mean)
                        predictions[name] = replay['group_states']
                        saved.update({name+'__'+key: value for key, value in replay.items()})
                # Prediction files are frozen before any evaluation/old artifact read.
                np.savez_compressed(output/'private'/f'policy-{split}-{sid}.npz', **saved)
                retained = {} if split == 'T_core' else common.load_artifacts(
                    paths['source_risk_run']/'private'/f'states-{sid}.npz', source.METHODS)
                if retained:
                    old._same(retained['WS_MEAN']['node_group_indices'], member, '133/114 WS membership')
                    old._same(retained['WS_MEAN']['group_states'], mean, '133 exactMean')
                    predictions.update({name: artifact['group_states'] for name, artifact in retained.items()})
                ev = common.evaluation(paths['native_run'], split, sid, motion)
                row, array, artifacts = common.evaluate_predictions(motion, groups, pack, ev, predictions, retained,
                                                                    method_names=methods, split=split)
                diag, labels = reference_diagnostic(pack, groups, ev, raw, old_q)
                # Independently bind reused T label arrays to the original evaluation contract.
                if split == 'T_core':
                    for key in ('labeled', 'sample_weight', 'errors_m'):
                        old._same(labels[key], original[key], '133 T evaluation label binding '+key)
                row.update(scene_id=sid, split=split, component_id=str(manifest['components'][sid]), node_count=len(motion['states']),
                    source_support_scope=source.support_counts(pack), feature_audit=pack['audit'],
                    risk_policy=policy_diagnostic(pack, raw, policies), reference_diagnostic=diag,
                    original133_exact_replays_verified=split == 'V_select',
                    fixed_population_invariants_verified=True, predictions_frozen_before_reference=True)
                common.save_artifacts(output/'private'/f'states-{split}-{sid}.npz', artifacts)
                np.savez_compressed(output/'private'/f'diagnostic-{split}-{sid}.npz',
                    **{key: labels[key] for key in ('labeled', 'sample_weight', 'errors_m')})
                old.frozen.write(output/'scenes'/split/f'scene-{sid}.json.gz', row)
                rows.append(row); arrays.append(array)
                count_key = 'completed_training_scene_count' if split == 'T_core' else 'completed_scene_count'
                check(**{count_key: len(rows)}, last_scene_id=sid)
            summaries[split] = common.summarize(rows, arrays, method_names=methods, primary_pairs=PRIMARY)
            summaries[split]['risk_diagnostics_by_scene'] = [{key: row[key] for key in
                ('scene_id', 'component_id', 'risk_policy', 'reference_diagnostic')} for row in rows]
            prefix = 'T-' if split == 'T_core' else ''
            old.frozen.write(output/(prefix+'scene-summary.json.gz'), rows)
            old.frozen.write(output/(prefix+'summary.json'), summaries[split])
            state['costs'][split+'_prediction_evaluation_seconds'] = time.monotonic()-stage
        if state['completed_training_scene_count'] != 32 or state['completed_scene_count'] != 32:
            raise ValueError('fixed T32/V32 endpoint incomplete')
        check(phase='finalizing')
        summary = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **summaries['V_select'],
            training_summary=summaries['T_core'], costs=state['costs'],
            elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'], analytic_head_fits=13,
            limitations=['Squared chord-risk regression is not direct fusion-position MSE training',
                'Finite feature-conditioned risk is not certified conditional on all decision geometry',
                'Convex surrogate and gap concern unnormalised3D vectors, not actual circular/radial output',
                'T LOCO isolates only this critic; upstream teacher and developedV restrictions persist',
                'Actual MTAD K2; generalK software is not empirical K>=3 or online validation',
                'Position-only: speed and heading remain frozenMean; no credible-peer superiority claim'])
        old.frozen.write(output/'summary.json', summary)
        check(status='COMPLETED', exit_code=0, phase='completed', finished_at=datetime.now(timezone.utc).isoformat())
        return summary
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes(),
            finished_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-run', 'value-run', 'partition-run', 'source-risk-run', 'manifest', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError('signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
