"""139: fixed T32/V32 single-factor radial-meter source-selection target."""
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

from association.evaluation import source_risk as risk
from association.evaluation import source_selection_evidence as selection
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source

old = source.old
IDEA = 'IDEA-SOURCE-SELECTION-EVIDENCE-20261030'
EQUAL, NEAREST = 'WS_SOURCE_EQUAL', 'WS_SOURCE_NEAREST'
T_LOG, T_METER, V_METER = 'WS_LOG_Q_OOF_HARD', 'WS_METER_Q_OOF_HARD', 'WS_METER_Q_HARD'
T_METHODS = (EQUAL, NEAREST, T_LOG, T_METER)
V_METHODS = source.METHODS+(NEAREST, V_METER)
T_PRIMARY = T_METER+'_vs_'+T_LOG
V_PRIMARY = V_METER+'_vs_WS_SHARED_Q_HARD'
INPUTS = ('native_run', 'value_run', 'partition_run', 'source_risk_run', 'quality_run', 'mixture_run')
RAM, ADDRESS_SPACE = 6*1024**3, 12*1024**3


def bind_equal(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc' and b.dtype.kind in 'fc'):
        raise ValueError(label)


def metadata(paths, manifest):
    tids, vids = common.metadata(*(paths[k] for k in INPUTS[:4]), manifest)
    quality = old.frozen.read(paths['quality_run']/'run.json')
    mixture = old.frozen.read(paths['mixture_run']/'run.json')
    if any(r.get('status') != 'COMPLETED' or r.get('exit_code') != 0 for r in (quality, mixture)):
        raise ValueError('136 and134 completed originals required')
    if (quality['training_scene_ids'] != tids or quality['scene_ids'] != vids
            or quality['completed_training_scene_count'] != 32 or quality['completed_scene_count'] != 32
            or mixture['scene_ids'] != tids or mixture['completed_scene_count'] != 32):
        raise ValueError('136/134 fixed scene populations differ')
    for key in INPUTS[:4]:
        if Path(quality[key]).resolve() != paths[key]:
            raise ValueError('136 lineage '+key)
    if (Path(mixture['native_run']).resolve() != paths['native_run']
            or Path(mixture['source_run']).resolve() != paths['source_risk_run']):
        raise ValueError('134 original lineage differs')
    return tids, vids


def policy(motion, groups, mean, pack, scores):
    return risk.recover_source_policy(states=motion['states'], support=motion['support'], groups=groups,
        pack=pack, scores=scores, kind='hard', mean_states=mean)


def combined_diagnostic(records):
    def aggregate(chosen):
        arrays = {key: np.concatenate([r['data'][key] for r in chosen])
                  for key in records[0]['data']}
        return selection.summarize_diagnostic(arrays)
    components = sorted({r['component_id'] for r in records})
    largest = max(components, key=lambda c: sum(r['component_id'] == c for r in records))
    return dict(all_components=aggregate(records),
        by_component={c: aggregate([r for r in records if r['component_id'] == c]) for c in components},
        without_largest_component=aggregate([r for r in records if r['component_id'] != largest]),
        largest_component_id=largest,
        scenes=[dict(scene_id=r['scene_id'], component_id=r['component_id'], **selection.summarize_diagnostic(r['data']))
                for r in records])


def run(args):
    paths = {key: Path(getattr(args, key)).resolve() for key in INPUTS}
    output = Path(args.output).resolve()
    if any(output == p or output in p.parents or p in output.parents for p in paths.values()):
        raise ValueError('new output overlaps frozen input')
    if not 1 <= args.cpu_threads <= 4:
        raise ValueError('CPU1..4 required')
    output.mkdir(parents=True, exist_ok=False)
    (output/'private').mkdir()
    for split in ('T_core', 'V_select'):
        (output/'scenes'/split).mkdir(parents=True)
    start = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), **{k: str(v) for k, v in paths.items()},
        manifest=str(Path(args.manifest).resolve()), command=sys.argv,
        methods=list(V_METHODS), training_methods=list(T_METHODS),
        imported_methods=list(V_METHODS[:-1]), imported_training_methods=[EQUAL, NEAREST],
        target_kind=selection.TARGET, analytic_head_fits=0, ridge_system_solves=0, no_label_fallback_heads=0,
        completed_label_scene_count=0, completed_training_scene_count=0, completed_scene_count=0,
        newly_recovered_old_T_log_scene_count=0, newly_recovered_policy_scene_count=0,
        feature_scene_read_count=0, evaluation_scene_read_count=0, training_pack_read_count=0,
        original_oof_read_count=0, neural_models_constructed=0, optimizer_updates=0, teacher_forward_calls=0,
        archive_read=False, csv_read=False, test_opened=False, gpu_used=False, new_state_estimates=0,
        new_groupings=0, new_oracle_searches=0, cpu_threads=args.cpu_threads, time_policy='advisory',
        full_T_models_frozen_before_V=False, source_features_frozen_before_labels=True, costs={})

    def checkpoint(**extra):
        state.update(extra, elapsed_seconds=time.monotonic()-start, peak_rss_bytes=old.frozen.peak_rss_bytes())
        state['advisory_exceeded'] = state['elapsed_seconds'] > 1800
        old.frozen.write(output/'run.json', state)
        if state['advisory_exceeded'] and not (output/'time_advisory.json').exists():
            old.frozen.write(output/'time_advisory.json', dict(action='complete fixed13 fits and T32/V32 only',
                                                             elapsed_seconds=state['elapsed_seconds']))
        if state['peak_rss_bytes'] > RAM:
            raise MemoryError('6GiB RSS exceeded')

    try:
        checkpoint(phase='binding')
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([ADDRESS_SPACE]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        manifest = old.frozen.read(args.manifest)
        tids, vids = metadata(paths, manifest)
        checkpoint(training_scene_ids=tids, scene_ids=vids, phase='T_labels')
        training, stage = [], time.monotonic()
        for sid in tids:
            members = common.membership(paths['value_run'], paths['partition_run'], 'T_core', sid)
            motion, groups, mean, pack, saved = common.physical(paths['native_run'], 'T_core', sid, members)
            state['feature_scene_read_count'] += 1
            original = common.load_npz(paths['source_risk_run']/'private'/f'training-{sid}.npz')
            state['training_pack_read_count'] += 1
            for key, value in saved.items():
                bind_equal(value, original[key], 'original133 feature '+key)
            ev = common.evaluation(paths['native_run'], 'T_core', sid, motion)
            state['evaluation_scene_read_count'] += 1
            labels = source.training_labels(pack, groups, ev)
            for key in ('y', 'labeled', 'sample_weight', 'errors_m'):
                bind_equal(labels[key], original[key], 'original133 label '+key)
            component = str(manifest['components'][sid])
            bind_equal(original['component_id'], np.asarray(component), 'original133 component')
            saved.update({k: v for k, v in labels.items() if isinstance(v, np.ndarray)})
            saved.update(component_id=np.asarray(component), y_log=labels['y'], y_meter=labels['errors_m']/1000.)
            np.savez_compressed(output/'private'/f'training-{sid}.npz', **saved)
            training.append(dict(scene_id=sid, component_id=component, saved=saved, labels=labels,
                                 label_scope=labels['audit']))
            checkpoint(completed_label_scene_count=len(training), last_scene_id=sid)
        state['costs']['T_labels_seconds'] = time.monotonic()-stage

        components = sorted({r['component_id'] for r in training})
        original_models = common.load_npz(paths['source_risk_run']/'private/models.npz')
        old_folds = {r['fold']: r for r in old.frozen.read(paths['source_risk_run']/'T-fold-index.json')}
        models, model_arrays, folds, oof_rows = {}, {}, [], []
        stage = time.monotonic()
        for heldout in components+[None]:
            name = 'full' if heldout is None else 'leave_'+heldout
            selected = [r for r in training if r['component_id'] != heldout]
            x = np.concatenate([r['saved']['features'][r['saved']['labeled']] for r in selected])
            e = np.concatenate([r['saved']['errors_m'][r['saved']['labeled']] for r in selected])
            w = np.concatenate([r['saved']['sample_weight'][r['saved']['labeled']] for r in selected])
            checkpoint(phase='T_fit', fold=name)
            model = selection.fit_meter_model(x, e, w)
            models[name] = model
            state['analytic_head_fits'] += 1
            state['ridge_system_solves'] += int(not model['no_labels_fallback'])
            state['no_label_fallback_heads'] += int(model['no_labels_fallback'])
            for key in ('mean', 'std'):
                bind_equal(model[key], original_models[name+'__QUADRATIC__'+key], 'same original133 '+name+'/'+key)
            fold = dict(fold=name, heldout_component=heldout,
                training_components=sorted({r['component_id'] for r in selected}),
                training_scene_ids=[r['scene_id'] for r in selected],
                labeled_source_query_count=len(e), training_weight_sum=float(w.sum()))
            if fold != old_folds[name]:
                raise ValueError('original133 fold membership/weights differs')
            folds.append(fold)
            model_arrays.update({name+'__QUADRATIC__'+key: np.asarray(value) for key, value in model.items()})
            np.savez_compressed(output/'private/models.npz', **model_arrays)
            old.frozen.write(output/'T-fold-index.json', folds)
            if heldout is not None:
                for row in training:
                    if row['component_id'] != heldout:
                        continue
                    sid, pack = row['scene_id'], row['saved']
                    old_score = common.load_npz(paths['source_risk_run']/'private'/f'oof-{sid}.npz',
                                                ('QUADRATIC__scores',))['QUADRATIC__scores']
                    state['original_oof_read_count'] += 1
                    meter_score = risk.predict_score(model, pack['features'].reshape(-1, 10)).reshape(pack['available'].shape)
                    np.savez_compressed(output/'private'/f'oof-{sid}.npz', LOG__scores=old_score,
                        METER__scores=meter_score, fold=np.asarray(name), component_id=np.asarray(heldout))
                    meter_labels = dict(row['labels'], y=pack['y_meter'])
                    oof_rows.append(dict(scene_id=sid, component_id=heldout,
                        heads=dict(LOG=source.score_diagnostic(pack, row['labels'], old_score),
                                   METER=source.score_diagnostic(pack, meter_labels, meter_score))))
            checkpoint()
        if state['analytic_head_fits'] != 13:
            raise ValueError('exactly13 single-Q fit attempts required')
        state['costs']['T_fit_oof_seconds'] = time.monotonic()-stage
        oof = dict(targets=dict(LOG='log1p(radial_error_m/1000)', METER=selection.TARGET),
            heads={h: source.aggregate_diagnostics([r['heads'][h] for r in oof_rows]) for h in ('LOG', 'METER')},
            scenes=oof_rows, components={c: {h: source.aggregate_diagnostics([r['heads'][h] for r in oof_rows if r['component_id'] == c])
                       for h in ('LOG', 'METER')} for c in components},
            scope=[dict(scene_id=r['scene_id'], component_id=r['component_id'], **r['label_scope']) for r in training])
        old.frozen.write(output/'T-oof.json', oof)
        checkpoint(phase='T_recovery', fold=None, full_T_models_frozen_before_V=True)

        summaries, diagnostics = {}, []
        for split, ids, methods, primary in (('T_core', tids, T_METHODS, T_PRIMARY),
                                             ('V_select', vids, V_METHODS, V_PRIMARY)):
            rows, arrays, stage = [], [], time.monotonic()
            checkpoint(phase=split+'_prediction_evaluation')
            for sid in ids:
                members = common.membership(paths['value_run'], paths['partition_run'], split, sid)
                motion, groups, mean, pack, saved = common.physical(paths['native_run'], split, sid, members)
                state['feature_scene_read_count'] += 1
                if split == 'T_core':
                    model = models['leave_'+str(manifest['components'][sid])]
                    stored_score = common.load_npz(output/'private'/f'oof-{sid}.npz')
                    scores = {h: stored_score[h+'__scores'] for h in ('LOG', 'METER')}
                    policies = {h: policy(motion, groups, mean, pack, scores[h]) for h in ('LOG', 'METER')}
                    predictions = {T_LOG: policies['LOG']['group_states'], T_METER: policies['METER']['group_states']}
                    policy_names = {'LOG': T_LOG, 'METER': T_METER}
                    state['newly_recovered_old_T_log_scene_count'] += 1
                else:
                    model = models['full']
                    scores = {'METER': risk.predict_score(model, pack['features'].reshape(-1, 10)).reshape(pack['available'].shape)}
                    policies = {'METER': policy(motion, groups, mean, pack, scores['METER'])}
                    predictions = {V_METER: policies['METER']['group_states']}
                    policy_names = {'METER': V_METER}
                state['newly_recovered_policy_scene_count'] += len(policies)
                for h, name in policy_names.items():
                    saved[name+'__scores'] = scores[h]
                    saved.update({name+'__'+key: value for key, value in policies[h].items()})
                np.savez_compressed(output/'private'/f'policy-{split}-{sid}.npz', **saved)
                # New scores, weights and physical states are fixed before old
                # artifact references or original evaluation cache is opened.
                quality = common.load_artifacts(paths['quality_run']/'private'/f'states-{split}-{sid}.npz',
                    (EQUAL, NEAREST), template=EQUAL)
                retained = quality if split == 'T_core' else dict(common.load_artifacts(
                    paths['source_risk_run']/'private'/f'states-{sid}.npz', source.METHODS), **{NEAREST: quality[NEAREST]})
                if split == 'V_select':
                    for key in quality[EQUAL]:
                        bind_equal(quality[EQUAL][key], retained[EQUAL][key], '136/133 source-equal '+key)
                    bind_equal(mean, retained['WS_MEAN']['group_states'], 'original Mean physical state')
                predictions.update({name: value['group_states'] for name, value in retained.items()})
                ev = common.evaluation(paths['native_run'], split, sid, motion)
                state['evaluation_scene_read_count'] += 1
                row, local_arrays, artifacts = common.evaluate_predictions(motion, groups, pack, ev, predictions,
                    retained, method_names=methods, split=split)
                labels = source.training_labels(pack, groups, ev)
                row.update(scene_id=sid, component_id=str(manifest['components'][sid]), split=split,
                    node_count=len(motion['states']), source_support_scope=source.support_counts(pack),
                    reference_diagnostic_scope=labels['audit'], feature_audit=pack['audit'],
                    score_diagnostics={h: selection.score_stats(scores[h], pack, policies[h]['source_weights'],
                        model if h == 'METER' else None) for h in policies},
                    fixed_population_invariants_verified=True, original_controls_exact_verified=True)
                if split == 'T_core':
                    original = common.load_npz(paths['mixture_run']/'private'/f'scene-{sid}.npz')
                    data = selection.build_diagnostic(original, dict(pack, **{k: labels[k] for k in ('labeled', 'errors_m')}),
                        scores, policies, motion, ev, members, groups)
                    np.savez_compressed(output/'private'/f'diagnostic-{sid}.npz', **data)
                    row['D2_diagnostic'] = selection.summarize_diagnostic(data)
                    diagnostics.append(dict(scene_id=sid, component_id=row['component_id'], data=data))
                common.save_artifacts(output/'private'/f'states-{split}-{sid}.npz', artifacts)
                old.frozen.write(output/'scenes'/split/f'scene-{sid}.json.gz', row)
                rows.append(row)
                arrays.append(local_arrays)
                checkpoint(**{('completed_training_scene_count' if split == 'T_core' else 'completed_scene_count'): len(rows)},
                           last_scene_id=sid)
            if any(len(r['methods']) != len(methods) or len(r['pairwise']) != len(methods)*(len(methods)-1)//2 for r in rows):
                raise ValueError('incomplete fixed arm/pair population')
            summaries[split] = common.summarize(rows, arrays, method_names=methods, primary_pairs=(primary,))
            old.frozen.write(output/('T-scene-summary.json.gz' if split == 'T_core' else 'scene-summary.json.gz'), rows)
            state['costs'][split+'_recovery_evaluation_seconds'] = time.monotonic()-stage
            if split == 'T_core':
                old.frozen.write(output/'T-summary.json', summaries[split])
                old.frozen.write(output/'T-selection-diagnostics.json', combined_diagnostic(diagnostics))
        checkpoint(phase='finalizing')
        summary = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **summaries['V_select'],
            primary_comparison=V_PRIMARY, training_primary_comparison=T_PRIMARY,
            target_kind=selection.TARGET, analytic_head_fits=13, costs=state['costs'],
            elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'],
            limitations=['Only target transform changes at fixed features/capacity/regularization/hard readout',
                'Finite scores may be negative; no health/Cov/calibration or Bayes-optimal claim',
                'Head-only component OOF, not full-chain independent generalization',
                'Actual K2, shared T/V components, historically developed V32',
                'Reference/anonymous roles only for training or declared evaluation, never source-priority input',
                'Old full T LOG recovery is newly executed and bound to original134 D2; no old fit repeated',
                'No new state estimate, association, neural update, oracle or combination with140'])
        old.frozen.write(output/'summary.json', summary)
        checkpoint(status='COMPLETED', exit_code=0, phase='completed', finished_at=datetime.now(timezone.utc).isoformat())
        return summary
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-start,
            failed_current_scene_partial_work='not fully counted', finished_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in INPUTS+('manifest', 'output'):
        parser.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
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
        raise SystemExit(1)


if __name__ == '__main__':
    main()
