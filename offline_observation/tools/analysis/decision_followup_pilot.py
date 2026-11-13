"""Independent143/144 pilots: relative decisions or actual-coordinate mixture gain."""
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
from association.evaluation import source_selection_evidence as diagnostic
from association.evaluation import relative_source_decision as relative
from association.evaluation import actual_mixture_gain as mixture
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source
from tools.analysis import source_selection_evidence_pilot as prior

old = common.old
INPUTS = ('native_run', 'value_run', 'partition_run', 'source_risk_run', 'quality_run', 'contribution_run', 'mixture_run')
T_OLD = ('WS_SOURCE_EQUAL', 'WS_SOURCE_NEAREST', 'WS_SHARED_Q_SOFT', 'WS_SHARED_Q_HARD')
V_OLD = source.METHODS+('WS_SOURCE_NEAREST',)


def primary_pairs(mode):
    names = relative.NEW if mode == 'selection' else mixture.NEW
    if mode == 'selection':
        return (names[1]+'_vs_WS_SHARED_Q_HARD', names[1]+'_vs_'+names[0])
    return (names[0]+'_vs_WS_SHARED_Q_HARD', names[1]+'_vs_'+names[0], names[1]+'_vs_WS_SHARED_Q_HARD')


def old_scores(paths, split, sid):
    if split == 'T_core':
        return common.load_npz(paths['source_risk_run']/'private'/f'oof-{sid}.npz', ('QUADRATIC__scores',))['QUADRATIC__scores']
    key = 'WS_SHARED_Q_HARD__scores'
    return common.load_npz(paths['source_risk_run']/'private'/f'policy-{sid}.npz', (key,))[key]


def actual_training_curve(motion, groups, mean, pack, scores, ev):
    context, hard, equal = mixture.context(pack, scores)
    labels = source.training_labels(pack, groups, ev)
    registry = {str(target): i for i, target in enumerate(ev['target_ids'])}
    gs, qs, ts = [], [], []
    for gid, (a, b) in enumerate(zip(pack['group_offsets'][:-1], pack['group_offsets'][1:])):
        indices = np.flatnonzero(labels['labeled'][a:b].any(0))
        if len(indices):
            gs.extend([gid]*len(indices)); qs.extend(indices.tolist())
            ts.extend([registry[str(ev['identities'][groups[gid][0]])]]*len(indices))
    gi, qi, ti = np.asarray(gs, int), np.asarray(qs, int), np.asarray(ts, int)
    truth = ev['target_states'][ti, qi, :2]
    errors = []
    for alpha in mixture.ALPHAS:
        states = mixture.recover(motion, groups, mean, pack, hard, equal, np.full(context.shape[:2], alpha))['group_states']
        errors.append(risk._position_distance(states[gi, qi, :2], truth))
    errors = np.asarray(errors).T
    return context[gi, qi], (errors-errors[:, [0]])/1000., labels


def harm_table(rows, arrays, pairs):
    def stat(d):
        return dict(count=len(d), improved=int((d < 0).sum()), harmed=int((d > 0).sum()), tied=int((d == 0).sum()),
            benefit_sum_m=float(-d[d < 0].sum()), harm_sum_m=float(d[d > 0].sum()),
            harm_gt1000_count=int((d > 1000).sum()), harm_gt1000_sum_m=float(d[d > 1000].sum()),
            maximum_harm_m=float(max(0., d.max())) if len(d) else None)
    components = sorted({r['component_id'] for r in rows})
    largest = max(components, key=lambda c: sum(r['component_id'] == c for r in rows))
    def block(indices):
        return {p: stat(np.concatenate([arrays[i]['pairs'][p]['all']['delta'] for i in indices])) for p in pairs}
    return dict(all_components=block(range(len(rows))), by_component={c: block([i for i, r in enumerate(rows) if r['component_id'] == c]) for c in components},
        without_largest_component=block([i for i, r in enumerate(rows) if r['component_id'] != largest]),
        scenes=[dict(scene_id=r['scene_id'], **block([i])) for i, r in enumerate(rows)], largest_component_id=largest)


def rename_diagnostic(value):
    if isinstance(value, dict):
        return {key.replace('METER', 'CANDIDATE').replace('meter', 'candidate'): rename_diagnostic(v) for key, v in value.items()}
    if isinstance(value, list):
        return [rename_diagnostic(v) for v in value]
    return value


def run(args):
    paths = {key: Path(getattr(args, key)).resolve() for key in INPUTS}
    output = Path(args.output).resolve()
    if any(output == p or output in p.parents or p in output.parents for p in paths.values()):
        raise ValueError('output overlaps immutable input')
    output.mkdir(parents=True, exist_ok=False); (output/'private').mkdir()
    start = time.monotonic()
    new = relative.NEW if args.mode == 'selection' else mixture.NEW
    pairs = primary_pairs(args.mode)
    idea = 'IDEA-RELATIVE-SOURCE-DECISION-20261030' if args.mode == 'selection' else 'IDEA-ACTUAL-MIXTURE-GAIN-20261030'
    state = dict(status='RUNNING', idea_id=idea, plan_version='1.0', mode=args.mode, command=sys.argv,
        started_at=datetime.now(timezone.utc).isoformat(), **{k: str(v) for k, v in paths.items()}, manifest=str(args.manifest),
        completed_label_scene_count=0, completed_training_scene_count=0, completed_scene_count=0,
        new_model_fits=0, newton_updates=0, global_alpha_selections=0, T_training_curve_recoveries=0,
        neural_updates=0, teacher_forward_calls=0, new_state_estimates=0, new_groupings=0, test_opened=False,
        gpu_used=False, time_policy='advisory', cpu_threads=args.cpu_threads, full_T_models_frozen_before_V=False, costs={})
    def check(**extra):
        state.update(extra, elapsed_seconds=time.monotonic()-start, peak_rss_bytes=old.frozen.peak_rss_bytes())
        state['advisory_exceeded'] = state['elapsed_seconds'] > 1800
        old.frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > 6*1024**3:
            raise MemoryError('6GiB RSS exceeded')
    try:
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (min([12*1024**3]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY]), hard))
        manifest = old.frozen.read(args.manifest)
        tids, vids = common.metadata(*(paths[k] for k in INPUTS[:4]), manifest)
        for key in ('quality_run', 'contribution_run'):
            record = old.frozen.read(paths[key]/'run.json')
            if record['status'] != 'COMPLETED' or record['exit_code'] != 0 or record['scene_ids'] != vids or record['training_scene_ids'] != tids:
                raise ValueError('original population '+key)
        prior_mix = old.frozen.read(paths['mixture_run']/'run.json')
        if prior_mix['status'] != 'COMPLETED' or prior_mix['exit_code'] != 0 or prior_mix['scene_ids'] != tids:
            raise ValueError('original134 completed T population required')
        check(training_scene_ids=tids, scene_ids=vids, phase='T_labels')
        training, stage = [], time.monotonic()
        for sid in tids:
            pack = common.load_npz(paths['source_risk_run']/'private'/f'training-{sid}.npz')
            component = str(manifest['components'][sid])
            old._same(pack['component_id'], np.asarray(component), '133 component')
            row = dict(scene_id=sid, component_id=component, pack=pack)
            if args.mode == 'mixture':
                members = common.membership(paths['value_run'], paths['partition_run'], 'T_core', sid)
                motion, groups, mean, current, saved = common.physical(paths['native_run'], 'T_core', sid, members)
                for key, value in saved.items():
                    old._same(value, pack[key], '133 immutable features '+key)
                ev = common.evaluation(paths['native_run'], 'T_core', sid, motion)
                context, gains, labels = actual_training_curve(motion, groups, mean, current, old_scores(paths, 'T_core', sid), ev)
                for key in ('labeled', 'errors_m', 'sample_weight', 'y'):
                    old._same(labels[key], pack[key], '133 original training '+key)
                row.update(context=context, gains=gains)
                np.savez_compressed(output/'private'/f'training-{sid}.npz', context=context, gains_km=gains)
                state['T_training_curve_recoveries'] += 9
            training.append(row)
            check(completed_label_scene_count=len(training), last_scene_id=sid)
        state['costs']['T_labels_seconds'] = time.monotonic()-stage
        components = sorted({r['component_id'] for r in training})
        originals = common.load_npz(paths['source_risk_run']/'private/models.npz')
        models, model_arrays, fold_index = {}, {}, []
        stage = time.monotonic()
        for heldout in components+[None]:
            fold = 'full' if heldout is None else 'leave_'+heldout
            selected = [r for r in training if r['component_id'] != heldout]
            check(phase='T_fit', fold=fold)
            if args.mode == 'selection':
                mean, std = (originals[fold+'__QUADRATIC__'+key] for key in ('mean', 'std'))
                result = {name: relative.fit_pair_model([r['pack'] for r in selected], mean, std, regret_weighted=weighted)
                          for name, weighted in zip(new, (False, True))}
                diagnostics = {name: {k: v for k, v in model.items() if k not in ('mean', 'std', 'beta', 'head')}
                               for name, model in result.items()}
                state['new_model_fits'] += 2
                state['newton_updates'] += sum(m['newton_updates'] for m in result.values())
            else:
                result = mixture.fit_gain_model(np.concatenate([r['context'] for r in selected]), np.concatenate([r['gains'] for r in selected]))
                diagnostics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items() if k not in ('mean', 'std', 'beta')}
                state['new_model_fits'] += 1; state['global_alpha_selections'] += 1
            models[fold] = result
            fold_index.append(dict(fold=fold, heldout_component=heldout, training_scene_ids=[r['scene_id'] for r in selected],
                training_components=sorted({r['component_id'] for r in selected}), fit_diagnostics=diagnostics))
            for name, model in (result.items() if args.mode == 'selection' else [('GAIN', result)]):
                for key in ('mean', 'std', 'beta'):
                    model_arrays[fold+'__'+name+'__'+key] = model[key]
            np.savez_compressed(output/'private/models.npz', **model_arrays)
            old.frozen.write(output/'T-fold-index.json', fold_index)
            check()
        if state['new_model_fits'] != (26 if args.mode == 'selection' else 13):
            raise ValueError('exact fixed fit count required')
        state['costs']['T_fit_seconds'] = time.monotonic()-stage
        check(phase='prediction_evaluation', full_T_models_frozen_before_V=True)
        summaries, diag = {}, {name: [] for name in new}
        for split, ids, old_names in (('T_core', tids, T_OLD), ('V_select', vids, V_OLD)):
            (output/'scenes'/split).mkdir(parents=True)
            names = old_names+new
            rows, arrays, stage = [], [], time.monotonic()
            for sid in ids:
                check(phase=split+'_prediction_evaluation', last_scene_id=sid)
                component = str(manifest['components'][sid])
                model = models['leave_'+component] if split == 'T_core' else models['full']
                members = common.membership(paths['value_run'], paths['partition_run'], split, sid)
                motion, groups, mean, pack, saved = common.physical(paths['native_run'], split, sid, members)
                scores = old_scores(paths, split, sid)
                old_policy = risk.recover_source_policy(states=motion['states'], support=motion['support'], groups=groups,
                    pack=pack, scores=scores, kind='hard', mean_states=mean)
                policies, prediction, info = {}, {}, {}
                if args.mode == 'selection':
                    new_scores = {name: relative.predict(model[name], pack['features']) for name in new}
                    for name in new:
                        policies[name] = risk.recover_source_policy(states=motion['states'], support=motion['support'], groups=groups,
                            pack=pack, scores=new_scores[name], kind='hard', mean_states=mean)
                        saved[name+'__scores'] = new_scores[name]
                else:
                    context, hard, equal = mixture.context(pack, scores)
                    conditional, gain, curvature = mixture.predict_alpha(model, context)
                    alphas = {new[0]: np.full(context.shape[:2], model['global_alpha']), new[1]: conditional}
                    live = np.asarray([pack['available'][a:b].sum(0) >= 2 for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:])])
                    for name, alpha in alphas.items():
                        alpha = np.where(live, alpha, 0.)
                        policies[name] = mixture.recover(motion, groups, mean, pack, hard, equal, alpha)
                        saved[name+'__alpha'] = alpha
                        v = alpha[live]
                        info[name] = dict(multi_source_queries=len(v), alpha_mean=float(v.mean()) if len(v) else None,
                            alpha_zero_count=int((v == 0).sum()), alpha_one_count=int((v == 1).sum()),
                            alpha_interior_count=int(((v > 0) & (v < 1)).sum()),
                            alpha_quantiles=np.quantile(v, [0, .25, .5, .75, 1]).tolist() if len(v) else [])
                    saved['conditional_predicted_gain_km'], saved['conditional_curvature'] = gain, curvature
                for name in new:
                    prediction[name] = policies[name]['group_states']
                    unchanged = np.asarray([pack['available'][a:b].sum(0) <= 1
                        for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:])])
                    old._same(prediction[name][unchanged], old_policy['group_states'][unchanged], 'zero/one-source unchanged '+name)
                    for key, value in policies[name].items():
                        if isinstance(value, np.ndarray): saved[name+'__'+key] = value
                np.savez_compressed(output/'private'/f'policy-{split}-{sid}.npz', **saved)
                # New actions and physical predictions are fixed before reference artifacts.
                control_file = paths['contribution_run']/'private'/f'states-{split}-{sid}.npz' if split == 'T_core' else paths['source_risk_run']/'private'/f'states-{sid}.npz'
                control_names = ('WS_SOURCE_EQUAL', 'WS_SHARED_Q_SOFT', 'WS_SHARED_Q_HARD') if split == 'T_core' else source.METHODS
                retained = common.load_artifacts(control_file, control_names, template='WS_SOURCE_EQUAL')
                retained.update(common.load_artifacts(paths['quality_run']/'private'/f'states-{split}-{sid}.npz', ('WS_SOURCE_NEAREST',), template='WS_SOURCE_EQUAL'))
                old._same(old_policy['group_states'], retained['WS_SHARED_Q_HARD']['group_states'], 'original Qhard physical binding')
                prediction.update({name: artifact['group_states'] for name, artifact in retained.items()})
                ev = common.evaluation(paths['native_run'], split, sid, motion)
                row, local, artifacts = common.evaluate_predictions(motion, groups, pack, ev, prediction, retained, method_names=names, split=split)
                row.update(scene_id=sid, component_id=component, split=split, original_controls_exact=True, policy_diagnostics=info)
                if args.mode == 'selection' and split == 'T_core':
                    original = common.load_npz(paths['mixture_run']/'private'/f'scene-{sid}.npz')
                    labels = source.training_labels(pack, groups, ev)
                    row['selection_diagnostics'] = {}
                    for name in new:
                        data = diagnostic.build_diagnostic(original, dict(pack, **{k: labels[k] for k in ('labeled', 'errors_m')}),
                            dict(LOG=scores, METER=new_scores[name]), dict(LOG=old_policy, METER=policies[name]), motion, ev, members, groups)
                        np.savez_compressed(output/'private'/f'diagnostic-{name}-{sid}.npz', **data)
                        row['selection_diagnostics'][name] = rename_diagnostic(diagnostic.summarize_diagnostic(data))
                        diag[name].append(dict(scene_id=sid, component_id=component, data=data))
                common.save_artifacts(output/'private'/f'states-{split}-{sid}.npz', artifacts)
                old.frozen.write(output/'scenes'/split/f'scene-{sid}.json.gz', row)
                rows.append(row); arrays.append(local)
                check(**{('completed_training_scene_count' if split == 'T_core' else 'completed_scene_count'): len(rows)})
            summaries[split] = common.summarize(rows, arrays, method_names=names, primary_pairs=pairs)
            summaries[split]['additional_actual_harm'] = harm_table(rows, arrays, pairs)
            old.frozen.write(output/('T-summary.json' if split == 'T_core' else 'summary.json'), summaries[split])
            old.frozen.write(output/('T-scene-summary.json.gz' if split == 'T_core' else 'scene-summary.json.gz'), rows)
            state['costs'][split+'_seconds'] = time.monotonic()-stage
        if args.mode == 'selection':
            old.frozen.write(output/'T-selection-diagnostics.json', {name: rename_diagnostic(prior.combined_diagnostic(diag[name])) for name in new})
        check(status='COMPLETED', exit_code=0, phase='completed', finished_at=datetime.now(timezone.utc).isoformat())
        return summaries['V_select']
    except BaseException as error:
        check(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}', finished_at=datetime.now(timezone.utc).isoformat())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('selection', 'mixture'), required=True)
    for key in INPUTS+('manifest', 'output'):
        parser.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args()
    def stop(signum, frame): raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try: run(args)
    except BaseException:
        traceback.print_exc(); raise SystemExit(1)


if __name__ == '__main__':
    main()
