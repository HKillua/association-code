"""133: fixed T32/V32 shared source error scores; general-K contribution pilot.

Reads only completed 127 native caches, 125 saved controls/T grouping and 114 WS.
No new neural forward, archive read, registration or sealed-test access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
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
from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.group_prediction import emit_group_outputs
from tools.analysis import query_contribution_control as qc

old = qc.old
IDEA = 'IDEA-SOURCE-RISK-20261030'
IMPORTED = ('WS_MEAN', 'WS_NEAREST_SUPPORTED', 'WS_LOW_SOURCE_SUPPORTED',
            'WS_FIXED_LOW_SOURCE_SUPPORTED', 'WS_QUADRATIC_VALUE_GATE')
POLICIES = {'WS_SOURCE_EQUAL': (None, 'equal'), 'WS_SHARED_Q_SOFT': ('QUADRATIC', 'soft'),
            'WS_SHARED_Q_HARD': ('QUADRATIC', 'hard'), 'WS_SHARED_LINEAR_SOFT': ('LINEAR', 'soft'),
            'WS_SHARED_SHUFFLED_SOFT': ('SHUFFLED', 'soft')}
METHODS = IMPORTED + tuple(POLICIES)
PRIMARY = 'WS_SHARED_Q_SOFT_vs_WS_SOURCE_EQUAL'
MOTION_KEYS = ('states', 'support', 'nearest_seconds', 'sources', 'native_count',
               'span_seconds', 'fit_rmse_m', 'ols_velocity_mps')
RAM, ADDRESS_SPACE = 6*1024**3, 12*1024**3


def load_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def artifact_load(path):
    with np.load(path, allow_pickle=False) as z:
        keys = [k[len('WS_MEAN_'):] for k in z.files if k.startswith('WS_MEAN_')]
        if not keys or 'group_states' not in keys:
            raise ValueError('missing original Mean artifact')
        return {name: {k: z[name+'_'+k] for k in keys} for name in IMPORTED}


def metadata(native, value, partition, manifest):
    nr, vr, pr = [old.frozen.read(p/'run.json') for p in (native, value, partition)]
    if any(r.get('status') != 'COMPLETED' or r.get('exit_code') != 0 for r in (nr, vr, pr)):
        raise ValueError('127/125/114 must be completed with exit0')
    if (any(Path(r['partition_run']).resolve() != partition for r in (nr, vr))
            or Path(vr['native_run']).resolve() != native
            or vr.get('completed_T_scenes') != 32 or vr.get('completed_V_scenes') != 32):
        raise ValueError('127/125 partition lineage differs')
    marker = old.frozen.read(native/'native-cache.json')
    t, v = vr['training_scene_ids'], vr['scene_ids']
    if (marker.get('status') != 'CACHE_COMPLETED' or marker.get('scene_ids') != {'T_core': t, 'V_select': v}
            or t != manifest['splits']['T_core'][:32] or len(t) != 32 or len(v) != 32
            or len(set(t)) != 32 or len(set(v)) != 32 or set(t) & set(v)
            or not set(v).issubset(manifest['splits']['V_select'])):
        raise ValueError('original completed T32/V32 cache populations differ')
    if len({manifest['components'][s] for s in t}) != 12 or len({manifest['components'][s] for s in v}) != 9:
        raise ValueError('original T12/V9 components required')
    return t, v


def physical(native, split, sid, membership):
    m = load_npz(native/'private'/'native-motion'/split/f'scene-{sid}.npz', MOTION_KEYS+('node_ids', 'query_seconds'))
    if m['states'].shape[1:] != (64, 4):
        raise ValueError('original Q64 four-state input required')
    groups = old.groups_from_membership(membership, len(m['states']))
    first = np.asarray([g[0] for g in groups], dtype=np.int64)
    mean = ordinary_fusion(m['states'], m['nearest_seconds'], m['support'], groups)['predictions']['mean_supported'][first]
    pack = risk.build_source_features(**{k: m[k] for k in MOTION_KEYS}, groups=groups)
    saved = {k: a for k, a in pack.items() if isinstance(a, np.ndarray)}
    saved.update(membership=membership, node_ids=m['node_ids'], query_seconds=m['query_seconds'])
    return m, groups, mean, pack, saved


def evaluation(native, split, sid, motion):
    ev = load_npz(native/'private'/'native-evaluation'/split/f'scene-{sid}.npz')
    for key in ('node_ids', 'query_seconds'):
        old._same(motion[key], ev[key], 'native motion/evaluation '+key)
    return ev


def training_labels(pack, groups, ev):
    return risk.source_labels(pack, groups, ev['identities'], ev['target_ids'], ev['target_states'], ev['target_valid'])


def support_counts(pack):
    counts = np.asarray([pack['available'][a:b].sum(0) for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:])])
    return {'zero': int((counts == 0).sum()), 'one': int((counts == 1).sum()), 'at_least_two': int((counts >= 2).sum())}


def score_diagnostic(pack, labels, scores):
    use, w = labels['labeled'], labels['sample_weight']
    delta = scores[use]-labels['y'][use]
    total = float(w[use].sum())
    correct = queries = 0.
    hard = risk.source_weights(scores, pack['available'], pack['group_offsets'], kind='hard')
    for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:]):
        qmask = use[a:b].any(0)
        if qmask.any():
            error = np.where(use[a:b, qmask], labels['errors_m'][a:b, qmask], np.inf)
            optimal = error == error.min(0)
            correct += float((hard[a:b, qmask]*optimal).sum())
            queries += int(qmask.sum())
    return dict(labeled_source_query_count=int(use.sum()), training_weight_sum=total,
        weighted_squared_error_sum=float((w[use]*delta**2).sum()),
        weighted_absolute_error_sum=float((w[use]*np.abs(delta)).sum()),
        weighted_mse=float((w[use]*delta**2).sum()/total) if total else None,
        weighted_mae=float((w[use]*np.abs(delta)).sum()/total) if total else None,
        selection_credit_sum=correct, selection_query_count=int(queries),
        source_selection_accuracy=correct/queries if queries else None)


def aggregate_diagnostics(rows):
    sums = {k: sum(r[k] for r in rows) for k in ('labeled_source_query_count', 'training_weight_sum',
        'weighted_squared_error_sum', 'weighted_absolute_error_sum', 'selection_credit_sum', 'selection_query_count')}
    w, n = sums['training_weight_sum'], sums['selection_query_count']
    sums.update(weighted_mse=sums['weighted_squared_error_sum']/w if w else None,
        weighted_mae=sums['weighted_absolute_error_sum']/w if w else None,
        source_selection_accuracy=sums['selection_credit_sum']/n if n else None)
    return sums


def contribution_summary(pack, weights):
    maxima, entropies, changes, equal_l1 = [], [], [], []
    slot_means = []
    for a, b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:]):
        available, w = pack['available'][a:b], weights[a:b]
        live = available.any(0)
        if live.any():
            maxima.extend(w[:, live].max(0).tolist())
            entropies.extend((-np.sum(w[:, live]*np.log(np.maximum(w[:, live], 1e-300)), axis=0)).tolist())
            equal = available[:, live]/available[:, live].sum(0)
            equal_l1.extend(np.abs(w[:, live]-equal).sum(0).tolist())
            slot_means.extend(w[:, live].mean(1).tolist())
        adjacent = live[1:] & live[:-1]
        changes.extend(np.abs(np.diff(w, axis=1)).sum(0)[adjacent].tolist())
    return dict(supported_group_queries=len(maxima), mean_max_source_mass=float(np.mean(maxima)) if maxima else None,
        mean_entropy=float(np.mean(entropies)) if entropies else None,
        mean_l1_from_source_equal=float(np.mean(equal_l1)) if equal_l1 else None,
        adjacent_supported_query_pairs=len(changes), mean_adjacent_l1_change=float(np.mean(changes)) if changes else None,
        source_group_slot_mean_mass_quantiles=np.quantile(slot_means, [0,.25,.5,.75,1]).tolist() if slot_means else [])


def evaluate_predictions(motion, groups, pack, ev, predictions, retained):
    # All predictions exist before this evaluation-only reference/identity use.
    fragment = dict(query_seconds=motion['query_seconds'], identities=ev['identities'],
                    support=motion['support'], nearest_seconds=motion['nearest_seconds'])
    regions = old.fixed_regions(fragment, ev['target_ids'])
    # Keep zero/one/multi-source states in the full population; strata do not filter outputs.
    group_counts = np.asarray([pack['available'][a:b].sum(0) for a,b in zip(pack['group_offsets'][:-1],pack['group_offsets'][1:])])
    counts = np.full((len(ev['target_ids']), motion['states'].shape[1]), -1, np.int64)
    registry = {s:i for i,s in enumerate(ev['target_ids'].tolist())}
    claimed = set()
    for gid, g in enumerate(groups):
        identity = ev['identities'][g[0]]
        if identity in registry and identity not in claimed:
            counts[registry[identity]] = group_counts[gid]; claimed.add(identity)
    regions.update(source_support_zero=counts == 0, source_support_one=counts == 1, source_support_multi=counts >= 2)
    artifacts, methods, arrays = {}, {}, {}
    for name in METHODS:
        out = emit_group_outputs(predictions[name], groups, node_count=len(motion['states']))
        report = old.evaluate_full_scene(out, node_identities=ev['identities'], target_identities=ev['target_ids'],
            target_states=ev['target_states'], target_valid=ev['target_valid'], node_observed=motion['support'],
            split='V_select', grouping_kind='predicted', correspondence_policy='anchor_identity_first',
            return_artifact=True, node_input_support=dict(support=motion['support'], nearest_seconds=motion['nearest_seconds']))
        artifact = report.pop('artifact')
        if name in retained:
            if set(artifact) != set(retained[name]):
                raise ValueError('old artifact schema differs')
            for k in artifact:
                old._same(artifact[k], retained[name][k], 'original125 exact '+name+'/'+k)
        if artifacts:
            for k in artifact:
                if k not in ('group_states', 'aligned_prediction'):
                    old._same(artifact[k], artifacts['WS_MEAN'][k], 'fixed population '+name+'/'+k)
        methods[name], arrays[name] = qc.describe(report, artifact, fragment, regions)
        artifacts[name] = artifact
    pairs, pair_arrays = {}, {}
    for base, candidate in combinations(METHODS, 2):
        key = candidate+'_vs_'+base
        pairs[key], pair_arrays[key], _ = old.compare_pair(artifacts[base], artifacts[candidate], regions)
        pairs[key].update(baseline_method=base, candidate_method=candidate, side_labels=dict(A0=base, candidate=candidate))
    return dict(methods=methods, pairwise=pairs), dict(methods=arrays, pairs=pair_arrays), artifacts


def component_bootstrap(rows, arrays):
    components = sorted({r['component_id'] for r in rows})
    sums, counts = [], []
    for c in components:
        delta = np.concatenate([a['pairs'][PRIMARY]['all']['delta'] for r,a in zip(rows, arrays) if r['component_id'] == c])
        sums.append(float(delta.sum())); counts.append(len(delta))
    sums, counts = np.asarray(sums), np.asarray(counts)
    draws = np.random.default_rng(20).integers(0,len(components),size=(2000,len(components)))
    denominator = counts[draws].sum(1)
    if not np.all(denominator > 0):
        raise ValueError('bootstrap resample without evaluable points')
    deltas = sums[draws].sum(1)/denominator
    return dict(comparison=PRIMARY, components=components, component_count=len(components), resamples=2000, seed=20,
        estimator='common-position-point-weighted MAE difference; candidate minus baseline',
        point_estimate_m=float(sums.sum()/counts.sum()), percentile95_m=np.quantile(deltas,[.025,.975]).tolist(),
        independent_test=False)


def run(args):
    paths = {k: Path(getattr(args,k)).resolve() for k in ('native_run','value_run','partition_run','output')}
    output = paths['output']
    for p in [v for k,v in paths.items() if k != 'output']:
        if output == p or output in p.parents or p in output.parents:
            raise ValueError('new output overlaps frozen input')
    if not 1 <= args.cpu_threads <= 4:
        raise ValueError('CPU1..4 required')
    output.mkdir(parents=True,exist_ok=False); (output/'private').mkdir(); (output/'scenes').mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0', started_at=datetime.now(timezone.utc).isoformat(),
        **{k:str(v) for k,v in paths.items() if k != 'output'}, manifest=str(Path(args.manifest).resolve()),
        methods=list(METHODS), imported_methods=list(IMPORTED), analytic_head_fits=0, ridge_system_solves=0,
        no_label_fallback_heads=0, completed_training_scene_count=0, completed_scene_count=0,
        teacher_forward_calls=0, neural_models_constructed=0, optimizer_updates=0, archive_read=False, csv_read=False,
        test_opened=False, cpu_threads=args.cpu_threads, gpu_used=False, time_policy='advisory',
        source_features_frozen_before_labels=True, full_T_models_frozen_before_V=False, costs={})
    def check(**extra):
        state.update(extra, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes())
        state['advisory_exceeded'] = state['elapsed_seconds'] > 1800
        old.frozen.write(output/'run.json',state)
        if state['advisory_exceeded'] and not (output/'time_advisory.json').exists():
            old.frozen.write(output/'time_advisory.json', dict(elapsed_seconds=state['elapsed_seconds'],action='continue fixed39 fits and originalT32/V32 only'))
        if state['peak_rss_bytes'] > RAM:
            raise MemoryError('6GiB resident budget exceeded')
    try:
        check(phase='binding')
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([ADDRESS_SPACE]+[v for v in (soft,hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS,(limit,hard))
        manifest = old.frozen.read(args.manifest)
        tids, vids = metadata(paths['native_run'], paths['value_run'], paths['partition_run'], manifest)
        check(training_scene_ids=tids, scene_ids=vids, phase='T_feature_labels')
        training=[]; stage=time.monotonic()
        for sid in tids:
            membership=load_npz(paths['value_run']/'private'/f'T-features-{sid}.npz',('membership',))['membership']
            m,g,mean,pack,saved=physical(paths['native_run'],'T_core',sid,membership)
            saved['component_id']=np.asarray(str(manifest['components'][sid]))
            np.savez_compressed(output/'private'/f'training-{sid}.npz',**saved)
            ev=evaluation(paths['native_run'],'T_core',sid,m)
            labels=training_labels(pack,g,ev)
            saved.update({k:v for k,v in labels.items() if isinstance(v,np.ndarray)})
            np.savez_compressed(output/'private'/f'training-{sid}.npz',**saved)
            training.append(dict(scene_id=sid,component_id=str(manifest['components'][sid]),pack=pack,labels=labels,
                scope=support_counts(pack),feature_audit=pack['audit']))
            check(completed_training_scene_count=len(training),last_scene_id=sid)
        state['costs']['T_feature_labels_seconds']=time.monotonic()-stage
        components=sorted({r['component_id'] for r in training}); model_arrays={}; fold_index=[]; oof_rows=[]
        stage=time.monotonic()
        for heldout in components+[None]:
            name='full' if heldout is None else 'leave_'+heldout
            selected=[r for r in training if r['component_id'] != heldout]
            x=np.concatenate([r['pack']['features'][r['labels']['labeled']] for r in selected])
            y=np.concatenate([r['labels']['y'][r['labels']['labeled']] for r in selected])
            w=np.concatenate([r['labels']['sample_weight'][r['labels']['labeled']] for r in selected])
            check(phase='T_fit',fold=name)
            models=risk.fit_score_models(x,y,w)
            for head,model in models.items():
                state['analytic_head_fits']+=1
                state['ridge_system_solves']+=int(not model['no_labels_fallback'])
                state['no_label_fallback_heads']+=int(model['no_labels_fallback'])
                model_arrays.update({name+'__'+head+'__'+k:np.asarray(v) for k,v in model.items()})
            np.savez_compressed(output/'private'/'models.npz',**model_arrays)
            fold_index.append(dict(fold=name,heldout_component=heldout,training_components=sorted({r['component_id'] for r in selected}),
                training_scene_ids=[r['scene_id'] for r in selected],labeled_source_query_count=len(y),training_weight_sum=float(w.sum())))
            old.frozen.write(output/'T-fold-index.json',fold_index)
            if heldout is not None:
                for row in training:
                    if row['component_id'] != heldout: continue
                    pack=row['pack']; scores={h:risk.predict_score(model,pack['features'].reshape(-1,10)).reshape(pack['available'].shape)
                        for h,model in models.items()}
                    np.savez_compressed(output/'private'/f"oof-{row['scene_id']}.npz",**{h+'__scores':s for h,s in scores.items()})
                    oof_rows.append(dict(scene_id=row['scene_id'],component_id=heldout,
                        heads={h:score_diagnostic(pack,row['labels'],s) for h,s in scores.items()}))
            check()
        if state['analytic_head_fits'] != 39: raise ValueError('exactly39 fit attempts required')
        state['costs']['T_fit_oof_seconds']=time.monotonic()-stage
        oof=dict(heads={h:aggregate_diagnostics([r['heads'][h] for r in oof_rows]) for h in risk.HEADS},scenes=oof_rows,
            components={c:{h:aggregate_diagnostics([r['heads'][h] for r in oof_rows if r['component_id']==c]) for h in risk.HEADS} for c in components},
            scope=[{k:r[k] for k in ('scene_id','component_id','scope','feature_audit')} for r in training])
        old.frozen.write(output/'T-oof.json',oof)
        check(full_T_models_frozen_before_V=True,phase='V_prediction_evaluation',fold=None)
        rows=[]; arrays=[]; stage=time.monotonic()
        for sid in vids:
            membership=np.asarray(old.frozen.read(paths['partition_run']/'private'/f'scene-{sid}.json.gz')['memberships']['WS'],np.int64)
            m,g,mean,pack,saved=physical(paths['native_run'],'V_select',sid,membership)
            scores={h:risk.predict_score(model,pack['features'].reshape(-1,10)).reshape(pack['available'].shape) for h,model in models.items()}
            predictions={}; wa={}
            for method,(head,kind) in POLICIES.items():
                score=np.zeros(pack['available'].shape) if head is None else scores[head]
                result=risk.recover_source_policy(states=m['states'],support=m['support'],groups=g,pack=pack,scores=score,kind=kind,mean_states=mean)
                predictions[method]=result['group_states']
                for k in ('source_weights','node_weights'): saved[method+'__'+k]=result[k]
                saved[method+'__scores']=score
                wa[method]=contribution_summary(pack,result['source_weights'])
            np.savez_compressed(output/'private'/f'policy-{sid}.npz',**saved)
            # Every new policy has been fixed before reference or old control artifact reads.
            retained=artifact_load(paths['value_run']/'private'/f'states-{sid}.npz')
            old._same(retained['WS_MEAN']['node_group_indices'],membership,'125/114 WS')
            old._same(mean,retained['WS_MEAN']['group_states'],'native exactMean')
            predictions.update({k:v['group_states'] for k,v in retained.items()})
            ev=evaluation(paths['native_run'],'V_select',sid,m)
            row,a,artifacts=evaluate_predictions(m,g,pack,ev,predictions,retained)
            labels=training_labels(pack,g,ev)  # diagnostic only, never fits or changes V policy
            row.update(scene_id=sid,component_id=str(manifest['components'][sid]),node_count=len(m['states']),
                fixed_R_population_invariants_verified=True,original125_exact_replays_verified=True,
                source_support_scope=support_counts(pack),feature_audit=pack['audit'],source_contributions=wa,
                source_score_diagnostics={h:score_diagnostic(pack,labels,s) for h,s in scores.items()})
            np.savez_compressed(output/'private'/f'states-{sid}.npz',**{n+'_'+k:v for n,d in artifacts.items() for k,v in d.items()})
            old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz',row)
            rows.append(row); arrays.append(a)
            check(completed_scene_count=len(rows),last_scene_id=sid)
        state['costs']['V_prediction_evaluation_seconds']=time.monotonic()-stage
        cs=sorted({r['component_id'] for r in rows})
        def aggregate(indices):
            return qc.aggregate([rows[i] for i in indices],[arrays[i] for i in indices],method_names=METHODS)
        largest=max(cs,key=lambda c:sum(r['component_id']==c for r in rows))
        summary=dict(status='COMPLETED',idea_id=IDEA,plan_version='1.0',all_components=aggregate(range(len(rows))),
            by_component={c:aggregate([i for i,r in enumerate(rows) if r['component_id']==c]) for c in cs},
            without_largest_component=aggregate([i for i,r in enumerate(rows) if r['component_id']!=largest]),
            largest_component_id=largest,primary_comparison=PRIMARY,primary_bootstrap=component_bootstrap(rows,arrays),T_oof=oof,
            V_score_diagnostics={h:aggregate_diagnostics([r['source_score_diagnostics'][h] for r in rows]) for h in risk.HEADS},
            limitations=['Actual MTAD sources=2; general-K software checks are not K>=3 empirical evidence',
                'T LOCO isolates only new heads; upstream teacher uses allT, threshold and V32 historically developed',
                'Softmax scores are not calibrated confidence or optimal marginal contributions; log error differs from meter MAE',
                'Source internal angle averaging and final node angle recovery need not be associative',
                'Position-only pilot; no online, cross-physical-sensor or credible peer superiority claim'])
        old.frozen.write(output/'scene-summary.json.gz',rows)
        check(phase='finalizing')
        summary.update(costs=state['costs'],elapsed_seconds=state['elapsed_seconds'],peak_rss_bytes=state['peak_rss_bytes'],analytic_head_fits=state['analytic_head_fits'])
        old.frozen.write(output/'summary.json',summary)
        check(status='COMPLETED',exit_code=0,phase='completed',finished_at=datetime.now(timezone.utc).isoformat())
        return summary
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'FAILED',exit_code=1,
            error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started,finished_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'run.json',state)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-run','value-run','partition-run','manifest','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--time-policy',choices=('advisory',),default='advisory')
    args=p.parse_args()
    def interrupted(signum,frame): raise InterruptedError(f'signal{signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try: run(args)
    except BaseException: traceback.print_exc();return 1
    print(f'COMPLETED: {args.output}/summary.json',flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
