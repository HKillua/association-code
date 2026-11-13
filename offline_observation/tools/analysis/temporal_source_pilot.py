"""149 zero-support candidates /150 temporal source evidence, fixed T32/V32.

Each mode is an independent fixed attempt.150 consumes only149's physical
descriptor files, never its new policies, results or validation selections.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import resource
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from association.evaluation import source_risk as risk
from association.evaluation import native_temporal as temporal
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source

old = common.old
BASE = 'WS_SHARED_Q_HARD'
ZERO = ('WS_Q_ZERO_EQUAL_OLS', 'WS_Q_ZERO_NEAREST_OLS', 'WS_Q_ZERO_SCALE_OLS')
EXT = ('WS_TEMPORAL_Q_HARD', 'WS_SHUFFLED_TEMPORAL_Q_HARD')
IDEAS = {'zero': 'IDEA-UNSUPPORTED-CANDIDATES-20261030', 'temporal': 'IDEA-TEMPORAL-SOURCE-EVIDENCE-20261030'}


def physical(paths, split, sid):
    membership = common.membership(paths['value_run'], paths['partition_run'], split, sid)
    return (*common.physical(paths['native_run'], split, sid, membership)[:4], membership)


def old_scores(paths, split, sid):
    path = paths['source_risk_run']/'private'
    return common.load_npz(path/(f'oof-{sid}.npz' if split == 'T_core' else f'policy-{sid}.npz'),
        ('QUADRATIC__scores',) if split == 'T_core' else (BASE+'__scores',))[
            'QUADRATIC__scores' if split == 'T_core' else BASE+'__scores']


def recover(motion, groups, pack, scores, mean):
    return risk.recover_source_policy(states=motion['states'], support=motion['support'],
        groups=groups, pack=pack, scores=scores, kind='hard', mean_states=mean)['group_states']


def add_harm(row, arrays):
    for key, value in row['pairwise'].items():
        d = arrays['pairs'][key]['all']['delta']
        value['actual_harm'] = dict(new_harm_over_1km=int((d > 1000).sum()),
            max_error_increase_m=float(max(0., d.max())) if len(d) else None)


def summarize(rows, arrays, methods):
    result = common.summarize(rows, arrays, method_names=methods,
        primary_pairs=[n+'_vs_'+BASE for n in methods[1:]])
    cs = sorted(set(r['component_id'] for r in rows))
    biggest = result['largest_component_id']
    subsets = {'all_components': list(range(len(rows))),
               'without_largest_component': [i for i,r in enumerate(rows) if r['component_id'] != biggest]}
    subsets.update({'component:'+c: [i for i,r in enumerate(rows) if r['component_id'] == c] for c in cs})
    for label, ids in subsets.items():
        target = result['by_component'][label.split(':',1)[1]] if label.startswith('component:') else result[label]
        for key in target['pairwise']:
            values = [rows[i]['pairwise'][key]['actual_harm'] for i in ids]
            target['pairwise'][key]['actual_harm'] = dict(new_harm_over_1km=sum(v['new_harm_over_1km'] for v in values),
                max_error_increase_m=max(v['max_error_increase_m'] or 0. for v in values))
    return result


def extras(pack, motion, evidence):
    extra = temporal.source_extra(evidence['node_features'], motion['support'], pack)
    shuffled = extra.copy()
    use = pack['available']
    values = shuffled[use].copy()
    shuffled[use] = values[np.random.default_rng(20).permutation(len(values))]
    return dict(WS_TEMPORAL_Q_HARD=np.concatenate((pack['features'], extra), axis=-1),
                WS_SHUFFLED_TEMPORAL_Q_HARD=np.concatenate((pack['features'], shuffled), axis=-1))


def selection(pack, labels, scores, original):
    weights = risk.source_weights(scores, pack['available'], pack['group_offsets'], kind='hard')
    baseline = risk.source_weights(original, pack['available'], pack['group_offsets'], kind='hard')
    values, old_values, best = [], [], []
    for a,b in zip(pack['group_offsets'][:-1], pack['group_offsets'][1:]):
        use = labels['labeled'][a:b]
        query = use.any(0)
        errors = np.where(use, labels['errors_m'][a:b], 0.)
        values.extend((weights[a:b]*errors).sum(0)[query])
        old_values.extend((baseline[a:b]*errors).sum(0)[query])
        best.extend(np.where(use, labels['errors_m'][a:b], np.inf).min(0)[query])
    values, old_values, best = map(np.asarray, (values, old_values, best))
    bad, old_bad = values-best > 1e-6, old_values-best > 1e-6
    return dict(eligible_group_queries=len(values), wrong=int(bad.sum()), baseline_wrong=int(old_bad.sum()),
        rescued=int((old_bad & ~bad).sum()), newly_wrong=int((~old_bad & bad).sum()),
        excess_sum_m=float((values-best).sum()), excess_over_1km_count=int((values-best > 1000).sum()),
        baseline_excess_sum_m=float((old_values-best).sum()))


def run(args):
    paths = {k: Path(getattr(args,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')}
    output = paths['output']; output.mkdir(exist_ok=False); (output/'private').mkdir(); (output/'scenes').mkdir()
    started = time.monotonic(); mode = args.mode
    methods = (BASE, *(ZERO if mode == 'zero' else EXT))
    state = dict(status='RUNNING', idea_id=IDEAS[mode], plan_version='1.0', mode=mode,
        started_at=datetime.now(timezone.utc).isoformat(), analytic_head_fits=0,
        optimizer_updates=0, teacher_forward_calls=0, neural_models_constructed=0,
        raw_reference_read=False, test_opened=False, groups_recomputed=False,
        completed_T_scenes=0, completed_V_scenes=0, methods=list(methods),
        **{k:str(v) for k,v in paths.items() if k != 'output'})
    def check(**kw):
        state.update(kw, elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=old.frozen.peak_rss_bytes())
        old.frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > 6*1024**3: raise MemoryError('RSS budget')
    try:
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([12*1024**3]+[x for x in (soft,hard) if x != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS,(limit,hard))
        manifest = old.frozen.read(paths['manifest'])
        tids, vids = common.metadata(paths['native_run'], paths['value_run'], paths['partition_run'], paths['source_risk_run'], manifest)
        check(training_scene_ids=tids, scene_ids=vids, phase='prepare')
        datasets = {}
        if mode == 'temporal':
            previous = Path(args.evidence_run).resolve()
            prior = old.frozen.read(previous/'run.json')
            if prior['status'] != 'COMPLETED' or prior['exit_code'] != 0 or prior['idea_id'] != IDEAS['zero']:
                raise ValueError('completed149 physical descriptor cache required')
            for key in ('native_run','value_run','partition_run','source_risk_run'):
                if Path(prior[key]).resolve() != paths[key]: raise ValueError('149 lineage differs')
            train = []
            for sid in tids:
                m,g,mean,pack,_ = physical(paths, 'T_core', sid)
                evidence = common.load_npz(previous/'private'/f'evidence-T_core-{sid}.npz')
                original = common.load_npz(paths['source_risk_run']/'private'/f'training-{sid}.npz')
                old._same(pack['features'], original['features'], 'original133 T features')
                x = extras(pack,m,evidence)
                train.append(dict(sid=sid, component_id=str(manifest['components'][sid]),
                    x=x, original=original, m=m,g=g,mean=mean,pack=pack))
            model_arrays, fold_models = {}, {}
            components = sorted({r['component_id'] for r in train})
            for held in components+[None]:
                fold = 'full' if held is None else 'leave_'+held
                fold_models[fold] = {}
                chosen = [r for r in train if r['component_id'] != held]
                y = np.concatenate([r['original']['y'][r['original']['labeled']] for r in chosen])
                w = np.concatenate([r['original']['sample_weight'][r['original']['labeled']] for r in chosen])
                for name in EXT:
                    x = np.concatenate([r['x'][name][r['original']['labeled']] for r in chosen])
                    model = temporal.fit_quadratic(x,y,w); fold_models[fold][name] = model
                    model_arrays.update({fold+'__'+name+'__'+k:v for k,v in model.items()})
                    state['analytic_head_fits'] += 1
                check(phase='T_fit', fold=fold)
            np.savez_compressed(output/'private/models.npz', **model_arrays)
            check(full_T_models_frozen_before_V=True)
        for split, ids, summary_name in (('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')):
            rows, arrays = [], []
            for sid in ids:
                m,g,mean,pack,membership = physical(paths,split,sid)
                original = old_scores(paths,split,sid)
                base = recover(m,g,pack,original,mean)
                predictions, scores_by_name = {BASE:base}, {BASE:original}
                if mode == 'zero':
                    cache = common.load_npz(paths['native_run']/'private/native-events'/split/f'scene-{sid}.npz')
                    for key in ('node_ids','sources','query_seconds'): old._same(cache[key],m[key], 'native '+key)
                    evidence = temporal.temporal_evidence(cache)
                    np.savez_compressed(output/'private'/f'evidence-{split}-{sid}.npz',**evidence,
                        node_ids=m['node_ids'], query_seconds=m['query_seconds'])
                    policy_audit = {}
                    for name, kind in zip(ZERO,('equal','nearest','scale')):
                        predictions[name],policy_audit[name] = temporal.unsupported_candidates(m,g,base,evidence,kind)
                else:
                    evidence = common.load_npz(previous/'private'/f'evidence-{split}-{sid}.npz')
                    for key in ('node_ids','query_seconds'): old._same(evidence[key],m[key], '149 '+key)
                    x = extras(pack,m,evidence)
                    models = fold_models['leave_'+str(manifest['components'][sid]) if split == 'T_core' else 'full']
                    for name in EXT:
                        score = temporal.predict(models[name],x[name].reshape(-1,14)).reshape(pack['available'].shape)
                        scores_by_name[name] = score
                        predictions[name] = recover(m,g,pack,score,mean)
                    np.savez_compressed(output/'private'/f'policy-{split}-{sid}.npz',
                        **{n+'__scores':s for n,s in scores_by_name.items()}, membership=membership)
                # New physical outputs are fixed and saved before opening evaluation.
                np.savez_compressed(output/'private'/f'raw-{split}-{sid}.npz',**predictions)
                if split == 'V_select':
                    with np.load(paths['source_risk_run']/'private'/f'states-{sid}.npz',allow_pickle=False) as z:
                        old._same(base,z[BASE+'_group_states'],'original133 Qhard')
                ev = common.evaluation(paths['native_run'],split,sid,m)
                row,arr,artifacts = common.evaluate_predictions(m,g,pack,ev,predictions,{},method_names=methods,
                    split=split,prediction_speed_unit='kn')
                add_harm(row,arr)
                if mode == 'zero': row['policy_audit'] = policy_audit
                else:
                    labels = source.training_labels(pack,g,ev)
                    row['selection'] = {n:selection(pack,labels,s,original) for n,s in scores_by_name.items()}
                row.update(scene_id=sid, component_id=str(manifest['components'][sid]))
                common.save_artifacts(output/'private'/f'states-{split}-{sid}.npz',artifacts)
                old.frozen.write(output/'scenes'/f'{split}-{sid}.json.gz',row)
                rows.append(row); arrays.append(arr)
                check(**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)},phase=split)
            result = summarize(rows,arrays,methods)
            result.update(status='COMPLETED', idea_id=IDEAS[mode], plan_version='1.0', analytic_head_fits=state['analytic_head_fits'])
            if mode=='temporal':
                result['selection'] = {n:{k:sum(r['selection'][n][k] for r in rows) for k in rows[0]['selection'][n]} for n in methods}
            old.frozen.write(output/summary_name,result)
            old.frozen.write(output/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,phase='completed',finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        check(status='FAILED',exit_code=1,error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('zero','temporal'),required=True)
    for key in ('native-run','value-run','partition-run','source-risk-run','manifest','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--evidence-run',type=Path)
    parser.add_argument('--time-policy',choices=('advisory',),required=True)
    args=parser.parse_args()
    if args.mode=='temporal' and args.evidence_run is None: parser.error('temporal mode requires --evidence-run')
    run(args)
