"""047 saved-array audit; no model, original archive or threshold reselection."""
from collections import Counter
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'HandOver/results/BASELINE-TAM-NODROPOUT-20261021-047'
OLD = ROOT / 'HandOver/results/BASELINE-TAM-FROM033-20261020-035'
spec = importlib.util.spec_from_file_location('saved_association_audit', HERE/'044-045-audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
read = audit.read


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k not in ('per_scene', 'membership')}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def aligned(a, b):
    assert [r['scene_id'] for r in a] == [r['scene_id'] for r in b]
    for left, right in zip(a, b):
        for key in ('pairs', 'labels', 'sources', 'node_ids'):
            assert np.array_equal(left[key], right[key]), (left['scene_id'], key)
        assert left['component_id'] == right['component_id']


def caches(path, support):
    result = {}
    with tarfile.open(path) as archive:
        for member in archive.getnames():
            if not member.endswith('/index.json'):
                continue
            index = json.load(archive.extractfile(member))
            rows = []
            for entry in index['records']:
                with np.load(io.BytesIO(archive.extractfile(str(Path(member).parent/entry['file'])).read())) as z:
                    r = {k: z[k].copy() for k in z.files}
                r.update(scene_id=entry['scene_id'], component_id=entry['component_id'],
                         node_ids=np.asarray(entry['node_ids']))
                r['labels'] = r['labels'].astype(bool)
                assert np.isfinite(r['scores']).all()
                n = len(r['node_ids'])
                assert np.array_equal(r['pairs'], np.column_stack(np.triu_indices(n, 1)))
                p = r['pairs']
                assert np.array_equal(r['labels'], r['identities'][p[:, 0]] == r['identities'][p[:, 1]])
                # Same node/pair arrays checked below; support counts come from
                # the already returned 045 metadata, not new preparation.
                r['joint_support_queries'] = support[r['scene_id']]
                rows.append(r)
            result[index['binding']['update']] = (rows, index['binding'])
    return result


def compare_saved(actual, expected):
    keys = {'tp':'tp', 'fp':'fp', 'fn':'fn', 'tn':'tn', 'macro_f1':'macro_f1',
            'micro_f1':'micro_f1', 'precision':'micro_precision', 'recall':'micro_recall'}
    for a, b in keys.items():
        assert np.isclose(actual[a], expected[b], rtol=1e-12, atol=1e-12), (a, actual[a], expected[b])
    lookup = {r['scene_id']: r for r in expected['per_scene']}
    for row in actual['per_scene']:
        original = lookup[row['scene_id']]
        for key in ('tp','fp','fn','tn','f1'):
            assert np.isclose(row[key], original[key], rtol=1e-12, atol=1e-12)


def window(rows):
    names = ('positive_distance_mean', 'negative_distance_mean', 'positive_class_loss_mean',
             'negative_class_loss_mean', 'negative_fraction_inside_margin')
    d = {name: float(np.mean([r['tam_train_batch'][name] for r in rows])) for name in names}
    # Every row has 32 per class: a window mean is also presentation-weighted.
    d.update(first_update=rows[0]['update'], last_update=rows[-1]['update'],
             positive_presentations=len(rows)*32, negative_presentations=len(rows)*32,
             loss_mean=float(np.mean([r['loss'] for r in rows])),
             loss_p10_p50_p90=np.quantile([r['loss'] for r in rows], [.1,.5,.9]).tolist(),
             gradient_norm_mean=float(np.mean([r['gradient_norm'] for r in rows])),
             gradient_norm_median=float(np.median([r['gradient_norm'] for r in rows])),
             gradient_clip_fraction=float(np.mean([r['gradient_norm']>1 for r in rows])),
             positive_distance_below_negative_batch_fraction=float(np.mean([
                 r['tam_train_batch']['positive_distance_mean'] < r['tam_train_batch']['negative_distance_mean']
                 for r in rows])))
    # For CL p=1, dl_i/dd=.5 for positives and -(m-d)_+ for negatives.
    # Negative RMS follows from recorded mean(.5*hinge^2), not parameter gradients.
    d['negative_distance_derivative_rms'] = float(np.sqrt(2*d['negative_class_loss_mean']))
    d['positive_distance_derivative'] = .5
    d['negative_vs_positive_derivative_rms_ratio'] = d['negative_distance_derivative_rms']/.5
    return d


def main():
    run, summary, history = [read(SOURCE/f'{name}.json.gz') for name in ('run','summary','history')]
    status, request, config = [read(SOURCE/f'{name}.json') for name in ('status','request','actual_config')]
    old_run, old_summary, old_history = [read(OLD/f'{name}.json.gz') for name in ('run','summary','history')]
    assert status['status'] == run['status'] == 'COMPLETED'
    assert status['returncode'] == int((SOURCE/'exit_code.txt').read_text()) == 0
    assert run['phase'] == 'tam' and run['training_stages'] == ['TAM_FROZEN']
    assert run['initialization']['source_tcm_updates'] == 4000
    assert run['config'] == config['training'] and not run['test_opened'] and not summary['test_opened']
    assert config['training']['updates'] == dict(TCM=0,TAM_FROZEN=1000,TFM_FROZEN=0,JOINT=0)
    config_without_override = json.loads(json.dumps(config))
    assert config_without_override['model'].pop('tam_dropout') == 0.
    assert config_without_override == read(OLD/'actual_config.json')
    assert run['requested_model_config']['tam_dropout'] == 0.
    assert summary['normalizer'] == old_summary['normalizer']
    assert len(history) == 1000 and [r['update'] for r in history] == list(range(1,1001))
    reconstruction_errors = []
    for row, old_row in zip(history, old_history):
        assert row['stage'] == 'TAM_FROZEN' and row['pair_count'] == 64 and row['positive_pairs'] == 32
        assert row['sampling'] == old_row['sampling']  # actual pair schedule, not just seed
        d = row['tam_train_batch']
        assert d['positive_count'] == d['negative_count'] == 32
        assert d['effective_margin'] == run['effective_margin']
        assert np.isfinite([row['loss'],row['gradient_norm'],*d.values()]).all()
        assert np.isclose(d['positive_class_loss_mean'],.5*d['positive_distance_mean'],rtol=1e-7)
        loss = .5*(d['positive_class_loss_mean']+d['negative_class_loss_mean'])
        reconstruction_errors.append(abs(loss-row['loss']))
        assert np.isclose(loss,row['loss'],rtol=3e-7,atol=3e-5)
    cal=run['margin_calibration'];old_cal=old_run['margin_calibration']
    dist=np.concatenate([r['distances'] for r in cal['samples']]);labels=np.concatenate([r['labels'] for r in cal['samples']]).astype(bool)
    assert len(cal['samples'])==cal['completed_batches']==64 and len(dist)==4096 and labels.sum()==2048
    for a,b in zip(cal['samples'],old_cal['samples']):
        assert a['groups']==b['groups'] and a['labels']==b['labels']
    margin=float(np.median(dist[~labels]));assert margin==run['effective_margin']==cal['effective_margin']
    hinge=np.maximum(float(np.float32(margin))-dist[~labels],0.)
    cal_loss=.25*(dist[labels].mean()+np.square(hinge).mean())
    assert np.isclose(cal_loss,cal['loss']['total'],rtol=1e-7)
    assert np.isclose(-hinge.mean(),cal['distance_derivative']['negative']['per_example_mean'],rtol=1e-7)

    candidate_summary=read(audit.DIRS['045']/'summary.json.gz')
    baseline_summary=read(audit.DIRS['045']/'baseline.json.gz')
    comp={r['scene_id']:r['component_id'] for r in candidate_summary['selected_result']['metrics']['per_scene']}
    candidate=audit.load('045','selected',comp);ols=audit.load('045','baseline',comp)
    support={r['scene_id']:r['joint_support_queries'] for r in candidate}
    data=caches(SOURCE/'association-scores.tar.gz',support)
    old_data=caches(OLD/'association-scores.tar.gz',support)
    assert sorted(data)==sorted(old_data)==list(range(100,1001,100))
    curve=[];endpoints={};original={r['update']:r for r in summary['validation_score_records']}
    thresholds={u:r['selection']['threshold'] for u,r in original.items()}
    selected_update=summary['selection']['update'];selected_threshold=summary['selection']['threshold']
    for u in sorted(data):
        rows,binding=data[u];aligned(rows,candidate);aligned(rows,old_data[u][0]);aligned(rows,ols)
        threshold=thresholds[u];assert threshold==binding['threshold']
        result=audit.summarize(rows,threshold)
        compare_saved(result['metrics'],original[u]['association'])
        for key,target in [('same_source','itsa'),('cross_source','mstsa')]:
            compare_saved(result['strata'][key],original[u]['association'][target])
        endpoints[u]=result
        scores=np.concatenate([r['scores'] for r in rows]);y=np.concatenate([r['labels'] for r in rows])
        curve.append(dict(update=u,threshold=threshold,metrics=compact(result['metrics']),
            distance_quantiles={k:np.quantile(-scores[mask],[0,.05,.5,.95,1]).tolist() for k,mask in [('positive',y),('negative',~y)]},
            train_window=window(history[u-100:u]), validation_seconds=history[u-1]['validation_seconds']))
    assert selected_update==max(curve,key=lambda r:r['metrics']['macro_f1'])['update']==400
    compare_saved(endpoints[selected_update]['metrics'],summary['evaluation']['association'])
    assert summary['evaluation']['inference_reused'] and not summary['evaluation']['physical_fusion_evaluated']
    old_threshold={r['update']:r['selection']['threshold'] for r in old_summary['validation_score_records']}
    all_selected = {
        '047_selected':(data[selected_update][0],selected_threshold,endpoints[selected_update]),
        '047_300':(data[300][0],thresholds[300],endpoints[300]),
        '047_1000':(data[1000][0],thresholds[1000],endpoints[1000]),
        '035_selected':(old_data[old_summary['selection']['update']][0],old_summary['selection']['threshold'],None),
        '035_300':(old_data[300][0],old_threshold[300],None),
        '035_1000':(old_data[1000][0],old_threshold[1000],None),
        'OLS':(ols,baseline_summary['selection']['threshold'],None),
        '045_selected':(candidate,candidate_summary['selected']['selection']['threshold'],None)}
    for key,(rows,t,value) in list(all_selected.items()):
        all_selected[key]=(rows,t,value if value is not None else audit.summarize(rows,t))
    comparisons={}
    for a,b in [('035_selected','047_selected'),('035_300','047_300'),('035_1000','047_1000'),
                ('OLS','047_selected'),('045_selected','047_selected')]:
        ar,at,am=all_selected[a];br,bt,bm=all_selected[b]
        comparisons[b+'_minus_'+a]=audit.comparison(ar,br,at,bt,am,bm)
    population=dict(scenes=len(candidate),nodes=sum(len(r['node_ids']) for r in candidate),
        pairs=sum(len(r['labels']) for r in candidate),positives=sum(int(r['labels'].sum()) for r in candidate),components=len(set(comp.values())))
    assert population==dict(scenes=64,nodes=1635,pairs=29008,positives=3297,components=17)
    result=dict(scope='Full V development; original thresholds only; no new inference or threshold search',
        execution=dict(command=request['argv'],status=run['status'],returncode=status['returncode'],
            initialization=run['initialization'],checkpoint_cursor=run['checkpoint_cursor'],cursor=run['cursor'],
            launcher_seconds=status['elapsed_seconds'],payload_seconds=run['elapsed_seconds'],
            training_seconds=sum(r['seconds'] for r in history),calibration_seconds=cal['seconds'],
            validation_seconds=sum(r['seconds'] for r in run['validation_timings']),
            final_cached_metrics_seconds=run['final_evaluation_seconds'],test_opened=False,physical_fusion_evaluated=False),
        population=population,config=config,selected_update=selected_update,
        calibration=dict(effective_margin=margin,old035_effective_margin=old_run['effective_margin'],
            batches=64,positive_presentations=2048,negative_presentations=2048,
            loss=cal['loss'],distance_derivative=cal['distance_derivative'],distance_quantiles=cal['distance_quantiles'],
            raw_negative_hinge_fraction=float(np.mean(hinge>0)),recomputed_balanced_cl=float(cal_loss)),
        training_windows_200=[window(history[i:i+200]) for i in range(0,1000,200)],
        training_diagnostics_scope='Original pre-update forward; equal 32+32 class counts. Derivatives are w.r.t. distance, not parameters.',
        history_max_loss_reconstruction_abs_error=max(reconstruction_errors),curve=curve,
        endpoints={key:compact(value) for key,(_,_,value) in all_selected.items()},
        comparisons={key: {k:v for k,v in value.items() if k!='per_scene'}
                     for key,value in comparisons.items()},
        selected_vs035_per_scene=[dict(scene_id=r['scene_id'],
            macro_f1_delta=r['macro_f1_delta'],
            net_corrected_edges=r['wrong_to_right']-r['right_to_wrong'])
            for r in comparisons['047_selected_minus_035_selected']['per_scene']],
        grouping_scope='Same canonical-order direct-anchor grouping for all score sets, not connected-component closure; no fusion inference.',
        support_scope='Joint query support borrowed from aligned 045 returned arrays; native times/data not reread.',
        checks=dict(ten_times_64_caches_verified=True,history_1000_and_same_sampling_as035=True,
            same_calibration_pairs_as035=True,source033_normalizer_equal=True,
            configs_only_tam_dropout_diff=True,all_returned_metrics_reproduced=True,
            threshold_reselected=False,model_constructed=False,raw_archive_read=False))
    (HERE/'047-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for key,(_,_,value) in all_selected.items():
        print(key,compact(value['metrics']),compact(value['grouping']))
    for key,value in comparisons.items():
        print(key,{k:v for k,v in value.items() if k not in ('per_scene','components')})
    print('training windows',result['training_windows_200'])


if __name__=='__main__':
    main()
