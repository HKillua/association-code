#!/usr/bin/env python3
"""Read saved 049 predictions only; recompute errors and output-collapse evidence.

No model loading, MTAD archive reads, training, new inference, or test-set access.
Tar members are streamed, never extracted, and unsafe paths/links are rejected.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import numpy as np

METHODS = ('oracle', 'simple_fusion_oracle', 'single_source:9001', 'single_source:9002')


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    if not len(v):
        return dict(n=0, mean=None, std=None, min=None, median=None, p90=None, max=None)
    return dict(n=len(v), mean=float(v.mean()), std=float(v.std()), min=float(v.min()),
                median=float(np.median(v)), p90=float(np.quantile(v, .9)), max=float(v.max()))


def errors(prediction, target):
    lat1, lat2 = np.deg2rad(prediction[..., 0]), np.deg2rad(target[..., 0])
    dlon = np.deg2rad((prediction[..., 1] - target[..., 1] + 180) % 360 - 180)
    h = np.clip(np.sin((lat1-lat2)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2, 0, 1)
    return {'position': 2*6371000*np.arctan2(np.sqrt(h), np.sqrt(1-h)),
            'speed': np.abs(prediction[..., 2]-target[..., 2]),
            'heading': np.abs((prediction[..., 3]-target[..., 3]+180)%360-180)}


def metric(pred, truth, pv, tv, region=None):
    error = errors(pred, truth)
    region = np.ones(truth.shape[:-1], bool) if region is None else region
    result = {}
    for name, indices in [('position', [0, 1]), ('speed', [2]), ('heading', [3])]:
        target_ok = tv[..., indices].all(-1) & np.isfinite(truth[..., indices]).all(-1) & region
        pred_ok = pv[..., indices].all(-1) & np.isfinite(pred[..., indices]).all(-1) & region
        common = target_ok & pred_ok
        e = error[name][common]
        result[name] = dict(target_count=int(target_ok.sum()), prediction_count=int(pred_ok.sum()),
                            common_count=int(common.sum()), missing_prediction_count=int((target_ok & ~pred_ok).sum()),
                            mae=float(e.mean()) if len(e) else None,
                            rmse=float(np.sqrt((e*e).mean())) if len(e) else None)
    return result


def load_endpoint(directory, update):
    prefix = f'update-{update:06d}'
    records = []
    with tarfile.open(directory/'fusion-evaluations'/(prefix+'.tar.gz')) as archive:
        names = []
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                raise ValueError(f'unsafe tar member: {member.name}')
            if member.isfile():
                names.append(member.name)
        if len(names) != len(set(names)):
            raise ValueError('duplicate tar member')
        evaluation = json.load(archive.extractfile(prefix+'/evaluation.json'))
        index = json.load(archive.extractfile(prefix+'/predictions/index.json'))
        assert index['test_opened'] is False and evaluation['test_opened'] is False
        assert len(index['scenes']) == 64
        for entry in index['scenes']:
            metadata = json.load(archive.extractfile(prefix+'/predictions/'+entry['metadata']))
            assert metadata['split'] == 'V_select' and metadata['test_opened'] is False
            with np.load(io.BytesIO(archive.extractfile(prefix+'/predictions/'+entry['npz']).read()), allow_pickle=False) as payload:
                record = {'scene_id': entry['scene_id'], 'component_id': metadata['component_id'],
                          'input_count': len(payload['scene__node_ids']), 'methods': {}}
                for name in METHODS:
                    key = metadata['methods'][name]['prefix']
                    record['methods'][name] = {suffix: payload[key+suffix] for suffix in
                        ('aligned_prediction', 'aligned_prediction_valid', 'aligned_truth', 'aligned_truth_valid')}
                key = metadata['methods']['oracle']['prefix']
                groups = payload[key+'aligned_group_indices']
                assert (groups >= 0).all(), 'oracle should cover each target'
                offsets = payload[key+'group_offsets']
                nodes = payload[key+'group_node_indices']
                record['size'] = np.diff(offsets)[groups]
                record['sources'] = np.asarray(['+'.join(sorted(set(payload['scene__sources'][nodes[offsets[g]:offsets[g+1]]]))) for g in groups])
                record['observed'] = payload[key+'aligned_observed']
                record['support'] = payload[key+'aligned_input_support_kind']
                record['nearest'] = payload[key+'aligned_nearest_input_seconds']
                record['target_ids'] = payload[key+'aligned_target_ids']
                records.append(record)
    return records, evaluation


def aggregate(records):
    return {method: {field: np.concatenate([r['methods'][method][field] for r in records]) for field in
                    ('aligned_prediction', 'aligned_prediction_valid', 'aligned_truth', 'aligned_truth_valid')} for method in METHODS}


def arrays_metric(arrays, region=None, common_valid=None):
    return metric(arrays['aligned_prediction'], arrays['aligned_truth'],
                  arrays['aligned_prediction_valid'] if common_valid is None else common_valid,
                  arrays['aligned_truth_valid'], region)


def paired_scene_summary(records, left='oracle', right='simple_fusion_oracle'):
    rows = []
    for record in records:
        a,b = record['methods'][left], record['methods'][right]
        ok = a['aligned_prediction_valid'] & b['aligned_prediction_valid']
        ma,mb=arrays_metric(a,common_valid=ok),arrays_metric(b,common_valid=ok)
        rows.append(dict(scene_id=record['scene_id'], component_id=record['component_id'], common_count=ma['position']['common_count'],
                         left_mae=ma['position']['mae'], right_mae=mb['position']['mae'],
                         difference=ma['position']['mae']-mb['position']['mae']))
    delta=np.asarray([r['difference'] for r in rows])
    # Scene bootstrap is descriptive within this fixed V, not a seed-replication CI.
    rng=np.random.default_rng(49)
    bootstrap=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    component_ids=sorted({r['component_id'] for r in rows})
    sums=np.asarray([sum(r['difference'] for r in rows if r['component_id']==cid) for cid in component_ids])
    counts=np.asarray([sum(r['component_id']==cid for r in rows) for cid in component_ids])
    draws=rng.integers(0,len(component_ids),(10000,len(component_ids)))
    clustered=sums[draws].sum(1)/counts[draws].sum(1)
    loco=(sums.sum()-sums)/(counts.sum()-counts)
    return dict(rows=rows, worse_scene_count=int((delta>0).sum()), better_scene_count=int((delta<0).sum()),
                shared_A_component_count=len(component_ids), worse_component_count=int((sums>0).sum()),
                macro_delta_component_bootstrap95=np.quantile(clustered,[.025,.975]).tolist(),
                leave_one_component_out_delta_range=[float(loco.min()),float(loco.max())],
                macro_delta=stats(delta), macro_delta_scene_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
                bootstrap_scope='fixed V exploratory uncertainty, 10000 resamples; component bootstrap accounts for shared-A; scene bootstrap descriptive only; no model seed uncertainty')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('HandOver/results/BASELINE-TFM-FROM033-20261021-049'))
    parser.add_argument('--output',type=Path,default=Path('docs/proposals/repair_before_retraining/049-audit.json'))
    args=parser.parse_args()
    read=lambda name: json.load(gzip.open(args.results/(name+'.json.gz')))
    run, history, coverage, preparation, summary = (read(x) for x in ['run','history','coverage','preparation','summary'])
    assert run['status']=='COMPLETED' and run['phase']=='tfm' and run['test_opened'] is False
    assert len(history)==300 and all(r['stage']=='TFM_FROZEN' and r['update']==i+1 for i,r in enumerate(history))
    scenes=preparation['dataset']['scenes']; train=[r for r in scenes if r['split']=='T_core']; validation=[r for r in scenes if r['split']=='V_select']
    report=dict(experiment='BASELINE-TFM-FROM033-20261021-049', evidence_status='saved predictions verified; checkpoint activations unavailable locally',
                resources='saved tar/gzip only; no MTAD ZIP, model forward, training or T_test',
                provenance=dict(status=run['status'],training_stages=run['training_stages'],updates=len(history),
                                selection_performed=run['selection_performed'],initialization=run['initialization'],
                                train_scene_count=len(train),validation_scene_count=len(validation),
                                train_fragments=sum(s['node_count'] for s in train),
                                train_positive_pairs=sum(s['positive_count'] for s in train),
                                test_opened=False), endpoints={})
    c=coverage['TFM_FROZEN']
    report['training_coverage']={k:v for k,v in c.items() if not isinstance(v,list)}
    report['training_coverage'].update(unique_scene_fraction=c['unique_scene_count']/len(train),
        unique_positive_pair_fraction=c['unique_pair_count']/sum(s['positive_count'] for s in train),
        unique_fragment_fraction=c['unique_track_count']/sum(s['node_count'] for s in train),
        repeated_pair_presentations=c['pair_presentations']-c['unique_pair_count'],
        expected_uniform_scene_coverage=len(train)*(1-(1-1/len(train))**len(history)))
    report['training_windows']=[]
    for a,b in [(0,1),(0,20),(20,100),(100,200),(200,300)]:
        window=history[a:b]
        report['training_windows'].append(dict(updates=[a+1,b],
            **{k:stats([r[k] for r in window]) for k in ['fusion_loss','gradient_norm','seconds']},
            components={k:stats([r['fusion_component_losses'][k] for r in window]) for k in ['lat','lon','vel','cou']}))
    report['training_clipped_updates']=sum(r['gradient_norm']>1 for r in history)
    report['normalizer']={k:v for k,v in summary['normalizer'].items() if k!='training_scene_ids'}
    previous_records=None
    for update in (0,100,300):
        records,evaluation=load_endpoint(args.results,update)
        data=aggregate(records)
        size=np.concatenate([r['size'] for r in records]); sources=np.concatenate([r['sources'] for r in records])
        observed=np.concatenate([r['observed'] for r in records]); support=np.concatenate([r['support'] for r in records]); nearest=np.concatenate([r['nearest'] for r in records])
        endpoint=dict(input_fragments=sum(r['input_count'] for r in records),target_groups=len(size),
                      group_size_counts=dict(sorted(Counter(map(int,size)).items())),
                      source_composition_counts=dict(Counter(sources.tolist())), own_support={},common_support={},strata={},
                      evaluation_elapsed_seconds=evaluation['elapsed_seconds'],fusion_forward_count=evaluation['fusion_forward_count'])
        for name in METHODS:
            a=data[name]
            endpoint['own_support'][name]=arrays_metric(a)
            for attribute in ('position','speed','heading'):
                expected=evaluation['fusion'][name]['all'][attribute]
                actual=endpoint['own_support'][name][attribute]
                assert actual['common_count']==expected['intersection_count']
                assert np.isclose(actual['mae'],expected['mae'],rtol=1e-12,atol=1e-8)
            if name!='oracle':
                mask=data['oracle']['aligned_prediction_valid']&a['aligned_prediction_valid']
                endpoint['common_support'][name]={m:arrays_metric(data[m],common_valid=mask) for m in ('oracle',name)}
                assert np.array_equal(a['aligned_truth'],data['oracle']['aligned_truth'],equal_nan=True)
                assert np.array_equal(a['aligned_truth_valid'],data['oracle']['aligned_truth_valid'])
        regions={'singleton':np.broadcast_to((size==1)[:,None],observed.shape),
                 'pair_only':np.broadcast_to((size==2)[:,None],observed.shape),
                 'recursive_ge3':np.broadcast_to((size>=3)[:,None],observed.shape),
                 'multi_fragment':np.broadcast_to((size>1)[:,None],observed.shape),'observed':observed,'missing':~observed,
                 'anchor_support':support==0,'helper_only_support':support==1,'unsupported':support==2,
                 'nearest_le300s':nearest<=300,'nearest_300to600s':(nearest>300)&(nearest<=600),
                 'nearest_600to1800s':(nearest>600)&(nearest<=1800),'nearest_gt1800s':(nearest>1800)&np.isfinite(nearest)}
        for source in sorted(set(sources)):
            regions['source_composition:'+source]=np.broadcast_to((sources==source)[:,None],observed.shape)
        for name,region in regions.items():
            endpoint['strata'][name]=arrays_metric(data['oracle'],region=region)
        a=data['oracle']; valid=a['aligned_prediction_valid'].all(-1)&a['aligned_truth_valid'].all(-1)
        multi=regions['multi_fragment']; raw_valid=a['aligned_prediction_valid'].all(-1)&multi
        pred=a['aligned_prediction'][raw_valid]; truth=a['aligned_truth'][valid&multi]
        endpoint['multi_fragment_output']={
            'all_prediction_point_count':len(pred),'truth_intersection_point_count':len(truth),
            'prediction_fields':{name:stats(pred[:,i]) for i,name in enumerate(('lat','lon','speed','heading'))},
            'target_fields_on_valid_points':{name:stats(truth[:,i]) for i,name in enumerate(('lat','lon','speed','heading'))},
            'exact_unique_position_count':len(np.unique(pred[:,:2],axis=0)),
            'median_within_trajectory_position_span_degrees':np.median(np.ptp(a['aligned_prediction'][size>1,:,:2],axis=1),axis=0).tolist(),
            'position_output_mean_minus_train_input_mean_in_std':((pred[:,:2].mean(0)-np.asarray(summary['normalizer']['mean'])[:2])/np.asarray(summary['normalizer']['std'])[:2]).tolist(),
            'group_size_output_diagnostics':{name:dict(groups=int(condition.sum()),
                position_unique_count=len(np.unique(a['aligned_prediction'][condition,:,:2].reshape(-1,2),axis=0)),
                position_std=a['aligned_prediction'][condition,:,:2].reshape(-1,2).std(0).tolist())
                for name,condition in [('pair_only',size==2),('recursive_ge3',size>=3)]},
            'median_within_trajectory_truth_span_degrees':np.median([np.ptp(a['aligned_truth'][i][valid[i],:2],axis=0)
                  for i in np.flatnonzero(size>1) if valid[i].sum()>1],axis=0).tolist()}
        endpoint['scene_paired_vs_simple']=paired_scene_summary(records)
        if previous_records is not None:
            for prev,now in zip(previous_records,records):
                assert prev['scene_id']==now['scene_id']
                for name in METHODS:
                    for field in ('aligned_truth','aligned_truth_valid','aligned_prediction_valid'):
                        assert np.array_equal(prev['methods'][name][field],now['methods'][name][field],equal_nan=True)
                    if name!='oracle':
                        assert np.array_equal(prev['methods'][name]['aligned_prediction'],now['methods'][name]['aligned_prediction'],equal_nan=True)
            endpoint['controls_masks_truth_unchanged_from_previous']=True
        previous_records=records
        report['endpoints'][str(update)]=endpoint
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'endpoint300':report['endpoints']['300']['own_support']['oracle'],
                      'multi_fragment_output':report['endpoints']['300']['multi_fragment_output'],
                      'training_coverage':report['training_coverage']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
