"""Three fixed V scenes only. CPU raw CSV roundtrip; no model/test/train access."""
import sys,json,gzip,time,math,collections
from pathlib import Path
import numpy as np
REPO=Path('/Users/a1/Projects/association-code');sys.path.insert(0,str(REPO))
from association.data import MTADArchive
from association.native_inputs import _record
from association.timed_inputs import prepare_timed_scene,pack_timed_scene
from association.reference_states import iter_reference_states
from association.fusion_targets import build_scene_fusion_targets
from association.full_scene_evaluation import simple_fusion_outputs,single_source_outputs,evaluate_full_scene
OUT=Path(__file__).parent;ZIP=Path('/Users/a1/Downloads/数据集level2.zip');t0=time.monotonic()
manifest=json.loads((REPO/'HandOver/results/EVIDENCE-20261018-012/prepared007/manifest.json').read_text())
align=json.loads((REPO/'configs/tmtaff-time-reference-alignment.json').read_text())
base=REPO/'HandOver/results/COOPERATIVE-DEVELOPMENT-20261018-018/files'
run=json.load(gzip.open(base/'run.json.gz','rt'));ev=json.load(gzip.open(base/'plain-evaluation.json.gz','rt'))
contract={s['scene_id']:s for s in run['scene_contract'][1]};server={s['scene_id']:s for s in ev['per_scene']}
selected=[manifest['splits']['V_select'][i] for i in [0,32,63]]
assert selected==['1671','4038','4676'];assert all(s in contract for s in selected)
old=json.load(gzip.open(REPO/'HandOver/results/TMTAFF-TIME-EVAL-20261018-014/files/summary.json.gz','rt'))
normalizer=old['normalizer'];assert np.array_equal(normalizer['std'],run['error_scales'])

def hav(a,b):
    p1,p2=np.deg2rad(a[:,0]),np.deg2rad(b[:,0]);dp=p2-p1;dl=np.deg2rad((b[:,1]-a[:,1]+180)%360-180)
    return 2*6371000*np.arctan2(np.sqrt(np.clip(np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2,0,1)),np.sqrt(np.clip(1-(np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2),0,1)))

def independent(reference,ident,q,offset):
    rows=[r for r in reference['rows'] if r['identity_id']==ident]
    times=np.array([r['time']-offset for r in rows]);vals=np.array([r['state'] for r in rows]);order=np.argsort(times);times,vals=times[order],vals[order]
    assert np.isfinite(times).all() and np.isfinite(vals).all() and np.all(np.diff(times)>0)
    assert np.all(np.abs(vals[:,0])<=90) and np.all(np.abs(vals[:,1])<=180) and np.all(vals[:,2]>=0) and np.all((vals[:,3]>=0)&(vals[:,3]<=360))
    vals[:,3]%=360;out=np.full((len(q),4),np.nan);mask=np.zeros_like(out,dtype=bool);exact=interior=outside=0
    for k,t in enumerate(q):
        j=int(np.searchsorted(times,t))
        if j<len(times) and times[j]==t:out[k]=vals[j];mask[k]=True;exact+=1
        elif j==0 or j==len(times):outside+=1
        elif times[j]-times[j-1]<=600:
            a=(t-times[j-1])/(times[j]-times[j-1]);out[k]=vals[j-1]+a*(vals[j]-vals[j-1]);dd=(vals[j,3]-vals[j-1,3]+180)%360-180
            assert abs(dd)!=180;out[k,3]=(vals[j-1,3]+a*dd)%360;mask[k]=True;interior+=1
    return out,mask,dict(exact=exact,interior=interior,outside=outside)

reports=[];reads=[];self_distances_all=[]
def extrap_summary(arr):
    arr=np.asarray(arr,dtype=float);outside=arr[arr>0]
    return {'fragment_query_count':len(arr),'inside_count':int(np.count_nonzero(arr==0)),'outside_count':len(outside),'outside_fraction':float(np.mean(arr>0)),'over_10min_fraction':float(np.mean(arr>600)),'over_30min_fraction':float(np.mean(arr>1800)),'over_60min_fraction':float(np.mean(arr>3600)),'p50_p95_max_seconds':np.percentile(arr,[50,95,100]).tolist(),'outside_only_p50_p95_max_seconds':np.percentile(outside,[50,95,100]).tolist() if len(outside) else None}

refs=iter_reference_states(ZIP,manifest,split='V_select',scene_ids=selected)
with MTADArchive(ZIP) as archive:
 for sid,ref in zip(selected,refs):
    assert sid==ref['scene_id'];scene=archive.load_scene(sid);labels=archive.read_labels(sid);native=_record(scene,manifest['components'][sid],'V_select',128)
    prep=prepare_timed_scene(native,length=64,max_observations=128,time_unit_seconds=20)
    targetids=list(dict.fromkeys('-'.join(r['mmsi'].split('-')[:2]) for r in labels))
    target=build_scene_fusion_targets(prep,ref,target_identity_ids=targetids,**align[sid]);contract_row=contract[sid]
    for key in ['node_ids','sources','identities','grid','intervals']:
        assert np.array_equal(np.asarray(prep[key]),np.asarray(contract_row[key])),(sid,key)
    assert targetids==contract_row['target_ids'];assert prep['native_observed'].sum(axis=1).tolist()==contract_row['native_observed_counts']
    x,mask=pack_timed_scene(prep,normalizer);m=prep['native_observed'];M=128
    restored=x[:,:M,:4][m].astype(np.float64)*normalizer['std']+normalizer['mean'];native_rt=np.max(np.abs(restored-prep['native_values'][m]),axis=0)
    pos_rt=hav(restored,prep['native_values'][m]);time_rt=np.max(np.abs(x[:,:M,4][m].astype(float)*20+prep['time_origin_seconds']-prep['native_times'][m]));query_rt=np.max(np.abs(x[0,M:,4].astype(float)*20+prep['time_origin_seconds']-prep['grid']))
    assert not mask[:,M:].any();assert np.array_equal(mask[:,:M],m)
    independent_max=0.;manual_counts=collections.Counter();target_physical_diffs=[]
    for ident in targetids:
        independent_states,im,cc=independent(ref,ident,prep['grid'],align[sid]['time_offset_seconds']);t=target[ident]
        assert np.array_equal(im,t['valid']);independent_max=max(independent_max,float(np.max(np.abs(independent_states[im]-t['states'][im]))));manual_counts.update(cc)
        ok=im[:,:2].all(1);target_physical_diffs.extend(hav(independent_states[ok],t['states'][ok]).tolist())
    by_state=collections.defaultdict(set)
    for rr in ref['rows']:
        state=list(rr['state']);state[3]%=360;by_state[(rr['identity_id'],tuple(state))].add(rr['time'])
    offsets=collections.Counter();point_matches=collections.Counter();native_range=collections.defaultdict(list)
    for tr in native['tracks']:
        src=tr['source'];point_matches[src+'_points']+=len(tr['canonical_times'])
        for t,state in zip(tr['canonical_times'],tr['canonical_values']):
            native_range[src].append(state)
            matches=by_state.get((tr['identity_id'],tuple(state)),set())
            if matches:
                point_matches[src+'_exact_four_state_matches']+=1
                diffs={rt-t for rt in matches}
                if align[sid]['time_offset_seconds'] in diffs:point_matches[src+'_matched_at_config_offset']+=1
                for diff in diffs:offsets[str(diff)]+=1
    truth=np.stack([target[k]['states'] for k in targetids]);valid=np.stack([target[k]['valid'] for k in targetids]);nv=np.repeat(prep['observed'][...,None],4,axis=-1)
    controls={'simple_fusion_oracle':simple_fusion_outputs(prep['values'],nv,prep['identities'],prep['sources'])}
    controls.update({f'single_source:{src}':single_source_outputs(prep['values'],nv,prep['identities'],prep['sources'],source=src) for src in ['9001','9002']})
    ctrl_reports={};control_max_diff=0.
    for name,outputs in controls.items():
        metric=evaluate_full_scene(outputs,node_identities=prep['identities'],target_identities=targetids,target_states=truth,target_valid=valid,node_observed=prep['observed'],split='V_select',grouping_kind='oracle',correspondence_policy='anchor_identity_first')
        for reg in ['all','observed','missing']:
            for attr in ['position','speed','heading']:
                ours=metric['full_target_metrics'][reg][attr];remote=server[sid]['methods'][name]['full_target_metrics'][reg][attr]
                for k,v in ours.items():
                    if k in remote and isinstance(v,(int,float)):
                        assert math.isclose(v,remote[k],rel_tol=1e-12,abs_tol=1e-8),(sid,name,reg,attr,k,v,remote[k])
                        control_max_diff=max(control_max_diff,abs(v-remote[k]))
        # independent spherical distance on the source/simple output against exact target identity, no evaluation matcher
        dd=[]
        for oo in outputs:
            ident=str(prep['identities'][oo['node_indices'][0]]);tt=target[ident];ok=oo['prediction_valid'][:,:2].all(1)&tt['valid'][:,:2].all(1)
            dd.extend(hav(oo['states'][ok],tt['states'][ok]).tolist())
        got=metric['full_target_metrics']['all']['position'];mae=float(np.mean(dd)) if dd else None
        assert mae is None or math.isclose(mae,got['mae'],rel_tol=1e-10,abs_tol=1e-7)
        ctrl_reports[name]={'position':got,'independent_haversine_mae':mae}
    observed_identity=0;exact_native_identity=0;nearestd=[];native_gap=[];self_distances=[]
    for tr in native['tracks']:
        ok=target[tr['identity_id']]['valid'].all(1);qq=prep['grid'][ok];lo,hi=tr['canonical_times'][0],tr['canonical_times'][-1]
        dist=np.maximum(np.maximum(lo-qq,qq-hi),0.);self_distances.extend(dist.tolist())
    self_distances_all.extend(self_distances)
    for ident in targetids:
        ids=np.flatnonzero(prep['identities']==ident);support=prep['observed'][ids].any(0);rawtimes=np.concatenate([t['canonical_times'] for t in native['tracks'] if t['identity_id']==ident]);ok=target[ident]['valid'][:,:2].all(1)
        for qi,t in enumerate(prep['grid']):
            if not(support[qi] and ok[qi]):continue
            observed_identity+=1;near=float(np.min(np.abs(rawtimes-t)));nearestd.append(near);exact_native_identity+=int(near==0)
        native_gap.extend([float(np.diff(t['canonical_times']).max()) for t in native['tracks'] if t['identity_id']==ident])
    rec={'scene_id':sid,'selection':'fixed V indices 0,32,63','offset_seconds':align[sid]['time_offset_seconds'],'native_nodes':len(scene.tracks),'canonical_points':sum(len(t.times) for t in scene.tracks),'selected_points':int(m.sum()),'target_identities':targetids,'reference_member':ref['member'],'reference_rows':len(ref['rows']),'reference_issues':ref['issues'],'matches_and_points':dict(point_matches),'exact_state_candidate_offsets':dict(offsets),'input_contract_matches_018':True,'reference_manual':{'max_feature_difference':independent_max,'max_haversine_difference_m':max(target_physical_diffs),'queries':dict(manual_counts)},'packing_roundtrip':{'normalizer_source':'014/013 saved same-protocol normalizer, std exactly equals018 error_scales; not018 checkpoint read','max_abs_state_error':native_rt.tolist(),'max_position_error_metres':float(pos_rt.max()),'max_native_time_error_seconds':float(time_rt),'max_query_time_error_seconds':float(query_rt),'query_observation_mask_all_false':True},'controls':ctrl_reports,'control_metric_max_difference_to_018':control_max_diff,'observed_semantics':{'reference_valid_observed_identity_queries':observed_identity,'at_any_exact_canonical_native_time':exact_native_identity,'nearest_any_native_time_seconds_p50_p95_max':np.percentile(nearestd,[50,95,100]).tolist(),'max_canonical_interpoint_gap':max(native_gap)},'native_value_ranges_by_source':{src:{'min':np.min(v,axis=0).tolist(),'max':np.max(v,axis=0).tolist()} for src,v in native_range.items()}}
    rec['self_extrapolation']=extrap_summary(self_distances)
    reports.append(rec)
 reads=archive.members_read
refs.close()
result={'scope':'3 fixed V scenes only; raw native/reference CPU audit; no training/model/GPU/test','archive':str(ZIP),'selected_scenes':selected,'test_opened':False,'observation_members_read':reads,'reference_members_read':[x['reference_member'] for x in reports],'elapsed_seconds':time.monotonic()-t0,'self_extrapolation':extrap_summary(self_distances_all),'reports':reports}
(OUT/'assembly_review_v2.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'seconds':result['elapsed_seconds'],'scenes':[{k:r[k] for k in ['scene_id','offset_seconds','native_nodes','canonical_points','selected_points','matches_and_points','exact_state_candidate_offsets','reference_manual','packing_roundtrip','observed_semantics','control_metric_max_difference_to_018','self_extrapolation']} for r in reports]},ensure_ascii=False,indent=2))
