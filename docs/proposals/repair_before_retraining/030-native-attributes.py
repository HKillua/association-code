from pathlib import Path
import json, subprocess, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from association.data.archive import MTADArchive
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from association.evaluation.physical import evaluate_aligned_fusion
SCENES=['1671','2056','290','113']; start=time.monotonic()
manifest=json.loads(subprocess.check_output(['git','show','612afea:HandOver/results/EVIDENCE-20261018-012/prepared007/manifest.json'],cwd=ROOT))
R=6371000.; rows=[]; scene_rows=[]
def arc(a):return (a+180)%360-180
def stat(a):
 a=np.asarray(a,dtype=float);return None if not a.size else {'n':int(a.size),'mean':float(a.mean()),'std':float(a.std()),'min':float(a.min()),'p05':float(np.quantile(a,.05)),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
def vec(a,b):return R*np.pi/180*np.stack((a[...,0]-b[...,0],arc(a[...,1]-b[...,1])*np.cos(np.deg2rad(b[...,0]))),axis=-1)
def interp(t,v,q):
 out=np.empty((len(q),4))
 for j,x in enumerate(q):
  right=int(np.searchsorted(t,x));left=max(right-1,0)
  if right==0 or right==len(t):out[j]=v[0 if right==0 else -1];continue
  alpha=(x-t[left])/(t[right]-t[left]);out[j]=v[left]+alpha*(v[right]-v[left])
  for col in [1,3]:
   raw=v[right,col]-v[left,col]; d=arc(raw)
   if abs(d)==180:d=180 if raw>0 else -180
   out[j,col]=v[left,col]+alpha*d
  out[j,1]=arc(out[j,1]);out[j,3]%=360
 return out
with MTADArchive(sys.argv[1] if len(sys.argv)>1 else Path.home()/'Downloads/数据集level2.zip') as ar:
 for sid in SCENES:
  n=_record(ar.load_scene(sid),manifest['components'][sid],'V_select',128)
  s=prepare_timed_scene(n,length=64,max_observations=128,time_unit_seconds=20.)
  z=np.load(ROOT/f'HandOver/results/COMPLETION-RESIDUAL-20261019-030/predictions-{sid}.npz')
  assert list(z['node_ids'])==s['node_ids'];np.testing.assert_array_equal(z['query_seconds'],s['grid'])
  for i,node in enumerate(s['node_ids']):
   m=s['native_observed'][i];t=s['native_times'][i,m];v=s['native_values'][i,m];p=interp(t,v,s['grid']); base=z['position_base'][i].astype(float);tar=z['target'][i]
   row={'scene':sid,'node':node,'source':s['sources'][i],'identity':str(s['identities'][i]),'times':t,'native':v,'grid':s['grid'],'control':p,'model':z['prediction'][i].astype(float),'target':tar,'valid':z['target_valid'][i],'regions':{k.removeprefix('region_'):z[k][i] for k in z.files if k.startswith('region_')},'needed_correction':vec(tar,p),'nearest':z['nearest_seconds'][i]}
   rows.append(row)
  scene_rows.append({'scene':sid,'fragments':len(n['tracks']),'query_span_seconds':float(np.ptp(s['grid'])),'query_step_seconds':float(np.diff(s['grid'])[0]),'fragment_duration_seconds':stat([x['times'][-1]-x['times'][0] for x in n['tracks']]),'native_points':int(s['native_observed'].sum())})
 members=list(ar.members_read)
regions=['all','inside_support','before_support','after_support','exact_observation']
def metrics(rs,region,method):
 out={}
 for field,idx in [('speed',2),('course',3)]:
  vals=[];targets=[];countpred=0;improved=0;fragn=0
  for r in rs:
   reg=r['regions'][region];countpred+=reg.sum();m=reg&r['valid'][:,idx]
   if not m.any():continue
   vals.append(r[method][m,idx]);targets.append(r['target'][m,idx]);fragn+=1
  if not vals:out[field]={'prediction_count':int(countpred),'valid_count':0};continue
  p=np.concatenate(vals);t=np.concatenate(targets)
  err=np.abs(p-t) if idx==2 else np.abs(arc(p-t))
  rawloss=(np.log(p+1e-6)-np.log(t+1e-6))**2 if idx==2 else 1-np.cos(np.deg2rad(p-t))
  out[field]={'prediction_count':int(countpred),'valid_count':len(p),'fragment_count':fragn,'mae':float(err.mean()),'rmse':float(np.sqrt(np.mean(err**2))),'median_abs_error':float(np.median(err)),'raw_loss':float(rawloss.mean()),'prediction':stat(p),'target':stat(t)}
 return out
results={source:{region:{method:metrics(rows if source=='all' else [r for r in rows if r['source']==source],region,method) for method in ['control','model']} for region in regions} for source in ['all','9001','9002']}
# Compare fixed observation-only control with teacher for each valid attribute query.
for source in results:
 rs=rows if source=='all' else [r for r in rows if r['source']==source]
 for region in regions:
  for field,idx in [('speed',2),('course',3)]:
   point=fn=improved=0;changes=[]
   for r in rs:
    m=r['regions'][region]&r['valid'][:,idx]
    if not m.any():continue
    t=r['target'][m,idx];a=r['control'][m,idx];b=r['model'][m,idx]
    ae=np.abs(a-t) if idx==2 else np.abs(arc(a-t));be=np.abs(b-t) if idx==2 else np.abs(arc(b-t))
    point+=m.sum();fn+=1;improved+=np.sum(ae<be);changes.append(float(np.mean(ae-be)))
   results[source][region][f'{field}_control_vs_model']={'valid_point_n':int(point),'control_better_points':int(improved),'valid_fragment_n':fn,'control_better_fragments':int(sum(x<0 for x in changes)),'mean_perfragment_delta_mae':float(np.mean(changes)) if changes else None}
# Truth based descriptive correction geometry. These means are NEVER used for control prediction.
corr_fragments=[]
for r in rows:
 m=r['regions']['inside_support']&np.all(r['valid'][:,:2],axis=-1)
 if not m.any():continue
 d=r['needed_correction'][m];mu=d.mean(axis=0);de=d-mu
 corr_fragments.append({'scene':r['scene'],'source':r['source'],'identity':r['identity'],'node':r['node'],'valid_inside_n':int(m.sum()),'mean_required_NE_m':mu.tolist(),'mean_magnitude_m':float(np.linalg.norm(mu)),'within_rms_m':float(np.sqrt(np.mean(np.sum(de**2,axis=-1)))),'time_span_seconds':float(np.ptp(r['grid'][m])),'query_first_seconds':float(r['grid'][m][0]),'query_last_seconds':float(r['grid'][m][-1])})
def group_geometry(keyfunc):
 groups={}
 for r in rows:
  if r['source']!='9002':continue
  m=r['regions']['inside_support']&np.all(r['valid'][:,:2],axis=-1)
  if not m.any():continue
  key=keyfunc(r);groups.setdefault(key,[]).append((r,r['needed_correction'][m]))
 out=[]
 for key,arr in groups.items():
  d=np.concatenate([v for r,v in arr]);mu=d.mean(0);rms=np.sqrt(np.mean(np.sum((d-mu)**2,axis=-1)));energy=np.mean(np.sum(d*d,axis=-1))
  out.append({'key':key,'n_points':len(d),'n_fragments':len(arr),'n_identities':len(set(r['identity'] for r,d in arr)),'mean_required_NE_m':mu.tolist(),'mean_magnitude_m':float(np.linalg.norm(mu)),'variation_rms_m':float(rms),'mean_fraction_energy':float(np.sum(mu*mu)/energy)})
 return out
# Compare same-identity fragments on simultaneously supported query points, no identity info enters any prediction.
shared=[]
for i,a in enumerate(rows):
 if a['source']!='9002':continue
 for b in rows[i+1:]:
  if b['source']!='9002' or a['scene']!=b['scene'] or a['identity']!=b['identity']:continue
  m=a['regions']['inside_support']&b['regions']['inside_support']&np.all(a['valid'][:,:2],-1)&np.all(b['valid'][:,:2],-1)
  if m.any():shared.append({'scene':a['scene'],'identity':a['identity'],'nodes':[a['node'],b['node']],'n':int(m.sum()),'correction_vector_difference_rms_m':float(np.sqrt(np.mean(np.sum((a['needed_correction'][m]-b['needed_correction'][m])**2,axis=-1))))})
# Empirical covariation native input speed/course vs target only descriptive; no fitting.
support={}
for source in ['all','9001','9002']:
 rs=rows if source=='all' else [r for r in rows if r['source']==source]
 support[source]={'fragment_count':len(rs),'duration_seconds':stat([r['times'][-1]-r['times'][0] for r in rs]),'observation_gap_seconds':stat(np.concatenate([np.diff(r['times']) for r in rs])),'nearest_seconds_all_queries':stat(np.concatenate([r['nearest'] for r in rs])),'zero_inside_query_fragments':sum(not r['regions']['inside_support'].any() for r in rs),'one_inside_query_fragments':sum(r['regions']['inside_support'].sum()==1 for r in rs)}
out={'status':'COMPLETED','scope':'Same original four V scenes, every86 fragment and5504 query preserved. Truth only for evaluation/descriptive geometry; ordinary attribute controls use no fitting or labels; no model or training. The separate oracle section uses other V identities and true correspondences for offset estimation; it is not deployable performance. No new reference read: existing030 NPZ truth reused. Control fixed before scoring: native speed linear interpolation, course signed-shortest-arc interpolation; endpoints held outside support. Native original units retained.','archive':str(sys.argv[1] if len(sys.argv)>1 else Path.home()/'Downloads/数据集level2.zip'),'manifest':'612afea:HandOver/results/EVIDENCE-20261018-012/prepared007/manifest.json','observation_members_read':members,'scenes':scene_rows,'population':{'scenes':4,'fragments':86,'queries':5504,'position_target_valid':4008,'attribute_target_valid_each':4008,'no_position_reference':1496},'support':support,'attribute_metrics':results,'correction_geometry':{'all9002':group_geometry(lambda r:'all9002'),'by_scene9002':group_geometry(lambda r:r['scene']),'by_scene_identity9002':group_geometry(lambda r:r['scene']+':'+r['identity']),'fragments':corr_fragments,'same_identity_overlapping_fragments':shared},'elapsed_seconds':time.monotonic()-start}
Path('/tmp/native-attribute-analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item()))
print(json.dumps({'elapsed':out['elapsed_seconds'],'scene_info':scene_rows,'source_all_summary':{s:results[s]['all'] for s in results},'sensor_geometry':out['correction_geometry']['by_scene9002']},indent=2))
# Disjoint fixed query-index thirds give descriptive temporal sensor-bias summaries.
by_time=[]
for sid in SCENES:
 sr=[r for r in rows if r['scene']==sid and r['source']=='9002']
 for label,lo,hi in [('first_third',0,21),('middle_third',21,42),('last_third',42,64)]:
  ds=[]; identities=set();nodes=set()
  for r in sr:
   m=r['regions']['inside_support']&np.all(r['valid'][:,:2],-1)&(np.arange(64)>=lo)&(np.arange(64)<hi)
   if m.any():ds.append(r['needed_correction'][m]);identities.add(r['identity']);nodes.add(r['node'])
  d=np.concatenate(ds) if ds else np.empty((0,2))
  by_time.append({'scene':sid,'fixed_query_index_third':label,'n_points':len(d),'n_identities':len(identities),'n_fragments':len(nodes),'mean_required_NE_m':d.mean(0).tolist() if len(d) else None,'within_rms_m':float(np.sqrt(np.mean(np.sum((d-d.mean(0))**2,axis=-1)))) if len(d) else None})
out['correction_geometry']['fixed_query_thirds9002']=by_time
# Independent implementation consistency with production prepared interpolation in support.
max_diff=0.
with MTADArchive(sys.argv[1] if len(sys.argv)>1 else Path.home()/'Downloads/数据集level2.zip') as ar:
 for sid in SCENES:
  scene=prepare_timed_scene(_record(ar.load_scene(sid),manifest['components'][sid],'V_select',128),length=64,max_observations=128,time_unit_seconds=20.)
  rr=[r for r in rows if r['scene']==sid]
  for i,r in enumerate(rr):
   m=r['regions']['inside_support']; diff=r['control'][m]-scene['values'][i,m];diff[:,1]=arc(diff[:,1]);diff[:,3]=arc(diff[:,3]);max_diff=max(max_diff,float(np.max(np.abs(diff))) if diff.size else 0.)
out['verification']={'native_control_inside_vs_independent_prepare_timed_scene_max_abs_field_difference':max_diff,'node_ids_and_query_times_exactly_match030':True,'every_source_region_metric_retains_all_valid_points':True}
Path('/tmp/native-attribute-analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item()))
print('TIME',json.dumps(by_time,indent=2))
# Requested oracle diagnostic: reference-free offset estimated from OTHER identities'
# true cross-source correspondences. Ground-truth identity labels are used, so this is
# never a deployable method or a benchmark result. No held-out identity coordinates enter fit; the original whole-scene query grid is retained.
def haversine(a,b):
 la=np.deg2rad(a[...,0]);lb=np.deg2rad(b[...,0]);dl=np.deg2rad(arc(a[...,1]-b[...,1]));h=np.clip(np.sin((la-lb)/2)**2+np.cos(la)*np.cos(lb)*np.sin(dl/2)**2,0,1);return 2*R*np.arctan2(np.sqrt(h),np.sqrt(1-h))
oracle_rows=[];oracle_identity=[];estimated_all=[];unestimated_all=[]
for sid in SCENES:
 sr=[r for r in rows if r['scene']==sid];left=[r for r in sr if r['source']=='9001'];right=[r for r in sr if r['source']=='9002']
 pairs=[]
 for a in left:
  for b in right:
   if a['identity']!=b['identity']:continue
   m=a['regions']['inside_support']&b['regions']['inside_support']
   if not m.any():continue
   d=a['control'][m,:2]-b['control'][m,:2];d[:,1]=arc(d[:,1])
   for j,q in enumerate(np.flatnonzero(m)):pairs.append({'identity':a['identity'],'query_index':int(q),'difference_degrees':d[j]})
 # Collapse duplicate evidence slots explicitly, without looking at reference.
 idq={}
 for p in pairs:idq.setdefault((p['identity'],p['query_index']),[]).append(p['difference_degrees'])
 unique_pairs=[{'identity':i,'query_index':q,'difference_degrees':np.mean(d,axis=0),'duplicates':len(d)} for (i,q),d in idq.items()]
 for ident in sorted(set(r['identity'] for r in right)):
  evidence=[p for p in unique_pairs if p['identity']!=ident]
  own=[r for r in right if r['identity']==ident]
  enough=len(evidence)>0
  offset=np.mean([p['difference_degrees'] for p in evidence],axis=0) if enough else np.zeros(2)
  summary={'scene':sid,'held_out_identity':ident,'evidence_identity_count':len(set(p['identity'] for p in evidence)),'evidence_overlap_query_count':len(evidence),'offset_degrees_lat_lon':offset.tolist(),'estimated':enough,'heldout_query_count':0}
  ep=[];eb=[]
  for r in own:
   m=r['regions']['inside_support']&np.all(r['valid'][:,:2],-1)
   if not m.any():continue
   p=r['control'][m,:2].copy();p+=offset;p[:,1]=arc(p[:,1]);truth=r['target'][m,:2];b=r['control'][m,:2]
   e=haversine(p,truth);base_error=haversine(b,truth)
   ep.extend(e);eb.extend(base_error)
   oracle_rows.append({'scene':sid,'node':r['node'],'identity':ident,'valid_n':int(m.sum()),'estimated':enough,'base_mae_m':float(np.mean(base_error)),'oracle_mae_m':float(np.mean(e)),'oracle_better_points':int(np.sum(e<base_error))})
   (estimated_all if enough else unestimated_all).append((e,base_error))
  summary.update(heldout_query_count=len(ep),base_mae_m=float(np.mean(eb)) if eb else None,oracle_mae_m=float(np.mean(ep)) if ep else None)
  oracle_identity.append(summary)
 arr=[(a,b) for a,b in estimated_all+unestimated_all];aa=np.concatenate([a for a,b in arr]);bb=np.concatenate([b for a,b in arr])
out['oracle_cross_source_observed_alignment']={'warning':'ORACLE DIAGNOSTIC ONLY: uses ground-truth identity correspondences on V. Offset is estimated from other identities same-scene original observation interpolation only; NO reference or held-out identity coordinates enter estimation; the original whole-scene query grid is retained. Not a deployable inference method or a claim of paper performance. References used solely to evaluate held-out source9002 inside support. All690 valid points retained; missing evidence would leave base unchanged.','mean_policy':'For each scene/heldout identity, equally weight unique(other identity,query index) overlap slots; average signed-shortest-lon/latitude differences9001-9002. No scale/rotation/search/hyperparameters.','valid_points':len(aa),'identities':len(oracle_identity),'fragments':len(oracle_rows),'unestimated_valid_points':sum(len(a) for a,b in unestimated_all),'base_mae_m':float(bb.mean()),'oracle_mae_m':float(aa.mean()),'oracle_better_points':int(np.sum(aa<bb)),'per_identity':oracle_identity,'per_fragment':oracle_rows}
Path('/tmp/native-attribute-analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item()))
print('ORACLE',json.dumps(out['oracle_cross_source_observed_alignment'],indent=2))
