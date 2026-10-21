"""Independently audit saved 048/046 arrays and 045 scores, without inference.

No model/checkpoint loading, raw ZIP reading, threshold fitting or T_test access.
The explicit output JSON contains the verification summary and recomputed metrics.
"""
import argparse
import io, json, gzip, tarfile
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
if args.output.exists():
 raise FileExistsError(args.output)
ROOT=Path(__file__).resolve().parents[3]
P=ROOT/'HandOver/results/FROZEN-EVIDENCE-COMBO-20261021-048'
ARMS=['ols046_mean_supported','ols046_learned','assoc045_mean_supported','assoc045_learned']
ATTRS=['position','speed','heading']
COMP={'association_mean':(2,0),'association_learned':(3,1),'fusion_ols':(1,0),'fusion_045':(3,2)}
def loadjson(p):return json.loads(gzip.decompress(p.read_bytes()))
def loadtar(p):
 out={}
 with tarfile.open(p) as t:
  for f in t:
   if f.isfile() and f.name.endswith('.npz'):
    with np.load(io.BytesIO(t.extractfile(f).read()),allow_pickle=False) as z:out[Path(f.name).stem.removeprefix('states-')]={k:z[k].copy() for k in z.files}
 return out
def sphere(a,b):
 def u(x):
  x=np.deg2rad(x.astype(float));return np.stack([np.cos(x[...,0])*np.cos(x[...,1]),np.cos(x[...,0])*np.sin(x[...,1]),np.sin(x[...,0])],-1)
 x,y=u(a),u(b);return 6371000*np.arctan2(np.linalg.norm(np.cross(x,y),axis=-1),(x*y).sum(-1))
def errs(a,b):return [sphere(a[...,:2],b[...,:2]),np.abs(a[...,2]-b[...,2]),np.abs((a[...,3]-b[...,3]+180)%360-180)]
def stats(e):
 return {'n':int(e.size),'mae':float(e.mean()) if e.size else None,'rmse':float(np.sqrt((e*e).mean())) if e.size else None,'p95':float(np.quantile(e,.95)) if e.size else None,'p99':float(np.quantile(e,.99)) if e.size else None,'max':float(e.max()) if e.size else None}
x=loadtar(P/'predictions.tar.gz');old=loadtar(ROOT/'HandOver/results/EVIDENCE-FUSION-20261021-046/predictions-final.tar.gz');report=loadjson(P/'evaluation.json.gz');replay=loadjson(P/'replay.json.gz');assoc=loadjson(P/'association-replay.json.gz')
assert len(x)==len(old)==len(report['per_scene'])==64 and x.keys()==old.keys()
common=['node_ids','identities','sources','query_seconds','target','target_valid','native_states','nearest_seconds']
common += [k for k in next(iter(old.values())) if k.startswith('region_')]
replay_max=np.zeros(3);weights_max=0;groups=0
for sid,a in x.items():
 b=old[sid];n=len(a['node_ids']);assert a['target'].shape==(n,64,4) and a['target_valid'].dtype==bool
 for k in common:np.testing.assert_array_equal(a[k],b[k])
 for k in ARMS:assert a[k].shape==a['target'].shape and np.isfinite(a[k]).all()
 np.testing.assert_array_equal(a[ARMS[0]],b['predicted_anchor_direct_mean_supported'])
 np.testing.assert_array_equal(a['ols046_membership'],b['predicted_anchor_direct_membership'])
 replay_max=np.maximum(replay_max,[z.max() for z in errs(a[ARMS[1]],b['predicted_anchor_direct_learned'])])
 weights_max=max(weights_max,float(np.abs(a['ols046_anchor_weights']-b['predicted_anchor_direct_anchor_weights']).max()))
 flags={}
 for group in ['ols046','assoc045']:
  mem=a[group+'_membership'];mixed=np.zeros(n,bool)
  for gid in np.unique(mem):
   ix=np.flatnonzero(mem==gid);mixed[ix]=len(set(a['identities'][ix]))>1
  flags[group]=mixed
  ha,hn=a[group+'_helper_anchor'],a[group+'_helper_node'];assert (ha!=hn).all();np.testing.assert_array_equal(mem[ha],mem[hn])
 for i,oi in [('pure',~flags['ols046']),('mixed',flags['ols046'])]:
  for j,nj in [('pure',~flags['assoc045']),('mixed',flags['assoc045'])]:np.testing.assert_array_equal(a['region_old_'+i+'_new_'+j],np.broadcast_to((oi&nj)[:,None],(n,64)))
 assert np.all(a['region_inside_support']^a['region_outside_support'])
 np.testing.assert_array_equal(a['region_far_gt_1800s'],a['nearest_seconds']>1800)
 np.testing.assert_array_equal(a['region_exact_observation'],a['nearest_seconds']==0)
regions=[k.removeprefix('region_') for k in next(iter(x.values())) if k.startswith('region_')]
pool={k:np.concatenate([a[k] for a in x.values()]) for k in ['sources','target','target_valid']+ARMS+['region_'+r for r in regions]}
errors={arm:errs(pool[arm],pool['target']) for arm in ARMS};masks=[pool['target_valid'][...,:2].all(-1),pool['target_valid'][...,2],pool['target_valid'][...,3]]
result={};maxdiff=0
for src in ['all','9001','9002']:
 result[src]={}
 for r in regions:
  rm=pool['region_'+r].copy()
  if src!='all':rm &= pool['sources'][:,None]==src
  result[src][r]={}
  for arm in ARMS:
   result[src][r][arm]={}
   for i,attr in enumerate(ATTRS):
    st=stats(errors[arm][i][rm&masks[i]]);result[src][r][arm][attr]=st
    claimed=(report['overall'] if src=='all' else report['by_source'][src])[r]['methods'][arm][attr]
    assert st['n']==claimed['intersection_count']
    for met in ['mae','rmse','p95','p99','max']:
     if st[met] is None:assert claimed[met] is None
     else:maxdiff=max(maxdiff,abs(st[met]-claimed[met]));np.testing.assert_allclose(st[met],claimed[met],atol=1e-6,rtol=1e-10)
scene_deltas={k:{attr:[] for attr in ATTRS} for k in COMP}
for a in x.values():
 for k,(ci,bi) in COMP.items():
  ce,be=errs(a[ARMS[ci]],a['target']),errs(a[ARMS[bi]],a['target'])
  for i,attr in enumerate(ATTRS):
   mask=a['target_valid'][...,:2].all(-1) if i==0 else a['target_valid'][...,i+1]
   scene_deltas[k][attr].append(float((ce[i][mask]-be[i][mask]).mean()))
scene_summary={k:{attr:{'mean_delta':float(np.mean(vals)),'improved':int((np.asarray(vals)<-1e-8).sum()),'worsened':int((np.asarray(vals)>1e-8).sum()),'unchanged':int((np.abs(vals)<=1e-8).sum())} for attr,vals in d.items()} for k,d in scene_deltas.items()}
with np.load(ROOT/'HandOver/results/EVIDENCE-ASSOC-FULL-20261021-045/selected-scores.npz',allow_pickle=False) as z:
 counts=np.zeros(4,dtype=int);f1=[]
 for sid in z['scene_ids']:
  a=x[str(sid)];np.testing.assert_array_equal(a['node_ids'],z[sid+'_node_ids']);np.testing.assert_array_equal(a['sources'],z[sid+'_sources'])
  pairs,scores,labels=z[sid+'_pairs'],z[sid+'_scores'],z[sid+'_labels'];pred=scores>=assoc['frozen_threshold'];truth=labels.astype(bool)
  c=np.array([(pred&truth).sum(),(pred&~truth).sum(),(~pred&truth).sum(),(~pred&~truth).sum()]);counts+=c;den=2*c[0]+c[1]+c[2];f1.append(2*c[0]/den if den else 1.)
  n=len(a['node_ids']);adj=np.zeros((n,n),bool);adj[pairs[:,0],pairs[:,1]]=pred;adj|=adj.T;mem=np.full(n,-1);gid=0
  for i in range(n):
   if mem[i]>=0:continue
   members=np.flatnonzero(adj[i]&(mem<0));mem[members]=gid;mem[i]=gid;gid+=1
  np.testing.assert_array_equal(mem,a['assoc045_membership'])
  groups+=gid
 assoc_ind={'tp':int(counts[0]),'fp':int(counts[1]),'fn':int(counts[2]),'tn':int(counts[3]),'macro_f1':float(np.mean(f1)),'micro_f1':float(2*counts[0]/(2*counts[0]+counts[1]+counts[2])),'group_count':groups}
 for k in ['tp','fp','fn','tn','macro_f1','micro_f1']:np.testing.assert_allclose(assoc_ind[k],assoc['metrics'][k],atol=1e-15)
summary={'scope':'saved048 fragment outputs independently recomputed; no model, archive or T_test','population':{'scenes':len(x),'fragments':len(pool['sources']),'queries':int(pool['target'].shape[0]*64),'valid_by_attribute':pool['target_valid'].sum((0,1)).tolist()},'common_arrays_to046_exact':len(common)*64,'all_four_arm_predictions_finite':True,'old_mean_control_exact':True,'membership_and_pure_mixed_masks_reconstructed_exact':True,'max_metric_difference':maxdiff,'replay_global_max_physical':dict(zip(ATTRS,map(float,replay_max))),'replay_max_anchor_weight_difference':weights_max,'association_independent':assoc_ind,'scene_paired':scene_summary,'metrics':result}
args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
