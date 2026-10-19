"""Fixed, input-only controls for the four already exposed 030 scenes.
No reference is used to predict; no coefficients are tuned against V.
Compare both endpoint-adjacent and whole-fragment secants, retain all outcomes.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from association.data.archive import MTADArchive
from association.data.native import _record
from association.data.timed import prepare_timed_scene
SRC=ROOT/'HandOver/results/COMPLETION-RESIDUAL-20261019-030'
manifest=json.loads(subprocess.check_output(['git','show','612afea:HandOver/results/EVIDENCE-20261018-012/prepared007/manifest.json'],cwd=ROOT))
R=6371000.
def angle(a,b):
 a,b=np.deg2rad(a),np.deg2rad(b);dl=(b[...,1]-a[...,1]+np.pi)%(2*np.pi)-np.pi
 h=np.clip(np.sin((b[...,0]-a[...,0])/2)**2+np.cos(a[...,0])*np.cos(b[...,0])*np.sin(dl/2)**2,0,1)
 return 2*np.arctan2(np.sqrt(h),np.sqrt(1-h))
def bearing(a,b):
 a,b=np.deg2rad(a),np.deg2rad(b);dl=b[1]-a[1]
 return np.arctan2(np.sin(dl)*np.cos(b[0]),np.cos(a[0])*np.sin(b[0])-np.sin(a[0])*np.cos(b[0])*np.cos(dl))
def destination(anchor,b,d):
 la,lo=np.deg2rad(anchor);lat=np.arcsin(np.clip(np.sin(la)*np.cos(d)+np.cos(la)*np.sin(d)*np.cos(b),-1,1))
 lon=lo+np.arctan2(np.sin(b)*np.sin(d)*np.cos(la),np.cos(d)-np.sin(la)*np.sin(lat))
 return np.stack([np.rad2deg(lat),(np.rad2deg(lon)+180)%360-180],-1)
rows=[];speed_rows=[]
with MTADArchive(sys.argv[1] if len(sys.argv)>1 else Path.home()/'Downloads/数据集level2.zip') as archive:
 for sid in ['1671','2056','290','113']:
  z=np.load(SRC/f'predictions-{sid}.npz')
  sc=prepare_timed_scene(_record(archive.load_scene(sid),manifest['components'][sid],'V_select',128),length=64,max_observations=128,time_unit_seconds=20.)
  assert np.array_equal(z['node_ids'],sc['node_ids']) and np.array_equal(z['query_seconds'],sc['grid'])
  preds={k:z['position_base'][...,:2].astype('float64').copy() for k in ['hold','adjacent_secant','whole_fragment_secant']};preds['teacher']=z['prediction'][...,:2].astype('float64')
  for i,mask in enumerate(sc['native_observed']):
   t=sc['native_times'][i,mask];p=sc['native_values'][i,mask,:2];q=sc['grid']
   for name in ['adjacent_secant','whole_fragment_secant']:
    for side in ['before','after']:
     if side=='before': a,b=0,(1 if name=='adjacent_secant' else len(t)-1); anchor=0; m=q<t[0]
     else:a,b=(len(t)-2 if name=='adjacent_secant' else 0),len(t)-1;anchor=-1;m=q>t[-1]
     angular_speed=angle(p[a],p[b])/(t[b]-t[a]);heading=bearing(p[a],p[b]) if side=='before' else bearing(p[b],p[a])+np.pi
     if m.any():preds[name][i,m]=destination(preds['hold'][i,np.flatnonzero(m)[0]],heading,angular_speed*(q[m]-t[anchor]))
     speed_rows.append(dict(scene=sid,node=sc['node_ids'][i],source=sc['sources'][i],method=name,side=side,derived_speed_mps=float(angular_speed*R)))
  errors={name:R*angle(pred,z['target'][...,:2]) for name,pred in preds.items()}
  for i,node in enumerate(sc['node_ids']):
   valid=z['target_valid'][i,:,:2].all(-1)
   regions={r:z['region_'+r][i]&valid for r in ['all','inside_support','before_support','after_support']}
   outside=(z['region_before_support'][i]|z['region_after_support'][i])&valid
   for lo,hi in [(0,300),(300,600),(600,1800),(1800,float('inf'))]:
    regions[f'outside_{lo}_{hi}']=outside&(z['nearest_seconds'][i]>lo)&(z['nearest_seconds'][i]<=hi)
   for region,m in regions.items():
    for name,e in errors.items():
     rows.append(dict(scene=sid,node=node,source=sc['sources'][i],region=region,method=name,n=int(m.sum()),error_sum=float(e[i,m].sum()),sq_error_sum=float((e[i,m]**2).sum()),improved_points=int((e[i,m]<errors['hold'][i,m]).sum())))
out={'scope':'four frozen V scenes, same 86 fragments/5504 query/4008 valid position points; exploratory CPU control, not trained or held-out result','prediction_rule':'Keep frozen030 base inside. Outside, great-circle constant angular speed from adjacent endpoint pair OR first-last pair. Anchor at frozen030 endpoint base; no fitted hyperparameters, speed attribute or target used in prediction. All controls reported.','optimizer_updates':0,'T_test_opened':False,'per_scene':{},'aggregate':{},'by_source':{},'derived_speed_mps':{}}
def aggregate(part):
 result={}
 for region in dict.fromkeys(r['region'] for r in part):
  result[region]={}
  for method in dict.fromkeys(r['method'] for r in part):
   rr=[r for r in part if r['region']==region and r['method']==method];n=sum(r['n'] for r in rr)
   result[region][method]={'n':n,'mae_m':sum(r['error_sum'] for r in rr)/n if n else None,'rmse_m':np.sqrt(sum(r['sq_error_sum'] for r in rr)/n) if n else None,'improved_points':sum(r['improved_points'] for r in rr)}
 return result
out['aggregate']=aggregate(rows)
for sid in ['1671','2056','290','113']:out['per_scene'][sid]=aggregate([r for r in rows if r['scene']==sid])
for s in ['9001','9002']:
 out['by_source'][s]=aggregate([r for r in rows if r['source']==s])
 out['derived_speed_mps'][s]={}
 for method in ['adjacent_secant','whole_fragment_secant']:
  v=[r['derived_speed_mps'] for r in speed_rows if r['source']==s and r['method']==method]
  out['derived_speed_mps'][s][method]=dict(zip(['min','median','p95','max'],np.quantile(v,[0,.5,.95,1]).tolist()))
Path('/tmp/completion_motion_controls.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'aggregate':out['aggregate'],'derived_speed_mps':out['derived_speed_mps']},indent=2))
