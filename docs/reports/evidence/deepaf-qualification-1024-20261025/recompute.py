import json, math
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Reaggregate saved DeepAF qualification errors; no inference.')
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--previous-probe',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ROOT,OLD=args.run,args.previous_probe
MODES=('all','CV','CA','CT')
def read(path): return json.loads(path.read_text())
def metrics(p,v):
 p=np.asarray(p);v=np.asarray(v)
 return dict(position_mae_m=float(p.mean()),position_rmse_m=float(np.sqrt((p*p).mean())),velocity_rmse_mps=float(np.sqrt((v*v).mean())))
def cmp(a,b): return {k:float(a[k]-b[k]) for k in a}
def arrays(rows,key): return np.asarray([r[key] for r in rows])
def audit_endpoint(p):
 d=read(p); result={'groups':{},'group_audit_max_abs_delta':0.0}; shadow=d['truth_reset_previous_position_only']; ss={r['scene_id']:r for r in shadow['scenes']}
 for mode in MODES:
  rows=[r for r in d['scenes'] if mode=='all' or r['motion']==mode]
  pp=arrays(rows,'position_errors_m'); vv=arrays(rows,'velocity_errors_mps')
  sp=arrays([ss[r['scene_id']] for r in rows],'position_errors_m');sv=arrays([ss[r['scene_id']] for r in rows],'velocity_errors_mps')
  cp=arrays(rows,'centroid_position_errors_m');cv=arrays(rows,'centroid_velocity_errors_mps');op=arrays(rows,'control_position_errors_m');ov=arrays(rows,'control_velocity_errors_mps')
  ms=metrics(pp,vv);cs=metrics(cp,cv);os=metrics(op,ov);sh=metrics(sp[:,1:],sv[:,1:]);same=metrics(pp[:,1:],vv[:,1:]);first=metrics(sp[:,:1],sv[:,:1])
  for typ,calc in [('deepaf',ms),('visible_centroid',cs),('visible_cv',os)]:
   result['group_audit_max_abs_delta']=max(result['group_audit_max_abs_delta'],max(abs(v-d['groups'][mode][typ][k]) for k,v in calc.items()))
  for typ,calc in [('reset_queries_only',sh),('first_query_without_reset',first)]:
   result['group_audit_max_abs_delta']=max(result['group_audit_max_abs_delta'],max(abs(v-shadow['groups'][mode][typ][k]) for k,v in calc.items()))
  result['groups'][mode]=dict(n=len(rows),closed_loop=ms,centroid=cs,ols=os,shadow=sh,closed_loop_on_same_reset_queries=same,first_query_mae_m=float(pp[:,0].mean()),last_query_mae_m=float(pp[:,-1].mean()),position_mae_by_query=pp.mean(0).tolist(),shadow_position_mae_by_query=sp.mean(0).tolist(),shadow_first_equal=bool(np.array_equal(pp[:,0],sp[:,0])),shadow_mae_reduction_percent=100*(1-sh['position_mae_m']/same['position_mae_m']),scene_mae_better_than_centroid=int((pp.mean(1)<cp.mean(1)).sum()),scene_mae_better_than_ols=int((pp.mean(1)<op.mean(1)).sum()))
 return result
out={'endpoints':{},'run':read(ROOT/'run.json')}
for step in (0,256,1024):
 p=ROOT/f'adapted_labels-step{step}-heldout.json'
 if p.exists(): out['endpoints'][str(step)]=audit_endpoint(p)
if (ROOT/'adapted_labels-training.json').exists(): log=read(ROOT/'adapted_labels-training.json')
else:
 text=(ROOT/'adapted_labels-sampling.jsonl').read_text();lines=text.splitlines();log=[]
 for line in lines:
  try: log.append(json.loads(line))
  except json.JSONDecodeError: pass
out['training_rows']=len(log)
if log:
 out['training_windows']={}
 for name,rows in [('first64',log[:64]),('last64',log[-64:]),('193_256',log[192:256]),('961_1024',log[960:1024])]:
  if rows: out['training_windows'][name]=dict(n=len(rows),**{key:float(np.mean([r[key] for r in rows])) for key in ('p_mse','v_mse')})
 if OLD is not None:
  old=read(OLD/'adapted_labels-training.json');n=min(len(old),len(log));
  out['old_256_training_comparison']=dict(compared=n,sampling_identical=all(a['scene_indices']==b['scene_indices'] for a,b in zip(old[:n],log[:n])),step_identical=all(a['step']==b['step'] for a,b in zip(old[:n],log[:n])),loss_max_abs_delta={key:max(abs(a[key]-b[key]) for a,b in zip(old[:n],log[:n])) for key in ('p_mse','v_mse')})
if '256' in out['endpoints'] and OLD is not None:
 old=read(OLD/'adapted_labels-heldout.json');new=read(ROOT/'adapted_labels-step256-heldout.json')
 out['old_256_heldout_comparison']=dict(scene_order_identical=[r['scene_id'] for r in old['scenes']]==[r['scene_id'] for r in new['scenes']],error_max_abs_delta={key:float(np.max(abs(arrays(old['scenes'],key)-arrays(new['scenes'],key)))) for key in ('position_errors_m','velocity_errors_mps')})
if '1024' in out['endpoints']:
 a=read(ROOT/'adapted_labels-step256-heldout.json');b=read(ROOT/'adapted_labels-step1024-heldout.json')
 out['256_to_1024']={}
 for mode in MODES:
  ra=[r for r in a['scenes'] if mode=='all' or r['motion']==mode];rb=[r for r in b['scenes'] if mode=='all' or r['motion']==mode]
  assert [r['scene_id'] for r in ra]==[r['scene_id'] for r in rb]
  pa=arrays(ra,'position_errors_m');pb=arrays(rb,'position_errors_m');va=arrays(ra,'velocity_errors_mps');vb=arrays(rb,'velocity_errors_mps')
  aa=metrics(pa,va);bb=metrics(pb,vb)
  out['256_to_1024'][mode]=dict(relative_percent={k:100*(bb[k]/aa[k]-1) for k in aa},scene_mae_better=int((pb.mean(1)<pa.mean(1)).sum()),scene_mae_worse=int((pb.mean(1)>pa.mean(1)).sum()),controls_exactly_unchanged=all(np.array_equal(arrays(ra,k),arrays(rb,k)) for k in ('centroid_position_errors_m','centroid_velocity_errors_mps','control_position_errors_m','control_velocity_errors_mps')))
args.output.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in out.items() if k not in ('run','endpoints')},indent=2))
print('endpoints',list(out['endpoints']))
for step,values in out['endpoints'].items():
 print('ENDPOINT',step,'audit_delta',values['group_audit_max_abs_delta'])
 for mode,row in values['groups'].items():
  print(mode,'main',row['closed_loop'],'first-last',[row['first_query_mae_m'],row['last_query_mae_m']],'shadow_vs_same_queries',row['shadow']['position_mae_m'],row['closed_loop_on_same_reset_queries']['position_mae_m'])
