import sys,json,gzip,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np,torch
from association.data.archive import MTADArchive
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from association.training.common import normalize_scene
from association.models.timed import TimedAssociationModel,continuous_time_encoding
from association.models.tmtaff import ModelConfig
R=Path(__file__).resolve().parents[3]
load=lambda p:json.loads(gzip.decompress((R/p).read_bytes()))
run=load('HandOver/results/STAGEWISE-COMPLETION-20261019-028/020/run.json.gz')
norm=load('HandOver/results/COMPLETION-CHECK-20261019-029/run.json.gz')['reference_normalizer']
old=load('HandOver/results/COMPLETION-CHECK-20261019-029/reference.json.gz')
manifest=json.loads(subprocess.check_output(['git','show','612afea:HandOver/results/EVIDENCE-20261018-012/prepared007/manifest.json'],cwd=R))
torch.set_num_threads(1);torch.manual_seed(run['config']['seed'])
model=TimedAssociationModel(ModelConfig(**run['requested_model_config']),run['config']['branch'],max_observations=128).eval()
W=model.input_embedding
embed=lambda x,t:torch.relu(W(x+continuous_time_encoding(t,4)))
scenes=[]
with MTADArchive(sys.argv[1] if len(sys.argv)>1 else Path.home()/'Downloads/数据集level2.zip') as archive:
 for sid in ['1671','2056','290','113']:
  scenes.append(prepare_timed_scene(_record(archive.load_scene(sid),manifest['components'][sid],'V_select',128),length=64,max_observations=128,time_unit_seconds=20.))
rows=[];totals={k:0. for k in ['n','hold_sse','copy_old_embedding_sse','time_only_change_sse','normalized_state_step_sse','state_time_cross','target_energy','target_within_track_sse','target_between_track_sse']}
attr=np.zeros(4);rawdelta=np.zeros(4);arc_heading_delta=0.;counter=np.zeros(4)
with torch.inference_mode():
 for scene in scenes:
  x,mask=normalize_scene(scene,norm);x=torch.from_numpy(x)
  for i,maskrow in enumerate(mask):
   count=int(maskrow.sum());v=x[i,:count,:4];t=x[i,:count,4]
   full=embed(v,t); mean=full.mean(0,keepdim=True)
   totals['target_energy']+=float(full.double().square().sum())
   totals['target_within_track_sse']+=float((full.double()-mean.double()).square().sum())
   for vs,ts in [(v,t),(v.flip(0),t.flip(0))]:
    truth=embed(vs[1:],ts[1:]);held=embed(vs[:-1],ts[1:]);previous=embed(vs[:-1],ts[:-1])
    totals['n']+=len(truth);sse=float((truth.double()-held.double()).square().sum());totals['hold_sse']+=sse
    totals['copy_old_embedding_sse']+=float((truth.double()-previous.double()).square().sum())
    totals['time_only_change_sse']+=float((held.double()-previous.double()).square().sum())
    for j in range(4):
     c=vs[:-1].clone();c[:,j]=vs[1:,j]
     counter[j]+=float((embed(c,ts[1:]).double()-held.double()).square().sum())
     attr[j]+=float((vs[1:,j].double()-vs[:-1,j].double()).square().sum())
    hd=(vs[1:,3].double()-vs[:-1,3].double())*norm['std'][3]
    arc_heading_delta+=float((((hd+180)%360-180)/norm['std'][3]).square().sum())
   rows.append(dict(scene=scene['scene_id'],node=scene['node_ids'][i],n=count))
 result=dict(scope='same four V scenes; CPU zero-update feature geometry; W reconstructed from original seed/config, not downloaded checkpoint',normalizer_std=norm['std'],fragment_count=len(rows),native_points=sum(r['n'] for r in rows),transition_count=int(totals['n']),hold_feature_mse=totals['hold_sse']/(totals['n']*128),saved029_hold_feature_mse=old['one_step']['state_hold']['all']['feature_mse'],copy_old_embedding_feature_mse=totals['copy_old_embedding_sse']/(totals['n']*128),pure_time_change_feature_mse=totals['time_only_change_sse']/(totals['n']*128),state_step_sum_squares=attr.tolist(),state_step_squared_shares=(attr/attr.sum()).tolist(),heading_short_arc_step_sse=arc_heading_delta,single_attribute_change_feature_mse=(counter/(totals['n']*128)).tolist(),single_attribute_sum_vs_joint=float(counter.sum()/totals['hold_sse']),within_track_variation_vs_total_energy=totals['target_within_track_sse']/totals['target_energy'])
 Path('/tmp/baseline-feature-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
qtot={k:dict(n=0,hold_sse=0.,single_attribute_sse=np.zeros(4)) for k in ['all','inside','before','after']}
heading_wrap=0;adjacent=0;native_values=[]
with torch.inference_mode():
 for scene in scenes:
  packed,mask=normalize_scene(scene,norm); packed=torch.from_numpy(packed)
  npz=np.load(R/f'HandOver/results/COMPLETION-RESIDUAL-20261019-030/predictions-{scene["scene_id"]}.npz')
  for i in range(len(packed)):
   count=int(mask[i].sum()); v=packed[i,:count,:4]; ts=packed[i,128:,4]
   observed=scene['native_values'][i,:count];native_values.append(observed)
   heading_wrap+=int((np.abs(np.diff(observed[:,3]))>180).sum());adjacent+=count-1
   b=npz['region_before_support'][i];a=npz['region_after_support'][i];valid=npz['target_valid'][i].all(-1)
   held_states=packed[i,128:,:4].clone();held_states[b]=v[0];held_states[a]=v[-1]
   target_states=torch.as_tensor((npz['target'][i]-norm['mean'])/norm['std'],dtype=torch.float32)
   held=embed(held_states,ts);truth=embed(target_states,ts)
   errs=(truth.double()-held.double()).square().sum(-1).numpy()
   changes=[]
   for j in range(4):
    c=held_states.clone();c[:,j]=target_states[:,j]
    changes.append((embed(c,ts).double()-held.double()).square().sum(-1).numpy())
   for k,m in [('all',np.ones(64,bool)),('inside',npz['region_inside_support'][i]),('before',b),('after',a)]:
    sel=m&valid;qtot[k]['n']+=int(sel.sum());qtot[k]['hold_sse']+=float(errs[sel].sum())
    qtot[k]['single_attribute_sse']+=np.array([c[sel].sum() for c in changes])
 for k,v in qtot.items():
  v['hold_mse']=v['hold_sse']/(v['n']*128);v['single_attribute_mse']=(v.pop('single_attribute_sse')/(v['n']*128)).tolist()
 result['query_hold_geometry']=qtot
 result['raw_heading_wrap_transition_count']=heading_wrap
 result['raw_heading_transition_count']=adjacent
 v=np.concatenate(native_values);result['native_speed_quantiles_p0_p50_p95_p99_max']=np.percentile(v[:,2],[0,50,95,99,100]).tolist()
 result['lat_lon_joint_step_energy_share']=float(attr[:2].sum()/attr.sum())
 result['heading_raw_vs_circular_step_energy_reduction']=float(1-arc_heading_delta/attr[3])
 Path('/tmp/baseline-feature-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k in ['query_hold_geometry','raw_heading_wrap_transition_count','raw_heading_transition_count','native_speed_quantiles_p0_p50_p95_p99_max','heading_raw_vs_circular_step_energy_reduction']},indent=2))
