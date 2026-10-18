"""Read existing evidence and CPU loss derivatives; no optimizer/model/GPU/data scan."""
import json,gzip,math,sys
from pathlib import Path
import numpy as np
import torch
REPO=Path('/Users/a1/Projects/association-code');sys.path.insert(0,str(REPO))
from association.fusion import FusionLoss
p=REPO/'HandOver/results/COOPERATIVE-DEVELOPMENT-20261018-018/files'
run=json.loads(gzip.decompress((p/'run.json.gz').read_bytes()))
history=json.loads((p/'history.json').read_text())
old=json.loads(gzip.decompress((REPO/'HandOver/results/TMTAFF-TIME-EVAL-20261018-014/files/summary.json.gz').read_bytes()))['normalizer']
assert old['std']==run['error_scales']
assert old['training_scene_ids']==[c['scene_id'] for c in run['scene_contract'][0]]
assert old['observed_point_count']==run['support_time_scaling']['valid_native_time_count']
assert old['training_splits']==['T_core']
assert old['adapter_config']['max_observations']==128
means=np.array(old['mean']);std=np.array(run['error_scales']);torch.set_num_threads(1)
objective=FusionLoss(position_loss_mode='standard_smooth_l1')
examples=[]
# Keep comparable position/course residuals while illustrating speed-head curvature.
for speed in [.001,.1,math.log(2),1.,10.,20.,100.,200.,1000.]:
 raw_speed=math.log(math.expm1(speed)) if speed<20 else speed
 desired=np.array([means[0]+.5,means[1]+.5,raw_speed,means[3]+90.])
 z=torch.tensor((desired-means)/std,dtype=torch.float32,requires_grad=True)
 raw=z*torch.tensor(std,dtype=torch.float32)+torch.tensor(means,dtype=torch.float32)
 pred=torch.stack([raw[0].clamp(-90,90),(raw[1]+180).remainder(360)-180,
                   torch.nn.functional.softplus(raw[2]),raw[3].remainder(360)])[None,None]
 truth=torch.tensor([means[0],means[1],20.,means[3]],dtype=torch.float32)[None,None]
 result=objective(pred,truth,torch.ones_like(truth,dtype=torch.bool))
 grad=torch.autograd.grad(result['total'],z)[0]
 examples.append(dict(predicted_speed=float(pred[0,0,2].detach()),target_speed=20.,
     normalized_speed_projection=float(z[2].detach()),four_losses={k:float(v.detach()) for k,v in result['losses'].items()},
     d_total_d_z=grad.detach().tolist(),scope='one synthetic valid point, log_variance=0; no network gradient or optimizer'))
rows={}
for r in history:rows.setdefault(r['stage']+(':'+r['routing'] if r['stage']=='adapt' else ''),[]).append(r)
clipping={}
for key, rs in rows.items():
 g=np.array([r['gradient_norm'] for r in rs]);clipping[key]=dict(updates=len(rs),preclip_min=float(g.min()),preclip_median=float(np.median(g)),preclip_max=float(g.max()),clip_count=int((g>1).sum()),
 gradient_norm_scope='combined preclip parameter norm; not postclip norm or individual task/module gradients',
 valid_supervision_per_attribute_range=[min(r['valid_counts'][0] for r in rs),max(r['valid_counts'][0] for r in rs)])
inspection=json.loads((REPO/'HandOver/results/COOPERATIVE-ARRAYS-20261018-016/files/inspection.json').read_text())
ref=inspection['speed']['plain']['oracle']['reference_valid']
map_examples=[]
for z in [-.3,-.28,-.26,-.24,-.2,0.,.1,1.]:
 raw=z*std[2]+means[2];v=float(torch.nn.functional.softplus(torch.tensor(raw,dtype=torch.float64)))
 map_examples.append(dict(z=z,raw=float(raw),speed=v))
result=dict(scope='CPU arithmetic only, no train/inference/ZIP/T_test; 018 std direct, 014 same-contract mean auxiliary',
 normalizer_direct018={'std':run['error_scales'],'mean':'NOT_RETURNED_IN_018'},
 normalizer_auxiliary014={'mean':old['mean'],'std':old['std'],'selected_native_count':old['observed_point_count'],
    'same_training_scene_order_as018':True,'same_std_as018':True,'same_selected_native_count_as018':True,
    'limit':'same-contract evidence; actual018 state_mean buffer and normalizer mean not locally returned'},
 speed_reference_distribution_016_saved_V=ref,
 speed_mapping_using014_mean_and018_std=map_examples,
 synthetic_loss_gradient_examples=examples,real018_preclip_norms=clipping,
 heading_input_boundary=dict(linear_z_distance_359_1=float(358/std[3]),shortest_angle_scaled_359_1=float(2/std[3]),
    interpretation='359 vs 1 degrees are close on the circle but scalar z-score input is discontinuous; source code fact, causal effect unmeasured'),
 main_objective='0.5*exp(-log_variance_k)*L_k + softplus(log_variance_k); position degree SmoothL1 beta=.001, speed log square, heading 1-cos; utility error_scales are not main loss scales',
 stage_learning_opportunities={'encoder_and_base_identity':100,'each_selected_decoder':200,'utility_head_query_pool':100,
    'all_run_updates':600,'note':'teacher100 plus each arm adapt100; no selected full system received all600 updates'})
Path(__file__).with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'018_std':run['error_scales'],'clipping':clipping,'speed_examples':map_examples,'grad_speed_at_v0.001':examples[0]['d_total_d_z'],'grad_speed_at_v200':examples[-2]['d_total_d_z']},indent=2))
