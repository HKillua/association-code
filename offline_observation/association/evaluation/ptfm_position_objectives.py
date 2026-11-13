"""091 T-exposure calibration and fixed training probe; no V input accepted."""
from __future__ import annotations
import copy,math
from pathlib import Path
import numpy as np
import torch
from association.models.ptfm import ptfm_inputs,position_objective_values,POSITION_OBJECTIVES
from association.training.evidence_fusion import TRAIN_MODES,training_members

CALIBRATION='full-source-T-initial-sample-mean-v1'
ERRORS=('normalized_squared','radial-linear','geodesic-linear')
REGIONS=('anchor_inside_support','anchor_outside_helper_supported','group_no_support')
PROBE_UPDATES=96


def replay_samples(train,rows,config,check=None):
    """Reconstruct every original logged sample; old logs store counts, not masks."""
    scenes={str(scene['scene_id']):scene for scene in train}
    if len(scenes)!=len(train) or any(s.get('split')!='T_core' for s in train):
        raise ValueError('091 calibration/probe requires distinct T_core scenes only')
    batch_size=config['training']['batch_size']
    for index,row in enumerate(rows,1):
        if check:check()
        mode=TRAIN_MODES[(index-1)%len(TRAIN_MODES)]
        if row.get('step')!=index or row.get('mode')!=mode or len(row.get('batch',[]))!=batch_size:
            raise ValueError('091 source successful-batch sequence differs')
        for sample in row['batch']:
            if check:check()
            sid=sample.get('scene_id');scene=scenes.get(sid);anchor=sample.get('anchor_index')
            if scene is None or type(anchor) is not int or not 0<=anchor<len(scene['identities']):
                raise ValueError('091 original T sample identity differs')
            members=training_members(scene,anchor,mode)
            target=torch.as_tensor(scene['target'][anchor],dtype=torch.float64)
            valid=torch.as_tensor(scene['valid'][anchor],dtype=torch.bool)
            eligible=~scene['motion']['support'][anchor]&scene['motion']['support'][members[1:]].any(0)
            counts=[int(valid[:,:2].all(-1).sum()),int(valid[:,2].sum()),int(valid[:,3].sum())]
            expected=dict(scene_id=sid,anchor_index=anchor,node_id=str(scene['node_ids'][anchor]),
                members=[int(x) for x in members],valid_position_points=counts[0],
                eligible_query_points=int(eligible.sum()),
                eligible_valid_position_points=int((eligible&valid[:,:2].all(-1).numpy()).sum()),
                task_valid_counts=counts)
            if sample!=expected:raise ValueError('091 reconstructed source members/task masks differ')
            yield row['step'],mode,sample,scene,members,target,valid


def validate_calibration(record,config):
    n=config['training']['updates']*config['training']['batch_size']
    if (not isinstance(record,dict) or record.get('method')!=CALIBRATION
            or record.get('split')!='T_core' or record.get('truth_used') is not True
            or record.get('model_forwards')!=0 or record.get('validation_used') is not False
            or record.get('successful_updates')!=config['training']['updates']
            or record.get('sample_count')!=n or record.get('global_sample_denominator')!=n
            or type(record.get('zero_supervision_samples')) is not int
            or not 0<=record['zero_supervision_samples']<=n
            or type(record.get('valid_position_points')) is not int or record['valid_position_points']<1
            or not isinstance(record.get('source_fit'),str) or not record['source_fit']):
        raise ValueError('091 calibration provenance/population differs')
    means=record.get('initial_sample_equal_means',{})
    scales=record.get('alpha',{})
    if set(means)!=set(ERRORS) or set(scales)!=set(POSITION_OBJECTIVES):
        raise ValueError('091 calibration error/alpha keys differ')
    for value in (*means.values(),*scales.values()):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('091 calibration requires positive finite means/scales')
    for name in POSITION_OBJECTIVES:
        if not math.isclose(scales[name],means['normalized_squared']/means[name],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('091 calibration alpha formula differs')
    return record


def calibrate_initial_inputs(train,rows,config,normalizer,source_fit,check=None):
    """No learned model is constructed/called; only the physical fixed095 base."""
    if len(rows)!=config['training']['updates']:
        raise ValueError('091 calibration requires the full original successful log')
    sums={key:0. for key in ERRORS};n=zero=points=0
    scale=torch.as_tensor(normalizer['scale'],dtype=torch.float64)
    for _,_,_,scene,members,target,valid in replay_samples(train,rows,config,check):
        base=ptfm_inputs(scene['motion'],members,include_prior=True)[4]
        values,mask=position_objective_values(base,target,valid,scale)
        n+=1;count=int(mask.sum());points+=count;zero+=int(not count)
        for key in ERRORS:sums[key]+=float(values[key][mask].mean()) if count else 0.
    means={key:value/n for key,value in sums.items()}
    if any(not math.isfinite(v) or v<=0 for v in means.values()):
        raise ValueError('091 initial loss calibration requires positive finite T means')
    record=dict(method=CALIBRATION,split='T_core',truth_used=True,validation_used=False,
        model_forwards=0,source_fit=str(Path(source_fit).resolve()),successful_updates=len(rows),
        sample_count=n,global_sample_denominator=n,zero_supervision_samples=zero,valid_position_points=points,
        initial_sample_equal_means=means,alpha={key:means['normalized_squared']/means[key] for key in POSITION_OBJECTIVES},
        interpretation='Match initial sample-mean data objective only; not gradient, Adam, clipping or L2 equivalence.',
        mask_boundary='Canonical T inputs and task mask counts reconstructed; old JSONL has no saved per-query mask.')
    return validate_calibration(record,config)


class _Totals:
    def __init__(self):
        self.samples=0;self.points=0;self.zero=0
        self.point_sums={k:0. for k in ERRORS};self.sample_sums={k:0. for k in ERRORS};self.square_km_sum=0.
    def add(self,values,mask,selection):
        self.samples+=1;n=int(mask.sum());use=mask&selection;count=int(use.sum())
        self.points+=count;self.zero+=int(not n)
        for key in ERRORS:
            total=float(values[key][use].sum());self.point_sums[key]+=total
            self.sample_sums[key]+=total/n if n else 0.
        self.square_km_sum+=float(values['geodesic-linear'][use].square().sum())
    def describe(self,global_sample_count):
        return dict(sample_count=self.samples,valid_point_count=self.points,zero_supervision_samples=self.zero,
            point_error_sums=self.point_sums,sample_mean_sums=self.sample_sums,
            point_geodesic_square_sum_km2=self.square_km_sum,
            conditional_sample_means={k:v/self.samples if self.samples else None for k,v in self.sample_sums.items()},
            global_sample_mean_contributions={k:v/global_sample_count for k,v in self.sample_sums.items()},
            global_sample_denominator=global_sample_count,
            point_mae_m=1000*self.point_sums['geodesic-linear']/self.points if self.points else None,
            point_rmse_m=1000*math.sqrt(self.square_km_sum/self.points) if self.points else None)


def fixed_training_probe(model,train,rows,config,*,source_fit,label,check=None):
    """One fixed endpoint pass; all regions retain the original sample denominator."""
    chosen=rows[:min(PROBE_UPDATES,config['training']['updates'])]
    expected_samples=len(chosen)*config['training']['batch_size']
    totals=_Totals();by_mode={key:_Totals() for key in TRAIN_MODES};by_region={key:_Totals() for key in REGIONS}
    # All three endpoints use CPU arithmetic, including CUDA-trained endpoints.
    # Copying avoids moving or mutating the live optimizer's parameters or RNG.
    model=copy.deepcopy(model).cpu();device='cpu';was_training=model.training;model.eval();samples=[]
    try:
        with torch.inference_mode():
            for step,mode,sample,scene,members,target,valid in replay_samples(train,chosen,config,check):
                args=ptfm_inputs(scene['motion'],members,device=device,include_prior=True)
                result=model(*args)
                values,mask=position_objective_values(result['states'].cpu(),target,valid,model.scale.cpu())
                all_points=torch.ones_like(mask);totals.add(values,mask,all_points);by_mode[mode].add(values,mask,all_points)
                inside=torch.as_tensor(scene['motion']['support'][members[0]],dtype=torch.bool)
                helper=torch.as_tensor(scene['motion']['support'][members[1:]].any(0),dtype=torch.bool)
                regions=dict(anchor_inside_support=inside,anchor_outside_helper_supported=~inside&helper,
                             group_no_support=~inside&~helper)
                for key,selection in regions.items():by_region[key].add(values,mask,selection)
                n=int(mask.sum())
                samples.append(dict(step=step,mode=mode,scene_id=sample['scene_id'],node_id=sample['node_id'],
                    anchor_index=sample['anchor_index'],valid_position_points=n,
                    sample_means={key:float(value[mask].mean()) if n else 0. for key,value in values.items()}))
            l2=float(config['model']['l2']*model.l2())
    finally:model.train(was_training)
    if len(samples)!=expected_samples:raise ValueError('091 fixed T probe population differs')
    return dict(label=label,source_fit=str(Path(source_fit).resolve()),split='T_core',truth_used=True,
        validation_used=False,selection=False,device='cpu',optimizer_updates=0,model_forwards=len(samples),
        source_updates=len(chosen),sample_count=len(samples),L2_penalty=l2,
        metrics=totals.describe(len(samples)),by_mode={k:v.describe(len(samples)) for k,v in by_mode.items()},
        by_original_region={k:v.describe(len(samples)) for k,v in by_region.items()},samples=samples,
        interpretation='Fixed repeated training exposures; not held-out validation, full T evaluation or convergence evidence.',
        denominator_note='Regional sample contributions retain each original sample n_valid and global N; mode conditional means have their own sample count.')
