"""087 full056-group evaluation with explicit recorded deterministic routing."""
import numpy as np
from association.evaluation.helper_sequence_position import evaluate_helper_sequence, _helper_predict
from association.models.helper_temporal import validate_plan
from association.utils.runtime import read_json,write_json

SCOPE='087-helper-temporal-evaluation'


def _temporal_predict(model,motion,groups,predictions,check,*,initial=False):
    if model.route_log: raise ValueError('temporal scene route log was not consumed')
    output,residual,extra=_helper_predict(model,motion,groups,predictions,check,initial=initial)
    if len(model.route_log)!=len(groups): raise ValueError('every full output group must have a route')
    q=motion['states'].shape[1];plans=[];applied=[]
    for group,record in zip(groups,model.route_log):
        h=len(group)-1
        plan=validate_plan(record['planned'],h,q);actual=validate_plan(record['applied'],h,q)
        identity=np.broadcast_to(np.arange(h),plan.shape)
        if not np.array_equal(actual,plan if model.correspondence=='shuffled' else identity):
            raise ValueError('087 actual evaluation route differs from declared mode')
        plans.append(plan.T);applied.append(actual.T)
        model.route_audit['group_count']+=1;model.route_audit['multihelper_groups']+=int(h>=2)
        model.route_audit['helper_query_slots']+=h*q
        model.route_audit['planned_changed_slots']+=int((plan!=identity).sum())
        model.route_audit['applied_changed_slots']+=int((actual!=identity).sum())
    extra['group_helper_query_route_plan']=np.concatenate(plans,axis=0) if plans else np.empty((0,q),np.int64)
    extra['group_helper_query_applied_route']=np.concatenate(applied,axis=0) if applied else np.empty((0,q),np.int64)
    model.route_log=[]
    return output,residual,extra


def evaluate_helper_temporal(model,validation_raw,source056,metadata,output,check=None,*,initial=False,route_seed=20348123):
    model.reset_route(route_seed)
    model.route_audit=dict(group_count=0,multihelper_groups=0,helper_query_slots=0,planned_changed_slots=0,applied_changed_slots=0)
    result=evaluate_helper_sequence(model,validation_raw,source056,metadata,output,check,initial=initial,
        predictor=_temporal_predict,evaluation_scope=SCOPE)
    result['routing']=dict(evaluation_seed=route_seed,correspondence=model.correspondence,
        exposure=model.route_audit,final_rng=model.route_rng.bit_generator.state)
    from pathlib import Path
    path=Path(output)
    write_json(path/'summary.json',result)
    state=read_json(path/'run.json');state['routing']=result['routing'];write_json(path/'run.json',state)
    return result
