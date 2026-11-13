"""Read-only Squeeze of149/150: anchor-side zero support and source-slot errors.

No predictions/fitting. Slot labels and reference only describe results; they
never enter either deployed policy. Saved originals remain unchanged.
"""
from __future__ import annotations
import argparse,json,sys,time,resource
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis import audit_source_risk_arrays as independent
from tools.analysis import audit_shared_control_arrays as core

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def run(root,out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();record=core.read(root/'run.json');native=Path(record['native_run']);source=Path(record['source_risk_run']);names=record['methods']
    result=dict(status='COMPLETED',exit_code=0,scientific_run=str(root),new_predictions=0,new_fits=0,splits={})
    for split,ids in [('T_core',record['training_scene_ids']),('V_select',record['scene_ids'])]:
        errors={};harms={};role_counts={}
        for sid in ids:
            motion=load(native/'private/native-motion'/split/f'scene-{sid}.npz');ev=load(native/'private/native-evaluation'/split/f'scene-{sid}.npz');nt=len(ev['target_ids'])
            cache=load(native/'private/native-events'/split/f'scene-{sid}.npz');z=load(root/'private'/f'states-{split}-{sid}.npz')
            base={k[len(names[0])+1:]:v for k,v in z.items() if k.startswith(names[0]+'_')}
            zero=np.zeros((nt,64),bool);before=zero.copy();after=zero.copy()
            for target,gid in enumerate(base['aligned_group_indices'][:nt]):
                if gid<0:continue
                a,b=base['group_offsets'][gid:gid+2];members=base['group_node_indices'][a:b];anchor=members[0]
                zero[target]=~motion['support'][members].any(0)
                l,r=cache['event_offsets'][anchor:anchor+2];t=cache['event_times_seconds'][l:r]
                before[target]=zero[target]&(motion['query_seconds']<t[0]);after[target]=zero[target]&(motion['query_seconds']>t[-1])
            region=dict(zero=zero,zero_before=before,zero_after=after)
            vectors={}
            for name in names:
                prefix=name+'_';prediction=z[prefix+'aligned_prediction'][:nt];valid=z[prefix+'aligned_prediction_valid'][:nt,...,:2].all(-1)&ev['target_valid'][...,:2].all(-1)
                e=independent.distances(prediction,ev['target_states']);vectors[name]=(e,valid)
                for r,mask in region.items():errors.setdefault((name,r),[]).extend(e[valid&mask].tolist())
            for name in names[1:]:
                for r,mask in region.items():
                    e,v=vectors[name];old,ov=vectors[names[0]];d=(e-old)[v&ov&mask]
                    harms.setdefault((name,r),[]).extend(d.tolist())
            if record['mode']=='temporal':
                pack=load(source/'private'/(f'training-{sid}.npz' if split=='T_core' else f'policy-{sid}.npz'))
                scores=load(root/'private'/f'policy-{split}-{sid}.npz');registry={str(t):i for i,t in enumerate(ev['target_ids'])};claimed=set()
                for gid,(a,b) in enumerate(zip(pack['group_offsets'][:-1],pack['group_offsets'][1:])):
                    members=np.flatnonzero(pack['membership']==gid);identity=str(ev['identities'][members[0]])
                    if identity not in registry or identity in claimed:continue
                    claimed.add(identity);target=registry[identity];available=pack['available'][a:b];use=ev['target_valid'][target,:,:2].all(-1)&(available.sum(0)>=2)
                    if not use.any():continue
                    e=independent.distances(pack['source_states'][a:b],ev['target_states'][target]);e=np.where(available,e,np.inf);best=e.min(0)
                    role=(e[0]>best+1e-6)&available[0]
                    selected={}
                    for name in names:
                        score=scores[name+'__scores'][a:b]
                        mass=np.where(available & (score==np.where(available,score,np.inf).min(0)),1.,0.)
                        mass/=np.maximum(mass.sum(0),1)
                        selected[name]=np.sum(mass*np.where(available,e,0),axis=0)
                    for label,mask in [('all',use),('slot0_not_best',use&role),('slot0_best',use&~role)]:
                        for name in names:
                            d=selected[name]-best;old=selected[names[0]]-best;bad=d>1e-6;obad=old>1e-6
                            data=role_counts.setdefault(label,{}).setdefault(name,dict(points=0,wrong=0,rescued=0,newly_wrong=0,excess_sum_m=0.,excess_over_1km=0))
                            data['points']+=int(mask.sum());data['wrong']+=int((mask&bad).sum());data['rescued']+=int((mask&obad&~bad).sum());data['newly_wrong']+=int((mask&~obad&bad).sum());data['excess_sum_m']+=float(d[mask].sum());data['excess_over_1km']+=int((mask&(d>1000)).sum())
        block=dict(zero_support={},actual_harm={},source_slot_selection=role_counts)
        for (name,region),v in errors.items():block['zero_support'].setdefault(name,{})[region]=core.stats(np.asarray(v))
        for (name,region),v in harms.items():
            d=np.asarray(v);block['actual_harm'].setdefault(name,{})[region]=dict(points=len(d),new_harm_over_1km=int((d>1000).sum()),max_harm_m=float(max(0.,d.max())) if len(d) else None)
        result['splits'][split]=block
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        scope='saved prediction-only Squeeze; fixed anchor before/after and label-only anonymous source slot strata, not causal mechanisms or policy inputs')
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run.resolve(),a.output.resolve())
