"""Independent 163 population/reference audit; no purification state refit or neural inference."""
from __future__ import annotations
import argparse,json,time,resource
from pathlib import Path
import numpy as np

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def distance(a,b):
    def unit(x):
        lat,lon=np.deg2rad(x[...,0]),np.deg2rad(x[...,1]);return np.stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)),axis=-1)
    a,b=unit(a),unit(b);return 6371000.*np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1))

def audit(run,out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();record=json.loads((run/'run.json').read_text());result=dict(status='RUNNING',run=str(run));checks=0;maximum=0.
    def exact(a,b,label):
        nonlocal checks
        if not np.array_equal(a,b):raise AssertionError(label)
        checks+=np.asarray(a).size
    try:
        exact(record['status'],'COMPLETED','science terminal');native=Path(record['native_run']);source=Path(record['source_risk_run']);expression=Path(record['expression_run']);result['splits']={}
        for split,ids,fn in [('T_core',record['training_scene_ids'],'T-summary.json'),('V_select',record['scene_ids'],'summary.json')]:
            sums=dict(queries=0,rare=0,rare_clean_multi=0,rare_clean_one=0,rare_clean_zero=0,reference_best_set_flips=0)
            for sid in ids:
                arrays=load(run/'private'/f'counterfactual-{split}-{sid}.npz');pack=load(source/'private'/(f'training-{sid}.npz' if split=='T_core' else f'policy-{sid}.npz'));ev=load(native/'private/native-evaluation'/split/f'scene-{sid}.npz');m=load(native/'private/native-motion'/split/f'scene-{sid}.npz');policy=load(expression/'private'/f'policy-{split}-{sid}.npz');membership=policy['membership'];groups=[np.flatnonzero(membership==i) for i in range(len(pack['group_offsets'])-1)];registry={str(t):i for i,t in enumerate(ev['target_ids'])};claimed=set();shape=arrays['eligible'].shape
                expected={k:np.zeros(shape,bool) for k in ('eligible','rare','pure','alt_seen','best_flips','clean_multi','clean_one','clean_zero','quadratic_wrong','network_wrong','quadratic_alt_correct','network_alt_correct')};names={'quadratic':'WS_T128_Q_HARD','network':'WS_T128_RESIDUAL_HARD'};reference_index={}
                for key in arrays:
                    if key.endswith('__target_identity'):
                        prefix=key[:-len('target_identity')];reference_index[(int(arrays[prefix+'group_index']),str(arrays[key]))]=prefix
                for gid,group in enumerate(groups):
                    identity=str(ev['identities'][group[0]]);expected['pure'][gid]=len(set(ev['identities'][group].tolist()))==1
                    if identity not in registry or identity in claimed:continue
                    claimed.add(identity);a,b=pack['group_offsets'][gid:gid+2];available=pack['available'][a:b];use=ev['target_valid'][registry[identity],:,:2].all(-1)&(available.sum(0)>=2);expected['eligible'][gid]=use
                    good=group[ev['identities'][group]==identity];counts=sum(m['support'][good[m['sources'][good]==s]].any(0) for s in np.unique(m['sources'][good]));expected['clean_multi'][gid]=use&(counts>=2);expected['clean_one'][gid]=use&(counts==1);expected['clean_zero'][gid]=use&(counts==0)
                    if not use.any():continue
                    pos=pack['source_states'][a:b];e=distance(pos,ev['target_states'][registry[identity]]);best=np.where(available,e,np.inf).min(0);bs=available&(e<=best[None,:]+1e-6);expected['rare'][gid]=use&available[0]&(e[0]>best+1e-6)
                    for tag,name in names.items():
                        s=policy[name+'__scores'][a:b];chosen=available&(s==np.where(available,s,np.inf).min(0));mass=chosen/np.maximum(chosen.sum(0),1);picked=(mass*np.where(available,e,0)).sum(0);expected[tag+'_wrong'][gid]=use&(picked-best>1e-6)
                    for other in set(map(str,ev['identities'][group].tolist()))-{identity}:
                        prefix=reference_index[(gid,other)];exact(arrays[prefix+'source_row_start'],a,'reference source rows')
                        if other not in registry:
                            exact(arrays[prefix+'target_index'],-1,'unknown reference');continue
                        target=registry[other];mask=use&ev['target_valid'][target,:,:2].all(-1);exact(arrays[prefix+'valid_original_query'],mask,'reference valid population');exact(arrays[prefix+'target_index'],target,'reference target')
                        alternative=distance(pos,ev['target_states'][target]);np.testing.assert_allclose(alternative[:,mask],arrays[prefix+'source_errors_m'][:,mask],rtol=1e-10,atol=1e-5);maximum=max(maximum,float(np.abs(alternative[:,mask]-arrays[prefix+'source_errors_m'][:,mask]).max()) if mask.any() else 0.)
                        other_best=np.where(available,alternative,np.inf).min(0);other_set=available&(alternative<=other_best[None,:]+1e-6);expected['alt_seen'][gid]|=mask;expected['best_flips'][gid]|=mask&(other_set!=bs).any(0)
                        for tag,name in names.items():
                            s=policy[name+'__scores'][a:b];chosen=available&(s==np.where(available,s,np.inf).min(0));mass=chosen/np.maximum(chosen.sum(0),1);picked=(mass*np.where(available,alternative,0)).sum(0);expected[tag+'_alt_correct'][gid]|=mask&(picked-other_best<=1e-6)
                for key,v in expected.items():exact(v,arrays[key],key)
                sums['queries']+=int(expected['eligible'].sum());sums['rare']+=int(expected['rare'].sum());sums['reference_best_set_flips']+=int(expected['best_flips'].sum())
                for key in ('multi','one','zero'):sums['rare_clean_'+key]+=int((expected['rare']&expected['clean_'+key]).sum())
            summary=json.loads((run/fn).read_text())['all_components'];exact(sums['queries'],summary['all']['queries'],'summary population');exact(sums['rare'],summary['rare']['queries'],'summary rare')
            for key in ('multi','one','zero'):exact(sums['rare_clean_'+key],summary['rare']['purification_'+key],'summary purified rare '+key)
            exact(sums['reference_best_set_flips'],summary['all']['reference_best_set_flips'],'summary reference flips');result['splits'][split]=sums
        result.update(status='COMPLETED',exit_code=0,scope='independent native/identity support counts, official choice errors and all alternative-reference geometry/flip masks; not purified-state features/forward/selected error, physical truth or deployed-method audit')
    except BaseException as e:result.update(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise
    finally:
        result.update(exact_elements_checked=checks,alternative_distance_max_difference_m=maximum,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);destination=out if result['status']=='COMPLETED' else out.with_suffix('.failed.json');destination.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.run.resolve(),a.output.resolve())
