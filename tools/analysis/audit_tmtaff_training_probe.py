"""Read-only saved153 output audit. No forward, fitting, V or raw CSV."""
from __future__ import annotations
import argparse,json,sys,time,resource
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis.audit_source_risk_arrays import distances

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def run(root,out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();record=json.loads((root/'run.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if record['status']!='COMPLETED' or record['completed_scenes']!=8:raise ValueError('completed153 required')
    native=Path(record['native_run']);values={};positions={};interventions={};pairs=nodes=checks=0
    for row in summary['scenes']:
        sid=row['scene_id'];z=load(root/'private'/f'predictions-{sid}.npz');ev=load(native/'private/native-evaluation/T_core'/f'scene-{sid}.npz');m=load(native/'private/native-motion/T_core'/f'scene-{sid}.npz')
        pair=z.pop('pairs');registry={str(t):i for i,t in enumerate(ev['target_ids'])};indices=[]
        for i,j in pair:
            if ev['identities'][i]!=ev['identities'][j] or m['sources'][i]==m['sources'][j]:raise ValueError('positive cross-source binding')
            indices.append(registry[str(ev['identities'][i])])
        pairs+=len(pair);nodes+=len(np.unique(pair));truth=ev['target_states'][indices];valid=ev['target_valid'][indices]
        counts=m['support'][pair].sum(1);regions={'all':np.ones(counts.shape,bool),'both_supported':counts==2,'one_supported':counts==1,'zero_supported':counts==0}
        for name,pred in z.items():
            if pred.shape!=truth.shape or not np.isfinite(pred).all():raise ValueError('saved shape/finite')
            errors={'position':distances(pred,truth),'speed':np.abs(pred[...,2]-truth[...,2]),'heading':np.abs((pred[...,3]-truth[...,3]+180)%360-180)}
            for region,mask in regions.items():
                for attr,cols in [('position',[0,1]),('speed',[2]),('heading',[3])]:values.setdefault((region,name,attr),[]).extend(errors[attr][mask&valid[...,cols].all(-1)].tolist())
            positions.setdefault(name,[]).append(pred[...,:2].reshape(-1,2))
            if name.startswith('TFM_'):
                d=np.abs(pred-z['TFM']);interventions.setdefault(name,[]).append(d.reshape(-1,4))
    for (region,name,attr),v in values.items():
        x=np.asarray(v);expected=summary['regions'][region][name][attr]
        actual={'count':len(x),'mae':float(x.mean()),'rmse':float(np.sqrt(np.mean(x*x))),'p95':float(np.quantile(x,.95)),'p99':float(np.quantile(x,.99))}
        for key,val in actual.items():
            if not np.isclose(val,expected[key],rtol=1e-9,atol=1e-5):raise ValueError(f'{region}/{name}/{attr}/{key}')
            checks+=1
    result=dict(status='COMPLETED',exit_code=0,scientific_run=str(root),scenes=8,pairs=pairs,unique_node_appearances=nodes,aggregate_checks=checks,
        saved_position_unique_rows={n:len(np.unique(np.concatenate(v),axis=0)) for n,v in positions.items()},
        saved_intervention_max_absolute_attribute_difference={n:np.max(np.concatenate(v),axis=0).tolist() for n,v in interventions.items()},
        elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        new_forwards=0,new_fits=0,scope='saved six outputs and native T reference/positive-pair binding; no independent TCM/head reconstruction, unique failure cause or physical truth certification')
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run.resolve(),a.output.resolve())
