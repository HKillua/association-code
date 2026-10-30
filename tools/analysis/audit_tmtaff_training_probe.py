"""Read-only saved153 output audit. No forward, fitting, V or raw CSV."""
from __future__ import annotations
import argparse,json,sys,time,resource
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis.audit_source_risk_arrays import distances

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def errors(pred,truth,*,double):
    # The original diagnostic evaluated float32 TFM arrays with float64 truth.
    # Reproduce that arithmetic explicitly; independent distances use float64.
    pred=np.asarray(pred,np.float64) if double else pred
    lat1,lat2=np.deg2rad(pred[...,0]),np.deg2rad(truth[...,0])
    dl=np.deg2rad((pred[...,1]-truth[...,1]+180)%360-180)
    h=np.clip(np.sin((lat1-lat2)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dl/2)**2,0,1)
    position=2*6371000.*np.arctan2(np.sqrt(h),np.sqrt(1-h))
    if double and not np.allclose(position,distances(pred,np.asarray(truth,np.float64)),rtol=1e-9,atol=1e-5,equal_nan=True):raise ValueError('independent double geometry')
    return {'position':position,'speed':np.abs(pred[...,2]-truth[...,2]),'heading':np.abs((pred[...,3]-truth[...,3]+180)%360-180)}

def run(root,out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();record=json.loads((root/'run.json').read_text());summary=json.loads((root/'summary.json').read_text())
    if record['status']!='COMPLETED' or record['completed_scenes']!=8:raise ValueError('completed153 required')
    native=Path(record['native_run']);values={};double_values={};positions={};interventions={};pairs=nodes=checks=0
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
            original_errors=errors(pred,truth,double=False);double_errors=errors(pred,truth,double=True)
            for region,mask in regions.items():
                for attr,cols in [('position',[0,1]),('speed',[2]),('heading',[3])]:
                    use=mask&valid[...,cols].all(-1);key=(region,name,attr)
                    values.setdefault(key,[]).extend(original_errors[attr][use].tolist());double_values.setdefault(key,[]).extend(double_errors[attr][use].tolist())
            positions.setdefault(name,[]).append(pred[...,:2].reshape(-1,2))
            if name.startswith('TFM_'):
                d=np.abs(pred-z['TFM']);interventions.setdefault(name,[]).append(d.reshape(-1,4))
    for (region,name,attr),v in values.items():
        x=np.asarray(v);expected=summary['regions'][region][name][attr]
        actual={'count':len(x),'mae':float(x.mean()),'rmse':float(np.sqrt(np.mean(x*x))),'p95':float(np.quantile(x,.95)),'p99':float(np.quantile(x,.99))}
        for key,val in actual.items():
            if not np.isclose(val,expected[key],rtol=1e-9,atol=1e-5):raise ValueError(f'{region}/{name}/{attr}/{key}')
            checks+=1
    double_summary={};max_difference=0.
    for (region,name,attr),v in double_values.items():
        x=np.asarray(v);old=np.asarray(values[(region,name,attr)]);max_difference=max(max_difference,float(np.max(np.abs(x-old))))
        double_summary.setdefault(region,{}).setdefault(name,{})[attr]=dict(count=len(x),mae=float(x.mean()),rmse=float(np.sqrt(np.mean(x*x))),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)))
    result=dict(status='COMPLETED',exit_code=0,scientific_run=str(root),scenes=8,pairs=pairs,unique_node_appearances=nodes,aggregate_checks=checks,
        double_precision_reanalysis=double_summary,max_original_vs_double_attribute_error_difference=max_difference,
        saved_position_unique_rows={n:len(np.unique(np.concatenate(v),axis=0)) for n,v in positions.items()},
        saved_intervention_max_absolute_attribute_difference={n:np.max(np.concatenate(v),axis=0).tolist() for n,v in interventions.items()},
        elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        new_forwards=0,new_fits=0,scope='original mixed-dtype arithmetic reproduced; separate float64 haversine/unit-vector cross-check; saved six outputs and T reference binding only, no TCM/head or unique failure cause certification')
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run.resolve(),a.output.resolve())
