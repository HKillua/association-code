"""Independent NumPy output audit of149/150; no production policy or fitting."""
from __future__ import annotations
import argparse,json,sys,time,resource
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis import audit_shared_control_arrays as core

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def audit(run,out):
    if out.exists():raise FileExistsError(out)
    t=time.monotonic();checks=core.Checks();result=dict(status='RUNNING',run=str(run))
    try:
        record=core.read(run/'run.json');checks.exact(record['status']=='COMPLETED' and record['exit_code']==0,'science terminal')
        native=Path(record['native_run']);names=tuple(record['methods']);result['splits']={}
        for split,ids,filename,rowsfile in [('T_core',record['training_scene_ids'],'T-summary.json','T-scene-summary.json.gz'),('V_select',record['scene_ids'],'summary.json','scene-summary.json.gz')]:
            summary,rows=core.read(run/filename),core.read(run/rowsfile);items=[]
            for sid,row in zip(ids,rows):
                z=load(run/'private'/f'states-{split}-{sid}.npz')
                saved={n:{k[len(n)+1:]:v for k,v in z.items() if k.startswith(n+'_')} for n in names}
                ev=load(native/'private/native-evaluation'/split/f'scene-{sid}.npz');motion=load(native/'private/native-motion'/split/f'scene-{sid}.npz')
                base=saved[names[0]];nt=len(ev['target_ids'])
                checks.array(base['target_ids'],ev['target_ids'],'native registry')
                for k,original in [('aligned_truth','target_states'),('aligned_truth_valid','target_valid')]:checks.array(base[k][:nt],ev[original],'native '+k)
                registry={str(v):i for i,v in enumerate(ev['target_ids'])}
                groups=[base['group_node_indices'][a:b] for a,b in zip(base['group_offsets'][:-1],base['group_offsets'][1:])]
                checks.array(base['group_anchor_target_indices'],np.array([registry[str(ev['identities'][g[0]])] for g in groups]),'anchor correspondence')
                counts=np.asarray([sum(motion['support'][g[motion['sources'][g]==s]].any(0) for s in np.unique(motion['sources'][g])) for g in groups])
                mask=counts>0 if record['mode']=='zero' else counts<=1
                for name in names[1:]:
                    checks.array(saved[name]['group_states'][mask],base['group_states'][mask],'unchanged action domain')
                    checks.array(saved[name]['group_states'][...,2:],base['group_states'][...,2:],'unchanged attributes')
                items.append(core.inspect_scene(saved,row,names,checks,expected_split=split))
            cs=sorted({r['component_id'] for r in rows});big=summary['largest_component_id']
            core.inspect_summary(summary['all_components'],list(range(len(rows))),items,rows,names,checks,'full')
            core.inspect_summary(summary['without_largest_component'],[i for i,r in enumerate(rows) if r['component_id']!=big],items,rows,names,checks,'drop')
            for c in cs:core.inspect_summary(summary['by_component'][c],[i for i,r in enumerate(rows) if r['component_id']==c],items,rows,names,checks,'component')
            for pair,expected in summary['primary_bootstrap'].items():
                right,left=pair.split('_vs_');totals=[];sizes=[]
                for c in cs:
                    d=np.concatenate([items[i][right]['errors']-items[i][left]['errors'] for i,r in enumerate(rows) if r['component_id']==c]);totals.append(d.sum());sizes.append(len(d))
                totals,sizes=np.asarray(totals),np.asarray(sizes);draw=np.random.default_rng(20).integers(0,len(cs),(2000,len(cs)))
                interval=np.quantile(totals[draw].sum(1)/sizes[draw].sum(1),[.025,.975])
                checks.value(float(totals.sum()/sizes.sum()),expected['point_estimate_m'],'delta','primary')
                for a,b in zip(interval,expected['percentile95_m']):checks.value(float(a),b,'bootstrap','primary')
                d=np.concatenate([it[right]['errors']-it[left]['errors'] for it in items]);harm=summary['all_components']['pairwise'][pair]['actual_harm']
                checks.value(int((d>1000).sum()),harm['new_harm_over_1km'],'harm_count','primary')
                checks.value(float(max(0.,d.max())),harm['max_error_increase_m'],'harm_max','primary')
            result['splits'][split]=dict(scenes=len(ids),methods=len(names),points=sum(len(it[names[0]]['errors']) for it in items))
        result.update(status='COMPLETED',exit_code=0,scope='saved native reference/population/all main position/tails/components/primary bootstrap+harm/action invariants; not new features/fitted heads/selection diagnostics/physical truth')
    except BaseException as e:
        result.update(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-t,checks=vars(checks),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        destination=out if result['status']=='COMPLETED' else out.with_suffix('.failed.json');destination.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.run.resolve(),a.output.resolve())
