"""186: T-only production list-group interface and label-free arithmetic audit."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.evaluation import native_motion_interventions as ni
from tools.analysis.native_motion_pilot import design,p
from tools.analysis.native_motion_closeout import reconstruct

def run(a):
    parent=p.common.old.frozen.read(a.parent_run/'run.json');out=a.output;out.mkdir(exist_ok=False);(out/'private').mkdir();start=time.monotonic();state=dict(status='RUNNING',idea_id='IDEA-NATIVE-LIST-INTERFACE-QUALIFICATION-186',plan_version='1.0',parent_run=str(a.parent_run),bridge_run=str(a.bridge_run),V_scenes=0,optimizer_updates=0,teacher_forward_calls=0,new_deployed_predictions=0,reference_read=False,native_likelihood_fits=0,completed_T_scenes=0,started_at=datetime.now(timezone.utc).isoformat())
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=p.common.old.frozen.peak_rss_bytes());p.common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>4*1024**3:raise MemoryError('RSS4GiB')
    try:
        rows=[]
        for sid in parent['training_scene_ids']:
            n=Path(parent['native_run'])/'private';m=p.common.load_npz(n/'native-motion/T_core'/f'scene-{sid}.npz');events=p.common.load_npz(n/'native-events/T_core'/f'scene-{sid}.npz');q=p.common.load_npz(a.parent_run/'private'/f'partitions-T_core-{sid}.npz');old=p.common.load_npz(a.parent_run/'private'/f'raw-T_core-{sid}.npz');op=p.common.load_npz(a.parent_run/'private'/f'policy-T_core-{sid}.npz');b=p.common.load_npz(a.bridge_run/'private'/f'raw-T_core-{sid}.npz');diagnostics={};public={}
            for front,short in [('HARD','H'),('STRUCT','S')]:
                groups=p.common.old.groups_from_membership(q[front],len(m['states']));assert all(isinstance(g,list) for g in groups);count=op[front+'__count'];base=old[short+'O_NN'];public[front]={}
                for kind in ('bridge','persistence'):
                    values=design(kind)[0][front];raw={};policy={}
                    if kind=='bridge':
                        outputs,flags,diag=ni.bridge(m,events,groups,count,op[front+'__scores'],base,b[short+'B_NN']);raw[values[2]+'_NN']=b[short+'B_NN'];raw[values[3]+'_NN']=b[short+'E_NN']
                        for key,name in zip(values[4:],('CL','CH','CM')):raw[key+'_NN']=outputs[name]
                    else:
                        outputs,flags,diag=ni.persistence(m,events,groups,count,op[short+'O__weights'],base);diagnostics[front]=diag;p.common.old.frozen.write(out/'private'/f'diagnostics-T_core-{sid}.json.gz',diagnostics);state['native_likelihood_fits']+=diag['native_likelihood_fits']
                        for key,name in zip(values[2:],('PH','PC','PD','PG')):raw[key+'_NN']=outputs[name]
                    policy[front+'__eligible']=flags['eligible'];r=dict(kind=kind,manifest=parent['manifest'],bridge_run=str(a.bridge_run));eligible,gap=reconstruct(out,r,'T_core',sid,front,m,events,[np.array(g,int) for g in groups],count,op,old,raw,policy);assert np.array_equal(eligible,flags['eligible'])
                    for key in values[2:]:assert np.array_equal(raw[key+'_NN'][...,2:],base[...,2:])
                    np.savez_compressed(out/'private'/f'{kind}-{front}-{sid}.npz',**raw,eligible=eligible);public[front][kind]=dict(eligible_queries=int(eligible.sum()),numpy_gap=gap,native_likelihood_fits=diag.get('native_likelihood_fits',0))
            rows.append(dict(scene_id=sid,heads=public));check(completed_T_scenes=len(rows))
        p.common.old.frozen.write(out/'summary.json',dict(status='COMPLETED',scenes=rows,arithmetic_checks=32*2*2,scope='T-only fixed formula/interface qualification, no reference/effect evaluation or V; failed184/185 kept'))
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('parent-run','bridge-run','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
