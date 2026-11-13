"""171: T-only saved object loss and exact full-label forest qualification."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.training.structured_forest import forest_contrast
from tools.analysis import observation_common as c


def point_coverage(a,prefix,target_ids):
    q=a[prefix+'_aligned_prediction_valid'][...,:2].all(-1)&a[prefix+'_aligned_truth_valid'][...,:2].all(-1)
    result=np.zeros((len(target_ids),q.shape[1]),bool);index={str(t):i for i,t in enumerate(target_ids)}
    for i,t in enumerate(a[prefix+'_aligned_target_ids']):
        if str(t) not in index:
            # Unmatched output rows have no reference: they carry no eligible
            # target-query. Keep them in the original group metrics, not here.
            if q[i].any():raise ValueError('valid aligned target absent from central population')
            continue
        result[index[str(t)]]|=q[i]
    return result


def object_rows(identities,target_ids,old,new,corrected_edges,pairs,coverage_old,coverage_new):
    identities=np.asarray(identities);n=len(identities)
    def anchors(membership):
        return {str(identities[np.flatnonzero(membership==g)[0]]) for g in np.unique(membership)}
    ao,an=anchors(old),anchors(new);rows=[]
    for t,identity in enumerate(target_ids):
        mask=identities==identity;size=int(mask.sum());incident=mask[pairs].any(1)
        members=np.flatnonzero(mask)
        rows.append(dict(object_index=t,fragments=size,positive_pairs=size*(size-1)//2,incident_pairs=int(incident.sum()),
            represented_before=str(identity) in ao,represented_after=str(identity) in an,
            lost=str(identity) in ao and str(identity) not in an,rescued=str(identity) not in ao and str(identity) in an,
            mixed_new_groups=sum(len(np.unique(identities[new==g]))>1 for g in np.unique(new[members])),
            forest_false_edges_incident=int((corrected_edges&incident).sum()),
            new_points=int((coverage_new[t]&~coverage_old[t]).sum()),lost_points=int((coverage_old[t]&~coverage_new[t]).sum())))
    return rows


def run(a):
    out=a.output.resolve();out.mkdir(exist_ok=False);(out/'scenes').mkdir();start=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-OBJECT-FOREST-QUALIFICATION-20261031-171' if a.run_id==171 else 'IDEA-OBJECT-FOREST-QUALIFICATION-REPAIR-20261031-172',plan_version='1.0' if a.run_id==171 else '1.1',
        optimizer_updates=0,V_opened=False,test_opened=False,started_at=datetime.now(timezone.utc).isoformat(),
        native_run=str(a.native_run),calibration_run=str(a.calibration_run),manifest=str(a.manifest))
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=c.old.frozen.peak_rss_bytes());c.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>4*1024**3:raise MemoryError('RSS4GiB')
    try:
        parent=c.old.frozen.read(a.calibration_run/'run.json');manifest=c.old.frozen.read(a.manifest)
        tids=parent['training_scene_ids']
        if parent['status']!='COMPLETED' or tids!=manifest['splits']['T_core'][:32]:raise ValueError('original completed170 T32 required')
        check(training_scene_ids=tids);rng=np.random.default_rng(20);allrows=[];scenes=[]
        for sid in tids:
            ev=c.load_npz(a.native_run/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
            motion=c.load_npz(a.native_run/'private/native-motion/T_core'/f'scene-{sid}.npz',('node_ids',))
            c.old._same(motion['node_ids'],ev['node_ids'],'original T nodes')
            q=c.load_npz(a.calibration_run/'private'/f'partitions-T_core-{sid}.npz');pairs=q['pairs'];y=ev['identities'];s=q['CTXCAL_scores']
            noise=rng.normal(0.,.1,(4,len(pairs)));co,offset=forest_contrast(s,pairs,y,noise);gap=float(co@s+offset)
            false=y[pairs[:,0]]!=y[pairs[:,1]];bad=(co>0)&false
            art=c.load_npz(a.calibration_run/'private'/f'states-T_core-{sid}.npz')
            target=ev['target_ids'];old=point_coverage(art,'WS_NN',target);new=point_coverage(art,'CTXCAL_NN',target)
            rows=object_rows(y,target,q['WS'],q['CTXCAL'],bad,pairs,old,new)
            record=dict(scene_id=sid,component_id=str(manifest['components'][sid]),nodes=len(y),classes=len(np.unique(y)),rank=len(y)-len(np.unique(y)),
                pairs=len(pairs),loss=gap,false_correctable_edges=int(bad.sum()),objects=rows)
            c.old.frozen.write(out/'scenes'/f'T_core-{sid}.json',record);scenes.append(record);allrows.extend(rows);check(completed_T_scenes=len(scenes))
        summary={}
        for name,subset in [('all',allrows),('lost',[r for r in allrows if r['lost']]),('rescued',[r for r in allrows if r['rescued']]),('retained',[r for r in allrows if r['represented_before'] and r['represented_after']])]:
            summary[name]=dict(objects=len(subset),single_fragment=sum(r['fragments']==1 for r in subset),
                fragments=[r['fragments'] for r in subset],positive_pair_mass=sum(r['positive_pairs'] for r in subset),
                touched_by_false_forest=sum(r['forest_false_edges_incident']>0 for r in subset),
                new_points=sum(r['new_points'] for r in subset),lost_points=sum(r['lost_points'] for r in subset))
        qualified=summary['lost']['touched_by_false_forest']>0 and any(s['loss']>1e-12 for s in scenes)
        c.old.frozen.write(out/'summary.json',dict(status='COMPLETED',T_only=True,qualified=qualified,diagnostic_only=True,
            objects=summary,scene_losses=[dict(scene_id=s['scene_id'],loss=s['loss'],false_correctable_edges=s['false_correctable_edges']) for s in scenes],
            interpretation='gradient touches some T lost-object bridges; not causal proof or deployment improvement'))
        check(status='COMPLETED',exit_code=0,qualified=qualified,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','calibration-run','manifest','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--run-id',type=int,choices=(171,172),default=171)
    p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
