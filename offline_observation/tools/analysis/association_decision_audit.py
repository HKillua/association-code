"""174: bounded T-only score/partition/object audit; oracle is diagnostic only."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis import observation_common as c
from tools.analysis.static_partition_pilot import agglomerate,THRESHOLD
from tools.analysis.object_forest_qualification import point_coverage

PREFIXES=('WS','CTX','BCE','STRUCT')

def dissect(pairs,scores,membership,identities,sources,support):
    n=len(identities);pairs=np.asarray(pairs);w=np.asarray(scores)-THRESHOLD
    y=identities[pairs[:,0]]==identities[pairs[:,1]];raw=w>=0
    decoded=membership[pairs[:,0]]==membership[pairs[:,1]]
    cross=sources[pairs[:,0]]!=sources[pairs[:,1]]
    joint=(support[pairs[:,0]]&support[pairs[:,1]]).any(1)
    counts={}
    for key,scope in [('all',np.ones(len(pairs),bool)),('cross',cross),('same_source',~cross),('cross_joint',cross&joint),('cross_no_joint',cross&~joint)]:
        counts[key]={k:int((mask&scope).sum()) for k,mask in dict(pairs=np.ones(len(pairs),bool),truth_positive=y,raw_positive=raw,decoded_positive=decoded,
            true_raw_kept=y&raw&decoded,true_raw_suppressed=y&raw&~decoded,true_raw_missed_but_grouped=y&~raw&decoded,
            true_missed_both=y&~raw&~decoded,false_raw_kept=~y&raw&decoded,false_raw_rejected=~y&raw&~decoded,
            false_raw_negative_forced=~y&~raw&decoded,false_rejected_both=~y&~raw&~decoded).items()}
    group_rows=[]
    for g in np.unique(membership):
        nodes=np.flatnonzero(membership==g);inside=(membership[pairs[:,0]]==g)&decoded
        before=float(w[inside].sum());after=float(w[inside&y].sum());delta=-float(w[inside&~y].sum())
        if not np.isclose(after-before,delta,atol=1e-9,rtol=1e-11):raise ValueError('oracle J decomposition mismatch')
        group_rows.append(dict(group=int(g),nodes=len(nodes),identities=len(np.unique(identities[nodes])),false_pairs=int((inside&~y).sum()),
            positive_false_pairs=int((inside&~y&raw).sum()),negative_false_pairs=int((inside&~y&~raw).sum()),oracle_split_J_change=delta))
    return dict(counts=counts,groups=group_rows,J=float(w[decoded].sum()),identity_partition_J=float(w[y].sum()),
        oracle_split_J_change=sum(g['oracle_split_J_change'] for g in group_rows))

def run(a):
    out=a.output.resolve();out.mkdir(exist_ok=False);(out/'scenes').mkdir();began=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-ASSOCIATION-DECISION-AUDIT-174',plan_version='1.0',parent_run=str(a.parent_run),
        optimizer_updates=0,V_opened=False,test_opened=False,oracle_deployment=False,started_at=datetime.now(timezone.utc).isoformat())
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=c.old.frozen.peak_rss_bytes());c.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>4*1024**3:raise MemoryError('RSS4GiB')
    try:
        parent=c.old.frozen.read(a.parent_run/'run.json');manifest=c.old.frozen.read(a.manifest);tids=parent['training_scene_ids']
        if parent['status']!='COMPLETED' or tids!=manifest['splits']['T_core'][:32]:raise ValueError('completed173 original T32 required')
        native=Path(parent['native_run']);check(training_scene_ids=tids,native_run=str(native));rows=[]
        for sid in tids:
            q=c.load_npz(a.parent_run/'private'/f'partitions-T_core-{sid}.npz');m=c.load_npz(native/'private/native-motion/T_core'/f'scene-{sid}.npz')
            ev=c.load_npz(native/'private/native-evaluation/T_core'/f'scene-{sid}.npz');art=c.load_npz(a.parent_run/'private'/f'states-T_core-{sid}.npz')
            c.old._same(m['node_ids'],ev['node_ids'],'T node binding');pairs=q['pairs'];n=len(m['node_ids']);y=ev['identities'];target=ev['target_ids']
            c.old._same(pairs,np.column_stack(np.triu_indices(n,1)),'complete original pairs')
            arms={};cover={};represented={}
            for p in PREFIXES:
                replay,_,_=agglomerate(n,pairs,q[p+'_scores'],'WS');c.old._same(replay,q[p],'saved WS replay')
                arms[p]=dissect(pairs,q[p+'_scores'],q[p],y,m['sources'],m['support'])
                cover[p]=point_coverage(art,p+'_NN',target)
                represented[p]={str(y[np.flatnonzero(q[p]==g)[0]]) for g in np.unique(q[p])}
            comparisons={}
            for old,new in [('WS','CTX'),('WS','BCE'),('WS','STRUCT'),('BCE','STRUCT')]:
                objs=[];truth=y[pairs[:,0]]==y[pairs[:,1]];cross=m['sources'][pairs[:,0]]!=m['sources'][pairs[:,1]]
                added=truth&cross&(q[old+'_scores']<THRESHOLD)&(q[new+'_scores']>=THRESHOLD)
                retained=q[new][pairs[:,0]]==q[new][pairs[:,1]]
                for t,identity in enumerate(target):
                    involved=[g for g in arms[new]['groups'] if g['group'] in q[new][y==identity]]
                    mixed=[g for g in involved if g['identities']>1]
                    objs.append(dict(object_index=t,fragments=int((y==identity).sum()),
                        lost=str(identity) in represented[old] and str(identity) not in represented[new],
                        rescued=str(identity) not in represented[old] and str(identity) in represented[new],
                        mixed_groups=len(mixed),J_improving_oracle_splits=sum(g['oracle_split_J_change']>1e-9 for g in mixed),
                        J_worsening_oracle_splits=sum(g['oracle_split_J_change'] < -1e-9 for g in mixed),
                        new_points=int((cover[new][t]&~cover[old][t]).sum()),lost_points=int((cover[old][t]&~cover[new][t]).sum())))
                comparisons[new+'_vs_'+old]=dict(objects=objs,new_raw_true_cross=int(added.sum()),new_raw_true_cross_retained=int((added&retained).sum()),new_raw_true_cross_suppressed=int((added&~retained).sum()))
            row=dict(scene_id=sid,component_id=str(manifest['components'][sid]),nodes=n,targets=len(target),arms=arms,comparisons=comparisons)
            c.old.frozen.write(out/'scenes'/f'T_core-{sid}.json',row);rows.append(row);check(completed_T_scenes=len(rows))
        summary=dict(status='COMPLETED',T_only=True,oracle_diagnostic_only=True,scenes=len(rows),objects=sum(r['targets'] for r in rows),arms={},comparisons={})
        for p in PREFIXES:
            groups=[g for r in rows for g in r['arms'][p]['groups'] if g['identities']>1]
            summary['arms'][p]=dict(counts={scope:{k:sum(r['arms'][p]['counts'][scope][k] for r in rows) for k in rows[0]['arms'][p]['counts'][scope]} for scope in rows[0]['arms'][p]['counts']},
                mixed_groups=len(groups),J_improving_oracle_splits=sum(g['oracle_split_J_change']>1e-9 for g in groups),J_worsening_oracle_splits=sum(g['oracle_split_J_change'] < -1e-9 for g in groups),
                oracle_split_J_change=sum(g['oracle_split_J_change'] for g in groups),identity_J_minus_WS=sum(r['arms'][p]['identity_partition_J']-r['arms'][p]['J'] for r in rows))
        for key in rows[0]['comparisons']:
            objects=[o for r in rows for o in r['comparisons'][key]['objects']];lost=[o for o in objects if o['lost']]
            summary['comparisons'][key]=dict(lost_objects=len(lost),rescued_objects=sum(o['rescued'] for o in objects),
                lost_with_J_improving_split=sum(o['J_improving_oracle_splits']>0 for o in lost),lost_with_J_worsening_split=sum(o['J_worsening_oracle_splits']>0 for o in lost),
                new_points=sum(o['new_points'] for o in objects),lost_points=sum(o['lost_points'] for o in objects),
                **{k:sum(r['comparisons'][key][k] for r in rows) for k in ['new_raw_true_cross','new_raw_true_cross_retained','new_raw_true_cross_suppressed']})
        c.old.frozen.write(out/'summary.json',summary);check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('parent-run','manifest','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
