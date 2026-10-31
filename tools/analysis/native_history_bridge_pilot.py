"""181: native bracket mask, nearest endpoint and conditional-time bridge controls."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.evaluation.native_history_bridge import controls
from tools.analysis import association_context_pilot as p

ARMS={'HARD':('HARD','HO','HR','HN','HE','HB'),'STRUCT':('STRUCT','SO','SR','SN','SE','SB')}
METHODS=tuple(k+'_'+f for arms in ARMS.values() for k in arms for f in ('MEAN','NN'))
PRIMARY=tuple(k+'_NN_vs_'+arms[1]+'_NN' for arms in ARMS.values() for k in arms[2:])+tuple(arms[5]+'_NN_vs_'+arms[i]+'_NN' for arms in ARMS.values() for i in (2,3,4))

def run(a):
    parent=p.common.old.frozen.read(a.parent_run/'run.json');qualification=p.common.old.frozen.read(a.qualification_run/'run.json')
    if parent['status']!='COMPLETED' or qualification['status']!='COMPLETED':raise ValueError('completed178/179 required')
    out=a.output;out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();began=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-NATIVE-HISTORY-BRIDGE-181',plan_version='1.0',mode='native_bridge',methods=METHODS,parent_run=str(a.parent_run),qualification_run=str(a.qualification_run),
        optimizer_updates=0,neural_head_fits=0,analytic_head_fits=0,teacher_forward_calls=0,test_opened=False,reference_used_for_prediction=False,started_at=datetime.now(timezone.utc).isoformat(),
        **{k:parent[k] for k in ('native_run','manifest','training_scene_ids','scene_ids')})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=p.common.old.frozen.peak_rss_bytes());p.common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        manifest=p.common.old.frozen.read(Path(parent['manifest']));eligible_T=0
        for split,ids,fn in [('T_core',parent['training_scene_ids'],'T-summary.json'),('V_select',parent['scene_ids'],'summary.json')]:
            rows=[];arrays=[]
            for sid in ids:
                m=p.common.load_npz(Path(parent['native_run'])/'private/native-motion'/split/f'scene-{sid}.npz');events=p.common.load_npz(Path(parent['native_run'])/'private/native-events'/split/f'scene-{sid}.npz');q=p.common.load_npz(a.parent_run/'private'/f'partitions-{split}-{sid}.npz');old=p.common.load_npz(a.parent_run/'private'/f'raw-{split}-{sid}.npz');policy=p.common.load_npz(a.parent_run/'private'/f'policy-{split}-{sid}.npz');raw={};parts=dict(pairs=q['pairs'],scores=q['scores']);saved={};groups={};diagnostic={}
                p.common.old._same(m['node_ids'],events['node_ids'],'native motion/events nodes');p.common.old._same(m['query_seconds'],events['query_seconds'],'native query times')
                for front,arms in ARMS.items():
                    g=p.common.old.groups_from_membership(q[front],len(m['states']));values,flags,diag=controls(m,events,g,policy[front+'__count'],policy[front+'__scores'],old[arms[1]+'_NN']);diagnostic[front]=diag
                    for prefix in arms:groups[prefix]=g;parts[prefix]=q[front];parts[prefix+'_scores']=q[front+'_scores'];raw[prefix+'_MEAN']=old[front+'_MEAN']
                    raw[arms[0]+'_NN']=old[arms[0]+'_NN'];raw[arms[1]+'_NN']=old[arms[1]+'_NN']
                    for prefix,key in zip(arms[2:],('R','N','E','B')):raw[prefix+'_NN']=values[key]
                    saved.update({front+'__'+k:v for k,v in flags.items()})
                    if split=='T_core':eligible_T+=diag['eligible_group_queries']
                np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw);np.savez_compressed(out/'private'/f'partitions-{split}-{sid}.npz',**parts);np.savez_compressed(out/'private'/f'policy-{split}-{sid}.npz',**saved);p.common.old.frozen.write(out/'private'/f'bridge-{split}-{sid}.json',diagnostic)
                ev=p.common.evaluation(Path(parent['native_run']),split,sid,m);row,arr,art=p.evaluate(m,ev,groups,{},raw,split,method_names=METHODS)
                original=p.common.old.frozen.read(a.parent_run/'scenes'/f'{split}-{sid}.json.gz');relation={}
                for front,arms in ARMS.items():
                    for prefix in arms:
                        for kind in ('raw','decoded'):relation[prefix+'_'+kind]=original['association'][front+'_'+kind]
                public={front:{k:v for k,v in diag.items() if k!='decisions'} for front,diag in diagnostic.items()}
                for v in public.values():
                    gaps=v.pop('selected_gap_seconds');v['gap_quantiles_seconds']=np.quantile(gaps,[0,.25,.5,.75,1]).tolist() if gaps else None
                row.update(scene_id=sid,component_id=str(manifest['components'][sid]),search={},association=relation,bridge=public);p.common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art);p.common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(phase=split,**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
            result=p.finish(rows,arrays,METHODS,PRIMARY);result.update(status='COMPLETED',idea_id=state['idea_id']);p.common.old.frozen.write(out/fn,result);p.common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
            if split=='T_core':
                check(T_eligible_group_queries=eligible_T,all_rules_frozen_before_V=True)
                if not eligible_T:check(status='COMPLETED',exit_code=0,phase='NO_T_CONDITION',completed_V_scenes=0);return
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('parent-run','qualification-run','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
