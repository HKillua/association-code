"""157: frozen155 head, full147 inputs and151 frozen-reference V300, no fitting."""
from __future__ import annotations
import argparse,sys,time,resource
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.evaluation import source_risk as risk
from association.evaluation.native_temporal import predict
from tools.analysis import observation_common as common
from tools.analysis.temporal_source_pilot import summarize,add_harm,selection,BASE
from tools.analysis.source_coverage_pilot import role_diagnostic,NEW
from tools.analysis import source_risk_pilot as source

METHODS=(BASE,'WS_MEAN','WS_NEAREST_SUPPORTED',NEW)

def population(a):
    data=common.old.frozen.read(a.data_run/'run.json');head=common.old.frozen.read(a.head_run/'run.json');baseline=common.old.frozen.read(a.baseline_run/'run.json')
    for z in (data,head,baseline):
        if z['status']!='COMPLETED' or z['exit_code']!=0:raise ValueError('completed inputs required')
    if data['completed_T_scenes']!=2100 or data['completed_V_scenes']!=300 or head['analytic_head_fits']!=13:raise ValueError('published input scope')
    index=common.old.frozen.read(a.data_run/'private/index.json')['V_select']
    if len(index)!=300 or set(index)!=set(baseline['scene_ids']):raise ValueError('full300 registry')
    return index,head['scene_ids']

def run(a):
    out=a.output.resolve();out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-T128-FROZEN-FULL-SCOPE-20261030',plan_version='1.0',mode='temporal',methods=METHODS,
        started_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,analytic_head_fits=0,teacher_forward_calls=0,test_opened=False,raw_reference_read=False,
        native_run=str(a.data_run),data_run=str(a.data_run),head_run=str(a.head_run),baseline_run=str(a.baseline_run),evaluation_splits=['V_select'],training_scene_ids=[],
        native_evaluation_subdir='evaluation',native_motion_subdir='motion',invariant_methods=[NEW])
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>6*1024**3:raise MemoryError('6GiB RSS')
    try:
        if sys.platform.startswith('linux'):
            soft,hard=resource.getrlimit(resource.RLIMIT_AS);limit=min([12*1024**3]+[x for x in (soft,hard) if x!=resource.RLIM_INFINITY]);resource.setrlimit(resource.RLIMIT_AS,(limit,hard))
        index,original32=population(a);saved=common.load_npz(a.head_run/'private/models.npz');model={k:saved['full__'+k] for k in ('mean','std','beta')};check(scene_ids=list(index),original32_ids=original32)
        rows=[];arrays=[]
        for sid,entry in index.items():
            m=common.load_npz(a.baseline_run/'private/native-motion'/f'scene-{sid}.npz');m['states']=m['states'].copy();m['states'][...,2]*=1852/3600
            pack=common.load_npz(a.baseline_run/'private'/f'policy-{sid}.npz');membership=pack['membership'];groups=common.old.groups_from_membership(membership,len(m['states']))
            for key in ('node_ids','query_seconds'):common.old._same(pack[key],m[key],'147 binding '+key)
            raw={}
            with np.load(a.baseline_run/'private'/f'states-{sid}.npz',allow_pickle=False) as z:
                for name in METHODS[:-1]:
                    if str(z[name+'_state_speed_unit'])!='mps':raise ValueError('147 output mps')
                    common.old._same(z[name+'_node_group_indices'],membership,'147 grouping');raw[name]=z[name+'_group_states'].copy()
            scores=predict(model,pack['features'].reshape(-1,10)).reshape(pack['available'].shape)
            raw[NEW]=risk.recover_source_policy(states=m['states'],support=m['support'],groups=groups,pack=pack,scores=scores,kind='hard',mean_states=raw['WS_MEAN'])['group_states']
            if sid in original32:
                previous=common.load_npz(a.head_run/'private'/f'raw-V_select-{sid}.npz');common.old._same(raw[NEW][...,:2],previous[NEW][...,:2],'155 original32 new positions')
            np.savez_compressed(out/'private'/f'raw-V_select-{sid}.npz',**raw);np.savez_compressed(out/'private'/f'policy-V_select-{sid}.npz',scores=scores)
            ev=common.load_npz(Path(entry['evaluation']))
            for key in ('node_ids','query_seconds'):common.old._same(ev[key],m[key],'151 reference binding '+key)
            row,arr,artifacts=common.evaluate_predictions(m,groups,pack,ev,raw,{},method_names=METHODS,split='V_select',prediction_speed_unit='mps');add_harm(row,arr);labels=source.training_labels(pack,groups,ev)
            row.update(scene_id=sid,component_id=entry['component_id'],stratum='original32' if sid in original32 else 'remaining268',selection={n:selection(pack,labels,s,pack['scores']) for n,s in [(BASE,pack['scores']),(NEW,scores)]},role_diagnostic={n:role_diagnostic(pack,labels,s) for n,s in [(BASE,pack['scores']),(NEW,scores)]})
            common.save_artifacts(out/'private'/f'states-V_select-{sid}.npz',artifacts);common.old.frozen.write(out/'scenes'/f'{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(completed_V_scenes=len(rows))
        result=summarize(rows,arrays,METHODS)
        for name in METHODS[1:-1]:
            key=NEW+'_vs_'+name;result['primary_bootstrap'][key]=common.component_bootstrap(rows,arrays,key)
        result['subsets']={}
        for label in ('original32','remaining268'):
            choose=[i for i,r in enumerate(rows) if r['stratum']==label];result['subsets'][label]=summarize([rows[i] for i in choose],[arrays[i] for i in choose],METHODS)
        result.update(status='COMPLETED',idea_id=state['idea_id'],analytic_head_fits=0,selection={n:{k:sum(r['selection'][n][k] for r in rows) for k in rows[0]['selection'][n]} for n in (BASE,NEW)})
        common.old.frozen.write(out/'summary.json',result);common.old.frozen.write(out/'scene-summary.json.gz',rows);check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data-run','head-run','baseline-run','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
