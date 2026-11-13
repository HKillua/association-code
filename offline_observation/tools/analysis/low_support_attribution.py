"""177 T-only saved-output attribution; every oracle is diagnostic, never deployed."""
from __future__ import annotations
import argparse,sys,time
from datetime import datetime,timezone
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis import association_context_pilot as p
from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation import source_risk as risk
from tools.analysis.audit_shared_control_arrays import position
PREFIXES=('WS','STRUCT','HARD','DEC')
COMPARISONS=(('WS','HARD'),('STRUCT','HARD'),('HARD','DEC'))


def arm_details(m,ev,a):
    groups=[a['group_node_indices'][b:e].tolist() for b,e in zip(a['group_offsets'][:-1],a['group_offsets'][1:])]
    nt=len(ev['target_ids']);q=m['support'].shape[1]
    p.common.old._same(a['aligned_target_ids'][:nt],ev['target_ids'],'central reference row order')
    valid=a['aligned_prediction_valid'][:nt,:,:2].all(-1)&ev['target_valid'][...,:2].all(-1)
    error=position(a['aligned_prediction'][:nt],ev['target_states'],valid)
    source_count=np.zeros((nt,q),int);fragments=np.zeros((nt,q),int);pure=np.zeros(nt,bool);anchors=np.full(nt,-1,int)
    fragment_best=np.full((nt,q),np.inf);supported_best=np.full((nt,q),np.inf);compatible_best=np.full((nt,q),np.inf);compatible_count=np.zeros(nt,int)
    states=a['group_states'];ids=ev['identities']
    for t,identity in enumerate(ev['target_ids']):
        g=int(a['aligned_group_indices'][t])
        if g<0:continue
        members=np.asarray(groups[g],int);anchors[t]=members[0]
        if str(ids[members[0]])!=str(identity):raise ValueError('declared first-anchor correspondence differs')
        pure[t]=len(set(str(x) for x in ids[members]))==1
        fragments[t]=m['support'][members].sum(0)
        source_count[t]=sum(m['support'][members[m['sources'][members]==s]].any(0) for s in np.unique(m['sources'][members]))
        truth=np.broadcast_to(ev['target_states'][t],m['states'][members].shape);mask=np.broadcast_to(ev['target_valid'][t,:,:2].all(-1),(len(members),q))
        distance=position(m['states'][members],truth,mask)
        fragment_best[t]=distance.min(0);supported_best[t]=np.where(m['support'][members],distance,np.inf).min(0)
        eligible=[i for i,group in enumerate(groups) if str(ids[group[0]])==str(identity)];compatible_count[t]=len(eligible)
        compatible_best[t]=position(states[eligible],np.broadcast_to(ev['target_states'][t],states[eligible].shape),np.broadcast_to(ev['target_valid'][t,:,:2].all(-1),(len(eligible),q))).min(0)
        if np.any(compatible_best[t,valid[t]]>error[t,valid[t]]+1e-6):raise ValueError('central group must be included in compatible-output oracle')
    mean=ordinary_fusion(m['states'],m['nearest_seconds'],m['support'],groups)['predictions']['mean_supported'][[g[0] for g in groups]]
    pack=risk.build_source_features(**{k:m[k] for k in p.source.MOTION_KEYS},groups=groups)
    equal=risk.recover_source_policy(states=m['states'],support=m['support'],groups=groups,pack=pack,scores=np.zeros_like(pack['available'],dtype=float),kind='equal',mean_states=mean)['group_states']
    count=np.asarray([pack['available'][b:e].sum(0) for b,e in zip(pack['group_offsets'][:-1],pack['group_offsets'][1:])]);low=count<=1
    p.common.old._same(equal[low,:2],states[low,:2],'0/1 support independent of source-risk scores')
    return dict(valid=valid,error=error,source_count=source_count,fragments=fragments,pure=pure,anchor=anchors,
                fragment_best=fragment_best,supported_best=supported_best,compatible_best=compatible_best,
                compatible_count=compatible_count,invariance_queries=int(low.sum()))


def compare(old,new):
    use=old['valid']&new['valid'];delta=new['error']-old['error'];bad=use&(delta>1000)
    shape=use.shape;broadcast=lambda x:np.broadcast_to(x[:,None],shape)
    fragment_oracle=np.where(new['source_count']==1,new['supported_best'],new['fragment_best'])
    masks=dict(all=np.ones(shape,bool),new_zero=new['source_count']==0,new_one=new['source_count']==1,new_multi=new['source_count']>=2,
               mixed=broadcast(~new['pure']),pure=broadcast(new['pure']),same_anchor=broadcast(new['anchor']==old['anchor']),
               pure_zero=broadcast(new['pure'])&(new['source_count']==0),pure_one=broadcast(new['pure'])&(new['source_count']==1),
               single_supported_fragment=new['fragments']==1,multiple_supported_fragments=new['fragments']>1,
               compatible_output_alternative=broadcast(new['compatible_count']>1))
    strata={}
    for name,mask in masks.items():
        sel=use&mask;n=int(sel.sum());harms=bad&mask
        strata[name]=dict(points=n,new_harm_over_1km=int(harms.sum()),mean_delta_m=float(delta[sel].mean()) if n else None,
          fragment_oracle_removes_new_harm=int((harms&(fragment_oracle-old['error']<=1000)).sum()),
          compatible_output_oracle_removes_new_harm=int((harms&(new['compatible_best']-old['error']<=1000)).sum()))
    transitions={f'{a}_to_{b}':dict(points=int((use&(np.minimum(old['source_count'],2)==a)&(np.minimum(new['source_count'],2)==b)).sum()),
             new_harm_over_1km=int((bad&(np.minimum(old['source_count'],2)==a)&(np.minimum(new['source_count'],2)==b)).sum())) for a in range(3) for b in range(3)}
    examples=[dict(target_index=int(t),query_index=int(j),increase_m=float(delta[t,j]),old_sources=int(old['source_count'][t,j]),new_sources=int(new['source_count'][t,j]),
         candidate_group_pure=bool(new['pure'][t]),supported_fragments=int(new['fragments'][t,j]),same_anchor=bool(new['anchor'][t]==old['anchor'][t]),
         fragment_oracle_removes_new_harm=bool(fragment_oracle[t,j]-old['error'][t,j]<=1000),compatible_output_count=int(new['compatible_count'][t]),
         compatible_output_oracle_removes_new_harm=bool(new['compatible_best'][t,j]-old['error'][t,j]<=1000)) for t,j in np.argwhere(bad)]
    return dict(strata=strata,transitions=transitions,examples=examples)


def run(a):
    out=a.output.resolve();out.mkdir(exist_ok=False);(out/'scenes').mkdir();began=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-LOW-SUPPORT-ATTRIBUTION-177',plan_version='1.0',parent_run=str(a.parent_run),
               optimizer_updates=0,V_opened=False,test_opened=False,oracle_deployment=False,started_at=datetime.now(timezone.utc).isoformat())
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=p.common.old.frozen.peak_rss_bytes());p.common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>4*1024**3:raise MemoryError('RSS4GiB')
    try:
        parent=p.common.old.frozen.read(a.parent_run/'run.json');manifest=p.common.old.frozen.read(a.manifest);ids=parent['training_scene_ids']
        if parent['status']!='COMPLETED' or ids!=manifest['splits']['T_core'][:32]:raise ValueError('completed175 original T32 required')
        old=p.common.old.frozen.read(a.parent_run/'T-summary.json')['all_components'];rows=[]
        for sid in ids:
            m=p.common.load_npz(Path(parent['native_run'])/'private/native-motion/T_core'/f'scene-{sid}.npz');ev=p.common.load_npz(Path(parent['native_run'])/'private/native-evaluation/T_core'/f'scene-{sid}.npz');z=p.common.load_npz(a.parent_run/'private'/f'states-T_core-{sid}.npz')
            arms={k:arm_details(m,ev,{name[len(k+'_NN_'):]:v for name,v in z.items() if name.startswith(k+'_NN_')}) for k in PREFIXES}
            row=dict(scene_id=sid,component_id=str(manifest['components'][sid]),objects=len(ev['target_ids']),invariance_queries={k:v['invariance_queries'] for k,v in arms.items()},
                     comparisons={right+'_vs_'+left:compare(arms[left],arms[right]) for left,right in COMPARISONS})
            p.common.old.frozen.write(out/'scenes'/f'T_core-{sid}.json.gz',row);rows.append(row);check(completed_T_scenes=len(rows))
        summary=dict(status='COMPLETED',T_only=True,oracle_diagnostic_only=True,objects=sum(r['objects'] for r in rows),comparisons={},invariance_queries={k:sum(r['invariance_queries'][k] for r in rows) for k in PREFIXES})
        for left,right in COMPARISONS:
            key=right+'_vs_'+left;units=[r['comparisons'][key] for r in rows];result=dict(strata={},transitions={})
            for name in units[0]['strata']:
                vs=[x['strata'][name] for x in units];n=sum(v['points'] for v in vs)
                result['strata'][name]={k:sum(v[k] for v in vs) for k in ('points','new_harm_over_1km','fragment_oracle_removes_new_harm','compatible_output_oracle_removes_new_harm')}
                result['strata'][name]['mean_delta_m']=sum(v['mean_delta_m']*v['points'] for v in vs if v['points'])/n if n else None
            result['transitions']={k:{f:sum(x['transitions'][k][f] for x in units) for f in ('points','new_harm_over_1km')} for k in units[0]['transitions']}
            result['anonymous_examples_count']=sum(len(x['examples']) for x in units);expected=old['pairwise'][right+'_NN_vs_'+left+'_NN']
            if result['strata']['all']['new_harm_over_1km']!=expected['actual_harm']['new_harm_over_1km']:raise ValueError('saved175 harm count differs')
            if abs(result['strata']['all']['mean_delta_m']-expected['strata']['all']['candidate_minus_A0']['mean_error_change'])>1e-6:raise ValueError('saved175 mean delta differs')
            summary['comparisons'][key]=result
        p.common.old.frozen.write(out/'summary.json',summary);check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    q=argparse.ArgumentParser(description=__doc__)
    for k in ('parent-run','manifest','output'):q.add_argument('--'+k,type=Path,required=True)
    q.add_argument('--time-policy',choices=('advisory',),required=True);run(q.parse_args())
