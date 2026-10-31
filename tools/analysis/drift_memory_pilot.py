"""189 T qualification / 191 fixed V validation of the same mean-velocity control."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.evaluation.native_motion_interventions import mean_velocity_control
from association.evaluation.native_compatibility import distance
from tools.analysis import association_context_pilot as p
from tools.analysis.history_tail_qualification import describe

ARMS={front:(front,short+'O')+tuple(short+k for k in ('PH','PC','PD','PG','MC')) for front,short in [('HARD','H'),('STRUCT','S')]}
METHODS=tuple(k+'_'+s for values in ARMS.values() for k in values for s in ('MEAN','NN'))
PRIMARY=tuple(key for values in ARMS.values() for key in [values[5]+'_NN_vs_'+values[1]+'_NN']+[values[6]+'_NN_vs_'+values[i]+'_NN' for i in (1,2,3,4,5)])
FIELDS=('absolute_endpoint_dt_seconds','group_nodes','group_sources','selected_nodes_count','mean_speed_mps','report_spread_mps','mean_secant_gap_mps','mean_ols_gap_mps','endpoint_mean_gap_mps','native_count','span_seconds','fit_rmse_m','memory_position_difference_m')

def main_rows(m,ev,groups,raw,diag,front):
    short='H' if front=='HARD' else 'S';registry={str(t):i for i,t in enumerate(ev['target_ids'])};claimed=set();rows=[];lookup={(r['group_index'],r['query_index']):r for r in diag['decisions']}
    for gid,nodes in enumerate(groups):
        nodes=np.asarray(nodes,int);identity=str(ev['identities'][nodes[0]])
        if identity not in registry or identity in claimed:continue
        claimed.add(identity);ti=registry[identity];pure=len(set(ev['identities'][nodes].tolist()))==1
        errors={key:distance(raw[short+key+'_NN'][gid],ev['target_states'][ti]) for key in ('O','PG','MC')};base=distance(raw[front+'_NN'][gid],ev['target_states'][ti])
        change=distance(raw[short+'PG_NN'][gid],raw[short+'MC_NN'][gid])
        for j in np.flatnonzero(ev['target_valid'][ti,:,:2].all(-1)):
            if (gid,int(j)) not in lookup:continue
            record={k:v for k,v in lookup[gid,int(j)].items() if k!='selected_nodes'}
            record.update(target_index=ti,pure=pure,old_error_m=float(errors['O'][j]),PG_error_m=float(errors['PG'][j]),MC_error_m=float(errors['MC'][j]),PG_delta_m=float(errors['PG'][j]-errors['O'][j]),MC_delta_m=float(errors['MC'][j]-errors['O'][j]),MC_minus_PG_m=float(errors['MC'][j]-errors['PG'][j]),PG_base_delta_m=float(errors['PG'][j]-base[j]),MC_base_delta_m=float(errors['MC'][j]-base[j]),memory_position_difference_m=float(change[j]))
            rows.append(record)
    return rows

def condition_summary(rows):
    groups={}
    conditions={'all':lambda r:True,'pure':lambda r:r['pure'],'mixed':lambda r:not r['pure'],'before':lambda r:r['before'],'after':lambda r:not r['before'],'PG_harm':lambda r:r['PG_delta_m']>1000,'PG_rescued':lambda r:r['PG_delta_m']< -1000,'pure_PG_harm':lambda r:r['pure'] and r['PG_delta_m']>1000,'pure_PG_rescued':lambda r:r['pure'] and r['PG_delta_m']< -1000,'MC_harm':lambda r:r['MC_delta_m']>1000,'MC_rescued':lambda r:r['MC_delta_m']< -1000,'memory_changes_over_1m':lambda r:r['memory_position_difference_m']>1.}
    for name,condition in conditions.items():
        use=[r for r in rows if condition(r)];groups[name]=dict(points=len(use),pure_points=sum(r['pure'] for r in use),PG_error_m=describe([r['PG_error_m'] for r in use]),MC_error_m=describe([r['MC_error_m'] for r in use]),MC_minus_PG_m=describe([r['MC_minus_PG_m'] for r in use]),PG_delta_m=describe([r['PG_delta_m'] for r in use]),MC_delta_m=describe([r['MC_delta_m'] for r in use]),PG_harms_retained_by_MC=sum(r['PG_delta_m']>1000 and r['MC_delta_m']>1000 for r in use),**{key:describe([r[key] for r in use]) for key in FIELDS})
    return groups

def run(a,*,split='T_core',number=189):
    if (split,number) not in (('T_core',189),('V_select',191)):raise ValueError('explicit scientific split/number')
    old=p.common.old.frozen;parent=old.read(a.parent_run/'run.json');previous=old.read(a.persistence_run/'run.json');assert parent['status']==previous['status']=='COMPLETED' and previous['parent_run']==str(a.parent_run) and previous['training_scene_ids']==parent['training_scene_ids'] and len(parent['training_scene_ids'])==32
    ids=parent['training_scene_ids'] if split=='T_core' else parent['scene_ids'];assert len(ids)==32
    qualification=None
    if split=='V_select':
        qualification=old.read(a.qualification_run/'run.json');audit=old.read(a.qualification_run/'independent-output-audit.json');assert qualification['status']==audit['status']=='COMPLETED' and qualification['evaluation_splits']==['T_core'] and qualification['completed_T_scenes']==32 and qualification['V_scenes']==0
    out=a.output;out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic()
    record=dict(status='RUNNING',idea_id='IDEA-DRIFT-MEMORY-ABLATION-189' if number==189 else 'IDEA-MEAN-VELOCITY-VALIDATION-191',plan_version='1.0',mode='native_drift_memory_ablation',parent_run=str(a.parent_run),persistence_run=str(a.persistence_run),qualification_run=str(a.qualification_run) if qualification else None,native_run=parent['native_run'],manifest=parent['manifest'],methods=METHODS,training_scene_ids=ids if split=='T_core' else [],scene_ids=ids if split=='V_select' else [],evaluation_splits=[split],V_scenes=len(ids) if split=='V_select' else 0,optimizer_updates=0,neural_head_fits=0,analytic_head_fits=0,native_likelihood_fits=0,native_mean_computations=0,teacher_forward_calls=0,reference_used_for_prediction=False,test_opened=False,all_rules_frozen_before_V=True,started_at=datetime.now(timezone.utc).isoformat())
    def check(**kw):
        record.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=old.peak_rss_bytes());old.write(out/'run.json',record)
        if record['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        rows=[];arrays=[];profiles={front:[] for front in ARMS}
        for sid in ids:
            native=Path(parent['native_run'])/'private';m=p.common.load_npz(native/'native-motion'/split/f'scene-{sid}.npz');events=p.common.load_npz(native/'native-events'/split/f'scene-{sid}.npz');ev=p.common.load_npz(native/'native-evaluation'/split/f'scene-{sid}.npz');q=p.common.load_npz(a.parent_run/'private'/f'partitions-{split}-{sid}.npz');op=p.common.load_npz(a.parent_run/'private'/f'policy-{split}-{sid}.npz');raw=p.common.load_npz(a.persistence_run/'private'/f'raw-{split}-{sid}.npz');p.common.old._same(m['node_ids'],events['node_ids'],'native node binding');p.common.old._same(m['query_seconds'],events['query_seconds'],'native query binding');parts=dict(pairs=q['pairs'],scores=q['scores']);groups={};policy={};diagnostics={}
            for front,values in ARMS.items():
                g=p.common.old.groups_from_membership(q[front],len(m['states']));candidate,flags,diag=mean_velocity_control(m,events,g,op[front+'__count'],op[values[1]+'__weights'],raw[values[1]+'_NN']);raw[values[-1]+'_NN']=candidate;raw[values[-1]+'_MEAN']=raw[front+'_MEAN'];record['native_mean_computations']+=diag['native_mean_computations'];diagnostics[front]=diag;policy[front+'__eligible']=flags['eligible']
                for prefix in values:groups[prefix]=g;parts[prefix]=q[front];parts[prefix+'_scores']=q[front+'_scores']
                conditions=main_rows(m,ev,g,raw,diag,front)
                for condition in conditions:condition['scene_id']=sid
                profiles[front].extend(conditions);old.write(out/'private'/f'conditions-{front}-{sid}.json.gz',conditions)
            np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw);np.savez_compressed(out/'private'/f'partitions-{split}-{sid}.npz',**parts);np.savez_compressed(out/'private'/f'policy-{split}-{sid}.npz',**policy);old.write(out/'private'/f'diagnostics-{split}-{sid}.json.gz',diagnostics)
            row,arr,artifact=p.evaluate(m,ev,groups,{},raw,split,method_names=METHODS);original=old.read(a.persistence_run/'scenes'/f'{split}-{sid}.json.gz');relation={}
            for front,values in ARMS.items():
                for prefix in values:
                    for kind in ('raw','decoded'):relation[prefix+'_'+kind]=original['association'][front+'_'+kind]
            row.update(scene_id=sid,component_id=str(old.read(Path(parent['manifest']))['components'][sid]),search={},association=relation);p.common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',artifact);old.write(out/'scenes'/f'{split}-{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
        result=p.finish(rows,arrays,METHODS,PRIMARY);result.update(status='COMPLETED',idea_id=record['idea_id']);old.write(out/('T-summary.json' if split=='T_core' else 'summary.json'),result);old.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        public={front:condition_summary(values) for front,values in profiles.items()}
        for front,expected_count,expected_pure_harms in ([('HARD',451,34),('STRUCT',344,1)] if split=='T_core' else [('HARD',784,11),('STRUCT',655,0)]):
            assert public[front]['all']['points']==expected_count and public[front]['pure_PG_harm']['points']==expected_pure_harms
            short='H' if front=='HARD' else 'S';assert public[front]['PG_harm']['points']==result['all_components']['pairwise'][short+'PG_NN_vs_'+short+'O_NN']['actual_harm']['new_harm_over_1km']
        old.write(out/'condition-profiles.json',dict(status='COMPLETED',split=split,heads=public,scope='declared-split reference-aligned eligible main points; visible native conditions with reference error cohorts, no deployed gate or causal diagnosis'))
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:check(status='FAILED',exit_code=1,error=f'{type(error).__name__}: {error}');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-run','persistence-run','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
