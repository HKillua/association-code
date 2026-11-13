"""151: bounded-memory fullT2100/V300 physical cache for common peer training.

States in this NEW cache are m/s. Original127/056 and archives stay untouched.
T references are explicitly read; V reference is copied from frozen056 only.
"""
from __future__ import annotations
import argparse,json,sys,time,resource
from contextlib import closing
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.data.archive import MTADArchive
from association.data.legacy_dataset import validate_archive
from association.data.native import _record
from association.data.motion_evidence import DATA,scene_motion_evidence
from association.data.timed import prepare_timed_scene
from association.data.reference_states import iter_reference_states
from association.data.targets import build_scene_fusion_targets
from association.models.fusion_peer_inputs import sensor_segments
from tools.analysis import native_cv_state_control as old

IDEA='IDEA-FUSION-PEER-FULL-DATA-20261030'

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def train_groups(motion,ev):
    registry={str(v):i for i,v in enumerate(ev['target_ids'])};groups=[]
    for identity in dict.fromkeys(ev['identities'].tolist()):
        members=np.flatnonzero(ev['identities']==identity);segments=sensor_segments(motion,members)
        if not segments or str(identity) not in registry:continue
        target=registry[str(identity)];valid=ev['target_valid'][target]
        kept=[s['query_indices'].tolist() for s in segments if valid[s['query_indices'],:2].all(-1).any()]
        if kept:groups.append(dict(members=members.tolist(),target_index=target,segments=kept))
    return groups

def run(a):
    output=a.output.resolve();output.mkdir(exist_ok=False);(output/'private').mkdir()
    started=time.monotonic();state=dict(status='RUNNING',idea_id=IDEA,plan_version='1.0',started_at=datetime.now(timezone.utc).isoformat(),
        completed_scenes=0,completed_T_scenes=0,completed_V_scenes=0,test_opened=False,optimizer_updates=0,
        state_speed_unit='mps',measurement_speed_unit='kn',reference_speed_unit='mps',reference_source='T raw declared reference; V frozen056',
        native_run=str(a.native_run.resolve()),source_system=str(a.source_system.resolve()),manifest=str(a.manifest.resolve()),
        archive=[str(Path(p).resolve()) for p in a.archive],alignment=str(a.alignment.resolve()))
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=old.old.frozen.peak_rss_bytes());old.old.frozen.write(output/'run.json',state)
        if state['peak_rss_bytes']>6*1024**3:raise MemoryError('6GiB resident budget')
    try:
        manifest=old.old.frozen.read(a.manifest);alignment=old.old.frozen.read(a.alignment)
        previous=old.old.frozen.read(a.native_run/'native-cache.json')
        prior_run=old.old.frozen.read(a.native_run/'run.json')
        if (previous['status']!='CACHE_COMPLETED' or previous['native_motion_policy']!='canonical_all'
                or prior_run.get('status')!='COMPLETED' or prior_run.get('exit_code')!=0):raise ValueError('completed original127 cache required')
        index={};rows=[];check(phase='cache')
        with MTADArchive(a.archive) as archive:
            validate_archive(archive,manifest)
            for split in ('T_core','V_select'):
                ids=manifest['splits'][split];index[split]={}
                for d in ('motion','evaluation'):(output/'private'/d/split).mkdir(parents=True)
                new_t=[s for s in ids if s not in previous['scene_ids'].get(split,[])] if split=='T_core' else []
                with closing(iter_reference_states(a.archive,manifest,split='T_core',scene_ids=new_t)) as refs:
                    for sid in ids:
                        borrowed=sid in previous['scene_ids'].get(split,[])
                        if borrowed:
                            motion=load(a.native_run/'private/native-motion'/split/f'scene-{sid}.npz')
                            ev=load(a.native_run/'private/native-evaluation'/split/f'scene-{sid}.npz')
                        if not borrowed or split=='V_select':
                            native=_record(archive.load_scene(sid),manifest['components'][sid],split,128,split_unit=manifest.get('split_unit'))
                            labels=archive.read_labels(sid);target_ids=list(dict.fromkeys('-'.join(r['mmsi'].split('-')[:2]) for r in labels))
                            query=None
                            if split=='V_select':
                                physical=load(a.source_system/'predictions'/f'states-{sid}.npz');query=physical['query_seconds']
                            elif not native['tracks']:
                                query=np.linspace(min(r['t_s'] for r in labels),max(r['t_e'] for r in labels),64)
                            scene=prepare_timed_scene(native,**DATA,query_times=query)
                            rebuilt=scene_motion_evidence(scene,motion_observation_policy='canonical_all')
                            if borrowed:
                                for k in ('states','support','nearest_seconds','sources','query_seconds'):old.old._same(rebuilt[k],motion[k],'127 '+k)
                            motion=dict(rebuilt,node_ids=np.asarray(scene['node_ids'],str))
                            if split=='T_core':
                                reference=next(refs)
                                if reference['scene_id']!=sid or alignment[sid]['max_gap_seconds']!=600.:raise ValueError('T reference/alignment differs')
                                targets=build_scene_fusion_targets(scene,reference,target_identity_ids=target_ids,longitude_policy='shortest_arc',**alignment[sid])
                                ev=dict(node_ids=motion['node_ids'],identities=np.asarray(scene['identities'],str),target_ids=np.asarray(target_ids,str),query_seconds=motion['query_seconds'],
                                    target_states=np.asarray([targets[t]['states'] for t in target_ids],float).reshape(-1,64,4),
                                    target_valid=np.asarray([targets[t]['valid'] for t in target_ids],bool).reshape(-1,64,4))
                            else:
                                for k,other in [('states','native_states'),('support','support'),('nearest_seconds','nearest_seconds'),('node_ids','node_ids')]:old.old._same(motion[k],physical[other],'056 '+k)
                                original=load(a.source_system/'system-predictions'/f'states-{sid}.npz');nt=len(physical['target_identity_ids'])
                                ev=dict(node_ids=motion['node_ids'],identities=physical['identities'],target_ids=physical['target_identity_ids'],query_seconds=motion['query_seconds'],
                                    target_states=original['learned_assoc_mean_supported_aligned_truth'][:nt],target_valid=original['learned_assoc_mean_supported_aligned_truth_valid'][:nt])
                                cap=scene_motion_evidence(scene,motion_observation_policy='selected_cap128')
                                for k in ('states','support','nearest_seconds'):motion['cap128_'+k]=cap[k]
                                motion['cap128_states'][...,2]*=1852/3600
                        motion['states']=motion['states'].copy();motion['states'][...,2]*=1852/3600
                        motion['state_speed_unit']=np.asarray('mps')
                        old.old._same(motion['node_ids'],ev['node_ids'],'motion/reference node order')
                        old.old._same(motion['query_seconds'],ev['query_seconds'],'motion/reference query')
                        mp=output/'private/motion'/split/f'scene-{sid}.npz';ep=output/'private/evaluation'/split/f'scene-{sid}.npz'
                        np.savez_compressed(mp,**motion);np.savez_compressed(ep,**ev)
                        entry=dict(motion=str(mp),evaluation=str(ep),component_id=str(manifest['components'][sid]),node_count=len(motion['states']),target_count=len(ev['target_ids']))
                        if split=='T_core':entry['groups']=train_groups(motion,ev)
                        index[split][sid]=entry
                        rows.append(dict(scene_id=sid,split=split,node_count=entry['node_count'],target_count=entry['target_count'],
                            trainable_groups=len(entry.get('groups',[])),reused127=borrowed,reference_valid_position_points=int(ev['target_valid'][...,:2].all(-1).sum())))
                        archive._label_cache.pop(sid,None)
                        check(completed_scenes=len(rows),**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(index[split])})
        old.old.frozen.write(output/'private/index.json',index)
        old.old.frozen.write(output/'scene-summary.json.gz',rows)
        old.old.frozen.write(output/'summary.json',dict(status='COMPLETED',idea_id=IDEA,scene_counts={s:len(v) for s,v in index.items()},
            trainable_groups=sum(len(v['groups']) for v in index['T_core'].values()),T_scenes_without_trainable_group=sum(not v['groups'] for v in index['T_core'].values()),
            state_speed_unit='mps',reference_speed_unit='mps',V_cap128_matched_controls=True,optimizer_updates=0,test_opened=False))
        check(status='COMPLETED',exit_code=0,phase='completed',finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:
        check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','source-system','manifest','alignment','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--archive',nargs='+',required=True);p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
