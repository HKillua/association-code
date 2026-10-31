import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from tests.integration.test_source_risk_pilot import scene_arrays
from association.evaluation import native_phase_gate as phase,native_motion_interventions as ni,history_candidates as h
from association.training import temporal_gate as gate
from tools.analysis import association_context_pilot as p,temporal_gate_pilot as pilot,temporal_gate_closeout as audit
from tools.analysis.association_context_closeout import closeout

class TemporalGateIOTests(unittest.TestCase):
    def test_saved_T44_actions_metrics_and_attributes(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'193';parent=root/'178';control=root/'189';native=root/'127'
            for folder in (run,parent,control):(folder/'private').mkdir(parents=True)
            (run/'scenes').mkdir();manifest=root/'manifest.json';p.common.old.frozen.write(manifest,dict(components={'1':'1','3':'3'}));p.common.old.frozen.write(parent/'run.json',dict(status='COMPLETED'));rows=[];arrays=[]
            for sid in ('1','3'):
                m,ev,member,_=scene_arrays(sid);m['support'][:,-1]=False;m['native_count']=np.full(4,2);m['span_seconds']=np.full(4,30.);groups=[[0,1],[2],[3]];_,count=h.descriptors(m,groups);scores=np.zeros(m['support'].shape);base=np.stack([m['states'][0],m['states'][2],m['states'][3]]);outputs,weights=h.readouts(m,groups,count,scores,base);o=outputs['O'];values=np.repeat(m['states'][:,0],2,axis=0).copy();values[:,2]/=1852/3600;values[1::2,2]+=1;events=dict(event_offsets=np.arange(5)*2,event_times_seconds=np.tile([-10.,20.],4),event_values=values,event_valid=np.ones(values.shape,bool));nat,node_mc,_=phase.node_context(m,events);case=phase.cases(m,nat,node_mc,groups,count,scores);ix,target=phase.labels(case,groups,ev);data=dict(old=case['old'][ix],mc=case['mc'][ix],target=target,w=np.ones(len(ix)));data.update({'x_'+k:case['x'][k][ix] for k in phase.KINDS});models,_,_=gate.fit(data,'cpu',steps=2);torch.save(models,run/'private'/f'model-leave_{sid}.pt');raw={};oldraw={};prior={};parts=dict(pairs=m['pairs'],scores=np.ones(len(m['pairs']))*2);oldparts=parts.copy();op={};policy={};groupmap={};relation={}
                for front,arms in pilot.ARMS.items():
                    op[front+'__count']=count;op[front+'__scores']=scores;op[arms[1]+'__weights']=weights['O'];oldparts[front]=member;oldparts[front+'_scores']=parts['scores'];persist,_,_=ni.persistence(m,events,groups,count,weights['O'],o);mean,_,_=ni.mean_velocity_control(m,events,groups,count,weights['O'],o)
                    for prefix in arms[:7]:raw[prefix+'_MEAN']=base;raw[prefix+'_NN']=base if prefix==front else o if prefix==arms[1] else mean if prefix==arms[6] else persist[prefix[1:]]
                    oldraw.update({k:v for k,v in raw.items() if k.startswith(front+'_') or k.startswith(arms[1]+'_')});prior.update(raw);gid,j=case['indices'].T;policy[front+'__eligible']=case['eligible'];policy[front+'__indices']=case['indices']
                    for prefix,kind in zip(arms[7:],('HALF',)+phase.KINDS):
                        alpha=np.full(len(gid),.5) if kind=='HALF' else gate.predict(models[kind],case['x'][kind],'cpu');candidate=o.copy();candidate[gid,j,:2]=phase.blend(case['old'],case['mc'],alpha);raw[prefix+'_NN']=candidate;raw[prefix+'_MEAN']=base;policy[prefix+'__alpha']=alpha
                        if kind!='HALF':policy[prefix+'__features']=case['x'][kind]
                    for prefix in arms:
                        groupmap[prefix]=groups;parts[prefix]=member;parts[prefix+'_scores']=parts['scores'];rec=dict(scene_id=sid,component_id=sid,pairs=m['pairs'],node_ids=m['node_ids'],sources=m['sources'],labels=ev['identities'][m['pairs'][:,0]]==ev['identities'][m['pairs'][:,1]],joint_support_queries=(m['support'][m['pairs'][:,0]]&m['support'][m['pairs'][:,1]]).sum(1));relation[prefix+'_raw']=p.partition.frozen.evaluate_scenes([dict(rec,scores=parts['scores'])],p.partition.THRESHOLD);co=member[m['pairs'][:,0]]==member[m['pairs'][:,1]];relation[prefix+'_decoded']=p.partition.frozen.evaluate_scenes([dict(rec,scores=co.astype(float))],.5)
                for name,val in [('native-motion',m),('native-events',events),('native-evaluation',ev)]:
                    folder=native/'private'/name/'T_core';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/f'scene-{sid}.npz',**val)
                for dest,name,val in [(parent,'partitions',oldparts),(parent,'policy',op),(parent,'raw',oldraw),(control,'raw',prior),(run,'partitions',parts),(run,'policy',policy),(run,'raw',raw)]:np.savez_compressed(dest/'private'/f'{name}-T_core-{sid}.npz',**val)
                row,arr,artifact=p.evaluate(m,ev,groupmap,{},raw,'T_core',method_names=pilot.METHODS);row.update(scene_id=sid,component_id=sid,search={},association=relation);p.common.old.frozen.write(run/'scenes'/f'T_core-{sid}.json.gz',row);p.common.save_artifacts(run/'private'/f'states-T_core-{sid}.npz',artifact);rows.append(row);arrays.append(arr)
            r=dict(status='COMPLETED',exit_code=0,mode='temporal_gate',methods=pilot.METHODS,parent_run=str(parent),control_run=str(control),native_run=str(native),manifest=str(manifest),training_scene_ids=['1','3'],scene_ids=[],evaluation_splits=['T_core']);p.common.old.frozen.write(run/'run.json',r);p.common.old.frozen.write(run/'T-summary.json',p.finish(rows,arrays,pilot.METHODS,pilot.PRIMARY));p.common.old.frozen.write(run/'T-scene-summary.json.gz',rows);closeout(run);audit.account(run,run/'history-followup-accounting.json',(pilot.ARMS,pilot.PRIMARY,audit.reconstruct,dict(fits=3,updates=6)));self.assertEqual(set(p.common.old.frozen.read(run/'independent-output-audit.json')['splits']),{'T_core'});self.assertFalse((run/'summary.json').exists());self.assertEqual(p.common.old.frozen.read(run/'four-attribute-accounting.json')['status'],'COMPLETED')
