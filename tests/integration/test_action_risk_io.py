"""Small synthetic cache exercises 195 CLI and audit wiring, not MTAD effects."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from tests.integration.test_source_risk_pilot import scene_arrays
from association.evaluation import history_candidates as h,native_motion_interventions as ni
from association.evaluation.local_source_evidence import ResidualScore
from tools.analysis import action_conditioned_risk_pilot as pilot,action_conditioned_risk_closeout as audit
from tools.analysis.association_context_closeout import closeout
p=pilot.p

class ActionRiskIOTests(unittest.TestCase):
    def test_real_cli_t32_saved_arrays_and_audits(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'178';control=root/'189';native=root/'127';out=root/'195'
            for d in (parent,control):(d/'private').mkdir(parents=True)
            (control/'scenes').mkdir();ids=[str(i) for i in range(32)];manifest=root/'manifest.json';p.common.old.frozen.write(manifest,dict(components={sid:str(i%12) for i,sid in enumerate(ids)}))
            record=dict(status='COMPLETED',native_run=str(native),training_scene_ids=ids,manifest=str(manifest))
            p.common.old.frozen.write(parent/'run.json',record);p.common.old.frozen.write(control/'run.json',dict(status='COMPLETED',parent_run=str(parent),evaluation_splits=['T_core']))
            net=ResidualScore(12)
            model=dict(dimension=12,mean=np.zeros(12),std=np.ones(12),base=dict(mean=np.zeros(12),std=np.ones(12),beta=np.zeros(91)),weights={k:torch.zeros_like(v) for k,v in net.state_dict().items()})
            for fold in range(12):torch.save(model,parent/'private'/f'model-leave_{fold}.pt')
            for sid in ids:
                m,ev,member,_=scene_arrays(sid);m['support'][:,-1]=False;m['native_count']=np.full(4,2);m['span_seconds']=np.full(4,30.);g=[[0,1],[2],[3]];x,count=h.descriptors(m,g);scores=np.zeros(m['support'].shape);base=m['states'][[0,2,3]].copy();outputs,weights=h.readouts(m,g,count,scores,base);o=outputs['O'];values=np.repeat(m['states'][:,0],2,axis=0);values[:,2]/=1852/3600;values[1::2,2]+=1
                events=dict(node_ids=m['node_ids'],query_seconds=m['query_seconds'],event_offsets=np.arange(5)*2,event_times_seconds=np.tile([-10.,20.],4),event_values=values,event_valid=np.ones(values.shape,bool));mean,_,_=ni.mean_velocity_control(m,events,g,count,weights['O'],o);persist,_,_=ni.persistence(m,events,g,count,weights['O'],o)
                parts=dict(pairs=m['pairs'],scores=np.ones(len(m['pairs']))*2);op={};raw={};oldraw={};relation={}
                for front,arms in pilot.ARMS.items():
                    parts[front]=member;parts[front+'_scores']=parts['scores'];op.update({front+'__features':x,front+'__scores':scores,front+'__count':count,arms[1]+'__weights':weights['O']})
                    for prefix in arms[:7]:raw[prefix+'_MEAN']=base;raw[prefix+'_NN']=base if prefix==front else o if prefix==arms[1] else mean if prefix==arms[6] else persist[prefix[1:]]
                    oldraw.update({k:v for k,v in raw.items() if k.startswith(front+'_') or k.startswith(arms[1]+'_')})
                    rec=dict(scene_id=sid,component_id=str(int(sid)%12),pairs=m['pairs'],node_ids=m['node_ids'],sources=m['sources'],labels=ev['identities'][m['pairs'][:,0]]==ev['identities'][m['pairs'][:,1]],joint_support_queries=(m['support'][m['pairs'][:,0]]&m['support'][m['pairs'][:,1]]).sum(1))
                    relation[front+'_raw']=p.partition.frozen.evaluate_scenes([dict(rec,scores=parts['scores'])],p.partition.THRESHOLD);co=member[m['pairs'][:,0]]==member[m['pairs'][:,1]];relation[front+'_decoded']=p.partition.frozen.evaluate_scenes([dict(rec,scores=co.astype(float))],.5)
                p.common.old.frozen.write(control/'scenes'/f'T_core-{sid}.json.gz',dict(association=relation))
                for name,z in [('native-motion',m),('native-events',events),('native-evaluation',ev)]:
                    d=native/'private'/name/'T_core';d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/f'scene-{sid}.npz',**z)
                for dest,name,z in [(parent,'partitions',parts),(parent,'policy',op),(parent,'raw',oldraw),(control,'raw',raw)]:np.savez_compressed(dest/'private'/f'{name}-T_core-{sid}.npz',**z)
            pilot.run(SimpleNamespace(parent_run=parent,control_run=control,output=out,time_policy='advisory'))
            closeout(out);audit.account(out,out/'history-followup-accounting.json',(pilot.ARMS,pilot.PRIMARY,audit.reconstruct,dict(fits=0,updates=0)));audit.opportunities(out)
            r=p.common.old.frozen.read(out/'run.json');self.assertEqual(r['status'],'COMPLETED');self.assertEqual(r['completed_T_scenes'],32);self.assertEqual(r['optimizer_updates'],0);self.assertFalse((out/'summary.json').exists());self.assertEqual(len(r['methods']),40)
            self.assertEqual(p.common.old.frozen.read(out/'action-opportunity-accounting.json')['status'],'COMPLETED')
