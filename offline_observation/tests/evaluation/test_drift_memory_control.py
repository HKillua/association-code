import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from association.evaluation import native_motion_interventions as ni,history_candidates as h
from tools.analysis.native_motion_closeout import forward_position
from tools.analysis import drift_memory_pilot as pilot,drift_memory_closeout as audit
from tools.analysis import association_context_pilot as p

class DriftMemoryControlTests(unittest.TestCase):
    def fixture(self):
        states=np.zeros((3,3,4));states[...,0]=10.;states[...,1]=179.999;states[...,2]=4.;states[...,3]=90.
        m=dict(states=states,sources=np.array(['a','b','c']),query_seconds=np.array([-5.,5.,25.]),ols_velocity_mps=np.zeros((3,2)),fit_rmse_m=np.ones(3))
        values=np.zeros((6,4));values[:,:2]=[10.,179.999];values[1::2,2]=10./(1852/3600);values[:,3]=90.
        events=dict(event_offsets=np.arange(4)*2,event_times_seconds=np.tile([0.,10.],3),event_values=values,event_valid=np.ones((6,4),bool))
        base=states[[0]].copy();count=np.array([[0,1,0]]);weights=np.ones((3,3))/3
        return m,events,base,count,weights

    def test_same_endpoint_mean_velocity_units_signed_time_and_preserved_domain(self):
        m,e,base,count,w=self.fixture();out,flags,diag=ni.mean_velocity_control(m,e,[[0,1,2]],count,w,base)
        for j,dt in [(0,-5.),(2,15.)]:np.testing.assert_allclose(out[0,j,:2],forward_position(base[0,j,:2],np.array([0.,5.])*dt),atol=1e-10)
        np.testing.assert_array_equal(flags['eligible'],[[True,False,True]]);np.testing.assert_array_equal(out[0,1],base[0,1]);np.testing.assert_array_equal(out[...,2:],base[...,2:]);self.assertEqual(diag['native_mean_computations'],3)
        self.assertTrue((-180<=out[...,1]).all() and (out[...,1]<=180).all())
        for dt in [-100.,100.]:
            v=np.array([0.,10.]);mu=np.array([0.,5.]);np.testing.assert_array_equal(ni.ou_displacement(v,mu,0,dt),v*dt);np.testing.assert_allclose(ni.ou_displacement(v,mu,1e6,dt),mu*dt,atol=1e-5)

    def test_list_K3_anonymous_order_and_common_missing_report_fallback(self):
        m,e,base,count,w=self.fixture();out,_,_=ni.mean_velocity_control(m,e,[[0,1,2]],count,w,base)
        other=dict(m,sources=np.array(['9','2','z']));again,_,_=ni.mean_velocity_control(other,e,[[2,0,1]],count,w,base);np.testing.assert_allclose(again,out,atol=1e-12)
        e['event_valid'][:,2:]=False;bad,_,d=ni.mean_velocity_control(m,e,[[0,1,2]],count,w,base);prior,_,_=ni.persistence(m,e,[[0,1,2]],count,w,base);np.testing.assert_array_equal(bad,prior['PG']);np.testing.assert_array_equal(bad[...,2:],base[...,2:]);self.assertEqual(d['unknown_selected_node_queries'],6);self.assertEqual(d['native_mean_computations'],0)

    def test_T_only_saved_pipeline_independent_actions_attributes_and_declared_splits(self):
        self._saved_pipeline('T_core')

    def test_V_only_saved_pipeline_independent_actions_attributes_and_declared_splits(self):
        self._saved_pipeline('V_select')

    def _saved_pipeline(self,split):
        from tests.integration.test_source_risk_pilot import scene_arrays
        from tools.analysis.association_context_closeout import closeout
        with TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'189';parent=root/'178';prior=root/'188';native=root/'127'
            for q in (run,parent,prior):(q/'private').mkdir(parents=True)
            p.common.old.frozen.write(parent/'run.json',dict(status='COMPLETED'))
            (run/'scenes').mkdir();rows=[];arrays=[];profiles={f:[] for f in pilot.ARMS};means=0
            record=dict(status='COMPLETED',exit_code=0,mode='native_drift_memory_ablation',methods=pilot.METHODS,parent_run=str(parent),persistence_run=str(prior),native_run=str(native),training_scene_ids=['1','3'] if split=='T_core' else [],scene_ids=['1','3'] if split=='V_select' else [],evaluation_splits=[split],native_mean_computations=0,native_likelihood_fits=0,optimizer_updates=0,neural_head_fits=0)
            for sid in ['1','3']:
                m,ev,member,_=scene_arrays(sid);m['support'][:,-1]=False;g=[[0,1],[2],[3]];_,count=h.descriptors(m,g);node_scores=np.zeros(m['support'].shape);base=np.stack([m['states'][0],m['states'][2],m['states'][3]])
                values=np.repeat(m['states'][:,0],2,axis=0).copy();values[:,2]/=1852/3600;values[1::2,2]+=1.;events=dict(event_offsets=np.arange(5)*2,event_times_seconds=np.tile([-10.,20.],4),event_values=values,event_valid=np.ones((8,4),bool))
                _,weights=h.readouts(m,g,count,node_scores,base);op={};raw={};original={};groups={};parts=dict(pairs=m['pairs'],scores=np.ones(len(m['pairs']))*2);parent_parts=dict(pairs=m['pairs'],scores=parts['scores']);policy={};prior_policy={};diags={};relation={}
                for front,arms in pilot.ARMS.items():
                    op[front+'__count']=count;op[arms[1]+'__weights']=weights['O'];op[front+'__scores']=node_scores;parent_parts[front]=member;parent_parts[front+'_scores']=parts['scores']
                    originals,_,_=ni.persistence(m,events,g,count,weights['O'],base)
                    for prefix in arms[:-1]:
                        raw[prefix+'_MEAN']=base;raw[prefix+'_NN']=base if prefix in arms[:2] else originals[prefix[1:]]
                    original.update({key:value for key,value in raw.items() if key.startswith(front+'_') or key.startswith(arms[1]+'_')})
                    out,flags,diag=ni.mean_velocity_control(m,events,g,count,weights['O'],base);raw[arms[-1]+'_MEAN']=base;raw[arms[-1]+'_NN']=out;policy[front+'__eligible']=flags['eligible'];prior_policy[front+'__eligible']=flags['eligible'];diags[front]=diag;means+=diag['native_mean_computations']
                    for prefix in arms:
                        groups[prefix]=g;parts[prefix]=member;parts[prefix+'_scores']=parts['scores'];rec=dict(scene_id=sid,component_id=sid,pairs=m['pairs'],node_ids=m['node_ids'],sources=m['sources'],labels=ev['identities'][m['pairs'][:,0]]==ev['identities'][m['pairs'][:,1]],joint_support_queries=(m['support'][m['pairs'][:,0]]&m['support'][m['pairs'][:,1]]).sum(1))
                        relation[prefix+'_raw']=p.partition.frozen.evaluate_scenes([dict(rec,scores=parts['scores'])],p.partition.THRESHOLD);decoded=member[m['pairs'][:,0]]==member[m['pairs'][:,1]];relation[prefix+'_decoded']=p.partition.frozen.evaluate_scenes([dict(rec,scores=decoded.astype(float))],.5)
                    condition=pilot.main_rows(m,ev,g,raw,diag,front)
                    for v in condition:v['scene_id']=sid
                    profiles[front].extend(condition);p.common.old.frozen.write(run/'private'/f'conditions-{front}-{sid}.json.gz',condition)
                for name,values in [('native-motion',m),('native-events',events),('native-evaluation',ev)]:
                    folder=native/'private'/name/split;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/f'scene-{sid}.npz',**values)
                for dest,filename,values in [(parent,'partitions',parent_parts),(parent,'policy',op),(parent,'raw',original),(prior,'raw',{k:v for k,v in raw.items() if 'MC_' not in k}),(prior,'policy',prior_policy),(run,'raw',raw),(run,'partitions',parts),(run,'policy',policy)]:np.savez_compressed(dest/'private'/f'{filename}-{split}-{sid}.npz',**values)
                p.common.old.frozen.write(run/'private'/f'diagnostics-{split}-{sid}.json.gz',diags)
                row,arr,artifact=p.evaluate(m,ev,groups,{},raw,split,method_names=pilot.METHODS);row.update(scene_id=sid,component_id=sid,search={},association=relation);p.common.old.frozen.write(run/'scenes'/f'{split}-{sid}.json.gz',row);p.common.save_artifacts(run/'private'/f'states-{split}-{sid}.npz',artifact);rows.append(row);arrays.append(arr)
            record['native_mean_computations']=means;p.common.old.frozen.write(run/'run.json',record);summary=p.finish(rows,arrays,pilot.METHODS,pilot.PRIMARY);p.common.old.frozen.write(run/('T-summary.json' if split=='T_core' else 'summary.json'),summary);p.common.old.frozen.write(run/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows);p.common.old.frozen.write(run/'condition-profiles.json',dict(heads={f:pilot.condition_summary(v) for f,v in profiles.items()}))
            closeout(run);audit.account(run,run/'history-followup-accounting.json',(pilot.ARMS,pilot.PRIMARY,audit.reconstruct,dict(fits=0,updates=0)));audit.verify_conditions(run,record)
            self.assertEqual(set(p.common.old.frozen.read(run/'independent-output-audit.json')['splits']),{split});self.assertFalse((run/('summary.json' if split=='T_core' else 'T-summary.json')).exists());self.assertEqual(p.common.old.frozen.read(run/'drift-condition-accounting.json')['heads']['HARD']['main_eligible_points'],4)
            for key in pilot.PRIMARY:
                right,left=key.split('_vs_');self.assertLess(pilot.METHODS.index(left),pilot.METHODS.index(right))
            record['evaluation_splits']=[];p.common.old.frozen.write(run/'run.json',record)
            with self.assertRaises((ValueError,AssertionError)):audit.account(run,run/'invalid-splits.json',(pilot.ARMS,pilot.PRIMARY,audit.reconstruct,dict(fits=0)))
            from tools.analysis import audit_association_context_outputs as generic
            with self.assertRaises((ValueError,AssertionError)):generic.parent.audit(run,run/'invalid-audit.json',extra_scene=generic.scene,extra_summary=generic.summary)

if __name__=='__main__':unittest.main()
