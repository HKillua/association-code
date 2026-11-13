import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from association.data.motion_evidence import build_motion_evidence
from association.evaluation import native_motion_interventions as ni
from tools.analysis.native_motion_closeout import forward_position
from tools.analysis import fitted_endpoint_pilot as pilot,fitted_endpoint_closeout as audit

class FittedEndpointTests(unittest.TestCase):
    def fixture(self,linear=False,date_line=False):
        values=np.zeros((3,3,4));values[...,0]=10.+np.array([0.,.001,.002] if linear else [0.,.003,0.]);values[...,1]=np.array([179.998,-179.999,-179.996]) if date_line else 179.999;values[...,2]=2./(1852/3600);times=np.tile([0.,10.,20.],(3,1));query=np.array([-10.,5.,30.]);m=build_motion_evidence(values,times,np.ones((3,3),bool),query,np.array(['a','b','c']));m['states'][...,2]*=1852/3600
        events=dict(event_offsets=np.arange(4)*3,event_times_seconds=times.ravel(),event_values=values.reshape(-1,4),event_valid=np.ones((9,4),bool));weights=np.ones((3,3))/3;count=np.array([[0,3,0]]);base=m['states'][[0]].copy();mc,_,_=ni.mean_velocity_control(m,events,[[0,1,2]],count,weights,base);return m,events,weights,count,mc
    def test_fitted_latent_state_and_factorial_match_analytic_linear_fit(self):
        m,e,w,count,mc=self.fixture();models,meta=ni.fit_endpoint_models(m,e);self.assertEqual(meta['native_OLS_fits'],3);self.assertLess(meta['max_old_coordinate_gap_deg'],1e-12);out,flags,_=ni.fitted_endpoint_controls(m,models,[[0,1,2]],count,w,mc)
        np.testing.assert_allclose(models[0]['fitted'][:,0],10.001,atol=1e-12)
        for j,dt in [(0,-10.),(2,10.)]:
            np.testing.assert_allclose(out['GL'][0,j,:2],[10.,179.999],atol=1e-10);np.testing.assert_allclose(out['FL'][0,j,:2],[10.001,179.999],atol=1e-10);np.testing.assert_allclose(out['FM'][0,j,:2],forward_position(np.array([10.001,179.999]),np.array([2.,0.])*dt),atol=1e-10)
        for v in out.values():np.testing.assert_array_equal(v[0,1],mc[0,1]);np.testing.assert_array_equal(v[...,2:],mc[...,2:])
        m,e,w,count,mc=self.fixture(True);models,_=ni.fit_endpoint_models(m,e);np.testing.assert_allclose(models[0]['native'],models[0]['fitted'],atol=1e-12)
    def test_K3_joint_domain_fallback_and_cache_binding_failure(self):
        m,e,w,count,mc=self.fixture();models,_=ni.fit_endpoint_models(m,e);out,_,_=ni.fitted_endpoint_controls(m,models,[[0,1,2]],count,w,mc);renamed=dict(m,sources=np.array(['x','00','19']));again,_,_=ni.fitted_endpoint_controls(renamed,models,[[2,0,1]],count,w,mc)
        for k in out:np.testing.assert_allclose(out[k],again[k],atol=1e-12)
        models[0]['fitted'][:,0]=91.;bad,flags,diag=ni.fitted_endpoint_controls(m,models,[[0,1,2]],count,w,mc);self.assertEqual(diag['joint_fallback_queries'],2)
        for v in bad.values():np.testing.assert_array_equal(v,mc)
        m['states'][0,0,0]+=.01
        with self.assertRaisesRegex(ValueError,'old native OLS state binding'):ni.fit_endpoint_models(m,e)
    def test_saved_actions_independent_lstsq_reconstruction_and_primary_order(self):
        m,e,w,count,mc=self.fixture(date_line=True);models,_=ni.fit_endpoint_models(m,e);outputs,flags,_=ni.fitted_endpoint_controls(m,models,[[0,1,2]],count,w,mc)
        for v in outputs.values():self.assertTrue((-180<=v[...,1]).all() and (v[...,1]<=180).all())
        with TemporaryDirectory() as tmp:
            run=Path(tmp)/'190';parent=Path(tmp)/'189';(run/'private').mkdir(parents=True);(parent/'private').mkdir(parents=True);raw={}
            for prefix in pilot.OLD_ARMS['HARD']:
                raw[prefix+'_NN']=mc.copy();raw[prefix+'_MEAN']=mc.copy()
            np.savez_compressed(parent/'private/raw-T_core-1.npz',**raw);np.savez_compressed(parent/'private/policy-T_core-1.npz',HARD__eligible=flags['eligible']);np.savez_compressed(run/'private/fitted-models-T_core-1.npz',**{key:np.stack([v[key] for v in models]) for key in models[0]})
            for k,v in outputs.items():raw['H'+k+'_NN']=v
            policy=dict(HARD__eligible=flags['eligible'],HARD__joint=flags['joint']);op=dict(HO__weights=w)
            audit.reconstruct(run,dict(ablation_run=str(parent)),'T_core','1','HARD',m,e,[np.arange(3)],count,op,{},raw,policy)
        for key in pilot.PRIMARY:
            right,left=key.split('_vs_');self.assertLess(pilot.METHODS.index(left),pilot.METHODS.index(right))
        self.assertEqual(len(pilot.METHODS),40)

if __name__=='__main__':unittest.main()
