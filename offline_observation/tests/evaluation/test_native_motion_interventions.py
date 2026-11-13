import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import torch
from association.evaluation import native_motion_interventions as ni
from association.training import native_motion_correction as nc
from tests.evaluation.test_cost_routing_bridge import NativeBridgeTests

class NativeMotionTests(unittest.TestCase):
    def test_hermite_matches_independent_WNOA_GP_and_bounded_curve(self):
        h=5.;tau=1.5;x0=np.array([0.,2.]);x1=np.array([10.,0.]);F=lambda t:np.array([[1.,t],[0.,1.]]);Q=lambda t:np.array([[t**3/3,t*t/2],[t*t/2,t]])
        psi=Q(tau)@F(h-tau).T@np.linalg.inv(Q(h));expected=(F(tau)-psi@F(h))@x0+psi@x1
        got=ni.hermite(tau/h,np.array([0.,0.]),np.array([10.,0.]),np.array([2.,0.]),np.array([0.,0.]),h,'raw');self.assertAlmostEqual(got[0],expected[0],places=11)
        for a in np.linspace(0,1,101):
            p=ni.hermite(a,np.array([0.,0.]),np.array([10.,0.]),np.array([-100.,900.]),np.array([800.,-900.]),h,'limited');self.assertGreaterEqual(p[0],0);self.assertLessEqual(p[0],10);self.assertEqual(p[1],0)
    def test_ou_beta_zero_units_profile_grid_and_long_horizon(self):
        v=np.array([2.,3.]);np.testing.assert_array_equal(ni.ou_displacement(v,np.zeros(2),0,10),v*10)
        long=ni.ou_displacement(v,np.zeros(2),.2,1e6);np.testing.assert_allclose(long,v/.2)
        for dt in [-100.,100.]:self.assertLess(np.linalg.norm(ni.ou_displacement(v,np.zeros(2),.2,dt)),np.linalg.norm(v*dt))
        times=np.array([0.,1.,3.,7.]);model=ni.fit_ou(times,np.tile(v,(4,1)),False);self.assertTrue(model['constant_limit']);self.assertEqual(model['beta'],0)
        z=ni.fit_ou(times,np.array([[1.,0.],[.8,.1],[.4,-.1],[.1,.05]]),False);self.assertGreater(z['beta'],0);self.assertEqual(z['grid_size'],66)
    def test_native_domains_K3_and_matching_missing_report(self):
        m,e,base,scores,count=NativeBridgeTests().fixture();groups=[np.arange(5)];out,flags,_=ni.bridge(m,e,groups,count,scores,base,base)
        for k in out:np.testing.assert_array_equal(out[k][0,1:],base[0,1:]);np.testing.assert_array_equal(out[k][...,2:],base[...,2:])
        weights=np.zeros((5,3));weights[0]=1;po,pf,d=ni.persistence(m,e,groups,count,weights,base);self.assertEqual(d['native_likelihood_fits'],10);np.testing.assert_array_equal(pf['eligible'],[[False,False,True]])
        listout,listflags,_=ni.bridge(m,e,[list(range(5))],count,scores,base,base);listpo,_,_=ni.persistence(m,e,[list(range(5))],count,weights,base)
        for k in out:np.testing.assert_array_equal(listout[k],out[k])
        for k in po:np.testing.assert_array_equal(listpo[k],po[k])
        mm=dict(m,sources=np.array(['99','99','00','00','x']));oo,_,_=ni.bridge(mm,e,[np.array([4,2,3,0,1])],count,scores,base,base)
        for k in oo:np.testing.assert_allclose(oo[k],out[k],atol=1e-12)
        e['event_valid'][:,2:]=False;bad,_,_=ni.bridge(m,e,groups,count,scores,base,base)
        for k in bad:np.testing.assert_array_equal(bad[k],base)
    def test_matched_actual_fit_freezes_old_base_and_label_free_roundtrip(self):
        torch.set_num_threads(1);rng=np.random.default_rng(20);x=rng.normal(size=(64,25));y=x[:,0]*.1;w=np.ones(64);old=np.zeros(64);models,draws=nc.fit(x,y,w,old,'cpu',steps=2)
        self.assertEqual(draws.shape,(2,4096))
        with TemporaryDirectory() as tmp:
            f=Path(tmp)/'m.pt';torch.save(models,f);loaded=torch.load(f,weights_only=False)
            for key,model in models.items():
                self.assertEqual(sum(v.numel() for v in model['weights'].values()),1921);np.testing.assert_array_equal(nc.predict(model,x.copy(),'cpu'),nc.predict(loaded[key],x.copy(),'cpu'))
            changed=x.copy();changed[:,12:]+=100
            np.testing.assert_array_equal(nc.predict(models['OLDINFO'],x.copy(),'cpu'),nc.predict(models['OLDINFO'],changed,'cpu'))
    def test_all_primary_comparisons_available_in_method_order(self):
        from tools.analysis.native_motion_pilot import design
        for kind in ('feature','bridge','persistence'):
            _,methods,primary=design(kind)
            for key in primary:
                right,left=key.split('_vs_');self.assertLess(methods.index(left),methods.index(right))
    def test_independent_reconstruction_of_all_three_scientific_paths(self):
        from tools.analysis.native_motion_closeout import reconstruct
        from tools.analysis.native_motion_pilot import design
        from tools.analysis.association_context_pilot import common
        from tests.integration.test_source_risk_pilot import scene_arrays
        from association.evaluation import history_candidates as h,native_compatibility as features
        from association.evaluation.native_history_bridge import controls
        m,_,member,_=scene_arrays('1');m['support'][:,-1]=False;groups=[np.array([0,1]),np.array([2]),np.array([3])];x,count=h.descriptors(m,groups);op={'HARD__features':x,'HARD__count':count,'HARD__scores':np.zeros((4,64))};base=np.stack([m['states'][0],m['states'][2],m['states'][3]])
        _,w=h.readouts(m,groups,count,op['HARD__scores'],base);op['HO__weights']=w['O'];oraw={'HARD_NN':base,'HO_NN':base}
        vals=np.repeat(m['states'][:,0],2,axis=0);e=dict(event_offsets=np.arange(5)*2,event_times_seconds=np.tile([-10.,20.],4),event_values=vals,event_valid=np.ones((8,4),bool))
        with TemporaryDirectory() as tmp:
            run=Path(tmp)/'run';(run/'private').mkdir(parents=True);bdir=Path(tmp)/'bridge';(bdir/'private').mkdir(parents=True);manifest=Path(tmp)/'manifest.json';common.old.frozen.write(manifest,dict(components={'1':'c'}))
            b,_,_=controls(m,e,groups,count,op['HARD__scores'],base);np.savez_compressed(bdir/'private/raw-V_select-1.npz',HB_NN=b['B'],HE_NN=b['E'])
            for kind in ('feature','bridge','persistence'):
                values=design(kind)[0]['HARD'];raw={};policy={};r=dict(kind=kind,manifest=str(manifest),bridge_run=str(bdir))
                if kind=='feature':
                    extra,_=features.descriptors(m,e,groups,speed_unit='kn');xx=np.concatenate([x,extra],axis=-1);models,_=nc.fit(xx.reshape(-1,25),np.zeros(256),np.ones(256),np.zeros(256),'cpu',steps=2);torch.save(models,run/'private/model-full.pt');policy['HARD__features']=xx
                    for prefix,k in zip(values[2:],('OLDINFO','NATIVE')):
                        scores=nc.predict(models[k],xx.reshape(-1,25),'cpu').reshape(4,64);outputs,weights=h.readouts(m,groups,count,scores,base);raw[prefix+'_NN']=outputs['1'];policy[prefix+'__scores']=scores;policy[prefix+'__weights']=weights['1']
                    flags={'eligible':count==1}
                elif kind=='bridge':
                    outputs,flags,_=ni.bridge(m,e,groups,count,op['HARD__scores'],base,b['B']);raw['HB_NN']=b['B'];raw['HE_NN']=b['E']
                    for prefix,k in zip(values[4:],('CL','CH','CM')):raw[prefix+'_NN']=outputs[k]
                else:
                    outputs,flags,diag=ni.persistence(m,e,groups,count,op['HO__weights'],base);common.old.frozen.write(run/'private/diagnostics-V_select-1.json.gz',{'HARD':diag})
                    for prefix,k in zip(values[2:],('PH','PC','PD','PG')):raw[prefix+'_NN']=outputs[k]
                policy['HARD__eligible']=flags['eligible'];eligible,gap=reconstruct(run,r,'V_select','1','HARD',m,e,groups,count,op,oraw,raw,policy);np.testing.assert_array_equal(eligible,flags['eligible']);self.assertLess(gap,1e-4)

if __name__=='__main__':unittest.main()
