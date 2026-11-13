import unittest
import numpy as np
from association.evaluation import native_compatibility as n

class NativeCompatibilityTests(unittest.TestCase):
    def fixture(self):
        times=np.tile([0.,10.],3);values=np.zeros((6,4));values[:,2:]=[10.,90.]
        values[:,:2]=n.advance(np.array([0.,179.999]),np.array([0.,10*1852/3600]),times)
        ev=dict(event_offsets=np.arange(4)*2,event_times_seconds=times,event_values=values,event_valid=np.ones((6,4),bool));query=np.array([-5.,5.,15.]);states=np.zeros((3,3,4));states[...,:2]=n.advance(np.array([0.,179.999]),np.array([0.,10*1852/3600]),query)
        m=dict(states=states,sources=np.array(['A','B','C']),query_seconds=query,support=np.zeros((3,3),bool),ols_velocity_mps=np.tile([0.,10*1852/3600],(3,1)))
        return m,ev
    def test_units_signed_sphere_dateline_and_exact_motion(self):
        m,e=self.fixture();s=n.stream(e,[0],speed_unit='kn');np.testing.assert_allclose(s['v'][:,1],10*1852/3600)
        a=n.at_queries(s,m['query_seconds']);np.testing.assert_allclose(a['reported'],m['states'][0,:,:2],atol=1e-11);np.testing.assert_array_equal(a['bracket'],[False,True,False]);self.assertTrue(a['before'][0] and a['after'][2])
        x,_=n.descriptors(m,e,[[0,1,2]],speed_unit='kn');self.assertLess(x[...,1].max(),1e-8);self.assertEqual(x.shape,(3,3,13))
        with self.assertRaises(ValueError):n.stream(e,[0],speed_unit=None)
    def test_k3_rename_member_order_and_missing_report(self):
        m,e=self.fixture();x,_=n.descriptors(m,e,[[0,1,2]],speed_unit='kn');mm=dict(m,sources=np.array(['z','00','9']));xx,_=n.descriptors(mm,e,[[2,0,1]],speed_unit='kn');np.testing.assert_allclose(x,xx,atol=1e-12)
        e['event_valid'][:,2:]=False;xx,_=n.descriptors(m,e,[[0,1,2]],speed_unit='kn');np.testing.assert_array_equal(xx[...,7],0);np.testing.assert_array_equal(xx[...,12],0);self.assertTrue(np.isfinite(xx).all())
    def test_native_jump_is_not_treated_as_healthy_motion(self):
        m,e=self.fixture();e['event_values'][1,0]=1.;x,_=n.descriptors(m,e,[[0,1,2]],speed_unit='kn');self.assertGreater(x[0,1,10],5.)
    def test_qualification_first_anchor_population_and_saved_harms(self):
        from tools.analysis.native_compatibility_qualification import analyze,summarize
        m,e=self.fixture();m['states'][0,0,0]=.02;x,d=n.descriptors(m,e,[[0,1,2]],speed_unit='kn');reference=m['states'][1].copy();ev=dict(target_ids=np.array(['t']),identities=np.array(['t','t','t']),target_states=reference[None],target_valid=np.ones((1,3,4),bool))
        raw={'HARD_NN':reference[None],'HO_NN':m['states'][:1]};policy={'HARD__count':np.zeros((1,3),int),'HO__weights':np.tile([[1.],[0.],[0.]],(1,3)),'HARD__scores':np.tile([[0.],[1.],[2.]],(1,3))};bridge=raw['HO_NN'].copy();bridge[0,2,0]=.02
        rows=analyze(m,ev,[np.array([0,1,2])],raw,policy,'HARD','HO',bridge,x,d);s=summarize(rows);self.assertEqual(s['all']['all']['points'],3);self.assertEqual(s['all']['old_harm']['points'],1);self.assertEqual(s['all']['bridge_harm']['points'],1)

if __name__=='__main__':unittest.main()
