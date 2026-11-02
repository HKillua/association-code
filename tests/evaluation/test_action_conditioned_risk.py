import unittest
import numpy as np
from association.evaluation import action_conditioned_risk as a,history_candidates as h
from tests.integration.test_source_risk_pilot import scene_arrays
from tools.analysis.action_conditioned_risk_closeout import native

class ActionRiskTests(unittest.TestCase):
    def fixture(self):
        m,_,_,_=scene_arrays('1');m['support'][:,[0,-1]]=False;g=[[0,1,2],[3]];m['sources']=np.array(['a','b','c','a']);m['native_count']=np.full(4,2);m['span_seconds']=np.full(4,30.)
        value=np.repeat(m['states'][:,0],2,axis=0);value[:,2]/=1852/3600;value[1::2,2]+=1
        e=dict(event_offsets=np.arange(5)*2,event_times_seconds=np.tile([10.,40.],4),event_values=value,event_valid=np.ones(value.shape,bool))
        x,c=h.descriptors(m,g);return m,e,g,x,c

    def test_native_kn_signed_time_and_missing_report(self):
        m,e,g,x,c=self.fixture();mc,v,b,k,_=a.candidates(m,e);mm,vv,bb,kk=native(m,e)
        np.testing.assert_allclose(mc,mm,atol=1e-10);np.testing.assert_allclose(v,vv,atol=1e-12);np.testing.assert_array_equal(k,kk)
        e['event_valid'][:,2:]=False;mc,v,b,k,n=a.candidates(m,e);np.testing.assert_array_equal(mc,m['states']);self.assertFalse(k.any());self.assertEqual(n,0)

    def test_frozen_context_and_only_three_action_columns(self):
        m,e,g,x,c=self.fixture();mc,v,b,k,_=a.candidates(m,e);xx,eligible=a.features(m,g,c,x,mc,v,b,k)
        np.testing.assert_array_equal(xx[...,[0,1,2,3,4,5,9,10,11]],x[...,[0,1,2,3,4,5,9,10,11]])
        self.assertTrue(eligible[:,[0,-1]].all());self.assertFalse(eligible[:,1:-1].any())
        # If the action is unchanged, the frozen-context features equal the original.
        same,_=a.features(m,g,c,x,m['states'],m['ols_velocity_mps'],b,k);np.testing.assert_allclose(same,x,atol=1e-12)

    def test_action_and_node_separation_preserves_unaffected_outputs(self):
        m,e,g,x,c=self.fixture();mc,v,b,k,_=a.candidates(m,e);_,eligible=a.features(m,g,c,x,mc,v,b,k);old=np.ones(m['support'].shape);old[0]=0;new=old.copy();new[1]=-2;new[0]=-1;base=m['states'][[0,3]].copy()
        out,w=a.readouts(m,g,old,new,mc,eligible,base)
        self.assertEqual(w['CF'][0,0,1],1);self.assertEqual(w['JM'][1,0,1],1);self.assertEqual(w['RM'][1,0,1],1)
        for key in out:
            np.testing.assert_array_equal(out[key][~eligible],base[~eligible]);np.testing.assert_array_equal(out[key][...,2:],base[...,2:])
        renamed=dict(m,sources=np.array(['8','2','z','8']))
        other,_=a.readouts(renamed,[list(reversed(z)) for z in g],old,new,mc,eligible,base)
        for key in out:np.testing.assert_allclose(out[key],other[key],atol=1e-12)

    def test_hierarchical_ties_do_not_overweight_extra_modes(self):
        nodes=np.array([0,0,1,2]);modes=np.array([0,1,0,1]);sources=np.array(['a','a','b']);w=a.tie_mass(nodes,modes,sources)
        np.testing.assert_allclose(w,[.125,.125,.25,.5]);np.testing.assert_allclose(a.tie_mass(nodes[::-1],modes[::-1],sources)[::-1],w)

if __name__=='__main__':unittest.main()
