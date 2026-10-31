import unittest
import numpy as np
from association.evaluation.set_conditioned_risk import conditional_features,fit_heads

class ContextTests(unittest.TestCase):
    def test_other_source_context_is_equivariant_and_ignores_unavailable_rows(self):
        x=np.arange(3*2*10,dtype=float).reshape(3,2,10)/10;use=np.array([[1,1],[1,0],[1,0]],bool);p=dict(features=x,available=use,group_offsets=np.array([0,3]));base=dict(mean=np.zeros(10),std=np.ones(10))
        c=conditional_features(p,base,context=True);np.testing.assert_allclose(c[0,0,10:],x[1:,0].mean(0));np.testing.assert_array_equal(c[0,1,10:],np.zeros(10))
        order=np.array([2,0,1]);cp=conditional_features(dict(p,features=x[order],available=use[order]),base,context=True);np.testing.assert_allclose(cp,c[order])
        own=conditional_features(p,base,context=False);np.testing.assert_array_equal(own[...,:10],own[...,10:])
    def test_equal_budget_heads_have_zero_initial_residual_and_real_updates(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(20,20));target=rng.normal(size=20);heads=fit_heads({'OWN':x,'CONTEXT':x},target,np.ones(20),np.zeros(20),device='cpu',steps=2)
        self.assertEqual(heads['OWN']['initial_mse'],heads['CONTEXT']['initial_mse']);self.assertEqual(heads['OWN']['final_mse'],heads['CONTEXT']['final_mse']);self.assertEqual(len(heads['OWN']['curve']),2)
        self.assertNotEqual(heads['OWN']['initial_mse'],heads['OWN']['final_mse'])

if __name__=='__main__':unittest.main()
