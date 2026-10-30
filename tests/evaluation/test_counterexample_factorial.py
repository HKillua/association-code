import unittest
import numpy as np
import torch
from association.evaluation.robust_native_endpoint import endpoint_line
from association.evaluation.local_source_evidence import local_features,ResidualScore,fit_residual,predict_residual
from association.evaluation.native_temporal import fit_quadratic
from association.evaluation.source_risk import build_source_features
from tests.integration.test_source_risk_pilot import scene_arrays

class FactorialTests(unittest.TestCase):
    def test_endpoint_is_not_forced_through_bad_last_observation(self):
        t=np.arange(8.);y=np.column_stack((2*t,3*t));y[-1]+=[1000,-1000]
        p,v=endpoint_line(t,y);np.testing.assert_allclose(p,[14,21]);np.testing.assert_allclose(v,[2,3])
        p2,v2=endpoint_line(t+100,y+[40,-20]);np.testing.assert_allclose(p2,p+[40,-20]);np.testing.assert_allclose(v2,v)
    def test_local_encoding_units_and_node_permutation(self):
        m,_,_,_=scene_arrays('1');groups=[[0,1],[2],[3]];keys=('states','support','nearest_seconds','sources','native_count','span_seconds','fit_rmse_m','ols_velocity_mps')
        pack=build_source_features(**{k:m[k] for k in keys},groups=groups);a=local_features(m,pack,speed_unit='kn');m2=dict(m,states=m['states'].copy());m2['states'][...,2]*=1852/3600
        np.testing.assert_allclose(a,local_features(m2,pack,speed_unit='mps'))
        order=np.array([1,0,3,2]);pm={k:(v[order] if k in keys else v) for k,v in m.items()};pp=dict(pack,node_to_source_row=pack['node_to_source_row'][order])
        # The new feature operator preserves source rows under a remapped node-to-source map.
        np.testing.assert_allclose(a,local_features(pm,pp,speed_unit='kn'))
    def test_residual_starts_at_base_and_has_real_optimizer_updates(self):
        torch.set_num_threads(2);rng=np.random.default_rng(20);x=rng.normal(size=(80,10));y=rng.normal(size=80);w=np.ones(80);base=fit_quadratic(x,y,w)
        np.testing.assert_array_equal(ResidualScore(10)(torch.tensor(x,dtype=torch.float32)).detach().numpy(),np.zeros(80))
        m=fit_residual(x,y,w,base,device='cpu',steps=4);self.assertEqual(m['updates'],4);self.assertTrue(np.isfinite(predict_residual(m,x,'cpu')).all());self.assertTrue(any(r['gradient_norm']>0 for r in m['curve']))

if __name__=='__main__':unittest.main()
