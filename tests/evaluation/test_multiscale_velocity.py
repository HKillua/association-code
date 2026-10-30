import unittest
import numpy as np
from association.evaluation.multiscale_velocity import geometric_median,native_multiscale
from association.evaluation.native_motion import predict_native_positions

class MultiscaleTests(unittest.TestCase):
    def test_robust_objective_rotation_and_coincident_point(self):
        x=np.array([[0.,0.],[0.,0.],[0.,0.],[10000.,5000.],[.01,0.]])
        z,ok,_=geometric_median(x);self.assertTrue(ok);np.testing.assert_allclose(z,[0,0],atol=1e-7)
        self.assertLessEqual(np.linalg.norm(x-z,axis=1).sum(),np.linalg.norm(x-x.mean(0),axis=1).sum())
        x=np.array([[0.,1.],[3.,2.],[-1.,4.],[7.,8.]])
        z,ok,_=geometric_median(x);r=np.array([[0.,-1.],[1.,0.]]);other,ok2,_=geometric_median(x@r+[20.,30.]);self.assertTrue(ok and ok2);np.testing.assert_allclose(other,z@r+[20.,30.],atol=1e-7)
    def test_two_points_exact_global_and_native_only(self):
        t=np.array([0.,10.]);v=np.array([[30.,179.99,1.,0.],[30.001,-179.99,1.,0.]]);q=np.array([-20.,5.,30.]);base=predict_native_positions(v[None],t[None],np.ones((1,2),bool),q)['predictions']['native_ols'];original=np.concatenate((base,np.ones((1,3,2))),axis=-1)
        cache=dict(event_offsets=np.array([0,2]),event_times_seconds=t,event_values=v,query_seconds=q)
        out,_,a=native_multiscale(cache,original)
        for values in out.values():np.testing.assert_array_equal(values,base)
        self.assertEqual(a['nonconverged_direction_global_fallbacks'],0)

if __name__=='__main__':unittest.main()
