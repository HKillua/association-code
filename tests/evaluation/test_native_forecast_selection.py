import unittest
import numpy as np
from association.evaluation.native_motion import predict_native_positions
from association.evaluation.native_forecast_selection import fragment_forecasts

def forecast(t,y,q):
    base=predict_native_positions(y[None],t[None],np.ones((1,len(t)),bool),q)['predictions']['native_ols'][0]
    return fragment_forecasts(t,y,q,base),base

class ForecastSelectionTests(unittest.TestCase):
    def test_constant_velocity_dateline_and_unchanged_inside(self):
        t=np.array([0.,2.,4.,7.,11.,16.,22.,29.,37.,46.]);y=np.column_stack((30+t*.0001,(179.99+t*.001+180)%360-180));q=np.array([-10.,2.,20.,60.])
        (out,a),base=forecast(t,y,q)
        np.testing.assert_allclose(out['cv'],base,atol=1e-10)
        for v in out.values():np.testing.assert_array_equal(v[1:3],base[1:3])
        self.assertGreater(a['validation_folds'],0)
        (shift,b),_=forecast(t+10000,y,q+10000)
        for k in out:np.testing.assert_allclose(shift[k],out[k],atol=1e-10)
    def test_two_observations_unknown_uses_global(self):
        (out,a),base=forecast(np.array([0.,10.]),np.array([[30.,120.],[30.001,120.002]]),np.array([-20.,30.]))
        np.testing.assert_array_equal(out['cv'],base);self.assertEqual(a['no_validation_queries'],2)
    def test_constant_source_translation_does_not_select_absolute_health(self):
        t=np.arange(12.)*10;y=np.column_stack((30+np.arange(12)*1e-5,120+np.arange(12)**2*1e-6));q=np.array([-20.,140.])
        (out,a),_=forecast(t,y,q);shift=y+np.array([0.,.1]);(other,b),_=forecast(t,shift,q)
        np.testing.assert_allclose(a['expert_mass'],b['expert_mass'])
        self.assertGreater(np.max(np.abs(other['cv']-out['cv'])),.09)

if __name__=='__main__':unittest.main()
