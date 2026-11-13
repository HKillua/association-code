import unittest
import numpy as np
from association.evaluation.native_sampling import velocity_measures
from tools.analysis.sampling_measure_qualification import join_conditions,cohorts

class SamplingMeasureTests(unittest.TestCase):
    def data(self,t,speed):
        v=np.zeros((len(t),4));v[:,2]=np.asarray(speed)*3600/1852;return np.asarray(t,float),v,np.ones(v.shape,bool)
    def test_linear_refinement_and_time_units(self):
        a=velocity_measures(*self.data([0,1,10],[0,1,10]));b=velocity_measures(*self.data([0,.5,1,10],[0,.5,1,10]));c=velocity_measures(*self.data([0,3600,36000],[0,1,10]));np.testing.assert_allclose(a['time_mean'],[5,0]);np.testing.assert_allclose(a['time_mean'],b['time_mean']);np.testing.assert_allclose(a['time_mean'],c['time_mean']);self.assertNotAlmostEqual(a['count_mean'][0],b['count_mean'][0])
    def test_constant_two_points_and_missing(self):
        x=self.data([2,5,9],[3,3,3]);a=velocity_measures(*x);np.testing.assert_allclose(a['count_mean'],a['time_mean']);b=velocity_measures(*self.data([0,10],[2,4]));np.testing.assert_allclose(b['count_mean'],b['time_mean']);x[2][0,2:]=False;x[2][1,2:]=False;self.assertFalse(velocity_measures(*x)['known'])
        with self.assertRaises(ValueError):velocity_measures(*self.data([0,0],[1,2]))
    def test_current_frozen_selection_join(self):
        model=velocity_measures(*self.data([0,1,10],[0,1,10]));model['times']=np.array([0,10]);m=dict(ols_velocity_mps=np.array([[2.,0]]),query_seconds=np.array([20.]));row=dict(group_index=0,query_index=0,scene_id='x',pure=True,before=False,PG_delta_m=1001.,group_nodes=1,mean_speed_mps=11/3,report_spread_mps=model['stats']['report_spread_mps'],native_count=3,span_seconds=10,mean_ols_gap_mps=5/3)
        rows=join_conditions(m,[model],[row],dict(decisions=[dict(group_index=0,query_index=0,selected_nodes=[0],before=False)]),{'HO__weights':np.ones((1,1))},'HARD');self.assertAlmostEqual(rows[0]['weighted_node_gap_times_absdt_m'],40/3);self.assertEqual(cohorts(rows)['PG_harm']['distinct_group_count'],1)
