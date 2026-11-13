import unittest

import numpy as np

from association.evaluation import native_temporal as p


def cache(times, values, query):
    return dict(node_ids=np.array(['a']), event_offsets=np.array([0,len(times)]),
                event_times_seconds=np.asarray(times,float),event_values=np.asarray(values,float),
                query_seconds=np.asarray(query,float))


class TemporalEvidenceTests(unittest.TestCase):
    def test_exact_line_and_endpoint_noise_multiplier(self):
        t=np.arange(8,dtype=float)*10
        values=np.column_stack((20+t*.00001,179.99+t*.0001,np.ones(8)*5,np.ones(8)*90))
        result=p.temporal_evidence(cache(t,values,[-10.,80.]))
        np.testing.assert_allclose(result['node_features'],0,atol=1e-7)
        self.assertTrue(result['scale_known'][0])
        noisy=values.copy();noisy[3,0]+=.01
        result=p.temporal_evidence(cache(t,noisy,[-10.,80.]))
        tc=t-t.mean();weights=tc/(tc@tc)
        explicit=np.sum((np.eye(8)[-1]+10*weights)**2)
        y=(noisy[:,0]-noisy[0,0])*p.RADIUS*np.pi/180
        residual=y-y.mean()-tc*(tc@y)/(tc@tc)
        expected=(residual@residual)/(len(t)-2)*explicit
        self.assertAlmostEqual(result['predictive_scale_m2'][0,1],expected,places=5)
        self.assertGreater(result['node_features'][0,0],0)
        shifted=p.temporal_evidence(cache(t+1000,noisy,[-10+1000,80+1000]))
        np.testing.assert_array_equal(result['node_features'],shifted['node_features'])

    def test_zero_only_source_rename_permutation_and_unknown_fallback(self):
        states=np.array([[[20.,179.9,5.,90.]]*3,[[20.,-179.9,6.,90.]]*3,[[20.,179.8,7.,90.]]*3])
        motion=dict(states=states,support=np.array([[False,True,False]]*3),
            nearest_seconds=np.array([[30.,0.,30.],[10.,0.,10.],[20.,0.,20.]]),sources=np.array(['x','y','x']))
        base=states[[0]].copy()
        evidence=dict(predictive_scale_m2=np.array([[30.,0.,30.],[10.,0.,10.],[20.,0.,20.]]),scale_known=np.ones(3,bool))
        result,_=p.unsupported_candidates(motion,[[0,1,2]],base,evidence,'scale')
        np.testing.assert_array_equal(result[0,[0,2],:2],states[1,[0,2],:2])
        np.testing.assert_array_equal(result[...,2:],base[...,2:])
        np.testing.assert_array_equal(result[:,1],base[:,1])
        permutation=np.array([2,1,0])
        perm={k:v[permutation] for k,v in motion.items()}
        perm['sources']=np.array(['renamed-a','renamed-b','renamed-a'])
        e={k:v[permutation] for k,v in evidence.items()}
        value,_=p.unsupported_candidates(perm,[[0,1,2]],base,e,'scale')
        np.testing.assert_allclose(value,result,atol=1e-10)
        evidence['scale_known'][0]=False
        value,audit=p.unsupported_candidates(motion,[[0,1,2]],base,evidence,'scale')
        np.testing.assert_array_equal(value,result)
        self.assertEqual(audit['scale_unknown_nearest_fallback_queries'],2)
        two=p.temporal_evidence(cache([0,1],[[20,120,5,0],[20.001,120,5,0]],[-1,2]))
        self.assertFalse(two['scale_known'][0])

    def test_extended_ridge_dimension_and_no_target_at_prediction(self):
        rng=np.random.default_rng(20);x=rng.normal(size=(250,14));y=x[:,0]-2*x[:,1]
        model=p.fit_quadratic(x,y,np.ones(250))
        self.assertEqual(model['beta'].shape,(120,))
        self.assertTrue(np.isfinite(p.predict(model,x)).all())
