import unittest
import numpy as np
from tools.analysis.source_coverage_pilot import fit_coverage,role_diagnostic
from association.evaluation.source_risk import fit_score_models,build_source_features,source_labels
from tests.integration.test_source_risk_pilot import scene_arrays

class CoverageTests(unittest.TestCase):
    def test_same_quadratic_and_component_exclusion(self):
        rng=np.random.default_rng(20);x=rng.normal(size=(200,10));y=rng.random(200);w=rng.random(200)+.1
        rows=[dict(component_id='a',x=x[:100],y=y[:100],weight=w[:100]),dict(component_id='b',x=x[100:],y=y[100:],weight=w[100:])]
        expected=fit_score_models(x,y,w)['QUADRATIC'];actual=fit_coverage(rows,None)
        for k in ('mean','std','beta'):np.testing.assert_allclose(actual[k],expected[k],rtol=1e-10,atol=1e-12)
        held=fit_coverage(rows,'a');rows[0]['y']=rows[0]['y']+10000
        np.testing.assert_array_equal(held['beta'],fit_coverage(rows,'a')['beta'])
    def test_role_counts_use_identical_eligible_population(self):
        m,ev,_,_=scene_arrays('1');groups=[[0,1],[2],[3]]
        keys=('states','support','nearest_seconds','sources','native_count','span_seconds','fit_rmse_m','ols_velocity_mps');pack=build_source_features(**{k:m[k] for k in keys},groups=groups)
        labels=source_labels(pack,groups,ev['identities'],ev['target_ids'],ev['target_states'],ev['target_valid']);d=role_diagnostic(pack,labels,np.zeros(pack['available'].shape))
        self.assertEqual(sum(v['points'] for v in d.values()),labels['audit']['labeled_group_query_count'])

if __name__=='__main__':unittest.main()
