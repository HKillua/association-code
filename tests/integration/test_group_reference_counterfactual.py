import unittest
import numpy as np
import torch
from association.evaluation.local_source_evidence import ResidualScore
from tools.analysis import group_reference_counterfactual as p
from tests.integration.test_source_risk_pilot import scene_arrays

class GroupReferenceTests(unittest.TestCase):
    def inputs(self,groups,identities=None):
        torch.set_num_threads(2);m,ev,_,_=scene_arrays('1')
        if identities is not None:ev=dict(ev,identities=np.array(identities))
        pack=p.risk.build_source_features(**{k:m[k] for k in p.source.MOTION_KEYS},groups=groups);base=dict(mean=np.zeros(10),std=np.ones(10),beta=np.zeros(66));net=dict(base=base,mean=np.zeros(10),std=np.ones(10),dimension=10,weights=ResidualScore(10).state_dict());scores={n:np.zeros(pack['available'].shape) for n in p.NAMES}
        return m,ev,pack,dict(quadratic=base,network=net),scores
    def test_reference_alternative_and_input_purification_are_separate(self):
        groups=[[0,1,2,3]];m,ev,pack,models,scores=self.inputs(groups);a,checks=p.scene_counterfactual(m,groups,pack,ev,models,scores)
        self.assertGreater(a['best_flips'].sum(),0);self.assertGreater(a['clean_multi'].sum(),0);self.assertEqual(checks['cpu_original_choice_changes'],0)
    def test_loss_of_multiple_support_is_not_counted_as_rescue(self):
        groups=[[0,1,2,3]];m,ev,pack,models,scores=self.inputs(groups,['t0','t1','t0','t1']);a,checks=p.scene_counterfactual(m,groups,pack,ev,models,scores);summary=p.summarize_arrays(a,np.ones_like(a['eligible']))
        self.assertEqual(summary['purification_multi'],0);self.assertEqual(summary['purification_one'],summary['queries']);self.assertEqual(summary['methods']['network']['common_rescued'],0)
    def test_pure_groups_remain_identical(self):
        groups=[[0,1],[2],[3]];m,ev,pack,models,scores=self.inputs(groups);a,checks=p.scene_counterfactual(m,groups,pack,ev,models,scores)
        np.testing.assert_array_equal(a['network_wrong'],a['network_clean_wrong']);self.assertEqual(a['best_flips'].sum(),0);self.assertGreater(checks['pure_group_feature_checks'],0)

if __name__=='__main__':unittest.main()
