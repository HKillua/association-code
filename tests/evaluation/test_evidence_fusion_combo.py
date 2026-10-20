import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from association.evaluation.evidence_fusion_combo import (
    aligned_motion, evaluate_combo, predict_partition, read_saved_predictions,
    replay_check, routing_diagnostics, saved_groups)
from association.models.cooperative import CooperativeConfig
from association.models.evidence_fusion import EvidenceFusion
from association.training.evidence_fusion import prepare_evidence_scenes, evaluate_evidence_fusion
from tools.experiments.cooperative_fit import synthetic_scenes


class FrozenEvidenceComboTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        _, validation = synthetic_scenes(CooperativeConfig(length=64, max_observations=4))
        self.scene = prepare_evidence_scenes(validation)[0]
        self.scene.update(target_identity_ids=list(dict.fromkeys(self.scene['identities'])),
                          observed=validation[0]['observed'])
        self.model = EvidenceFusion()
        with torch.no_grad():
            self.model.utility[-1].weight.normal_(0., 0.01)
            self.model.utility[-1].bias.fill_(0.5)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        evaluate_evidence_fusion(self.model, [self.scene], device='cpu', prediction_dir=root/'old')
        self.saved = read_saved_predictions(root/'old')[self.scene['scene_id']]
        self.scores = dict(node_ids=self.scene['node_ids'], sources=self.scene['sources'],
            pairs=np.column_stack(np.triu_indices(4, 1)), scores=np.array([-1., 1., -1., -1., -1., -1.]))

    def test_full_frozen_two_by_two_preserves_population_and_saved_membership(self):
        sid = self.scene['scene_id']
        # Freeze a membership different from what any current OLS threshold might
        # reconstruct. The evaluation must reproduce this exact saved partition.
        groups = [[0], [1, 2], [3]]
        values, routing = predict_partition(self.model, self.scene['motion'], groups, 'cpu')
        for name, value in values.items():
            self.saved['predicted_anchor_direct_'+name] = value
        for key in ('membership', 'anchor_weights'):
            self.saved['predicted_anchor_direct_'+key] = routing[key]
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        output = Path(self.temp.name)/'combo'
        report = evaluate_combo(self.model, [self.scene], {sid: self.saved}, {sid: self.scores},
                                0., output=output)
        self.assertEqual(report['fragment_count'], 4)
        self.assertEqual(report['query_count'], 256)
        self.assertTrue(report['replay_passed'])
        self.assertEqual(report['optimizer_updates'], 0)
        self.assertFalse(report['test_opened'])
        with np.load(output/'predictions'/('states-'+sid+'.npz')) as z:
            np.testing.assert_array_equal(z['ols046_membership'], [0, 1, 1, 2])
            np.testing.assert_array_equal(z['assoc045_membership'], [0, 1, 0, 2])
            self.assertEqual(z['ols046_utility'].shape, (2, 64, 3))
            a, h = z['ols046_helper_anchor'], z['ols046_helper_node']
            self.assertEqual(set(zip(a, h)), {(1, 2), (2, 1)})
            np.testing.assert_allclose(z['ols046_anchor_weights'][a]+z['ols046_helper_weights'], 1.)
            regions = [z['region_old_'+old+'_new_'+new] for old in ('pure', 'mixed') for new in ('pure', 'mixed')]
            np.testing.assert_array_equal(np.sum(regions, axis=0), np.ones((4, 64)))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertTrue(self.model.training)
        system = json.loads((output/'system-evaluation.json').read_text())
        self.assertEqual(system['target_identity_count'], 2)
        self.assertEqual(set(system['overall']), set(report['overall']['all']['methods']))
        common_counts = [system['common_valid_target_points'][arm]['all']['position']['intersection_count']
                         for arm in system['overall']]
        self.assertEqual(len(set(common_counts)), 1)
        self.assertTrue((output/'system-predictions'/('states-'+sid+'.npz')).exists())

    def test_node_order_and_replay_drift_fail_without_silent_repair(self):
        bad = copy.deepcopy(self.scores)
        bad['node_ids'] = bad['node_ids'][::-1]
        with self.assertRaisesRegex(ValueError, 'input order'):
            aligned_motion(self.scene, self.saved, bad)
        groups = saved_groups(self.saved['predicted_anchor_direct_membership'])
        values, routing = predict_partition(self.model, self.scene['motion'], groups, 'cpu')
        self.assertTrue(replay_check(self.saved, values, routing)['passed'])
        values['learned'][0, 0, 0] += 0.01
        self.assertFalse(replay_check(self.saved, values, routing)['passed'])

    def test_wrong_helper_utility_diagnostic_uses_anchor_truth(self):
        # A helper equals its OWN target but is harmful to the anchor. It must
        # be classified as harmful, not helpful from its own-identity error.
        q = 64
        states = np.zeros((2, q, 4))
        states[0, :, :2] = [20., 120.]
        states[1, :, :2] = [21., 121.]
        saved = dict(native_states=states, target=states.copy(), target_valid=np.ones_like(states, bool),
                     identities=np.array(['A', 'B']))
        routing = dict(helper_anchor=np.array([0]), helper_node=np.array([1]),
            utility=np.ones((1, q, 3)), helper_weights=np.full((1, q, 3), 0.5),
            anchor_weights=np.full((2, q, 3), 0.5))
        report = routing_diagnostics(saved, routing)
        self.assertEqual(report['position']['helpful_queries'], 0)
        self.assertEqual(report['position']['nonhelpful_accepted'], q)
        self.assertEqual(report['position']['wrong_identity_helpful'], 0)


if __name__ == '__main__':
    unittest.main()
