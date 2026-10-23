"""The scalar control must fit the original training objective, never V."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.support_gate import evaluate_support_gate
from association.evaluation.support_gate_replication import _fit
from association.models.support_gate import ConstantSupportGate, gate_inputs, support_gate_loss
from association.training.constant_gate import collect_training_points, search_constant, candidate_logits
from association.training.evidence_fusion import prepare_evidence_scenes
from association.training.support_gate import train_support_gate
from tests.training.test_support_gate import tiny_config, tiny_train
from tests.evaluation.test_support_gate import support_gate_cache_fixture


class ConstantGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_grid_matches_original_anchor_objective_with_missing_supervision(self):
        raw = tiny_train()
        # Partial and empty targets must retain each sampled anchor's original denominator.
        first = next(iter(raw[0]['fusion_targets'].values()))
        first['valid'][::2, :2] = False
        list(raw[0]['fusion_targets'].values())[-1]['valid'][:, :2] = False
        cfg = tiny_config(); cfg['position_loss'] = 'linear'
        with TemporaryDirectory() as directory:
            train_support_gate(raw, cfg, Path(directory)/'fit', record_objectives=True)
            record, rows, _, _, exposure = _fit(Path(directory)/'fit', 'linear')
            points = collect_training_points(raw, rows, record)
            self.assertEqual(points['sampled_anchors'], exposure['sampled_anchors'])
            self.assertEqual(points['valid_position_points'], exposure['valid_position_points'])
            logits = candidate_logits()[[0, 200, 500, 950, 1000]]
            fit = search_constant(points, logits=logits)
            prepared = {str(s['scene_id']): s for s in prepare_evidence_scenes(raw, motion_observation_policy='canonical_all')}
            for i, logit in enumerate(logits):
                losses = []
                model = ConstantSupportGate(logit)
                for row in rows:
                    for item in row['batch']:
                        scene = prepared[item['scene_id']]; node = item['anchor_index']
                        result = model(*gate_inputs(scene['motion'], item['members']))
                        losses.append(float(support_gate_loss(result, torch.from_numpy(scene['target'][node]),
                            torch.from_numpy(scene['valid'][node]), position_loss='linear')))
                self.assertAlmostEqual(fit['training_losses'][i], float(np.mean(losses)), places=10)
            self.assertEqual(fit['selected_index'], int(np.argmin(fit['training_losses'])))
            raw[0]['split'] = 'V_select'
            with self.assertRaisesRegex(ValueError, 'T_core'):
                collect_training_points(raw, rows, record)

    def test_scalar_finite_grid_finds_known_midpoint_and_exact_endpoints(self):
        anchor = torch.tensor([[0., 179., 1., 350.]], dtype=torch.float64)
        mean = torch.tensor([[0., -179., 4., 10.]], dtype=torch.float64)
        target = torch.tensor([[0., -180., 0., 0.]], dtype=torch.float64)
        points = dict(anchor=anchor, mean=mean, target=target, weight=torch.ones(1, dtype=torch.float64),
                      fixed_loss=2., sampled_anchors=1, valid_position_points=1, eligible_valid_position_points=1)
        result = search_constant(points)
        self.assertEqual(result['candidate_count'], 1001)
        self.assertEqual(result['selected_index'], 500)
        self.assertEqual(result['selected_gate'], .5)
        self.assertEqual(result['effective_gates'][0], 0.)
        self.assertEqual(result['effective_gates'][-1], 1.)
        self.assertAlmostEqual(result['selected_training_loss'], 2.)

    def test_complete_cpu_evaluation_preserves_groups_and_saved_gate_relation(self):
        for logit in (-100., 0., 100.):
            with self.subTest(logit=logit), TemporaryDirectory() as directory:
                root = Path(directory)
                source, metadata, validation = support_gate_cache_fixture(root)
                model = ConstantSupportGate(logit)
                self.assertEqual(len(list(model.parameters())), 0)
                with patch('association.evaluation.support_gate.frozen_motion', side_effect=lambda raw, *_: raw):
                    result = evaluate_support_gate(model, validation, source, metadata, root/'final')
                self.assertEqual(result['model_forwards'], 4)
                population = result['methods']['learned_gate']['population']
                self.assertEqual(population['duplicate_prediction_group_count'], 1)
                self.assertEqual(population['target_identity_count'], 5)
                with np.load(root/'final/predictions/states-3035.npz') as saved:
                    np.testing.assert_array_equal(saved['group_logits'], logit)
                    np.testing.assert_array_equal(saved['group_gate'], float(torch.sigmoid(model.logit)))


if __name__ == '__main__':
    unittest.main()
