"""074 harm magnitude accounting and complete cached population checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.sequence_position import evaluate_sequence_position
from association.models.sequence_position import SequencePosition
from tests.fixtures.cache import support_gate_cache_fixture
from tools.analysis.compare_position_harm import harm_statistics, scene_pair, scientific_screen, summarize, verify_loss_pair


class HarmComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_real_paired_training_sampling_and_initial_checkpoint_binding(self):
        import json
        from copy import deepcopy
        from association.training.sequence_position import train_sequence_position
        from tests.training.test_sequence_position import tiny_configuration, tiny_training
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_configuration('transformer')
            train_sequence_position(tiny_training(), config, root/'linear')
            changed = deepcopy(config)
            changed.update(scope='sequence-position-relative-harm-v1', position_loss='linear_relative_harm')
            train_sequence_position(tiny_training(), changed, root/'harm', paired_reference_fit=root/'linear')
            result = verify_loss_pair(root/'harm', root/'linear')
            self.assertEqual(result['matched_updates'], 3)
            self.assertEqual(result['matched_anchors'], 6)
            checkpoint_path = root/'harm/step-0.pt'
            original = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            tampered = deepcopy(original)
            tampered['loss_semantics'] = 'old-linear-semantics'
            torch.save(tampered, checkpoint_path)
            with self.assertRaisesRegex(ValueError, 'contract'):
                verify_loss_pair(root/'harm', root/'linear')
            torch.save(original, checkpoint_path)
            rng_before = torch.get_rng_state().clone()
            verify_loss_pair(root/'harm', root/'linear')
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            path = root/'harm/training.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['batch'][0]['members'].append(999)
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'sampling'):
                verify_loss_pair(root/'harm', root/'linear')

    def test_signed_magnitude_severe_thresholds_and_squared_error_accounting(self):
        baseline = np.array([1., 100000., 2., 5., 4.])
        current = np.array([2., 90000., 102., 1006., 20005.])
        result = harm_statistics(current, baseline)
        self.assertEqual(result['improvement_sum'], 10000.)
        self.assertEqual(result['harm_sum'], 21103.)
        self.assertEqual(result['net_identity_residual_m'], 0.)
        self.assertEqual(result['severe_harm']['100']['count'], 2)  # Strict >, not >=.
        self.assertEqual(result['severe_harm']['1000']['count'], 2)
        self.assertEqual(result['severe_harm']['10000']['count'], 1)
        self.assertEqual(result['squared_error_change_sum_m2'], float(np.square(current).sum()-np.square(baseline).sum()))
        self.assertIsNone(harm_statistics([], [])['mean_error_change'])
        with self.assertRaises(ValueError):
            harm_statistics([1., np.nan], [1., 2.])

    def test_complete_sequence_initial_and_final_cache_comparison_and_tamper_rejection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, metadata, validation = support_gate_cache_fixture(root, sparse=True)
            arms = {name: dict(path=root/name) for name in ('linear', 'harm')}
            for name in arms:
                model = SequencePosition(backbone='transformer')
                with patch('association.evaluation.sequence_position.frozen_motion', side_effect=lambda raw, *_: raw):
                    evaluate_sequence_position(model, validation, source, metadata, root/name/'initial', initial=True)
                    with torch.no_grad():
                        model.output.bias[0] = .1 if name == 'linear' else .2
                    evaluate_sequence_position(model, validation, source, metadata, root/name/'final')
            row, arrays = scene_pair(arms, '3035')
            row['component_id'] = 'fixture'
            result = summarize([row], [arrays])
            self.assertEqual(result['population']['predicted_group_count'], 4)
            self.assertEqual(result['population']['duplicate_prediction_group_count'], 1)
            self.assertEqual(result['position_population']['covered_target_points'], 1)
            self.assertEqual(result['relative_harm']['all']['harm']['common_point_count'], 1)
            self.assertEqual(result['relative_harm']['no_predicted_group']['harm']['common_point_count'], 0)
            path = root/'harm/initial/system-predictions/states-3035.npz'
            with np.load(path, allow_pickle=False) as data:
                bad = dict(data)
            bad['learned_sequence_group_states'][0, 0, 0] += 1.
            np.savez_compressed(path, **bad)
            with self.assertRaisesRegex(ValueError, 'step0 equals'):
                scene_pair(arms, '3035')

    def test_direction_screen_does_not_promote_noop_or_tiny_numerical_signs(self):
        def summary(mae, rmse, p99, harm):
            return dict(position_weightings={'all': {
                'linear': {'point_equal': dict(mae=10., rmse=20., p95=30., p99=40., unit_count=100)},
                'harm': {'point_equal': dict(mae=mae, rmse=rmse, p95=30., p99=p99, unit_count=100)}}},
                relative_harm={'all': {'linear': dict(harm_sum=100.), 'harm': dict(harm_sum=harm)}})
        self.assertFalse(scientific_screen(summary(10., 20., 40., 100.))['direction_screen_passed'])
        self.assertFalse(scientific_screen(summary(10., 20.-1e-9, 40., 100.-1e-7))['direction_screen_passed'])
        self.assertTrue(scientific_screen(summary(9., 19., 39., 90.))['direction_screen_passed'])
        self.assertFalse(scientific_screen(summary(11., 19., 39., 90.))['direction_screen_passed'])


if __name__ == '__main__':
    unittest.main()
