"""Cross-layer physical input, native TCM and complete development execution."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.tmtaff import evaluate_complete
from association.data.archive import DataError
from association.models.fusion import PairFusion, PairFusionConfig
from association.models.tmtaff import ModelConfig
from association.training.common import fit_normalizer, normalize_scene
from association.data.timed import prepare_timed_scene
from association.models.timed import TimedAssociationModel


from tests.fixtures.scenes import native_scene, timed_scene

class TimedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(19)
        self.model_config = ModelConfig(length=4, d_model=8, nhead=2, num_layers=1,
                                        dim_feedforward=16, dropout=0.)
        self.fusion_config = PairFusionConfig(length=4, d_model=8, nhead=2, num_layers=1,
                                              dim_feedforward=16, dropout=0., fc_hidden=8)
        self.model = TimedAssociationModel(self.model_config, max_observations=3).eval()
        self.fusion = PairFusion(self.fusion_config).eval()
        self.train = [timed_scene('train', 'T_core')]
        self.validation = timed_scene('validation', 'V_select')
        self.normalizer = fit_normalizer(self.train)

    def evaluate(self, example):
        return evaluate_complete(self.model, self.fusion, self.normalizer, [example], device='cpu',
                                 batch_tracks=2, pair_batch_size=3, select_on_validation=False,
                                 threshold=-1e6, singleton_policy='observed')

    def test_real_native_short_track_reaches_tcm_tam_tfm_and_full_population(self):
        example = self.validation
        self.assertFalse(example['observed'][0].any())
        self.assertEqual(example['native_observed'][0].sum(), 2)
        x, mask = normalize_scene(example, self.normalizer)
        self.assertEqual(x.shape, (4, 7, 5))
        self.assertFalse(mask[:, 3:].any())
        self.assertEqual(mask[0].sum(), 2)
        calls = {'encoder': 0, 'tam': 0, 'tfm': 0}
        def hook(key):
            def count(*args):
                calls[key] += 1
            return count
        handles = [module.register_forward_hook(hook(key)) for key, module in
                   [('encoder', self.model.tcm_encoder), ('tam', self.model.tam_encoder), ('tfm', self.fusion)]]
        try:
            result = self.evaluate(example)
        finally:
            for handle in handles:
                handle.remove()
        self.assertGreater(calls['encoder'], 0)
        self.assertEqual(calls['tam'], 2)
        self.assertEqual(calls['tfm'], 5)
        self.assertEqual(len(result['records'][0]['scores']), 6)
        np.testing.assert_equal(result['records'][0]['pairs'], example['pairs'])
        np.testing.assert_equal(result['records'][0]['labels'], example['labels'])
        self.assertEqual(result['fusion']['oracle']['all']['position']['available_target_count'], 8)
        self.assertEqual(result['fusion']['oracle']['all']['position']['intersection_count'], 8)
        self.assertEqual(result['fusion']['simple_fusion_oracle']['missing']['position']['intersection_count'], 0)
        self.assertFalse(result['test_opened'])

    def test_joint_time_shift_preserves_packed_input_and_model_outputs(self):
        shifted = timed_scene('validation', 'V_select', shift=1_000_000.)
        x, mask = normalize_scene(self.validation, self.normalizer)
        y, shifted_mask = normalize_scene(shifted, self.normalizer)
        np.testing.assert_equal(x, y)
        np.testing.assert_equal(mask, shifted_mask)
        with torch.no_grad():
            expected = self.model.complete(torch.from_numpy(x), torch.from_numpy(mask))
            actual = self.model.complete(torch.from_numpy(y), torch.from_numpy(shifted_mask))
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_reference_and_outside_query_values_do_not_change_predictions(self):
        changed = copy.deepcopy(self.validation)
        changed['values'][~changed['observed']] = np.inf
        for target in changed['fusion_targets'].values():
            target['states'][:, 0] += .1
            target['states'][:, 2] += 10
        x, mask = normalize_scene(self.validation, self.normalizer)
        y, changed_mask = normalize_scene(changed, self.normalizer)
        np.testing.assert_equal(x, y)
        np.testing.assert_equal(mask, changed_mask)
        captured = []
        handle = self.fusion.register_forward_hook(
            lambda module, args, output: captured.append(output['states'].detach().clone()))
        try:
            first = self.evaluate(self.validation); before = captured[:]; captured.clear()
            second = self.evaluate(changed); after = captured[:]
        finally:
            handle.remove()
        for left, right in zip(before, after):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(len(before), 5)
        np.testing.assert_equal(first['records'][0]['scores'], second['records'][0]['scores'])
        self.assertNotEqual(first['fusion']['oracle']['all']['position']['mae'],
                            second['fusion']['oracle']['all']['position']['mae'])

    def test_each_rollout_slot_keeps_its_own_next_time_in_both_directions(self):
        original = torch.randn(1, 3, 8)
        cases = [(torch.tensor([[2., 5., 9.]]), torch.tensor([[12., 20.]]),
                  [[5., 9., 12.], [5., 9., 12., 20.]]),
                 (torch.tensor([[9., 5., 2.]]), torch.tensor([[0., -4.]]),
                  [[5., 2., 0.], [5., 2., 0., -4.]])]
        for times, queries, expected_times in cases:
            calls, predictions = [], []
            def capture(module, args, output):
                calls.append((args[0].detach().clone(), args[1]))
                predictions.append(output[:, -1:].detach().clone())
            handle = self.model.tcm_decoder.register_forward_hook(capture)
            try:
                with patch.object(self.model.tcm_encoder, 'forward', wraps=self.model.tcm_encoder.forward) as encoder:
                    self.model._extrapolate(original, times, queries)
                    self.assertEqual(encoder.call_count, 1)
            finally:
                handle.remove()
            for step, (conditioned, memory) in enumerate(calls):
                history = torch.cat([original, *predictions[:step]], dim=1)
                timestamp = torch.tensor(expected_times[step])
                independent = torch.empty(len(timestamp), 8)
                for pair in range(4):
                    angle = timestamp * math.pow(10000., -2 * pair / 8)
                    independent[:, 2 * pair] = angle.sin()
                    independent[:, 2 * pair + 1] = angle.cos()
                torch.testing.assert_close(conditioned - history, independent[None], rtol=1e-5, atol=1e-6)
                self.assertEqual(memory.data_ptr(), calls[0][1].data_ptr())

    def test_support_rounding_cannot_silently_change_known_query_regions(self):
        native = native_scene('collision', 'V_select')
        times = ([.50000001, .6], [0., .25, 1.], [.1, .3, .9], [.1, .4, .8])
        for track, stamps in zip(native['tracks'], times):
            track['canonical_times'] = np.asarray(stamps)
        native['intervals'] = np.asarray([[x[0], x[-1]] for x in times])
        example = prepare_timed_scene(native, length=3, max_observations=3, time_unit_seconds=1.)
        training = copy.deepcopy(example)
        training.update(scene_id='collision-train', split='T_core')
        normalizer = fit_normalizer([training])
        self.assertFalse(example['observed'][0, 1])
        with self.assertRaises(DataError):
            normalize_scene(example, normalizer)


if __name__ == '__main__':
    unittest.main()
