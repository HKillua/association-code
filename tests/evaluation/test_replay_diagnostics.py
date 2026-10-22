"""063 evidence must identify failed elements and preserve the failed forward."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.data.motion_evidence import scene_motion_evidence
from association.evaluation.evidence_fusion_combo import predict_partition
from association.evaluation.replay_diagnostics import array_difference, diagnose_replay
from association.models.evidence_fusion import EvidenceFusion, fusion_inputs
from tests.fixtures.scenes import timed_scene


class ArrayDifferenceTests(unittest.TestCase):
    def test_largest_absolute_difference_is_not_the_tolerance_failure(self):
        previous = np.array([1000., 0., 0.])
        current = np.array([1000.001, 1e-5, -2e-7])
        result = array_difference(current, previous, atol=1e-6, rtol=1e-5)
        self.assertFalse(result['passed'])
        self.assertEqual(result['exceed_count'], 1)
        self.assertEqual(result['max_absolute_difference']['index'], [0])
        failure = result['worst_tolerance_ratio']
        self.assertEqual(failure['index'], [1])
        self.assertEqual(failure['previous'], 0.)
        self.assertEqual(failure['current'], 1e-5)
        self.assertEqual(failure['tolerance_bound'], 1e-6)
        self.assertAlmostEqual(failure['tolerance_ratio'], 10.)
        self.assertEqual(result['first_violations'][0]['index'], [1])
        json.dumps(result, allow_nan=False)

    def test_zero_bound_near_zero_and_exact_boundary(self):
        equal = array_difference(np.array([0.]), np.array([0.]), atol=0., rtol=0.)
        self.assertTrue(equal['passed'])
        self.assertEqual(equal['exceed_count'], 0)
        unequal = array_difference(np.array([1e-30]), np.array([0.]), atol=0., rtol=0.)
        self.assertFalse(unequal['passed'])
        self.assertEqual(unequal['exceed_count'], 1)
        worst = unequal['worst_tolerance_ratio']
        self.assertEqual(worst['index'], [0])
        self.assertEqual(worst['ratio_kind'], 'zero_bound_nonzero_difference')
        self.assertIsNone(worst['tolerance_ratio'])
        json.dumps(unequal, allow_nan=False)
        boundary = array_difference(np.array([.125, .12500001]), np.zeros(2),
                                    atol=.125, rtol=0.)
        self.assertEqual(boundary['exceed_count'], 1)
        self.assertEqual(boundary['first_violations'][0]['index'], [1])

    def test_nonfinite_and_shape_mismatch_are_json_safe_failures(self):
        current = np.array([np.nan, np.inf, -np.inf, 0.])
        previous = np.array([np.nan, np.inf, 0., 0.])
        result = array_difference(current, previous, atol=1e-6, rtol=1e-5)
        self.assertFalse(result['passed'])
        self.assertEqual(result['current_nonfinite_count'], 3)
        self.assertEqual(result['previous_nonfinite_count'], 2)
        self.assertEqual(result['exceed_count'], 3)
        self.assertEqual(result['first_violations'][0]['current_kind'], 'nan')
        self.assertEqual(result['first_violations'][1]['current_kind'], 'positive_infinity')
        self.assertEqual(result['first_violations'][2]['current_kind'], 'negative_infinity')
        json.dumps(result, allow_nan=False)
        mismatch = array_difference(np.zeros((2, 1)), np.zeros(2), atol=1e-6, rtol=1e-5)
        self.assertFalse(mismatch['passed'])
        self.assertFalse(mismatch['shape_equal'])
        self.assertEqual(mismatch['current_shape'], [2, 1])
        self.assertEqual(mismatch['previous_shape'], [2])
        self.assertIsNone(mismatch['exceed_count'])
        json.dumps(mismatch, allow_nan=False)


class ReplaySnapshotTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)
        example = timed_scene('diagnostic', 'V_select')
        self.motion = scene_motion_evidence(example, motion_observation_policy='canonical_all')
        # Rotating the anchor yields calls [1, 0, 2] and [2, 0, 1]; preserve
        # this real helper order as well as the singleton forward.
        self.groups = [[0, 1, 2], [3]]
        self.model = EvidenceFusion(utility_output_bias=.01).eval()
        with torch.no_grad():
            self.model.utility[-1].weight.normal_(0., .01)
        self.prediction, self.routing = predict_partition(self.model, self.motion, self.groups, 'cpu')
        self.old = {'ols_' + key: value.copy() for key, value in self.prediction.items()}
        self.old.update({'ols_routing_' + key: value.copy() for key, value in self.routing.items()})
        # A saved near-zero utility differs enough to fail. The snapshot must
        # retain both original arrays, never overwrite them with the repeat.
        self.old['ols_routing_utility'][0, 0, 0] = 0.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def hook_counts(model):
        return [(len(module._forward_hooks), len(module._forward_pre_hooks))
                for module in model.modules()]

    def test_snapshot_preserves_first_forward_and_one_trace_repeats_actual_helper_order(self):
        original = deepcopy((self.motion, self.prediction, self.routing))
        parameters = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
        hooks = self.hook_counts(self.model)
        with patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
            report = diagnose_replay(self.model, self.motion, self.groups, self.prediction,
                                     self.routing, self.old, self.root)
        self.assertEqual(forward.call_count, 4)  # One additional partition, no retry sweep.
        self.assertTrue(report['repeat_vs_first']['utility']['passed'])
        self.assertFalse(report['first_vs_old']['utility']['passed'])
        self.assertEqual(self.hook_counts(self.model), hooks)
        self.assertFalse(self.model.training)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, parameters[key]))
        for before, after in zip(original, (self.motion, self.prediction, self.routing)):
            self.assertEqual(set(before), set(after))
            for key in before:
                np.testing.assert_array_equal(after[key], before[key])
        with np.load(self.root/'first-forward.npz', allow_pickle=False) as first:
            for key, value in self.motion.items():
                np.testing.assert_array_equal(first['motion_' + key], value)
            for prefix, original_values in (('prediction_', self.prediction), ('routing_', self.routing)):
                for key, value in original_values.items():
                    np.testing.assert_array_equal(first[prefix + key], value)
            np.testing.assert_array_equal(first['old_ols_routing_utility'], self.old['ols_routing_utility'])
        calls = [[anchor] + [node for node in group if node != anchor]
                 for group in self.groups for anchor in group]
        with np.load(self.root/'trace-second-forward.npz', allow_pickle=False) as trace:
            utility_rows = []
            helpers, anchors = [], []
            with torch.inference_mode():
                for call, members in enumerate(calls):
                    prefix = f'call_{call:05d}_'
                    states, features, exact = fusion_inputs(self.motion, members, device='cpu')
                    np.testing.assert_array_equal(trace[prefix + 'members'], members)
                    for key, value in (('states', states), ('features', features), ('exact_observation', exact)):
                        np.testing.assert_array_equal(trace[prefix + key], value.numpy())
                    layer_value = features
                    for index, layer in enumerate(self.model.utility):
                        layer_value = layer(layer_value)
                        np.testing.assert_array_equal(trace[prefix + f'utility_layer_{index}'], layer_value.numpy())
                    np.testing.assert_array_equal(trace[prefix + 'utility'], layer_value.numpy())
                    utility_rows.extend(trace[prefix + 'utility'])
                    anchors.extend([members[0]] * (len(members) - 1))
                    helpers.extend(members[1:])
            np.testing.assert_array_equal(anchors, self.routing['helper_anchor'])
            np.testing.assert_array_equal(helpers, self.routing['helper_node'])
            np.testing.assert_array_equal(np.asarray(utility_rows), self.routing['utility'])
            for prefix, original_values in (('repeat_prediction_', self.prediction), ('repeat_routing_', self.routing)):
                for key, value in original_values.items():
                    np.testing.assert_array_equal(trace[prefix + key], value)
        json.dumps(report, allow_nan=False)

    def test_trace_failure_preserves_first_pass_and_removes_only_its_own_hooks(self):
        observed = []
        handle = self.model.utility[0].register_forward_hook(
            lambda module, args, output: observed.append(output.shape))
        self.addCleanup(handle.remove)
        hooks = self.hook_counts(self.model)
        with patch.object(self.model.utility[2], 'forward', side_effect=RuntimeError('trace aborted')):
            with self.assertRaisesRegex(RuntimeError, 'trace aborted'):
                diagnose_replay(self.model, self.motion, self.groups, self.prediction,
                                self.routing, self.old, self.root)
        self.assertEqual(len(observed), 1)
        self.assertEqual(self.hook_counts(self.model), hooks)
        with np.load(self.root/'first-forward.npz', allow_pickle=False) as first:
            np.testing.assert_array_equal(first['prediction_learned'], self.prediction['learned'])
            np.testing.assert_array_equal(first['routing_utility'], self.routing['utility'])
        with np.load(self.root/'trace-second-forward.npz', allow_pickle=False) as trace:
            self.assertIn('call_00000_utility_layer_0', trace.files)

    def test_probe_checks_missing_reliability_features_without_more_network_calls(self):
        probe = deepcopy(self.motion)
        probe['native_count'][1] += 1
        with patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
            report = diagnose_replay(self.model, self.motion, self.groups, self.prediction,
                                    self.routing, self.old, self.root, probe_motion=probe)
        self.assertEqual(forward.call_count, 4)
        comparison = report['probe_input_comparison']
        self.assertEqual(comparison['status'], 'compared')
        self.assertTrue(comparison['fields']['states']['passed'])
        self.assertFalse(comparison['fields']['native_count']['passed'])
        self.assertEqual(len(comparison['inputs']), 4)
        for row in comparison['inputs'][:3]:
            self.assertTrue(row['states']['passed'])
            self.assertTrue(row['exact_observation']['passed'])
            self.assertFalse(row['features']['passed'])
        self.assertTrue(comparison['inputs'][3]['features']['passed'])
        json.dumps(report, allow_nan=False)

    def test_broken_deferred_probe_preserves_the_unhooked_first_forward(self):
        def broken_probe():
            self.assertTrue((self.root/'first-forward.npz').exists())
            raise ValueError('corrupt optional probe')
        hooks = self.hook_counts(self.model)
        with patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
            with self.assertRaisesRegex(ValueError, 'corrupt optional probe'):
                diagnose_replay(self.model, self.motion, self.groups, self.prediction,
                                self.routing, self.old, self.root, probe_motion=broken_probe)
        self.assertEqual(forward.call_count, 0)
        self.assertEqual(self.hook_counts(self.model), hooks)
        self.assertFalse((self.root/'trace-second-forward.npz').exists())
        with np.load(self.root/'first-forward.npz', allow_pickle=False) as first:
            np.testing.assert_array_equal(first['routing_utility'], self.routing['utility'])
            np.testing.assert_array_equal(first['motion_native_count'], self.motion['native_count'])

    def test_reference_fields_are_rejected_and_existing_evidence_is_never_overwritten(self):
        contaminated = dict(self.motion, target=np.zeros((4, 4, 4)))
        with patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
            with self.assertRaisesRegex(ValueError, 'visible motion'):
                diagnose_replay(self.model, contaminated, self.groups, self.prediction,
                                self.routing, self.old, self.root)
        self.assertEqual(forward.call_count, 0)
        self.assertEqual(list(self.root.iterdir()), [])
        diagnose_replay(self.model, self.motion, self.groups, self.prediction,
                        self.routing, self.old, self.root)
        original = {p.name: p.read_bytes() for p in self.root.iterdir()}
        with patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
            with self.assertRaises(FileExistsError):
                diagnose_replay(self.model, self.motion, self.groups, self.prediction,
                                self.routing, self.old, self.root)
        self.assertEqual(forward.call_count, 0)
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, original)


if __name__ == '__main__':
    unittest.main()
