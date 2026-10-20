"""Diagnostic hooks must observe the original target-free, frozen self forward."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.data.cooperative import fit_cooperative_normalizer, normalize_cooperative_scene
from association.evaluation.cooperative_dynamics import evaluate_cooperative_dynamics
from association.models.cooperative import CooperativeConfig, CooperativeModel
from tests.fixtures.scenes import empty_timed_scene, timed_scene


class _NoReferenceScene(dict):
    def __getitem__(self, key):
        if key in ('fusion_targets', 'reference'):
            raise AssertionError('dynamics must not access reference')
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in ('fusion_targets', 'reference'):
            raise AssertionError('dynamics must not access reference')
        return super().get(key, default)


class CooperativeDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(91)
        self.scene = _NoReferenceScene(timed_scene('diagnostic', 'V_select'))
        config = CooperativeConfig(length=4, max_observations=3, d_model=8, nhead=2,
            num_layers=2, dim_feedforward=16, dropout=.1, head_hidden=8,
            model_variant='support_residual_v2', support_time_policy='train_absmax_v1',
            speed_output_policy='scaled_log', heading_output_policy='observation_residual')
        self.normalizer = fit_cooperative_normalizer([timed_scene('train', 'T_core')], config)
        self.model = CooperativeModel(config, self.normalizer, support_time_scale=10.)
        with torch.no_grad():
            self.model.state_projection.weight.normal_(0., .08)
            self.model.state_projection.bias.copy_(torch.tensor([.1, -.2, .3, -.4]))

    def test_one_forward_hooks_reconstruct_and_preserve_state_without_reference(self):
        self.model.train()
        self.model.encoder.eval()  # Mixed modes must survive, too.
        next(self.model.parameters()).requires_grad_(False)
        modes = [module.training for module in self.model.modules()]
        flags = [parameter.requires_grad for parameter in self.model.parameters()]
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        original_decode = self.model.decode_group
        observed_outputs = []
        def decode(encoded, members, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertEqual(members, [0])
            self.assertFalse(any(module.training for module in self.model.modules()))
            result = original_decode(encoded, members, **kwargs)
            observed_outputs.append(result.detach().clone())
            return result
        with patch.object(self.model, 'encode', wraps=self.model.encode) as encode, \
             patch.object(self.model, 'decode_group', side_effect=decode) as decoding, \
             patch.object(self.model, 'score_pairs', side_effect=AssertionError('identity forbidden')), \
             patch.object(self.model, 'utility', side_effect=AssertionError('helper forbidden')):
            report = evaluate_cooperative_dynamics(self.model, self.normalizer,
                [self.scene, _NoReferenceScene(empty_timed_scene())])
        self.assertEqual((encode.call_count, decoding.call_count), (4, 4))
        self.assertEqual(report['fragment_count'], 4)
        self.assertEqual(report['per_scene'][1]['fragment_count'], 0)
        self.assertEqual(report['decoder_layer_count'], 2)
        self.assertEqual([row['native_count'] for row in report['per_fragment']], [2, 3, 3, 3])
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertEqual(flags, [parameter.requires_grad for parameter in self.model.parameters()])
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertFalse(self.model.decoder._forward_pre_hooks)
        self.assertFalse(self.model.decoder._forward_hooks)
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.decoder.layers))
        # Hooks must observe the same predictions as the ordinary inference path.
        self.model.eval().requires_grad_(False)
        values, observed = normalize_cooperative_scene(self.scene, self.normalizer, self.model.config)
        with torch.inference_mode():
            for node, hooked in enumerate(observed_outputs):
                encoded = self.model.encode(torch.as_tensor(values[node:node+1]),
                                            torch.as_tensor(observed[node:node+1]))
                ordinary = self.model.decode_group(encoded, [0], routing='plain')
                torch.testing.assert_close(ordinary, hooked, rtol=0, atol=0)
        for row, prediction in zip(report['per_fragment'], observed_outputs):
            metrics = row['metrics']
            self.assertEqual(row['before_support_query_count']+row['inside_support_query_count']
                             +row['after_support_query_count'], row['query_count'])
            self.assertGreater(metrics['query_pe_centered_rms'], 0.)
            self.assertGreater(metrics['support_projection_centered_rms'], 0.)
            self.assertGreater(metrics['decoder_layer_1_centered_rms'], 0.)
            self.assertGreater(metrics['decoder_layer_2_centered_rms'], 0.)
            self.assertGreater(metrics['speed_residual_std'], 0.)
            self.assertEqual(metrics['physical_reconstruction_max_abs'], 0.)
            self.assertAlmostEqual(metrics['output_speed_rms'],
                                   prediction.double().cpu().numpy()[:, 2].std(), places=12)
            for attribute in ('latitude', 'longitude', 'speed', 'heading'):
                ratio = metrics[f'{attribute}_head_normalized_dynamic_projection']
                self.assertGreaterEqual(ratio, 0.)
                self.assertLessEqual(ratio, 1.+1e-12)
        json.dumps(report, allow_nan=False)

    def test_uses_saved_cooperative_normalizer_and_rejects_mismatch(self):
        inputs = []
        original_encode = self.model.encode
        def capture(values, observed):
            inputs.append(values.detach().cpu().numpy().copy())
            return original_encode(values, observed)
        with patch.object(self.model, 'encode', side_effect=capture):
            evaluate_cooperative_dynamics(self.model, self.normalizer, [self.scene])
        expected, _ = normalize_cooperative_scene(self.scene, self.normalizer, self.model.config)
        np.testing.assert_array_equal(np.concatenate(inputs), expected)
        mismatch = copy.deepcopy(self.normalizer)
        mismatch['std'][0] *= 2.
        with self.assertRaisesRegex(ValueError, 'normalizer differs'):
            evaluate_cooperative_dynamics(self.model, mismatch, [self.scene])
        mismatch = copy.deepcopy(self.normalizer)
        mismatch['cooperative_representation']['log_speed_std'] *= 2.
        with self.assertRaisesRegex(ValueError, 'representation differs'):
            evaluate_cooperative_dynamics(self.model, mismatch, [self.scene])

    def test_exception_cleans_hooks_and_modes_and_zero_head_has_null_ratio(self):
        modes = [module.training for module in self.model.modules()]
        with patch.object(self.model, 'decode_group', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                evaluate_cooperative_dynamics(self.model, self.normalizer, [self.scene])
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertFalse(self.model.decoder._forward_pre_hooks)
        self.assertFalse(self.model.decoder._forward_hooks)
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.decoder.layers))
        with torch.no_grad():
            self.model.state_projection.weight.zero_()
        report = evaluate_cooperative_dynamics(self.model, self.normalizer, [self.scene])
        for row in report['per_fragment']:
            for attribute in ('latitude', 'longitude', 'speed', 'heading'):
                self.assertIsNone(row['metrics'][f'{attribute}_head_normalized_dynamic_projection'])
                self.assertEqual(row['metrics'][f'{attribute}_residual_std'], 0.)
        self.assertEqual(report['aggregate']['speed_head_normalized_dynamic_projection']['count'], 0)

    def test_wrap_safe_output_amplitudes(self):
        from association.evaluation.cooperative_dynamics import _physical_rms, _physical_difference
        states = torch.tensor([[0., 179., 4., 359.], [0., -179., 6., 1.]])
        result = _physical_rms(states)
        self.assertAlmostEqual(result['longitude_circular_rms_deg'], 1.)
        self.assertAlmostEqual(result['heading_circular_rms_deg'], 1.)
        equivalent = states.clone()
        equivalent[:, 1] += 360.
        equivalent[:, 3] += 360.
        difference = _physical_difference(states, equivalent)
        self.assertEqual(difference['heading_rms'], 0.)
        self.assertEqual(difference['longitude_rms'], 0.)
        self.assertEqual(difference['position_rms_metres'], 0.)
