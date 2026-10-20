"""CPU acceptance cases for the approved visible-support candidate revision.

These exercise physical behavior and information boundaries, not MTAD quality.
"""
import copy
from dataclasses import asdict
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from association.models.cooperative import CooperativeConfig, CooperativeModel
from association.data.cooperative import fit_cooperative_normalizer
from association.training.cooperative import endpoint_view
from association.models.fusion import FusionLoss
from tests.fixtures.scenes import timed_scene


ROUTES = ('plain', 'scalar-u', 'attribute-u', 'learned-attribute')


def revision_model(*, length=7, maximum=6, speed_output_policy='physical_softplus',
                   heading_output_policy='learned_absolute', num_layers=1,
                   decoder_norm_first=False, decoder_num_layers=None):
    config = CooperativeConfig(length=length, max_observations=maximum, d_model=8,
        nhead=2, num_layers=num_layers, dim_feedforward=16, dropout=0., head_hidden=6,
        model_variant='support_residual_v2', support_time_policy='train_absmax_v1',
        speed_output_policy=speed_output_policy, heading_output_policy=heading_output_policy,
        decoder_norm_first=decoder_norm_first, decoder_num_layers=decoder_num_layers)
    normalizer = fit_cooperative_normalizer([timed_scene('train', 'T_core')], config)
    # Simple physical scales make the geometry expectations independently exact.
    normalizer['mean'] = [0., 0., 109., 0.]
    normalizer['std'] = [1., 1., 415., 1.]
    torch.manual_seed(2026)
    return CooperativeModel(config, normalizer, support_time_scale=10.), normalizer


def packed(model, rows, queries=None):
    """Rows contain physical (time, latitude, longitude, speed, course) tokens."""
    m, q = model.config.max_observations, model.config.length
    queries = list(range(q)) if queries is None else queries
    values = torch.full((len(rows), m + q, 5), float('nan'))
    mask = torch.zeros(len(rows), m + q, dtype=torch.bool)
    for i, row in enumerate(rows):
        for j, (t, lat, lon, speed, course) in enumerate(row):
            values[i, j] = torch.tensor([lat, lon,
                (math.log1p(speed)-float(model.log_speed_mean))/float(model.log_speed_std),
                course, t])
            mask[i, j] = True
        values[i, m:, 4] = torch.tensor(queries)
    return values, mask


class CooperativeRevisionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_decoder_defaults_preserve_historical_configuration_and_weights(self):
        historical, normalizer = revision_model(length=3, num_layers=6,
            speed_output_policy='scaled_log', heading_output_policy='observation_residual')
        old_config = asdict(historical.config)
        del old_config['decoder_norm_first'], old_config['decoder_num_layers']
        restored = CooperativeModel(CooperativeConfig(**old_config), normalizer, support_time_scale=10.)
        explicit, _ = revision_model(length=3, num_layers=6, decoder_norm_first=False,
            decoder_num_layers=6, speed_output_policy='scaled_log',
            heading_output_policy='observation_residual')
        self.assertNotIn('decoder_semantics', historical.state_dict())
        self.assertEqual(historical.state_dict().keys(), explicit.state_dict().keys())
        for name, value in historical.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            historical.state_projection.weight.fill_(.025)
        restored.load_state_dict(historical.state_dict())
        explicit.load_state_dict(historical.state_dict())
        x, mask = packed(historical, [[(0, 10, 20, 3, 359), (2, 12, 21, 7, 1)]])
        for model in (historical, restored, explicit):
            model.eval()
        with torch.no_grad():
            expected = historical(x, mask)
            torch.testing.assert_close(restored(x, mask), expected, rtol=0, atol=0)
            torch.testing.assert_close(explicit(x, mask), expected, rtol=0, atol=0)

    def test_decoder_ablation_initialization_rng_and_actual_forward(self):
        options = dict(length=3, num_layers=6, speed_output_policy='scaled_log',
                       heading_output_policy='observation_residual')
        control, normalizer = revision_model(**options)
        control_rng = torch.get_rng_state().clone()
        before = dict(control.named_parameters())
        pre, _ = revision_model(**options, decoder_norm_first=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), control_rng))
        short, _ = revision_model(**options, decoder_num_layers=2)
        self.assertTrue(torch.equal(torch.get_rng_state(), control_rng))
        self.assertEqual(dict(pre.named_parameters()).keys(), before.keys())
        for model in (pre, short):
            self.assertEqual(len(model.encoder.layers), 6)
            self.assertTrue(all(not layer.norm_first for layer in model.encoder.layers))
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        self.assertEqual(len(pre.decoder.layers), 6)
        self.assertTrue(all(layer.norm_first for layer in pre.decoder.layers))
        self.assertIsNone(pre.decoder.norm)
        self.assertEqual(len(short.decoder.layers), 2)
        self.assertEqual(short.decoder.num_layers, 2)
        self.assertTrue(all(not layer.norm_first for layer in short.decoder.layers))
        self.assertFalse(any(name.startswith('decoder.layers.2.') for name, _ in short.named_parameters()))
        self.assertEqual(set(pre.state_dict()) - set(control.state_dict()), {'decoder_semantics'})
        x, mask = packed(control, [[(0, 10, 20, 3, 359), (2, 12, 21, 7, 1)]])
        for model in (control, pre, short):
            model.eval()
            with torch.no_grad():
                # Zero heads hide architecture differences; an identical,
                # nonzero head exposes the decoder's actual forward behavior.
                model.state_projection.weight.copy_(torch.arange(32).reshape(4, 8) * .001)
        with torch.no_grad():
            original = control(x, mask)
            for model in (pre, short):
                actual = model(x, mask)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertGreater(float((actual-original).abs().max()), 1e-4)
        for model in (pre, short):
            optimizer = torch.optim.SGD(model.parameters(), lr=.001)
            previous = model.decoder.layers[0].linear1.weight.detach().clone()
            optimizer.zero_grad()
            model(x, mask)[:, 0].sum().backward()
            optimizer.step()
            self.assertFalse(torch.equal(previous, model.decoder.layers[0].linear1.weight))
            restored = CooperativeModel(model.config, normalizer, support_time_scale=10.)
            restored.load_state_dict(model.state_dict())
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(x, mask), model(x, mask), rtol=0, atol=0)

    def test_decoder_semantics_cannot_be_cross_loaded_even_non_strictly(self):
        options = dict(length=3, num_layers=6)
        control, _ = revision_model(**options)
        pre, _ = revision_model(**options, decoder_norm_first=True)
        short, _ = revision_model(**options, decoder_num_layers=2)
        for strict in (True, False):
            for destination, source in ((control, pre), (pre, control), (short, control),
                                        (control, short), (pre, short), (short, pre)):
                with self.subTest(strict=strict, config=destination.config), \
                     self.assertRaisesRegex(ValueError, 'decoder_semantics'):
                    destination.load_state_dict(source.state_dict(), strict=strict)
        bad = copy.deepcopy(pre.state_dict())
        bad['decoder_semantics'][1] = 0
        with self.assertRaisesRegex(ValueError, 'decoder_semantics'):
            pre.load_state_dict(bad)

    def test_decoder_configuration_rejects_ambiguous_or_unsupported_values(self):
        defaults = CooperativeConfig()
        self.assertFalse(defaults.decoder_norm_first)
        self.assertIsNone(defaults.decoder_num_layers)
        v2 = dict(model_variant='support_residual_v2', support_time_policy='train_absmax_v1')
        for invalid in (None, 0, 1, 'true'):
            with self.subTest(norm=invalid), self.assertRaisesRegex(ValueError, 'must be a boolean'):
                CooperativeConfig(**v2, decoder_norm_first=invalid)
        for invalid in (True, False, 0, -1, 7, 2., '2'):
            with self.subTest(depth=invalid), self.assertRaisesRegex(ValueError, 'decoder_num_layers'):
                CooperativeConfig(**v2, decoder_num_layers=invalid)
        for options in ({'decoder_norm_first': True}, {'decoder_num_layers': 2}):
            with self.assertRaisesRegex(ValueError, 'require support_residual_v2'):
                CooperativeConfig(**options)

    def test_visible_base_interpolation_short_longitude_arc_and_endpoint_hold(self):
        model, normalizer = revision_model()
        model.eval()
        x, mask = packed(model, [[(0, 10, 179, 3, 0), (4, 18, -179, 4, 0),
                                  (10, 30, -173, 5, 0)]], [-2, 0, 2, 4, 7, 10, 12])
        expected = torch.tensor([[10, 179], [10, 179], [14, -180], [18, -179],
                                 [24, -176], [30, -173], [30, -173]], dtype=torch.float32)
        with torch.no_grad():
            result = model(x, mask)
        torch.testing.assert_close(result[:, :2], expected, rtol=0, atol=1e-5)
        typical = normalizer['cooperative_representation']['typical_speed']
        torch.testing.assert_close(result[:, 2], torch.full((7,), typical))
        with torch.no_grad():
            model.state_projection.bias[0] = .25
            model.state_projection.bias[1] = -.5
            changed = model(x, mask)
        torch.testing.assert_close(changed[:, 0], expected[:, 0] + .25)
        expected_lon = (expected[:, 1] - .5 + 180).remainder(360)-180
        torch.testing.assert_close(changed[:, 1], expected_lon)

    def test_exact_antipodal_longitude_tie_uses_negative_arc(self):
        model, _ = revision_model(length=3)
        x, mask = packed(model, [[(0, 10, 0, 0, 0), (2, 12, 180, 0, 0)]], [0, 1, 2])
        base = model.position_base(model.encode(x, mask), [0])[0]
        torch.testing.assert_close(base[:, 1], torch.tensor([0., -90., -180.]))

    def test_endpoint_mask_recomputes_base_and_nearest_support_without_hidden_leak(self):
        model, _ = revision_model(length=5)
        model.eval()
        row = [(0, 0, 0, 2, 10), (2, 20, 2, 2, 10),
               (4, 40, 4, 2, 10), (6, 60, 6, 2, 10)]
        x, mask = packed(model, [row], [-1, 0, 1, 3, 7])
        class Prefix:
            def integers(self, *args):
                return 1
        clipped, visible, removed = endpoint_view(x, mask, 6, Prefix(), fraction=.25)
        self.assertEqual(removed, [1])
        changed = clipped.clone()
        changed[:, :6][~visible[:, :6]] = float('nan')
        changed[:, 6:, :4] = float('inf')
        with torch.no_grad():
            result = model(clipped, visible)
            other = model(changed, visible)
        torch.testing.assert_close(result, other, rtol=0, atol=0)
        torch.testing.assert_close(result[:, 0], torch.tensor([20., 20., 20., 30., 60.]))
        encoded = model.encode(clipped, visible)
        support = model.support_features(encoded)[0]
        torch.testing.assert_close(support[2], torch.tensor([0., .1, 0., 1., 0.]))
        torch.testing.assert_close(support[3], torch.tensor([.1, .1, 1., 1., 1.]))
        torch.testing.assert_close(support[-1], torch.tensor([.1, 0., 1., 0., 0.]))
        # A real retained point, unlike a hidden value, must change the base.
        changed = clipped.clone(); changed[0, 0, 0] += 1
        with torch.no_grad():
            self.assertNotEqual(float(model(changed, visible)[0, 0]), float(result[0, 0]))

    def test_support_query_describes_internal_gap_not_only_first_last(self):
        model, _ = revision_model(length=4)
        x, mask = packed(model, [[(0, 10, 20, 1, 0), (2, 10, 20, 1, 0),
                                  (10, 10, 20, 1, 0)]], [-1, 2, 5, 12])
        expected = torch.tensor([[0, .1, 0, 1, 0], [0, 0, 1, 1, 1],
                                 [.3, .5, 1, 1, 1], [.2, 0, 1, 0, 0]])
        torch.testing.assert_close(model.support_features(model.encode(x, mask))[0], expected)

    def test_course_projection_is_circular_and_query_states_have_no_gradient(self):
        model, _ = revision_model(length=3)
        model.eval()
        rows = [[(0, 10, 20, 3, angle), (1, 11, 21, 4, angle)] for angle in (359, 1, 361)]
        x, mask = packed(model, rows)
        captured = []
        hook = model.input_projection.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        try:
            model.encode(x, mask)
        finally:
            hook.remove()
        features = captured[0]
        self.assertEqual(features.shape[-1], 5)
        torch.testing.assert_close(features[1, :2], features[2, :2], atol=1e-6, rtol=1e-6)
        distance = torch.linalg.vector_norm(features[0, 0, -2:]-features[1, 0, -2:])
        self.assertAlmostEqual(float(distance), 2*math.sin(math.pi/180), places=6)
        x = x.clone().requires_grad_()
        model(x[:1], mask[:1]).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(float(x.grad[:, 6:, :4].abs().sum()), 0.)
        self.assertEqual(float(x.grad[:, :6][~mask[:, :6]].abs().sum()), 0.)

    def test_observed_heading_short_arc_half_turns_and_cross_anchor(self):
        model, _ = revision_model(length=5, heading_output_policy='observation_residual')
        model.eval()
        rows = [[(0, 10, 20, 3, a), (2, 12, 21, 7, b)]
                for a, b in ((359, 1), (1, 359), (0, 180), (180, 0))]
        x, mask = packed(model, rows, [-1, 0, 1, 2, 3])
        expected = torch.tensor([[359., 359., 0., 1., 1.], [1., 1., 0., 359., 359.],
                                 [0., 0., 90., 180., 180.], [180., 180., 90., 0., 0.]])
        encoded = model.encode(x, mask)
        torch.testing.assert_close(model.heading_base(encoded, list(range(4))), expected, rtol=0, atol=0)
        for anchor, helper in ((0, 1), (1, 0), (2, 3)):
            for members in ([anchor], [anchor, helper]):
                result = model.decode_group(encoded, members)
                torch.testing.assert_close(result[:, 3], expected[anchor], rtol=0, atol=0)
        with torch.no_grad():
            model.state_projection.bias[3] = 3.
        prediction = model.decode_group(encoded, [1, 0])
        torch.testing.assert_close(prediction[:, 3], (expected[1]+3).remainder(360), rtol=0, atol=0)
        # The head can correct exact observed points; zero initialization does
        # not freeze the residual or impose zero correction at observations.
        truth = prediction.detach().clone()[None]
        truth[..., 3] = (truth[..., 3]+15).remainder(360)
        objective = FusionLoss(position_loss_mode='standard_smooth_l1')
        objective(prediction[None], truth, torch.ones_like(truth, dtype=torch.bool))['total'].backward()
        self.assertGreater(float(model.state_projection.weight.grad[3].abs().sum()), 0.)
        self.assertTrue(torch.isfinite(model.state_projection.weight.grad).all())
        # Real native inputs are standardized before their FP32 inverse. A
        # true signed half turn must not flip its arc from roundoff alone.
        with torch.no_grad():
            model.state_mean[3] = 123.456
            model.state_std[3] = 104.69449368349127
        rows = [[(0, 10, 20, 3, a), (2, 12, 21, 7, b)] for a, b in ((154, 334), (334, 154))]
        x, mask = packed(model, rows, [-1, 0, 1, 2, 3])
        x[:, :2, 3] = (x[:, :2, 3]-model.state_mean[3])/model.state_std[3]
        expected = torch.tensor([[154., 154., 244., 334., 334.], [334., 334., 244., 154., 154.]])
        torch.testing.assert_close(model.heading_base(model.encode(x, mask), [0, 1]), expected, rtol=0, atol=5e-5)

    def test_observed_heading_uses_clipped_native_values_without_hidden_leak(self):
        model, _ = revision_model(length=5, heading_output_policy='observation_residual')
        model.eval()
        row = [(0, 0, 0, 2, 300), (2, 20, 2, 2, 350),
               (4, 40, 4, 2, 10), (6, 60, 6, 2, 40)]
        x, mask = packed(model, [row, [(t, a, b, v, 100.) for t, a, b, v, _ in row]], [-1, 0, 1, 3, 7])
        class Prefix:
            def integers(self, *args):
                return 1
        clipped, visible, removed = endpoint_view(x, mask, 6, Prefix(), fraction=.25)
        self.assertEqual(removed, [1, 1])
        poisoned = clipped.clone()
        poisoned[:, :6][~visible[:, :6]] = float('nan')
        poisoned[:, 6:, :4] = float('inf')
        with torch.no_grad():
            expected = model(clipped, visible, [0, 1])
            actual = model(poisoned, visible, [0, 1])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(expected[:, 3], torch.tensor([350., 350., 350., 0., 40.]), rtol=0, atol=0)
        clipped.requires_grad_()
        model(clipped, visible, [0, 1]).sum().backward()
        self.assertTrue(torch.isfinite(clipped.grad).all())
        self.assertEqual(float(clipped.grad[:, 6:, :4].abs().sum()), 0.)
        self.assertEqual(float(clipped.grad[:, :6][~visible[:, :6]].abs().sum()), 0.)

    def test_heading_policy_preserves_other_initialization_and_forbids_cross_loading(self):
        original, _ = revision_model(length=3, speed_output_policy='scaled_log')
        rng = torch.get_rng_state().clone()
        revised, _ = revision_model(length=3, speed_output_policy='scaled_log',
                                    heading_output_policy='observation_residual')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for (name, before), (other_name, after) in zip(original.named_parameters(), revised.named_parameters()):
            self.assertEqual(name, other_name)
            if name in ('state_projection.weight', 'state_projection.bias'):
                torch.testing.assert_close(before[:3], after[:3], rtol=0, atol=0)
                self.assertEqual(float(after[3].detach().abs().sum()), 0.)
            else:
                torch.testing.assert_close(before, after, rtol=0, atol=0)
        original.eval(); revised.eval()
        x, mask = packed(original, [[(0, 10, 20, 3, 359), (2, 12, 21, 7, 1)],
                                    [(0, 11, 22, 5, 30), (2, 13, 24, 6, 40)]])
        with torch.no_grad():
            torch.testing.assert_close(original(x, mask)[:, :3], revised(x, mask)[:, :3], rtol=0, atol=0)
            torch.testing.assert_close(original.score_pairs(original.encode(x, mask), [[0, 1]]),
                                       revised.score_pairs(revised.encode(x, mask), [[0, 1]]), rtol=0, atol=0)
        self.assertEqual(set(revised.state_dict())-set(original.state_dict()), {'heading_base_version'})
        for strict in (True, False):
            with self.assertRaisesRegex(ValueError, 'heading_base_version'):
                revised.load_state_dict(original.state_dict(), strict=strict)
            with self.assertRaisesRegex(ValueError, 'heading_output_policy'):
                original.load_state_dict(revised.state_dict(), strict=strict)
        bad = copy.deepcopy(revised.state_dict()); bad['heading_base_version'] += 1
        with self.assertRaisesRegex(ValueError, 'heading_base_version'):
            revised.load_state_dict(bad)
        # Old serialized configurations have no heading policy and keep the
        # historical output, without a new state-dict key.
        self.assertEqual(CooperativeConfig().heading_output_policy, 'learned_absolute')
        with self.assertRaisesRegex(ValueError, 'requires support_residual_v2'):
            CooperativeConfig(heading_output_policy='observation_residual')

    def test_speed_output_does_not_use_native_std_and_loss_accepts_zero_and_tail(self):
        model, _ = revision_model(length=3)
        model.eval()
        x, mask = packed(model, [[(0, 10, 20, 0, 0), (2, 12, 22, 1e6, 10)]])
        with torch.no_grad():
            model.state_projection.bias[2] = 20.
        output = model(x, mask)
        expected = F.softplus(model.speed_bias + 20.)
        torch.testing.assert_close(output[:, 2], expected.expand(3))
        # Direct raw-speed Jacobian is <=1, irrespective of native std=415.
        gradient = torch.autograd.grad(output[:, 2].sum(), model.state_projection.bias,
                                       retain_graph=True)[0][2]
        self.assertGreater(float(gradient), 0.)
        self.assertLessEqual(float(gradient), 3.000001)
        target = output.detach().clone()[None]
        target[0, :, 2] = torch.tensor([0., 20., 1e6])
        loss = FusionLoss(position_loss_mode='standard_smooth_l1')(output[None], target, torch.ones_like(target, dtype=torch.bool))['total']
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.state_projection.weight.grad).all())
        self.assertGreater(float(model.state_projection.weight.grad[:3].abs().sum()), 0.)
        with torch.no_grad():
            model.state_projection.bias[2] = 1e6
            tail = model(x, mask)
        self.assertGreater(float(tail[:, 2].min()), 999999.)

    def test_scaled_log_preserves_initialization_rng_and_non_speed_outputs(self):
        original, _ = revision_model(length=3)
        original_rng = torch.get_rng_state().clone()
        revised, _ = revision_model(length=3, speed_output_policy='scaled_log')
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        before = dict(original.named_parameters())
        after = dict(revised.named_parameters())
        self.assertEqual(before.keys(), after.keys())
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        self.assertEqual(set(revised.state_dict()) - set(original.state_dict()), {'speed_reference'})
        original.eval(); revised.eval()
        x, mask = packed(original, [[(0, 10, 20, 3, 5), (2, 12, 21, 7, 15)]])
        with torch.no_grad():
            torch.testing.assert_close(original(x, mask), revised(x, mask), rtol=0, atol=0)
            for model in (original, revised):
                model.state_projection.bias.copy_(torch.tensor([.25, -.5, 1., .7]))
            old, new = original(x, mask), revised(x, mask)
        torch.testing.assert_close(old[:, [0, 1, 3]], new[:, [0, 1, 3]], rtol=0, atol=0)
        self.assertFalse(torch.equal(old[:, 2], new[:, 2]))

    def test_scaled_log_one_residual_unit_is_one_log_scale_with_physical_gradient(self):
        model, _ = revision_model(length=3, speed_output_policy='scaled_log')
        model.eval()
        x, mask = packed(model, [[(0, 10, 20, 3, 5), (2, 12, 21, 7, 15)]])
        with torch.no_grad():
            model.state_projection.bias[2] = 1.
        output = model(x, mask)
        expected = F.softplus(model.speed_bias) * model.log_speed_std.exp()
        torch.testing.assert_close(output[:, 2], expected.expand(3))
        derivative = torch.autograd.grad(output[:, 2].sum(), model.state_projection.bias)[0][2]
        torch.testing.assert_close(derivative, 3 * model.log_speed_std * expected)

    def test_speed_policy_state_cannot_be_cross_loaded_or_recalibrated(self):
        original, _ = revision_model(length=3)
        revised, _ = revision_model(length=3, speed_output_policy='scaled_log')
        with torch.no_grad():
            revised.state_projection.bias[2] = .75
        # Even non-strict loads must not silently reinterpret the head weights.
        for strict in (True, False):
            with self.subTest(strict=strict), self.assertRaisesRegex(ValueError, 'speed_reference'):
                revised.load_state_dict(original.state_dict(), strict=strict)
            with self.subTest(strict=strict), self.assertRaisesRegex(ValueError, 'speed_output_policy'):
                original.load_state_dict(revised.state_dict(), strict=strict)
        bad = copy.deepcopy(revised.state_dict()); bad['speed_reference'] += 1.
        with self.assertRaisesRegex(ValueError, 'speed_reference'):
            revised.load_state_dict(bad)
        restored, _ = revision_model(length=3, speed_output_policy='scaled_log')
        restored.load_state_dict(revised.state_dict())
        revised.eval(); restored.eval()
        x, mask = packed(revised, [[(0, 10, 20, 3, 5), (2, 12, 21, 7, 15)]])
        with torch.no_grad():
            torch.testing.assert_close(revised(x, mask), restored(x, mask), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'requires support_residual_v2'):
            CooperativeConfig(speed_output_policy='scaled_log')

    def test_scaled_log_overflow_is_reported_instead_of_clipped(self):
        model, _ = revision_model(length=3, speed_output_policy='scaled_log')
        model.eval()
        x, mask = packed(model, [[(0, 10, 20, 3, 5), (2, 12, 21, 7, 15)]])
        with torch.no_grad():
            model.state_projection.bias[2] = 1e6
            with self.assertRaisesRegex(ValueError, 'physical output is nonfinite'):
                model(x, mask)

    def test_local_identity_is_symmetric_at_tied_cutoff_and_nonoverlap_uses_endpoints(self):
        model, _ = revision_model(length=3)
        model.eval()
        rows = [[(t, 10+t, 20, 1, 0) for t in times] for times in
                ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9], [12, 13, 14])]
        encoded = model.encode(*packed(model, rows))
        pairs = torch.tensor([[0, 1], [0, 2], [2, 0]])
        forward, available = model.local_pair_features(encoded, pairs)
        reverse, reverse_available = model.local_pair_features(encoded, pairs.flip(1))
        self.assertTrue(available.all())
        torch.testing.assert_close(forward, reverse, rtol=0, atol=0)
        torch.testing.assert_close(available, reverse_available)
        torch.testing.assert_close(forward[1, -4:], torch.tensor([.4, 0., .25, .4]))
        # Perturb unselected encoded tokens: the nonoverlap local evidence is
        # specifically early-last/late-first, not an average of all endpoints.
        changed = {key:value.clone() for key,value in encoded.items()}
        changed['memory'][0, :4] += 100
        changed['memory'][2, 1:] -= 50
        after, _ = model.local_pair_features(changed, [[0, 2]])
        torch.testing.assert_close(after[0], forward[1], rtol=0, atol=0)
        with torch.no_grad():
            model.local_identity[-1].weight.fill_(.25)
        scores = model.score_pairs(encoded, pairs)
        torch.testing.assert_close(scores, model.score_pairs(encoded, pairs.flip(1)), rtol=0, atol=0)

    def test_overlap_without_real_tokens_falls_back_to_global_identity(self):
        model, _ = revision_model(length=3)
        encoded = model.encode(*packed(model, [[(0, 10, 20, 1, 0), (10, 10, 20, 1, 0)],
                                              [(4, 10, 20, 1, 0), (6, 10, 20, 1, 0)]]))
        _, available = model.local_pair_features(encoded, [[0, 1]])
        self.assertFalse(bool(available[0]))
        with patch.object(model.identity_base, 'forward', return_value=torch.tensor([[.75]])), \
             patch.object(model.local_identity, 'forward', return_value=torch.tensor([[100.]])):
            torch.testing.assert_close(model.score_pairs(encoded, [[0, 1]]), torch.tensor([.75]))

    def test_singleton_is_exact_same_self_path_for_every_route(self):
        model, _ = revision_model(length=3)
        model.eval()
        encoded = model.encode(*packed(model, [[(0, 10, 20, 1, 0), (2, 12, 21, 1, 0)]]))
        with patch.object(model, 'summarize', side_effect=AssertionError('no helper summary')), \
             patch.object(model, 'utility', side_effect=AssertionError('no helper utility')), \
             patch.object(model, 'score_pairs', side_effect=AssertionError('no helper identity')):
            outputs = [model.decode_group(encoded, [0], routing=route) for route in ROUTES]
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)

    def test_attribute_route_changes_only_its_attribute_while_scalar_uses_one_mean(self):
        model, _ = revision_model(length=3)
        model.eval()
        encoded = model.encode(*packed(model, [[(0, 10, 20, 1, 0), (2, 10, 20, 1, 0)],
                                              [(0, 11, 21, 1, 0), (2, 11, 21, 1, 0)]]))
        with torch.no_grad():
            model.fusion_projection.weight.copy_(torch.eye(8))
            model.state_projection.weight.zero_()
            model.state_projection.weight[:, :4].copy_(torch.eye(4))
            model.state_projection.bias.zero_()
        summaries = torch.zeros(2, 3, 8)
        summaries[1, :, :4] = torch.tensor([1., 2., 3., 4.])
        u = torch.tensor([0., 1., .2, .8]).expand(1, 3, 4)
        shifted = u.clone(); shifted[..., 0] = .8
        options = dict(identity_logits=torch.tensor([0.]))
        with patch.object(model.decoder, 'forward', return_value=torch.zeros(1, 3, 8)), \
             patch.object(model, 'summarize', return_value=summaries):
            before = model.decode_group(encoded, [0, 1], routing='attribute-u', utility=u, **options)
            after = model.decode_group(encoded, [0, 1], routing='attribute-u', utility=shifted, **options)
            scalar = model.decode_group(encoded, [0, 1], routing='scalar-u', utility=u, **options)
            permuted = model.decode_group(encoded, [0, 1], routing='scalar-u', utility=u.flip(-1), **options)
            plain = model.decode_group(encoded, [0, 1], routing='plain')
        torch.testing.assert_close(before[:, 1:], after[:, 1:], rtol=0, atol=0)
        self.assertGreater(float((after[:, 0]-before[:, 0]).abs().max().detach()), .1)
        torch.testing.assert_close(scalar, permuted, rtol=0, atol=0)
        torch.testing.assert_close(plain[:, 0], torch.full((3,), 10.5))
        torch.testing.assert_close(plain[:, 1], torch.full((3,), 21.))
        # Same identity probability and affine floor in ordinary learned route.
        with patch.object(model.ordinary_gate, 'forward', return_value=torch.zeros(1, 3, 4)):
            learned = model.attribute_route_weights(encoded, [0, 1], 'learned-attribute',
                identity_logits=torch.zeros(1), summaries=summaries)
        torch.testing.assert_close(learned[1], torch.full((3, 4), .001 + .999*.25))

    def test_declared_v2_statistics_and_checkpoint_buffers_cannot_silently_disagree(self):
        model, normalizer = revision_model()
        for key in ('log_speed_mean', 'log_speed_std', 'speed_bias', 'state_mean', 'state_std'):
            bad = copy.deepcopy(model.state_dict()); bad[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                model.load_state_dict(bad)
        bad = copy.deepcopy(normalizer)
        del bad['cooperative_representation']['speed_bias']
        with self.assertRaises(ValueError):
            CooperativeModel(model.config, bad, support_time_scale=10.)
        with self.assertRaises(ValueError):
            CooperativeConfig(model_variant='support_residual_v2')
        legacy = CooperativeConfig()
        with self.assertRaises(ValueError):
            CooperativeModel(legacy, normalizer)
