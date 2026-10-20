"""Visible-only motion base: geometry, cropping and checkpoint semantics."""
import copy
from dataclasses import asdict
import unittest

import numpy as np
import torch

from association.evaluation.native_motion import predict_native_positions
from association.models.cooperative import CooperativeConfig, CooperativeModel
from association.training.cooperative import endpoint_view
from tests.models.test_cooperative_revision_model import packed, revision_model


class CooperativeMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def model(self, **kwargs):
        model, norm = revision_model(position_base_policy='native_ols',
            speed_output_policy='scaled_log', heading_output_policy='observation_residual', **kwargs)
        return model.eval(), norm

    def assert_numpy_control(self, model, values, mask):
        with torch.no_grad():
            encoded = model.encode(values, mask)
            actual = model.position_base(encoded, list(range(len(values))))
        # Independent float64 formula, using precisely the model-visible FP32
        # coordinates/times. Original native FP64 precision is not recoverable.
        expected = predict_native_positions(encoded['native_positions'].numpy(),
            encoded['times'].numpy(), encoded['observed'].numpy(),
            encoded['queries'][0].numpy())['predictions']['native_ols']
        difference = actual.numpy().astype(np.float64)-expected
        difference[..., 1] = (difference[..., 1]+180.) % 360.-180.
        np.testing.assert_allclose(difference, 0., rtol=0, atol=6e-5)
        return actual

    def test_zero_residual_matches_motion_control_for_short_dateline_and_noisy_tracks(self):
        model, _ = self.model()
        values, mask = packed(model, [
            [(0, 10, 179, 4, 359), (4, 14, -177, 5, 1)],
            [(0, 20, 30, 4, 40), (1, 21.25, 30.5, 4, 40),
             (3, 22.75, 32, 4, 40), (4, 24, 32.5, 4, 40)],
        ], [-3., 0., 1., 2., 3., 4., 7.])
        base = self.assert_numpy_control(model, values, mask)
        with torch.no_grad():
            for i in range(len(values)):
                prediction = model(values[i:i+1], mask[i:i+1])
                torch.testing.assert_close(prediction[:, :2], base[i], rtol=0, atol=0)
            encoded = model.encode(values, mask)
            # Unchanged support interpolation and all unaffected learned parts.
            hold, _ = revision_model(speed_output_policy='scaled_log',
                                    heading_output_policy='observation_residual')
            inside = (encoded['queries'] >= 0) & (encoded['queries'] <= 4)
            torch.testing.assert_close(hold.position_base(encoded, [0, 1])[inside], base[inside], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(base).all())
        # The new native OLS policy keeps the raw sign of exact half turns
        # both inside support and in the extrapolation slope. Old hold keeps
        # its historical negative-half-turn convention.
        half, half_mask = packed(model, [[(0, 0, 0, 4, 40), (2, 0, 180, 4, 40)]],
                                [-1., 0., .5, 1., 1.5, 2., 3.])
        half_base = self.assert_numpy_control(model, half, half_mask)
        self.assertAlmostEqual(float(half_base[0, 3, 1]), 90., places=4)
        with torch.no_grad():
            half_encoded = model.encode(half, half_mask)
            self.assertAlmostEqual(float(hold.position_base(half_encoded, [0])[0, 3, 1]), -90., places=4)

    def test_endpoint_crop_recomputes_slope_and_hidden_tokens_cannot_supply_a_trend(self):
        class CropPrefix:
            def integers(self, *args):
                return 1

        model, _ = self.model()
        rows = [[(0, 20, 30, 4, 40), (1, 21, 31, 4, 40), (2, 22, 32, 4, 40),
                 (3, 23, 33, 4, 40), (4, 23.1, 33.1, 4, 40), (5, 23.2, 33.2, 4, 40)]]
        x, mask = packed(model, rows, [-2., 0., 2., 3., 4., 5., 8.])
        full = self.assert_numpy_control(model, x, mask)
        cropped, visible, removed = endpoint_view(x, mask, 6, CropPrefix(), fraction=.5)
        self.assertEqual(removed, [3])
        result = self.assert_numpy_control(model, cropped, visible)
        self.assertGreater(float((full-result).abs().max()), 1.)
        # Poison all unavailable native states/times and every query state;
        # they must neither alter the base nor the complete model prediction.
        poisoned = cropped.clone()
        poisoned[:, :6][~visible[:, :6]] = float('nan')
        poisoned[:, 6:, :4] = float('nan')
        with torch.no_grad():
            torch.testing.assert_close(model(poisoned, visible), model(cropped, visible), rtol=0, atol=0)
        # Interior holes are allowed at the model boundary, even though the
        # training crop normally repacks them into a visible prefix.
        holes, holemask = torch.full_like(cropped, float('nan')), torch.zeros_like(visible)
        holes[:, [0, 2, 5]] = cropped[:, :3]
        holemask[:, [0, 2, 5]] = True
        holes[:, 6:, 4] = cropped[:, 6:, 4]
        with torch.no_grad():
            encoded = model.encode(holes, holemask)
            torch.testing.assert_close(model.position_base(encoded, [0]), result, rtol=0, atol=0)

    def test_time_units_and_offset_leave_motion_prediction_unchanged(self):
        model, _ = self.model()
        x, mask = packed(model, [[(0, 10, 179, 4, 359), (1, 10.1, 179.5, 4, 1),
                                (3, 10.4, -179.75, 4, 1), (4, 10.5, -179, 4, 1)]],
                         [-3., 0., 1., 2., 3., 4., 7.])
        original = self.assert_numpy_control(model, x, mask)
        converted = x.clone()
        converted[..., 4] = converted[..., 4]*20.+400.
        changed = self.assert_numpy_control(model, converted, mask)
        torch.testing.assert_close(changed, original, rtol=0, atol=3e-5)

    def test_policy_is_explicit_preserves_old_weights_and_rejects_cross_loading(self):
        ols, norm = self.model()
        config = asdict(ols.config)
        del config['position_base_policy']
        historical = CooperativeModel(CooperativeConfig(**config), norm, support_time_scale=10.)
        hold, _ = revision_model(speed_output_policy='scaled_log', heading_output_policy='observation_residual')
        self.assertEqual(historical.config.position_base_policy, 'native_hold')
        self.assertNotIn('position_base_version', historical.state_dict())
        historical.load_state_dict(hold.state_dict())
        self.assertEqual(set(ols.state_dict())-set(hold.state_dict()), {'position_base_version'})
        for name, value in hold.state_dict().items():
            torch.testing.assert_close(value, ols.state_dict()[name], rtol=0, atol=0)
        for strict in (True, False):
            for destination, source in ((ols, hold), (hold, ols)):
                with self.subTest(strict=strict, target=destination.config.position_base_policy):
                    with self.assertRaisesRegex(ValueError, 'position_base'):
                        destination.load_state_dict(source.state_dict(), strict=strict)
        wrong = copy.deepcopy(ols.state_dict())
        wrong['position_base_version'].fill_(99)
        with self.assertRaisesRegex(ValueError, 'position_base_version'):
            ols.load_state_dict(wrong)
        with self.assertRaisesRegex(ValueError, 'requires support_residual_v2'):
            CooperativeConfig(position_base_policy='native_ols')
