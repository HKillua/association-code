import io
import math
import unittest

import torch

from association.models.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.models.tmtaff import AssociationModel, ModelConfig


class PairFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20)
        self.config = PairFusionConfig(fc_hidden=12, length=6, d_model=8,
                                       nhead=2, num_layers=2,
                                       dim_feedforward=16, dropout=0)

    def test_independent_encoders_and_depths_and_native_output(self):
        model = PairFusion(self.config).eval()
        left_ids = {id(p) for p in model.left_encoder.parameters()}
        self.assertTrue(left_ids.isdisjoint(id(p) for p in model.right_encoder.parameters()))
        left = model.left_encoder.layers
        right = model.right_encoder.layers
        self.assertFalse(torch.equal(left[0].self_attn.in_proj_weight, right[0].self_attn.in_proj_weight))
        self.assertFalse(torch.equal(left[0].self_attn.in_proj_weight, left[1].self_attn.in_proj_weight))
        self.assertFalse(torch.equal(model.decoder.layers[0].self_attn.in_proj_weight,
                                     model.decoder.layers[1].self_attn.in_proj_weight))
        values = torch.randn(2, 6, 8)
        output = model(values, values + 1)
        self.assertEqual(tuple(output["states"].shape), (2, 6, 4))
        self.assertEqual(tuple(output["fused_features"].shape), (2, 6, 8))
        self.assertTrue(all(torch.isfinite(v).all() for v in output.values()))
        self.assertTrue((output["states"][..., 2] >= 0).all())
        with torch.no_grad():
            model.state_projection.weight.zero_()
            model.state_projection.bias.copy_(torch.tensor([123., 200., 0., 720.]))
        states = model(values, values)["states"]
        expected = torch.tensor([123., 200., math.log(2), 720.]).expand_as(states)
        torch.testing.assert_close(states, expected)

    def test_decoder_receives_only_bos_then_own_latent_and_causal_mask(self):
        model = PairFusion(self.config).eval()
        calls, outputs = [], []

        def capture_inputs(module, args, kwargs):
            calls.append((args[0].detach().clone(), args[1].detach().clone(),
                          kwargs["tgt_mask"].clone(), kwargs.get("memory_mask")))

        def capture_outputs(module, args, output):
            outputs.append(output[:, -1:].detach().clone())

        before = model.decoder.register_forward_pre_hook(capture_inputs, with_kwargs=True)
        after = model.decoder.register_forward_hook(capture_outputs)
        values = torch.randn(2, 6, 8)
        try:
            actual = model(values, values + 1)
        finally:
            before.remove()
            after.remove()
        self.assertEqual(len(calls), 6)
        for step, (history, memory, mask, cross_mask) in enumerate(calls):
            expected_history = torch.cat([model.bos.expand(2, -1, -1).detach()] + outputs[:step], dim=1)
            torch.testing.assert_close(history, expected_history, rtol=0, atol=0)
            torch.testing.assert_close(memory, actual["fused_features"], rtol=0, atol=0)
            self.assertTrue(torch.equal(mask, torch.ones(step + 1, step + 1, dtype=torch.bool).triu(1)))
            self.assertIsNone(cross_mask)
        with self.assertRaises(TypeError):
            model(values, values, target=torch.zeros(2, 6, 4))

    def test_eval_roundtrip_and_pair_gather(self):
        model = PairFusion(self.config).eval()
        tracks = torch.randn(3, 6, 8)
        pairs = torch.tensor([[0, 2], [1, 0]])
        expected = model(tracks[pairs[:, 0]], tracks[pairs[:, 1]])
        for key, value in model.forward_pairs(tracks, pairs).items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        clone = PairFusion(self.config).eval()
        clone.load_state_dict(torch.load(buffer, weights_only=True))
        for key, value in clone.forward_pairs(tracks, pairs).items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        empty = model.forward_pairs(tracks, torch.empty((0, 2), dtype=torch.long))
        self.assertEqual(tuple(empty["states"].shape), (0, 6, 4))
        self.assertEqual(tuple(empty["fused_features"].shape), (0, 6, 8))


    def test_missing_endpoint_fusion_gradient_and_joint_three_losses_step(self):
        for branch in ("ce", "contrastive"):
            with self.subTest(branch=branch):
                torch.manual_seed(20)
                self._assert_joint_three_losses_step(branch)

    def _assert_joint_three_losses_step(self, branch):
        association = AssociationModel(ModelConfig(length=6, d_model=8, nhead=2,
                                       num_layers=1, dim_feedforward=16, dropout=0), branch=branch)
        fusion = PairFusion(self.config)
        objective = FusionLoss(position_loss_mode="standard_smooth_l1")
        values = torch.randn(3, 6, 4)
        observed = torch.tensor([[False, True, True, True, False, False],
                                 [True, True, True, False, False, False],
                                 [False, False, True, True, True, True]])
        values[~observed] = float("nan")
        completed = association.complete(values, observed)
        # Only the same-identity pair is given reference supervision. The third
        # fragment supplies negative TAM pairs, not a false fusion target.
        states = fusion.forward_pairs(completed, torch.tensor([[0, 1]]))["states"]
        target = torch.tensor([30., 120., 5., 359.]).expand_as(states).clone()
        fusion_total = objective(states, target, torch.ones_like(target, dtype=torch.bool))["total"]
        tcm_parameters = tuple(association.tcm_encoder.parameters()) + tuple(association.tcm_decoder.parameters())
        gradients = torch.autograd.grad(fusion_total, (completed,) + tcm_parameters, retain_graph=True)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(gradients[0][:2][~observed[:2]].abs().sum().item(), 0)
        encoder_count = len(tuple(association.tcm_encoder.parameters()))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients[1:1 + encoder_count]), 0)
        self.assertGreater(sum(g.abs().sum().item() for g in gradients[1 + encoder_count:]), 0)
        # A separate TAM objective is required; fusion does not traverse TAM.
        tam_gradients = torch.autograd.grad(fusion_total, tuple(association.tam_encoder.parameters()),
                                            retain_graph=True, allow_unused=True)
        self.assertTrue(all(g is None for g in tam_gradients))
        encoded = association.tam_encoder(completed).flatten(start_dim=1)
        scores = association.score_pairs(encoded, torch.tensor([[0, 1], [0, 2], [1, 2]]))
        association_total = association.association_loss(scores, torch.tensor([1., 0., 0.]))
        completion_total = association.completion_loss(values, observed)
        optimizer = torch.optim.Adam(list(association.parameters()) + list(fusion.parameters())
                                     + list(objective.parameters()), lr=1e-3)
        modules = (association.tcm_encoder, association.tcm_decoder, association.tam_encoder,
                   fusion.left_encoder, fusion.right_encoder, fusion.fc1, fusion.fc2,
                   fusion.decoder, fusion.state_projection, objective)
        before = [next(module.parameters()).detach().clone() for module in modules]
        (completion_total + association_total + fusion_total).backward()
        for module in modules:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        optimizer.step()
        for old, module in zip(before, modules):
            self.assertFalse(torch.equal(old, next(module.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in fusion.parameters()))


class FusionLossTests(unittest.TestCase):
    def test_hand_calculated_native_terms_and_weighting(self):
        prediction = torch.tensor([[[0.0005, 0.001, 3., 359.]]], dtype=torch.float64)
        target = torch.tensor([[[0., 0., 1., 1.]]], dtype=torch.float64)
        valid = torch.ones_like(target, dtype=torch.bool)
        loss = FusionLoss(position_loss_mode="paper_printed").double()
        expected = {"lat": 0.5 * 0.0005 ** 2, "lon": 0.001 - 0.0005,
                    "vel": (math.log(3 + 1e-6) - math.log(1 + 1e-6)) ** 2,
                    "cou": 1 - math.cos(math.radians(358))}
        actual = loss(prediction, target, valid)
        for name, value in expected.items():
            self.assertAlmostEqual(actual["losses"][name].item(), value, places=14)
        self.assertAlmostEqual(actual["total"].item(), 0.5 * sum(expected.values()) + 4 * math.log(2), places=13)
        torch.testing.assert_close(actual["variance"], torch.ones(4, dtype=torch.float64), rtol=0, atol=0)
        with torch.no_grad():
            loss.log_variance.copy_(torch.tensor([-1., 0., 1., 2.]))
        actual = loss(prediction, target, valid)
        weighted = sum(0.5 * math.exp(-s) * value + math.log1p(math.exp(s))
                       for s, value in zip([-1, 0, 1, 2], expected.values()))
        self.assertAlmostEqual(actual["total"].item(), weighted, places=13)
        buffer = io.BytesIO()
        torch.save(loss.state_dict(), buffer)
        buffer.seek(0)
        clone = FusionLoss(position_loss_mode="paper_printed").double()
        clone.load_state_dict(torch.load(buffer, weights_only=True))
        restored = clone(prediction, target, valid)
        torch.testing.assert_close(restored["total"], actual["total"], rtol=0, atol=0)
        torch.testing.assert_close(restored["variance"], actual["variance"], rtol=0, atol=0)

    def test_position_boundary_modes_remain_distinct(self):
        errors = [0.000999, 0.001, 0.001001, -0.001]
        prediction = torch.tensor([[[e, e, 0., 360.] for e in errors]], dtype=torch.float64)
        target = torch.zeros_like(prediction)
        valid = torch.ones_like(target, dtype=torch.bool)
        printed = FusionLoss(position_loss_mode="paper_printed").double()(prediction, target, valid)
        corrected = FusionLoss(position_loss_mode="standard_smooth_l1").double()(prediction, target, valid)
        expected_printed = sum(0.5 * e ** 2 if abs(e) < 0.001 else abs(e) - 0.0005 for e in errors) / 4
        expected_corrected = sum(0.5 * e ** 2 / 0.001 if abs(e) < 0.001 else abs(e) - 0.0005 for e in errors) / 4
        self.assertAlmostEqual(printed["losses"]["lat"].item(), expected_printed, places=15)
        self.assertAlmostEqual(corrected["losses"]["lat"].item(), expected_corrected, places=15)
        self.assertNotAlmostEqual(expected_printed, expected_corrected, places=6)
        self.assertEqual(printed["losses"]["vel"].item(), 0)
        self.assertEqual(printed["losses"]["cou"].item(), 0)

    def test_per_feature_mask_excludes_nan_and_negative_invalid_speed_before_math(self):
        prediction = torch.tensor([[[0.002, float("nan"), 2., float("nan")],
                                    [float("nan"), 0.003, -3., 180.]]], requires_grad=True)
        target = torch.tensor([[[0., float("nan"), 1., float("nan")],
                                [float("nan"), 0., float("nan"), 0.]]])
        valid = torch.tensor([[[True, False, True, False], [False, True, False, True]]])
        result = FusionLoss(position_loss_mode="paper_printed")(prediction, target, valid)
        self.assertAlmostEqual(result["losses"]["lat"].item(), 0.0015, places=8)
        self.assertAlmostEqual(result["losses"]["lon"].item(), 0.0025, places=8)
        self.assertEqual(result["losses"]["cou"].item(), 2)
        result["total"].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue((prediction.grad[~valid] == 0).all())


if __name__ == "__main__":
    unittest.main()
