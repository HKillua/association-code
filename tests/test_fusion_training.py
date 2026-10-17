import copy
import unittest
from unittest.mock import patch

import torch

from association.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.fusion_training import fusion_training_parameters, fusion_training_step
from association.models import AssociationModel, ModelConfig


class FusionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def models(self, branch="contrastive", dropout=0):
        torch.manual_seed(42)
        model = AssociationModel(ModelConfig(length=4, d_model=8, nhead=2,
                                 num_layers=1, dim_feedforward=16, dropout=dropout), branch).double()
        fusion = PairFusion(PairFusionConfig(fc_hidden=8, length=4, d_model=8, nhead=2,
                            num_layers=1, dim_feedforward=16, dropout=dropout)).double()
        objective = FusionLoss(position_loss_mode="paper_printed").double()
        return model, fusion, objective

    def batch(self):
        values = torch.randn(5, 4, 4, dtype=torch.float64)
        observed = torch.tensor([[False, True, True, False], [True, True, True, False],
                                 [False, True, True, True], [True, True, False, False],
                                 [True, True, True, True]])
        values[~observed] = float("nan")
        # An unused fifth endpoint must not contribute auxiliary TCM targets.
        pairs = torch.tensor([[0, 2], [1, 3], [0, 1], [0, 1], [2, 3]])
        labels = torch.tensor([0., 0., 1., 1., 1.], dtype=torch.float64)
        target = torch.tensor([30., 120., 5., 359.], dtype=torch.float64).expand(3, 4, 4).clone()
        valid = torch.ones_like(target, dtype=torch.bool)
        valid[:2, :, 0] = False  # First positive micro has no latitude target.
        valid[:2, :2, 1] = False
        valid[2, :, 2] = False
        valid[2, 1:, 3] = False
        target[~valid] = float("nan")
        return dict(values=values, observed=observed, pairs=pairs, labels=labels, target=target, valid=valid)

    def optimizer(self, modules, stage):
        parameters = fusion_training_parameters(*modules, stage=stage)
        return torch.optim.SGD(parameters, lr=0.001)

    def test_joint_micro_gradients_match_independent_full_objective(self):
        for branch in ("ce", "contrastive"):
            with self.subTest(branch=branch):
                modules = self.models(branch)
                full = copy.deepcopy(modules)
                batch = self.batch()
                optimizer = self.optimizer(modules, "JOINT")
                reference_optimizer = self.optimizer(full, "JOINT")
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    report = fusion_training_step(*modules, optimizer, stage="JOINT", **batch,
                        pair_microbatch=1, track_microbatch=2, completion_weight=0.1, gradient_clip=1e9)
                    self.assertEqual(step.call_count, 1)
                model, fusion, objective = full
                nodes, inverse = torch.unique(batch["pairs"], return_inverse=True)
                pairs = inverse.reshape(-1, 2)
                x, mask = batch["values"][nodes], batch["observed"][nodes]
                completed = model.complete(x, mask)
                scores = model.score_pairs(model.tam_encoder(completed).flatten(1), pairs)
                association = model.association_loss(scores, batch["labels"])
                completion = model.completion_loss(x, mask)
                states = fusion.forward_pairs(completed, pairs[batch["labels"] == 1])["states"]
                fusion_loss = objective(states, batch["target"], batch["valid"])["total"]
                total = association + 0.1 * completion + fusion_loss
                reference_optimizer.zero_grad()
                total.backward()
                reference_optimizer.step()
                self.assertAlmostEqual(report["loss"], float(total.detach()), places=9)
                self.assertEqual(report["unique_endpoints"], 4)
                self.assertEqual(report["completion_target_elements"], 96)
                self.assertEqual(report["microbatches"], 5)
                self.assertEqual(report["positive_pairs"], 3)
                for actual_module, expected_module in zip(modules, full):
                    for actual, expected in zip(actual_module.parameters(), expected_module.parameters()):
                        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)
                        if expected.grad is not None:
                            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-8, rtol=1e-7)

    def test_frozen_tfm_updates_only_fusion_and_uncertainty(self):
        modules = self.models()
        batch = self.batch()
        batch["pairs"], batch["labels"] = batch["pairs"][2:], batch["labels"][2:]
        optimizer = self.optimizer(modules, "TFM_FROZEN")
        before = [{key: value.clone() for key, value in module.state_dict().items()} for module in modules]
        report = fusion_training_step(*modules, optimizer, stage="TFM_FROZEN", **batch,
            pair_microbatch=1, track_microbatch=2, completion_weight=0, gradient_clip=1)
        self.assertEqual(report["association_loss"], 0)
        self.assertEqual(report["completion_loss"], 0)
        self.assertFalse(modules[0].training)
        self.assertFalse(modules[0].tcm_encoder.training)
        for key, old in before[0].items():
            torch.testing.assert_close(modules[0].state_dict()[key], old, atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in modules[0].parameters()))
        for module, old in zip(modules[1:], before[1:]):
            self.assertTrue(any(not torch.equal(value, old[key]) for key, value in module.state_dict().items()))

    def test_unusable_effective_batch_never_steps_or_resamples(self):
        modules = self.models()
        batch = self.batch()
        batch["valid"][..., 3] = False
        optimizer = self.optimizer(modules, "JOINT")
        before = [p.clone() for module in modules for p in module.parameters()]
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaisesRegex(ValueError, "no valid supervision"):
                fusion_training_step(*modules, optimizer, stage="JOINT", **batch,
                    pair_microbatch=2, track_microbatch=2, completion_weight=.1, gradient_clip=1)
            self.assertEqual(step.call_count, 0)
        for actual, old in zip((p for module in modules for p in module.parameters()), before):
            torch.testing.assert_close(actual, old, atol=0, rtol=0)
            self.assertIsNone(actual.grad)

    def test_missing_loss_parameters_in_optimizer_is_rejected(self):
        modules = self.models()
        parameters = fusion_training_parameters(*modules, stage="JOINT")
        optimizer = torch.optim.SGD(parameters[:-1], lr=.001)
        with self.assertRaisesRegex(ValueError, "optimizer must contain exactly"):
            fusion_training_step(*modules, optimizer, stage="JOINT", **self.batch(),
                pair_microbatch=2, track_microbatch=2, completion_weight=.1, gradient_clip=1)

    def test_deadline_during_accumulation_clears_gradients_without_partial_step(self):
        modules = self.models()
        optimizer = self.optimizer(modules, "JOINT")
        before = [p.clone() for module in modules for p in module.parameters()]
        calls = []

        def check():
            calls.append(1)
            if len(calls) == 3:
                raise TimeoutError("effective update interrupted")

        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaises(TimeoutError):
                fusion_training_step(*modules, optimizer, stage="JOINT", **self.batch(),
                    pair_microbatch=2, track_microbatch=2, completion_weight=.1, gradient_clip=1, check=check)
            self.assertEqual(step.call_count, 0)
        for actual, old in zip((p for module in modules for p in module.parameters()), before):
            torch.testing.assert_close(actual, old, atol=0, rtol=0)
            self.assertIsNone(actual.grad)

    def test_dropout_joint_is_finite_and_reaches_each_module(self):
        modules = self.models(dropout=.1)
        optimizer = self.optimizer(modules, "JOINT")
        report = fusion_training_step(*modules, optimizer, stage="JOINT", **self.batch(),
            pair_microbatch=2, track_microbatch=2, completion_weight=.1, gradient_clip=1)
        self.assertTrue(torch.isfinite(torch.tensor(report["loss"])))
        model, fusion, objective = modules
        for module in (model.tcm_encoder, model.tcm_decoder, model.tam_encoder, fusion, objective):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_zero_transition_joint_keeps_association_and_fusion(self):
        modules = self.models()
        batch = self.batch()
        batch["observed"][:] = False
        batch["observed"][:, 1] = True
        optimizer = self.optimizer(modules, "JOINT")
        report = fusion_training_step(*modules, optimizer, stage="JOINT", **batch,
            pair_microbatch=2, track_microbatch=2, completion_weight=.1, gradient_clip=1)
        self.assertEqual(report["completion_target_elements"], 0)
        self.assertEqual(report["completion_loss"], 0)
        self.assertGreater(report["fusion_loss"], 0)


class FusionLossAccumulationTests(unittest.TestCase):
    def test_masked_unequal_feature_counts_and_regularizer_sum_match_full(self):
        torch.manual_seed(5)
        prediction = torch.rand(3, 2, 4, dtype=torch.float64, requires_grad=True)
        target = torch.ones_like(prediction)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        valid[0, :, :2] = False
        valid[1, :, 2] = False
        target[~valid] = float("nan")
        full = FusionLoss(position_loss_mode="standard_smooth_l1").double()
        accumulated = copy.deepcopy(full)
        p2 = prediction.detach().clone().requires_grad_()
        expected = full(prediction, target, valid)["total"]
        expected.backward()
        actual = 0
        for i in range(3):
            result = accumulated(p2[i:i+1], target[i:i+1], valid[i:i+1],
                normalization_counts=valid.sum((0, 1)), regularizer_weight=float(i == 0))
            actual += float(result["total"].detach())
            result["total"].backward()
        self.assertAlmostEqual(actual, float(expected.detach()), places=12)
        torch.testing.assert_close(prediction.grad, p2.grad)
        torch.testing.assert_close(full.log_variance.grad, accumulated.log_variance.grad)

    def test_bad_global_counts_and_weights_fail(self):
        objective = FusionLoss(position_loss_mode="paper_printed")
        prediction = torch.ones(1, 2, 4)
        valid = torch.ones_like(prediction, dtype=torch.bool)
        for counts in ([1, 2, 2, 2], [0, 2, 2, 2], [2., 2., 2., 2.], [2, 2]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                objective(prediction, prediction, valid, normalization_counts=counts)
        for weight in (-1., 2., float("nan"), True):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                objective(prediction, prediction, valid, regularizer_weight=weight)


if __name__ == "__main__":
    unittest.main()
