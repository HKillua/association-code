import io
import unittest

import torch

from association.models import AssociationModel, ModelConfig


class AssociationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20)
        self.config = ModelConfig(d_model=8, nhead=2, num_layers=1,
                                  dim_feedforward=16, dropout=0, length=6)
        self.values = torch.randn(3, 6, 4)
        self.observed = torch.tensor([
            [False, True, True, True, False, False],
            [True, True, True, True, True, True],
            [False, False, True, False, False, False],
        ])
        self.pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])
        self.labels = torch.tensor([1., 0., 0.])

    def test_completion_preserves_observed_and_predicts_both_endpoints(self):
        model = AssociationModel(self.config).eval()
        embedded = model.embed_inputs(self.values, self.observed)
        result = model.complete(self.values, self.observed)
        self.assertEqual(tuple(result.shape), (3, 6, 8))
        self.assertTrue(torch.equal(result[self.observed], embedded[self.observed]))
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(torch.equal(result[0, 0], embedded[0, 0]))
        self.assertFalse(torch.equal(result[0, -1], embedded[0, -1]))

    def test_unobserved_values_are_not_model_inputs(self):
        model = AssociationModel(self.config).eval()
        changed = self.values.clone()
        changed[~self.observed] = float("nan")
        torch.testing.assert_close(model.complete(self.values, self.observed),
                                   model.complete(changed, self.observed), rtol=0, atol=0)

    def test_teacher_forcing_cannot_access_future_encoder_or_decoder_values(self):
        model = AssociationModel(self.config).eval()
        history = torch.randn(2, 5, 8)
        changed = history.clone()
        changed[:, 3:] += torch.randn_like(changed[:, 3:]) * 100
        first = model._teacher_predictions(history)
        second = model._teacher_predictions(changed)
        torch.testing.assert_close(first[:, :3], second[:, :3], rtol=1e-5, atol=1e-6)

    def test_completion_loss_gradients_reach_encoder_and_decoder(self):
        model = AssociationModel(self.config)
        for param in model.input_embedding.parameters():
            param.requires_grad_(False)
        loss = model.completion_loss(self.values, self.observed)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (model.tcm_encoder, model.tcm_decoder):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_both_association_branches_symmetric_and_differentiable(self):
        for branch in ("ce", "contrastive"):
            with self.subTest(branch=branch):
                model = AssociationModel(self.config, branch).eval()
                encoded = model.encode_tracks(self.values, self.observed)
                self.assertEqual(tuple(encoded.shape), (3, 48))
                scores = model.score_pairs(encoded, self.pairs)
                torch.testing.assert_close(scores, model.score_pairs(encoded, self.pairs.flip(1)),
                                           rtol=0, atol=0)
                loss = model.association_loss(scores, self.labels)
                loss.backward()
                for module in (model.tcm_decoder, model.tam_encoder):
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_contrastive_equation_and_identical_embedding_gradient(self):
        model = AssociationModel(self.config, "contrastive")
        scores = torch.tensor([-0.25, -0.25, -2.0], requires_grad=True)
        labels = torch.tensor([1., 0., 0.])
        actual = model.association_loss(scores, labels)
        self.assertAlmostEqual(actual.item(), (0.25 + 0.75 ** 2) / 6, places=7)
        embeddings = torch.ones(2, 48, requires_grad=True)
        zero_scores = model.score_pairs(embeddings, torch.tensor([[0, 1]]))
        model.association_loss(zero_scores, torch.tensor([1.])).backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_checkpoint_roundtrip(self):
        model = AssociationModel(self.config).eval()
        before = model.score_pairs(model.encode_tracks(self.values, self.observed), self.pairs)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        clone = AssociationModel(self.config).eval()
        clone.load_state_dict(torch.load(buffer, weights_only=True))
        after = clone.score_pairs(clone.encode_tracks(self.values, self.observed), self.pairs)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_singleton_completion_and_empty_pairs_are_finite(self):
        model = AssociationModel(self.config)
        model.input_embedding.requires_grad_(False)
        loss = model.completion_loss(self.values[2:], self.observed[2:])
        self.assertEqual(loss.item(), 0)
        loss.backward()
        encoded = model.encode_tracks(self.values[2:], self.observed[2:])
        self.assertTrue(torch.isfinite(encoded).all())
        scores = model.score_pairs(encoded, torch.empty((0, 2), dtype=torch.long))
        self.assertEqual(tuple(scores.shape), (0,))
        self.assertEqual(model.association_loss(scores, torch.empty(0)).item(), 0)

    def test_internal_gap_and_empty_track_are_rejected(self):
        model = AssociationModel(self.config)
        for support in ([True, False, True, False, False, False], [False] * 6):
            with self.assertRaises(ValueError):
                model.complete(self.values[:1], torch.tensor([support]))

    def test_identical_tam_input_reuses_shared_encoder(self):
        model = AssociationModel(self.config).eval()
        repeated = self.values[:1].repeat(2, 1, 1)
        support = self.observed[:1].repeat(2, 1)
        encoded = model.encode_tracks(repeated, support)
        torch.testing.assert_close(encoded[0], encoded[1], rtol=0, atol=0)
        score = model.score_pairs(encoded, torch.tensor([[0, 1]]))
        self.assertEqual(score.item(), 0.0)

    def test_explicit_frozen_and_joint_stage_controls(self):
        model = AssociationModel(self.config)
        model.set_completion_trainable(False)
        model.train()
        self.assertFalse(model.tcm_encoder.training)
        self.assertFalse(model.tcm_decoder.training)
        self.assertTrue(model.tam_encoder.training)
        self.assertFalse(any(p.requires_grad for p in model.completion_parameters(True)))
        model.set_completion_trainable(True)
        self.assertTrue(model.tcm_encoder.training)
        self.assertTrue(model.tcm_decoder.training)
        self.assertTrue(all(p.requires_grad for p in model.completion_parameters(True)))


if __name__ == "__main__":
    unittest.main()
