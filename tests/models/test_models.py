import unittest
from dataclasses import replace

import torch

from association.models.tmtaff import AssociationModel, ModelConfig


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

    def test_tam_only_dropout_keeps_initial_weights_rng_and_completion_unchanged(self):
        config = replace(self.config, dropout=.35)
        torch.manual_seed(20)
        original = AssociationModel(config, 'contrastive')
        original_rng = torch.get_rng_state().clone()
        torch.manual_seed(20)
        changed = AssociationModel(replace(config, tam_dropout=0.), 'contrastive')
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, changed.state_dict()[key], rtol=0, atol=0)
        for root, expected in ((changed.tcm_encoder, .35), (changed.tcm_decoder, .35),
                               (changed.tam_encoder, 0.)):
            for module in root.modules():
                if isinstance(module, torch.nn.Dropout):
                    self.assertEqual(module.p, expected)
                if isinstance(module, torch.nn.MultiheadAttention):
                    self.assertEqual(module.dropout, expected)
        self.assertEqual(changed.output_dropout.p, 0.)
        original.eval()
        changed.eval()
        complete = changed.complete(self.values, self.observed).detach()
        torch.testing.assert_close(complete, original.complete(self.values, self.observed), rtol=0, atol=0)
        expected = changed.tam_encoder(complete)
        changed.set_completion_trainable(False)
        changed.train()
        self.assertFalse(changed.tcm_encoder.training)
        self.assertFalse(changed.tcm_decoder.training)
        torch.testing.assert_close(changed.tam_encoder(complete), expected, rtol=0, atol=0)
        self.assertEqual(ModelConfig().effective_tam_dropout, ModelConfig().dropout)


    def test_track_encoding_is_independent_of_batch_neighbors_and_order(self):
        model = AssociationModel(self.config).eval()
        batch = model.encode_tracks(self.values, self.observed)
        separate = torch.cat([
            model.encode_tracks(self.values[i:i + 1], self.observed[i:i + 1])
            for i in range(len(self.values))
        ])
        order = torch.tensor([2, 0, 1])
        permuted = model.encode_tracks(self.values[order], self.observed[order])
        torch.testing.assert_close(batch, separate, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(batch[order], permuted, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
