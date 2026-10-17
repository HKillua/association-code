import math
import unittest

import torch

from baselines.tsactfer import TSAConfig, TSActfer, symmetric_info_nce


class TSActferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20)
        self.config = TSAConfig(length=8, d_model=8, nhead=2, num_layers=2,
                                dim_feedforward=32, dropout=0, time_bins=20)
        self.model = TSActfer(self.config).eval()
        self.positions = torch.randn(3, 6, 2)
        self.times = torch.arange(6).expand(3, -1)
        self.observed = torch.tensor([
            [True, True, True, False, False, False],
            [True, True, True, True, True, True],
            [True, False, False, False, False, False],
        ])

    def test_paper_dimensions_and_pre_norm_architecture(self):
        model = TSActfer(TSAConfig()).eval()
        self.assertEqual(model.config.length, 128)
        self.assertEqual(model.config.d_model, 64)
        self.assertEqual(len(model.encoder.layers), 4)
        for layer in model.encoder.layers:
            self.assertTrue(layer.norm_first)
            self.assertEqual(layer.self_attn.num_heads, 4)
            self.assertEqual(layer.linear1.out_features, 256)
            self.assertEqual(layer.dropout.p, 0.1)
        self.assertIsNone(model.encoder.norm)
        self.assertIsNone(model.continuous.bias)
        for head in model.heads.values():
            self.assertEqual(tuple(head.weight.shape), (64, 64))
            self.assertIsNone(head.bias)
        result = model(self.positions, self.times, self.observed)
        self.assertEqual(tuple(result.shape), (3, 64))
        self.assertTrue(torch.isfinite(result).all())

    def test_padding_values_and_appended_padding_do_not_change_output(self):
        before = self.model(self.positions, self.times, self.observed)
        positions = self.positions.clone()
        times = self.times.clone()
        positions[~self.observed] = float("nan")
        times[~self.observed] = -999
        after = self.model(positions, times, self.observed)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        padded_positions = torch.cat([positions, torch.full((3, 2, 2), float("nan"))], 1)
        padded_times = torch.cat([times, torch.full((3, 2), 999, dtype=torch.long)], 1)
        padded_observed = torch.cat([self.observed, torch.zeros(3, 2, dtype=torch.bool)], 1)
        extended = self.model(padded_positions, padded_times, padded_observed)
        torch.testing.assert_close(before, extended, rtol=1e-5, atol=1e-6)

    def test_heads_are_independent_and_encoder_is_shared(self):
        pooled = self.model.encode(self.positions, self.times, self.observed)
        itsa_before = self.model.project(pooled, "itsa")
        mstsa_before = self.model.project(pooled, "mstsa")
        with torch.no_grad():
            self.model.heads["itsa"].weight.zero_()
        torch.testing.assert_close(self.model.project(pooled, "mstsa"), mstsa_before,
                                   rtol=0, atol=0)
        self.assertFalse(torch.equal(itsa_before, self.model.project(pooled, "itsa")))
        self.model.zero_grad(set_to_none=True)
        self.model(self.positions, self.times, self.observed, "mstsa").square().mean().backward()
        self.assertIsNone(self.model.heads["itsa"].weight.grad)
        self.assertGreater(self.model.heads["mstsa"].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.continuous.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()
                            if p.grad is not None))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.model(self.positions, self.times, torch.zeros_like(self.observed))
        positions = self.positions.clone()
        positions[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model(positions, self.times, self.observed)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.model(self.positions, self.times + 20, self.observed)
        with self.assertRaisesRegex(ValueError, "long"):
            self.model(self.positions, self.times.float(), self.observed)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.model(self.positions, self.times[:, :-1], self.observed)
        with self.assertRaisesRegex(ValueError, "task"):
            self.model(self.positions, self.times, self.observed, "other")
        with self.assertRaisesRegex(ValueError, "L <="):
            self.model(torch.randn(3, 9, 2), torch.zeros(3, 9, dtype=torch.long),
                       torch.ones(3, 9, dtype=torch.bool))

    def test_info_nce_matches_manual_pairing_and_has_finite_gradients(self):
        embeddings = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]],
                                  requires_grad=True)
        temperature = 0.5
        loss = symmetric_info_nce(embeddings, temperature)
        expected = math.log(math.exp(1 / temperature) + 2) - 1 / temperature
        self.assertAlmostEqual(loss.item(), expected, places=6)
        wrong_partners = embeddings.detach()[[0, 1, 3, 2]]
        self.assertGreater(symmetric_info_nce(wrong_partners, temperature).item(),
                           loss.item())
        loss.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertGreater(embeddings.grad.abs().sum().item(), 0)
        torch.testing.assert_close(loss.detach(),
                                   symmetric_info_nce(embeddings.detach()[[2, 3, 0, 1]],
                                                      temperature))

    def test_info_nce_and_config_reject_invalid_values(self):
        for embeddings in (torch.zeros(2, 4), torch.zeros(5, 4), torch.zeros(4, 0),
                           torch.full((4, 2), float("nan")), torch.ones(4, 2).long()):
            with self.subTest(shape=embeddings.shape), self.assertRaises(ValueError):
                symmetric_info_nce(embeddings)
        for temperature in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                symmetric_info_nce(torch.randn(4, 2), temperature)
        for kwargs in ({"length": 0}, {"nhead": 3}, {"time_bins": True},
                       {"dropout": float("nan")}, {"dropout": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TSAConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
