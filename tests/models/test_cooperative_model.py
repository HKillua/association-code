"""Synthetic CPU behavior checks; these do not establish MTAD performance."""
import unittest
from unittest.mock import patch

import torch

from association.models.cooperative import CooperativeConfig, CooperativeModel


class CooperativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, dropout=0.):
        torch.manual_seed(20)
        config = CooperativeConfig(length=5, max_observations=4, d_model=8,
            nhead=2, num_layers=2, dim_feedforward=16, dropout=dropout, head_hidden=6)
        return CooperativeModel(config, {'mean': [10., 20., 3., 180.], 'std': [1., 2., 1., 10.]})

    def packed(self):
        values = torch.full((3, 9, 5), float('nan'))
        observed = torch.zeros(3, 9, dtype=torch.bool)
        for row, times in enumerate(((1., 2., 3.), (1.5, 2.5), (.7, .8))):
            n = len(times)
            observed[row, :n] = True
            t = torch.tensor(times)
            values[row, :n, 4] = t
            values[row, :n, :4] = torch.stack((t + row, .2 * t, .1 * t, -.3 * t), -1)
            values[row, 4:, 4] = torch.arange(5.)
        return values, observed


    def test_query_states_padding_and_hidden_native_content_never_leak(self):
        model = self.model().eval()
        values, observed = self.packed()
        # A temporary native mask may exclude an endpoint before compaction.
        observed[0, 0] = False
        changed = values.clone()
        changed[:, :4][~observed[:, :4]] = -1e25
        changed[:, 4:, :4] = float('inf')
        with torch.no_grad():
            expected = model(values, observed)
            actual = model(changed, observed)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        differentiable = values.clone().requires_grad_()
        model(differentiable, observed).sum().backward()
        self.assertTrue(torch.isfinite(differentiable.grad).all())
        self.assertEqual(float(differentiable.grad[:, 4:, :4].abs().sum()), 0.)
        self.assertEqual(float(differentiable.grad[:, :4][~observed[:, :4]].abs().sum()), 0.)
        self.assertGreater(float(differentiable.grad[:, :4, :4].abs().sum()), 0.)


    def test_utility_direction_swap_and_identity_symmetry(self):
        model = self.model().eval()
        encoded = model.encode(*self.packed())
        pairs = torch.tensor([[0, 1], [0, 2]])
        summaries = model.summarize(encoded)
        direct = model.utility(encoded, pairs, summaries)
        reverse = model.utility(encoded, pairs.flip(1), summaries)
        torch.testing.assert_close(direct, reverse.flip(1), rtol=0, atol=0)
        self.assertGreater(float((direct[:, 0] - direct[:, 1]).abs().max().detach()), 1e-5)
        for use_utility in (False, True):
            torch.testing.assert_close(model.score_pairs(encoded, pairs, use_utility),
                model.score_pairs(encoded, pairs.flip(1), use_utility), rtol=1e-6, atol=1e-6)


    def test_length_normalized_bias_applies_to_anchor_and_helper(self):
        model = self.model().eval()
        encoded = model.encode(*self.packed())
        with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as decoder:
            model.decode_group(encoded, [0, 1], routing='plain')
        bias = decoder.call_args.kwargs['memory_mask']
        self.assertEqual(tuple(bias.shape), (5, 5))
        torch.testing.assert_close(bias[:, :3].logsumexp(1), torch.zeros(5), atol=1e-7, rtol=0)
        torch.testing.assert_close(bias[:, 3:].logsumexp(1), torch.zeros(5), atol=1e-7, rtol=0)
        with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as decoder:
            model.decode_group(encoded, [0, 1], routing='utility', utility=torch.full((1, 5, 4), -2.))
        bias = decoder.call_args.kwargs['memory_mask']
        torch.testing.assert_close(bias[:, 3:].logsumexp(1).exp(), torch.full((5,), model.config.gate_floor))


    def test_utility_routing_combines_fixed_identity_and_utility_with_affine_floor(self):
        model = self.model().eval()
        encoded = model.encode(*self.packed())
        for logit, utility in ((0., .6), (-5., .6), (5., .6), (5., -2.), (20., 1.)):
            with self.subTest(logit=logit, utility=utility), \
                 patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as decoder:
                model.decode_group(encoded, [0, 1], routing='utility',
                    identity_logits=torch.tensor([logit]), utility=torch.full((1, 5, 4), utility))
                mass = decoder.call_args.kwargs['memory_mask'][:, 3:].logsumexp(1).exp()
                expected = model.config.gate_floor + (1 - model.config.gate_floor) * \
                    torch.sigmoid(torch.tensor(logit)) * min(max(utility, 0.), 1.)
                torch.testing.assert_close(mass, expected.expand(5))
        with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as decoder:
            model.decode_group(encoded, [0, 1], routing='identity', identity_logits=torch.zeros(1))
        mass = decoder.call_args.kwargs['memory_mask'][:, 3:].logsumexp(1).exp()
        torch.testing.assert_close(mass, torch.full((5,), .5 + .5 * model.config.gate_floor))

    def test_identity_utility_summary_is_bounded_symmetric_and_retains_negative_sign(self):
        model = self.model().eval()
        encoded = model.encode(*self.packed())
        u = torch.tensor([-100., -1., 1., 100.]).expand(1, 2, 5, 4)
        with patch.object(model, 'utility', return_value=u), \
             patch.object(model.identity_with_utility, 'forward', wraps=model.identity_with_utility.forward) as head:
            model.score_pairs(encoded, [[0, 1]], use_utility=True)
        supplied = head.call_args.args[0][:, -4:]
        torch.testing.assert_close(supplied, u.tanh().mean((1, 2)))
        self.assertTrue((supplied.abs() <= 1.).all())
        self.assertTrue((supplied[:, :2] < 0).all())


    def test_negative_utility_is_preserved_and_external_routing_changes_fusion(self):
        model = self.model().eval()
        encoded = model.encode(*self.packed())
        with torch.no_grad():
            model.utility_head[-1].weight.zero_()
            model.utility_head[-1].bias.fill_(-2.)
            prediction = model.utility(encoded, [[0, 1]])
            torch.testing.assert_close(prediction, torch.full((1, 2, 5, 4), -2.))
            low = model.decode_group(encoded, [0, 1], routing='utility', utility=prediction[:, 0])
            high = model.decode_group(encoded, [0, 1], routing='utility', utility=torch.ones(1, 5, 4))
            self.assertGreater(float((low - high).abs().max()), 1e-5)


if __name__ == '__main__':
    unittest.main()
