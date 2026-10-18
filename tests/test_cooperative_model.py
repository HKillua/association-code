"""Synthetic CPU behavior checks; these do not establish MTAD performance."""
import unittest
from unittest.mock import patch

import torch

from association.cooperative import CooperativeConfig, CooperativeModel


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

    def test_config_and_capacity_matched_public_parts(self):
        formal = CooperativeConfig()
        self.assertEqual((formal.d_model, formal.num_layers, formal.length, formal.max_observations),
                         (128, 6, 64, 128))
        model = self.model()
        for name in ('input_projection', 'encoder', 'decoder', 'state_projection', 'role_embedding',
                     'query_pool', 'utility_head', 'identity_base', 'identity_with_utility', 'ordinary_gate'):
            self.assertIsInstance(getattr(model, name), torch.nn.Module)
        count = lambda module: sum(p.numel() for p in module.parameters())
        self.assertEqual(count(model.utility_head), count(model.ordinary_gate))
        self.assertEqual(count(model.identity_base), count(model.identity_with_utility))
        self.assertFalse(any(isinstance(m, torch.nn.Dropout) for m in model.ordinary_gate.modules()))
        self.assertFalse(torch.equal(model.encoder.layers[0].linear1.weight,
                                     model.encoder.layers[1].linear1.weight))
        self.assertFalse(torch.equal(model.decoder.layers[0].self_attn.in_proj_weight,
                                     model.decoder.layers[1].self_attn.in_proj_weight))
        self.assertIn('state_mean', model.state_dict())
        for kwargs in ({'max_observations': 1}, {'nhead': 3}, {'gate_floor': 0}, {'dropout': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CooperativeConfig(**kwargs)

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

    def test_physical_time_and_two_point_no_query_support_track_affect_output(self):
        model = self.model().eval()
        values, observed = self.packed()
        with torch.no_grad():
            expected = model(values[2:3], observed[2:3])
            states = values[2:3].clone()
            states[:, :2, :4] += .4
            times = values[2:3].clone()
            times[:, 1, 4] = .9
            queries = values[2:3].clone()
            queries[:, -1, 4] = 6.
            for changed in (states, times, queries):
                actual = model(changed, observed[2:3])
                self.assertGreater(float((actual - expected).abs().max()), 1e-5)
        self.assertEqual(tuple(expected.shape), (5, 4))
        self.assertTrue(torch.isfinite(expected).all())

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

    def test_plain_support_and_base_identity_do_not_execute_utility_or_pool(self):
        model = self.model().eval()
        encoded = model.encode(*self.packed())
        with patch.object(model.query_pool, 'forward', side_effect=AssertionError('pool used')), \
             patch.object(model.utility_head, 'forward', side_effect=AssertionError('utility used')):
            expected = model.decode_group(encoded, [0, 1])
            actual = model.decode_group(encoded, [0, 1], utility=torch.full((1, 5, 4), float('nan')))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            model.decode_group(encoded, [0, 1], routing='support')
            model.decode_group(encoded, [0, 1], routing='identity')
            model.score_pairs(encoded, [[0, 1]], use_utility=False)
        with patch.object(model.utility_head, 'forward', side_effect=AssertionError('utility used')):
            self.assertTrue(torch.isfinite(model.decode_group(encoded, [0, 1], routing='learned')).all())

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

    def test_all_routes_output_physical_singleton_and_groups(self):
        model = self.model().eval()
        encoded = model.encode(*self.packed())
        for route in ('plain', 'identity', 'utility', 'support', 'learned'):
            for members in ([0], [0, 1], [2, 0, 1]):
                with self.subTest(route=route, members=members):
                    actual = model.decode_group(encoded, members, routing=route)
                    self.assertEqual(tuple(actual.shape), (5, 4))
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue((actual[:, 0].abs() <= 90).all())
                    self.assertTrue(((actual[:, 1] >= -180) & (actual[:, 1] < 180)).all())
                    self.assertTrue((actual[:, 2] >= 0).all())
                    self.assertTrue(((actual[:, 3] >= 0) & (actual[:, 3] < 360)).all())

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

    def test_identity_loss_does_not_backpropagate_through_utility(self):
        model = self.model()
        encoded = model.encode(*self.packed())
        model.score_pairs(encoded, [[0, 1], [0, 2]], use_utility=True).sum().backward()
        self.assertTrue(all(p.grad is None for p in model.utility_head.parameters()))
        self.assertTrue(all(p.grad is None for p in model.query_pool.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.identity_with_utility.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))

    def test_frozen_teacher_utility_training_has_only_head_and_pool_gradients(self):
        model = self.model(dropout=.3).eval()
        model.requires_grad_(False)
        model.utility_head.requires_grad_(True)
        model.query_pool.requires_grad_(True)
        encoded = model.encode(*self.packed())
        first = model.utility(encoded, [[0, 1]])
        second = model.utility(encoded, [[0, 1]])
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        first.square().mean().backward()
        for name, parameter in model.named_parameters():
            with self.subTest(name=name):
                if name.startswith(('query_pool.', 'utility_head.')):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                else:
                    self.assertIsNone(parameter.grad)

    def test_ordinary_gate_can_learn_without_unfreezing_utility_or_teacher(self):
        model = self.model().eval()
        model.requires_grad_(False)
        model.ordinary_gate.requires_grad_(True)
        encoded = model.encode(*self.packed())
        model.decode_group(encoded, [0, 1], routing='learned').square().mean().backward()
        grads = [p.grad for p in model.ordinary_gate.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters()
                            if not name.startswith('ordinary_gate.')))

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

    def test_bad_inputs_routes_and_auxiliary_shape_fail(self):
        model = self.model()
        values, observed = self.packed()
        for change in ('one_point', 'native_nan', 'repeated_time', 'query_nan', 'query_mask'):
            v, m = values.clone(), observed.clone()
            if change == 'one_point':
                m[1, 1] = False
            elif change == 'native_nan':
                v[1, 0, 0] = float('nan')
            elif change == 'repeated_time':
                v[0, 1, 4] = v[0, 0, 4]
            elif change == 'query_nan':
                v[0, -1, 4] = float('nan')
            else:
                m[0, -1] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                model.encode(v, m)
        encoded = model.encode(values, observed)
        for kwargs in ({'routing': 'unknown'}, {'routing': 'identity', 'identity_logits': torch.ones(2)},
                       {'routing': 'identity', 'identity_logits': torch.tensor([float('inf')])},
                       {'routing': 'utility', 'utility': torch.ones(2, 5, 4)},
                       {'routing': 'utility', 'utility': torch.full((1, 5, 4), float('inf'))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model.decode_group(encoded, [0, 1], **kwargs)
        for members in ([], [0, 0], [-1], [3]):
            with self.assertRaises(ValueError):
                model.decode_group(encoded, members)
        encoded['queries'] = encoded['queries'].clone()
        encoded['queries'][1, -1] += 1
        with self.assertRaises(ValueError):
            model.decode_group(encoded, [0, 1])
        with self.assertRaises(ValueError):
            model.utility(encoded, [[0, 1]])

    def test_empty_encode_and_pair_interfaces(self):
        model = self.model().eval()
        values, observed = self.packed()
        encoded = model.encode(values[:0], observed[:0])
        self.assertEqual(tuple(encoded['memory'].shape), (0, 4, 8))
        self.assertEqual(tuple(model.summarize(encoded).shape), (0, 5, 8))
        pairs = torch.empty((0, 2), dtype=torch.long)
        self.assertEqual(tuple(model.utility(encoded, pairs).shape), (0, 2, 5, 4))
        self.assertEqual(tuple(model.score_pairs(encoded, pairs).shape), (0,))
        for bad in ([], torch.empty((0, 2)), torch.empty((0, 3), dtype=torch.long)):
            with self.assertRaises(ValueError):
                model.utility(encoded, bad)


if __name__ == '__main__':
    unittest.main()
