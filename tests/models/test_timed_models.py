"""Native-time conditioning, causal consistency and unchanged learned capacity."""
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from association.models.tmtaff import AssociationModel, ModelConfig
from association.models.timed import TimedAssociationModel, continuous_time_encoding


class TimedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, *, layers=2, dropout=0, branch='ce'):
        torch.manual_seed(17)
        return TimedAssociationModel(ModelConfig(length=6, d_model=8, nhead=2,
            num_layers=layers, dim_feedforward=16, dropout=dropout), branch, max_observations=5)

    def packed(self, *, native_times=((1., 2., 4.), (1.5, 2.5)), query_times=(0., 1., 2., 3., 4., 5.)):
        values = torch.full((len(native_times), 11, 5), float('nan'))
        observed = torch.zeros(len(native_times), 11, dtype=torch.bool)
        query = torch.tensor(query_times)
        for row, times in enumerate(native_times):
            times = torch.tensor(times)
            n = len(times)
            observed[row, :n] = True
            values[row, :n, 4] = times
            values[row, :n, :4] = torch.stack((times * .1 + row, times * .2, times * 0 + 1, times * .3), dim=-1)
            values[row, 5:, 4] = query
            inside = (query >= times[0]) & (query <= times[-1])
            values[row, 5:, :4][inside] = torch.stack((query[inside] * .1 + row,
                query[inside] * .2, query[inside] * 0 + 1, query[inside] * .3), dim=-1)
        return values, observed

    def test_paper_scale_parameter_count_and_shapes_are_unchanged(self):
        for branch in ('ce', 'contrastive'):
            config = ModelConfig()
            old = AssociationModel(config, branch)
            timed = TimedAssociationModel(config, branch, max_observations=128)
            self.assertEqual(sum(p.numel() for p in old.parameters()), sum(p.numel() for p in timed.parameters()))
            self.assertEqual({k: tuple(v.shape) for k, v in old.state_dict().items()},
                             {k: tuple(v.shape) for k, v in timed.state_dict().items()})
            self.assertEqual(len(timed.tcm_encoder.layers), 6)
            self.assertEqual(timed.config.d_model, 128)
            self.assertEqual(timed.input_adapter, 'native_time_v1')

    def test_short_native_track_with_zero_support_on_query_grid_is_retained(self):
        model = self.model().eval()
        values, observed = self.packed(native_times=((1.2, 1.4),), query_times=(0., 1., 2., 3., 4., 5.))
        self.assertTrue(torch.isnan(values[:, 5:, :4]).all())
        with torch.no_grad():
            result = model.complete(values, observed)
            changed = values.clone()
            changed[:, :2, :4] += .5
            second = model.complete(changed, observed)
        self.assertEqual(tuple(result.shape), (1, 6, 8))
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(float((second - result).abs().max()), 1e-4)
        self.assertEqual(tuple(model.encode_tracks(values, observed).shape), (1, 48))

    def test_teacher_prefix_last_prediction_equals_first_rollout_for_both_directions(self):
        for layers in (1, 3):
            model = self.model(layers=layers).eval()
            history = torch.randn(2, 4, 8)
            times = torch.tensor([[.1, .3, .9, 2., 5.], [.2, .6, 1., 3., 4.]])
            for reverse in (False, True):
                source_times = times.flip(1) if reverse else times
                source = history.flip(1) if reverse else history
                with self.subTest(layers=layers, reverse=reverse), torch.no_grad():
                    teacher = model._teacher_predictions(source, source_times[:, 1:])[:, -1:]
                    rollout = model._extrapolate(source, source_times[:, :-1], source_times[:, -1:])
                    torch.testing.assert_close(teacher, rollout, rtol=1e-5, atol=1e-6)

    def test_future_states_and_next_times_cannot_change_earlier_teacher_predictions(self):
        model = self.model(layers=3).eval()
        history = torch.randn(2, 5, 8)
        times = torch.tensor([[1., 2., 4., 8., 10.]]).repeat(2, 1)
        changed_history, changed_times = history.clone(), times.clone()
        changed_history[:, 3:] += 50
        changed_times[:, 3:] += 20
        with torch.no_grad():
            expected = model._teacher_predictions(history, times)
            actual = model._teacher_predictions(changed_history, changed_times)
            prefix = model._teacher_predictions(history[:, :3], times[:, :3])
        torch.testing.assert_close(expected[:, :3], actual[:, :3], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(expected[:, :3], prefix, rtol=1e-5, atol=1e-6)
        self.assertGreater(float((expected[:, 3:] - actual[:, 3:]).abs().max()), 1e-4)

    def test_query_time_and_native_time_both_change_endpoint_predictions(self):
        model = self.model().eval()
        values, observed = self.packed(native_times=((1., 2., 4.),))
        later_query = values.clone()
        later_query[:, -1, 4] = 6.
        different_history = values.clone()
        different_history[:, 1, 4] = 3.
        with torch.no_grad():
            expected = model.complete(values, observed)
            actual_query = model.complete(later_query, observed)
            actual_history = model.complete(different_history, observed)
        self.assertGreater(float((expected[:, -1] - actual_query[:, -1]).abs().max()), 1e-4)
        self.assertGreater(float((expected[:, -1] - actual_history[:, -1]).abs().max()), 1e-4)
        torch.testing.assert_close(actual_query[:, :-1], expected[:, :-1], rtol=0, atol=0)

    def test_padding_and_outside_query_state_values_have_no_effect_or_gradient(self):
        model = self.model().eval()
        values, observed = self.packed()
        changed = values.clone()
        changed[:, :5][~observed[:, :5]] = 1e10
        for row in range(len(values)):
            outside = torch.isnan(values[row, 5:, 0])
            changed[row, 5:, :4][outside] = torch.tensor([float('inf'), -1e12, float('nan'), 1e12])
        with torch.no_grad():
            torch.testing.assert_close(model.complete(values, observed), model.complete(changed, observed), rtol=0, atol=0)
            torch.testing.assert_close(model.completion_loss(values, observed), model.completion_loss(changed, observed), rtol=0, atol=0)
        differentiable = values.clone().requires_grad_()
        (model.complete(differentiable, observed) * torch.randn(2, 6, 8)).sum().backward()
        self.assertTrue(torch.isfinite(differentiable.grad).all())
        self.assertEqual(float(differentiable.grad[:, :5][~observed[:, :5]].abs().sum()), 0)
        outside = torch.isnan(values[:, 5:, 0])
        self.assertEqual(float(differentiable.grad[:, 5:, :4][outside].abs().sum()), 0)

    def test_complete_preserves_inside_queries_and_both_endpoint_directions(self):
        model = self.model().eval()
        values, observed = self.packed(native_times=((1., 2., 4.),))
        with torch.no_grad():
            embedded = model.embed_inputs(values, observed)
            complete = model.complete(values, observed)
            expected_before = model._teacher_predictions(embedded[:, :3].flip(1),
                                                          torch.tensor([[2., 1., 0.]]))[:, -1]
            expected_after = model._teacher_predictions(embedded[:, :3], torch.tensor([[2., 4., 5.]]))[:, -1]
        torch.testing.assert_close(complete[:, 1:5], embedded[:, 6:10], rtol=0, atol=0)
        torch.testing.assert_close(complete[:, 0], expected_before, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(complete[:, -1], expected_after, rtol=1e-5, atol=1e-6)

    def test_rollout_keeps_fixed_native_memory_and_rectangular_causal_mask(self):
        model = self.model().eval()
        native = torch.randn(1, 3, 8)
        with patch.object(model.tcm_decoder, 'forward', wraps=model.tcm_decoder.forward) as decoder:
            model._extrapolate(native, torch.tensor([[1., 2., 4.]]), torch.tensor([[5., 8., 10.]]))
        self.assertEqual([tuple(c.kwargs['memory_mask'].shape) for c in decoder.call_args_list],
                         [(3, 3), (4, 3), (5, 3)])
        self.assertEqual([c.args[1].shape[1] for c in decoder.call_args_list], [3, 3, 3])
        torch.testing.assert_close(decoder.call_args_list[-1].kwargs['memory_mask'], torch.tensor([
            [False, True, True], [False, False, True], [False, False, False],
            [False, False, False], [False, False, False]]))

    def test_completion_loss_weights_actual_native_transitions_and_detaches_targets(self):
        model = self.model().eval()
        values, observed = self.packed()
        # Check against independent one-step causal prefixes, not the batched
        # teacher routine or a mean over tracks of unequal native lengths.
        embedded = model.embed_inputs(values, observed)
        total, count = 0., 0
        for row, n in enumerate((3, 2)):
            for reverse in (False, True):
                sequence, times = embedded[row:row+1, :n], values[row:row+1, :n, 4]
                if reverse:
                    sequence, times = sequence.flip(1), times.flip(1)
                for endpoint in range(1, n):
                    prediction = model._extrapolate(sequence[:, :endpoint], times[:, :endpoint], times[:, endpoint:endpoint+1])
                    target = sequence[:, endpoint:endpoint+1].detach()
                    total = total + F.mse_loss(prediction, target, reduction='sum')
                    count += target.numel()
        expected = total / count
        actual = model.completion_loss(values, observed)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected_grad = torch.autograd.grad(expected, model.input_embedding.weight, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(actual, model.input_embedding.weight)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
        changed = values.clone()
        inside = torch.isfinite(changed[:, 5:, 0])
        changed[:, 5:, :4][inside] += 500
        torch.testing.assert_close(model.completion_loss(changed, observed), actual, rtol=0, atol=0)
        self.assertEqual(count, 2 * ((3 - 1) + (2 - 1)) * 8)

    def test_endpoint_gradients_and_frozen_completion_remain_compatible(self):
        model = self.model(dropout=.2)
        values, observed = self.packed()
        model.set_completion_trainable(False)
        model.train()
        first, second = model.complete(values, observed), model.complete(values, observed)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(model.tcm_encoder.training)
        self.assertFalse(model.tcm_decoder.training)
        score = model.score_pairs(model.encode_tracks(values, observed), [[0, 1]])
        model.association_loss(score, torch.tensor([0.])).backward()
        self.assertTrue(all(p.grad is None for p in model.completion_parameters(True)))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.tam_encoder.parameters()))
        model.zero_grad(set_to_none=True)
        model.set_completion_trainable(True)
        model.eval()
        complete = model.complete(values, observed)
        (complete[:, [0, 5]] * torch.randn_like(complete[:, [0, 5]])).sum().backward()
        for module in (model.input_embedding, model.tcm_encoder, model.tcm_decoder):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_circular_projection_preserves_other_initialization_and_global_rng(self):
        config = ModelConfig(length=6, d_model=8, nhead=2, num_layers=2,
                             dim_feedforward=16, dropout=0)
        normalizer = {'mean': [1., 2., 3., 173.], 'std': [2., 3., 4., 91.]}
        torch.manual_seed(20)
        linear = TimedAssociationModel(config, 'contrastive', max_observations=5)
        next_random = torch.rand(16)
        torch.manual_seed(20)
        circular = TimedAssociationModel(replace(config, heading_representation='circular_sincos'),
            'contrastive', max_observations=5, normalizer=normalizer)
        torch.testing.assert_close(torch.rand(16), next_random, rtol=0, atol=0)
        for key, value in linear.state_dict().items():
            actual = circular.state_dict()[key]
            if key == 'input_embedding.weight':
                actual = actual[:, :4]
            torch.testing.assert_close(actual, value, rtol=0, atol=0, msg=key)
        self.assertNotIn('heading_mean_std', linear.state_dict())
        self.assertEqual(circular.input_embedding.in_features, 5)
        self.assertEqual(sum(p.numel() for p in circular.parameters()) -
                         sum(p.numel() for p in linear.parameters()), config.d_model)
        circular.set_completion_trainable(False)
        self.assertFalse(circular.input_embedding.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in circular.tam_encoder.parameters()))

    def test_circular_heading_restores_degrees_and_keeps_original_time_frequencies(self):
        normalizer = {'mean': [1., 2., 3., 173.], 'std': [2., 3., 4., 91.]}
        torch.manual_seed(20)
        config = ModelConfig(length=6, d_model=8, nhead=2, num_layers=1,
            dim_feedforward=16, dropout=0, heading_representation='circular_sincos')
        model = TimedAssociationModel(config, max_observations=5, normalizer=normalizer).double()
        degrees = torch.tensor([0., 360., 359., 1., 180.], dtype=torch.float64)
        states = torch.tensor([.2, -.3, .7, 0.], dtype=torch.float64).repeat(5, 1)
        states[:, 3] = (degrees - 173.) / 91.
        times = torch.full((5,), 7.3, dtype=torch.float64)
        embedded = model.embed_states(states, times)
        torch.testing.assert_close(embedded[0], embedded[1], rtol=0, atol=1e-14)
        self.assertLess(float((embedded[2] - embedded[3]).detach().norm()),
                        float((embedded[0] - embedded[4]).detach().norm()) / 10)
        # Independent physical-angle construction fixes the time protocol:
        # PE4 is retained, and the newly added cosine has a zero time channel.
        physical = torch.cat((states[:, :3], torch.deg2rad(degrees).sin()[:, None],
                              torch.deg2rad(degrees).cos()[:, None]), dim=-1)
        time = torch.cat((continuous_time_encoding(times, 4), times[:, None] * 0), dim=-1)
        expected = F.relu(F.linear(physical + time, model.input_embedding.weight))
        torch.testing.assert_close(embedded, expected, rtol=1e-13, atol=1e-14)
        linear = self.model().double()
        torch.testing.assert_close(linear.embed_states(states, times),
            F.relu(linear.input_embedding(states + continuous_time_encoding(times, 4))), rtol=0, atol=0)

    def test_circular_normalizer_is_required_fixed_and_persisted(self):
        config = replace(self.model().config, heading_representation='circular_sincos')
        normalizer = {'mean': [1., 2., 3., 173.], 'std': [2., 3., 4., 91.]}
        with self.assertRaisesRegex(ValueError, 'native-time'):
            AssociationModel(config)
        for invalid in (None, {}, {'mean': [0.] * 4, 'std': [1., 1., 1., 0.]},
                        {'mean': [0., 0., 0., float('nan')], 'std': [1.] * 4}):
            with self.subTest(normalizer=invalid), self.assertRaises(ValueError):
                TimedAssociationModel(config, max_observations=5, normalizer=invalid)
        model = TimedAssociationModel(config, max_observations=5, normalizer=normalizer)
        model.validate_normalizer(normalizer)
        with self.assertRaisesRegex(ValueError, 'normalizer'):
            model.validate_normalizer({**normalizer, 'mean': [1., 2., 3., 174.]})
        restored = TimedAssociationModel(config, max_observations=5, normalizer=normalizer)
        restored.load_state_dict(model.state_dict())
        restored.validate_normalizer(normalizer)
        with torch.no_grad():
            restored.heading_mean_std[1] += 1.
        with self.assertRaisesRegex(ValueError, 'normalizer'):
            restored.validate_normalizer(normalizer)

    def test_circular_invalid_tokens_stay_ignored_during_completion_and_training(self):
        config = replace(self.model().config, heading_representation='circular_sincos')
        model = TimedAssociationModel(config, max_observations=5,
            normalizer={'mean': [0., 0., 0., 180.], 'std': [1., 1., 1., 90.]}).eval()
        values, observed = self.packed()
        changed = values.clone()
        changed[:, :5][~observed[:, :5]] = float('inf')
        outside = torch.isnan(values[:, 5:, 0])
        changed[:, 5:, :4][outside] = float('inf')
        torch.testing.assert_close(model.complete(values, observed), model.complete(changed, observed),
                                   rtol=0, atol=0)
        torch.testing.assert_close(model.completion_loss(values, observed),
                                   model.completion_loss(changed, observed), rtol=0, atol=0)
        model.input_embedding.weight.requires_grad_(False)
        model.completion_loss(values, observed).backward()
        self.assertIsNone(model.input_embedding.weight.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.tcm_encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
