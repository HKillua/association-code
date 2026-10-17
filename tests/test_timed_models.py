"""Native-time conditioning, causal consistency and unchanged learned capacity."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from association.models import AssociationModel, ModelConfig
from association.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.fusion_training import fusion_training_parameters, fusion_training_step
from association.staged import FrozenCompletionCache
from association.timed_models import TimedAssociationModel, continuous_time_encoding


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

    def test_empty_batch_and_continuous_encoding_shapes(self):
        model = self.model()
        self.assertEqual(model.complete(torch.empty(0, 11, 5), torch.empty(0, 11, dtype=torch.bool)).shape, (0, 6, 8))
        self.assertEqual(model.encode_tracks(torch.empty(0, 11, 5), torch.empty(0, 11, dtype=torch.bool)).shape, (0, 48))
        times = torch.tensor([[0., .1]], dtype=torch.float64)
        encoding = continuous_time_encoding(times, 5)
        self.assertEqual(encoding.shape, (1, 2, 5))
        self.assertEqual(encoding.dtype, torch.float64)
        torch.testing.assert_close(encoding[0, 0], torch.tensor([0., 1., 0., 1., 0.], dtype=torch.float64))

    def test_existing_frozen_cache_and_joint_fusion_step_accept_packed_native_inputs(self):
        model = self.model(dropout=.1)
        values, observed = self.packed()
        model.set_completion_trainable(False)
        model.train()
        cache = FrozenCompletionCache(model, [(values.numpy(), observed.numpy())], 'cpu', 10000, 2)
        with torch.no_grad():
            direct = model.complete(values, observed)
        torch.testing.assert_close(cache.get([(0, 0), (0, 1)]), direct, rtol=0, atol=0)
        fusion = PairFusion(PairFusionConfig(fc_hidden=8, length=6, d_model=8, nhead=2,
            num_layers=1, dim_feedforward=16, dropout=0))
        objective = FusionLoss(position_loss_mode='standard_smooth_l1')
        parameters = fusion_training_parameters(model, fusion, objective, stage='JOINT')
        optimizer = torch.optim.SGD(parameters, lr=.001)
        target = torch.tensor([30., 120., 5., 45.]).expand(1, 6, 4).clone()
        report = fusion_training_step(model, fusion, objective, optimizer, stage='JOINT',
            values=values, observed=observed, pairs=torch.tensor([[0, 1]]), labels=torch.ones(1),
            target=target, valid=torch.ones_like(target, dtype=torch.bool), pair_microbatch=1,
            track_microbatch=1, completion_weight=.1, gradient_clip=1.)
        self.assertEqual(report['completion_target_elements'], 48)
        self.assertTrue(torch.isfinite(torch.tensor(report['loss'])))
        for module in (model.tcm_encoder, model.tcm_decoder, model.tam_encoder, fusion):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_bad_time_shape_mask_and_support_inputs_fail_explicitly(self):
        model = self.model()
        for issue in ('shape', 'dtype', 'prefix', 'singleton', 'query_mask', 'native_nan',
                      'native_order', 'query_order', 'query_nan', 'inside_nan'):
            values, observed = self.packed()
            if issue == 'shape':
                values = values[:, :-1]
            elif issue == 'dtype':
                observed = observed.float()
            elif issue == 'prefix':
                observed[0, 1] = False
            elif issue == 'singleton':
                observed[1, 1] = False
            elif issue == 'query_mask':
                observed[0, -1] = True
            elif issue == 'native_nan':
                values[0, 1, 4] = float('nan')
            elif issue == 'native_order':
                values[0, 1, 4] = values[0, 0, 4]
            elif issue == 'query_order':
                values[0, -1, 4] = values[0, -2, 4]
            elif issue == 'query_nan':
                values[0, -1, 4] = float('nan')
            elif issue == 'inside_nan':
                values[0, 7, 0] = float('nan')
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                model.complete(values, observed)
        for cap in (1, 0, True, 2.5):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                TimedAssociationModel(ModelConfig(), max_observations=cap)


if __name__ == '__main__':
    unittest.main()
