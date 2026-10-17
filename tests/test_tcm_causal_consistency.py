"""TCM prefix semantics across teacher forcing and actual endpoint rollout."""

import unittest
from unittest.mock import patch

import torch

from association.models import AssociationModel, ModelConfig


class TCMPrefixConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20)

    def model(self, layers=3, dropout=0):
        return AssociationModel(ModelConfig(
            length=8, d_model=8, nhead=2, num_layers=layers,
            dim_feedforward=16, dropout=dropout,
        ))

    def test_first_rollout_step_equals_teacher_for_single_and_multiple_layers(self):
        for layers in (1, 3):
            model = self.model(layers).eval()
            for length in (1, 4):
                with self.subTest(layers=layers, length=length), torch.no_grad():
                    history = torch.randn(2, length, 8)
                    teacher = model._teacher_predictions(history)[:, -1:]
                    rollout = model._extrapolate(history, 1)
                    torch.testing.assert_close(rollout, teacher, rtol=1e-5, atol=1e-6)

    def test_future_teacher_values_cannot_change_earlier_multilayer_predictions(self):
        model = self.model().eval()
        history = torch.randn(2, 6, 8)
        changed = history.clone()
        changed[:, 3:] += 100 * torch.randn_like(changed[:, 3:])
        with torch.no_grad():
            expected = model._teacher_predictions(history)
            actual = model._teacher_predictions(changed)
            prefix_only = model._teacher_predictions(history[:, :3])
        torch.testing.assert_close(actual[:, :3], expected[:, :3], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(prefix_only, expected[:, :3], rtol=1e-5, atol=1e-6)
        self.assertGreater((actual[:, 3:] - expected[:, 3:]).abs().max().item(), 1e-4)

    def test_rectangular_cross_attention_keeps_observed_prefix_queries_causal(self):
        model = self.model().eval()
        history = torch.randn(2, 5, 8)
        # This directly exercises the longer-history/fixed-memory rollout
        # geometry. Memory token 2 is future to queries 0 and 1, but available
        # to generated queries 3 and 4.
        memory = torch.randn(2, 3, 8)
        changed = memory.clone()
        changed[:, 2] += 100 * torch.randn_like(changed[:, 2])
        with torch.no_grad():
            expected = model._decode_causal_history(history, memory)
            actual = model._decode_causal_history(history, changed)
        torch.testing.assert_close(actual[:, :2], expected[:, :2], rtol=1e-5, atol=1e-6)
        self.assertGreater((actual[:, 3:] - expected[:, 3:]).abs().max().item(), 1e-4)

    def test_rollout_uses_fixed_memory_and_rectangular_mask_after_first_step(self):
        model = self.model().eval()
        history = torch.randn(2, 3, 8)
        with patch.object(model.tcm_decoder, "forward", wraps=model.tcm_decoder.forward) as decoder:
            with torch.no_grad():
                result = model._extrapolate(history, 3)
        self.assertEqual(tuple(result.shape), (2, 3, 8))
        masks = [call.kwargs["memory_mask"] for call in decoder.call_args_list]
        self.assertEqual([tuple(mask.shape) for mask in masks], [(3, 3), (4, 3), (5, 3)])
        torch.testing.assert_close(masks[-1], torch.tensor([
            [False, True, True],
            [False, False, True],
            [False, False, False],
            [False, False, False],
            [False, False, False],
        ]))
        self.assertEqual([call.args[1].shape[1] for call in decoder.call_args_list], [3, 3, 3])

    def test_both_endpoint_directions_start_from_the_corresponding_teacher_prediction(self):
        model = self.model().eval()
        values = torch.randn(2, 8, 4)
        observed = torch.zeros(2, 8, dtype=torch.bool)
        observed[:, 2:5] = True
        values[~observed] = float("nan")
        with torch.no_grad():
            embedded = model.embed_inputs(values, observed)
            sequence = embedded[:, 2:5]
            expected_before = model._teacher_predictions(sequence.flip(1))[:, -1]
            expected_after = model._teacher_predictions(sequence)[:, -1]
            completed = model.complete(values, observed)
        self.assertTrue(torch.equal(completed[observed], embedded[observed]))
        self.assertTrue(torch.isfinite(completed).all())
        torch.testing.assert_close(completed[:, 1], expected_before, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(completed[:, 5], expected_after, rtol=1e-5, atol=1e-6)

    def test_multistep_gradients_and_frozen_completion_modes(self):
        model = self.model(dropout=.2)
        values = torch.randn(3, 8, 4)
        observed = torch.zeros(3, 8, dtype=torch.bool)
        observed[:, 2:5] = True
        pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])

        model.set_completion_trainable(False)
        model.train()
        self.assertFalse(model.tcm_encoder.training)
        self.assertFalse(model.tcm_decoder.training)
        first = model.complete(values, observed)
        second = model.complete(values, observed)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        scores = model.score_pairs(model.encode_tracks(values, observed), pairs)
        model.association_loss(scores, torch.tensor([1., 0., 0.])).backward()
        self.assertTrue(all(p.grad is None for p in model.completion_parameters(True)))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.tam_encoder.parameters()))

        model.zero_grad(set_to_none=True)
        model.set_completion_trainable(True)
        model.eval()
        completed = model.complete(values, observed)
        # Only generated endpoints contribute, so gradients cannot reach the
        # embedding merely through the unchanged observed output positions.
        weights = torch.randn_like(completed[~observed])
        (completed[~observed] * weights).sum().backward()
        for module in (model.input_embedding, model.tcm_encoder, model.tcm_decoder):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)


if __name__ == "__main__":
    unittest.main()
