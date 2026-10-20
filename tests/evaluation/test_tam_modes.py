"""A mode diagnostic cannot manufacture stochastic gaps or mutate its source."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from association.evaluation.tam_modes import MODES, inspect_tam_modes
from association.models.tmtaff import AssociationModel, ModelConfig


class TamModesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.model = AssociationModel(ModelConfig(length=4, d_model=8, nhead=2,
            num_layers=1, dim_feedforward=16, dropout=.35, margin=7.), 'contrastive')
        rng = np.random.default_rng(7)
        self.samples = [dict(metadata=dict(scene_id='x', split='T_core'),
            values=rng.normal(size=(3, 4, 4)).astype(np.float32),
            observed=np.ones((3, 4), dtype=bool), pairs=np.array([[0, 1], [0, 2]]),
            labels=np.array([1., 0.], dtype=np.float32))]

    def test_same_input_mc_raw_loss_and_all_dropout_disabled_without_source_mutation(self):
        self.model.train()
        self.model.tcm_decoder.eval()  # Deliberately heterogeneous caller modes.
        parameters = list(self.model.parameters())
        parameters[0].grad = torch.ones_like(parameters[0])
        state = copy.deepcopy(self.model.state_dict())
        modes = [m.training for m in self.model.modules()]
        original_rng = torch.get_rng_state().clone()
        original_grad = parameters[0].grad
        original_fastpath = torch.backends.mha.get_fastpath_enabled()
        attention_calls, completion_calls = [], []

        def attention_hook(module, inputs):
            attention_calls.append((module.training, module.dropout))

        def completion(values, observed):
            completion_calls.append(len(values))
            self.assertFalse(self.model.tcm_encoder.training)
            self.assertFalse(self.model.tcm_decoder.training)
            self.assertFalse(torch.is_grad_enabled())
            return original_complete(values, observed)

        original_complete = self.model.complete
        handles = [m.register_forward_pre_hook(attention_hook) for m in self.model.tam_encoder.modules()
                   if isinstance(m, nn.MultiheadAttention)]
        try:
            with patch.object(self.model, 'complete', side_effect=completion):
                report, arrays = inspect_tam_modes(self.model, self.samples, repeats=8)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(completion_calls, [3])
        self.assertTrue(report['inference_mode_during_inspection'])
        self.assertEqual(report['default_fastpath_eval_reference']['repeats'], 1)
        self.assertTrue(np.isfinite(arrays['eval_default_fastpath_sample_0_distances']).all())
        self.assertTrue(all(report['state_restoration'].values()))
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertIs(original_grad, parameters[0].grad)
        self.assertTrue(torch.equal(parameters[0].grad, torch.ones_like(parameters[0])))
        self.assertEqual(original_fastpath, torch.backends.mha.get_fastpath_enabled())
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[key]) for key, value in state.items()))
        self.assertIn((True, .35), attention_calls)
        self.assertIn((True, 0.), attention_calls)
        for mode in ('eval', 'train_all_dropout_zero'):
            noise = arrays[f'{mode}_sample_0_same_track_noise']
            np.testing.assert_array_equal(noise, np.zeros_like(noise))
            self.assertEqual(report['modes'][mode]['samples'][0]['repeat_pair_distance_std_mean'], 0.)
        self.assertGreater(arrays['train_sample_0_same_track_noise'].mean(), 0.)
        np.testing.assert_array_equal(arrays['eval_sample_0_distances'],
                                      arrays['train_all_dropout_zero_sample_0_distances'])
        self.assertEqual(arrays['train_sample_0_distances'].shape, (8, 2))
        for mode in MODES:
            distances = arrays[f'{mode}_sample_0_distances']
            expected = .25*(distances[:, 0] + np.maximum(7.-distances[:, 1], 0.)**2)
            actual = report['modes'][mode]['samples'][0]
            np.testing.assert_allclose(actual['class_balanced_loss_per_repeat'], expected, rtol=2e-7)
            np.testing.assert_allclose(actual['model_original_loss_per_repeat'], expected, rtol=2e-7)
            self.assertAlmostEqual(actual['class_balanced_loss'], float(expected.mean()), places=5)

    def test_exception_restores_rng_modes_grad_flags_and_attention_dropout(self):
        self.model.train()
        self.model.tcm_encoder.eval()
        parameters = list(self.model.parameters())
        parameters[0].requires_grad_(False)
        parameters[1].grad = torch.ones_like(parameters[1])
        state = copy.deepcopy(self.model.state_dict())
        modes = [m.training for m in self.model.modules()]
        flags = [p.requires_grad for p in parameters]
        dropout = [(m, m.p) for m in self.model.modules() if isinstance(m, nn.Dropout)]
        attention = [(m, m.dropout) for m in self.model.modules() if isinstance(m, nn.MultiheadAttention)]
        original_rng = torch.get_rng_state().clone()
        original_forward = self.model.tam_encoder.forward

        def fail_when_train(*args, **kwargs):
            if self.model.tam_encoder.training and all(m.dropout == 0. for m, _ in attention
                    if m in self.model.tam_encoder.modules()):
                raise RuntimeError('injected TAM failure')
            return original_forward(*args, **kwargs)

        with patch.object(self.model.tam_encoder, 'forward', side_effect=fail_when_train):
            with self.assertRaisesRegex(RuntimeError, 'injected TAM failure'):
                inspect_tam_modes(self.model, self.samples, repeats=2)
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertEqual(flags, [p.requires_grad for p in parameters])
        self.assertTrue(all(m.p == value for m, value in dropout))
        self.assertTrue(all(m.dropout == value for m, value in attention))
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[key]) for key, value in state.items()))
        self.assertTrue(torch.equal(parameters[1].grad, torch.ones_like(parameters[1])))


if __name__ == '__main__':
    unittest.main()
