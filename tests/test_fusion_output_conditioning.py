"""Explicit physical output conditioning and historical snapshot compatibility."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from association import runtime
from association.complete_training import STAGES, load_complete_snapshot, train_complete
from association.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.models import ModelConfig
from tests.test_timed_pipeline import timed_scene


class FusionOutputConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.raw_config = PairFusionConfig(fc_hidden=8, length=4, d_model=8, nhead=2,
            num_layers=1, dim_feedforward=16, dropout=.1)
        self.affine_config = replace(self.raw_config, output_parameterization='train_position_affine')
        self.normalizer = dict(mean=[20., 131., 5., 180.], std=[.5, .6, 2., 90.])
        torch.manual_seed(71)
        self.left, self.right = torch.randn(2, 4, 8), torch.randn(2, 4, 8)

    def test_only_position_maps_to_physical_units_speed_and_heading_stay_raw(self):
        model = PairFusion(self.affine_config, self.normalizer).eval()
        with torch.no_grad():
            model.state_projection.weight.zero_()
            model.state_projection.bias.zero_()
            zero = model(self.left, self.right)['states']
            physical = torch.tensor([20., 131., F.softplus(torch.tensor(0.)).item(), 0.])
            torch.testing.assert_close(zero, physical.expand_as(zero), rtol=0, atol=0)
            model.state_projection.bias.copy_(torch.tensor([1., -2., .5, 3.]))
            actual = model(self.left, self.right)['states']
            expected = torch.tensor([20.5, 129.8, .5, 3.])
            expected[2] = F.softplus(expected[2])
            torch.testing.assert_close(actual, expected.expand_as(actual), rtol=0, atol=0)
            # No new heading wrap or geographic clamp is part of this adaptation.
            model.state_projection.bias[0] = 200.
            self.assertTrue((model(self.left, self.right)['states'][..., 0] > 90.).all())

    def test_raw_default_keeps_keys_rng_initialization_and_outputs(self):
        torch.manual_seed(18)
        raw = PairFusion(self.raw_config).eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(18)
        explicit = PairFusion(replace(self.raw_config, output_parameterization='raw'), self.normalizer).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertNotIn('position_mean', raw.state_dict())
        self.assertNotIn('position_std', raw.state_dict())
        self.assertEqual(raw.state_dict().keys(), explicit.state_dict().keys())
        for name, value in raw.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(raw(self.left, self.right)['states'],
                                       explicit(self.left, self.right)['states'], rtol=0, atol=0)
        torch.manual_seed(18)
        affine = PairFusion(self.affine_config, self.normalizer)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in raw.parameters()), sum(p.numel() for p in affine.parameters()))
        self.assertEqual(set(affine.state_dict()) - set(raw.state_dict()), {'position_mean', 'position_std'})
        self.assertNotIn('position_mean', dict(affine.named_parameters()))
        self.assertFalse(affine.position_mean.requires_grad)
        self.assertFalse(affine.position_std.requires_grad)

    def test_invalid_or_unrepresentable_statistics_fail_without_fallback(self):
        cases = [None, {}, dict(mean=[0.]*3, std=[1.]*4),
                 dict(mean=[0.]*4, std=[[1.]*4]), dict(mean=[0., 0., float('nan'), 0.], std=[1.]*4),
                 dict(mean=[0.]*4, std=[1., 0., 1., 1.]),
                 dict(mean=[0.]*4, std=[1., -1., 1., 1.]),
                 dict(mean=[0.]*4, std=[1., float('inf'), 1., 1.]),
                 dict(mean=[1e100]*4, std=[1.]*4), dict(mean=[0.]*4, std=[1e-100]*4)]
        for normalizer in cases:
            with self.subTest(normalizer=normalizer), self.assertRaisesRegex(ValueError, 'train_position_affine'):
                PairFusion(self.affine_config, normalizer)
        with self.assertRaisesRegex(ValueError, 'output_parameterization'):
            replace(self.raw_config, output_parameterization='automatic')

    def test_normalizer_buffers_are_detached_copies_and_snapshot_cannot_replace_them(self):
        source = dict(mean=torch.tensor(self.normalizer['mean'], requires_grad=True),
                      std=torch.tensor(self.normalizer['std'], requires_grad=True))
        model = PairFusion(self.affine_config, source)
        original = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            source['mean'].add_(10.)
        torch.testing.assert_close(model.position_mean, original['position_mean'], rtol=0, atol=0)
        for key in ('position_mean', 'position_std'):
            for kind in ('changed', 'missing', 'wrong_shape', 'nonfinite'):
                with self.subTest(key=key, kind=kind):
                    state = copy.deepcopy(original)
                    if kind == 'changed':
                        state[key][0] += 1.
                    elif kind == 'missing':
                        del state[key]
                    elif kind == 'wrong_shape':
                        state[key] = state[key][:1]
                    else:
                        state[key][0] = float('nan')
                    state['state_projection.weight'].add_(1.)
                    with self.assertRaisesRegex(ValueError, 'disagrees with its normalizer'):
                        model.load_state_dict(state, strict=False)
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        # The values originate in the same float32 constructor, even when moved
        # to another precision before restore.
        model.double().load_state_dict(original)
        torch.testing.assert_close(model.position_mean, original['position_mean'].double(), rtol=0, atol=0)

    def test_real_forward_backward_has_finite_gradients_and_fixed_statistics(self):
        model = PairFusion(self.affine_config, self.normalizer).train()
        left, right = self.left.clone().requires_grad_(), self.right.clone().requires_grad_()
        states = model(left, right)['states']
        target = torch.tensor(self.normalizer['mean']).expand_as(states).clone()
        objective = FusionLoss(position_loss_mode='standard_smooth_l1')
        loss = objective(states, target, torch.ones_like(states, dtype=torch.bool))['total']
        loss.backward()
        for parameter in (left, right, model.state_projection.weight,
                          model.fc1.weight, model.decoder.layers[0].linear1.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        torch.testing.assert_close(model.position_mean, torch.tensor(self.normalizer['mean'][:2]), rtol=0, atol=0)
        self.assertIsNone(model.position_mean.grad)
        self.assertIsNone(model.position_std.grad)

    def test_extreme_speed_statistics_do_not_change_speed_outputs_or_gradients(self):
        extreme = copy.deepcopy(self.normalizer)
        # Actual 014 train input speed moments. Mapping z=-1 through these
        # moments would put softplus at -306 and lose its FP32 gradient.
        extreme['mean'][2:] = [109.00519634850676, 177.95173703514496]
        extreme['std'][2:] = [415.2732000413263, 104.69449368349127]
        torch.manual_seed(11)
        first = PairFusion(self.affine_config, self.normalizer).eval()
        second = PairFusion(self.affine_config, extreme).eval()
        second.load_state_dict(first.state_dict())
        with torch.no_grad():
            for model in (first, second):
                model.state_projection.weight.zero_()
                model.state_projection.bias.copy_(torch.tensor([0., 0., -1., 3.]))
        results = []
        for model in (first, second):
            states = model(self.left, self.right)['states']
            states[..., 2].mean().backward()
            results.append((states.detach(), model.state_projection.bias.grad.clone()))
            torch.testing.assert_close(states[..., 2], F.softplus(torch.tensor(-1.)).expand_as(states[..., 2]))
            self.assertGreater(float(model.state_projection.bias.grad[2]), .26)
            torch.testing.assert_close(model.state_projection.bias.grad[2], torch.sigmoid(torch.tensor(-1.)))
        for actual, expected in zip(results[0], results[1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class FusionOutputSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.train = [timed_scene('training', 'T_core')]
        self.validation = [timed_scene('validation', 'V_select')]
        self.model = ModelConfig(length=4, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=.1)
        self.fusion = PairFusionConfig(length=4, d_model=8, nhead=2, num_layers=1,
            dim_feedforward=16, dropout=.1, fc_hidden=8, output_parameterization='train_position_affine')
        self.config = dict(updates=dict.fromkeys(STAGES, 2), learning_rates=dict.fromkeys(STAGES, .001),
            seed=13, branch='ce', batch_tracks=3, pair_batch_size=4, pair_microbatch=2,
            track_microbatch=2, completion_weight=.1, gradient_clip=1., max_hours=.1,
            validate_every_updates=1, completion_cache_mib=1,
            position_loss_mode='standard_smooth_l1', lr_schedule='constant',
            singleton_policy='observed', validation_batches=dict(tracks=2, pairs=3))

    def fit(self, name, **kwargs):
        return train_complete(self.train, self.validation, self.model, self.fusion,
                              self.config, self.root / name, **kwargs)

    def checkpoint(self, name):
        return torch.load(self.root / name / 'checkpoint.pt', weights_only=True)

    def assert_states_equal(self, left, right):
        for expected, actual in zip(left, right):
            self.assertEqual(set(expected), set(actual))
            for name, value in expected.items():
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0)

    def test_affine_native_four_stages_selected_last_and_exact_resume(self):
        result = self.fit('full')
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['fusion_output_parameterization'], 'train_position_affine')
        self.assertIn('output-scale-adapted', result['scope'])
        full = self.checkpoint('full')
        self.assertEqual(full['normalizer']['training_splits'], ['T_core'])
        self.assertEqual([row['stage'] for row in full['history']], [stage for stage in STAGES for _ in range(2)])
        self.assertTrue(all(np.isfinite(row['loss']) for row in full['history']))
        for selected in (True, False):
            loaded = load_complete_snapshot(self.root / 'full' / 'checkpoint.pt', selected=selected)
            expected = full['selected']['states'] if selected else full['states']
            self.assert_states_equal(expected, [loaded[key].state_dict() for key in ('model', 'fusion', 'objective')])
            torch.testing.assert_close(loaded['fusion'].position_mean,
                torch.tensor(full['normalizer']['mean'][:2], dtype=torch.float32), rtol=0, atol=0)
            torch.testing.assert_close(loaded['fusion'].position_std,
                torch.tensor(full['normalizer']['std'][:2], dtype=torch.float32), rtol=0, atol=0)
        self.assertEqual(self.fit('part', max_updates_this_call=5)['status'], 'INTERRUPTED')
        self.assertEqual(self.fit('resumed', resume_from=self.root/'part')['status'], 'COMPLETED')
        resumed = self.checkpoint('resumed')
        self.assert_states_equal(full['states'], resumed['states'])
        self.assert_states_equal(full['selected']['states'], resumed['selected']['states'])
        self.assertEqual([row['loss'] for row in full['history']], [row['loss'] for row in resumed['history']])
        self.assertEqual(runtime.read_json(self.root/'full'/'summary.json')['evaluation']['fusion'],
                         runtime.read_json(self.root/'resumed'/'summary.json')['evaluation']['fusion'])

    def test_legacy_raw_payload_without_field_loads_selected_last_and_resumes(self):
        self.fusion = replace(self.fusion, output_parameterization='raw')
        self.fit('full')
        self.fit('old-part', max_updates_this_call=5)
        for name in ('full', 'old-part'):
            path = self.root/name/'checkpoint.pt'
            saved = self.checkpoint(name)
            del saved['fusion_config']['output_parameterization']
            torch.save(saved, path)
            for selected in (True, False):
                loaded = load_complete_snapshot(path, selected=selected)
                self.assertEqual(loaded['fusion'].config.output_parameterization, 'raw')
                self.assertNotIn('position_mean', loaded['fusion'].state_dict())
                expected = saved['selected']['states'] if selected else saved['states']
                self.assert_states_equal(expected, [loaded[key].state_dict() for key in ('model','fusion','objective')])
        self.fit('resumed', resume_from=self.root/'old-part')
        self.assert_states_equal(self.checkpoint('full')['states'], self.checkpoint('resumed')['states'])
        self.fusion = replace(self.fusion, output_parameterization='train_position_affine')
        with self.assertRaisesRegex(ValueError, 'resume configuration'):
            self.fit('forbidden-upgrade', resume_from=self.root/'old-part')
        self.assertFalse((self.root/'forbidden-upgrade').exists())

    def test_snapshot_statistics_conflicts_fail_for_selected_last_and_resume(self):
        self.fit('part', max_updates_this_call=5)
        original = self.checkpoint('part')
        for target in ('normalizer', 'last', 'selected'):
            with self.subTest(target=target):
                saved = copy.deepcopy(original)
                if target == 'normalizer':
                    saved['normalizer']['mean'][0] += 1.
                elif target == 'last':
                    saved['states'][1]['position_mean'][0] += 1.
                else:
                    saved['selected']['states'][1]['position_std'][0] += 1.
                path = self.root/f'{target}.pt'
                torch.save(saved, path)
                selected = target != 'last'
                with self.assertRaisesRegex(ValueError, 'disagrees with its normalizer'):
                    load_complete_snapshot(path, selected=selected)
                if target != 'normalizer':
                    torch.save(saved, self.root/'part'/'checkpoint.pt')
                    with (patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('must fail before optimizer update')),
                          self.assertRaisesRegex(ValueError, 'disagrees with its normalizer')):
                        self.fit(f'bad-{target}-resume', resume_from=self.root/'part')
        # Resume's input-moment semantic check rejects changed checkpoint metadata
        # before a new run directory exists.
        changed = copy.deepcopy(original)
        changed['normalizer']['mean'][0] += 1.
        torch.save(changed, self.root/'part'/'checkpoint.pt')
        with self.assertRaisesRegex(ValueError, 'resume configuration'):
            self.fit('bad-resume', resume_from=self.root/'part')
        self.assertFalse((self.root/'bad-resume').exists())


if __name__ == '__main__':
    unittest.main()
