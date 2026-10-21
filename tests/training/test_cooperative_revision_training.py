"""Behavioral integration checks for v2 training/statistics/checkpoint isolation."""
import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from association.models.cooperative import CooperativeConfig, CooperativeModel
from association.data.cooperative import fit_cooperative_normalizer, normalize_cooperative_scene
from association.training.cooperative import (
    _snapshot, load_cooperative_checkpoint, stage_parameters, train_batch, train_cooperative,
    training_configuration,
)
from association.models.fusion import FusionLoss
from tests.fixtures.scenes import timed_scene


ROUTES = ['plain', 'scalar-u', 'attribute-u', 'learned-attribute']


class CooperativeRevisionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.scene = timed_scene('train-v2', 'T_core')
        self.validation = timed_scene('validation-v2', 'V_select')
        self.config = CooperativeConfig(length=4, max_observations=3, d_model=8,
            nhead=2, num_layers=1, dim_feedforward=16, dropout=0., head_hidden=6,
            model_variant='support_residual_v2', support_time_policy='train_absmax_v1')
        self.normalizer = fit_cooperative_normalizer([self.scene], self.config)
        torch.manual_seed(23)
        self.model = CooperativeModel(self.config, self.normalizer, support_time_scale=3.)

    def items(self):
        values, observed = normalize_cooperative_scene(self.scene, self.normalizer, self.config)
        result = []
        for nodes, label in (([0, 1], 1.), ([0, 2], 0.)):
            targets = [self.scene['fusion_targets'][str(self.scene['identities'][i])] for i in nodes]
            result.append(dict(values=torch.from_numpy(values[nodes]),
                observed=torch.from_numpy(observed[nodes]),
                target=torch.tensor(np.stack([t['states'] for t in targets]), dtype=torch.float32),
                valid=torch.tensor(np.stack([t['valid'] for t in targets])), label=label,
                removed=[0, 0], scene_id='train-v2', pair_row=len(result)))
        return result

    def optimizer(self, model, objective, stage, route='plain'):
        parameters = stage_parameters(model, stage, routing=route)
        if stage in ('teacher', 'adapt'):
            parameters += list(objective.parameters())
        return torch.optim.Adam(parameters, lr=.001)

    def fit_config(self):
        return dict(seed=20, batch_size=2, teacher_updates=2, utility_updates=1,
            adapt_updates=1, learning_rate=.001, gradient_clip=1., endpoint_fraction=.25,
            utility_epsilon=.001, error_scale_policy='training_native_std', routes=ROUTES,
            max_seconds=30., batch_tracks=2, pair_batch_size=3)

    def test_log_speed_statistics_use_only_retained_training_native_points(self):
        mask = self.scene['native_observed']
        native = self.scene['native_values']
        sample = np.array([0., 1e-12, 1., 20., 1e6, 4., 5., 6., 7., 8., 9.])
        self.assertEqual(mask.sum(), len(sample))
        native[..., 2][mask] = sample
        # Keep the legacy registered observed view consistent with the native
        # synthetic speeds; this test changes no physical validity contract.
        for row in range(len(native)):
            keep = mask[row]
            grid_keep = self.scene['observed'][row]
            self.scene['values'][row, grid_keep, 2] = np.interp(
                self.scene['grid'][grid_keep], self.scene['native_times'][row, keep],
                native[row, keep, 2])
        expected = np.log1p(sample)
        norm = fit_cooperative_normalizer([self.scene], self.config)
        stats = norm['cooperative_representation']
        self.assertAlmostEqual(stats['log_speed_mean'], float(expected.mean()), places=13)
        self.assertAlmostEqual(stats['log_speed_std'], float(expected.std()), places=13)
        self.assertEqual(stats['observed_point_count'], len(sample))
        before = copy.deepcopy(norm)
        for target in self.scene['fusion_targets'].values():
            target['states'][:] = float('nan')
        # Original assembled padding remains safe-zero; packed hidden NaN
        # isolation is covered separately at the model boundary.
        self.assertEqual(fit_cooperative_normalizer([self.scene], self.config), before)
        validation = copy.deepcopy(self.validation)
        validation['native_values'][validation['native_observed'], 2] = 1e12
        values, observed = normalize_cooperative_scene(validation, norm, self.config)
        self.assertTrue(np.isfinite(values[:, :3][validation['native_observed']]).all())
        self.assertTrue(np.isfinite(values[:, 3:, 4]).all())
        self.assertEqual(norm, before)  # V packing cannot refit any statistic.
        for bad_scenes in ([validation], [self.scene, validation]):
            with self.assertRaises(ValueError):
                fit_cooperative_normalizer(bad_scenes, self.config)
        actual, _ = normalize_cooperative_scene(self.scene, norm, self.config)
        np.testing.assert_array_equal(actual[:, :3, 2][mask],
            ((expected-expected.mean())/expected.std()).astype(np.float32))
        # Zero and tiny positive values use raw float64 log1p, with no inverse
        # FP32 native-speed rounding and no clipping of the 1e6 tail.
        self.assertGreater(actual[:, :3, 2][mask][4], actual[:, :3, 2][mask][3])
        self.assertFalse(observed[:, 3:].any())

    def test_constant_zero_speed_and_invalid_speed_contract(self):
        self.scene['native_values'][self.scene['native_observed'], 2] = 0.
        norm = fit_cooperative_normalizer([self.scene], self.config)
        representation = norm['cooperative_representation']
        self.assertEqual(representation['log_speed_mean'], 0.)
        self.assertEqual(representation['log_speed_std'], 1.)
        self.assertTrue(representation['log_speed_constant'])
        self.assertEqual(representation['typical_speed'], 1e-6)
        values, observed = normalize_cooperative_scene(self.scene, norm, self.config)
        self.assertTrue((values[:, :3, 2][self.scene['native_observed']] == 0).all())
        model = CooperativeModel(self.config, norm, support_time_scale=3.).eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(values[:1]), torch.from_numpy(observed[:1]))
        torch.testing.assert_close(pred[:, 2], torch.full((4,), 1e-6), rtol=2e-6, atol=1e-12)
        for value in (-1e-12, float('nan'), float('inf')):
            bad = copy.deepcopy(self.scene); bad['native_values'][0, 0, 2] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                fit_cooperative_normalizer([bad], self.config)
            with self.subTest(pack_value=value), self.assertRaises(ValueError):
                normalize_cooperative_scene(bad, norm, self.config)


    def test_stage_whitelists_include_new_modules_and_freeze_query_pool_after_teacher(self):
        teacher = {'input_projection', 'encoder', 'support_projection', 'query_pool', 'decoder',
                   'state_projection', 'fusion_projection', 'role_embedding', 'identity_base', 'local_identity'}
        adapt = {'decoder', 'state_projection', 'role_embedding', 'fusion_projection'}
        cases = [('teacher', 'plain', teacher), ('utility', 'plain', {'utility_head'})]
        cases += [('adapt', route, adapt | ({'ordinary_gate'} if route == 'learned-attribute' else set()))
                  for route in ROUTES]
        for stage, route, expected in cases:
            with self.subTest(stage=stage, route=route):
                members = stage_parameters(self.model, stage, routing=route)
                actual = {name.split('.')[0] for name, parameter in self.model.named_parameters()
                          if parameter.requires_grad}
                self.assertEqual(actual, expected)
                self.assertEqual({id(x) for x in members},
                    {id(p) for p in self.model.parameters() if p.requires_grad})
                self.assertEqual(len(members), len({id(x) for x in members}))

    def test_teacher_updates_connect_new_support_local_and_fusion_modules(self):
        objective = FusionLoss(position_loss_mode='standard_smooth_l1')
        before = {name:p.detach().clone() for name,p in self.model.named_parameters()}
        optimizer = self.optimizer(self.model, objective, 'teacher')
        # Zero-initialized output/local last rows intentionally block some
        # upstream gradients on the first step. Two steps test the resumed path.
        items = self.items()
        # The other fixtures have overlapping spans but no common-range real
        # token on one side, so they correctly contribute no local evidence.
        values, observed = normalize_cooperative_scene(self.scene, self.normalizer, self.config)
        target = self.scene['fusion_targets'][str(self.scene['identities'][2])]
        items.append(dict(values=torch.from_numpy(values[[2, 3]]),
            observed=torch.from_numpy(observed[[2, 3]]),
            target=torch.tensor(np.stack([target['states']]*2), dtype=torch.float32),
            valid=torch.tensor(np.stack([target['valid']]*2)), label=1.,
            removed=[0, 0], scene_id='train-v2', pair_row=5))
        self.assertTrue(bool(self.model.local_pair_features(
            self.model.encode(items[-1]['values'], items[-1]['observed']), [[0, 1]])[1][0]))
        for _ in range(2):
            train_batch(self.model, objective, optimizer, items, stage='teacher')
        changed = {name for name,p in self.model.named_parameters() if not torch.equal(p, before[name])}
        for module in ('input_projection', 'encoder', 'support_projection', 'query_pool',
                       'local_identity', 'fusion_projection', 'state_projection'):
            with self.subTest(module=module):
                self.assertTrue(any(name.startswith(module+'.') for name in changed))
        self.assertFalse(any(name.startswith(('utility_head.', 'ordinary_gate.', 'identity_with_utility.'))
                             for name in changed))


    def test_utility_step_changes_only_u_and_not_frozen_teacher_or_query_pool(self):
        model = copy.deepcopy(self.model)
        teacher = copy.deepcopy(self.model).eval().requires_grad_(False)
        objective = FusionLoss(position_loss_mode='standard_smooth_l1')
        before = {name:p.detach().clone() for name,p in model.named_parameters()}
        teacher_before = copy.deepcopy(teacher.state_dict())
        objective_before = copy.deepcopy(objective.state_dict())
        optimizer = self.optimizer(model, objective, 'utility')
        report = train_batch(model, objective, optimizer, self.items(), stage='utility',
            teacher=teacher, error_scales=self.normalizer['std'])
        changed = {name for name,p in model.named_parameters() if not torch.equal(p, before[name])}
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith('utility_head.') for name in changed))
        for name,value in teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
        for name,value in objective.state_dict().items():
            torch.testing.assert_close(value, objective_before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertEqual(report['loss_components']['physical'], 0.)
        self.assertEqual(report['loss_components']['identity_bce'], 0.)


    def test_motion_teacher_updates_and_restores_explicit_base_without_reinterpreting_weights(self):
        config = replace(self.config, position_base_policy='native_ols',
                         speed_output_policy='scaled_log', heading_output_policy='observation_residual')
        fit = {**self.fit_config(), 'utility_updates': 0, 'adapt_updates': 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'teacher'
            run = train_cooperative([self.scene], [self.validation], config, fit, root,
                                    device='cpu', phase='teacher', save_predictions=True)
            self.assertEqual(run['completed_updates'], 2)
            self.assertEqual(run['model_config']['position_base_policy'], 'native_ols')
            restored, payload = load_cooperative_checkpoint(root/'teacher.pt', device='cpu')
            self.assertTrue(payload['optimizer_state']['state'])
            # These position head rows initialize to exactly zero. Their update
            # proves physical training actually traverses the new base path.
            self.assertGreater(float(restored.state_projection.weight[:2].detach().abs().max()), 0.)
            again, _ = load_cooperative_checkpoint(root/'teacher.pt', device='cpu')
            values, observed = normalize_cooperative_scene(self.validation, payload['normalizer'], config)
            with torch.no_grad():
                x, mask = torch.from_numpy(values[:1]), torch.from_numpy(observed[:1])
                torch.testing.assert_close(restored(x, mask), again(x, mask), rtol=0, atol=0)
            changed = copy.deepcopy(payload)
            del changed['model_config']['position_base_policy']
            torch.save(changed, root/'mislabeled.pt')
            with self.assertRaisesRegex(ValueError, 'position_base_policy'):
                load_cooperative_checkpoint(root/'mislabeled.pt', device='cpu')

    def test_scene_protocol_teacher_retains_real_components_and_split_contract(self):
        train, validation = copy.deepcopy(self.scene), copy.deepcopy(self.validation)
        validation['component_id'] = train['component_id']
        fit = {**self.fit_config(), 'utility_updates': 0, 'adapt_updates': 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'components overlap'):
                train_cooperative([train], [validation], self.config, fit,
                                  root/'legacy-overlap', device='cpu', phase='teacher')
            train['split_unit'] = 'scene'
            with self.assertRaisesRegex(ValueError, 'split_unit'):
                train_cooperative([train], [validation], self.config, fit,
                                  root/'mixed-protocol', device='cpu', phase='teacher')
            validation['split_unit'] = 'scene'
            run = train_cooperative([train], [validation], self.config, fit,
                                    root/'scene-teacher', device='cpu', phase='teacher')
            self.assertEqual(run['completed_updates'], fit['teacher_updates'])
            _, checkpoint = load_cooperative_checkpoint(root/'scene-teacher'/'teacher.pt', device='cpu')
            self.assertEqual(checkpoint['normalizer']['training_scene_ids'], [train['scene_id']])
            for population in run['scene_contract']:
                self.assertEqual(population[0]['split_unit'], 'scene')
                self.assertEqual(population[0]['component_id'], train['component_id'])
