"""Behavioral integration checks for v2 training/statistics/checkpoint isolation."""
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.models.cooperative import CooperativeConfig, CooperativeModel
from association.data.cooperative import fit_cooperative_normalizer, normalize_cooperative_scene
from association.training.cooperative import (
    _snapshot, load_cooperative_checkpoint, stage_parameters, train_batch, train_cooperative,
)
from association.models.fusion import FusionLoss
from association.training.common import fit_normalizer, normalize_scene
from association.utils import runtime
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

    def test_original_physical_error_scales_are_not_replaced_by_log_statistics(self):
        physical = fit_normalizer([self.scene])
        self.assertEqual(self.normalizer['std'], physical['std'])
        self.assertEqual(self.normalizer['mean'], physical['mean'])
        legacy = CooperativeConfig(**{**asdict(self.config), 'model_variant':'legacy'})
        old = fit_cooperative_normalizer([self.scene], legacy)
        self.assertEqual(old, physical)
        expected = normalize_scene(self.scene, old)
        actual = normalize_cooperative_scene(self.scene, old, legacy)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)

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

    def test_actual_adaptation_update_preserves_frozen_identity_support_and_pool(self):
        for route in ROUTES:
            with self.subTest(route=route):
                model = copy.deepcopy(self.model)
                objective = FusionLoss(position_loss_mode='standard_smooth_l1')
                # Pretrained nonzero heads make a physical ordinary-gate gradient
                # observable; initial zero rows need not backpropagate through E.
                with torch.no_grad():
                    model.state_projection.weight[:3].fill_(.1)
                before = {name:p.detach().clone() for name,p in model.named_parameters()}
                optimizer = self.optimizer(model, objective, 'adapt', route)
                report = train_batch(model, objective, optimizer, self.items(), stage='adapt', routing=route)
                changed = {name for name,p in model.named_parameters() if not torch.equal(p, before[name])}
                self.assertTrue(any(name.startswith('state_projection.') for name in changed))
                forbidden = ('input_projection.', 'encoder.', 'support_projection.', 'query_pool.',
                             'identity_base.', 'local_identity.', 'identity_with_utility.', 'utility_head.')
                self.assertFalse(any(name.startswith(forbidden) for name in changed))
                if route == 'learned-attribute':
                    self.assertTrue(any(name.startswith('ordinary_gate.') for name in changed))
                else:
                    self.assertFalse(any(name.startswith('ordinary_gate.') for name in changed))
                self.assertEqual(report['used_pairs'], 1)

    def test_four_arm_fit_shares_initialization_sampling_identity_and_new_metadata(self):
        starts, samples, events = {}, {}, []
        original = train_batch
        def capture(model, objective, optimizer, items, **kwargs):
            events.append(kwargs['stage'])
            if kwargs['stage'] == 'adapt':
                route = kwargs['routing']
                starts[route] = copy.deepcopy(model.state_dict())
                samples[route] = [(x['scene_id'], x['pair_row'], x['removed'], x['label']) for x in items]
            return original(model, objective, optimizer, items, **kwargs)
        from association.evaluation.cooperative import score_cooperative
        def score(*args, **kwargs):
            events.append('select_identity')
            return score_cooperative(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'fit'
            with patch('association.training.cooperative.train_batch', side_effect=capture), \
                 patch('association.evaluation.cooperative.score_cooperative', side_effect=score):
                run = train_cooperative([self.scene], [self.validation], self.config,
                                       self.fit_config(), output, device='cpu')
            self.assertEqual(run['status'], 'COMPLETED')
            self.assertEqual(run['completed_updates'], 7)
            self.assertEqual(run['completed_routes'], ROUTES)
            self.assertLess(events.index('select_identity'), events.index('utility'))
            self.assertEqual(events.count('select_identity'), 1)
            self.assertEqual(run['identity_selection_phase'], 'teacher_endpoint_before_utility')
            self.assertEqual(run['error_scales'], self.normalizer['std'])
            self.assertEqual(run['training_parameters']['utility']['names'],
                             ['model.'+name for name,_ in self.model.utility_head.named_parameters(prefix='utility_head')])
            expected_scores = np.load(output/'plain-scores.npz')
            expected_normalizer = None
            for route in ROUTES:
                for name,value in starts['plain'].items():
                    torch.testing.assert_close(starts[route][name], value, rtol=0, atol=0)
                self.assertEqual(samples[route], samples['plain'])
                restored, payload = load_cooperative_checkpoint(output/f'adapt-{route}.pt', device='cpu')
                self.assertEqual(payload['format'], 'cooperative-physical-v2')
                self.assertEqual(payload['model_variant'], 'support_residual_v2')
                self.assertEqual(payload['representation_version'], 'support_residual_v2')
                if expected_normalizer is None:
                    expected_normalizer = payload['normalizer']
                self.assertEqual(payload['normalizer'], expected_normalizer)
                for name,value in restored.state_dict().items():
                    if name.startswith(('encoder.', 'input_projection.', 'query_pool.', 'support_projection.',
                                        'identity_base.', 'local_identity.', 'utility_head.')):
                        torch.testing.assert_close(value, starts[route][name], rtol=0, atol=0)
                scores = np.load(output/f'{route}-scores.npz')
                for key in expected_scores.files:
                    np.testing.assert_array_equal(scores[key], expected_scores[key])
                evaluation = json.loads((output/f'{route}-evaluation.json').read_text())
                self.assertEqual(evaluation['model_variant'], 'support_residual_v2')
                self.assertEqual(evaluation['representation_version'], 'support_residual_v2')
                self.assertFalse(evaluation['test_opened'])
                x, mask = normalize_cooperative_scene(self.validation, payload['normalizer'], self.config)
                with torch.no_grad():
                    first = restored(torch.from_numpy(x[:1]), torch.from_numpy(mask[:1]))
                again, _ = load_cooperative_checkpoint(output/f'adapt-{route}.pt', device='cpu')
                with torch.no_grad():
                    second = again(torch.from_numpy(x[:1]), torch.from_numpy(mask[:1]))
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_checkpoint_rejects_conflicting_v2_version_and_loads_old_legacy_without_variant(self):
        objective = FusionLoss(position_loss_mode='standard_smooth_l1')
        optimizer = self.optimizer(self.model, objective, 'teacher')
        payload = _snapshot(self.model, objective, optimizer, self.normalizer, self.fit_config(),
                            stage='teacher', route='plain', updates=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            for field,value in [('format','cooperative-physical-v1'), ('model_variant','legacy'),
                                ('representation_version','missing')]:
                altered = copy.deepcopy(payload); altered[field] = value
                torch.save(altered,path)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_cooperative_checkpoint(path, device='cpu')
            legacy_config = CooperativeConfig(length=4, max_observations=3, d_model=8,
                nhead=2, num_layers=1, dim_feedforward=16, dropout=0., head_hidden=6)
            norm = fit_normalizer([self.scene])
            legacy = CooperativeModel(legacy_config, norm).eval()
            optimizer = self.optimizer(legacy, objective, 'teacher')
            old = _snapshot(legacy, objective, optimizer, norm, self.fit_config(),
                            stage='teacher', route='plain', updates=0)
            del old['model_config']['model_variant']
            del old['model_config']['support_time_policy']
            del old['model_config']['speed_output_policy']
            del old['support_time_scale_tau']
            torch.save(old,path)
            restored, metadata = load_cooperative_checkpoint(path, device='cpu')
            x, mask = normalize_scene(self.scene, norm)
            with torch.no_grad():
                expected = legacy(torch.from_numpy(x[:1]), torch.from_numpy(mask[:1]))
                actual = restored(torch.from_numpy(x[:1]), torch.from_numpy(mask[:1]))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(metadata['format'], 'cooperative-physical-v1')

    def test_split_shared_and_four_arms_equal_full_fit_with_real_dropout(self):
        config = replace(self.config, dropout=.25)
        cfg = {**self.fit_config(), 'adapt_updates': 2}

        def assert_same(left, right):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    assert_same(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    assert_same(a, b)
            else:
                self.assertEqual(left, right)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_cooperative([self.scene], [self.validation], config, cfg, root/'full',
                              device='cpu', save_predictions=True)
            shared = train_cooperative([self.scene], [self.validation], config,
                {**cfg, 'max_seconds': 60.}, root/'shared', device='cpu', phase='shared')
            self.assertEqual(shared['completed_updates'], cfg['teacher_updates']+cfg['utility_updates'])
            self.assertEqual(shared['completed_routes'], [])
            self.assertFalse(list((root/'shared').glob('adapt-*.pt')))
            # Stopping at teacher must preserve the exact original training
            # endpoint, including dropout/sampling and learned loss weights.
            with patch('association.evaluation.cooperative.score_cooperative', side_effect=AssertionError('no scores')), \
                 patch('association.evaluation.association.select_threshold', side_effect=AssertionError('no threshold')):
                teacher = train_cooperative([self.scene], [self.validation], config,
                    {**cfg, 'utility_updates': 0, 'adapt_updates': 0}, root/'teacher',
                    device='cpu', phase='teacher', save_predictions=True)
            self.assertEqual(teacher['completed_updates'], cfg['teacher_updates'])
            self.assertEqual(set(teacher['phases']),
                {'external_preparation', 'local_preparation', 'teacher', 'completion_evaluation'})
            self.assertNotIn('threshold', teacher)
            original = torch.load(root/'shared'/'teacher.pt', weights_only=True)
            stopped = torch.load(root/'teacher'/'teacher.pt', weights_only=True)
            for key in ('model_state', 'objective_state', 'optimizer_state', 'normalizer',
                        'support_time_scale_tau'):
                assert_same(original[key], stopped[key])
            # A pre-policy shared checkpoint/run must still feed legacy-formula
            # arms without interpreting those weights under a different chart.
            old_bundle = torch.load(root/'shared'/'shared.pt', weights_only=True)
            del old_bundle['model_config']['speed_output_policy']
            del shared['model_config']['speed_output_policy']
            torch.save(old_bundle, root/'shared'/'shared.pt')
            runtime.write_json(root/'shared'/'run.json', shared)
            for route in ROUTES:
                # Fresh process initialization may consume any RNG amount. Each
                # adaptation stage must restore the same original seed itself.
                torch.manual_seed(987); torch.rand(51)
                with patch('association.data.cooperative.fit_cooperative_normalizer', side_effect=AssertionError('no refit')), \
                     patch('association.training.cooperative.fit_support_time_scaling', side_effect=AssertionError('no S refit')), \
                     patch('association.training.cooperative.teacher_error_pair', side_effect=AssertionError('no teacher training')), \
                     patch('association.evaluation.cooperative.score_cooperative', side_effect=AssertionError('no new scores')), \
                     patch('association.evaluation.association.select_threshold', side_effect=AssertionError('no reselection')):
                    arm = train_cooperative([self.scene], [self.validation], config,
                        {**cfg, 'max_seconds': 1e-12, 'time_policy': 'advisory'},
                        root/route, device='cpu', phase='arm',
                        shared_from=root/'shared', route=route, save_predictions=True)
                self.assertEqual(arm['completed_updates'], cfg['adapt_updates'])
                self.assertEqual(arm['completed_routes'], [route])
                self.assertNotIn('teacher', arm['phases'])
                self.assertEqual(arm['identity_selection'], shared['identity_selection'])
                before = torch.load(root/'full'/f'adapt-{route}.pt', weights_only=True)
                after = torch.load(root/route/f'adapt-{route}.pt', weights_only=True)
                for key in ('model_state', 'objective_state', 'optimizer_state', 'normalizer',
                            'support_time_scale_tau'):
                    assert_same(before[key], after[key])
                full_history = [row for row in runtime.read_json(root/'full'/'history.json')
                                if row['stage'] == 'adapt' and row['routing'] == route]
                arm_history = runtime.read_json(root/route/'history.json')
                for row in [*full_history, *arm_history]:
                    row.pop('seconds')
                self.assertEqual(full_history, arm_history)
                full_report = runtime.read_json(root/'full'/f'{route}-evaluation.json')
                arm_report = runtime.read_json(root/route/f'{route}-evaluation.json')
                full_report.pop('elapsed_seconds'); arm_report.pop('elapsed_seconds')
                self.assertEqual(full_report, arm_report)
                for source in [root/'full'/f'{route}-scores.npz',
                               *(root/'full'/f'{route}-predictions').glob('*.npz')]:
                    target = root/route/source.relative_to(root/'full')
                    with np.load(source, allow_pickle=False) as expected, np.load(target, allow_pickle=False) as actual:
                        self.assertEqual(expected.files, actual.files)
                        for key in expected.files:
                            np.testing.assert_array_equal(actual[key], expected[key])

    def test_arm_rejects_unfinished_shared_or_changed_protocol_and_population(self):
        cfg = self.fit_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_cooperative([self.scene], [self.validation], self.config, cfg,
                              root/'shared', device='cpu', phase='shared')
            saved = runtime.read_json(root/'shared'/'run.json')
            runtime.write_json(root/'shared'/'run.json', {**saved, 'status': 'RUNNING'})
            with self.assertRaisesRegex(ValueError, 'COMPLETED shared'):
                train_cooperative([self.scene], [self.validation], self.config, cfg,
                    root/'unfinished', device='cpu', phase='arm', shared_from=root/'shared', route='plain')
            runtime.write_json(root/'shared'/'run.json', saved)
            changed = copy.deepcopy(self.scene)
            changed['component_id'] = 'different-component'
            cases = [('protocol', [self.scene], {**cfg, 'learning_rate': cfg['learning_rate']*2}, 'protocol'),
                     ('population', [changed], cfg, 'scene contract')]
            for name, scenes, training, message in cases:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    train_cooperative(scenes, [self.validation], self.config, training,
                        root/name, device='cpu', phase='arm', shared_from=root/'shared', route='plain')
            labels = copy.deepcopy(self.validation)
            labels['labels'][0] = 1-labels['labels'][0]
            with self.assertRaisesRegex(ValueError, 'labels disagree'):
                train_cooperative([self.scene], [labels], self.config, cfg,
                    root/'labels', device='cpu', phase='arm', shared_from=root/'shared', route='plain')
