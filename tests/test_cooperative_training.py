"""Independent synthetic training checks; no MTAD file or accelerator access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from association.cooperative import CooperativeConfig, CooperativeModel
from association.cooperative_training import (
    _validate_training_scenes, endpoint_view, load_cooperative_checkpoint, physical_errors, stage_parameters,
    teacher_error_pair, train_batch, train_cooperative, utility_objective,
)
from association.fusion import FusionLoss
from association.reference import fit_normalizer, normalize_scene
from tests.test_timed_pipeline import timed_scene


class CooperativeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(20)
        self.scene = timed_scene('training', 'T_core')
        self.validation = timed_scene('validation', 'V_select')
        self.normalizer = fit_normalizer([self.scene])
        self.config = CooperativeConfig(length=4, max_observations=3, d_model=8,
            nhead=2, num_layers=1, dim_feedforward=16, dropout=0., head_hidden=6)
        self.model = CooperativeModel(self.config, self.normalizer)
        self.objective = FusionLoss(position_loss_mode='standard_smooth_l1')

    def items(self):
        values, observed = normalize_scene(self.scene, self.normalizer)
        result = []
        for nodes, label in (([0, 1], 1.), ([0, 2], 0.)):
            targets = [self.scene['fusion_targets'][str(self.scene['identities'][node])] for node in nodes]
            target = torch.as_tensor(np.stack([t['states'] for t in targets]), dtype=torch.float32)
            valid = torch.ones_like(target, dtype=torch.bool)
            valid[0, 1, 0] = False
            if not label:
                valid[..., 2] = False  # One chunk has an empty feature, effective batch does not.
            target[~valid] = float('nan')
            result.append(dict(values=torch.from_numpy(values[nodes]),
                observed=torch.from_numpy(observed[nodes]), target=target, valid=valid,
                label=label, removed=[0, 0], scene_id='training', pair_row=len(result)))
        return result

    def optimizer(self, model, objective, stage, route='plain', lr=.001):
        parameters = stage_parameters(model, stage, routing=route)
        if stage in ('teacher', 'adapt'):
            parameters += list(objective.parameters())
        return torch.optim.SGD(parameters, lr=lr)

    def fit_config(self, routes=None):
        return dict(seed=20, batch_size=2, teacher_updates=1, utility_updates=1,
                    adapt_updates=1, learning_rate=.001, gradient_clip=1.,
                    endpoint_fraction=.25, utility_epsilon=.001,
                    error_scale_policy='training_native_std',
                    routes=routes or ['plain', 'utility', 'learned'],
                    max_seconds=30., batch_tracks=2, pair_batch_size=3)

    def test_physical_error_masks_nan_before_arithmetic_and_wraps_angles(self):
        pred = torch.tensor([[[1., 179., 3., 359.], [float('nan')]*4]], requires_grad=True)
        target = torch.tensor([[[2., -179., 1., 1.], [float('nan')]*4]])
        valid = torch.tensor([[[True]*4, [False]*4]])
        error = physical_errors(pred, target, valid, [1., 1., 2., 1.])
        torch.testing.assert_close(error, torch.tensor([[[1., 2., 1., 2.], [0.]*4]]))
        error.sum().backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(float(pred.grad[:, 1].abs().sum()), 0.)

    def test_full_data_validation_counts_against_cumulative_budget(self):
        from association.complete_training import _scenes
        clock = [0.]
        def expensive_validation(*args):
            result = _scenes(*args)
            clock[0] += 31.
            return result
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'over-budget'
            with patch('association.cooperative_training.time.monotonic', side_effect=lambda: clock[0]), \
                 patch('association.complete_training._scenes', side_effect=expensive_validation), \
                 patch('association.cooperative.CooperativeModel') as constructor:
                with self.assertRaisesRegex(TimeoutError, 'budget exhausted'):
                    train_cooperative([self.scene], [self.validation], self.config,
                                      self.fit_config(), output, device='cpu')
                constructor.assert_not_called()
            report = json.loads((output / 'run.json').read_text())
            self.assertEqual(report['status'], 'INTERRUPTED')
            self.assertEqual(report['elapsed_seconds'], 31.)

    def test_utility_gradient_direction_detaches_teacher_and_ignores_invalid_labels(self):
        u = torch.zeros(2, 2, 4, requires_grad=True)
        e0 = torch.full_like(u, 2., requires_grad=True)
        e1 = torch.full_like(u, 1., requires_grad=True)
        with torch.no_grad():
            e1[1] = 3.  # First direction benefits, reverse direction is harmed.
            e0[:, 1, 0] = float('nan')
            e1[:, 1, 0] = float('nan')
        valid = torch.ones_like(u, dtype=torch.bool)
        valid[:, 1, 0] = False
        loss = utility_objective(u, e0, e1, valid, epsilon=.01)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertLess(float(u.grad[0, 0, 0]), 0.)
        self.assertGreater(float(u.grad[1, 0, 0]), 0.)
        self.assertEqual(float(u.grad[:, 1, 0].abs().sum()), 0.)
        self.assertIsNone(e0.grad)
        self.assertIsNone(e1.grad)

    def test_utility_streaming_has_exact_effective_batch_denominator(self):
        u = torch.randn(4, 4, 4, requires_grad=True)
        v = u.detach().clone().requires_grad_()
        e0, e1 = torch.rand_like(u) + .1, torch.rand_like(u)
        valid = torch.ones_like(u, dtype=torch.bool)
        valid[:2, :, 2] = False
        valid[2, :, 0] = False
        e0[~valid] = float('nan'); e1[~valid] = float('nan')
        counts = valid.sum((0, 1))
        whole = utility_objective(u, e0, e1, valid, epsilon=.01)
        parts = sum(utility_objective(v[s], e0[s], e1[s], valid[s], epsilon=.01,
                                     normalization_counts=counts) for s in (slice(0, 2), slice(2, 4)))
        whole.backward(); parts.backward()
        torch.testing.assert_close(whole, parts)
        torch.testing.assert_close(u.grad, v.grad)
        with self.assertRaisesRegex(ValueError, 'supervision'):
            utility_objective(u[:2], e0[:2], e1[:2], valid[:2], epsilon=.01)

    def test_teacher_eval_repeated_labels_are_identical_and_anchor_specific(self):
        config = CooperativeConfig(**{**self.config.__dict__, 'dropout': .3})
        teacher = CooperativeModel(config, self.normalizer).train()
        item = self.items()[0]
        args = (teacher, item['values'], item['observed'], item['target'], item['valid'])
        first = teacher_error_pair(*args, scales=self.normalizer['std'])
        second = teacher_error_pair(*args, scales=self.normalizer['std'])
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        changed = item['target'].clone(); changed[0, :, 2] += 5.
        third = teacher_error_pair(teacher, item['values'], item['observed'], changed,
                                   item['valid'], scales=self.normalizer['std'])
        torch.testing.assert_close(first[0][1], third[0][1], rtol=0, atol=0)
        self.assertFalse(torch.equal(first[0][0, :, 2], third[0][0, :, 2]))
        self.assertFalse(any(module.training for module in teacher.modules()))
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_teacher_pair_streaming_matches_whole_batch_value_and_update(self):
        actual, expected = copy.deepcopy(self.model), copy.deepcopy(self.model)
        actual_loss, expected_loss = copy.deepcopy(self.objective), copy.deepcopy(self.objective)
        items = self.items()
        optimizer = self.optimizer(actual, actual_loss, 'teacher')
        report = train_batch(actual, actual_loss, optimizer, items, stage='teacher', gradient_clip=1e8)
        ref_optimizer = self.optimizer(expected, expected_loss, 'teacher')
        predictions, targets, masks, identity = [], [], [], []
        for item in items:
            encoded = expected.encode(item['values'], item['observed'])
            groups = [[0], [1]] + ([[0, 1], [1, 0]] if item['label'] else [])
            for index, members in enumerate(groups):
                predictions.append(expected.decode_group(encoded, members))
                targets.append(item['target'][index % 2]); masks.append(item['valid'][index % 2])
            score = expected.score_pairs(encoded, [[0, 1]])
            identity.append(F.binary_cross_entropy_with_logits(score, score.new_tensor([item['label']])))
        physical = expected_loss(torch.stack(predictions), torch.stack(targets), torch.stack(masks))['total']
        identity_bce = torch.stack(identity).mean()
        objective = physical + identity_bce
        ref_optimizer.zero_grad(set_to_none=True); objective.backward(); ref_optimizer.step()
        self.assertAlmostEqual(report['loss'], float(objective.detach()), places=5)
        self.assertAlmostEqual(report['loss_components']['physical'], float(physical.detach()), places=5)
        self.assertAlmostEqual(report['loss_components']['identity_bce'], float(identity_bce.detach()), places=5)
        self.assertEqual(report['loss_components']['utility'], 0.)
        self.assertAlmostEqual(sum(report['loss_components'].values()), report['loss'], places=5)
        self.assertEqual(report['valid_counts'], torch.stack(masks).sum((0, 1)).tolist())
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(actual_loss.log_variance, expected_loss.log_variance)

    def test_utility_updates_only_utility_parts_and_preserves_s0(self):
        teacher = copy.deepcopy(self.model)
        teacher_before = copy.deepcopy(teacher.state_dict())
        model_before = copy.deepcopy(self.model.state_dict())
        item = self.items()[0]
        self.model.eval()
        with torch.no_grad():
            score_before = self.model.score_pairs(self.model.encode(item['values'], item['observed']), [[0, 1]])
        optimizer = self.optimizer(self.model, self.objective, 'utility')
        report = train_batch(self.model, self.objective, optimizer, self.items(), stage='utility',
                             teacher=teacher, error_scales=self.normalizer['std'])
        self.assertEqual(report['loss_components'], dict(physical=0., identity_bce=0., utility=report['loss']))
        changed = []
        for key, value in self.model.state_dict().items():
            if not torch.equal(value, model_before[key]):
                changed.append(key)
        self.assertTrue(changed)
        self.assertTrue(all(key.startswith(('query_pool.', 'utility_head.')) for key in changed))
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
        self.model.eval()
        with torch.no_grad():
            score_after = self.model.score_pairs(self.model.encode(item['values'], item['observed']), [[0, 1]])
        torch.testing.assert_close(score_before, score_after, rtol=0, atol=0)
        self.assertFalse(self.model.encoder.training)

    def test_endpoint_view_clears_hidden_and_query_values_preserves_two_point_track(self):
        class Prefix:
            def integers(self, *args): return 1
        x, observed = normalize_scene(self.scene, self.normalizer)
        x, observed = torch.from_numpy(x), torch.from_numpy(observed)
        a, mask, removed = endpoint_view(x, observed, 3, Prefix(), fraction=.4)
        altered = x.clone(); altered[1:, 0, :4] = 1e20; altered[:, 3:, :4] = float('nan')
        b, other_mask, other_removed = endpoint_view(altered, observed, 3, Prefix(), fraction=.4)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(mask, other_mask))
        self.assertEqual(removed, other_removed)
        self.assertEqual(removed[0], 0)
        self.assertTrue((mask[:, :3].sum(1) == 2).all())
        self.assertEqual(float(a[:, 3:, :4].abs().sum()), 0.)
        self.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(self.model(a, mask), self.model(b, other_mask), rtol=0, atol=0)

    def test_tiny_fit_matches_arm_initialization_sampling_s0_and_final_snapshots(self):
        initial, sequences = {}, {}
        original = train_batch
        def capture(model, objective, optimizer, items, **kwargs):
            if kwargs['stage'] == 'adapt':
                route = kwargs['routing']
                initial[route] = copy.deepcopy(model.state_dict())
                sequences[route] = [(x['scene_id'], x['pair_row'], x['removed'], x['label']) for x in items]
            return original(model, objective, optimizer, items, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'fit'
            with patch('association.cooperative_training.train_batch', side_effect=capture):
                run = train_cooperative([self.scene], [self.validation], self.config, self.fit_config(), out, device='cpu')
            self.assertEqual(run['status'], 'COMPLETED')
            self.assertEqual(run['completed_routes'], ['plain', 'utility', 'learned'])
            history = json.loads((out / 'history.json').read_text())
            for item in history:
                self.assertAlmostEqual(sum(item['loss_components'].values()), item['loss'], places=5)
                if item['stage'] == 'adapt':
                    self.assertEqual(item['loss_components'], dict(physical=item['loss'], identity_bce=0., utility=0.))
            for route in ('utility', 'learned'):
                self.assertEqual(sequences['plain'], sequences[route])
                for key, value in initial['plain'].items():
                    torch.testing.assert_close(value, initial[route][key], rtol=0, atol=0)
            base_scores = np.load(out / 'plain-scores.npz')
            for route in ('plain', 'utility', 'learned'):
                restored, payload = load_cooperative_checkpoint(out / f'adapt-{route}.pt', device='cpu')
                self.assertEqual(payload['stage'], 'adapt')
                self.assertEqual(payload['completed_updates'], 1)
                self.assertFalse(payload['resume_supported'])
                for key, value in restored.state_dict().items():
                    if key.startswith(('input_projection.', 'encoder.', 'identity_base.', 'query_pool.', 'utility_head.')):
                        torch.testing.assert_close(value, initial[route][key], rtol=0, atol=0)
                scores = np.load(out / f'{route}-scores.npz')
                for key in base_scores:
                    np.testing.assert_array_equal(scores[key], base_scores[key])

    def test_failure_and_timeout_never_report_completed_or_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            failed = Path(folder) / 'failed'
            with patch('association.cooperative_training.train_batch', side_effect=ValueError('injected invalid supervision')):
                with self.assertRaisesRegex(ValueError, 'injected'):
                    train_cooperative([self.scene], [self.validation], self.config, self.fit_config(), failed, device='cpu')
            payload = json.loads((failed / 'run.json').read_text())
            self.assertEqual(payload['status'], 'FAILED')
            self.assertEqual(payload['completed_routes'], [])
            before = (failed / 'run.json').read_bytes()
            with self.assertRaises(FileExistsError):
                train_cooperative([self.scene], [self.validation], self.config, self.fit_config(), failed, device='cpu')
            self.assertEqual((failed / 'run.json').read_bytes(), before)
            interrupted = Path(folder) / 'timeout'
            cfg = self.fit_config(); cfg['max_seconds'] = 1e-10
            with self.assertRaises(TimeoutError):
                train_cooperative([self.scene], [self.validation], self.config, cfg, interrupted, device='cpu')
            self.assertEqual(json.loads((interrupted / 'run.json').read_text())['status'], 'INTERRUPTED')

    def test_training_contract_rejects_changed_candidates_labels_and_invalid_targets(self):
        changed = []
        labels = copy.deepcopy(self.scene)
        labels['labels'][1] = 1.  # Original [0,2] joins different identities.
        changed.append(labels)
        partial = copy.deepcopy(self.scene)
        partial['pairs'], partial['labels'] = partial['pairs'][:-1], partial['labels'][:-1]
        changed.append(partial)
        wrong_shape = copy.deepcopy(self.scene)
        key = str(wrong_shape['identities'][0])
        wrong_shape['fusion_targets'][key]['states'] = wrong_shape['fusion_targets'][key]['states'][:-1]
        changed.append(wrong_shape)
        invalid_mask = copy.deepcopy(self.scene)
        invalid_mask['fusion_targets'][key]['valid'] = invalid_mask['fusion_targets'][key]['valid'].astype(np.int32)
        changed.append(invalid_mask)
        nonfinite = copy.deepcopy(self.scene)
        nonfinite['fusion_targets'][key]['states'][0, 0] = np.nan
        changed.append(nonfinite)
        for index, scene in enumerate(changed):
            with self.subTest(index=index), self.assertRaises(ValueError):
                _validate_training_scenes([scene])


if __name__ == '__main__':
    unittest.main()
