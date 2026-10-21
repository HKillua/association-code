"""Approved fixed calibration: synthetic CPU math, state isolation and resume."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.utils import runtime
from association.training import tmtaff as training
from association.training.tmtaff import STAGES, load_complete_snapshot, train_complete
from association.models.fusion import PairFusionConfig
from association.training.calibration import MARGIN_POLICY, calibrate_training_margin
from association.models.tmtaff import AssociationModel, ModelConfig
from association.training.common import fit_normalizer, normalize_scene
from association.training.sampling import BalancedScenePairSampler
from association.training.batches import FrozenCompletionCache
from tests.fixtures.scenes import registered_scene


class MarginCalibrationTests(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = [registered_scene('a', 'T_core'), registered_scene('b', 'T_core')]
        self.validation = [registered_scene('v', 'V_select')]
        self.mc = ModelConfig(length=4, d_model=8, nhead=2, num_layers=1,
                              dim_feedforward=16, dropout=.1, margin=1.)
        self.fc = PairFusionConfig(length=4, d_model=8, nhead=2, num_layers=1,
                                   dim_feedforward=16, dropout=.1, fc_hidden=8)
        self.config = dict(updates=dict.fromkeys(STAGES, 2), learning_rates=dict.fromkeys(STAGES, .001),
            seed=7, branch='contrastive', batch_tracks=3, pair_batch_size=4, pair_microbatch=2,
            track_microbatch=2, completion_weight=.1, gradient_clip=1., max_hours=.1,
            validate_every_updates=1, completion_cache_mib=1, margin_policy=MARGIN_POLICY,
            position_loss_mode='standard_smooth_l1', lr_schedule='constant',
            singleton_policy='observed', validation_batches=dict(tracks=3, pairs=3))
        self.model = AssociationModel(self.mc, 'contrastive')
        self.model.set_completion_trainable(False)
        self.model.eval()
        self.normalized = [normalize_scene(s, fit_normalizer(self.train)) for s in self.train]
        self.cache = FrozenCompletionCache(self.model, self.normalized, 'cpu', 1024 * 1024, 3)

    def calibrate(self, *, check=lambda: None, report=None):
        return calibrate_training_margin(self.model, self.train, self.normalized, self.cache,
            device='cpu', batch_tracks=3, check=check, report={} if report is None else report)

    def fit(self, name, **kwargs):
        return train_complete(self.train, self.validation, self.mc, self.fc, self.config,
                              self.root / name, **kwargs)

    def saved(self, name):
        return torch.load(self.root / name / 'checkpoint.pt', weights_only=True)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_real_forward_median_diagnostics_rng_modes_parameters_and_sampler_isolation(self):
        main_sampler = BalancedScenePairSampler(self.train, 7)
        track_rng, tfm_rng = np.random.default_rng(7), np.random.default_rng(8)
        state = training._rng_state(track_rng, tfm_rng, main_sampler, torch.device('cpu'))
        weights = copy.deepcopy(self.model.state_dict())
        modes = [m.training for m in self.model.modules()]
        flags = [p.requires_grad for p in self.model.parameters()]
        called = []
        hook = self.model.tam_encoder.register_forward_pre_hook(
            lambda module, args: called.append((module.training, torch.is_grad_enabled(),
                                                self.model.tcm_encoder.training)))
        report = {}
        with patch.object(torch.cuda, 'set_rng_state', side_effect=AssertionError('CPU must not touch CUDA')):
            margin = self.calibrate(report=report)
        hook.remove()
        after = training._rng_state(track_rng, tfm_rng, main_sampler, torch.device('cpu'))
        self.assert_tree_equal(state, after)
        self.assert_tree_equal(weights, self.model.state_dict())
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        self.assertEqual(flags, [p.requires_grad for p in self.model.parameters()])
        self.assertTrue(called and all(item == (True, False, False) for item in called))
        distance = np.array([x for row in report['samples'] for x in row['distances']])
        label = np.array([x for row in report['samples'] for x in row['labels']])
        negative = sorted(distance[label == 0])
        self.assertEqual(margin, (negative[1023] + negative[1024]) / 2)
        self.assertGreater(margin, 0)
        self.assertEqual(report['pair_presentations'], 4096)
        self.assertEqual(report['positive_presentations'], 2048)
        self.assertEqual(report['negative_presentations'], 2048)
        self.assertEqual(report['duplicate_pair_count'], 4096 - report['unique_pair_count'])
        self.assertGreater(report['duplicate_pair_count'], 0)
        actual_margin = np.float32(margin)
        self.assertEqual(report['training_dtype_margin'], float(actual_margin))
        self.assertEqual(report['negative_hinge']['strict_active_count'], int((np.array(negative) < actual_margin).sum()))
        self.assertEqual(report['negative_hinge']['ties_count'], int((np.array(negative) == actual_margin).sum()))
        actual_hinge = torch.relu(margin - torch.tensor(distance, dtype=torch.float32)).double().numpy()
        expected_loss = .5 * np.mean(label * distance + (1 - label) * actual_hinge**2)
        self.assertAlmostEqual(report['loss']['total'], expected_loss)
        self.assertEqual(report['distance_derivative']['positive']['batch_mean_mean'], .5 / 64)
        self.assertAlmostEqual(report['distance_derivative']['negative']['per_example_mean'],
                               -actual_hinge[label == 0].mean())
        self.assertGreater(report['cache_after']['misses'], 0)
        self.assertGreater(report['cache_after']['hits'], 0)
        second = {}
        self.assertEqual(self.calibrate(report=second), margin)
        self.assertEqual(second['samples'], report['samples'])
        self.assertEqual(self.model.config.margin, 1.)  # caller applies only after successful return


    def test_budget_failure_keeps_partial_diagnostics_and_restores_state(self):
        report, rng = {}, torch.get_rng_state()
        original = TimeoutError('calibration budget expired')
        def check():
            if report.get('completed_batches', 0) >= 2:
                raise original
        with self.assertRaises(TimeoutError) as caught:
            self.calibrate(check=check, report=report)
        self.assertIs(caught.exception, original)
        self.assertEqual(report['completed_batches'], 2)
        self.assertEqual(report['status'], 'FAILED')
        self.assertNotIn('effective_margin', report)
        self.assertGreater(report['seconds'], 0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(self.model.training)

    def test_inverse_margin_changes_reported_objective_but_not_calibration_samples_or_margin(self):
        paper, scaled = {}, {}
        margin = self.calibrate(report=paper)
        self.model.config = replace(self.model.config, contrastive_negative_scale='inverse_margin')
        self.assertEqual(self.calibrate(report=scaled), margin)
        self.assertEqual(paper['samples'], scaled['samples'])
        divisor = scaled['training_dtype_margin']
        self.assertEqual(scaled['training_dtype_negative_divisor'], divisor)
        self.assertEqual(scaled['loss']['positive_class_mean'], paper['loss']['positive_class_mean'])
        self.assertAlmostEqual(scaled['loss']['negative_class_mean'],
                               paper['loss']['negative_class_mean'] / divisor)
        self.assertAlmostEqual(scaled['distance_derivative']['negative']['per_example_mean'],
                               paper['distance_derivative']['negative']['per_example_mean'] / divisor)
        self.assertEqual(scaled['distance_derivative']['positive'], paper['distance_derivative']['positive'])
        quantiles = scaled['distance_derivative']['negative']['per_example_quantiles']
        self.assertGreaterEqual(quantiles['min'], -1.)
        self.assertLessEqual(quantiles['max'], 0.)


    def test_adjacent_float_median_keeps_rule_and_reports_actual_training_hinge(self):
        from association.training import calibration
        pair_batch = calibration._pair_batch
        current_labels = []
        def capture(*args):
            batch = pair_batch(*args)
            current_labels[:] = [batch[4]]
            return batch
        upper = float(np.nextafter(np.float32(1.), np.float32(2.)))
        def scores(encoded, pairs):
            result = torch.full((64,), -2.)
            result[current_labels[0] == 0] = -torch.tensor([1., upper] * 16)
            return result
        report = {}
        with (patch.object(calibration, '_pair_batch', side_effect=capture),
              patch.object(self.model, 'score_pairs', side_effect=scores)):
            margin = self.calibrate(report=report)
        self.assertEqual(margin, (1. + upper) / 2)
        self.assertEqual(report['training_dtype_margin'], 1.)
        self.assertEqual(report['mathematical_median_hinge']['strict_active_count'], 1024)
        self.assertEqual(report['negative_hinge']['strict_active_count'], 0)
        self.assertEqual(report['negative_hinge']['ties_count'], 1024)
        self.assertEqual(report['loss']['negative_contribution'], 0.)
        self.assertEqual(report['distance_derivative']['negative']['per_example_mean'], 0.)


    def test_calibration_once_boundary_resume_effective_snapshot_and_training_replay(self):
        original = training.calibrate_training_margin
        with patch.object(training, 'calibrate_training_margin', wraps=original) as call:
            result = self.fit('full')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result['status'], 'COMPLETED')
        expected = self.saved('full')
        margin = expected['effective_margin']
        self.assertNotEqual(margin, self.mc.margin)
        self.assertEqual(expected['model_config']['margin'], 1.)
        self.assertEqual(expected['selected']['effective_margin'], margin)
        for selected in (True, False):
            snapshot = load_complete_snapshot(self.root / 'full' / 'checkpoint.pt', selected=selected)
            self.assertEqual(snapshot['model'].config.margin, margin)
        for cut in (2, 3, 6):
            with self.subTest(cut=cut):
                with patch.object(training, 'calibrate_training_margin', wraps=original) as first_call:
                    self.fit(f'first-{cut}', max_updates_this_call=cut)
                with patch.object(training, 'calibrate_training_margin', wraps=original) as resumed_call:
                    self.fit(f'resumed-{cut}', resume_from=self.root / f'first-{cut}')
                self.assertEqual(first_call.call_count, int(cut > 2))
                self.assertEqual(resumed_call.call_count, int(cut == 2))
                actual = self.saved(f'resumed-{cut}')
                self.assertEqual(actual['effective_margin'], margin)
                for key in ('states', 'optimizer', 'cursor', 'coverage', 'selected', 'rng'):
                    self.assert_tree_equal(actual[key], expected[key])
                self.assertEqual([r['loss'] for r in actual['history']], [r['loss'] for r in expected['history']])

    def test_calibration_failure_preserves_tcm_checkpoint_without_partial_margin(self):
        original = training.calibrate_training_margin
        failure = TimeoutError('inside calibration')
        def fail(*args, **kwargs):
            report = kwargs['report']
            def check():
                if report.get('completed_batches', 0) == 1:
                    raise failure
            return original(*args, **{**kwargs, 'check': check})
        with patch.object(training, 'calibrate_training_margin', side_effect=fail):
            with self.assertRaises(TimeoutError) as caught:
                self.fit('failed')
        self.assertIs(caught.exception, failure)
        saved = self.saved('failed')
        self.assertEqual(saved['cursor'], dict(stage=0, update=2, completed_updates=2))
        self.assertIsNone(saved['margin_calibration'])
        self.assertEqual(saved['effective_margin'], 1.)
        run = runtime.read_json(self.root / 'failed' / 'run.json')
        self.assertEqual(run['status'], 'FAILED')
        self.assertEqual(run['margin_calibration']['status'], 'FAILED')
        with self.assertRaisesRegex(ValueError, 'only an INTERRUPTED'):
            self.fit('no-retry', resume_from=self.root / 'failed')


    def test_native_time_four_stages_calibrate_once_and_snapshot_resume_effective_margin(self):
        from association.models.timed import TimedAssociationModel
        from tests.fixtures.scenes import timed_scene
        self.train = [timed_scene('native-train', 'T_core')]
        self.validation = [timed_scene('native-validation', 'V_select')]
        self.config['updates'] = dict.fromkeys(STAGES, 1)
        original = training.calibrate_training_margin
        with patch.object(training, 'calibrate_training_margin', wraps=original) as call:
            result = self.fit('native-full')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result['status'], 'COMPLETED')
        expected = self.saved('native-full')
        self.assertEqual([r['stage'] for r in expected['history']], list(STAGES))
        self.assertTrue(all(np.isfinite(row['loss']) for row in expected['history']))
        loaded = load_complete_snapshot(self.root / 'native-full' / 'checkpoint.pt')
        self.assertIsInstance(loaded['model'], TimedAssociationModel)
        self.assertEqual(loaded['model'].config.margin, expected['effective_margin'])
        self.fit('native-part', max_updates_this_call=2)
        with patch.object(training, 'calibrate_training_margin', side_effect=AssertionError('must not recalibrate')):
            resumed = self.fit('native-resume', resume_from=self.root / 'native-part')
        self.assertEqual(resumed['status'], 'COMPLETED')
        actual = self.saved('native-resume')
        for key in ('states', 'optimizer', 'cursor', 'coverage', 'selected', 'rng'):
            self.assert_tree_equal(actual[key], expected[key])
        self.assertEqual(actual['effective_margin'], expected['effective_margin'])
        self.assertEqual([r['loss'] for r in actual['history']], [r['loss'] for r in expected['history']])


if __name__ == '__main__':
    unittest.main()
