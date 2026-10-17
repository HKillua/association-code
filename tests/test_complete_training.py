import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association import runtime
from association.complete_training import STAGES, _compatible_device, train_complete
from association.fusion import PairFusionConfig
from association.models import ModelConfig


def registered_scene(name, split, *, singleton=False):
    grid = np.arange(4, dtype=np.float64) * 10
    values = np.empty((4, 4, 4), dtype=np.float32)
    for node in range(4):
        values[node] = np.array([30 + .01 * (node // 2), 120, 5 + node / 10, 45])
        values[node, :, 0] += np.arange(4) * .0001
        values[node, :, 1] += np.arange(4) * .0002 + (node % 2) * .00001
    observed = np.array([[1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0]], dtype=bool)
    if singleton:
        observed[:] = False
        observed[:, 1] = True
    targets = {identity: dict(states=values[node].copy(), valid=np.ones((4, 4), dtype=bool), query_times=grid.copy())
               for identity, node in (('a', 0), ('b', 2))}
    values[~observed] = np.nan
    pairs = np.array([(a, b) for a in range(4) for b in range(a + 1, 4)], dtype=np.int64)
    identities = np.array(['a', 'a', 'b', 'b'])
    return dict(scene_id=name, component_id=name + '-component', split=split,
                grid=grid, observation_times=np.broadcast_to(grid, observed.shape).copy(),
                values=values, observed=observed, fusion_targets=targets,
                node_ids=np.array(['n0', 'n1', 'n2', 'n3']), identities=identities,
                sources=np.array(['A', 'B', 'A', 'B']), pairs=pairs,
                labels=(identities[pairs[:, 0]] == identities[pairs[:, 1]]).astype(np.float32),
                intervals=np.array([[grid[np.flatnonzero(mask)[0]], grid[np.flatnonzero(mask)[-1]]]
                                    for mask in observed]))


class CompleteTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = [registered_scene('train-a', 'T_core'), registered_scene('train-b', 'T_head')]
        self.validation = [registered_scene('validation', 'V_select')]
        self.model = ModelConfig(length=4, d_model=8, nhead=2, num_layers=1,
                                 dim_feedforward=16, dropout=.1)
        self.fusion = PairFusionConfig(fc_hidden=8, length=4, d_model=8, nhead=2,
                                       num_layers=1, dim_feedforward=16, dropout=.1)
        self.config = dict(updates=dict.fromkeys(STAGES, 2), learning_rates=dict.fromkeys(STAGES, .001),
            seed=7, branch='ce', batch_tracks=3, pair_batch_size=4, pair_microbatch=2,
            track_microbatch=2, completion_weight=.1, gradient_clip=1., max_hours=.1,
            validate_every_updates=1, completion_cache_mib=1,
            position_loss_mode='standard_smooth_l1', lr_schedule='constant',
            singleton_policy='observed', validation_batches=dict(tracks=3, pairs=3))

    def fit(self, name, **kwargs):
        return train_complete(self.train, self.validation, self.model, self.fusion,
                              self.config, self.root / name, **kwargs)

    def checkpoint(self, name):
        return torch.load(self.root / name / 'checkpoint.pt', weights_only=True)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_four_stages_have_real_updates_full_evaluation_and_coverage(self):
        result = self.fit('full')
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertFalse(result['test_opened'])
        saved = self.checkpoint('full')
        self.assertEqual(saved['cursor']['completed_updates'], 8)
        self.assertEqual([row['stage'] for row in saved['history']], [s for s in STAGES for _ in range(2)])
        for row in saved['history']:
            self.assertTrue(np.isfinite(row['loss']))
            self.assertGreater(row['gradient_norm'], 0)
            if row['stage'] in ('TAM_FROZEN', 'JOINT'):
                self.assertEqual(row['positive_pairs'], 2)
            if row['stage'] == 'TFM_FROZEN':
                self.assertEqual(row['positive_pairs'], 4)
        for stage in STAGES:
            item = saved['coverage'][stage]
            self.assertGreater(item['unique_track_count'], 0)
            self.assertEqual(item['unique_scene_count'], len(item['scene_ids']))
        summary = runtime.read_json(self.root / 'full' / 'summary.json')
        self.assertTrue({'predicted', 'oracle', 'single_source:A', 'single_source:B'}.issubset(
            summary['evaluation']['fusion']))
        self.assertEqual(summary['selection']['threshold'], summary['evaluation']['threshold'])
        self.assertEqual(saved['normalizer']['training_scene_ids'], ['train-a', 'train-b'])
        self.assertIn('not full-task optimality', summary['selection_scope'])

    def test_exact_resume_including_dropout_optimizer_rng_and_selected_state(self):
        self.fit('full')
        expected = self.checkpoint('full')
        # Interior TCM, exact stage transition, and interior TFM interruption.
        for cut in (1, 2, 3, 4, 5, 6, 7):
            with self.subTest(cut=cut):
                first = self.fit(f'first-{cut}', max_updates_this_call=cut)
                self.assertEqual(first['status'], 'INTERRUPTED')
                self.assertFalse((self.root / f'first-{cut}' / 'summary.json').exists())
                result = self.fit(f'resume-{cut}', resume_from=self.root / f'first-{cut}')
                self.assertEqual(result['status'], 'COMPLETED')
                actual = self.checkpoint(f'resume-{cut}')
                for key in ('states', 'optimizer', 'cursor', 'coverage', 'selected', 'rng'):
                    self.assert_nested_equal(actual[key], expected[key])
                self.assertEqual([r['loss'] for r in actual['history']], [r['loss'] for r in expected['history']])
                self.assertGreater(result['elapsed_seconds'], first['elapsed_seconds'])

    def test_stage_freezing_and_joint_fallback_keeps_trained_fusion(self):
        self.config['updates'] = dict.fromkeys(STAGES, 1)
        self.fit('tcm', max_updates_this_call=1)
        self.fit('tam', resume_from=self.root / 'tcm', max_updates_this_call=1)
        self.fit('tfm', resume_from=self.root / 'tam', max_updates_this_call=1)
        tcm, tam, tfm = [self.checkpoint(s) for s in ('tcm', 'tam', 'tfm')]
        for key in tcm['states'][0]:
            if key.startswith(('input_embedding.', 'tcm_encoder.', 'tcm_decoder.')):
                self.assert_nested_equal(tcm['states'][0][key], tam['states'][0][key])
        self.assertTrue(any(not torch.equal(tcm['states'][0][k], tam['states'][0][k])
                            for k in tcm['states'][0] if k.startswith('tam_encoder.')))
        self.assert_nested_equal(tam['selected']['states'][0], tfm['states'][0])
        self.assertTrue(any(not torch.equal(tam['states'][1][k], tfm['states'][1][k]) for k in tam['states'][1]))
        self.assert_nested_equal(tfm['selected']['states'], tfm['states'])
        # Force only the joint selection scalar lower; final evaluation remains
        # real and must use the stored TFM-complete selection, never TAM-only.
        from association.complete_training import select_threshold
        def lower(records):
            selected = select_threshold(records)
            return {**selected, 'macro_f1': -1.}
        with patch('association.complete_training.select_threshold', side_effect=lower):
            self.fit('joint', resume_from=self.root / 'tfm')
        joint = self.checkpoint('joint')
        self.assert_nested_equal(joint['selected']['states'], tfm['selected']['states'])
        self.assertTrue(any(not torch.equal(joint['states'][0][k], tfm['states'][0][k]) for k in tfm['states'][0]))

    def test_all_singleton_skips_tcm_without_fake_steps(self):
        self.train = [registered_scene('singleton', 'T_core', singleton=True)]
        self.config['updates'] = dict.fromkeys(STAGES, 1)
        self.fit('singletons')
        saved = self.checkpoint('singletons')
        self.assertEqual(saved['history'][0]['status'], 'SKIPPED')
        self.assertEqual(saved['history'][0]['completed_updates'], 0)
        self.assertEqual(saved['cursor']['completed_updates'], 3)
        self.assertEqual(saved['coverage']['TCM']['track_presentations'], 0)

    def test_registration_splits_and_zero_budget_rejected_before_output(self):
        cases = ('time', 'empty', 'component', 'scene', 'test', 'zero_budget', 'gap',
                 'missing_pair', 'noncanonical_pairs', 'extra_bad_target', 'target_time')
        for name in cases:
            with self.subTest(name=name):
                train, validation, config = copy.deepcopy((self.train, self.validation, self.config))
                if name == 'time':
                    train[0]['observation_times'][0, 0] += 150
                elif name == 'empty':
                    train[0]['observed'][0] = False
                elif name == 'component':
                    validation[0]['component_id'] = train[0]['component_id']
                elif name == 'scene':
                    validation[0]['scene_id'] = train[0]['scene_id']
                elif name == 'test':
                    validation[0]['split'] = 'T_test'
                elif name == 'zero_budget':
                    config['max_hours'] = 0
                elif name == 'gap':
                    train[0]['observed'][0, 1] = False
                elif name == 'missing_pair':
                    train[0]['pairs'] = train[0]['pairs'][:-1]
                    train[0]['labels'] = train[0]['labels'][:-1]
                elif name == 'noncanonical_pairs':
                    train[0]['pairs'] = train[0]['pairs'][::-1]
                    train[0]['labels'] = train[0]['labels'][::-1]
                elif name == 'extra_bad_target':
                    train[0]['fusion_targets']['not-observed'] = copy.deepcopy(train[0]['fusion_targets']['a'])
                    train[0]['fusion_targets']['not-observed']['states'][0, 0] = 91
                elif name == 'target_time':
                    train[0]['fusion_targets']['a']['query_times'][0] += 150
                with self.assertRaises(ValueError):
                    train_complete(train, validation, self.model, self.fusion, config, self.root / name)
                self.assertFalse((self.root / name).exists())

    def test_zero_node_scenes_and_unobserved_targets_stay_in_full_evaluation(self):
        missing = registered_scene('no-nodes', 'V_select')
        for key in ('values', 'observed', 'observation_times', 'node_ids', 'identities', 'sources', 'intervals', 'pairs', 'labels'):
            missing[key] = missing[key][:0]
        empty = copy.deepcopy(missing)
        empty.update(scene_id='empty', component_id='empty-component', fusion_targets={})
        self.validation.extend([missing, empty])
        unused_train = copy.deepcopy(empty)
        unused_train.update(scene_id='empty-train', component_id='empty-train-component', split='T_core')
        self.train.append(unused_train)
        self.config['updates'] = dict.fromkeys(STAGES, 1)
        self.fit('empty-scenes')
        evaluation = runtime.read_json(self.root / 'empty-scenes' / 'summary.json')['evaluation']
        self.assertEqual([s['scene_id'] for s in evaluation['per_scene']], ['validation', 'no-nodes', 'empty'])
        self.assertEqual(evaluation['fusion']['oracle']['all']['position']['available_target_count'], 16)
        self.assertEqual(evaluation['fusion']['oracle']['all']['position']['missing_prediction_count'], 8)

    def test_all_reject_threshold_none_is_a_valid_final_selection(self):
        scene = self.validation[0]
        scene['identities'] = np.array(['w', 'x', 'y', 'z'])
        scene['labels'][:] = 0
        base = scene['fusion_targets']
        scene['fusion_targets'] = {key: copy.deepcopy(base['a' if i < 2 else 'b'])
                                   for i, key in enumerate(scene['identities'])}
        self.config['updates'] = dict.fromkeys(STAGES, 1)
        result = self.fit('reject')
        self.assertEqual(result['status'], 'COMPLETED')
        summary = runtime.read_json(self.root / 'reject' / 'summary.json')
        self.assertIsNone(summary['selection']['threshold'])
        self.assertIsNone(summary['evaluation']['threshold'])

    def test_checkpoint_frequency_and_explicit_interrupt(self):
        self.config['updates'] = dict.fromkeys(STAGES, 3)
        self.config['validate_every_updates'] = 2
        original_save = torch.save
        cursors = []
        def save(payload, *args, **kwargs):
            cursors.append((payload['cursor']['stage'], payload['cursor']['update']))
            return original_save(payload, *args, **kwargs)
        with patch('association.complete_training.torch.save', side_effect=save):
            self.fit('periodic')
        self.assertEqual(cursors, [(stage, update) for stage in range(4) for update in (2, 3)])
        cursors.clear()
        with patch('association.complete_training.torch.save', side_effect=save):
            result = self.fit('explicit-interrupt', max_updates_this_call=1)
        self.assertEqual(result['status'], 'INTERRUPTED')
        self.assertEqual(cursors, [(0, 1)])

    def test_failure_never_saves_partial_step_as_resumable(self):
        from association.complete_training import _update
        calls = 0
        def fail_after_step(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = _update(*args, **kwargs)
            if calls == 2:
                raise RuntimeError('failure after parameter mutation')
            return result
        with patch('association.complete_training._update', side_effect=fail_after_step):
            with self.assertRaisesRegex(RuntimeError, 'parameter mutation'):
                self.fit('failed')
        self.assertEqual(runtime.read_json(self.root / 'failed' / 'run.json')['status'], 'FAILED')
        self.assertEqual(self.checkpoint('failed')['cursor']['completed_updates'], 1)
        with self.assertRaisesRegex(ValueError, 'only an INTERRUPTED'):
            self.fit('retry', resume_from=self.root / 'failed')
        self.assertFalse((self.root / 'retry').exists())

    def test_final_evaluation_failure_is_not_completion(self):
        self.config['updates'] = dict.fromkeys(STAGES, 1)
        def fail(*args, **kwargs):
            kwargs['check']()
            raise TimeoutError('full evaluation budget exhausted')
        with self.assertRaisesRegex(TimeoutError, 'evaluation budget'):
            self.fit('failed-eval', evaluation_fn=fail)
        self.assertEqual(self.checkpoint('failed-eval')['cursor']['completed_updates'], 4)
        self.assertEqual(runtime.read_json(self.root / 'failed-eval' / 'run.json')['status'], 'FAILED')
        self.assertFalse((self.root / 'failed-eval' / 'summary.json').exists())

    def test_resume_preserves_budget_semantics_and_output_immutability(self):
        self.fit('first', max_updates_this_call=1)
        with self.assertRaises(FileExistsError):
            self.fit('first')
        self.config['learning_rates']['JOINT'] *= 2
        with self.assertRaisesRegex(ValueError, 'resume configuration'):
            self.fit('changed', resume_from=self.root / 'first')
        self.config['learning_rates']['JOINT'] /= 2
        report = runtime.read_json(self.root / 'first' / 'run.json')
        report['elapsed_seconds'] = self.config['max_hours'] * 3600
        runtime.write_json(self.root / 'first' / 'run.json', report)
        with self.assertRaisesRegex(TimeoutError, 'exhausted'):
            self.fit('exhausted', resume_from=self.root / 'first')
        self.assertFalse((self.root / 'exhausted').exists())

    def test_gpu_device_indices_can_change_without_implicit_cpu_fallback(self):
        self.assertTrue(_compatible_device('cuda:0', 'cuda:3'))
        self.assertTrue(_compatible_device('cuda', 'cuda:1'))
        self.assertTrue(_compatible_device('cpu', 'cpu'))
        self.assertFalse(_compatible_device('cuda:0', 'cpu'))
        self.assertFalse(_compatible_device('cpu', 'cuda:0'))

    def test_contrastive_branch_executes_all_four_stages(self):
        self.config['branch'] = 'contrastive'
        self.config['updates'] = dict.fromkeys(STAGES, 1)
        self.assertEqual(self.fit('contrastive')['status'], 'COMPLETED')
        self.assertEqual(self.checkpoint('contrastive')['cursor']['completed_updates'], 4)

    def test_positive_but_exhausted_budget_leaves_failed_evidence(self):
        self.config['max_hours'] = 1e-12
        with self.assertRaisesRegex(TimeoutError, 'budget exceeded'):
            self.fit('tiny-budget')
        self.assertEqual(runtime.read_json(self.root / 'tiny-budget' / 'run.json')['status'], 'FAILED')
        self.assertFalse((self.root / 'tiny-budget' / 'checkpoint.pt').exists())


if __name__ == '__main__':
    unittest.main()
