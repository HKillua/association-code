import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.utils import runtime
from association.training.tmtaff import STAGES, _ordered_tcm_indices, _scenes, train_complete
from association.models.fusion import PairFusionConfig
from association.models.tmtaff import ModelConfig


from tests.fixtures.scenes import registered_scene

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

    def test_scene_protocol_relaxes_only_component_overlap(self):
        self.validation[0]['component_id'] = self.train[0]['component_id']
        with self.assertRaisesRegex(ValueError, 'components overlap'):
            _scenes(self.train, self.validation, self.model.length)
        for scene in [*self.train, *self.validation]:
            scene['split_unit'] = 'scene'
        contract = _scenes(self.train, self.validation, self.model.length)
        self.assertTrue(all(row['split_unit'] == 'scene' for rows in contract for row in rows))
        self.assertEqual(contract[0][0]['component_id'], contract[1][0]['component_id'])
        self.validation[0]['scene_id'] = self.train[0]['scene_id']
        with self.assertRaisesRegex(ValueError, 'duplicate/overlapping scene'):
            _scenes(self.train, self.validation, self.model.length)

    def test_ordered_tcm_batches_preserve_cross_scene_order_and_small_population(self):
        pool = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
        first, report = _ordered_tcm_indices(len(pool), 3, 1)
        tail, tail_report = _ordered_tcm_indices(len(pool), 3, 2)
        self.assertEqual([pool[int(i)] for i in np.concatenate((first, tail))], pool)
        self.assertEqual(first.tolist(), [0, 1, 2])
        self.assertEqual(tail.tolist(), [3, 4])
        self.assertEqual(report['completed_epochs'], 0)
        self.assertEqual(tail_report['completed_epochs'], 1)
        next_epoch, next_report = _ordered_tcm_indices(len(pool), 3, 3)
        self.assertEqual(next_epoch.tolist(), first.tolist())
        self.assertEqual(next_report['epoch'], 2)
        small, small_report = _ordered_tcm_indices(2, 64, 1)
        self.assertEqual(small.tolist(), [0, 1])
        self.assertEqual(small_report['completed_epochs'], 1)

    def test_mixed_split_protocol_rejected_and_legacy_contract_unchanged(self):
        contract = _scenes(self.train, self.validation, self.model.length)
        self.assertTrue(all('split_unit' not in row for rows in contract for row in rows))
        self.train[0]['split_unit'] = 'scene'
        with self.assertRaisesRegex(ValueError, 'one supported split_unit'):
            _scenes(self.train, self.validation, self.model.length)

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

    def test_cost_timings_separate_updates_validation_snapshots_and_final_evaluation(self):
        from association.training import tmtaff as training
        clock = [0.]

        def costing(function, seconds):
            def call(*args, **kwargs):
                result = function(*args, **kwargs)
                clock[0] += seconds
                return result
            return call

        self.config['max_hours'] = 1
        # CPU work is real; only time and forbidden CUDA telemetry are mocked.
        with (patch.object(training.time, 'monotonic', side_effect=lambda: clock[0]),
              patch.object(training, '_update', side_effect=costing(training._update, 2)),
              patch.object(training, 'fusion_training_step', side_effect=costing(training.fusion_training_step, 3)),
              patch.object(training, 'score_reference', side_effect=costing(training.score_reference, 5)),
              patch.object(training, 'select_threshold', side_effect=costing(training.select_threshold, 7)),
              patch.object(training, '_states', side_effect=costing(training._states, 11)),
              patch.object(training, 'evaluate_scenes', side_effect=costing(training.evaluate_scenes, 13)),
              patch.object(training, '_restore', side_effect=costing(training._restore, 17)),
              patch.object(training.torch, 'save', side_effect=costing(training.torch.save, 29)),
              patch.object(runtime, 'write_json', side_effect=costing(runtime.write_json, 23)),
              patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('CPU must not synchronize CUDA')),
              patch.object(torch.cuda, 'reset_peak_memory_stats', side_effect=AssertionError('CPU must not reset CUDA')),
              patch.object(torch.cuda, 'max_memory_allocated', side_effect=AssertionError('CPU must not read CUDA')),
              patch.object(torch.cuda, 'max_memory_reserved', side_effect=AssertionError('CPU must not read CUDA'))):
            result = self.fit('timed', evaluation_fn=costing(training.evaluate_complete, 19))
        history = self.checkpoint('timed')['history']
        for row in history:
            self.assertEqual(row['seconds'], 2 if row['stage'] in ('TCM', 'TAM_FROZEN') else 3)
        timed_rows = [row for row in history if 'validation_seconds' in row]
        self.assertEqual(len(timed_rows), 4)
        self.assertTrue(any(item['selected_snapshot_copied'] for item in result['validation_timings']))
        for row, timing in zip(timed_rows, result['validation_timings']):
            expected = 12 + (24 if timing['selected_snapshot_copied'] else 0)
            self.assertEqual(timing['status'], 'COMPLETED')
            self.assertEqual(timing['seconds'], expected)
            self.assertEqual(row['validation_seconds'], expected)
        self.assertEqual(result['final_evaluation_seconds'], 36)
        self.assertEqual(result['final_evaluation_status'], 'COMPLETED')
        summary = runtime.read_json(self.root / 'timed' / 'summary.json')
        self.assertEqual(summary['evaluation_seconds'], 36)
        self.assertIsNone(result['cuda_process_peak_allocated_bytes'])
        self.assertIsNone(result['cuda_process_peak_reserved_bytes'])
        self.assertIsNone(result['cuda_peak_reset_completed'])


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
        from association.training.tmtaff import select_threshold
        def lower(records):
            selected = select_threshold(records)
            return {**selected, 'macro_f1': -1.}
        with patch('association.training.tmtaff.select_threshold', side_effect=lower):
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


    def test_failure_never_saves_partial_step_as_resumable(self):
        from association.training.tmtaff import _update
        calls = 0
        def fail_after_step(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = _update(*args, **kwargs)
            if calls == 2:
                raise RuntimeError('failure after parameter mutation')
            return result
        with patch('association.training.tmtaff._update', side_effect=fail_after_step):
            with self.assertRaisesRegex(RuntimeError, 'parameter mutation'):
                self.fit('failed')
        self.assertEqual(runtime.read_json(self.root / 'failed' / 'run.json')['status'], 'FAILED')
        self.assertEqual(self.checkpoint('failed')['cursor']['completed_updates'], 1)
        with self.assertRaisesRegex(ValueError, 'only an INTERRUPTED'):
            self.fit('retry', resume_from=self.root / 'failed')
        self.assertFalse((self.root / 'retry').exists())

    def test_failure_flushes_completed_updates_after_last_checkpoint(self):
        from association.training import tmtaff as training
        self.config['updates'] = dict(TCM=1, TAM_FROZEN=1, TFM_FROZEN=1, JOINT=4)
        self.config['validate_every_updates'] = 4
        original = training.fusion_training_step
        joint_calls = 0

        def interrupted_step(*args, **kwargs):
            nonlocal joint_calls
            if kwargs['stage'] == 'JOINT':
                joint_calls += 1
                if joint_calls == 3:
                    raise InterruptedError('SIGTERM during third JOINT update')
            return original(*args, **kwargs)

        with patch.object(training, 'fusion_training_step', side_effect=interrupted_step):
            with self.assertRaisesRegex(InterruptedError, 'third JOINT'):
                self.fit('failed-joint')
        output = self.root / 'failed-joint'
        run = runtime.read_json(output / 'run.json')
        history = runtime.read_json(output / 'history.json')
        coverage = runtime.read_json(output / 'coverage.json')
        saved = self.checkpoint('failed-joint')
        self.assertEqual(run['status'], 'FAILED')
        self.assertEqual(run['cursor'], dict(stage=3, update=2, completed_updates=5))
        self.assertEqual([r['update'] for r in history if r['stage'] == 'JOINT'], [1, 2])
        self.assertEqual(len(history), 5)
        self.assertEqual(coverage['JOINT']['pair_presentations'], 8)
        self.assertEqual(coverage['JOINT']['positive_pair_presentations'], 4)
        self.assertEqual(run['checkpoint_cursor'], saved['cursor'])
        self.assertEqual(saved['cursor']['completed_updates'], 3)
        self.assertEqual(len(saved['history']), 3)
        self.assertEqual(saved['coverage']['JOINT']['pair_presentations'], 0)
        self.assertEqual(run['final_evaluation_status'], 'NOT_STARTED')
        self.assertFalse((output / 'summary.json').exists())

    def test_validation_interruption_retains_its_completed_training_update(self):
        with patch('association.training.tmtaff.score_reference',
                   side_effect=InterruptedError('SIGTERM during validation')):
            with self.assertRaisesRegex(InterruptedError, 'during validation'):
                self.fit('failed-validation')
        output = self.root / 'failed-validation'
        run = runtime.read_json(output / 'run.json')
        history = runtime.read_json(output / 'history.json')
        coverage = runtime.read_json(output / 'coverage.json')
        self.assertEqual(run['cursor']['completed_updates'], 3)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[-1]['stage'], 'TAM_FROZEN')
        self.assertNotIn('selection', history[-1])
        self.assertEqual(run['validation_timings'][0]['status'], 'FAILED')
        self.assertEqual(history[-1]['validation_seconds'], run['validation_timings'][0]['seconds'])
        self.assertEqual(coverage['TAM_FROZEN']['pair_presentations'], 4)
        self.assertEqual(run['checkpoint_cursor']['completed_updates'], 2)
        self.assertEqual(self.checkpoint('failed-validation')['cursor']['completed_updates'], 2)

    def test_failure_artifact_error_does_not_replace_training_error(self):
        from association.training import tmtaff as training
        write_json = runtime.write_json

        def cannot_write_history(path, value):
            if Path(path).name == 'history.json':
                raise OSError('journal unavailable')
            return write_json(path, value)

        with (patch.object(training, '_update', side_effect=InterruptedError('original SIGTERM')),
              patch.object(runtime, 'write_json', side_effect=cannot_write_history)):
            with self.assertRaisesRegex(InterruptedError, 'original SIGTERM'):
                self.fit('failed-artifact')
        run = runtime.read_json(self.root / 'failed-artifact' / 'run.json')
        self.assertEqual(run['failure_type'], 'InterruptedError')
        self.assertIn('history.json: OSError: journal unavailable', run['failure_artifact_errors'])
        self.assertIsNone(run['checkpoint_cursor'])

    def test_final_evaluation_failure_is_not_completion(self):
        self.config['updates'] = dict.fromkeys(STAGES, 1)
        clock = [0.]
        def fail(*args, **kwargs):
            kwargs['check']()
            clock[0] += 7
            raise TimeoutError('full evaluation budget exhausted')
        with patch('association.training.tmtaff.time.monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(TimeoutError, 'evaluation budget'):
                self.fit('failed-eval', evaluation_fn=fail)
        self.assertEqual(self.checkpoint('failed-eval')['cursor']['completed_updates'], 4)
        run = runtime.read_json(self.root / 'failed-eval' / 'run.json')
        self.assertEqual(run['status'], 'FAILED')
        self.assertEqual(run['final_evaluation_status'], 'FAILED')
        self.assertEqual(run['final_evaluation_seconds'], 7)
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
        self.config['time_policy'] = 'advisory'
        continued = self.fit('advisory-resume', resume_from=self.root / 'first')
        self.assertEqual(continued['status'], 'COMPLETED')
        self.assertEqual(continued['cursor']['completed_updates'], 8)
        self.assertGreaterEqual(continued['elapsed_seconds'], report['elapsed_seconds'])
        self.assertEqual(continued['config']['time_policy'], 'advisory')

    def test_cli_advisory_covers_preparation_and_complete_training(self):
        from contextlib import redirect_stderr, redirect_stdout
        import io
        from tests.fixtures.native_zip import make_native_zip_fixture
        from tools.experiments.tmtaff_complete_fit import main
        fixture = make_native_zip_fixture(self)
        fixture.config['training']['max_hours'] = 1e-12
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--device', 'cpu', *fixture.cli_args('hard')]), 1)
            self.assertFalse((fixture.root/'hard').exists())
            self.assertEqual(main(['--device', 'cpu', '--time-policy', 'advisory',
                                   *fixture.cli_args('advisory')]), 0)
        run = runtime.read_json(fixture.root/'advisory'/'run.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['cursor']['completed_updates'], 4)
        self.assertEqual(run['config']['time_policy'], 'advisory')
        prepared = runtime.read_json(fixture.root/'advisory'/'preparation.json')
        self.assertEqual(prepared['configuration']['training']['time_policy'], 'advisory')


if __name__ == '__main__':
    unittest.main()
