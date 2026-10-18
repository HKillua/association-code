"""Evaluation-only reuse preserves the frozen selected model and full V population."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
from dataclasses import asdict
import io
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

import torch

from association import runtime
from association.complete_training import STAGES, _configuration, _scenes, _states, load_complete_snapshot
from association.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.models import ModelConfig
from association.reference import fit_normalizer
from association.timed_models import TimedAssociationModel
from tools.experiments import tmtaff_complete_evaluate as cli
from tests import test_timed_dataset as native_fixture


class CompleteEvaluateCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        fixture = native_fixture.TimedDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.root = fixture, fixture.root
        dataset = fixture.load()
        config = _configuration(fixture.config['training'])
        config['updates'] = dict.fromkeys(STAGES, 2)
        model_config, fusion_config = ModelConfig(**fixture.config['model']), PairFusionConfig(**fixture.config['fusion'])
        model = TimedAssociationModel(model_config, config['branch'], max_observations=4)
        fusion = PairFusion(fusion_config)
        objective = FusionLoss(position_loss_mode=config['position_loss_mode'])
        states = _states((model, fusion, objective))
        # This is a synthetic selected-checkpoint fixture, not a claimed training run.
        self.saved = dict(format_version=1, input_adapter='native_time_v1',
            max_observations=4, config=config, model_config=asdict(model_config),
            fusion_config=asdict(fusion_config), normalizer=fit_normalizer(dataset['train']),
            data_contract=_scenes(dataset['train'], dataset['validation'], 4),
            cursor=dict(stage=3, update=2, completed_updates=8),
            history=[dict(stage=stage, update=update) for stage in STAGES for update in (1, 2)],
            selected=dict(states=states, threshold=.25, macro_f1=.5, stage='TAM_FROZEN',
                          update=2, fusion_stage='TFM_FROZEN', fusion_update=2),
            states=[{key: torch.zeros_like(value) for key, value in state.items()} for state in states])
        self.source = self.root/'source'
        self.source.mkdir()
        self.checkpoint = self.source/'checkpoint.pt'
        torch.save(self.saved, self.checkpoint)
        runtime.write_json(self.source/'run.json', dict(status='FAILED', failure_type='InterruptedError'))
        runtime.write_json(self.root/'manifest.json', fixture.manifest)
        runtime.write_json(self.root/'alignment.json', fixture.alignment)

    def args(self, output='evaluation'):
        return ['--checkpoint', str(self.checkpoint), '--archive', str(self.fixture.path),
                '--manifest', str(self.root/'manifest.json'), '--alignment', str(self.root/'alignment.json'),
                '--output', str(self.root/output), '--device', 'cpu', '--cpu-threads', '1',
                '--max-seconds', '60']

    def invoke(self, args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return cli.main(args)

    def assert_status(self, output, status):
        run = runtime.read_json(self.root/output/'run.json')
        self.assertEqual(run['status'], status)
        return run

    def test_full_zip_selected_snapshot_without_training_reselection_or_test_reads(self):
        original_source = {p.name: p.read_bytes() for p in self.source.iterdir()}
        with ExitStack() as stack:
            for target in ('association.complete_training.train_complete',
                           'association.complete_training._optimizer',
                           'association.complete_training.calibrate_training_margin',
                           'association.complete_training.fit_normalizer',
                           'association.reference.fit_normalizer',
                           'association.timed_inputs.fit_timed_normalizer',
                           'association.complete_evaluation.select_threshold',
                           'torch.optim.Adam', 'torch.optim.AdamW', 'torch.optim.SGD',
                           'association.data.MTADArchive.archive_sha256'):
                stack.enter_context(patch(target, side_effect=AssertionError('Forbidden in evaluation: '+target)))
            restored = stack.enter_context(patch('association.complete_training.load_complete_snapshot',
                                                wraps=load_complete_snapshot))
            self.assertEqual(self.invoke(self.args()), 0)
            restored.assert_called_once_with(str(self.checkpoint), device=torch.device('cpu'), selected=True)
        run = self.assert_status('evaluation', 'COMPLETED')
        self.assertEqual(run['source_cursor']['completed_updates'], 8)
        self.assertEqual(run['validation_scene_ids'], ['2', '4'])
        self.assertEqual(run['validation_target_count'], 4)
        self.assertEqual(run['optimizer_updates'], 0)
        self.assertFalse(run['threshold_reselected'])
        summary = runtime.read_json(self.root/'evaluation'/'summary.json')
        self.assertEqual(summary['normalizer'], self.saved['normalizer'])
        self.assertEqual(summary['evaluation']['threshold'], .25)
        self.assertIsNone(summary['evaluation']['selection'])
        self.assertEqual([r['scene_id'] for r in summary['evaluation']['records']], ['2', '4'])
        position = summary['evaluation']['fusion']['oracle']['all']['position']
        self.assertEqual(position['available_target_count'], 16)
        self.assertEqual(position['missing_prediction_count'], 8)
        self.assertGreater(summary['evaluation']['fusion_forward_count'], 0)
        self.assertFalse(summary['test_opened'])
        progress = runtime.read_json(self.root/'evaluation'/'evaluation-progress.json')
        self.assertEqual(progress['stage'], 'completed')
        self.assertEqual(progress['completed_scenes'], 2)
        self.assertEqual(original_source, {p.name: p.read_bytes() for p in self.source.iterdir()})
        original_run = (self.root/'evaluation'/'run.json').read_bytes()
        self.assertEqual(self.invoke(self.args()), 1)
        self.assertEqual((self.root/'evaluation'/'run.json').read_bytes(), original_run)

    def test_all_reject_saved_threshold_remains_none(self):
        self.saved['selected']['threshold'] = None
        torch.save(self.saved, self.checkpoint)
        with patch('association.complete_evaluation.select_threshold', side_effect=AssertionError('No reselection')):
            self.assertEqual(self.invoke(self.args()), 0)
        result = runtime.read_json(self.root/'evaluation'/'summary.json')['evaluation']
        self.assertIsNone(result['threshold'])
        self.assertTrue(all(value['tp'] == value['fp'] == 0 for value in result['association']['per_scene']))

    def test_legacy_checkpoint_without_longitude_fields_keeps_exact_contract(self):
        self.saved['config'].pop('longitude_loss_mode', None)
        self.assertTrue(all('longitude_policy' not in row['reference_alignment']
                            for population in self.saved['data_contract'] for row in population))
        torch.save(self.saved, self.checkpoint)
        restored = load_complete_snapshot(self.checkpoint)
        self.assertEqual(restored['objective'].longitude_loss_mode, 'linear')
        dataset = self.fixture.load()
        self.assertEqual(_scenes(dataset['train'], dataset['validation'], 4), self.saved['data_contract'])
        self.assertEqual(self.invoke(self.args('legacy-longitude')), 0)
        run = self.assert_status('legacy-longitude', 'COMPLETED')
        self.assertNotIn('reference_longitude_policy', run['data'])

    def test_shortest_checkpoint_reuses_actual_dateline_targets_during_evaluation(self):
        from association.complete_evaluation import evaluate_complete
        original_csv = native_fixture.csv_bytes
        def dateline_csv(header, rows):
            rows = copy.deepcopy(rows)
            if header[0] == 'batch':
                for row in rows:
                    if row[2] == 30:
                        row[2] = 29  # query 29/3 lies between reference t=0 and t=10.
            elif header[0] == 'MMSI':
                for row in rows:
                    row[3] = 179. if row[1] < 1010. else -179.
            return original_csv(header, rows)
        with patch.object(native_fixture, 'csv_bytes', side_effect=dateline_csv):
            self.fixture.write_zip()
        dataset = self.fixture.load(data_config=dict(self.fixture.data, reference_longitude_policy='shortest_arc'))
        self.saved['normalizer'] = fit_normalizer(dataset['train'])
        self.saved['data_contract'] = _scenes(dataset['train'], dataset['validation'], 4)
        self.saved['config']['longitude_loss_mode'] = 'shortest_arc'
        torch.save(self.saved, self.checkpoint)
        restored = load_complete_snapshot(self.checkpoint)
        self.assertEqual(restored['objective'].longitude_loss_mode, 'shortest_arc')
        captured = []
        def checked_evaluation(model, fusion, normalizer, scenes, **kwargs):
            target = scenes[0]['fusion_targets']['target2a-0']
            expected = 179. + (29./3.)/10.*2. - 360.
            self.assertAlmostEqual(target['states'][1, 1], expected)
            self.assertTrue(target['valid'][1, 1])
            captured.append(scenes[0]['reference_alignment']['longitude_policy'])
            return evaluate_complete(model, fusion, normalizer, scenes, **kwargs)
        with patch('association.complete_evaluation.evaluate_complete', side_effect=checked_evaluation):
            self.assertEqual(self.invoke(self.args('shortest-longitude')), 0)
        self.assertEqual(captured, ['shortest_arc'])
        run = self.assert_status('shortest-longitude', 'COMPLETED')
        self.assertEqual(run['data']['reference_longitude_policy'], 'shortest_arc')
        self.assertEqual(self.saved['data_contract'], _scenes(dataset['train'], dataset['validation'], 4))

    def test_changed_manifest_alignment_and_incomplete_updates_fail_before_archive(self):
        for case in ('manifest', 'alignment', 'updates', 'time_unit', 'no_threshold', 'missing_fusion'):
            with self.subTest(case=case):
                manifest, alignment, saved = copy.deepcopy(self.fixture.manifest), copy.deepcopy(self.fixture.alignment), copy.deepcopy(self.saved)
                if case == 'manifest':
                    manifest['splits']['V_select'] = ['2']
                elif case == 'alignment':
                    alignment['2']['time_offset_seconds'] += 1
                elif case == 'updates':
                    saved['cursor']['completed_updates'] -= 1
                elif case == 'time_unit':
                    saved['normalizer']['adapter_config']['time_unit_seconds'] += 1
                elif case == 'no_threshold':
                    del saved['selected']['threshold']
                else:
                    saved['selected']['states'] = saved['selected']['states'][:1]
                runtime.write_json(self.root/'manifest.json', manifest)
                runtime.write_json(self.root/'alignment.json', alignment)
                torch.save(saved, self.checkpoint)
                with patch('association.timed_dataset.load_timed_dataset', side_effect=AssertionError('No archive access')) as load:
                    self.assertEqual(self.invoke(self.args(case)), 1)
                    load.assert_not_called()
                self.assert_status(case, 'FAILED')

    def test_prepared_contract_changes_are_rejected_before_model_allocation(self):
        self.saved['data_contract'][1][0]['node_ids'][0] = 'different-node'
        torch.save(self.saved, self.checkpoint)
        with patch('association.complete_training.load_complete_snapshot') as restore:
            self.assertEqual(self.invoke(self.args()), 1)
            restore.assert_not_called()
        run = self.assert_status('evaluation', 'FAILED')
        self.assertIn('prepared scene contract', run['failure'])

    def test_preparation_time_counts_and_failure_is_preserved(self):
        for kind in ('deadline', 'data_error'):
            with self.subTest(kind=kind):
                clock = [0.]
                def load(*args, **kwargs):
                    if kind == 'data_error':
                        raise ValueError('malformed input')
                    clock[0] = 61.
                    kwargs['check']()
                with patch.object(cli.time, 'monotonic', side_effect=lambda: clock[0]), \
                        patch('association.timed_dataset.load_timed_dataset', side_effect=load), \
                        patch('association.complete_training.load_complete_snapshot') as restore:
                    self.assertEqual(self.invoke(self.args(kind)), 124 if kind == 'deadline' else 1)
                    restore.assert_not_called()
                run = self.assert_status(kind, 'INTERRUPTED' if kind == 'deadline' else 'FAILED')
                self.assertEqual(run['stage'], 'preparation')
                self.assertFalse((self.root/kind/'summary.json').exists())

    def test_sigterm_retains_last_physical_fusion_progress(self):
        def interrupted(*args, **kwargs):
            kwargs['progress'](dict(stage='physical_fusion', completed_scenes=0, scene_id='2',
                                    total_scenes=2, fusion_forward_count=0, elapsed_seconds=.1))
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        with patch('association.complete_evaluation.evaluate_complete', side_effect=interrupted):
            self.assertEqual(self.invoke(self.args()), 130)
        run = self.assert_status('evaluation', 'INTERRUPTED')
        self.assertEqual(run['failure_type'], 'InterruptedError')
        progress = runtime.read_json(self.root/'evaluation'/'evaluation-progress.json')
        self.assertEqual(progress['stage'], 'physical_fusion')
        self.assertEqual(progress, run['evaluation_progress'])
        self.assertFalse((self.root/'evaluation'/'summary.json').exists())

    def test_result_io_overrun_does_not_leave_success_summary(self):
        clock = [0.]
        original_write = runtime.write_json
        def write(path, value):
            original_write(path, value)
            if Path(path).name == 'summary.json':
                clock[0] = 61.
        with patch.object(cli.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(runtime, 'write_json', side_effect=write):
            self.assertEqual(self.invoke(self.args()), 124)
        self.assert_status('evaluation', 'INTERRUPTED')
        self.assertFalse((self.root/'evaluation'/'summary.json').exists())
        self.assertTrue((self.root/'evaluation'/'summary.incomplete.json').exists())

    def test_device_and_budget_must_be_explicit(self):
        for field in ('--device', '--max-seconds'):
            args = self.args()
            index = args.index(field)
            del args[index:index+2]
            with self.subTest(field=field), self.assertRaises(SystemExit) as error:
                self.invoke(args)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((self.root/'evaluation').exists())


if __name__ == '__main__':
    unittest.main()
