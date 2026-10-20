"""Evaluation-only reuse preserves the frozen selected model and full V population."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
from dataclasses import asdict
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.utils import runtime
from association.training.tmtaff import STAGES, _configuration, _scenes, _states, load_complete_snapshot
from association.models.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.models.tmtaff import ModelConfig
from association.training.common import fit_normalizer
from association.models.timed import TimedAssociationModel
from tools.experiments import tmtaff_complete_evaluate as cli
from tests.fixtures import native_zip as native_fixture


class CompleteEvaluateCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        fixture = native_fixture.make_native_zip_fixture(self)
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
            for target in ('association.training.tmtaff.train_complete',
                           'association.training.tmtaff._optimizer',
                           'association.training.tmtaff.calibrate_training_margin',
                           'association.training.tmtaff.fit_normalizer',
                           'association.training.common.fit_normalizer',
                           'association.data.timed.fit_timed_normalizer',
                           'association.evaluation.tmtaff.select_threshold',
                           'torch.optim.Adam', 'torch.optim.AdamW', 'torch.optim.SGD',
                           'association.data.archive.MTADArchive.archive_sha256'):
                stack.enter_context(patch(target, side_effect=AssertionError('Forbidden in evaluation: '+target)))
            restored = stack.enter_context(patch('association.training.tmtaff.load_complete_snapshot',
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

    def test_tcm_feedback_cli_accepts_completed_run_without_candidate_or_reference_reads(self):
        from tools.experiments import completion_check
        saved = copy.deepcopy(self.saved)
        saved.update(phase='tcm', cursor=dict(stage=0, update=2, completed_updates=2), selected=None,
                     effective_margin=saved['model_config']['margin'], states=self.saved['selected']['states'])
        saved['config']['updates'] = dict(TCM=2, TAM_FROZEN=0, TFM_FROZEN=0, JOINT=0)
        torch.save(saved, self.checkpoint)
        runtime.write_json(self.source/'run.json', dict(status='COMPLETED', phase='tcm',
            cursor=dict(stage=1, update=0, completed_updates=2)))
        args = ['--archive', str(self.fixture.path), '--manifest', str(self.root/'manifest.json'),
            '--alignment', str(self.root/'alignment.json'), '--reference-checkpoint', str(self.source),
            '--output', str(self.root/'feedback'), '--device', 'cpu', '--cpu-threads', '1']
        with patch.object(completion_check, 'SCENE_IDS', ['2', '4']), \
             patch('association.data.reference_states.iter_reference_states', side_effect=AssertionError('no reference')) , \
             patch('association.training.cooperative.load_cooperative_checkpoint', side_effect=AssertionError('no candidate')), \
             patch('association.evaluation.tmtaff.evaluate_tcm_completion', side_effect=AssertionError('no query evaluation')), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                completion_check.main(args)  # Original mode still requires the candidate.
            self.assertEqual(completion_check.main(args+['--reference-feedback']), 0)
        run = runtime.read_json(self.root/'feedback'/'run.json')
        report = runtime.read_json(self.root/'feedback'/'reference-feedback.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['reference_members_read'], [])
        self.assertEqual(run['optimizer_updates'], 0)
        self.assertEqual(report['fragment_count'], 4)
        self.assertEqual(report['insufficient_native_fragment_count'], 2)
        self.assertEqual(report['first_step_check']['all_identical'], True)
        self.assertFalse(run['test_opened'])

    def test_candidate_dynamics_cli_reads_only_native_inputs_from_completed_teacher(self):
        from association.models.cooperative import CooperativeConfig, CooperativeModel
        from association.data.cooperative import fit_cooperative_normalizer
        from tools.experiments import completion_check
        dataset = self.fixture.load()
        config = CooperativeConfig(length=4, max_observations=4, d_model=8, nhead=2,
            num_layers=2, dim_feedforward=16, dropout=.1, head_hidden=8,
            model_variant='support_residual_v2', support_time_policy='train_absmax_v1',
            heading_output_policy='observation_residual', speed_output_policy='scaled_log')
        normalizer = fit_cooperative_normalizer(dataset['train'], config)
        model = CooperativeModel(config, normalizer, support_time_scale=3.)
        with torch.no_grad():
            model.state_projection.weight.uniform_(-.01, .01)
        teacher = self.source/'teacher.pt'
        torch.save(dict(format='cooperative-physical-v2', model_variant='support_residual_v2',
            representation_version='support_residual_v2', model_config=asdict(config),
            model_state=model.state_dict(), normalizer=normalizer,
            support_time_scale_tau=3., stage='teacher', completed_updates=2,
            training=dict(teacher_updates=2)), teacher)
        runtime.write_json(self.source/'run.json', dict(status='COMPLETED', phase='teacher',
            completed_updates=2, scene_contract=_scenes(dataset['train'], dataset['validation'], 4)))
        original = teacher.read_bytes()
        args = ['--archive', str(self.fixture.path), '--manifest', str(self.root/'manifest.json'),
            '--alignment', str(self.root/'alignment.json'), '--candidate-checkpoint', str(teacher),
            '--output', str(self.root/'dynamics'), '--device', 'cpu', '--cpu-threads', '1',
            '--candidate-dynamics']
        with ExitStack() as stack:
            stack.enter_context(patch.object(completion_check, 'SCENE_IDS', ['2', '4']))
            for target in ('association.data.reference_states.iter_reference_states',
                           'association.training.cooperative.train_cooperative',
                           'association.data.cooperative.fit_cooperative_normalizer',
                           'association.models.cooperative.CooperativeModel.score_pairs',
                           'association.evaluation.cooperative.evaluate_cooperative_completion',
                           'association.evaluation.tmtaff.evaluate_tcm_completion',
                           'torch.optim.Adam'):
                stack.enter_context(patch(target, side_effect=AssertionError('forbidden: '+target)))
            decode = stack.enter_context(patch.object(CooperativeModel, 'decode_group', autospec=True,
                side_effect=CooperativeModel.decode_group))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            self.assertEqual(completion_check.main(args), 0)
            self.assertEqual(decode.call_count, 4)
            with self.assertRaises(SystemExit):
                completion_check.main(args+['--reference-checkpoint', str(self.checkpoint)])
        report = runtime.read_json(self.root/'dynamics'/'candidate-dynamics.json')
        self.assertEqual(report['fragment_count'], 4)
        self.assertEqual(report['completion_forward_count'], 4)
        run = runtime.read_json(self.root/'dynamics'/'run.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['reference_members_read'], [])
        self.assertEqual(run['optimizer_updates'], 0)
        self.assertFalse(run['test_opened'])
        self.assertEqual(teacher.read_bytes(), original)

    def test_completion_cli_uses_partial_reference_and_teacher_without_downstream(self):
        from association.models.cooperative import CooperativeConfig, CooperativeModel
        from association.data.cooperative import fit_cooperative_normalizer
        from tools.experiments import completion_check
        dataset = self.fixture.load(data_config={**self.fixture.data, 'reference_longitude_policy': 'shortest_arc'})
        saved = copy.deepcopy(self.saved)
        saved.update(cursor=dict(stage=1, update=300, completed_updates=1300),
                     effective_margin=saved['model_config']['margin'],
                     states=saved['selected']['states'],
                     data_contract=_scenes(dataset['train'], dataset['validation'], 4))
        torch.save(saved, self.checkpoint)
        config = CooperativeConfig(length=4, max_observations=4, d_model=8, nhead=2,
            num_layers=1, dim_feedforward=16, dropout=0, head_hidden=8,
            model_variant='support_residual_v2', support_time_policy='train_absmax_v1')
        normalizer = fit_cooperative_normalizer(dataset['train'], config)
        candidate = CooperativeModel(config, normalizer, support_time_scale=3.)
        objective = FusionLoss(position_loss_mode='standard_smooth_l1', longitude_loss_mode='shortest_arc')
        with torch.no_grad():
            objective.log_variance.copy_(torch.tensor([-.2, .1, .3, .4]))
            candidate.state_projection.bias[:2].copy_(torch.tensor([.001, .002]))
        teacher = self.source/'teacher.pt'
        torch.save(dict(format='cooperative-physical-v2', model_variant='support_residual_v2',
            representation_version='support_residual_v2', model_config=asdict(config),
            model_state=candidate.state_dict(), normalizer=normalizer,
            support_time_scale_tau=3., stage='teacher', completed_updates=1000,
            objective_state=objective.state_dict(), training=dict(longitude_loss_mode='shortest_arc')), teacher)
        originals = [path.read_bytes() for path in (self.checkpoint, teacher)]
        args = ['--archive', str(self.fixture.path), '--manifest', str(self.root/'manifest.json'),
            '--alignment', str(self.root/'alignment.json'), '--reference-checkpoint', str(self.checkpoint),
            '--candidate-checkpoint', str(teacher), '--output', str(self.root/'completion'),
            '--device', 'cpu', '--cpu-threads', '1']
        with ExitStack() as stack:
            stack.enter_context(patch.object(completion_check, 'SCENE_IDS', ['2', '4']))
            for target in ('association.training.tmtaff.train_complete', 'association.models.fusion.PairFusion',
                           'association.training.cooperative.train_cooperative',
                           'association.data.timed.fit_timed_normalizer',
                           'association.data.cooperative.fit_cooperative_normalizer',
                           'association.models.cooperative.CooperativeModel.score_pairs',
                           'association.evaluation.tmtaff.select_threshold'):
                stack.enter_context(patch(target, side_effect=AssertionError('forbidden: '+target)))
            self.assertEqual(completion_check.main(args), 0)
            # Evidence export must reuse the same predictions and never repeat reference inference.
            with patch('association.evaluation.tmtaff.evaluate_tcm_completion',
                       side_effect=AssertionError('reference inference forbidden')), \
                 patch.object(CooperativeModel, 'decode_group', autospec=True,
                              side_effect=CooperativeModel.decode_group) as decode:
                self.assertEqual(completion_check.main(args + ['--candidate-residuals',
                    '--output', str(self.root/'residuals')]), 0)
                self.assertEqual(decode.call_count, 4)
        report = runtime.read_json(self.root/'completion'/'summary.json')
        self.assertEqual(report['reference']['fragment_count'], 4)
        self.assertEqual(report['candidate']['fragment_count'], 4)
        self.assertEqual(report['reference']['scene_count'], 2)
        run = runtime.read_json(self.root/'completion'/'run.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['optimizer_updates'], 0)
        self.assertEqual([row['scene_id'] for row in run['reference_members_read']], ['2', '4'])
        self.assertFalse(run['test_opened'])
        residuals = runtime.read_json(self.root/'residuals'/'residual-diagnostics.json')
        self.assertEqual(residuals['objective']['log_variance'], objective.log_variance.detach().tolist())
        self.assertEqual(residuals['objective']['longitude_loss_mode'], 'shortest_arc')
        base = residuals['position_base_regions']['all']['attributes']
        self.assertIsNone(base['vel']['raw_data_loss'])
        self.assertIsNone(base['cou']['weighted_contribution'])
        summary = runtime.read_json(self.root/'residuals'/'candidate-summary.json')
        self.assertEqual(summary['methods'], report['candidate']['methods'])
        self.assertEqual(summary['per_scene'], report['candidate']['per_scene'])
        self.assertFalse((self.root/'residuals'/'reference.json').exists())
        fragments = valid_lat = 0
        for entry in residuals['prediction_files']:
            with np.load(self.root/'residuals'/entry['path'], allow_pickle=False) as arrays:
                fragments += len(arrays['node_ids'])
                valid_lat += int(arrays['target_valid'][..., 0].sum())
                self.assertEqual(str(arrays['scene_id']), entry['scene_id'])
                self.assertEqual(arrays['prediction'].shape, arrays['target_valid'].shape)
                self.assertEqual(arrays['prediction'].dtype, np.float32)
                self.assertEqual(arrays['target_valid'].dtype, bool)
                self.assertEqual(arrays['region_inside_support'].dtype, bool)
                np.testing.assert_array_equal(arrays['region_all'], True)
                self.assertTrue(np.isnan(arrays['position_base'][..., 2:]).all())
        self.assertEqual(fragments, 4)
        self.assertEqual(valid_lat, residuals['objective']['regions']['all']['attributes']['lat']['valid_count'])
        self.assertEqual(originals, [path.read_bytes() for path in (self.checkpoint, teacher)])


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
        from association.evaluation.tmtaff import evaluate_complete
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
        with patch('association.evaluation.tmtaff.evaluate_complete', side_effect=checked_evaluation):
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
                with patch('association.data.dataset.load_timed_dataset', side_effect=AssertionError('No archive access')) as load:
                    self.assertEqual(self.invoke(self.args(case)), 1)
                    load.assert_not_called()
                self.assert_status(case, 'FAILED')


if __name__ == '__main__':
    unittest.main()
