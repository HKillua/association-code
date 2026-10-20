"""V2 public CLI: real control flow over synthetic native files, no real MTAD."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.utils import runtime
from association.models.cooperative import CooperativeConfig
from association.training.cooperative import load_cooperative_checkpoint
from tools.experiments.cooperative_fit import main, synthetic_configuration
from tests.fixtures import native_zip as native_fixture


class CooperativeRevisionCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, self.threads)

    def invoke(self, args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(args)

    def configuration(self):
        return dict(scope='cooperative-physical-v2',
            model=asdict(CooperativeConfig(length=4, max_observations=4, d_model=8,
                nhead=2, num_layers=1, dim_feedforward=16, head_hidden=8, dropout=0.,
                model_variant='support_residual_v2', support_time_policy='train_absmax_v1')),
            data={'time_unit_seconds': 10., 'reference_longitude_policy': 'shortest_arc'}, train_splits=['T_core'],
            training=synthetic_configuration('support_residual_v2'))

    def test_public_synthetic_v2_runs_and_reloads_all_arms_with_one_identity(self):
        output = self.root/'check'
        args = ['check', '--device', 'cpu', '--output', str(output), '--cpu-threads', '1',
                '--model-variant', 'support_residual_v2', '--save-predictions']
        self.assertEqual(self.invoke(args), 0)
        run = runtime.read_json(output/'run.json')
        self.assertEqual(run['completed_updates'], 6)
        self.assertEqual(run['completed_routes'], ['plain', 'scalar-u', 'attribute-u', 'learned-attribute'])
        self.assertEqual(run['identity_selection_phase'], 'teacher_endpoint_before_utility')
        self.assertEqual(run['training']['longitude_loss_mode'], 'shortest_arc')
        first = None
        for route in run['completed_routes']:
            model, payload = load_cooperative_checkpoint(output/('adapt-'+route+'.pt'), device='cpu')
            self.assertEqual(model.config.model_variant, 'support_residual_v2')
            self.assertEqual(payload['format'], 'cooperative-physical-v2')
            self.assertEqual(payload['completed_updates'], 1)
            self.assertEqual(payload['training']['longitude_loss_mode'], 'shortest_arc')
            with np.load(output/(route+'-scores.npz')) as arrays:
                scores = arrays['scores_0'].copy()
                if first is None:
                    first = scores
                np.testing.assert_array_equal(scores, first)
            report = runtime.read_json(output/(route+'-evaluation.json'))
            self.assertEqual(report['threshold'], run['threshold'])
            self.assertFalse(report['test_opened'])
            for kind in ('predicted', 'oracle'):
                self.assertIn('input_support_metrics', report['fusion'][kind])
                self.assertIn('input_distance_metrics', report['fusion'][kind])
            self.assertTrue((output/(route+'-predictions/index.json')).is_file())
        prior = (output/'run.json').read_bytes()
        self.assertEqual(self.invoke(args), 1)
        self.assertEqual((output/'run.json').read_bytes(), prior)

    def test_v2_native_zip_keeps_reference_only_scene_in_population(self):
        fixture = native_fixture.make_native_zip_fixture(self)
        fixture.config = self.configuration()
        fixture.config['training']['max_seconds'] = 1e-12
        self.assertEqual(self.invoke(['fit', '--device', 'cpu', '--cpu-threads', '1',
                         '--time-policy', 'advisory', '--save-predictions', *fixture.cli_args('v2')]), 0)
        output = fixture.root/'v2'
        run = runtime.read_json(output/'run.json')
        self.assertEqual(run['completed_updates'], 6)
        self.assertEqual(run['training']['time_policy'], 'advisory')
        preparation = runtime.read_json(output/'execution.json')['preparation']
        self.assertFalse(preparation['test_opened'])
        self.assertEqual(preparation['reference_longitude_policy'], 'shortest_arc')
        self.assertTrue(all(scene['reference_alignment']['longitude_policy'] == 'shortest_arc'
                            for population in run['scene_contract'] for scene in population))
        for route in run['completed_routes']:
            report = runtime.read_json(output/(route+'-evaluation.json'))
            self.assertEqual(len(report['per_scene']), 2)
            empty = report['per_scene'][1]['methods']['predicted']['full_target_metrics']['all']['position']
            self.assertGreater(empty['available_target_count'], 0)
            self.assertEqual(empty['intersection_count'], 0)
            unsupported = report['per_scene'][1]['methods']['predicted']['input_support_metrics']['unmatched_target']['position']
            self.assertEqual(unsupported['available_target_count'], empty['available_target_count'])
            self.assertIsNone(unsupported['mae'])
            entries = runtime.read_json(output/(route+'-predictions/index.json'))['scenes']
            with np.load(output/(route+'-predictions')/entries[1]['npz']) as arrays:
                self.assertGreater(arrays['m000__aligned_truth_valid'].sum(), 0)
                self.assertEqual(arrays['m000__aligned_prediction_valid'].sum(), 0)

    def test_public_synthetic_teacher_is_v2_only_and_saves_completion(self):
        output = self.root/'teacher-check'
        common = ['check', '--device', 'cpu', '--cpu-threads', '1',
                  '--phase', 'teacher', '--save-predictions', '--output', str(output)]
        self.assertEqual(self.invoke(common), 1)
        self.assertFalse(output.exists())
        self.assertEqual(self.invoke([*common, '--model-variant', 'support_residual_v2']), 0)
        run = runtime.read_json(output/'run.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['phase'], 'teacher')
        self.assertEqual(run['completed_updates'], 1)
        self.assertEqual(run['training']['utility_updates'], 0)
        self.assertEqual(run['training']['adapt_updates'], 0)
        self.assertNotIn('threshold', run)
        self.assertNotIn('identity_selection', run)
        self.assertEqual(runtime.read_json(output/'completion-evaluation.json')['fragment_count'], 4)
        self.assertTrue(list((output/'completion-predictions').glob('*.npz')))

    def test_native_scaled_log_teacher_has_no_downstream_training_or_selection(self):
        fixture = native_fixture.make_native_zip_fixture(self)
        fixture.config = self.configuration()
        fixture.config['model']['speed_output_policy'] = 'scaled_log'
        fixture.config['training'].update(utility_updates=0, adapt_updates=0, max_seconds=1e-12)
        output = fixture.root/'teacher-scaled-log'
        with (patch('association.evaluation.association.select_threshold',
                    side_effect=AssertionError('teacher does not select an association threshold')),
              patch('association.evaluation.cooperative.score_cooperative',
                    side_effect=AssertionError('teacher does not score validation associations')),
              patch('association.evaluation.cooperative.evaluate_cooperative',
                    side_effect=AssertionError('teacher does not run grouped fusion'))):
            self.assertEqual(self.invoke(['fit', '--device', 'cpu', '--cpu-threads', '1',
                '--phase', 'teacher', '--time-policy', 'advisory', '--save-predictions',
                *fixture.cli_args('teacher-scaled-log')]), 0)
        run = runtime.read_json(output/'run.json')
        self.assertEqual(run['status'], 'COMPLETED')
        self.assertEqual(run['phase'], 'teacher')
        self.assertEqual(run['completed_updates'], 1)
        self.assertEqual(run['completed_routes'], [])
        self.assertEqual(run['training']['time_policy'], 'advisory')
        self.assertEqual(run['phases']['teacher']['completed_updates'], 1)
        self.assertFalse(set(run['phases']) & {'identity_scoring_selection', 'utility', 'shared_export'})
        self.assertFalse(any(name.startswith('adapt') for name in run['phases']))
        self.assertNotIn('threshold', run)
        self.assertNotIn('identity_selection', run)
        model, payload = load_cooperative_checkpoint(output/'teacher.pt', device='cpu')
        self.assertEqual(model.config.speed_output_policy, 'scaled_log')
        self.assertEqual(payload['stage'], 'teacher')
        self.assertEqual(payload['completed_updates'], 1)
        self.assertFalse((output/'utility.pt').exists())
        self.assertFalse((output/'shared.pt').exists())
        self.assertFalse(list(output.glob('adapt-*.pt')))
        report = runtime.read_json(output/'completion-evaluation.json')
        self.assertEqual(report['scene_count'], 2)
        self.assertEqual(report['fragment_count'], 4)
        self.assertEqual(report['completion_forward_count'], 4)
        self.assertFalse(report['test_opened'])
        self.assertEqual(report['per_scene'][1]['fragment_count'], 0)
        self.assertTrue(report['per_scene'][1]['reference_targets_without_input'])
        files = sorted((output/'completion-predictions').glob('*.npz'))
        self.assertEqual(len(files), 2)
        fragments = 0
        for path in files:
            with np.load(path) as arrays:
                prediction = arrays['prediction']
                self.assertEqual(prediction.shape[1:], (4, 4))
                self.assertEqual(arrays['target'].shape, prediction.shape)
                self.assertEqual(arrays['target_valid'].shape, prediction.shape)
                self.assertTrue(np.isfinite(prediction).all())
                self.assertEqual(len(arrays['node_ids']), len(prediction))
                fragments += len(prediction)
        self.assertEqual(fragments, 4)
        execution = runtime.read_json(output/'execution.json')
        self.assertEqual(execution['phase'], 'teacher')
        self.assertEqual(execution['configuration']['model']['speed_output_policy'], 'scaled_log')
        self.assertFalse(execution['preparation']['test_opened'])

    def test_observed_heading_teacher_exports_reloads_and_retains_group_cache(self):
        from association.evaluation.cooperative import _score_and_cache, _take
        from association.data.cooperative import normalize_cooperative_scene
        fixture = native_fixture.make_native_zip_fixture(self)
        fixture.config = self.configuration()
        fixture.config['model'].update(speed_output_policy='scaled_log', heading_output_policy='observation_residual')
        fixture.config['training'].update(utility_updates=0, adapt_updates=0)
        self.assertEqual(self.invoke(['fit', '--device', 'cpu', '--cpu-threads', '1',
            '--phase', 'teacher', '--save-predictions', *fixture.cli_args('heading')]), 0)
        output = fixture.root/'heading'
        model, payload = load_cooperative_checkpoint(output/'teacher.pt', device='cpu')
        self.assertEqual(model.config.heading_output_policy, 'observation_residual')
        self.assertEqual(payload['model_config']['heading_output_policy'], 'observation_residual')
        self.assertEqual(model.heading_base_version.item(), 1)
        self.assertEqual(runtime.read_json(output/'run.json')['status'], 'COMPLETED')
        data = fixture.load(data_config={**fixture.data, 'reference_longitude_policy': 'shortest_arc'})
        scenes = data['validation']
        # A cached pair/group view must carry the raw native heading alongside
        # its mask and time. This is additional tiny inference, never training.
        _, cache = _score_and_cache(model, payload['normalizer'], scenes, torch.device('cpu'),
            torch.float32, 2, 2, False, None, None)
        self.assertIsNone(cache[1])
        values, observed = normalize_cooperative_scene(scenes[0], payload['normalizer'], model.config)
        direct = model.encode(torch.as_tensor(values), torch.as_tensor(observed))
        subset = _take(cache[0], [1, 0], torch.device('cpu'))
        torch.testing.assert_close(subset['native_headings'], direct['native_headings'][[1, 0]], rtol=0, atol=0)
        with torch.no_grad():
            cached_prediction = model.decode_group(subset, [0, 1])
            direct_prediction = model.decode_group(direct, [1, 0])
        torch.testing.assert_close(cached_prediction, direct_prediction, rtol=1e-6, atol=1e-5)
        entries = runtime.read_json(output/'completion-predictions'/'index.json')
        self.assertEqual([item['scene_id'] for item in entries], ['2', '4'])
        for entry in entries:
            with np.load(output/'completion-predictions'/entry['path']) as arrays:
                self.assertTrue(np.isfinite(arrays['prediction']).all())
                if entry['scene_id'] == '2':
                    with torch.no_grad():
                        prediction = model.decode_group(direct, [0]).cpu().numpy()
                    np.testing.assert_allclose(arrays['prediction'][0], prediction, rtol=1e-6, atol=1e-5)
                else:
                    self.assertEqual(arrays['prediction'].shape, (0, 4, 4))

    def test_native_teacher_decoder_ablations_train_reload_and_export_full_population(self):
        from association.data.cooperative import normalize_cooperative_scene
        from association.evaluation.cooperative_dynamics import evaluate_cooperative_dynamics
        variants = (
            ('pre-norm', dict(decoder_norm_first=True), 1., 3, True),
            ('shallow-decoder', dict(decoder_num_layers=2), 1., 2, False),
            ('without-identity', {}, 0., 3, False),
        )
        for name, model_change, identity_weight, decoder_depth, norm_first in variants:
            with self.subTest(variant=name):
                fixture = native_fixture.make_native_zip_fixture(self)
                fixture.config = self.configuration()
                fixture.config['model'].update(
                    num_layers=3, speed_output_policy='scaled_log',
                    heading_output_policy='observation_residual', **model_change)
                fixture.config['training'].update(
                    teacher_updates=2, utility_updates=0, adapt_updates=0,
                    teacher_identity_weight=identity_weight)
                output = fixture.root/name
                with (patch('association.evaluation.association.select_threshold',
                            side_effect=AssertionError('teacher must not select an association threshold')),
                      patch('association.evaluation.cooperative.score_cooperative',
                            side_effect=AssertionError('teacher must not score validation associations')),
                      patch('association.evaluation.cooperative.evaluate_cooperative',
                            side_effect=AssertionError('teacher must not run grouped fusion'))):
                    self.assertEqual(self.invoke([
                        'fit', '--device', 'cpu', '--phase', 'teacher', '--save-predictions',
                        *fixture.cli_args(name)]), 0)
                run = runtime.read_json(output/'run.json')
                self.assertEqual(run['status'], 'COMPLETED')
                self.assertEqual(run['phase'], 'teacher')
                self.assertEqual(run['completed_updates'], 2)
                self.assertEqual(run['completed_routes'], [])
                self.assertEqual(run['training']['teacher_identity_weight'], identity_weight)
                self.assertFalse(run['identity_selection_performed'])
                self.assertNotIn('threshold', run)
                self.assertEqual(set(run['phases']), {
                    'external_preparation', 'local_preparation', 'teacher', 'completion_evaluation'})
                self.assertFalse((output/'shared.pt').exists())
                self.assertFalse((output/'utility.pt').exists())
                self.assertFalse(list(output.glob('adapt-*.pt')))

                model, payload = load_cooperative_checkpoint(output/'teacher.pt', device='cpu')
                self.assertEqual(payload['completed_updates'], 2)
                self.assertEqual(payload['training']['teacher_identity_weight'], identity_weight)
                self.assertEqual(len(model.encoder.layers), 3)
                self.assertEqual(len(model.decoder.layers), decoder_depth)
                self.assertTrue(all(not layer.norm_first for layer in model.encoder.layers))
                self.assertTrue(all(layer.norm_first == norm_first for layer in model.decoder.layers))
                # The residual head starts at zero; nonzero saved weights and
                # Adam step counters establish actual updates through the CLI.
                self.assertGreater(torch.count_nonzero(model.state_projection.weight).item(), 0)
                self.assertTrue(payload['optimizer_state']['state'])
                self.assertEqual(
                    {int(state['step']) for state in payload['optimizer_state']['state'].values()}, {2})
                history = runtime.read_json(output/'history.json')
                self.assertEqual(len(history), 2)
                for row in history:
                    self.assertEqual(row['stage'], 'teacher')
                    self.assertTrue(np.isfinite(row['gradient_norm']))
                    components = row['loss_components']
                    self.assertEqual(row['teacher_identity_weight'], identity_weight)
                    self.assertGreater(row['identity_bce_raw'], 0.)
                    self.assertAlmostEqual(components['identity_bce'],
                                           identity_weight*row['identity_bce_raw'], places=6)
                    self.assertAlmostEqual(row['loss'],
                                           components['physical']+components['identity_bce'], places=5)

                report = runtime.read_json(output/'completion-evaluation.json')
                self.assertEqual(report['scene_count'], 2)
                self.assertEqual(report['fragment_count'], 4)
                self.assertEqual(report['completion_forward_count'], 4)
                self.assertFalse(report['test_opened'])
                self.assertTrue(report['per_scene'][1]['reference_targets_without_input'])
                execution = runtime.read_json(output/'execution.json')
                self.assertFalse(execution['preparation']['test_opened'])
                self.assertEqual(execution['configuration']['training']['teacher_identity_weight'],
                                 identity_weight)
                entries = runtime.read_json(output/'completion-predictions'/'index.json')
                self.assertEqual([entry['scene_id'] for entry in entries], ['2', '4'])
                dataset = fixture.load(data_config={
                    **fixture.data, 'reference_longitude_policy': 'shortest_arc'})
                model.eval()
                if name != 'without-identity':
                    dynamics = evaluate_cooperative_dynamics(
                        model, payload['normalizer'], dataset['validation'], device='cpu')
                    self.assertEqual(dynamics['decoder_layer_count'], decoder_depth)
                    self.assertEqual(dynamics['completion_forward_count'], 4)
                    self.assertTrue(dynamics['checks']['input_and_physical_reconstruction_exact'])
                    self.assertEqual(
                        {key for key in dynamics['aggregate']
                         if key.startswith('decoder_layer_')},
                        {f'decoder_layer_{layer}_centered_rms'
                         for layer in range(1, decoder_depth+1)})
                for scene, entry in zip(dataset['validation'], entries):
                    with np.load(output/'completion-predictions'/entry['path']) as arrays:
                        prediction = arrays['prediction']
                        self.assertEqual(prediction.shape, (len(scene['node_ids']), 4, 4))
                        self.assertEqual(arrays['target'].shape, prediction.shape)
                        self.assertEqual(arrays['target_valid'].shape, prediction.shape)
                        self.assertTrue(np.isfinite(prediction).all())
                        if not scene['node_ids']:
                            continue
                        values, observed = normalize_cooperative_scene(
                            scene, payload['normalizer'], model.config)
                        with torch.no_grad():
                            encoded = model.encode(torch.as_tensor(values), torch.as_tensor(observed))
                            reloaded = torch.stack([
                                model.decode_group(encoded, [node])
                                for node in range(len(scene['node_ids']))]).numpy()
                        np.testing.assert_allclose(prediction, reloaded, rtol=1e-6, atol=1e-5)

    def test_shared_and_one_independent_arm_use_public_cli(self):
        common = ['check', '--device', 'cpu', '--cpu-threads', '1',
                  '--model-variant', 'support_residual_v2']
        shared, arm = self.root/'shared', self.root/'arm'
        self.assertEqual(self.invoke([*common, '--phase', 'shared', '--output', str(shared)]), 0)
        self.assertTrue((shared/'shared.pt').is_file())
        self.assertFalse(list(shared.glob('*-evaluation.json')))
        # Existing 021 endpoints predate explicit timing/output/ablation fields.
        old_run = runtime.read_json(shared/'run.json')
        old_run['training'].pop('time_policy')
        old_run['training'].pop('teacher_identity_weight')
        old_run['model_config'].pop('speed_output_policy')
        old_run['model_config'].pop('heading_output_policy')
        old_run['model_config'].pop('decoder_norm_first')
        old_run['model_config'].pop('decoder_num_layers')
        runtime.write_json(shared/'run.json', old_run)
        old_checkpoint = torch.load(shared/'shared.pt', weights_only=True)
        old_checkpoint['training'].pop('time_policy')
        old_checkpoint['training'].pop('teacher_identity_weight')
        old_checkpoint['model_config'].pop('speed_output_policy')
        old_checkpoint['model_config'].pop('heading_output_policy')
        old_checkpoint['model_config'].pop('decoder_norm_first')
        old_checkpoint['model_config'].pop('decoder_num_layers')
        torch.save(old_checkpoint, shared/'shared.pt')
        self.assertEqual(self.invoke([*common, '--phase', 'arm', '--shared-from', str(shared),
                         '--time-policy', 'advisory',
                         '--route', 'attribute-u', '--save-predictions', '--output', str(arm)]), 0)
        run = runtime.read_json(arm/'run.json')
        self.assertEqual(run['completed_updates'], 1)
        self.assertEqual(run['completed_routes'], ['attribute-u'])
        self.assertEqual(run['training']['time_policy'], 'advisory')
        self.assertEqual(runtime.read_json(arm/'execution.json')['phase'], 'arm')
        self.assertTrue((arm/'attribute-u-predictions/index.json').is_file())
        self.assertFalse((arm/'teacher.pt').exists())
        self.assertFalse((arm/'utility.pt').exists())

    def test_second_sigterm_during_terminal_write_keeps_original_failure(self):
        from tools.experiments.tmtaff_complete_fit import main as reference_main
        for name, entry, update, terminal in (
                ('reference', reference_main, 'association.training.tmtaff._update', 'FAILED'),
                ('candidate', main, 'association.training.cooperative.train_batch', 'INTERRUPTED')):
            with self.subTest(name=name):
                fixture = native_fixture.make_native_zip_fixture(self)
                prefix = ['--device', 'cpu']
                if name == 'candidate':
                    fixture.config = self.configuration()
                    prefix.insert(0, 'fit')
                output = fixture.root/name
                original_signal = signal.getsignal(signal.SIGTERM)
                write_json = runtime.write_json
                repeated = []
                def write_terminal(path, value):
                    if Path(path) == output/'run.json' and value.get('status') == terminal:
                        repeated.append(True)
                        signal.raise_signal(signal.SIGTERM)
                    return write_json(path, value)
                with (patch(update, side_effect=lambda *a, **kw: signal.raise_signal(signal.SIGTERM)),
                      patch.object(runtime, 'write_json', side_effect=write_terminal),
                      redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                    self.assertEqual(entry([*prefix, *fixture.cli_args(name)]), 1)
                self.assertEqual(len(repeated), 1)
                self.assertEqual(runtime.read_json(output/'run.json')['status'], terminal)
                self.assertEqual(signal.getsignal(signal.SIGTERM), original_signal)
                self.assertTrue((output/('preparation.json' if name == 'reference' else 'execution.json')).exists())

    def test_no_budget_or_inconsistent_variant_fails_before_opening_archive(self):
        cases = []
        value = self.configuration(); value['training']['max_seconds'] = None; cases.append(value)
        value = self.configuration(); value['scope'] = 'cooperative-physical-v1'; cases.append(value)
        value = self.configuration(); value['training']['routes'] = ['plain', 'utility']; cases.append(value)
        for index, config in enumerate(cases):
            with self.subTest(index=index):
                path = self.root/f'config{index}.json'
                runtime.write_json(path, config)
                output = self.root/f'blocked{index}'
                with patch('association.data.dataset.load_timed_dataset', side_effect=AssertionError('archive must stay closed')) as loader:
                    self.assertEqual(self.invoke(['fit', '--device', 'cpu', '--cpu-threads', '1',
                        '--archive', 'not-opened.zip', '--manifest', 'not-opened.json',
                        '--alignment', 'not-opened.json', '--config', str(path), '--output', str(output)]), 1)
                    loader.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
