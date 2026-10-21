"""V2 public CLI: real control flow over synthetic native files, no real MTAD."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
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


if __name__ == '__main__':
    unittest.main()
