"""CPU-only observational timing, interruption and scientific-invariance checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association import runtime
from association.cooperative import CooperativeConfig, CooperativeModel
from association.cooperative_evaluation import evaluate_cooperative, score_cooperative
from association.cooperative_training import train_cooperative
from association.reference import fit_normalizer
from tests.test_timed_pipeline import timed_scene


class CooperativeProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.train = [timed_scene('train', 'T_core')]
        self.validation = [timed_scene(name, 'V_select') for name in ('val-one', 'val-two')]
        self.config = CooperativeConfig(length=4, max_observations=3, d_model=8,
            nhead=2, num_layers=1, dim_feedforward=16, dropout=.2, head_hidden=6)
        self.training = dict(seed=20, batch_size=2, teacher_updates=1, utility_updates=1,
            adapt_updates=1, learning_rate=.001, gradient_clip=1., endpoint_fraction=.25,
            utility_epsilon=.001, error_scale_policy='training_native_std',
            routes=['plain', 'utility'], max_seconds=300., batch_tracks=2, pair_batch_size=3)

    def fit(self, output, **kwargs):
        return train_cooperative(self.train, self.validation, self.config, self.training,
                                 output, device='cpu', **kwargs)

    def assert_same(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_same(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_same(a, b)
        else:
            self.assertEqual(left, right)

    def test_evaluation_observer_preserves_all_scores_metrics_population_and_rng(self):
        torch.manual_seed(42)
        normalizer = fit_normalizer(self.train)
        model = CooperativeModel(self.config, normalizer)
        options = dict(device='cpu', batch_tracks=2, pair_batch_size=3,
                       select_on_validation=True, threshold=None, routing='utility')
        rng = torch.get_rng_state().clone()
        expected = evaluate_cooperative(model, normalizer, self.validation, **options)
        events = []
        actual = evaluate_cooperative(model, normalizer, self.validation, progress=events.append, **options)
        expected.pop('elapsed_seconds'); actual.pop('elapsed_seconds')
        self.assert_same(expected, actual)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(events[-1]['stage'], 'completed')
        self.assertEqual(events[-1]['completed_scenes'], 2)
        scenes = [item['scene_id'] for item in events if item['stage'] == 'physical_fusion'
                  and item['completed_scenes'] == 0 and item['scene_id'] is not None]
        self.assertEqual(scenes, ['val-one'])
        scoring = []
        scored = score_cooperative(model, normalizer, self.validation, 'cpu', 2, 3, progress=scoring.append)
        self.assert_same(scored, expected['records'])
        self.assertEqual(scoring[-1]['completed_scenes'], 2)
        self.assertTrue(all('memory' not in item and 'states' not in item for item in events))

    def test_training_observer_preserves_checkpoint_optimizer_history_scores_and_rng(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            left, right = [Path(directory) / name for name in ('no-observer', 'observer')]
            first = self.fit(left)
            rng = torch.get_rng_state().clone()
            second = self.fit(right, progress=events.append)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            for name in ('teacher', 'utility', 'adapt-plain', 'adapt-utility'):
                self.assert_same(torch.load(left / (name+'.pt'), weights_only=True),
                                 torch.load(right / (name+'.pt'), weights_only=True))
            histories = [runtime.read_json(path/'history.json') for path in (left, right)]
            for history in histories:
                for row in history:
                    row.pop('seconds')
            self.assert_same(*histories)
            for route in self.training['routes']:
                reports = [runtime.read_json(path/(route+'-evaluation.json')) for path in (left, right)]
                for report in reports:
                    report.pop('elapsed_seconds')
                self.assert_same(*reports)
                with np.load(left/(route+'-scores.npz')) as a, np.load(right/(route+'-scores.npz')) as b:
                    self.assert_same(dict(a), dict(b))
            self.assertEqual(first['threshold'], second['threshold'])
            self.assertEqual(second['completed_updates'], 4)
            self.assertEqual(set(second['phases']), {'external_preparation', 'local_preparation', 'teacher',
                'utility', 'identity_scoring_selection', 'adapt:plain', 'evaluation:plain',
                'adapt:utility', 'evaluation:utility'})
            self.assertTrue(all(phase['status'] == 'COMPLETED' for phase in second['phases'].values()))
            final = runtime.read_json(right/'progress.json')
            self.assertEqual(final['status'], 'COMPLETED')
            self.assertEqual(final['completed_routes'], self.training['routes'])
            selected = [item for item in events if item['stage'] == 'threshold_selected']
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]['selection'], second['identity_selection'])

    def test_prediction_export_preserves_metrics_and_reloads_actual_aligned_arrays(self):
        from association.fusion_metrics import evaluate_aligned_fusion
        from association.prediction_artifacts import ScenePredictionWriter
        torch.manual_seed(42)
        normalizer = fit_normalizer(self.train)
        model = CooperativeModel(self.config, normalizer)
        options = dict(device='cpu', batch_tracks=2, pair_batch_size=3,
                       select_on_validation=False, threshold=None, routing='utility')
        expected = evaluate_cooperative(model, normalizer, self.validation, **options)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'predictions'
            writer = ScenePredictionWriter(output)
            actual = evaluate_cooperative(model, normalizer, self.validation,
                                          prediction_writer=writer, **options)
            expected.pop('elapsed_seconds'); actual.pop('elapsed_seconds')
            self.assert_same(expected, actual)
            index = runtime.read_json(output/'index.json')
            self.assertEqual(len(index['scenes']), 2)
            for scene_index, entry in enumerate(index['scenes']):
                scene = self.validation[scene_index]
                metadata = runtime.read_json(output/entry['metadata'])
                self.assertEqual(entry['scene_id'], scene['scene_id'])
                with np.load(output/entry['npz'], allow_pickle=False) as arrays:
                    self.assertTrue(all(not arrays[key].dtype.hasobject for key in arrays.files))
                    np.testing.assert_array_equal(arrays['scene__native_values'], scene['native_values'])
                    for name, method in metadata['methods'].items():
                        prefix = method['prefix']
                        prediction, valid, truth, truth_valid, observed = [arrays[prefix+key]
                            for key in ('aligned_prediction', 'aligned_prediction_valid',
                                        'aligned_truth', 'aligned_truth_valid', 'aligned_observed')]
                        target_ids = arrays[prefix+'target_ids']
                        for row, identity in enumerate(target_ids):
                            np.testing.assert_array_equal(truth[row], scene['fusion_targets'][identity]['states'])
                            np.testing.assert_array_equal(truth_valid[row], scene['fusion_targets'][identity]['valid'])
                        for region, mask in (('all', np.ones_like(observed)), ('observed', observed),
                                             ('missing', ~observed)):
                            recovered = evaluate_aligned_fusion(prediction, truth, truth_valid,
                                                                prediction_valid=valid, region_mask=mask)
                            self.assert_same(recovered, actual['per_scene'][scene_index]['methods'][name]
                                             ['full_target_metrics'][region])
                        if name == 'predicted':
                            # All-reject leaves four singletons for two true identities.
                            self.assertEqual(len(prediction), 4)
                            self.assertFalse(truth_valid[2:].any())
                            self.assertEqual(arrays[prefix+'aligned_target_ids'][2:].tolist(), ['', ''])

    def test_phase_wall_time_includes_sampling_checkpoint_selection_and_writes(self):
        from association.cooperative_training import _batch
        from association.metrics import select_threshold
        clock = [0.]
        save, write = torch.save, runtime.write_json
        def sample(*args, **kwargs):
            result = _batch(*args, **kwargs); clock[0] += 2.; return result
        def checkpoint(*args, **kwargs):
            result = save(*args, **kwargs); clock[0] += 3.; return result
        def selection(*args, **kwargs):
            result = select_threshold(*args, **kwargs); clock[0] += 4.; return result
        def writing(*args, **kwargs):
            result = write(*args, **kwargs); clock[0] += .25; return result
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'timed'
            with patch('association.cooperative_training.time.monotonic', side_effect=lambda: clock[0]), \
                 patch('association.cooperative_training._batch', side_effect=sample), \
                 patch('association.cooperative_training.torch.save', side_effect=checkpoint), \
                 patch('association.metrics.select_threshold', side_effect=selection), \
                 patch('association.cooperative_training.runtime.write_json', side_effect=writing):
                result = self.fit(out, initial_elapsed_seconds=7.)
            self.assertEqual(result['phases']['external_preparation']['seconds'], 7.)
            for phase in ('teacher', 'utility', 'adapt:plain', 'adapt:utility'):
                self.assertGreater(result['phases'][phase]['seconds'], 5.)
                self.assertEqual(result['phases'][phase]['checkpoint_updates'], 1)
            self.assertGreater(result['phases']['identity_scoring_selection']['seconds'], 4.)
            self.assertGreater(result['phases']['evaluation:plain']['seconds'], .5)
            self.assertTrue(all(row['seconds'] == 0. for row in runtime.read_json(out/'history.json')))

    def test_prediction_write_counts_toward_budget_and_reports_unfinished_scene(self):
        from association.prediction_artifacts import ScenePredictionWriter
        normalizer = fit_normalizer(self.train)
        model = CooperativeModel(self.config, normalizer)
        written, events = [], []
        def check():
            if written:
                raise TimeoutError('budget exhausted during prediction write')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'predictions'
            writer = ScenePredictionWriter(output)
            def saving(*args):
                result = writer(*args)
                written.append(result)
                return result
            with self.assertRaisesRegex(TimeoutError, 'during prediction write'):
                evaluate_cooperative(model, normalizer, self.validation, device='cpu',
                    batch_tracks=2, pair_batch_size=3, select_on_validation=False, threshold=None,
                    prediction_writer=saving, check=check, progress=events.append)
            self.assertEqual(events[-1]['stage'], 'writing_predictions')
            self.assertEqual(events[-1]['scene_id'], 'val-one')
            self.assertEqual(events[-1]['completed_scenes'], 0)
            self.assertEqual(len(runtime.read_json(output/'index.json')['scenes']), 1)

    def test_evaluation_interruption_retains_route_scene_threshold_completed_work_and_elapsed(self):
        events = []
        def interrupt(item):
            events.append(item)
            if (item['phase'] == 'evaluation' and item['route'] == 'utility'
                    and item['stage'] == 'physical_fusion' and item['scene_id'] == 'val-two'
                    and item['completed_scenes'] == 1):
                raise InterruptedError('injected evaluation stop')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'interrupted'
            with self.assertRaisesRegex(InterruptedError, 'injected evaluation stop'):
                self.fit(out, progress=interrupt)
            run = runtime.read_json(out/'run.json')
            progress = runtime.read_json(out/'progress.json')
            self.assertEqual(run['status'], 'INTERRUPTED')
            self.assertEqual(run['completed_routes'], ['plain'])
            self.assertEqual(run['completed_updates'], 4)
            self.assertEqual(run['phases']['evaluation:utility']['status'], 'INTERRUPTED')
            self.assertGreater(run['phases']['evaluation:utility']['seconds'], 0.)
            for key, value in dict(phase='evaluation', route='utility', stage='physical_fusion',
                                   scene_id='val-two', completed_scenes=1).items():
                self.assertEqual(run['last_progress'][key], value)
                self.assertEqual(progress[key], value)
            self.assertEqual(progress['status'], 'INTERRUPTED')
            self.assertGreater(run['elapsed_seconds'], 0.)
            self.assertEqual(run['threshold'], run['identity_selection']['threshold'])
            self.assertEqual(progress['identity_selection'], run['identity_selection'])
            self.assertEqual(progress['threshold'], run['threshold'])
            self.assertTrue((out/'plain-evaluation.json').exists())
            self.assertFalse((out/'utility-evaluation.json').exists())
            self.assertTrue((out/'adapt-utility.pt').exists())

    def test_terminal_log_failure_preserves_original_exception_and_other_terminal_file(self):
        write = runtime.write_json
        for failed in ('progress.json', 'run.json'):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)/'failed'
                original = ValueError('original training problem')
                def writing(path, payload):
                    if Path(path).name == failed and payload.get('status') == 'FAILED':
                        raise OSError('secondary evidence storage problem')
                    return write(path, payload)
                with patch('association.cooperative_training.train_batch', side_effect=original), \
                     patch('association.cooperative_training.runtime.write_json', side_effect=writing):
                    with self.assertRaises(ValueError) as raised:
                        self.fit(out)
                self.assertIs(raised.exception, original)
                retained = runtime.read_json(out/('run.json' if failed == 'progress.json' else 'progress.json'))
                self.assertEqual(retained['status'], 'FAILED')
                self.assertIn('original training problem', retained['error'])
                self.assertTrue(any('secondary evidence storage problem' in note
                                    for note in getattr(original, '__notes__', [])))
                if failed == 'progress.json':
                    self.assertIn('secondary evidence storage problem', retained['evidence_write_errors'][0])


if __name__ == '__main__':
    unittest.main()
