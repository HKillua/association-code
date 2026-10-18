"""Saved physical outputs reproduce actual metrics, without changing inference."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association import runtime
from association.full_scene_evaluation import evaluate_full_scene, _regions
from association.prediction_artifacts import ScenePredictionWriter, load_scene_prediction
from tests import test_complete_evaluation as evaluation_fixtures
from tests import test_timed_dataset as dataset_fixtures
from tests.test_complete_evaluation import scene, empty_scene
from tools.experiments import tmtaff_complete_fit


def artifact_for(example):
    identities = list(example['fusion_targets'])
    targets = example['fusion_targets']
    # Split identity A and leave identity C without any input/prediction. This
    # exercises duplicated predictions and missing targets simultaneously.
    outputs = [dict(node_indices=[0], states=example['values'][0],
                    prediction_valid=np.repeat(example['observed'][0, :, None], 4, 1)),
               dict(node_indices=[1], states=None),
               dict(node_indices=[2, 3], states=targets['B']['states'])]
    kwargs = dict(node_identities=example['identities'], target_identities=identities,
        target_states=np.stack([targets[i]['states'] for i in identities]),
        target_valid=np.stack([targets[i]['valid'] for i in identities]),
        node_observed=example['observed'], split=example['split'], grouping_kind='predicted',
        correspondence_policy='anchor_identity_first')
    return outputs, kwargs


class PredictionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_artifact_retains_exact_metric_arrays_and_group_correspondence(self):
        example = scene(include_missing_target=True)
        outputs, kwargs = artifact_for(example)
        baseline = evaluate_full_scene(outputs, **kwargs)
        exported = evaluate_full_scene(outputs, **kwargs, return_artifact=True)
        artifact = exported.pop('artifact')
        self.assertEqual(exported, baseline)
        self.assertEqual(artifact['aligned_prediction'].shape, (4, 4, 4))
        self.assertEqual(artifact['aligned_target_ids'].tolist(), ['A', 'B', 'C', ''])
        self.assertEqual(artifact['aligned_group_indices'].tolist(), [0, 2, -1, 1])
        self.assertEqual(artifact['group_anchor_target_indices'].tolist(), [0, 0, 1])
        self.assertEqual(artifact['group_offsets'].tolist(), [0, 1, 2, 4])
        self.assertEqual(artifact['group_node_indices'].tolist(), [0, 1, 2, 3])
        self.assertEqual(artifact['node_group_indices'].tolist(), [0, 1, 2, 2])
        self.assertFalse(artifact['aligned_prediction_valid'][2].any())
        self.assertTrue(artifact['aligned_truth_valid'][2].all())
        self.assertFalse(artifact['aligned_truth_valid'][3].any())
        self.assertEqual(self.metrics(artifact), baseline['full_target_metrics'])

    @staticmethod
    def metrics(artifact):
        return _regions(artifact['aligned_prediction'], artifact['aligned_truth'],
            artifact['aligned_truth_valid'], artifact['aligned_prediction_valid'], artifact['aligned_observed'])

    def test_reference_export_reload_including_empty_scenes_preserves_all_results(self):
        fixture = evaluation_fixtures.CompleteEvaluationTests()
        fixture.setUp()
        examples = [scene(include_missing_target=True), empty_scene('empty-target', with_target=True), empty_scene('empty')]
        expected = fixture.evaluate(examples)
        writer = ScenePredictionWriter(self.root/'predictions')
        actual = fixture.evaluate(examples, prediction_writer=writer)
        for key in ('association', 'fusion', 'per_scene', 'threshold', 'fusion_forward_count'):
            self.assertEqual(actual[key], expected[key])
        for left, right in zip(actual['records'], expected['records']):
            np.testing.assert_array_equal(left['scores'], right['scores'])
        index = runtime.read_json(writer.directory/'index.json')
        self.assertEqual(len(index['scenes']), 3)
        for entry, report in zip(index['scenes'], actual['per_scene']):
            saved = load_scene_prediction(writer.directory/entry['metadata'])
            self.assertEqual(saved['metadata']['scene_id'], report['scene_id'])
            self.assertEqual(set(saved['methods']), set(report['methods']))
            np.testing.assert_array_equal(saved['scene']['grid'], examples[entry['scene_index']]['grid'])
            for method, artifact in saved['methods'].items():
                self.assertEqual(self.metrics(artifact), report['methods'][method]['full_target_metrics'])
                self.assertNotIn('artifact', report['methods'][method])
            with np.load(writer.directory/entry['npz'], allow_pickle=False) as arrays:
                self.assertTrue(all(not arrays[key].dtype.hasobject for key in arrays.files))
        self.assertEqual(load_scene_prediction(writer.directory/'000002.json')['methods']['predicted']['aligned_truth'].shape,
                         (0, 4, 4))

    def test_writer_refuses_overwrite_and_object_arrays(self):
        example = scene(include_missing_target=True)
        outputs, kwargs = artifact_for(example)
        artifact = evaluate_full_scene(outputs, **kwargs, return_artifact=True)['artifact']
        writer = ScenePredictionWriter(self.root/'predictions')
        writer(0, example, {'predicted': artifact})
        original = {p.name: p.read_bytes() for p in writer.directory.iterdir()}
        with self.assertRaises(FileExistsError):
            ScenePredictionWriter(writer.directory)
        with self.assertRaises(FileExistsError):
            writer(0, example, {'predicted': artifact})
        with self.assertRaises(FileExistsError):
            writer(1, example, {'predicted': artifact})
        self.assertEqual(original, {p.name: p.read_bytes() for p in writer.directory.iterdir()})
        broken = dict(artifact, extra=np.array([{'unsafe': True}], dtype=object))
        with self.assertRaisesRegex(ValueError, 'object/pickle'):
            writer(1, {**example, 'scene_id': 'new'}, {'predicted': broken})
        self.assertFalse((writer.directory/'000001.npz').exists())

    def test_loader_rejects_inconsistent_shape_metadata(self):
        example = scene(include_missing_target=True)
        outputs, kwargs = artifact_for(example)
        writer = ScenePredictionWriter(self.root/'predictions')
        writer(0, example, {'predicted': evaluate_full_scene(outputs, **kwargs, return_artifact=True)['artifact']})
        metadata_path = writer.directory/'000000.json'
        metadata = runtime.read_json(metadata_path)
        metadata['methods']['predicted']['target_count'] += 1
        runtime.write_json(metadata_path, metadata)
        with self.assertRaisesRegex(ValueError, 'metadata disagrees'):
            load_scene_prediction(metadata_path)

    def test_export_failure_and_deadline_do_not_report_evaluation_completed(self):
        fixture = evaluation_fixtures.CompleteEvaluationTests()
        fixture.setUp()
        events = []
        def fail(*args):
            raise OSError('prediction disk full')
        with self.assertRaisesRegex(OSError, 'prediction disk full'):
            fixture.evaluate([scene()], prediction_writer=fail, progress=events.append)
        self.assertNotIn('completed', [item['stage'] for item in events])
        written = [False]
        def writer(*args):
            written[0] = True
        def check():
            if written[0]:
                raise TimeoutError('export exhausted evaluation budget')
        with self.assertRaisesRegex(TimeoutError, 'export exhausted'):
            fixture.evaluate([scene()], prediction_writer=writer, check=check)

    def test_native_cli_explicit_export_and_default_disabled(self):
        fixture = dataset_fixtures.TimedDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for enabled in (False, True):
            name = 'saved' if enabled else 'default'
            args = fixture.cli_args(name) + ['--device', 'cpu'] + (['--save-predictions'] if enabled else [])
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(tmtaff_complete_fit.main(args), 0)
            summary = runtime.read_json(fixture.root/name/'summary.json')
            self.assertEqual((fixture.root/name/'predictions').exists(), enabled)
            if enabled:
                self.assertEqual(summary['prediction_artifacts'], 'predictions/index.json')
                saved = load_scene_prediction(fixture.root/name/'predictions'/'000000.json')
                self.assertIn('native_times', saved['scene'])
                for method, artifact in saved['methods'].items():
                    self.assertEqual(self.metrics(artifact), summary['evaluation']['per_scene'][0]['methods'][method]['full_target_metrics'])
            else:
                self.assertNotIn('prediction_artifacts', summary)
        baseline = torch.load(fixture.root/'default'/'checkpoint.pt', map_location='cpu', weights_only=True)
        exported = torch.load(fixture.root/'saved'/'checkpoint.pt', map_location='cpu', weights_only=True)
        for left, right in zip(baseline['states'], exported['states']):
            self.assertEqual(set(left), set(right))
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        args = fixture.cli_args('inspect') + ['--inspect-only', '--save-predictions']
        with redirect_stderr(io.StringIO()), patch('tools.experiments.tmtaff_complete_fit.load_timed_dataset') as load:
            self.assertEqual(tmtaff_complete_fit.main(args), 1)
            load.assert_not_called()

    def test_training_export_error_keeps_checkpoint_and_marks_failed(self):
        fixture = dataset_fixtures.TimedDatasetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        args = fixture.cli_args('failed-export') + ['--device', 'cpu', '--save-predictions']
        with patch.object(ScenePredictionWriter, '__call__', side_effect=OSError('artifact disk error')), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(tmtaff_complete_fit.main(args), 1)
        directory = fixture.root/'failed-export'
        run = runtime.read_json(directory/'run.json')
        self.assertEqual(run['status'], 'FAILED')
        self.assertEqual(run['final_evaluation_status'], 'FAILED')
        self.assertEqual(run['failure'], 'artifact disk error')
        self.assertTrue((directory/'checkpoint.pt').exists())
        self.assertFalse((directory/'summary.json').exists())
        self.assertEqual(runtime.read_json(directory/'predictions'/'index.json')['scenes'], [])


if __name__ == '__main__':
    unittest.main()
