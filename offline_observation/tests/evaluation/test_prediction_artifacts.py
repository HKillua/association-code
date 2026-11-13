"""Saved physical outputs reproduce actual metrics, without changing inference."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from association.utils import runtime
from association.evaluation.scene import evaluate_full_scene, _regions
from association.evaluation.artifacts import ScenePredictionWriter, load_scene_prediction
from tests.fixtures import models as evaluation_fixtures
from tests.fixtures.scenes import scene, empty_scene


from tests.fixtures.scenes import artifact_for

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
        fixture = evaluation_fixtures.CompleteEvaluationFixture()
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


    def test_export_failure_and_deadline_do_not_report_evaluation_completed(self):
        fixture = evaluation_fixtures.CompleteEvaluationFixture()
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


if __name__ == '__main__':
    unittest.main()
