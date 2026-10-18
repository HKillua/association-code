from tests.fixtures.scenes import scene, empty_scene, empty_timed_scene
"""Actual-input strata preserve all rows and never repair wrong associations."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from association.evaluation.scene import evaluate_full_scene, INPUT_SUPPORT_REGIONS
from association.evaluation.support import build_input_support
from association.evaluation.tmtaff import aggregate_fusion_reports
from association.evaluation.artifacts import ScenePredictionWriter, load_scene_prediction
from association.evaluation.physical import evaluate_aligned_fusion
from tests.fixtures import models as complete_fixtures
from tests.fixtures import models as cooperative_fixtures


class InputSupportEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def fixture(self):
        q = np.array([0., 10., 20., 500., 700., 900., 1400., 2000., 3000.])
        times = np.array([[0., 20.], [20., 600.], [700., 900.]])
        prepared = build_input_support(q, times, np.ones_like(times, dtype=bool))
        truth = np.zeros((3, len(q), 4))
        # Wrongly attach B to A; A's other fragment remains in another group.
        outputs = [dict(node_indices=[0, 2], states=truth[0].copy()),
                   dict(node_indices=[1], states=truth[0].copy())]
        kwargs = dict(node_identities=['A', 'A', 'B'], target_identities=['A', 'B', 'C'],
            target_states=truth, target_valid=np.ones_like(truth, dtype=bool),
            node_observed=prepared['support'], split='V_select', grouping_kind='predicted',
            correspondence_policy='anchor_identity_first')
        return outputs, kwargs, prepared

    def test_wrong_groups_keep_actual_helpers_and_do_not_borrow_true_identity(self):
        outputs, kwargs, prepared = self.fixture()
        legacy = evaluate_full_scene(outputs, **kwargs, return_artifact=True)
        report = evaluate_full_scene(outputs, **kwargs, node_input_support=prepared, return_artifact=True)
        self.assertEqual(report['full_target_metrics'], legacy['full_target_metrics'])
        self.assertEqual(report['selected_group_by_target'], [0, None, None])
        artifact = report['artifact']
        # At t=500 another true-A fragment exists, but not in predicted group 0.
        self.assertEqual(artifact['aligned_input_support_kind'][0, 3], 2)
        # At t=700 the actually attached wrong-B fragment is a helper.
        self.assertEqual(artifact['aligned_input_support_kind'][0, 4], 1)
        self.assertTrue((artifact['aligned_input_support_kind'][1:3] == 3).all())
        self.assertEqual(artifact['aligned_input_support_kind'][3, 3], 0)
        self.assertEqual(artifact['aligned_nearest_input_seconds'][0, 3], 200)
        self.assertEqual(artifact['aligned_nearest_input_seconds'][0, 1], 10)
        self.assertTrue(np.isinf(artifact['aligned_nearest_input_seconds'][1:3]).all())
        # Every point remains, including missing targets and duplicated outputs.
        for key in ('input_support_metrics', 'input_distance_metrics'):
            rows = report[key].values()
            self.assertEqual(sum(x['region_point_count'] for x in rows), 4 * 9)
            for feature in ('position', 'speed', 'heading'):
                for count in ('available_target_count', 'intersection_count', 'missing_prediction_count',
                              'available_prediction_count', 'prediction_without_target_count'):
                    self.assertEqual(sum(x[feature][count] for x in rows),
                                     report['full_target_metrics']['all'][feature][count])
        self.assertEqual(report['input_support_metrics']['unmatched_target']['position']['missing_prediction_count'], 18)
        self.assertIsNone(report['input_support_metrics']['unmatched_target']['position']['mae'])
        self.assertNotIn('input_support_metrics', legacy)
        for key in legacy['artifact']:
            np.testing.assert_equal(artifact[key], legacy['artifact'][key])
        report.pop('artifact')
        json.dumps(report, allow_nan=False)

    def test_partial_reference_and_pooled_counts_preserve_primary_results(self):
        outputs, kwargs, prepared = self.fixture()
        kwargs['target_valid'][0, 4, 3] = False
        kwargs['target_states'][0, 4, 3] = np.nan
        report = evaluate_full_scene(outputs, **kwargs, node_input_support=prepared)
        original = evaluate_full_scene(outputs, **kwargs)
        pooled = aggregate_fusion_reports([report, report])
        self.assertEqual({k: pooled[k] for k in ('all', 'observed', 'missing')},
                         aggregate_fusion_reports([original, original]))
        for name in INPUT_SUPPORT_REGIONS:
            cell = pooled['input_support_metrics'][name]['heading']
            self.assertEqual(cell['intersection_count'],
                             2 * report['input_support_metrics'][name]['heading']['intersection_count'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            aggregate_fusion_reports([report, original])

    def test_both_evaluators_export_exact_strata(self):
        fixtures = (complete_fixtures.CompleteEvaluationFixture(), cooperative_fixtures.CooperativeEvaluationFixture())
        for fixture in fixtures:
            with self.subTest(system=type(fixture).__name__), tempfile.TemporaryDirectory() as directory:
                fixture.setUp()
                if isinstance(fixture, complete_fixtures.CompleteEvaluationFixture):
                    scenes = [scene(include_missing_target=True), empty_scene(with_target=True)]
                else:
                    scenes = [fixture.scene, empty_timed_scene(with_target=True)]
                writer = ScenePredictionWriter(Path(directory)/'predictions')
                actual = fixture.evaluate(scenes, prediction_writer=writer)
                self.assertIn('input_support_metrics', actual['fusion']['predicted'])
                empty_report = actual['per_scene'][1]['methods']['predicted']
                self.assertEqual(empty_report['input_support_metrics']['unmatched_target']['position']['available_target_count'], 4)
                for method in ('predicted', 'oracle'):
                    saved = load_scene_prediction(writer.directory/'000000.json')['methods'][method]
                    report = actual['per_scene'][0]['methods'][method]
                    for index, name in enumerate(INPUT_SUPPORT_REGIONS):
                        metrics = evaluate_aligned_fusion(saved['aligned_prediction'], saved['aligned_truth'],
                            saved['aligned_truth_valid'], prediction_valid=saved['aligned_prediction_valid'],
                            region_mask=saved['aligned_input_support_kind'] == index)
                        self.assertEqual(metrics, report['input_support_metrics'][name])
                # Simple/single-source diagnostics retain their distinct input policy.
                self.assertNotIn('input_support_metrics', actual['fusion']['simple_fusion_oracle'])

    def test_loader_rejects_corrupt_optional_support_without_breaking_legacy(self):
        outputs, kwargs, prepared = self.fixture()
        artifact = evaluate_full_scene(outputs, **kwargs, node_input_support=prepared,
                                       return_artifact=True)['artifact']
        q = np.array([0., 10., 20., 500., 700., 900., 1400., 2000., 3000.])
        example = dict(scene_id='fixture', component_id='component', split='V_select',
            grid=q, values=np.zeros((3, len(q), 4)), observed=prepared['support'],
            pairs=np.array([[0, 1], [0, 2], [1, 2]]), labels=np.array([1, 0, 0]),
            intervals=np.array([[0, 20], [20, 600], [700, 900]]),
            node_ids=['0', '1', '2'], identities=['A', 'A', 'B'], sources=['S1', 'S2', 'S1'])
        with tempfile.TemporaryDirectory() as directory:
            writer = ScenePredictionWriter(Path(directory)/'predictions')
            writer(0, example, {'predicted': artifact})
            path = writer.directory/'000000.npz'
            with np.load(path, allow_pickle=False) as values:
                arrays = {key: values[key] for key in values.files}
            arrays['m000__aligned_nearest_input_seconds'][0, 0] = np.nan
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, 'input-support'):
                load_scene_prediction(writer.directory/'000000.json')


if __name__ == '__main__':
    unittest.main()
