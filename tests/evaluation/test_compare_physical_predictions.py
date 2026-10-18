"""Common support is diagnostic; original missing targets remain counted."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from association.evaluation.scene import evaluate_full_scene
from association.evaluation.artifacts import ScenePredictionWriter
from tools.analysis.compare_physical_predictions import compare


class PhysicalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.truth = np.broadcast_to([30., 120., 5., 40.], (2, 3, 4)).copy()

    def write(self, name, *, states=None, valid=None, truth=None, scene_change=None,
              nodes=('A', 'B'), group_outputs=None, route=None, status='COMPLETED'):
        run = self.root/name
        directory = run/('predictions' if route is None else route+'-predictions')
        q, n = 3, len(nodes)
        pairs = np.column_stack(np.triu_indices(n, 1))
        observed = np.broadcast_to([True, False, False], (n, q)).copy()
        scene = dict(scene_id='scene', component_id='component', split='V_select',
            input_adapter='native_time_v1', node_ids=np.array(['n'+str(i) for i in range(n)]),
            identities=np.array(nodes), sources=np.array(['9001']*n), grid=np.arange(q, dtype=float),
            pairs=pairs, labels=np.array([nodes[a] == nodes[b] for a, b in pairs]),
            intervals=np.tile([0., 1.], (n, 1)), observed=observed,
            values=np.broadcast_to(self.truth[0], (n, q, 4)).copy(),
            native_values=np.broadcast_to(self.truth[0, :2], (n, 2, 4)).copy(),
            native_times=np.tile([0., 1.], (n, 1)), native_observed=np.ones((n, 2), bool))
        if scene_change:
            scene.update(scene_change)
        target = self.truth if truth is None else truth
        prediction = target if states is None else states
        mask = np.ones_like(prediction, bool) if valid is None else valid
        outputs = group_outputs or [dict(node_indices=[i], states=prediction[i], prediction_valid=mask[i]) for i in range(n)]
        artifact = evaluate_full_scene(outputs, node_identities=nodes, target_identities=['A', 'B'],
            target_states=target, target_valid=np.ones_like(target, bool), node_observed=observed,
            split='V_select', grouping_kind='predicted', correspondence_policy='anchor_identity_first',
            return_artifact=True)['artifact']
        ScenePredictionWriter(directory)(0, scene, {'predicted': artifact})
        (run/'run.json').write_text(json.dumps(dict(status=status, test_opened=False, completed_routes=[route] if route else [])))
        evaluation = dict(per_scene=[{'scene_id': 'scene'}])
        report = evaluation if route else {'evaluation': evaluation}
        (run/(route+'-evaluation.json' if route else 'summary.json')).write_text(json.dumps(report))
        return directory

    def test_identical_predictions_keep_all_population_and_refuse_overwrite(self):
        left, right = self.write('left'), self.write('right', route='utility')
        result = compare(left, right, self.root/'out')
        self.assertEqual(result['target_count'], 2)
        self.assertEqual(result['common']['left']['all']['position']['intersection_count'], 6)
        self.assertEqual(result['common']['right']['all']['position']['rmse'], 0)
        self.assertEqual(result['improved'], [])
        self.assertEqual(result['worsened'], [])
        self.assertTrue((self.root/'out/comparison.json').is_file())
        with self.assertRaises(FileExistsError):
            compare(left, right, self.root/'out')

    def test_different_coverage_uses_common_points_without_hiding_original_misses(self):
        valid = np.ones_like(self.truth, bool)
        valid[:, 1:] = False
        left_states = self.truth.copy(); left_states[..., 0] += .002
        right_states = self.truth.copy(); right_states[0, :, 0] += .001
        right_states[1, :, 0] += .003
        result = compare(self.write('left', states=left_states, valid=valid),
                         self.write('right', states=right_states), self.root/'out')
        self.assertEqual(result['original']['left']['all']['position']['missing_prediction_count'], 4)
        self.assertEqual(result['original']['right']['all']['position']['missing_prediction_count'], 0)
        self.assertEqual(result['common']['left']['all']['position']['intersection_count'], 2)
        self.assertEqual(result['common']['left']['all']['position']['coverage'], 1/3)
        self.assertIsNone(result['common']['right']['missing']['position']['rmse'])
        self.assertEqual([x['target_id'] for x in result['improved']], ['A'])
        self.assertEqual([x['target_id'] for x in result['worsened']], ['B'])
        self.assertTrue(Path(result['improved'][0]['left_metadata']).is_file())

    def test_zero_common_points_produce_null_errors_and_keep_denominator(self):
        left_mask = np.zeros_like(self.truth, bool); left_mask[:, 0] = True
        right_mask = np.zeros_like(self.truth, bool); right_mask[:, 1] = True
        result = compare(self.write('left', valid=left_mask), self.write('right', valid=right_mask), self.root/'out')
        metric = result['common']['left']['all']['position']
        self.assertEqual(metric['available_target_count'], 6)
        self.assertEqual(metric['intersection_count'], 0)
        self.assertIsNone(metric['rmse'])
        self.assertEqual(result['improved'], [])

    def test_common_masks_remain_feature_specific(self):
        left_mask = np.ones_like(self.truth, bool); left_mask[0, :, 2] = False
        right_mask = np.ones_like(self.truth, bool); right_mask[1, :, 3] = False
        result = compare(self.write('left', valid=left_mask), self.write('right', valid=right_mask), self.root/'out')
        metrics = result['common']['left']['all']
        self.assertEqual(metrics['position']['intersection_count'], 6)
        self.assertEqual(metrics['speed']['intersection_count'], 3)
        self.assertEqual(metrics['heading']['intersection_count'], 3)

    def test_different_grouping_retains_duplicate_predictions_in_original_report(self):
        nodes = ('A', 'A', 'B')
        separated = [dict(node_indices=[i], states=self.truth[0 if i < 2 else 1]) for i in range(3)]
        grouped = [dict(node_indices=[0, 1], states=self.truth[0]), dict(node_indices=[2], states=self.truth[1])]
        result = compare(self.write('left', nodes=nodes, group_outputs=separated),
                         self.write('right', nodes=nodes, group_outputs=grouped), self.root/'out')
        self.assertEqual(result['original']['left']['all']['position']['prediction_without_target_count'], 3)
        self.assertEqual(result['common']['left']['all']['position']['prediction_without_target_count'], 0)
        self.assertEqual(result['target_count'], 2)

    def test_population_truth_and_input_mismatch_are_rejected(self):
        left = self.write('left')
        changed = self.truth.copy(); changed[0, 0, 0] += .01
        variants = [dict(truth=changed), dict(scene_change={'node_ids': np.array(['wrong', 'n1'])}),
                    dict(scene_change={'native_times': np.tile([0., 2.], (2, 1))})]
        for i, change in enumerate(variants):
            with self.subTest(change=i), self.assertRaises(ValueError):
                compare(left, self.write('right'+str(i), **change), self.root/('out'+str(i)))

    def test_incomplete_run_or_partial_index_rejected(self):
        left, right = self.write('left'), self.write('right', status='INTERRUPTED')
        with self.assertRaises(ValueError):
            compare(left, right, self.root/'out')
        (right.parent/'run.json').write_text(json.dumps(dict(status='COMPLETED', test_opened=False)))
        index = json.loads((right/'index.json').read_text()); index['scenes'] = []
        (right/'index.json').write_text(json.dumps(index))
        with self.assertRaises(ValueError):
            compare(left, right, self.root/'out')


if __name__ == '__main__':
    unittest.main()
