"""Distinct point, target and scene weightings must remain explicit."""
from copy import deepcopy
import unittest
import numpy as np
from association.evaluation.support_gate_objectives import METHODS, REGIONS, summarize

class PositionWeightingTests(unittest.TestCase):
    def test_unequal_points_targets_scenes_have_explicitly_different_weights(self):
        def row(sid, targets, values):
            regions = {name: np.zeros(values.shape, bool) for name in REGIONS}
            regions['all'][:] = True
            regions['anchor_inside_support'][:] = True
            return dict(scene_id=sid, target_ids=np.array(targets),
                errors={method: values*(.5 if method == 'final' else 1.) for method in METHODS},
                regions=regions, population=dict(target_identity_count=len(targets)), model_state_comparison={})
        values = np.zeros((2, 9))
        values[0] = np.nan
        values[0, 0] = 10000.
        rows = [row('a', ['shared', 'other'], values), row('b', ['shared'], np.array([[100.]]))]
        result = summarize(rows, {'a': 'same_component', 'b': 'same_component'})
        region = result['regions']['all']
        self.assertEqual(result['scene_target_record_count'], 3)
        self.assertEqual(result['distinct_identity_string_count'], 2)
        self.assertEqual(result['component_count'], 1)
        self.assertEqual(region['point_count'], 11)
        self.assertEqual(region['scene_target_record_count'], 3)
        self.assertEqual(region['nonempty_scene_count'], 2)
        metrics = region['methods']['initial']
        self.assertAlmostEqual(metrics['point_equal']['mae'], 10100./11)
        self.assertAlmostEqual(metrics['scene_target_equal']['mae'], 10100./3)
        self.assertEqual(metrics['scene_equal_point_mean']['mae'], 550.)
        self.assertEqual(metrics['scene_equal_target_mean']['mae'], 2550.)
        self.assertAlmostEqual(metrics['point_equal']['log1p'], (np.log1p(10.)+np.log1p(.1))/11)
        self.assertAlmostEqual(metrics['scene_equal_point_mean']['rmse'], np.sqrt((10000.**2/10+100.**2)/2))
        paired = region['paired']['final__minus__initial']['point_equal']
        self.assertEqual(paired['unit_count'], 11)
        self.assertEqual(paired['mae']['negative_unit_count'], 2)
        self.assertEqual(paired['mae']['exact_tie_unit_count'], 9)
        self.assertAlmostEqual(paired['mae']['mean_difference'], -5050./11)
        wrong = deepcopy(rows)
        wrong[0]['errors']['final'][1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'paired masks'):
            summarize(wrong, {'a': 'shared', 'b': 'shared'})


