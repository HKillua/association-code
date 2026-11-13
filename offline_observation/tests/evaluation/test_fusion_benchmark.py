import copy
import unittest

import numpy as np

from association.data.physical_units import KNOT_TO_MPS
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.tmtaff import aggregate_fusion_reports


class FusionBenchmarkTests(unittest.TestCase):
    def fixture(self):
        truth = np.broadcast_to([10., 20., 10*KNOT_TO_MPS, 1.], (3, 2, 4)).copy()
        states = np.broadcast_to([10., 20., 10., 359.], (2, 4)).copy()
        outputs = [dict(node_indices=[0], states=states), dict(node_indices=[1], states=states),
                   dict(node_indices=[2], states=None)]
        return outputs, dict(node_identities=np.array(['A', 'A', 'C']),
            target_ids=np.array(['A', 'B', 'C']), target_states=truth,
            target_valid=np.ones_like(truth, bool), support=np.ones((3, 2), bool),
            nearest_seconds=np.zeros((3, 2)), target_speed_unit='mps')

    def test_units_misses_duplicates_and_circular_heading_without_mutation(self):
        outputs, kwargs = self.fixture()
        before = copy.deepcopy(outputs)
        mps = copy.deepcopy(outputs)
        for row in mps:
            if row['states'] is not None:
                row['states'] = row['states'].copy()
                row['states'][:, 2] *= KNOT_TO_MPS
        reports, artifacts = evaluate_fusion_methods({'kn': outputs, 'mps': mps},
            prediction_speed_units={'kn': 'kn', 'mps': 'mps'}, **kwargs)
        self.assertEqual(reports['kn']['full_target_metrics'], reports['mps']['full_target_metrics'])
        metric = aggregate_fusion_reports([reports['kn']])['all']
        self.assertEqual(metric['speed']['mae'], 0.)
        self.assertEqual(metric['speed']['unit'], 'mps')
        self.assertAlmostEqual(metric['heading']['mae'], 2.)
        self.assertEqual(metric['position']['available_target_count'], 6)
        self.assertEqual(metric['position']['intersection_count'], 2)
        self.assertEqual(metric['position']['missing_prediction_count'], 4)
        self.assertEqual(len(artifacts['kn']['group_states']), 3)
        self.assertFalse(artifacts['kn']['aligned_prediction_valid'][1:3].any())
        np.testing.assert_array_equal(outputs[0]['states'], before[0]['states'])

    def test_methods_keep_their_own_missing_mask_and_require_units(self):
        outputs, kwargs = self.fixture()
        complete = copy.deepcopy(outputs)
        complete[2]['states'] = outputs[0]['states'].copy()
        reports, _ = evaluate_fusion_methods({'partial': outputs, 'complete': complete},
            prediction_speed_units={'partial': 'kn', 'complete': 'kn'}, **kwargs)
        self.assertEqual(reports['partial']['full_target_metrics']['all']['position']['intersection_count'], 2)
        self.assertEqual(reports['complete']['full_target_metrics']['all']['position']['intersection_count'], 4)
        with self.assertRaises(ValueError):
            evaluate_fusion_methods({'a': outputs}, prediction_speed_units={}, **kwargs)
        with self.assertRaises(ValueError):
            evaluate_fusion_methods({'a': outputs}, prediction_speed_units={'a': 'native'}, **kwargs)
        with self.assertRaises(ValueError):
            evaluate_fusion_methods({'a': outputs}, prediction_speed_units={'a': 'kn'}, **dict(kwargs, split='T_test'))
