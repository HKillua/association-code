"""Fixed representatives, complete target registry and repeated-target integrity."""
import copy
import unittest

import numpy as np

from association.evaluation.frozen_group_view import unique_target_view
from association.evaluation.tmtaff import aggregate_fusion_reports


def fixture():
    target = np.zeros((3, 2, 4))
    target[:, :, 2] = np.array([1., 1., 10.])[:, None]
    valid = np.ones_like(target, bool)
    target[2, 0, 3] = np.nan
    valid[2, 0, 3] = False
    observed = np.array([[True, False], [False, True], [False, True]])
    scene = dict(split='V_select', node_ids=['n0', 'n1', 'n2'], identities=['A', 'A', 'B'],
        target_identity_ids=['B', 'A'], observed=observed,
        motion=dict(query_seconds=np.array([0., 1.]), support=observed.copy(),
                    nearest_seconds=np.ones((3, 2))))
    saved = dict(node_ids=scene['node_ids'], identities=scene['identities'],
                 query_seconds=scene['motion']['query_seconds'], target=target, target_valid=valid)
    prediction = np.zeros_like(target)
    prediction[:, :, 2] = np.array([9., 1., 10.])[:, None]
    partitions = {'ols046': [[0, 2], [1]]}
    predictions = {'ols046': {'learned': prediction}}
    return scene, saved, partitions, predictions


class FrozenGroupViewTests(unittest.TestCase):
    def test_fixed_first_output_retains_missed_target_duplicate_and_attribute_masks(self):
        scene, saved, groups, predictions = fixture()
        report = unique_target_view(scene, saved, groups, predictions,
                                    return_artifact=True)['ols046']['learned']
        self.assertEqual(report['target_identity_count'], 2)
        self.assertEqual(report['unmatched_target_ids'], ['B'])
        self.assertEqual(report['duplicate_prediction_group_count'], 1)
        self.assertEqual(report['selected_group_by_target'], [None, 0])
        metric = report['full_target_metrics']['all']['speed']
        self.assertEqual(metric['available_target_count'], 4)
        self.assertEqual(metric['intersection_count'], 2)
        self.assertEqual(metric['prediction_without_target_count'], 2)
        self.assertEqual(metric['mae'], 8.)  # Neither closer identity B nor better duplicate n1 is selected.
        self.assertEqual(report['full_target_metrics']['all']['heading']['available_target_count'], 3)
        self.assertEqual(report['full_target_metrics']['observed']['speed']['available_target_count'], 3)
        self.assertEqual(report['full_target_metrics']['missing']['speed']['available_target_count'], 1)
        np.testing.assert_array_equal(report['artifact']['target_ids'], ['B', 'A'])
        np.testing.assert_array_equal(report['artifact']['group_states'][0], predictions['ols046']['learned'][0])
        pooled = aggregate_fusion_reports([report])
        self.assertEqual(pooled['all']['speed']['coverage'], .5)
        self.assertEqual(pooled['all']['speed']['mae'], 8.)

    def test_incomplete_registry_and_disagreeing_repeated_targets_cannot_silently_change_population(self):
        scene, saved, groups, predictions = fixture()
        missing = copy.deepcopy(scene)
        missing['target_identity_ids'].append('C')
        with self.assertRaisesRegex(ValueError, 'cannot be reconstructed'):
            unique_target_view(missing, saved, groups, predictions)
        for key in ('target', 'target_valid'):
            bad = copy.deepcopy(saved)
            bad[key][1, 0, 2] = 20. if key == 'target' else False
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'inconsistent repeated'):
                unique_target_view(scene, bad, groups, predictions)


if __name__ == '__main__':
    unittest.main()
