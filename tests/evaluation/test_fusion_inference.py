"""Predicted partitions retain input order, threshold semantics and full population."""
import unittest

import numpy as np

from association.data.archive import DataError
from association.evaluation.grouping import group_predicted_tracks, group_tracks_by_identity


def predicted(node_count, pairs, scores, threshold=0.5):
    return group_predicted_tracks(node_count, pairs, scores, threshold=threshold,
                                  grouping_policy='anchor_direct', node_order='input')


class PredictedGroupTests(unittest.TestCase):
    def test_anchor_neighbors_do_not_form_a_transitive_closure(self):
        # 0-1-2 and 0-3-4 are accepted paths, but only direct 0 neighbors
        # enter its group. Nodes already taken by 0 cannot enter later groups.
        pairs = np.asarray([[0,1], [1,2], [0,3], [3,4], [2,5]])
        groups = predicted(6, pairs, [0.5, 0.9, 0.8, 0.9, 0.5])
        self.assertEqual(groups, [[0,1,3], [2,5], [4]])
        flattened = [index for group in groups for index in group]
        self.assertEqual(sorted(flattened), list(range(6)))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_threshold_ties_and_absent_edges_have_exact_meaning(self):
        self.assertEqual(predicted(4, np.asarray([[0,1], [0,2]]), [0.5, 0.499]),
                         [[0,1], [2], [3]])
        self.assertEqual(predicted(3, np.asarray([[0,1], [1,2]]), [-2., -1.], threshold=-2.),
                         [[0,1], [2]])

    def test_existing_all_reject_and_float64_threshold_contract(self):
        pairs = np.asarray([[0,1]])
        scores = np.asarray([0.5], dtype=np.float32)
        self.assertEqual(predicted(2, pairs, scores, threshold=None), [[0], [1]])
        # Do not round a float64 threshold down to the float32 score before
        # applying >=; this must match association.evaluation.association.predict exactly.
        just_above = float(np.nextafter(0.5, 1.0))
        self.assertEqual(predicted(2, pairs, scores, threshold=just_above), [[0], [1]])
        self.assertEqual(predicted(2, pairs, scores, threshold=0.5), [[0,1]])

    def test_candidate_order_and_orientation_do_not_change_node_order(self):
        pairs = np.asarray([[0,3], [0,1], [2,4], [1,2]])
        scores = np.asarray([0.8, 0.9, 0.7, 0.9])
        before_pairs, before_scores = pairs.copy(), scores.copy()
        expected = [[0,1,3], [2,4]]
        self.assertEqual(predicted(5, pairs, scores), expected)
        order = [3, 1, 0, 2]
        self.assertEqual(predicted(5, pairs[order, ::-1], scores[order]), expected)
        np.testing.assert_array_equal(pairs, before_pairs)
        np.testing.assert_array_equal(scores, before_scores)

    def test_empty_scene_and_all_singletons_are_retained(self):
        empty = np.empty((0,2), dtype=np.int64)
        self.assertEqual(predicted(0, empty, []), [])
        self.assertEqual(predicted(3, empty, []), [[0], [1], [2]])
        self.assertEqual(predicted(3, np.asarray([[0,1], [1,2]]), [0.1,0.2]), [[0], [1], [2]])


    def test_identity_groups_are_a_separate_oracle_comparison(self):
        identities = ['ship-0', 'other-0', 'ship-0', 'lonely-0']
        self.assertEqual(group_tracks_by_identity(identities, node_order='input'), [[0,2], [1], [3]])
        # Prediction has no identity argument and must retain this wrong merge.
        actual = predicted(4, np.asarray([[0,1], [0,2]]), [0.8,0.9])
        self.assertEqual(actual, [[0,1,2], [3]])
        identities[1] = 'ship-0'
        self.assertEqual(predicted(4, np.asarray([[0,1], [0,2]]), [0.8,0.9]), actual)
        self.assertEqual(group_tracks_by_identity([], node_order='input'), [])
        for invalid in ([None], [''], ['ship-0', 2], [['ship-0']]):
            with self.subTest(identities=invalid), self.assertRaises(DataError):
                group_tracks_by_identity(invalid, node_order='input')


if __name__ == '__main__':
    unittest.main()
