import unittest

import numpy as np

from tools.analysis import association_position_smoothing as study


class PositionSmoothingTest(unittest.TestCase):
    def test_irregular_motion_dateline_and_no_update_across_long_gap(self):
        t = np.array([0., 10., 30., 91., 100.])
        xy = np.column_stack((.01 * t, (179.8 + .02 * t + 180.) % 360. - 180.))
        original = xy.copy()
        result, eligible = study.smooth_positions(t, xy)
        self.assertEqual(eligible, 1)
        np.testing.assert_allclose(result, xy, rtol=0., atol=1e-12)
        xy[1, 0] += 1.
        result, _ = study.smooth_positions(t, xy)
        self.assertAlmostEqual(result[1, 0], original[1, 0] + .5)
        np.testing.assert_array_equal(result[[0, 2, 3, 4]], original[[0, 2, 3, 4]])
        np.testing.assert_array_equal(study.smooth_positions(t[:2], xy[:2])[0], xy[:2])

    def test_old_graph_previous_quota_and_ties(self):
        # The former winner gets more expensive than the original cutoff.
        # A new near-zero cost on an originally forbidden edge must be ignored.
        sources = ['9001', '9002', '9002', '9002']
        edges = np.array([[0, 1], [0, 2], [0, 3]])
        old = np.array([5., 20., 6000.])
        original = np.array([[0, 1]])
        np.testing.assert_array_equal(study.matched_links(sources, edges, old,
                                      np.array([7000., 6500., 0.]), original), [[0, 2]])
        np.testing.assert_array_equal(study.matched_links(sources, edges, old,
                                      np.array([7000., 7000., 0.]), original), original)

    def test_expanded_mixed_chain_ties_in_roc_and_pr(self):
        d=np.array([1., 1., 6000.])
        y,n=np.array([1,0,1]),np.array([0,2,1])
        stats,curve=study.ranking(d,y,n)
        np.testing.assert_array_equal(curve['true_positive'], [1,2])
        np.testing.assert_array_equal(curve['false_positive'], [2,3])
        self.assertAlmostEqual(stats['roc_auc'], 5/12)
        self.assertAlmostEqual(stats['average_precision'], (1/3+2/5)/2)
        self.assertEqual(stats['at_5000']['tp'], 1)
        self.assertEqual(stats['at_recall']['0.99']['fp'], 3)
        self.assertIsNone(study.ranking(np.empty(0),np.empty(0,int),np.empty(0,int))[0]['average_precision'])

    def test_both_sources_and_missing_support_follow_actual_groups(self):
        saved = dict(sources=np.array(['9001','9001','9002']),
            support=np.array([[0,1],[1,1],[1,1]],bool), chain_sources=np.array(['9001','9001','9002']),
            chain_nodes=np.arange(3),chain_offsets=np.arange(4), RAW_membership=np.array([0,1,0]),
            SMOOTH_membership=np.array([0,1,1]), candidate_pairs=np.array([[0,2],[1,2]]),
            RAW_cost=np.array([0.,2.]), SMOOTH_cost=np.array([2.,0.]))
        row, ev=study.evaluate(saved,np.array(['a','b','a']))
        self.assertEqual(row['supply']['same_ship_missing_lost'],1)
        self.assertEqual(row['arms']['SMOOTH']['MSTSA']['fp'],1)
        np.testing.assert_array_equal(ev['positive'], [1,0])

    def test_costs_use_only_joint_support(self):
        support=np.array([[1,1,0],[1,1,0]],bool)
        pos=np.zeros((2,3,2));pos[0,2,0]=80.
        costs=study.score_existing_pairs(np.array([[0,1]]),support,pos)
        np.testing.assert_array_equal(costs,[0.])
