import math
import unittest

from tools.analysis.offline_baseline_assignment import assign, maximum_pair_weight


class OfflineBaselineAssignmentTests(unittest.TestCase):
    def test_fixed_quota_swaps_greedy_links_within_component(self):
        edges = [(0, 2, 1.0), (0, 3, 2.0), (1, 2, 2.0), (1, 3, 4.0)]
        arms, audit = assign(['a', 'a', 'b', 'b'], edges, [(0, 2), (1, 3)])
        self.assertEqual(arms['GLOBAL_FIXED'], [(0, 3), (1, 2)])
        self.assertEqual(arms['EXPLICIT_NULL'], arms['GLOBAL_FIXED'])
        self.assertEqual(audit[0]['greedy_count'], audit[0]['global_fixed_count'])
        self.assertEqual(audit[0]['greedy_cost_m'], 5.0)
        self.assertEqual(audit[0]['global_fixed_cost_m'], 4.0)

    def test_sparse_quota_is_local_and_explicit_null_can_add_links(self):
        edges = [(0, 3, 1000.0), (0, 4, 1500.0), (1, 3, 1500.0),
                 (2, 5, 5000.0), (1, 5, 5000.1)]
        arms, audit = assign(['a'] * 3 + ['b'] * 4, edges, [(0, 3), (2, 5)])
        self.assertEqual(arms['GLOBAL_FIXED'], [(0, 3), (2, 5)])
        self.assertEqual(arms['EXPLICIT_NULL'], [(0, 4), (1, 3)])
        self.assertEqual([(x['eligible_edges'], x['greedy_count'], x['explicit_null_count']) for x in audit],
                         [(3, 1, 2), (1, 1, 0)])

    def test_null_breaks_exact_utility_ties_in_favor_of_fewer_links(self):
        # All three edges have positive individual utility; the two-link swap adds zero.
        edges = [(0, 2, 1000.0), (0, 3, 3000.0), (1, 2, 3000.0)]
        arms, _ = assign(['a', 'a', 'b', 'b'], edges, [(0, 2)])
        self.assertEqual(arms['EXPLICIT_NULL'], [(0, 2)])
        edges[1] = (0, 3, math.nextafter(3000.0, 0))
        arms, _ = assign(['a', 'a', 'b', 'b'], edges, [(0, 2)])
        self.assertEqual(arms['EXPLICIT_NULL'], [(0, 3), (1, 2)])

    def test_truth_oracle_counts_fragment_pair_weights_not_links(self):
        edges = [(0, 3, 7), (3, 1, 4), (4, 0, 5), (2, 5, 2), (1, 5, 0)]
        self.assertEqual(maximum_pair_weight(['b'] * 3 + ['a'] * 3, edges), 11)
        self.assertEqual(maximum_pair_weight(['a'], []), 0)


if __name__ == '__main__':
    unittest.main()
