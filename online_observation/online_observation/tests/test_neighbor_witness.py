import unittest

from online_observation.neighbor_witness import NeighborVote, NeighborWitnessIndex, summarize_votes


class NeighborWitnessTest(unittest.TestCase):
    def test_bridge_uses_only_arrived_points_and_excludes_candidate_endpoints(self):
        left, right = ("a", "new"), ("b", "old")
        left_a, left_b = ("a", "first"), ("a", "second")
        right_a, right_b = ("b", "first"), ("b", "second")
        pending = ("b", "future_bracket")
        long_gap = ("a", "wide_gap")
        histories = {
            left: [(10, (1100, 0))],
            right: [(0, (5000, 0)), (8, (5080, 0))],
            left_a: [(7, (70, 0)), (9, (90, 0))],
            left_b: [(7, (70, 1000)), (9, (90, 1000))],
            right_a: [(7, (4070, 0)), (9, (4090, 0))],
            right_b: [(7, (4070, 1000)), (9, (4090, 1000))],
            pending: [(7, (4070, 1100)), (11, (4110, 1100))],
            long_gap: [(-50, (0, 0)), (9, (90, 500))],
        }

        view = NeighborWitnessIndex(10, histories, histories).lookup((left, right))

        self.assertEqual((view.epoch, view.latest_epoch, view.clock), (8, 10, "bridge"))
        self.assertIsNone(view.strict_offset)
        self.assertEqual(view.bridge_offset, (4000, 0))
        self.assertEqual(view.nearby_counts, (3, 3))
        self.assertEqual(view.cotimed_counts, (2, 2))
        self.assertEqual(view.consensus.best.witnesses, ((left_a, right_a), (left_b, right_b)))
        self.assertEqual(view.consensus.residual_m, 0)
        self.assertTrue(all(set(vote.pair).isdisjoint({left, right, pending, long_gap}) for vote in view.votes))

    def test_strict_offset_precedes_bridge_and_bridge_is_bounded(self):
        left, right = ("a", "old"), ("b", "current")
        histories = {left: [(0, (0, 0)), (10, (10, 0))], right: [(0, (20, 0)), (20, (100, 0))]}
        view = NeighborWitnessIndex(20, histories, histories).lookup((left, right))
        self.assertEqual((view.strict_offset, view.bridge_offset, view.clock), ((50, 0), (80, 0), "strict"))
        self.assertEqual(view.candidate_offset, view.strict_offset)

        histories[right] = [(31, (100, 0))]
        view = NeighborWitnessIndex(31, histories, histories).lookup((left, right))
        self.assertEqual(view.clock, "unavailable")
        self.assertIsNone(view.candidate_offset)

    def test_neighbor_cap_counts_only_co_timed_local_candidates(self):
        left, right = ("a", "target"), ("b", "target")
        histories = {left: [(20, (0, 0))], right: [(20, (1000, 0))]}
        histories.update({("a", str(index)): [(20, (index * 500, 0))] for index in range(1, 7)})
        histories[("b", "neighbor")] = [(20, (1100, 0))]
        histories[("b", "out_of_radius")] = [(20, (9100, 0))]
        view = NeighborWitnessIndex(20, histories, histories).lookup((left, right))

        self.assertEqual(view.eligible_counts, (6, 1))
        self.assertEqual(view.cotimed_counts, (4, 1))
        self.assertEqual(view.truncated, (True, False))
        self.assertEqual({vote.pair[0][1] for vote in view.votes}, {"1", "2", "3", "4"})

    def test_equal_support_parallel_solution_is_retained(self):
        left = [("a", "1"), ("a", "2")]
        right = [("b", "1"), ("b", "2"), ("b", "3")]
        votes = [NeighborVote((a, b), (4000 + 1000 * (j - i), 0))
                 for i, a in enumerate(left) for j, b in enumerate(right)]

        result = summarize_votes(votes, (5000, 0))

        self.assertEqual((result.best.offset, result.best.support), ((4000, 0), 2))
        self.assertEqual((result.competing.offset, result.competing.support), ((5000, 0), 2))
        self.assertEqual((result.candidate.support, result.residual_m), (2, 1000))

    def test_shared_endpoints_count_once_and_assignment_ambiguity_survives_cost_order(self):
        first, second = ("a", "1"), ("a", "2")
        one, two = ("b", "1"), ("b", "2")
        votes = [NeighborVote((first, one), (0, 0)), NeighborVote((first, two), (100, 0)),
                 NeighborVote((second, one), (-100, 0)), NeighborVote((second, two), (0, 0))]

        result = summarize_votes(votes, (0, 0))

        self.assertEqual(result.best.witnesses, ((first, one), (second, two)))
        self.assertTrue(result.best.assignment_ambiguous)
        self.assertEqual(summarize_votes(votes[:2], (0, 0)).best.support, 1)


if __name__ == "__main__":
    unittest.main()
