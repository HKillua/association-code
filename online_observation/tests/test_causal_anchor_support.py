import unittest

from online_observation.causal_anchor_support import AnchorVote, fit_causal_anchor


LEFT = ("a", "target")
RIGHT = ("b", "target")


def _vote(left, right, east, north=0):
    return AnchorVote(("a", left), ("b", right), ((east, north),) * 3)


class CausalAnchorSupportTests(unittest.TestCase):
    def test_exclude_both_target_endpoints_before_mutual_uniqueness(self):
        votes = [_vote("target", "one", 0), _vote("one", "target", 0),
                 _vote("target", "target", 0), _vote("one", "one", 4500, -40),
                 _vote("two", "two", 4700, 0)]
        expected = fit_causal_anchor(votes, LEFT, RIGHT)
        self.assertEqual(expected.status, "fitted")
        self.assertEqual(expected.translation, (4600, -20))
        self.assertEqual(expected.witnesses, ((("a", "one"), ("b", "one")),
                                             (("a", "two"), ("b", "two"))))
        self.assertEqual(expected.candidate_pairs, 2)
        self.assertEqual(expected, fit_causal_anchor(reversed(votes), LEFT, RIGHT))

    def test_single_witness_is_only_a_diagnostic_and_repetition_is_not_independent(self):
        vote = _vote("one", "one", 4700)
        fit = fit_causal_anchor([vote, vote], LEFT, RIGHT)
        self.assertEqual(fit.status, "single")
        self.assertIsNone(fit.translation)
        self.assertEqual(fit.single_translation, (4700, 0))
        self.assertEqual(fit.stable_pairs, 1)

    def test_mutual_uniqueness_rejects_shared_endpoint_before_stability_check(self):
        vote = _vote("one", "one", 4700)
        conflicting = AnchorVote(("a", "one"), ("b", "two"), ((5000, 0), (5000, 0), (5600, 0)))
        fit = fit_causal_anchor([vote, conflicting], LEFT, RIGHT)
        self.assertEqual(fit.status, "ambiguous")
        self.assertEqual(fit.candidate_pairs, 2)
        self.assertEqual(fit.unique_pairs, 0)
        self.assertIsNone(fit.translation)

    def test_within_witness_and_across_witness_limits(self):
        stable = AnchorVote(("a", "one"), ("b", "one"), ((4000, 0), (4500, 0), (5000, 0)))
        self.assertEqual(fit_causal_anchor([stable, _vote("two", "two", 5000)],
                                         LEFT, RIGHT).translation, (4750, 0))
        bad_consensus = fit_causal_anchor([stable, _vote("two", "two", 5000.001)], LEFT, RIGHT)
        self.assertEqual(bad_consensus.status, "disagreement")
        self.assertIsNone(bad_consensus.translation)
        unstable = AnchorVote(("a", "one"), ("b", "one"), ((4000, 0), (4500, 0), (5000.001, 0)))
        self.assertEqual(fit_causal_anchor([unstable], LEFT, RIGHT).status, "unstable")


if __name__ == "__main__":
    unittest.main()
