import unittest

from online_observation.interval_assignment import Action, _accept, decide


A1, A2, A3, A4, A5 = ("a", "1"), ("a", "2"), ("a", "3"), ("a", "4"), ("a", "5")
B1, B2, B3, B4, B5 = ("b", "1"), ("b", "2"), ("b", "3"), ("b", "4"), ("b", "5")
ORIGINAL = ((A1, B1), (A2, B2))
CROSSED = ((A1, B2), (A2, B1))


class IntervalAssignmentTests(unittest.TestCase):
    def test_integral_resolves_equal_endpoint_costs(self):
        history = {A1: [(0, (0, 0)), (20, (50, 0))], A2: [(0, (100, 0)), (20, (50, 0))]}
        history.update({B1: history[A2], B2: history[A1]})
        result, diagnostics = decide(20, {A1}, dict.fromkeys(ORIGINAL + CROSSED, 1),
                                     set(ORIGINAL), history, {})
        self.assertEqual(result["point_swap"], ())
        action, = result["interval_swap"]
        self.assertEqual(action.added, CROSSED)
        self.assertAlmostEqual(action.gain_m2, 2 * 100**2 / 3)
        self.assertEqual(diagnostics["groups_eligible"], 1)

    def test_null_pays_for_both_unmatched_endpoints(self):
        positions = {A1: 0, B1: 3000, A2: -3000, B2: 0}
        history = {key: [(0, (x, 0)), (20, (x, 0))] for key, x in positions.items()}
        result, _ = decide(20, {A1}, dict.fromkeys(ORIGINAL + CROSSED[:1], 1), set(ORIGINAL), history, {})
        self.assertEqual(result["interval_swap"], ())
        action, = result["interval_null"]
        self.assertFalse(action.complete)
        self.assertEqual(action.added, CROSSED[:1])
        self.assertEqual(action.gain_m2, 2 * 3000**2 - 4000**2)

    def test_does_not_interpolate_across_unsupported_gap(self):
        histories = {A1: [(0, (0, 0)), (80, (0, 0))],
                     A2: [(20, (-3000, 0)), (60, (-3000, 0))],
                     B1: [(20, (3000, 0)), (60, (3000, 0))],
                     B2: [(20, (0, 0)), (60, (0, 0))]}
        result, diagnostics = decide(80, {A1}, dict.fromkeys(ORIGINAL + CROSSED[:1], 1),
                                     set(ORIGINAL), histories, {})
        self.assertFalse(any(result.values()))
        self.assertEqual(diagnostics["groups_insufficient"], 1)

    def test_exact_conflicting_tie_does_not_use_fragment_ids(self):
        unrelated = Action(((A4, B4), (A5, B5)), ((A4, B5),), 1, False, 20)
        swap = Action(ORIGINAL, CROSSED, 1, True, 20)
        conflict = Action(((A1, B1), (A3, B3)), ((A1, B3),), 1, False, 20)
        self.assertEqual(_accept([swap, unrelated, conflict]), (unrelated,))
