import unittest

from online_observation.causal_common_motion import compare_common_motion


def _turn(stamp):
    return 5 * min(stamp, 80), 5 * max(0, stamp - 80)


def _history(stamps, position, shift=(0, 0)):
    return [(stamp, tuple(value + offset for value, offset in zip(position(stamp), shift)))
            for stamp in stamps]


class CommonMotionTests(unittest.TestCase):
    def test_asynchronous_brackets_cancel_translation_and_ignore_future_coordinates(self):
        left = _history(range(0, 141, 20), _turn)
        right = _history(range(5, 146, 20), _turn, (4800, -1200))
        evidence = compare_common_motion(left, right, 145)
        self.assertTrue(evidence.qualified)
        self.assertEqual((evidence.interval_end, evidence.available_at), (140, 145))
        self.assertEqual(evidence.left_points, ((100, 0), (400, 0), (400, 300)))
        self.assertAlmostEqual(evidence.score_m, compare_common_motion(
            left, _history(range(5, 146, 20), _turn), 145).score_m)
        left.append((150, (9000, -9000)))
        right.append((160, (-9000, 9000)))
        self.assertEqual(compare_common_motion(left, right, 145), evidence)

    def test_refuses_missing_brackets_and_excessive_internal_gap(self):
        left = _history(range(0, 141, 20), _turn)
        right = _history((5, 25, 45, 65, 125, 145), _turn)
        gap = compare_common_motion(left, right, 145)
        self.assertEqual(gap.status, "gap_too_wide")
        self.assertFalse(gap.supported)
        self.assertEqual(compare_common_motion(left, right, 20).status, "missing_window")
        self.assertEqual(compare_common_motion(left, right, -1).status, "missing_window")

    def test_low_motion_steady_motion_and_inconsistent_turn_abstain(self):
        stamps = range(0, 141, 20)
        still = _history(stamps, lambda stamp: (0, 0))
        self.assertEqual(compare_common_motion(still, still, 140).status, "low_motion")
        straight = _history(stamps, lambda stamp: (stamp * 5, 0))
        steady = compare_common_motion(straight, straight, 140)
        self.assertEqual(steady.status, "degenerate")
        self.assertTrue(steady.supported)
        self.assertEqual(steady.score_m, 0)
        reverse_turn = _history(stamps, lambda stamp: (_turn(stamp)[0], -_turn(stamp)[1]))
        other = compare_common_motion(_history(stamps, _turn), reverse_turn, 140)
        self.assertEqual(other.status, "mismatch")
        self.assertFalse(other.qualified)

    def test_exact_gap_and_path_bounds_are_inclusive(self):
        history = _history((0, 45, 60, 90, 120),
                           lambda stamp: (2 * min(stamp, 60), 2 * max(0, stamp - 60)))
        evidence = compare_common_motion(history, history, 120)
        self.assertTrue(evidence.qualified)
        self.assertEqual(evidence.available_at, 120)
