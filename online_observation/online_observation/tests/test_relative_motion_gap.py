import unittest

from online_observation.hypothesis_gap_diagnostic import Gap, Switch
from online_observation.relative_motion_gap import Trace, _decisions, _decompose, _motion


A, C = ("left", "1"), ("left", "2")
B, D = ("right", "1"), ("right", "2")
SWITCH = Switch(((A, B), (C, D)), ((A, D), (C, B)))


class RelativeMotionGapTest(unittest.TestCase):
    def test_single_source_transverse_shift_flips_relative_order(self):
        old = {A: (1., 0.), C: (0., 0.), B: (-1., 2.), D: (0., 0.)}
        new = old | {A: (1., 1.)}
        result = _decompose(SWITCH, old, new, {A})
        self.assertEqual((result["masked"], result["actual"]), (-2., 2.))
        self.assertEqual(result["terms"], (4., 0., 0.))
        self.assertEqual((result["axial_only"], result["transverse_only"]), (-2., 2.))
        self.assertFalse(result["axis_reversed"])
        quiet = _decompose(SWITCH, old, old, set())
        self.assertEqual(quiet["increment"], 0.)

    def test_motion_uses_one_bracket_and_masks_only_current_values(self):
        history = {
            A: [(0., (1., 0.)), (20., (3., 0.))],
            C: [(0., (0., 0.)), (25., (0., 0.))],
            B: [(0., (1., 0.)), (25., (3.5, 0.))],
            D: [(0., (0., 0.)), (25., (0., 0.))],
        }
        masked = {key: (0., 0.) for key in history}
        result = _motion(SWITCH, history, 25., masked, {B})
        self.assertEqual((result["start"], result["stop"]), (0., 20.))
        self.assertEqual(result["support"][B], (0., 25.))
        self.assertEqual(result["updated_reports"], 1)
        self.assertEqual(result["displacement"], 8.)
        self.assertAlmostEqual(result["mask_displacement"], -3.2)
        self.assertIsNone(_motion(SWITCH, history, 24., masked, set()))

    def test_two_distinct_source_batches_share_second_decision_time(self):
        gap = Gap(SWITCH, 10., ((A, D),), True, True, False, False, 0.)
        trace = Trace(gap)
        def sample(delay, source, gain):
            return dict(delay=delay, sources=1, updated_source_slot=source, masked=-2.,
                        actual=2., increment=gain)
        trace._new_point(sample(0., 0, -1.))
        trace._new_point(sample(1., 1, 1.))
        trace._new_point(sample(2., 1, 1.))
        rows = _decisions(trace.public())
        self.assertEqual(rows["first_cross"]["delay"], 1.)
        self.assertTrue(rows["first_cross"]["single"])
        self.assertFalse(rows["first_cross"]["consecutive"])
        self.assertTrue(rows["first_same"]["consecutive"])


if __name__ == "__main__":
    unittest.main()
