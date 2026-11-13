import unittest

from online_observation.hypothesis_gap_diagnostic import Switch
from online_observation.past_motion_shape import score_past_motion


A0, A1, B0, B1 = (("a", "0"), ("a", "1"), ("b", "0"), ("b", "1"))
SWITCH = Switch(release=((A0, B0), (A1, B1)), retain=((A0, B1), (A1, B0)))


def _sample(stamp, innovation=(0, 0)):
    return dict(stamp=stamp, reporter=A0, keys=tuple(sorted(SWITCH.keys)), self=innovation)


def _histories(reporter_times=range(0, 121, 20)):
    def track(key, times):
        sign = 1 if key[1] == "0" else -1
        return [(time, (time, sign * min(time, 120 - time))) for time in times]

    return {key: track(key, reporter_times if key == A0 else range(0, 121, 20))
            for key in SWITCH.keys}


class PastMotionShapeTest(unittest.TestCase):
    def test_opposite_turns_distinguish_equal_net_motion(self):
        histories = _histories()
        result = score_past_motion(_sample(120), SWITCH, histories)
        self.assertEqual(result["actual"], {"net": 0, "change": 115200, "two_block": 57600, "recent": 3200})
        self.assertEqual(result["actual"], result["masked"])
        self.assertEqual((result["end"], result["available_at"], result["right_age"]), (120, 120, 0))
        self.assertTrue(result["current_used"])
        self.assertEqual(result["provenance"][A0], (0, 60, 100, 120))

        for key, track in histories.items():
            direction = 1 if key[0] == "a" else -1
            histories[key] = [(time, (point[0] + direction * time**2, point[1] + 300 * direction))
                              for time, point in track]
        self.assertEqual(score_past_motion(_sample(120), SWITCH, histories), result)

    def test_masks_only_reporter_bracket_and_ignores_future_reports(self):
        histories = _histories((0, 20, 40, 60, 80, 100, 115, 125))
        result = score_past_motion(_sample(125, (0, 10)), SWITCH, histories)
        self.assertEqual((result["end"], result["available_at"], result["right_age"]), (120, 125, 5))
        self.assertEqual(result["provenance"][A0], (0, 60, 100, 115, 125))
        self.assertEqual(result["actual"]["change"], 115200)
        self.assertEqual(result["masked"], {"net": 0, "change": 117600, "two_block": 58800, "recent": 3600})

        for track in histories.values():
            track.append((145, (1e5, -1e5)))
        self.assertEqual(score_past_motion(_sample(125, (0, 10)), SWITCH, histories), result)

    def test_new_report_may_not_supply_a_used_bracket(self):
        histories = _histories((0, 20, 40, 60, 80, 100, 120, 125))
        result = score_past_motion(_sample(125, (0, 1000)), SWITCH, histories)
        self.assertFalse(result["current_used"])
        self.assertEqual(result["available_at"], 120)
        self.assertEqual(result["actual"], result["masked"])

    def test_missing_and_wide_internal_brackets_abstain(self):
        histories = _histories()
        histories[B1] = [(time, (0, 0)) for time in (0, 20, 40, 100, 120)]
        wide = score_past_motion(_sample(120), SWITCH, histories)
        self.assertEqual((wide["status"], wide["end"]), ("gap_too_wide", 120))
        self.assertIsNone(wide["actual"])
        histories[B0] = histories[B0][1:]
        missing = score_past_motion(_sample(120), SWITCH, histories)
        self.assertEqual((missing["status"], missing["end"]), ("missing_window", 120))
        self.assertIsNone(missing["provenance"])
