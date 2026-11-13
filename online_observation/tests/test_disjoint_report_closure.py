import unittest

from online_observation.disjoint_report_closure import pair_kind, score_pair, select_pairs


A0, A1, B0, B1 = (("a", "0"), ("a", "1"), ("b", "0"), ("b", "1"))
RELEASE = {A0: B0, A1: B1, B0: A0, B1: A1}
RETAIN = {A0: B1, A1: B0, B0: A1, B1: A0}


def sample(stamp, reporter, actual=(3., 10.), masked=(1., 8.), shift=(0., 0.), ages=(2., 1.)):
    return dict(stamp=stamp, reporter=reporter, partners={"release": RELEASE[reporter], "retain": RETAIN[reporter]},
                cross={"release": (actual[0], 0.), "retain": (actual[1], 0.)},
                pred_delta={"release": (masked[0], 0.), "retain": (masked[1], 0.)},
                age={"release": ages[0], "retain": ages[1]}, shift_delta=shift)


def translated(report, translation):
    def shift(vector):
        return tuple(value + offset for value, offset in zip(vector, translation, strict=True))

    return report | {key: {side: shift(vector) for side, vector in report[key].items()}
                     for key in ("cross", "pred_delta")} | {"shift_delta": translation}


class DisjointReportClosureTest(unittest.TestCase):
    def test_reports_classify_by_overlap_under_both_matchings(self):
        older = sample(10., A0)
        self.assertEqual(pair_kind(older, sample(12., A0)), "repeat")
        self.assertEqual(pair_kind(older, sample(12., A1)), "disjoint")
        self.assertEqual(pair_kind(older, sample(12., B0)), "reciprocal_release")
        self.assertEqual(pair_kind(older, sample(12., B1)), "reciprocal_retain")

    def test_selection_uses_only_latest_causally_eligible_report_per_kind(self):
        current = sample(100., A0)
        latest = sample(90., A1)
        history = [sample(99., A1, shift=(1., 0.)), sample(45., B0), sample(85., A1),
                   latest, sample(100., A1), sample(101., A1), sample(40., A0), sample(39., B1)]
        self.assertEqual(select_pairs(history, current), {"reciprocal_release": history[1],
                                                          "disjoint": latest, "repeat": history[6]})
        with self.assertRaises(ValueError):
            score_pair(history[4], current)

    def test_each_mask_removes_only_the_intended_observation(self):
        older = sample(10., A0)
        newer = sample(12., A1, actual=(4., 7.), masked=(5., 8.))
        score = score_pair(older, newer)
        self.assertEqual((score["older_stamp"], score["newer_stamp"], score["lag_s"]), (10., 12., 2.))
        self.assertEqual({key: score[key] for key in ("actual", "masked", "earlier_masked", "both_masked", "delta")},
                         {"actual": 8., "masked": 0., "earlier_masked": -8., "both_masked": -16., "delta": 8.})
        self.assertEqual(score["sides"]["release"],
                         {"actual": 1., "masked": 4., "earlier_masked": 9., "both_masked": 16.})
        self.assertEqual(score_pair(translated(older, (120., -30.)), translated(newer, (120., -30.))), score)

    def test_reciprocal_orients_shift_and_audits_forecast_reuse(self):
        older = sample(10., A0)
        newer = sample(12., B0, actual=(-4., -7.), masked=(-5., -8.))
        score = score_pair(older, newer)
        self.assertEqual(score["actual"], 8.)
        self.assertEqual(score["kind"], "reciprocal_release")
        self.assertTrue(score["reused_older_report"])
        self.assertFalse(score_pair(older, newer | {"age": {"release": 2.01, "retain": 2.}})["reused_older_report"])
        self.assertTrue(score_pair(older, sample(12., B1, ages=(1., 2.)))["reused_older_report"])
        self.assertEqual(score_pair(translated(older, (120., -30.)), translated(newer, (-120., 30.))), score)


if __name__ == "__main__":
    unittest.main()
