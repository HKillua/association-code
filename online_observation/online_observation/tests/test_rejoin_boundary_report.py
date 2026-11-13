"""Verify the shared trigger and strictly prior update counterfactuals."""

import unittest

from online_observation.async_report_scores import score_batch
from online_observation.hypothesis_gap_diagnostic import Switch
from online_observation.nemot_replay import _registration_shifts
from online_observation.rejoin_async_innovation import _prior_states
from online_observation.rejoin_boundary_report import score_boundary


A, C = ("a", "1"), ("a", "2")
B, D = ("b", "1"), ("b", "2")
SWITCH = Switch(((A, B), (C, D)), ((A, D), (C, B)))
FIT = dict(translation_east_north_m=(10.0, 1.0), source_order=("a", "b"))
STOP = 650.0
SELECTED = dict(release=(B, 635.0), retain=(D, 643.0))


def reports(stamps, east):
    return [(stamp, (east, 0.0)) for stamp in stamps]


HISTORIES = {
    A: reports((600.0, 620.0, 640.0), 10.0) + [(STOP, (30.0, 4.0))],
    B: reports((600.0, 610.0, 630.0), 0.0) + reports((635.0,), 80.0),
    C: reports((600.0, 620.0, 640.0), 140.0),
    D: reports((600.0, 620.0, 630.0), 100.0) + reports((643.0,), 130.0) + reports((645.0,), 100.0),
}


def read(histories=HISTORIES, selections=SELECTED):
    return score_boundary(SWITCH, histories, selections, A, STOP, FIT)


class RejoinBoundaryReportTest(unittest.TestCase):
    def test_full_absorption_matches_the_existing_same_time_batch(self):
        found = read()
        self.assertEqual(found["status"], "supported")
        self.assertEqual(found["trigger_innovation_m"], (20.0, 4.0))
        priors = {key: _prior_states(history, STOP, STOP)[0] for key, history in HISTORIES.items()}
        previous = score_batch(SWITCH, STOP, {A: HISTORIES[A][-1][1]}, priors, _registration_shifts(FIT))
        assimilated = found["scores"]["assimilate"]
        self.assertAlmostEqual(assimilated["actual_m2"], previous["position_actual"])
        self.assertAlmostEqual(assimilated["masked_m2"], previous["position_masked"])
        self.assertAlmostEqual(assimilated["local"]["actual_m2"], previous["euclidean_actual"])
        self.assertEqual(found["focals"]["retain"]["intervening_stamps"], (645.0,))

    def test_single_update_ablation_affects_only_its_track_in_both_matchings(self):
        found = read()["scores"]
        baseline, skip_release, skip_retain, skip_both = (
            found[arm] for arm in ("assimilate", "skip_release", "skip_retain", "skip_both"))
        self.assertEqual(skip_release["edges"]["retain"]["trigger_m2"], baseline["edges"]["retain"]["trigger_m2"])
        self.assertNotEqual(skip_release["edges"]["release"]["trigger_m2"], baseline["edges"]["release"]["trigger_m2"])
        self.assertNotEqual(skip_release["edges"]["retain"]["companion_m2"], baseline["edges"]["retain"]["companion_m2"])
        self.assertEqual(skip_retain["edges"]["release"]["trigger_m2"], baseline["edges"]["release"]["trigger_m2"])
        self.assertNotEqual(skip_retain["edges"]["retain"]["trigger_m2"], baseline["edges"]["retain"]["trigger_m2"])
        self.assertEqual(skip_both["edges"]["release"]["trigger_m2"], skip_release["edges"]["release"]["trigger_m2"])
        self.assertEqual(skip_both["edges"]["retain"]["trigger_m2"], skip_retain["edges"]["retain"]["trigger_m2"])
        for score in found.values():
            self.assertAlmostEqual(score["increment_m2"], score["local"]["increment_m2"])
            self.assertEqual(score["prior_stamps"]["trigger"], 640.0)
            self.assertEqual(score["prior_stamps"]["companion"], 640.0)
            self.assertEqual(score["prior_stamps"]["retain"], 645.0)

    def test_trigger_changes_only_actual_view_and_future_is_ignored(self):
        baseline = read()
        later = {key: history + [(STOP + 1, (9999.0, -9999.0))] for key, history in HISTORIES.items()}
        self.assertEqual(baseline, read(later))
        changed = read(HISTORIES | {A: HISTORIES[A][:-1] + [(STOP, (31.0, 4.0))]})
        self.assertEqual(changed["focals"], baseline["focals"])
        for arm, score in changed["scores"].items():
            original = baseline["scores"][arm]
            self.assertEqual(score["prior_stamps"], original["prior_stamps"])
            self.assertEqual(score["masked_m2"], original["masked_m2"])
            self.assertEqual(score["local"]["variance_m2"], original["local"]["variance_m2"])
            self.assertNotEqual(score["actual_m2"], original["actual_m2"])

    def test_lost_skip_support_keeps_assimilation_but_prevents_full_comparison(self):
        partial = read(HISTORIES | {B: reports((580.0, 635.0), 0.0)})
        self.assertEqual(partial["status"], "partial_support")
        self.assertEqual(partial["scores"]["assimilate"]["status"], "supported")
        self.assertEqual(partial["scores"]["skip_release"]["status"], "stale_prior")
        self.assertIsNone(partial["scores"]["skip_release"]["actual_m2"])
        missing = read(HISTORIES | {B: reports((635.0,), 0.0)})
        self.assertEqual(missing["focals"]["release"]["status"], "missing_prior")
        self.assertEqual(missing["scores"]["assimilate"]["status"], "supported")
        self.assertEqual(missing["scores"]["skip_release"]["status"], "missing_prior")

    def test_frozen_trigger_and_opponent_membership_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "one fixed-group trigger"):
            read(HISTORIES | {C: HISTORIES[C] + [(STOP, (140.0, 0.0))]})
        with self.assertRaisesRegex(ValueError, "original two opposing candidates"):
            read(selections=SELECTED | {"release": (D, 643.0)})


if __name__ == "__main__":
    unittest.main()
