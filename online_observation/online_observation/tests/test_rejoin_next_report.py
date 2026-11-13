"""Exercise fixed witness selection and the single-update counterfactual."""

import unittest

from online_observation.hypothesis_gap_diagnostic import Switch
from online_observation.rejoin_next_report import score_next_report


A, C = ("a", "1"), ("a", "2")
B, D = ("b", "1"), ("b", "2")
SWITCH = Switch(((A, B), (C, D)), ((A, D), (C, B)))
FIT = dict(translation_east_north_m=(10.0, 1.0), source_order=("a", "b"))


def reports(stamps, east=0.0):
    return [(stamp, (east, 0.0)) for stamp in stamps]


def read(histories, witness=B, cache=(), *, after=None, stop=660.0):
    return score_next_report(SWITCH, histories, (B, 630.0), witness, cache, stop, FIT, after=after)


class RejoinNextReportTest(unittest.TestCase):
    def setUp(self):
        self.history = {
            A: reports((600.0, 610.0, 630.0, 646.0, 660.0), east=10.0),
            C: reports((600.0, 620.0, 642.0, 660.0), east=40.0),
            B: reports((600.0, 610.0, 620.0)) + [(630.0, (90.0, 0.0))] + reports((645.0, 660.0)),
            D: reports((600.0, 620.0, 641.0, 646.0), east=70.0),
        }

    def test_no_legal_witness_keeps_later_cache_and_boundary_visible(self):
        row = read(self.history, cache=(645.0,))
        self.assertEqual(row["status"], "cached_only")
        self.assertEqual(row["raw_after_focal_before_stop"], (645.0,))
        self.assertEqual(row["cached_after_floor_before_stop"], (645.0,))
        self.assertEqual(row["novel_after_floor_before_stop"], ())
        self.assertTrue(row["at_stop"])
        self.assertIsNone(row["scores"])
        at_floor = read(self.history, after=645.0)
        self.assertEqual(at_floor["status"], "no_later_raw")
        self.assertEqual(at_floor["raw_after_focal_before_stop"], (645.0,))
        self.assertEqual(at_floor["raw_after_floor_before_stop"], ())

    def test_same_track_holdout_is_not_a_candidate_rank_change(self):
        row = read(self.history)
        self.assertEqual(row["status"], "supported")
        self.assertEqual(row["witness_stamp"], 645.0)
        self.assertEqual(row["intervening_focal_stamps"], ())
        self.assertEqual(row["focal_prior_stamps"], dict(assimilate=630.0, skip=620.0))
        assimilated, skipped = (row["scores"][arm] for arm in ("assimilate", "skip"))
        self.assertEqual(assimilated["actual_m2"], skipped["actual_m2"])
        self.assertNotEqual(assimilated["masked_m2"], skipped["masked_m2"])
        self.assertGreater(assimilated["own_loss_m2"], skipped["own_loss_m2"])
        self.assertGreater(skipped["focal_innovation_variance_m2"], assimilated["focal_innovation_variance_m2"])

    def test_cross_track_witness_changes_only_affected_edge_and_holds_out_its_batch(self):
        row = read(self.history, A, after=642.0)
        self.assertEqual(row["witness_stamp"], 646.0)
        self.assertEqual(row["intervening_focal_stamps"], (645.0,))
        assimilated, skipped = (row["scores"][arm] for arm in ("assimilate", "skip"))
        self.assertEqual(assimilated["prior_stamps"], dict(own=630.0, retain=641.0, release=645.0))
        self.assertEqual(assimilated["prior_stamps"], skipped["prior_stamps"])
        self.assertEqual(assimilated["own_loss_m2"], skipped["own_loss_m2"])
        self.assertEqual(assimilated["retain_actual_m2"], skipped["retain_actual_m2"])
        self.assertNotEqual(assimilated["affected_edge_loss_m2"], skipped["affected_edge_loss_m2"])
        self.assertEqual(assimilated["affected_edge"], "release")
        other = self.history | {D: self.history[D][:-1] + [(646.0, (-10000.0, 3.0))]}
        self.assertEqual(row, read(other, A, after=642.0))

    def test_cached_intervening_update_and_strict_witness_bounds(self):
        history = self.history | {B: self.history[B][:-1] + reports((655.0, 660.0, 661.0))}
        row = read(history, cache=(645.0,))
        self.assertEqual(row["witness_stamp"], 655.0)
        self.assertEqual(row["intervening_focal_stamps"], (645.0,))
        self.assertEqual(row["focal_prior_stamps"], dict(assimilate=645.0, skip=645.0))
        self.assertEqual(row["raw_after_floor_before_stop"], (645.0, 655.0))
        self.assertEqual(row["novel_after_floor_before_stop"], (655.0,))

    def test_same_track_loss_survives_missing_partners_but_earliest_stale_skip_abstains(self):
        row = read(self.history | {C: []})
        self.assertEqual(row["status"], "own_only")
        self.assertIsNotNone(row["scores"]["skip"]["own_loss_m2"])
        self.assertIsNone(row["scores"]["skip"]["actual_m2"])
        stale = self.history | {B: reports((500.0, 630.0, 645.0, 655.0))}
        row = read(stale)
        self.assertEqual(row["status"], "stale_prior")
        self.assertEqual(row["witness_stamp"], 645.0)
        self.assertIsNone(row["scores"])


if __name__ == "__main__":
    unittest.main()
