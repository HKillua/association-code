"""Check report membership and displacement for the fixed-clock rejoin diagnostic."""
import unittest

from online_observation.rejoin_fresh_displacement import score_fresh_displacement


REPORTER = ("source-a", "track-1")
RELEASE = ("source-b", "track-2")
RETAIN = ("source-b", "track-3")
SAMPLE = dict(stamp=80.0, reporter=REPORTER, partners={"release": RELEASE, "retain": RETAIN})
HISTORIES = {
    REPORTER: [(30.0, (30.0, 0.0)), (50.0, (50.0, 0.0))],
    RELEASE: [(20.0, (20.0, 0.0)), (40.0, (40.0, 0.0)), (60.0, (60.0, 0.0))],
    RETAIN: [(20.0, (20.0, 0.0)), (40.0, (40.0, 0.0)), (70.0, (70.0, 30.0))],
}


class FreshDisplacementTest(unittest.TestCase):
    def test_removed_cache_matches_track_key_and_timestamp_and_keeps_the_query(self):
        histories = HISTORIES | {RETAIN: [*HISTORIES[RETAIN][:2], (55.0, (55.0, 30.0)), HISTORIES[RETAIN][-1]]}
        provenance = {REPORTER: (50.0,), RELEASE: (40.0,), RETAIN: (55.0,)}
        result = score_fresh_displacement(SAMPLE, histories, provenance, 65.0)
        self.assertEqual((result["left"], result["right"]), (30.0, 50.0))
        self.assertTrue(result["cache_verified"])
        decached = result["decached"]
        self.assertEqual(decached["status"], "supported")
        self.assertEqual(decached["brackets"], {
            "reporter": ((30.0, 30.0), (50.0, 50.0)),
            "release": ((20.0, 60.0), (20.0, 60.0)),
            "retain": ((20.0, 40.0), (40.0, 70.0)),
        })
        self.assertEqual(decached["coefficients"]["release"], {20.0: -0.5, 60.0: 0.5})
        self.assertAlmostEqual(decached["mismatch_m2"]["retain"], 100.0)
        self.assertAlmostEqual(decached["margin_m2"], 100.0)
        self.assertEqual(decached["available_at"], 70.0)
        self.assertEqual(decached["post_onset_roles"], ("retain",))
        self.assertTrue(decached["qualified"])
        self.assertEqual(result["opportunities"]["retain"], dict(
            before_batch=4, removed_cache=1, post_onset=1, post_onset_outside_cache=1))
        self.assertEqual(result["post_onset_masked"]["status"], "missing_window")

    def test_cancels_a_point_used_with_equal_weights_at_both_ends(self):
        result = score_fresh_displacement(SAMPLE, HISTORIES, {RELEASE: (), RETAIN: ()}, 35.0)["decached"]
        self.assertEqual(result["brackets"]["release"], ((20.0, 40.0), (40.0, 60.0)))
        self.assertEqual(result["coefficients"]["release"], {20.0: -0.5, 60.0: 0.5})
        self.assertEqual(result["post_onset_roles"], ("release", "retain"))
        self.assertEqual(result["post_onset_count"], 3)
        self.assertEqual(result["latest_contributor_at"], 70.0)

    def test_current_and_future_reports_do_not_appear_in_readouts_or_counts(self):
        provenance = {RELEASE: (), RETAIN: ()}
        padded = {key: points + [(80.0, (100.0, 0.0)), (100.0, (20.0, 30.0))]
                  for key, points in HISTORIES.items()}
        self.assertEqual(score_fresh_displacement(SAMPLE, padded, provenance, 69.0),
                         score_fresh_displacement(SAMPLE, HISTORIES, provenance, 69.0))
        with self.assertRaisesRegex(ValueError, "current or future"):
            score_fresh_displacement(SAMPLE, padded, provenance | {RETAIN: (80.0,)}, 69.0)

    def test_missing_cache_provenance_and_structural_abstentions_are_explicit(self):
        without_cache = score_fresh_displacement(SAMPLE, HISTORIES, {RELEASE: ()}, 55.0)
        self.assertFalse(without_cache["cache_verified"])
        self.assertEqual(without_cache["unfiltered"]["status"], "supported")
        self.assertEqual(without_cache["decached"]["status"], "missing_cache_provenance")

        for retain, removed, status in (
            ([(10.0, (10.0, 0.0)), (70.0, (70.0, 0.0))], (), "gap_too_wide"),
            ([(10.0, (10.0, 0.0)), (70.0, (70.0, 0.0))], (70.0,), "stale"),
            (HISTORIES[RETAIN], (70.0,), "missing_window"),
            (HISTORIES[RETAIN], (20.0, 40.0, 70.0), "missing_report"),
        ):
            with self.subTest(status=status):
                result = score_fresh_displacement(SAMPLE, HISTORIES | {RETAIN: retain},
                                                  {RELEASE: (), RETAIN: removed}, 55.0)["decached"]
                self.assertEqual(result["status"], status)
                self.assertFalse(result["qualified"])

    def test_constant_track_translation_does_not_change_displacement(self):
        shifted = HISTORIES | {RETAIN: [(t, (east + 500.0, north - 120.0))
                                       for t, (east, north) in HISTORIES[RETAIN]]}
        provenance = {RELEASE: (), RETAIN: ()}
        self.assertEqual(score_fresh_displacement(SAMPLE, shifted, provenance, 70.0),
                         score_fresh_displacement(SAMPLE, HISTORIES, provenance, 70.0))
        decached = score_fresh_displacement(SAMPLE, shifted, provenance, 70.0)["decached"]
        self.assertFalse(decached["qualified"])
        self.assertEqual(decached["post_onset_count"], 0)


if __name__ == "__main__":
    unittest.main()
