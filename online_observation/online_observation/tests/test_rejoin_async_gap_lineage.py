"""Verify that raw-time support and cache-induced gaps remain distinguishable."""
import unittest

from online_observation.rejoin_async_gap_lineage import trace_gap_lineage


REPORTER, RELEASE, RETAIN = ("a", "1"), ("b", "2"), ("b", "3")
SAMPLE = dict(stamp=80.0, reporter=REPORTER, partners={"release": RELEASE, "retain": RETAIN})


def reports(*times):
    return [(time, None) for time in times]


class GapLineageTest(unittest.TestCase):
    def test_deletion_creates_fifty_second_gap_from_five_native_steps(self):
        histories = {REPORTER: reports(40, 60), RELEASE: reports(20, 30, 40, 50, 60, 70),
                     RETAIN: reports(30, 40, 50, 60, 75)}
        cached = {RELEASE: (30, 40, 50, 60), RETAIN: (60,), REPORTER: (60,)}
        result = trace_gap_lineage(SAMPLE, histories, cached, 65)
        self.assertEqual((result["left"], result["right"]), (40, 60))
        self.assertEqual((result["raw_status"], result["retained_status"]), ("supported", "gap_too_wide"))
        self.assertEqual(result["novel_roles"], ("release", "retain"))
        release = result["roles"]["release"]
        self.assertEqual(release["raw_brackets"], ((40, 40), (60, 60)))
        self.assertEqual(release["retained_brackets"], ((20, 70), (20, 70)))
        self.assertEqual(release["violations"], (dict(start=20, end=70, span_s=50,
                          raw_interior_times=(30, 40, 50, 60), removed_times=(30, 40, 50, 60),
                          native_steps_s=(10, 10, 10, 10, 10), native_max_s=10, kind="cache_induced"),))
        self.assertEqual((release["raw_age_s"], release["retained_age_s"]), (10, 10))
        self.assertEqual(result["roles"]["reporter"]["old_cache_count"], 0)
        self.assertEqual(result["roles"]["retain"]["old_cache_count"], 1)

    def test_distinguishes_native_and_mixed_gap_with_one_common_clock(self):
        histories = {REPORTER: reports(40, 60), RELEASE: reports(20, 70), RETAIN: reports(10, 58, 70)}
        result = trace_gap_lineage(SAMPLE, histories, {RELEASE: (), RETAIN: (58,)}, 60)
        self.assertEqual((result["raw_status"], result["retained_status"]), ("gap_too_wide", "gap_too_wide"))
        self.assertEqual(result["roles"]["release"]["violations"][0]["kind"], "native")
        mixed = result["roles"]["retain"]["violations"][0]
        self.assertEqual((mixed["kind"], mixed["native_steps_s"]), ("mixed", (48, 12)))

    def test_tracks_unbracketed_retained_stream_and_rejects_future_cache(self):
        histories = {REPORTER: reports(40, 60, 80), RELEASE: reports(20, 50, 80, 81),
                     RETAIN: reports(40, 60, 70, 80)}
        result = trace_gap_lineage(SAMPLE, histories, {RELEASE: (), RETAIN: (70,)}, 65)
        self.assertEqual((result["raw_status"], result["retained_status"]), ("missing_window", "missing_window"))
        self.assertEqual(result["roles"]["release"]["retained_brackets"], ((20, 50), None))
        self.assertEqual(result["roles"]["retain"]["raw_age_s"], 10)
        self.assertEqual(result["roles"]["retain"]["retained_age_s"], 20)
        unknown = trace_gap_lineage(SAMPLE, histories, {RELEASE: ()}, 65)
        self.assertFalse(unknown["cache_verified"])
        self.assertEqual(unknown["retained_status"], "missing_cache_provenance")
        self.assertIsNone(unknown["roles"]["retain"]["novel_post_onset"])
        with self.assertRaisesRegex(ValueError, "current or future"):
            trace_gap_lineage(SAMPLE, histories, {RELEASE: (), RETAIN: (80,)}, 65)
