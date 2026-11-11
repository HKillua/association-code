import unittest

from online_observation.continuous_metrics import evaluate_publications


def snapshot(query_time, *groups):
    return dict(query_time=query_time,
                tracks=[dict(track_id=track_id, members=members) for track_id, members in groups])


class ContinuousMetricsTests(unittest.TestCase):
    def test_membership_pairs_mixed_duplicates_and_first_clean_link(self):
        left, sibling, right, other = ("A", "left"), ("A", "sibling"), ("B", "right"), ("C", "other")
        events = [(0, (left, sibling, other)), (.5, (right,))]
        labels = {left: "one", sibling: "one", right: "one", other: "two"}
        report = evaluate_publications(events, [
            snapshot(0, ("x", (left, sibling)), ("y", (other,))),
            snapshot(1, ("x", (left, right, other)), ("z", (sibling,))),
            snapshot(2, ("x", (left, right)), ("z", (sibling,)), ("y", (other,))),
            snapshot(3, ("x", (left, sibling, right)), ("y", (other,))),
        ], labels)
        self.assertEqual(report["pairs"]["cross_source"],
                         dict(truth_pairs=6, correct_pairs=4, missed_pairs=2, false_pairs=2))
        self.assertEqual(report["pairs"]["same_source"],
                         dict(truth_pairs=4, correct_pairs=2, missed_pairs=2, false_pairs=0))
        counts = report["counts"]
        self.assertEqual(counts["report_supported_target_seconds"], 8)
        self.assertEqual(counts["pure_tracked_target_seconds"], 7)
        self.assertEqual(counts["uniquely_tracked_target_seconds"], 6)
        self.assertEqual(counts["mixed_output_seconds"], 1)
        self.assertEqual(counts["duplicate_target_seconds"], 1)
        self.assertEqual(counts["id_switches_consecutive"], 1)
        self.assertEqual(counts["id_switches_after_gap"], 1)
        self.assertEqual(counts["reacquired_same_id"], 1)
        self.assertEqual(counts["reacquired_changed_id"], 1)
        self.assertEqual(counts["reacquired_while_report_supported"], 2)
        self.assertEqual(report["target_seconds_by_active_source_count"], {1: 5, 2: 3})
        self.assertEqual(report["by_source"]["A"]["report_supported_fragment_seconds"], 8)
        births = report["births"]
        self.assertEqual(births["first_publication"], dict(correct=0, incorrect=1, rejected=3, unpublished=0))
        self.assertEqual(births["by_opportunity"]["never_observed"]["births"], 3)
        self.assertEqual(births["by_opportunity"]["fresh"]["births"], 1)
        self.assertEqual(births["ever_correct"], 3)
        self.assertEqual(births["unconfirmed_by_end"], 1)
        self.assertEqual(births["first_correct_delay_seconds"]["sum"], 6.5)
        self.assertEqual(births["first_correct_delay_seconds"]["ceil_seconds_histogram"], {2: 2, 3: 1})

    def test_age_boundary_support_gap_and_identity_transfer(self):
        left, right, third = ("A", "left"), ("B", "right"), ("A", "other")
        events = [(0, (left,)), (0, (right,)), (62, (left,)), (63, (third,))]
        published = {
            0: (("x", (left, right)),),
            60: (("x", (left, right)),),
            61: (("x", (left, right)),),
            62: (("new", (left,)),),
            63: (("new", (left,)), ("x", (third,))),
        }
        report = evaluate_publications(events, (snapshot(tick, *published.get(tick, ())) for tick in range(64)),
                                       {left: "one", right: "one", third: "two"})
        self.assertEqual(report["pairs"]["cross_source"]["truth_pairs"], 61)
        self.assertEqual(report["births"]["by_opportunity"]["fresh"]["births"], 2)
        self.assertEqual(report["births"]["first_correct_delay_seconds"]["ceil_seconds_histogram"], {0: 2})
        counts = report["counts"]
        self.assertEqual(counts["reacquired_while_report_supported"], 1)
        self.assertEqual(counts["reacquired_after_report_support_gap"], 1)
        self.assertEqual(counts["system_id_identity_transfers"], 1)
        self.assertEqual(counts["unsupported_output_seconds"], 1)
        self.assertEqual(counts["stale_membership_entries"], 2)

    def test_repeated_members_deduplicate_pairs_and_leave_unpublished_births(self):
        left, right, missing = ("A", "left"), ("B", "right"), ("A", "missing")
        events = [(0, (left, right, missing))]
        report = evaluate_publications(events, [
            snapshot(0, ("x", (left, left, right)), ("y", (left, right))),
        ], {left: "one", right: "one", missing: "two"})
        self.assertEqual(report["pairs"]["cross_source"]["correct_pairs"], 1)
        self.assertEqual(report["counts"]["duplicate_membership_entries"], 1)
        self.assertEqual(report["counts"]["repeated_member_seconds"], 2)
        self.assertEqual(report["counts"]["duplicate_pure_outputs"], 1)
        self.assertEqual(report["counts"]["uniquely_tracked_target_seconds"], 0)
        self.assertEqual(report["births"]["first_publication"],
                         dict(correct=2, incorrect=0, rejected=0, unpublished=1))

    def test_rejects_future_members_and_missing_ticks(self):
        future = ("A", "future")
        with self.assertRaisesRegex(ValueError, "before its first report"):
            evaluate_publications([(1, (future,))], [snapshot(0, ("x", (future,)))], {future: "one"})
        with self.assertRaisesRegex(ValueError, "every consecutive"):
            evaluate_publications([], [snapshot(0), snapshot(2)], {})


if __name__ == "__main__":
    unittest.main()
