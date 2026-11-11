import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from online_observation import recent_motion_support as probe
from online_observation.positions import event_batches


def row(time, source, batch, longitude=0):
    return dict(time=time, source=source, batch=batch, lat=0, lon=longitude, vel="masked", cou="masked")


def scored_event(prepared, result, key):
    index = list(prepared.first_seen).index(key) + 1
    return next(event for event in result["events"] if event["fragment_index"] == index)


class RecentMotionSupportTests(unittest.TestCase):
    def test_native_boundaries_exact_tie_and_timeout(self):
        data = [
            row(40, "B", "truth"), row(99, "B", "expired"), row(100, "B", "truth"),
            row(160, "A", "query"), row(160, "A", "query"), row(160, "B", "same-time"),
            row(200, "B", "truth"), row(200, "A", "late", 1),
            row(280, "A", "query"), row(280, "B", "truth"),
            row(320.01, "A", "late", 1), row(350, "A", "missing", 1),
        ]
        prepared = probe.prepare_support(event_batches(data))
        key = ("A", "query")
        self.assertEqual(prepared.first[key].points, 1)
        self.assertEqual(prepared.second[key].points, 2)
        self.assertEqual(prepared.second[key].stamp, 280)
        candidates = {candidate.key: candidate for candidate in prepared.first[key].candidates}
        self.assertEqual(set(candidates), {("B", "truth"), ("B", "same-time")})
        self.assertEqual(candidates["B", "truth"].points, 2)
        self.assertEqual(candidates["B", "truth"].distance_rank, candidates["B", "same-time"].distance_rank)
        labels = {key: key[1] for key in prepared.first_seen} | {key: "truth"}
        scored = probe.score_support(prepared, labels)
        event = scored_event(prepared, scored, key)
        self.assertEqual(event["first"]["nearest_outcome"], "tie")
        self.assertTrue(event["first"]["true_opposite_history_visible"])
        self.assertEqual(event["second"]["nearest_outcome"], "true")
        self.assertEqual(event["paired_change"]["gained_true_both_sides_2_points"], 1)
        self.assertEqual(event["paired_change"]["true_candidate_with_new_opposite_report"], 1)
        self.assertEqual(scored_event(prepared, scored, ("A", "late"))["second_status"], "timeout")
        self.assertEqual(scored_event(prepared, scored, ("A", "missing"))["second_status"], "missing")
        self.assertEqual(sum(scored["paired"][field] for field in
                             ("second_observed", "second_timed_out", "second_missing")),
                         scored["paired"]["first_births"])

    def test_same_source_neighbors_and_newly_seen_truth(self):
        data = [row(stamp, source, "neighbor", .004) for source in ("A", "B")
                for stamp in (0, 20, 40, 60, 80)]
        data.extend((row(80, "A", "query"), row(90, "B", "true"),
                     row(100, "A", "query"), row(100, "B", "true")))
        key = ("A", "query")
        prepared = probe.prepare_support(event_batches(data))
        self.assertEqual(prepared.first[key], probe.prepare_support(
            event_batches(report for report in data if report["time"] <= 80)).first[key])
        self.assertEqual(prepared.first[key].neighbors, 1)
        self.assertEqual(prepared.second[key].neighbors, 1)
        latest_true = next(candidate for candidate in prepared.second[key].candidates if candidate.key == ("B", "true"))
        self.assertEqual((latest_true.points, latest_true.neighbors), (2, 1))
        labels = {key: "-".join(key) for key in prepared.first_seen} | {
            ("A", "query"): "target", ("B", "true"): "target"}
        scored = probe.score_support(prepared, labels)
        event = scored_event(prepared, scored, key)
        self.assertFalse(event["first"]["true_opposite_history_visible"])
        self.assertEqual(event["first"]["nearest_outcome"], "false")
        self.assertEqual(event["second"]["nearest_outcome"], "true")
        self.assertEqual(event["paired_change"]["gained_true_reachability"], 1)
        self.assertEqual(event["paired_change"]["newly_seen_true_candidate"], 1)
        self.assertEqual(event["paired_change"]["gained_true_both_sides_5_points_or_neighbor"], 1)
        self.assertEqual(scored["cohorts"]["second"]["query_points_at_least_5"], 0)

    def test_prefix_shift_applies_only_strictly_after_600_and_gate_is_inclusive(self):
        seed = {(source, str(i)): (0, longitude + shift)
                for source, shift in (("A", 0), ("B", .03)) for i, longitude in enumerate((0, .3, .6))}
        key, candidate = ("A", "query"), ("B", "candidate")
        events = [(stamp, seed) for stamp in (30, 60, 90)]
        events.extend((stamp, {key: (0, 0), candidate: (0, .05)}) for stamp in (600, 600.25))
        prepared = probe.prepare_support(events)
        self.assertEqual(prepared.fit["status"], "fitted")
        self.assertEqual(prepared.first[key].candidates, ())
        self.assertEqual(prepared.second[key].candidates[0].key, candidate)

        a, b, c = ("A", "a"), ("B", "b"), ("B", "c")
        histories = {key: probe.History(0, 0, point, probe.deque((0,))) for key, point in
                     ((a, (0, 0)), (b, (4000, 0)), (c, (4000.001, 0)))}
        sample, = probe._snapshot(0, (a,), histories, {})
        self.assertEqual([item.key for item in sample.candidates], [b])

    def test_cli_freezes_all_scenes_before_label_reads_and_writes_anonymous_events(self):
        calls = []

        def read_rows(parts, manifest, scene, *, required_split):
            self.assertEqual(required_split, "T_core")
            calls.append(("measurement", scene))
            return [row(0, "private-source", "private-batch")]

        def read_labels(parts, manifest, scene):
            self.assertEqual(calls[:2], [("measurement", "1"), ("measurement", "2")])
            calls.append(("label", scene))
            return {("private-source", "private-batch"): "private-identity"}

        with TemporaryDirectory() as temporary:
            destination = Path(temporary) / "summary.json"
            args = ["--archive", "one", "two", "three", "--manifest", "synthetic", "--output", str(destination)]
            with (patch.object(probe, "ROSTER", {"stratum": ("1", "2")}),
                  patch.object(probe, "read_archive_rows", side_effect=read_rows),
                  patch.object(probe, "read_labels", side_effect=read_labels)):
                probe.main(args)
                with self.assertRaises(FileExistsError):
                    probe.main(args)
            payload = destination.read_text()
        self.assertEqual(calls, [("measurement", "1"), ("measurement", "2"), ("label", "1"), ("label", "2")])
        self.assertNotIn("private-", payload)
        result = json.loads(payload)
        self.assertEqual(result["aggregate"]["paired"]["first_births"], 2)
        self.assertEqual(result["by_roster_stratum"]["stratum"]["paired"]["second_missing"], 2)


if __name__ == "__main__":
    unittest.main()
