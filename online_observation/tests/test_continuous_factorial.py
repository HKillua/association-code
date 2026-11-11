import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from online_observation import continuous_factorial as factorial
from online_observation.continuous_tracker import ContinuousTracker, LocalTrack
from online_observation.stream import EARTH_RADIUS_M


def point(metres):
    return 0.0, math.degrees(metres / EARTH_RADIUS_M)


def groups(snapshot):
    return {frozenset(track["members"]) for track in snapshot["tracks"]}


class TrackerTest(unittest.TestCase):
    def test_global_competition_does_not_force_forbidden_or_unmatched_pairs(self):
        a, b, c, d = ("A", "a"), ("A", "b"), ("B", "c"), ("B", "d")
        event = {a: point(0), b: point(3000), c: point(1000), d: point(-2000)}
        local = ContinuousTracker("hold_local")
        global_ = ContinuousTracker("hold_global")
        local.ingest(0, event)
        global_.ingest(0, event)
        self.assertEqual(groups(local.publish(0)), {frozenset((a, c)), frozenset((b,)), frozenset((d,))})
        self.assertEqual(groups(global_.publish(0)), {frozenset((a, d)), frozenset((b, c))})

    def test_kalman_only_changes_old_position_on_irregular_events(self):
        initial = LocalTrack(1, (0.0, 0.0), (0.0, 0.0))
        updated = initial.updated(61, (600.0, 0.0))
        self.assertEqual(updated.position(61, True), (600.0, 0.0))
        self.assertGreater(updated.position(111, True)[0], 1000)
        self.assertEqual(updated.position(1000, True), updated.position(121, True))
        a, b, c = ("A", "a"), ("A", "b"), ("B", "c")
        events = [(0, {a: point(0), b: point(1100)}), (60, {a: point(600), b: point(1100)}),
                  (119, {c: point(1170)})]
        for arm, target in (("hold_local", b), ("kf_local", a)):
            tracker = ContinuousTracker(arm)
            for stamp, reports in events:
                tracker.ingest(stamp, reports)
            self.assertIn(frozenset((target, c)), groups(tracker.publish(119)))

    def test_dormant_identity_outlives_position_and_same_source_replacement(self):
        a, b, later = ("A", "old"), ("B", "other"), ("A", "new")
        tracker = ContinuousTracker("hold_global")
        tracker.ingest(0, {a: point(1)})
        identity = tracker.publish(0)["tracks"][0]["track_id"]
        self.assertFalse(tracker.publish(61)["tracks"])
        tracker.ingest(70, {b: point(2)})
        self.assertEqual(tracker.publish(70)["tracks"][0]["track_id"], identity)
        self.assertEqual(groups(tracker.publish(71)), {frozenset((b,))})
        tracker.ingest(80, {later: point(2)})
        self.assertEqual(groups(tracker.publish(80)), {frozenset((b, later))})
        with self.assertRaises(ValueError):
            tracker.ingest(79, {a: point(1)})

    def test_equal_geometry_stays_pending_under_source_permutations(self):
        tracker = ContinuousTracker("hold_global")
        tracker.ingest(0, {("A", "1"): point(0), ("B", "1"): point(20), ("A", "2"): point(40)})
        self.assertEqual(len(tracker.publish(0)["tracks"]), 3)
        renamed = ContinuousTracker("hold_global")
        renamed.ingest(0, {("ZZ", "2"): point(40), ("AA", "1"): point(20), ("ZZ", "1"): point(0)})
        self.assertEqual(len(renamed.publish(0)["tracks"]), 3)


class FactorialTest(unittest.TestCase):
    def test_all_arms_and_scenes_predicted_before_labels(self):
        calls = []
        raw = [{"source": "A", "batch": "a", "time": "0", "lat": "0", "lon": "0"},
               {"source": "B", "batch": "b", "time": "1.2", "lat": "0", "lon": "0"}]
        mapping = {("A", "a"): "1-2", ("B", "b"): "1-2"}

        def measurements(_archive, _manifest, scene, *, required_split):
            calls.append(("prediction", scene))
            self.assertEqual(required_split, "T_core")
            return raw

        def labels(_archive, _manifest, scene):
            calls.append(("scoring", scene))
            return mapping

        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text('{"components":{"1":"x","2":"y"}}', encoding="utf-8")
            with (patch.object(factorial, "ROSTER", {"sample": ("1", "2")}),
                  patch.object(factorial, "read_archive_rows", side_effect=measurements),
                  patch.object(factorial, "read_labels", side_effect=labels)):
                result = factorial._run((), manifest, root)
        self.assertEqual(calls, [("prediction", "1"), ("prediction", "2"), ("scoring", "1"), ("scoring", "2")])
        for arm in result["aggregate"].values():
            self.assertEqual(arm["counts"]["ticks"], 6)
            self.assertEqual(arm["pairs"]["cross_source"]["correct_pairs"], 2)
            self.assertEqual(arm["births"]["first_correct_by_seconds"]["1"], 2)


if __name__ == "__main__":
    unittest.main()
