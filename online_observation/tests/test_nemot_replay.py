import math
import unittest

from online_observation import nemot_replay as replay
from online_observation.stream import EARTH_RADIUS_M


def position(east):
    return 0.0, math.degrees(east / EARTH_RADIUS_M)


def matching(result, arm, second):
    return next(frame for frame in replay.publications(result, arm, second) if frame["query_time"] == second)


class NemotReplayTests(unittest.TestCase):
    def test_prediction_reorders_but_cannot_open_a_candidate_edge(self):
        a, b, c = ("A", "a"), ("B", "b"), ("B", "c")
        calls = []

        def predict(context, stamp, keys):
            calls.append((stamp, keys))
            self.assertEqual(context.tracks[b].stamp, stamp)
            self.assertEqual(context.tracks[c].stamp, stamp)
            # History would prefer c if its learned position could open the HOLD gate.
            return {"history": {a: (5000, 0)}, "neighbors": {b: (10000, 0)}}

        result = replay.replay([(0.0, {a: position(0)}), (10.0, {b: position(100), c: position(4100)})], predict)
        self.assertEqual(calls, [(10.0, (a,))])
        self.assertEqual(result.first_candidates[b].peers, (a,))
        self.assertEqual(result.first_candidates[c].peers, ())
        self.assertEqual(result.diagnostics["fixed_candidate_edge_events"], 1)
        self.assertEqual(result.diagnostics["first_peers_le3"], 3)
        scores = replay.score_native_births(result, {a: "target", b: "target", c: "other"})
        self.assertEqual(scores["kf"]["births"]["ever_correct"], 2)
        self.assertEqual(scores["history"]["births"]["ever_correct"], 0)
        self.assertEqual(scores["neighbors"]["births"]["ever_correct"], 2)
        self.assertEqual(scores["history"]["candidates"]["true_reachable"], 1)
        self.assertEqual(scores["history"]["candidates"]["first_rejected_when_true_reachable"], 1)

    def test_unrelated_report_reuses_edge_and_scoring_filters_expiry(self):
        a, b, other = ("A", "a"), ("B", "b"), ("A", "other")
        calls = []

        def predict(_, stamp, keys):
            calls.append((stamp, keys))
            return {}

        result = replay.replay([
            (0.0, {a: position(0)}),
            (10.25, {b: position(100)}),
            (11.5, {other: position(10000)}),
        ], predict)
        self.assertEqual(calls, [(10.25, (a,))])
        self.assertEqual(result.diagnostics["fixed_candidate_edge_events"], 2)
        self.assertEqual(result.diagnostics["rescored_edge_events"], 1)
        at_sixty = matching(result, "kf", 60)
        after_expiry = matching(result, "kf", 61)
        self.assertEqual(at_sixty["tracks"][0]["members"], (a, b))
        self.assertEqual(after_expiry["tracks"][0]["members"], (b,))
        self.assertEqual(at_sixty["tracks"][0]["track_id"], after_expiry["tracks"][0]["track_id"])
        scores = replay.score_native_births(result, {a: "target", b: "target", other: "other"})
        delays = scores["kf"]["births"]
        self.assertEqual(delays["first_correct_delay_seconds"]["ceil_seconds_histogram"], {0: 1, 11: 1})
        self.assertEqual(delays["delay_milliseconds_histogram"], {0: 1, 10250: 1})
        self.assertEqual(delays["first_publication"]["correct"], 1)

    def test_registration_activation_rescores_even_unchanged_edges(self):
        a, b, other = ("A", "a"), ("B", "b"), ("A", "other")
        c, d = ("A", "c"), ("B", "d")
        fit = {"translation_east_north_m": [1000, 0], "source_order": ["A", "B"]}
        calls = []

        def predict(_, stamp, keys):
            calls.append((stamp, keys))
            return {}

        result = replay.replay([
            (590.0, {a: position(0), b: position(4500), c: position(20000), d: position(21000)}),
            (600.0, {other: position(10000)}),
            (601.0, {other: position(10000)}),
        ], predict, fit=fit)
        self.assertEqual(calls, [(601.0, (a, c, b, d))])
        self.assertEqual(len(matching(result, "kf", 600)["tracks"]), 4)
        self.assertEqual(matching(result, "kf", 601)["tracks"][0]["members"], (a, b))

    def test_id_reuse_within_memory_and_repeated_source_conflicts(self):
        a, b = ("A", "a"), ("B", "b")
        result = replay.replay([
            (0.0, {a: position(0), b: position(100)}),
            (200.0, {a: position(0)}),
            (900.0, {a: position(0)}),
        ], lambda *_: {})
        first = matching(result, "kf", 0)["tracks"][0]["track_id"]
        self.assertEqual(matching(result, "kf", 200)["tracks"][0]["track_id"], first)
        self.assertNotEqual(matching(result, "kf", 900)["tracks"][0]["track_id"], first)

        a2, c = ("A", "a2"), ("C", "c")
        result = replay.replay([(0.0, {a: position(-3500), b: position(0), c: position(3500),
                                      a2: position(7000)})], lambda *_: {})
        self.assertEqual(result.diagnostics["fixed_candidate_edge_events"], 3)
        self.assertEqual(len(matching(result, "kf", 0)["tracks"]), 4)


if __name__ == "__main__":
    unittest.main()
