import math
import unittest

from online_observation.continuous_tracker import NULL_COST
from online_observation.nemot_edge_diagnostics import EdgeAuditRecorder, merge_edge_audits, score_edge_audit
from online_observation.nemot_replay import ARMS, replay
from online_observation.stream import EARTH_RADIUS_M


def position(east):
    return 0.0, math.degrees(east / EARTH_RADIUS_M)


class NemotEdgeDiagnosticTests(unittest.TestCase):
    def test_replay_keeps_original_decisions_and_audits_rank_reversal(self):
        first, true, false = ("A", "a"), ("B", "b"), ("B", "c")
        events = [(0.0, {true: position(100), false: position(300)}), (10.0, {first: position(0)})]
        recorder = EdgeAuditRecorder()

        def prediction(*_):
            return {"history": {true: (-20, 0), false: (-250, 0)}, "neighbors": {true: (5000, 0)}}

        def observe(stamp, updated, born, fresh, costs, memberships):
            with self.assertRaises(TypeError):
                costs["kf"] = {}
            with self.assertRaises(TypeError):
                costs["kf"][(first, true)] = 0
            with self.assertRaises(TypeError):
                memberships["kf"] = ()
            self.assertIsInstance(updated, frozenset)
            recorder(stamp, updated, born, fresh, costs, memberships)

        with_audit = replay(events, prediction, observer=observe)
        plain = replay(events, prediction)
        self.assertEqual(with_audit.runs, plain.runs)
        self.assertEqual(with_audit.diagnostics, plain.diagnostics)
        self.assertEqual(with_audit.first_candidates, plain.first_candidates)
        self.assertEqual(len(recorder.frames[0].queries), 2)
        self.assertTrue(all(not query.candidates for query in recorder.frames[0].queries))
        result = score_edge_audit(recorder.frames, {first: "target", true: "target", false: "other"})
        first_row = result["first_fresh_0_3"]
        self.assertEqual(first_row["support"]["true_never_seen"], 2)
        self.assertEqual(first_row["support"]["mixed_candidates"], 1)
        self.assertEqual(first_row["arms"]["kf"]["counts"]["true_rank_win"], 1)
        self.assertEqual(first_row["arms"]["history"]["counts"]["unselected_true_below_null_loss"], 1)
        self.assertEqual(first_row["arms"]["neighbors"]["counts"]["unselected_all_true_above_null"], 1)
        comparison = first_row["comparisons"]["kf_to_history"]["counts"]
        self.assertEqual(comparison["rank_win_to_loss"], 1)
        self.assertEqual(comparison["min_true_decreased_margin_decreased"], 1)
        self.assertEqual(comparison["correct_to_wrong"], 1)
        self.assertEqual(comparison["changed_selected_partner"], 1)

    def test_null_boundary_and_jointly_arrived_reports(self):
        a, b = ("A", "a"), ("B", "b")
        recorder = EdgeAuditRecorder()
        costs = {arm: {(a, b): 2 * NULL_COST} for arm in ARMS}
        singles = {arm: (("1", (a,)), ("2", (b,))) for arm in ARMS}
        recorder(0.0, {a, b}, {a, b}, {a, b}, costs, singles)
        result = score_edge_audit(recorder.frames, {a: "same", b: "same"})
        row = result["all"]
        self.assertEqual(row["support"]["queries"], 2)
        self.assertEqual(row["support"]["true_reachable"], 2)
        self.assertEqual(row["support"]["true_never_seen"], 0)
        self.assertEqual(row["arms"]["kf"]["counts"]["unselected_best_true_at_null"], 2)
        self.assertEqual(row["arms"]["kf"]["true_residual_m"]["le_4000"], 2)
        self.assertEqual(merge_edge_audits([result, result])["all"]["support"]["true_reachable"], 4)

    def test_outside_graph_stale_and_single_false_unseen(self):
        a, b, c = ("A", "a"), ("B", "b"), ("A", "c")
        recorder = EdgeAuditRecorder()
        replay([(0.0, {b: position(0)}), (10.0, {a: position(6000), c: position(0)}),
                (70.0, {a: position(6000)})], lambda *_: {}, observer=recorder)
        scores = score_edge_audit(recorder.frames, {a: "same", b: "same", c: "false"})
        first = scores["first"]
        self.assertEqual(first["support"]["true_outside_graph"], 1)
        self.assertEqual(first["support"]["true_never_seen"], 2)
        self.assertEqual(first["support"]["single_false_unseen"], 1)
        self.assertEqual(first["arms"]["kf"]["counts"]["single_false_unseen_accepted"], 1)
        self.assertEqual(scores["continuation"]["support"]["true_stale_only"], 1)


if __name__ == "__main__":
    unittest.main()
