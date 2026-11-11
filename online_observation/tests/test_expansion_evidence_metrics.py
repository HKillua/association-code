import json
from types import SimpleNamespace
import unittest

from online_observation.expansion_evidence_metrics import pool_expansion_evidence, score_expansion_evidence


def _edge(left, right, *, wide=False, witnesses=(), **features):
    return SimpleNamespace(pair=(left, right), wide=wide, features=features, witnesses=witnesses)


def _frame(stamp, updated, fresh, *edges):
    return SimpleNamespace(stamp=stamp, updated=updated, fresh=fresh, edges=edges)


def _fixture():
    ax, ay, aw, ap = (("A", name) for name in ("x", "y", "w", "p"))
    bx, by, bw, bq, br = (("B", name) for name in ("x", "y", "w", "q", "r"))
    initial = {ax, by, aw, bw, ap, bq, br}
    frames = (
        _frame(0, initial, initial, _edge(ax, by, witness_count=0)),
        _frame(10, {bx}, initial | {bx}, _edge(ax, by, wide=True, repeat_both=0, range_m=4500), _edge(ax, bx)),
        _frame(30, {ay}, initial | {bx, ay},
               _edge(ax, by, wide=True, witnesses=((aw, bw), (ap, bq)), witness_count=2, range_m=7000,
                     past_drift_m=400, repeat_offset_drift_m=399, repeat_both=1, singleton=1, episode_age_s=20),
               _edge(ax, bx), _edge(ay, by)),
        _frame(100, {ax, by}, {ax, by}, _edge(ax, by, wide=True, witness_count=0)),
    )
    labels = {key: key[1] for key in initial | {bx, ay}}
    return frames, labels


class ExpansionEvidenceMetricsTests(unittest.TestCase):
    def test_truth_competition_uses_arrived_history_and_first_wide_is_independent(self):
        frames, labels = _fixture()
        scores = score_expansion_evidence(frames, labels)
        populations = scores["populations"]
        open_unseen = populations["all_open_events"]["false_unseen_truth"]["counts"]
        self.assertEqual(open_unseen["edges"], 2)
        self.assertEqual(open_unseen["seen_truth_ends_0"], 1)
        self.assertEqual(open_unseen["seen_truth_ends_1"], 1)
        self.assertEqual(open_unseen["fresh_truth_ends_0"], 1)
        self.assertEqual(open_unseen["open_truth_ends_1"], 1)
        self.assertEqual(open_unseen["sparse_ends_1"], 2)
        self.assertEqual(open_unseen["short_revisit"], 1)

        wide_seen = populations["wide_events"]["false_both_seen"]
        self.assertEqual(wide_seen["counts"]["edges"], 2)
        self.assertEqual(wide_seen["counts"]["fresh_truth_ends_0"], 1)
        self.assertEqual(wide_seen["counts"]["open_truth_ends_2"], 1)
        self.assertEqual(wide_seen["counts"]["short_revisit"], 1)
        self.assertEqual(populations["first_wide_per_pair"]["false_both_seen"]["counts"]["edges"], 0)
        first_wide = populations["first_wide_per_pair"]["false_unseen_truth"]["counts"]
        self.assertEqual(first_wide["fresh_truth_ends_1"], 1)
        first_open = populations["first_open_per_pair"]["false_unseen_truth"]["counts"]
        self.assertEqual(first_open["fresh_truth_ends_0"], 1)
        self.assertEqual(populations["first_open_per_pair"]["true"]["counts"]["edges"], 2)

    def test_fixed_bins_joint_missing_and_witness_truth_survive_pooling(self):
        frames, labels = _fixture()
        scores = score_expansion_evidence(frames, labels)
        wide_seen = scores["populations"]["wide_events"]["false_both_seen"]
        self.assertEqual(wide_seen["histograms"]["range_m"], {"above": 1, "missing": 1})
        self.assertEqual(wide_seen["histograms"]["past_drift_m"], {"le_400": 1, "missing": 1})
        self.assertEqual(wide_seen["histograms"]["witness_count"], {"le_2": 1, "le_0": 1})
        self.assertEqual(wide_seen["histograms"]["witness_residual_m"], {"missing": 2})
        self.assertEqual(wide_seen["histograms"]["episode_age_s"], {"le_20": 1, "missing": 1})
        self.assertEqual(wide_seen["counts"]["witness_truth_mixed"], 1)
        self.assertEqual(wide_seen["counts"]["witness_true_edges"], 1)
        self.assertEqual(wide_seen["counts"]["witness_false_edges"], 1)
        for outcomes in wide_seen["joint_support"].values():
            self.assertEqual(outcomes, {"yes": 1, "missing": 1})

        reopened = json.loads(json.dumps(scores))
        pooled = pool_expansion_evidence([reopened, reopened])
        self.assertEqual(pooled["events"], 8)
        self.assertEqual(pooled["populations"]["first_open_per_pair"]["true"]["counts"]["edges"], 4)
        self.assertEqual(pooled["populations"]["wide_events"]["false_both_seen"]["counts"]["edges"], 4)
