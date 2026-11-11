import json
import unittest

from online_observation.continuous_metrics import evaluate_publications
from online_observation.nemot_episodes import pool_episodes, score_episodes
from online_observation.nemot_replay import ARMS, ReplayResult, publications


A = ("source1", "a")
B = ("source2", "b")
C = ("source2", "c")
D = ("source2", "d")


def membership(*groups):
    return tuple((str(i), tuple(sorted(members))) for i, members in enumerate(groups))


class NemotEpisodeTests(unittest.TestCase):
    def test_age_boundary_state_changes_censoring_and_original_metric_parity(self):
        wrong = membership((A, B), (C,))
        right = membership((A, C), (B,))
        reports = [(0, (A, B, C)), (60.25, (A, C)), (80, (B,)), (100, (A, C))]
        replay = ReplayResult(
            runs={"kf": [(0, wrong)], "history": [(0, wrong), (1.3, right), (81.5, wrong)],
                  "neighbors": [(0, right)]},
            native_reports=reports, first_candidates={}, diagnostics={}, fit={}, timing={},
        )
        labels = {A: "true", C: "true", B: "other"}
        score = score_episodes(replay, labels)
        self.assertEqual(score["ticks"], 101)
        for arm in ARMS:
            reference = evaluate_publications(reports, publications(replay, arm, 100), labels)
            self.assertEqual(score["arms"][arm]["pairs"], reference["pairs"]["cross_source"])

        kf_false = score["arms"]["kf"]["false"]
        self.assertEqual(kf_false["observed_length_histogram"], {21: 1, 61: 1})
        self.assertEqual(kf_false["by_end"]["support_lost"], dict(episodes=1, pair_seconds=61))
        self.assertEqual(kf_false["by_end"]["right_censored"], dict(episodes=1, pair_seconds=21))
        self.assertEqual(kf_false["by_start"]["returning_support"], dict(episodes=1, pair_seconds=21))
        self.assertEqual(kf_false["by_length"]["61+"], dict(episodes=1, pair_seconds=61))

        history_false = score["arms"]["history"]["false"]
        self.assertEqual(history_false["by_end"]["coactive_state_change"], dict(episodes=1, pair_seconds=2))
        self.assertEqual(history_false["by_length"]["1-5"], dict(episodes=1, pair_seconds=2))
        self.assertEqual(history_false["by_start"]["joint_support_already_present"], dict(episodes=1, pair_seconds=19))
        self.assertEqual(score["arms"]["kf"]["missed"]["observed_length_histogram"], {101: 1})
        self.assertEqual(score["arms"]["neighbors"]["missed"]["observed_length_seconds"],
                         {"p50": None, "p95": None, "max": None})

        diff = score["comparisons"]["history_minus_kf"]["false"]
        self.assertEqual({key: count for key, count in diff.items() if key.endswith("_pair_seconds")},
                         dict(added_pair_seconds=0, removed_pair_seconds=61,
                              common_pair_seconds=21, net_pair_seconds=-61))
        self.assertEqual(diff["removed_episodes"]["observed_length_histogram"], {2: 1, 59: 1})
        self.assertEqual(score["comparisons"]["neighbors_minus_kf"]["false"]["removed_episodes"]["by_length"]["61+"],
                         dict(episodes=1, pair_seconds=61))
        pooled = pool_episodes([score, json.loads(json.dumps(score))])
        self.assertEqual(pooled["ticks"], 202)
        self.assertEqual(pooled["arms"]["kf"]["false"]["observed_length_histogram"], {21: 2, 61: 2})
        self.assertEqual(pooled["arms"]["kf"]["false"]["observed_length_seconds"],
                         {"p50": 21, "p95": 61, "max": 61})
        self.assertEqual(pooled["comparisons"]["history_minus_kf"]["false"]["removed_pair_seconds"], 122)

    def test_equal_wrong_counts_can_replace_one_false_link_with_another(self):
        first = membership((A, B), (D,))
        second = membership((A, D), (B,))
        replay = ReplayResult(
            runs={"kf": [(0.25, first)], "history": [(0.25, second)], "neighbors": [(0.25, first)]},
            native_reports=[(0.25, (A, B, D)), (5, (A, B, D))],
            first_candidates={}, diagnostics={}, fit={}, timing={},
        )
        score = score_episodes(replay, {A: "A", B: "B", D: "D"})
        self.assertEqual(score["arms"]["kf"]["false"]["pair_seconds"], 5)
        self.assertEqual(score["arms"]["history"]["false"]["pair_seconds"], 5)
        self.assertEqual(score["arms"]["kf"]["false"]["by_start"]["first_observed_endpoint"],
                         dict(episodes=1, pair_seconds=5))
        diff = score["comparisons"]["history_minus_kf"]["false"]
        self.assertEqual({key: count for key, count in diff.items() if key.endswith("_pair_seconds")},
                         dict(added_pair_seconds=5, removed_pair_seconds=5,
                              common_pair_seconds=0, net_pair_seconds=0))
        self.assertEqual(diff["added_episodes"]["by_end"]["right_censored"], dict(episodes=1, pair_seconds=5))


if __name__ == "__main__":
    unittest.main()
