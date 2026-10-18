import json
import unittest

import numpy as np

from association.evaluation.association import compare_predictions, evaluate_scenes, grouping_diagnostics, paired_cluster_bootstrap, predict, select_threshold


def record(scene, labels, scores, n=None, component=None, sources=None):
    if n is None:
        n = int((1 + np.sqrt(1 + 8 * len(labels))) / 2)
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64).reshape(-1, 2)
    return {"scene_id": str(scene), "component_id": str(component if component is not None else scene),
            "scores": np.asarray(scores), "labels": np.asarray(labels), "pairs": pairs,
            "sources": np.asarray(sources if sources is not None else np.arange(n))}


class MetricsTests(unittest.TestCase):
    def test_fixed_population_and_empty_positive_convention(self):
        data = [record("positive", [1, 0, 0], [.8, .9, .1], sources=[1, 1, 2]),
                record("negative", [0], [.2]), record("singleton", [], [], n=1)]
        result = evaluate_scenes(data, .5)
        self.assertEqual(result["scene_count"], 2)
        self.assertAlmostEqual(result["macro_f1"], 5 / 6)
        self.assertAlmostEqual(result["micro_f1"], 2 / 3)
        self.assertAlmostEqual(result["positive_scene_macro_f1"], 2 / 3)
        self.assertEqual(result["negative_scene_false_acceptance_rate"], 0)
        self.assertEqual(result["coverage"]["scenes_without_pairs"], 1)
        self.assertEqual(result["itsa"]["tp"], 1)
        self.assertEqual(result["mstsa"]["fp"], 1)
        rejected = evaluate_scenes(data, None)
        self.assertEqual(rejected["scene_count"], result["scene_count"])
        self.assertEqual(rejected["macro_f1"], .5)
        self.assertIsNone(rejected["micro_precision"])
        self.assertIsNone(rejected["p_fa"])
        json.dumps(rejected, allow_nan=False)

    def test_threshold_tie_prefers_all_reject_and_scores_are_inclusive(self):
        data = [record("p", [1], [0]), record("n", [0], [0])]
        result = select_threshold(data)
        self.assertIsNone(result["threshold"])
        self.assertEqual(result["macro_f1"], .5)
        self.assertEqual(result["candidate_count"], 2)
        np.testing.assert_array_equal(predict([0, 1], 0), [True, True])
        np.testing.assert_array_equal(predict([0, 1], None), [False, False])

    def test_incremental_threshold_matches_exhaustive_oracle(self):
        rng = np.random.default_rng(9)
        for _ in range(20):
            data = [record(i, rng.integers(0, 2, 6), rng.integers(-4, 5, 6)) for i in range(8)]
            candidates = [None] + sorted({float(s) for row in data for s in row["scores"]}, reverse=True)
            values = [evaluate_scenes(data, threshold)["macro_f1"] for threshold in candidates]
            expected_index = next(i for i, value in enumerate(values) if abs(value - max(values)) < 1e-12)
            result = select_threshold(data)
            self.assertEqual(result["threshold"], candidates[expected_index])
            self.assertAlmostEqual(result["macro_f1"], values[expected_index])

    def test_bridge_probe_reveals_transitive_false_positives(self):
        # True clusters {0,1}, {2,3}; one false bridge 1--2 joins both.
        data = record(0, [1, 0, 0, 0, 0, 1], [1, 0, 0, 1, 0, 1], sources=[0, 0, 1, 1])
        result = grouping_diagnostics(data, .5)
        self.assertEqual(result["component_count"], 1)
        self.assertEqual(result["mixed_identity_components"], 1)
        self.assertEqual(result["false_bridge_edges"], 1)
        self.assertEqual(result["closure"]["fp"], 4)
        self.assertEqual(result["added_closure_fp"], 3)
        singleton = grouping_diagnostics(record(1, [0], [0]), .5)
        self.assertEqual(singleton["singleton_count"], 2)

    def test_wrong_right_reports_both_directions(self):
        baseline = [record(0, [1, 0, 1], [.1, .9, .9])]
        candidate = [record(0, [1, 0, 1], [.9, .1, .1])]
        result = compare_predictions(baseline, candidate, .5, .5)
        self.assertEqual(result["wrong_to_right"], 2)
        self.assertEqual(result["right_to_wrong"], 1)
        self.assertEqual(result["corrected_fp"], 1)
        self.assertEqual(result["corrected_fn"], 1)
        self.assertEqual(result["introduced_fn"], 1)
        self.assertEqual(result["net_corrected_edges"], 1)

    def test_cluster_bootstrap_is_paired_scene_weighted_and_reproducible(self):
        baseline = [record(i, [1], [0], component=i // 2) for i in range(20)]
        candidate = [record(i, [1], [1], component=i // 2) for i in range(20)]
        result = paired_cluster_bootstrap(baseline, candidate, .5, .5, seed=4)
        self.assertEqual(result["cluster_count"], 10)
        self.assertEqual(result["scene_count"], 20)
        self.assertEqual(result["ci95"], [1., 1.])
        self.assertEqual(result["difference_pp"], 100)
        repeated = paired_cluster_bootstrap(baseline, candidate, .5, .5, seed=4)
        self.assertEqual(result, repeated)
        descriptive = paired_cluster_bootstrap(baseline[:18], candidate[:18], .5, .5)
        self.assertIsNone(descriptive["ci95"])
        self.assertEqual(descriptive["replicates"], 0)
        json.dumps(result, allow_nan=False)

    def test_unequal_cluster_sizes_do_not_change_primary_scene_estimand(self):
        baseline = [record(0, [1], [0], component="a"),
                    record(1, [1], [1], component="b"), record(2, [1], [1], component="b")]
        candidate = [record(0, [1], [1], component="a"),
                     record(1, [1], [1], component="b"), record(2, [1], [1], component="b")]
        result = paired_cluster_bootstrap(baseline, candidate, .5, .5)
        self.assertAlmostEqual(result["difference"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
