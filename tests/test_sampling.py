import copy
import json
import unittest

import numpy as np

from association.sampling import BalancedScenePairSampler


def make_scene(scene_id, labels):
    """Tiny scene-local candidate list; all nodes deliberately share a source."""
    labels = np.asarray(labels, dtype=np.float32)
    nodes = len(labels) + 1
    return {"scene_id": scene_id, "labels": labels,
            "pairs": np.column_stack((np.zeros(len(labels), dtype=np.int64),
                                       np.arange(1, nodes, dtype=np.int64))),
            "values": np.zeros((nodes, 2, 4), dtype=np.float32),
            "sources": ["same-source"] * nodes}


def serial_groups(groups):
    return [(index, pairs.tolist()) for index, pairs in groups]


class BalancedScenePairSamplerTests(unittest.TestCase):
    def test_every_batch_is_balanced_scene_local_and_preserves_inputs(self):
        scenes = [make_scene("mixed", [1, 0, 0, 1, 0]),
                  make_scene("positive-only", [1, 1]),
                  make_scene("negative-only", [0, 0, 0]), make_scene("empty", [])]
        before = copy.deepcopy(scenes)
        sampler = BalancedScenePairSampler(scenes, seed=7)
        for size in (2, 4, 6, 16):
            for _ in range(20):
                groups = sampler.sample(size)
                self.assertLessEqual(len(groups), 2)
                self.assertEqual(len({index for index, _ in groups}), len(groups))
                labels = []
                for scene_index, indices in groups:
                    self.assertEqual(indices.dtype, np.dtype(np.int64))
                    self.assertEqual(indices.ndim, 1)
                    self.assertTrue(((indices >= 0) & (indices < len(scenes[scene_index]["pairs"]))).all())
                    pairs = scenes[scene_index]["pairs"][indices]
                    self.assertTrue((pairs < len(scenes[scene_index]["values"])).all())
                    labels.extend(scenes[scene_index]["labels"][indices])
                self.assertEqual(len(labels), size)
                self.assertEqual(sum(labels), size // 2)
                metadata = sampler.batch_metadata(groups)
                self.assertEqual(metadata["positive_count"], size // 2)
                self.assertEqual(metadata["negative_count"], size // 2)
        for original, current in zip(before, scenes):
            for key in ("pairs", "labels", "values"):
                np.testing.assert_array_equal(current[key], original[key])
            self.assertEqual(current["sources"], original["sources"])

    def test_same_source_positive_is_retained(self):
        scenes = [make_scene("one", [1, 0])]
        sampler = BalancedScenePairSampler(scenes, seed=0)
        groups = sampler.sample(2)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0][1]), {0, 1})
        pair = scenes[0]["pairs"][0]
        self.assertEqual(scenes[0]["sources"][pair[0]], scenes[0]["sources"][pair[1]])
        self.assertEqual(sampler.batch_metadata(groups)["positive_count"], 1)

    def test_zero_pair_and_single_class_scenes_are_eligible_as_appropriate(self):
        scenes = [make_scene("empty", []), make_scene("positive", [1]),
                  make_scene("negative", [0]), make_scene("both", [0, 1])]
        sampler = BalancedScenePairSampler(scenes, seed=23)
        info = sampler.describe()
        self.assertEqual(info["eligible_scene_ids"],
                         {"positive": ["positive", "both"], "negative": ["negative", "both"]})
        self.assertEqual(info["eligible_scene_indices"], {"positive": [1, 3], "negative": [2, 3]})
        self.assertEqual(info["zero_pair_scene_ids"], ["empty"])
        self.assertEqual((info["scene_count"], info["pair_count"],
                          info["positive_pair_count"], info["negative_pair_count"]), (4, 4, 2, 2))
        seen = {0: set(), 1: set()}
        for _ in range(100):
            for scene_index, indices in sampler.sample(2):
                for index in indices:
                    seen[int(scenes[scene_index]["labels"][index])].add(scene_index)
        self.assertEqual(seen, {1: {1, 3}, 0: {2, 3}})
        json.dumps(info, allow_nan=False)

    def test_scene_selection_is_not_weighted_by_pair_pool_size(self):
        scenes = [make_scene("small", [1, 0]), make_scene("large", [1] * 9 + [0] * 9)]
        sampler = BalancedScenePairSampler(scenes, seed=84)
        selections = {0: np.zeros(2, dtype=int), 1: np.zeros(2, dtype=int)}
        large_pair_counts = np.zeros(18, dtype=int)
        for _ in range(2000):
            for scene_index, indices in sampler.sample(2):
                for index in indices:
                    label = int(scenes[scene_index]["labels"][index])
                    selections[label][scene_index] += 1
                    if scene_index == 1:
                        large_pair_counts[index] += 1
        for counts in selections.values():
            self.assertTrue((counts > 850).all(), counts)
            self.assertTrue((counts < 1150).all(), counts)
        # Broad bounds check uniform within-scene draws without brittle exact RNG fixtures.
        self.assertTrue((large_pair_counts > 60).all(), large_pair_counts)
        self.assertTrue((large_pair_counts < 170).all(), large_pair_counts)

    def test_seed_reproduces_class_draws_and_final_order(self):
        scenes = [make_scene("a", [1, 0, 1, 0]), make_scene("b", [1, 0, 0])]
        first = BalancedScenePairSampler(scenes, seed=123)
        second = BalancedScenePairSampler(scenes, seed=np.int64(123))
        for size in (2, 8, 4, 6, 2, 10):
            left = first.sample(size)
            first.describe()
            first.batch_metadata(left)
            self.assertEqual(serial_groups(left), serial_groups(second.sample(size)))
        other = BalancedScenePairSampler(scenes, seed=124)
        self.assertNotEqual([serial_groups(first.sample(8)) for _ in range(5)],
                            [serial_groups(other.sample(8)) for _ in range(5)])

    def test_no_replacement_when_each_selected_class_pool_is_large_enough(self):
        sampler = BalancedScenePairSampler([make_scene("both", [1] * 4 + [0] * 4)], seed=5)
        for size in (2, 4, 8):
            for _ in range(10):
                groups = sampler.sample(size)
                self.assertEqual(len(groups), 1)
                self.assertEqual(len(np.unique(groups[0][1])), size)
                self.assertEqual(sampler.batch_metadata(groups)["duplicate_pair_count"], 0)
        self.assertEqual(set(sampler.sample(8)[0][1]), set(range(8)))

    def test_replacement_only_for_the_undersized_class_pool(self):
        sampler = BalancedScenePairSampler([make_scene("both", [1] + [0] * 5)], seed=11)
        groups = sampler.sample(8)
        indices = groups[0][1]
        self.assertEqual(np.count_nonzero(indices == 0), 4)
        negatives = indices[indices != 0]
        self.assertEqual(len(np.unique(negatives)), 4)
        info = sampler.batch_metadata(groups)
        self.assertEqual((info["pair_count"], info["unique_pair_count"], info["duplicate_pair_count"]), (8, 5, 3))
        self.assertEqual(info["groups"][0]["duplicate_pair_indices"], [0])
        self.assertEqual(info["groups"][0]["duplicate_pair_counts"], [4])
        json.dumps(info, allow_nan=False)

    def test_equal_pair_indices_in_different_scenes_are_not_duplicates(self):
        sampler = BalancedScenePairSampler([make_scene("positive", [1]),
                                           make_scene("negative", [0])], seed=6)
        info = sampler.batch_metadata(sampler.sample(2))
        self.assertEqual(info["unique_pair_count"], 2)
        self.assertEqual(info["duplicate_pair_count"], 0)
        self.assertEqual(set(info["scene_ids"]), {"positive", "negative"})

    def test_missing_classes_fail_clearly(self):
        for scenes, message in (([], "no positive or negative"),
                                ([make_scene("empty", [])], "no positive or negative"),
                                ([make_scene("negative", [0, 0])], "no positive"),
                                ([make_scene("positive", [1, 1])], "no negative")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                BalancedScenePairSampler(scenes, seed=0)

    def test_invalid_batch_sizes_do_not_advance_rng(self):
        scenes = [make_scene("both", [1, 0, 1, 0])]
        sampler = BalancedScenePairSampler(scenes, seed=1)
        fresh = BalancedScenePairSampler(scenes, seed=1)
        for size in (0, -2, 1, 3, 2.0, "2", True, np.bool_(True), None):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "even integer >= 2"):
                sampler.sample(size)
        self.assertEqual(serial_groups(sampler.sample(np.int64(4))), serial_groups(fresh.sample(4)))

    def test_invalid_seed_labels_and_candidate_shapes_fail(self):
        for seed in (-1, 0.5, "1", True, np.bool_(False), None):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                BalancedScenePairSampler([make_scene("both", [1, 0])], seed=seed)
        for labels in ([1, 2], [0, np.nan], [0, np.inf], [[1, 0]], ["0", "1"], [1]):
            scene = make_scene("bad", [1, 0])
            scene["labels"] = np.asarray(labels)
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "labels"):
                BalancedScenePairSampler([scene], seed=0)
        for pairs in (np.array([0, 1]), np.array([[0., 1.], [0., 2.]]), np.array([[-1, 1], [0, 2]])):
            scene = make_scene("bad", [1, 0])
            scene["pairs"] = pairs
            with self.subTest(pairs=pairs), self.assertRaisesRegex(ValueError, "pair"):
                BalancedScenePairSampler([scene], seed=0)

    def test_batch_metadata_rejects_nonlocal_indices(self):
        sampler = BalancedScenePairSampler([make_scene("both", [1, 0])], seed=3)
        for groups in ([(1, np.array([0]))], [(0, np.array([-1]))],
                       [(0, np.array([2]))], [(0, np.array([0.5]))]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                sampler.batch_metadata(groups)


if __name__ == "__main__":
    unittest.main()
