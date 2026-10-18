import copy
import unittest

import numpy as np

from baselines.tsactfer_data import (build_positive_pools, fit_normalizer,
                                    pack_scene, sample_batch)


def make_scene(tracks, scene_id="train"):
    """Shared registration-shaped fixture with deliberate unequal time supports."""
    values = np.zeros((len(tracks), 64, 4), dtype=np.float32)
    observed = np.zeros((len(tracks), 64), dtype=bool)
    actual_times = np.full((len(tracks), 64), np.nan, dtype=np.float64)
    for node, (_, _, times) in enumerate(tracks):
        times = np.asarray(times, dtype=np.float64)
        slots = np.linspace(1, 62, len(times), dtype=int)
        observed[node, slots] = True
        actual_times[node, slots] = times
        values[node, slots] = np.column_stack((20 + times / 10000, 130 + times / 20000,
                                               np.full(len(times), 5), np.full(len(times), 180)))
    pairs = np.column_stack(np.triu_indices(len(tracks), 1)).astype(np.int64)
    identities = [row[0] for row in tracks]
    return {"scene_id": scene_id, "component_id": scene_id, "values": values,
            "observed": observed, "observation_times": actual_times,
            "grid": np.linspace(min(row[2][0] for row in tracks), max(row[2][-1] for row in tracks), 64),
            "sources": [row[1] for row in tracks], "identities": identities,
            "node_ids": [scene_id + ":" + str(index) for index in range(len(tracks))],
            "intervals": np.asarray([[row[2][0], row[2][-1]] for row in tracks], dtype=np.float64),
            "pairs": pairs,
            "labels": np.asarray([identities[a] == identities[b] for a, b in pairs], dtype=np.float32),
            "audit": {"registration": {"policy": "fixture-shared-64"}}}


class MidpointRng:
    """Deterministic uniform draws at distribution midpoints for crop assertions."""
    def choice(self, count, size, replace):
        assert not replace and count >= size
        return np.arange(size)

    def integers(self, count):
        assert count > 0
        return 0

    def uniform(self, low, high):
        return (low + high) / 2


class TSACtferDataTests(unittest.TestCase):
    def test_packing_preserves_every_registered_point_time_and_metadata(self):
        scene = make_scene([("A-0", "a", [10, 30, 85]), ("B-0", "b", [0, 55, 100, 140])])
        before = copy.deepcopy(scene)
        norm = fit_normalizer([scene])
        packed = pack_scene(scene, norm)
        self.assertEqual(packed["positions"].shape, (2, 128, 2))
        self.assertEqual(packed["positions"].dtype, np.float32)
        self.assertEqual(packed["time_ids"].dtype, np.int64)
        self.assertEqual(packed["times"].dtype, np.float64)
        self.assertEqual(packed["observed"].dtype, np.bool_)
        for key in ("values", "pairs", "labels", "sources", "identities", "intervals", "grid", "audit"):
            self.assertIs(packed[key], scene[key])
        for node in range(2):
            mask = scene["observed"][node]
            count = int(mask.sum())
            points = scene["values"][node, mask][:, [1, 0]]
            recovered = packed["positions"][node, :count] * norm["std"] + norm["mean"]
            np.testing.assert_allclose(recovered, points, rtol=0, atol=1e-7)
            np.testing.assert_array_equal(packed["times"][node, :count], scene["observation_times"][node, mask])
            np.testing.assert_array_equal(packed["time_ids"][node, :count],
                                          np.floor((scene["observation_times"][node, mask] - scene["grid"][0]) / 20))
            self.assertTrue(packed["observed"][node, :count].all())
            self.assertFalse(packed["observed"][node, count:].any())
        for field in ("positions", "time_ids", "times"):
            self.assertTrue(np.isfinite(packed[field]).all())
            self.assertTrue((packed[field][~packed["observed"]] == 0).all())
        self.assertEqual(packed["packing_audit"]["valid_points"], int(scene["observed"].sum()))
        self.assertEqual(packed["packing_audit"]["clipped_time_count"], 0)
        for key in ("values", "observed", "observation_times", "pairs", "labels"):
            np.testing.assert_array_equal(scene[key], before[key])

    def test_moments_use_only_supplied_training_observations_and_lon_lat_order(self):
        train = make_scene([("A-0", "a", [0, 200, 400])], "train-only")
        train["values"][0, train["observed"][0], 1] = 130
        train["values"][~train["observed"]] = 1e10
        norm = fit_normalizer([train])
        self.assertEqual(norm["training_scene_ids"], ["train-only"])
        self.assertEqual(norm["observed_point_count"], 3)
        self.assertEqual(norm["constant_features"], [0])
        self.assertEqual(norm["std"][0], 1)
        self.assertEqual(norm["mean"][0], 130)
        expected = train["values"][train["observed"]][:, [1, 0]].astype(np.float64)
        np.testing.assert_allclose(norm["mean"], expected.mean(axis=0))
        self.assertAlmostEqual(norm["std"][1], expected.std(axis=0)[1])
        validation = make_scene([("V-0", "b", [0, 200, 400])], "validation")
        validation["values"][validation["observed"]] += 100
        before = copy.deepcopy(norm)
        packed = pack_scene(validation, norm)
        self.assertEqual(norm, before)
        self.assertAlmostEqual(float(packed["positions"][0, 0, 0]), 100)

    def test_timestamp_clipping_is_counted_and_truncation_is_rejected(self):
        scene = make_scene([("A-0", "a", [0, 20, 40, 100])])
        norm = fit_normalizer([scene])
        packed = pack_scene(scene, norm, time_bins=2)
        self.assertEqual(packed["packing_audit"]["clipped_time_count"], 2)
        np.testing.assert_array_equal(packed["time_ids"][0, :4], [0, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "truncate"):
            pack_scene(scene, norm, length=2)
        scene["grid"] += 20
        packed = pack_scene(scene, norm)
        self.assertEqual(packed["packing_audit"]["clipped_time_count"], 1)
        with self.assertRaisesRegex(ValueError, "no observed"):
            fit_normalizer([])

    def test_positive_pool_strict_boundaries_original_labels_and_old_young_order(self):
        tracks = [
            ("touch", "a", [200, 400]), ("touch", "a", [0, 200]),
            ("gap-ok", "a", [0, 121]), ("gap-ok", "a", [1320, 1441]),
            ("gap-edge", "a", [0, 200]), ("gap-edge", "a", [1400, 1600]),
            ("duration-edge", "a", [0, 120]), ("duration-edge", "a", [200, 400]),
            ("same-overlap", "a", [0, 400]), ("same-overlap", "a", [200, 600]),
            ("cross-ok", "a", [0, 400]), ("cross-ok", "b", [279, 500]),
            ("cross-edge", "a", [0, 400]), ("cross-edge", "b", [280, 500]),
            ("negative-label", "a", [0, 400]), ("negative-label", "b", [100, 500]),
        ]
        scene = make_scene(tracks)
        negative_pair = np.flatnonzero((scene["pairs"] == [14, 15]).all(axis=1))[0]
        scene["labels"][negative_pair] = 0
        pools, audit = build_positive_pools([scene])
        self.assertEqual(set(pools["ITSA"]), {"touch", "gap-ok"})
        self.assertEqual(pools["ITSA"]["touch"], [(0, 1, 0)])
        self.assertEqual(set(pools["MSTSA"]), {"cross-ok"})
        self.assertEqual(audit["positive_pair_count"], 7)
        self.assertEqual(audit["eligible_pair_counts"], {"ITSA": 2, "MSTSA": 1})
        self.assertEqual(audit["eligible_identity_counts"], {"ITSA": 2, "MSTSA": 1})
        scene["labels"][0] = 0  # Does not create a positive by identity equality alone.
        pools, _ = build_positive_pools([scene])
        self.assertNotIn("touch", pools["ITSA"])

    def test_global_identity_sampling_avoids_false_negatives_and_fails_small_pool(self):
        tracks = [(identity, source, [offset, offset + 200, offset + 400])
                  for identity in ("A-0", "B-0", "C-0")
                  for source, offset in (("a", 0), ("b", 100))]
        scenes = [make_scene(tracks, "s0"), make_scene(tracks, "s1")]
        normalizer = fit_normalizer(scenes)
        packed = [pack_scene(scene, normalizer) for scene in scenes]
        pools, audit = build_positive_pools(scenes)
        self.assertEqual(audit["eligible_pair_counts"]["MSTSA"], 6)
        self.assertEqual(audit["eligible_identity_counts"]["MSTSA"], 3)
        first, second = np.random.default_rng(12), np.random.default_rng(12)
        for _ in range(10):
            batch = sample_batch(packed, pools, "MSTSA", 3, first)
            repeated = sample_batch(packed, pools, "MSTSA", 3, second)
            self.assertEqual(len(set(batch["identities"])), 3)
            np.testing.assert_array_equal(batch["pairs"], repeated["pairs"])
            np.testing.assert_array_equal(batch["positions"], repeated["positions"])
            for row, (scene_index, left, right) in enumerate(batch["pairs"]):
                self.assertEqual(packed[scene_index]["identities"][left], batch["identities"][row])
                self.assertEqual(packed[scene_index]["identities"][right], batch["identities"][row])
            self.assertEqual(batch["positions"].shape, (6, 128, 2))
        with self.assertRaisesRegex(ValueError, "found 3"):
            sample_batch(packed, pools, "MSTSA", 4, first)
        with self.assertRaisesRegex(ValueError, ">= 2"):
            sample_batch(packed, pools, "MSTSA", 1, first)

    def test_mstsa_crop_uses_one_actual_time_window_with_unequal_sampling(self):
        left_times = [0, 50, 100, 150, 200, 250, 300, 350, 400]
        right_times = [25, 100, 175, 250, 325, 400, 475]
        scene = make_scene([(identity, source, times) for identity in ("A-0", "B-0")
                            for source, times in (("a", left_times), ("b", right_times))])
        packed = pack_scene(scene, fit_normalizer([scene]))
        pools, _ = build_positive_pools([scene])
        batch = sample_batch([packed], pools, "MSTSA", 2, MidpointRng())
        self.assertEqual(batch["augmentation"]["fallback_pairs"], 0)
        self.assertEqual(batch["augmentation"]["cropped_pairs"], 2)
        for pair, window in enumerate(batch["crop_windows"]):
            for node in (pair, pair + 2):
                retained = batch["times"][node, batch["observed"][node]]
                self.assertTrue((retained >= window["start"]).all())
                self.assertTrue((retained <= window["stop"]).all())
                np.testing.assert_array_equal(batch["time_ids"][node, batch["observed"][node]],
                                              np.floor(retained / 20))
            np.testing.assert_array_equal(batch["times"][pair, batch["observed"][pair]],
                                          [100, 150, 200, 250, 300, 350])
            np.testing.assert_array_equal(batch["times"][pair + 2, batch["observed"][pair + 2]],
                                          [100, 175, 250, 325])
        for key in ("positions", "times", "time_ids"):
            self.assertTrue((batch[key][~batch["observed"]] == 0).all())

    def test_mstsa_sparse_crop_falls_back_for_both_tracks(self):
        scene = make_scene([(identity, source, times) for identity in ("A-0", "B-0")
                            for source, times in (("a", [0, 500]), ("b", [100, 200, 300, 400, 600]))])
        packed = pack_scene(scene, fit_normalizer([scene]))
        pools, _ = build_positive_pools([scene])
        batch = sample_batch([packed], pools, "MSTSA", 2, MidpointRng())
        self.assertEqual(batch["augmentation"]["fallback_pairs"], 2)
        self.assertEqual(batch["augmentation"]["cropped_pairs"], 0)
        for pair, (_, left, right) in enumerate(batch["pairs"]):
            for destination, original in ((pair, left), (pair + 2, right)):
                np.testing.assert_array_equal(batch["observed"][destination], packed["observed"][original])
                np.testing.assert_array_equal(batch["positions"][destination], packed["positions"][original])
                np.testing.assert_array_equal(batch["times"][destination], packed["times"][original])

    def test_itsa_crops_only_young_tail_by_time_and_keeps_at_least_two_points(self):
        scene = make_scene([("A-0", "a", [0, 100, 200]),
                            ("A-0", "a", [500, 550, 600, 650, 700, 750, 800]),
                            ("B-0", "a", [0, 100, 200]), ("B-0", "a", [500, 800])])
        packed = pack_scene(scene, fit_normalizer([scene]))
        pools, _ = build_positive_pools([scene])
        batch = sample_batch([packed], pools, "ITSA", 2, MidpointRng())
        for node in (0, 1):
            np.testing.assert_array_equal(batch["times"][node, batch["observed"][node]], [0, 100, 200])
        np.testing.assert_array_equal(batch["times"][2, batch["observed"][2]], [500, 550, 600, 650, 700])
        np.testing.assert_array_equal(batch["times"][3, batch["observed"][3]], [500, 800])
        np.testing.assert_array_equal(batch["time_ids"][2, batch["observed"][2]], [25, 27, 30, 32, 35])


if __name__ == "__main__":
    unittest.main()
