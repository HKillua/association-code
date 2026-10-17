"""Balanced training batches drawn from existing, scene-local pair indices.

This is an explicit sampling policy, not an evaluation candidate filter. Callers
must supply only the prepared training scenes for the current fit. The sampler
does not inspect sources/identities, generate pairs, or alter the supplied labels.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


class BalancedScenePairSampler:
    """Choose one scene per class, then half a batch from that scene's class.

    Scene selection is uniform over scenes containing the requested class, not
    weighted by their pair counts. A class is sampled with replacement only if
    its chosen scene has fewer pairs than half the requested batch size.

    Returned indices address the original ``scenes[scene_index]['pairs']`` and
    ``labels`` arrays. The caller must keep their order and contents unchanged
    after construction. Only class-index pools and compact label metadata are
    retained; observation arrays are neither copied nor read.
    """

    def __init__(self, scenes, seed):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("sampler seed must be a nonnegative integer")
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._scene_ids = []
        self._labels = []
        self._pools = {1: [], 0: []}
        for scene_index, scene in enumerate(scenes):
            scene_id = str(scene["scene_id"])
            pairs = np.asarray(scene["pairs"])
            labels = np.asarray(scene["labels"])
            if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
                raise ValueError(f"scene {scene_id}: pairs must be an integer array of shape (N, 2)")
            if (pairs < 0).any():
                raise ValueError(f"scene {scene_id}: pair endpoints must be nonnegative")
            if labels.ndim != 1 or len(labels) != len(pairs):
                raise ValueError(f"scene {scene_id}: labels must have one entry per pair")
            if labels.dtype.kind not in "biuf" or not ((labels == 0) | (labels == 1)).all():
                raise ValueError(f"scene {scene_id}: labels must be binary 0/1")
            self._scene_ids.append(scene_id)
            self._labels.append(labels.astype(bool, copy=True))
            for label in (1, 0):
                self._pools[label].append(np.flatnonzero(labels == label).astype(np.int64, copy=False))
        self._eligible = {
            label: np.asarray([index for index, pool in enumerate(pools) if len(pool)], dtype=np.int64)
            for label, pools in self._pools.items()
        }
        missing = [name for label, name in ((1, "positive"), (0, "negative"))
                   if not len(self._eligible[label])]
        if missing:
            raise ValueError("training scenes contain no " + " or ".join(missing) + " pairs")

    def sample(self, batch_size):
        """Return at most two ``(scene_index, int64 pair_indices)`` groups.

        The total is exactly ``batch_size``, half positive and half negative.
        Positive sampling precedes negative sampling. Selections sharing a scene
        are merged, each group's indices are shuffled, then groups are shuffled.
        The same seed, scene order, and call sequence reproduce all these orders.
        Each class selects its scene independently; no draw spans scenes.
        """
        if (isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral)
                or batch_size < 2 or batch_size % 2):
            raise ValueError("batch_size must be an even integer >= 2")
        half = int(batch_size) // 2
        selected = {}
        for label in (1, 0):
            scene_index = int(self._rng.choice(self._eligible[label]))
            pool = self._pools[label][scene_index]
            indices = self._rng.choice(pool, size=half, replace=len(pool) < half)
            selected.setdefault(scene_index, []).append(indices)
        groups = []
        for scene_index, chunks in selected.items():
            indices = np.concatenate(chunks).astype(np.int64, copy=False)
            groups.append((scene_index, self._rng.permutation(indices)))
        return [groups[int(index)] for index in self._rng.permutation(len(groups))]

    def describe(self):
        """Return JSON-compatible pool counts and the exact selection policy.

        Counts include all provided scenes, including zero-pair and single-class
        scenes. This method does not advance the sampler's random state.
        """
        scenes = []
        for index, scene_id in enumerate(self._scene_ids):
            positive = len(self._pools[1][index])
            negative = len(self._pools[0][index])
            scenes.append({"scene_index": index, "scene_id": scene_id,
                           "pair_count": positive + negative,
                           "positive_pair_count": positive, "negative_pair_count": negative})
        return {
            "strategy": "balanced_scene_then_pair_v1", "seed": self._seed,
            "scene_count": len(scenes), "pair_count": sum(row["pair_count"] for row in scenes),
            "positive_pair_count": sum(row["positive_pair_count"] for row in scenes),
            "negative_pair_count": sum(row["negative_pair_count"] for row in scenes),
            "eligible_scene_ids": {
                name: [self._scene_ids[int(index)] for index in self._eligible[label]]
                for label, name in ((1, "positive"), (0, "negative"))
            },
            "eligible_scene_indices": {
                name: self._eligible[label].tolist()
                for label, name in ((1, "positive"), (0, "negative"))
            },
            "zero_pair_scene_ids": [row["scene_id"] for row in scenes if not row["pair_count"]],
            "selection_policy": {
                "class_counts": "exactly half positive and half negative; even batch_size >= 2",
                "scene": "independent uniform choice of one eligible scene for each class",
                "pair": "uniform choice among original pair indices of that class in the chosen scene",
                "replacement": "with replacement iff the chosen class pool is smaller than half the batch",
                "ordering": "positive draw, negative draw; merge by scene; shuffle indices then groups",
                "labels": "unchanged prepared labels; no source, identity, or temporal filtering",
            },
            "scenes": scenes,
        }

    def batch_metadata(self, groups):
        """Count a sampled batch, with scene-local duplicate indices recorded.

        ``duplicate_pair_count`` counts extra occurrences, not the number of
        distinct indices repeated. A pair index in a different scene is a
        different pair. This method does not advance the random state.
        """
        merged = {}
        for scene_index, indices in groups:
            if (isinstance(scene_index, (bool, np.bool_)) or not isinstance(scene_index, Integral)
                    or not 0 <= scene_index < len(self._scene_ids)):
                raise ValueError("batch group contains an invalid scene_index")
            scene_index = int(scene_index)
            indices = np.asarray(indices)
            if indices.ndim != 1 or indices.dtype.kind not in "iu":
                raise ValueError("pair_indices must be a one-dimensional integer array")
            if ((indices < 0) | (indices >= len(self._labels[scene_index]))).any():
                raise ValueError("batch group contains an out-of-range pair index")
            merged.setdefault(scene_index, []).append(indices)
        rows = []
        for scene_index, chunks in merged.items():
            indices = np.concatenate(chunks)
            unique, counts = np.unique(indices, return_counts=True)
            positive = int(self._labels[scene_index][indices].sum())
            repeated = counts > 1
            rows.append({
                "scene_index": scene_index, "scene_id": self._scene_ids[scene_index],
                "pair_count": len(indices), "positive_count": positive,
                "negative_count": len(indices) - positive, "unique_pair_count": len(unique),
                "duplicate_pair_count": len(indices) - len(unique),
                "duplicate_pair_indices": unique[repeated].tolist(),
                "duplicate_pair_counts": counts[repeated].tolist(),
            })
        return {
            **{key: sum(row[key] for row in rows)
               for key in ("pair_count", "positive_count", "negative_count",
                           "unique_pair_count", "duplicate_pair_count")},
            "scene_ids": [row["scene_id"] for row in rows], "groups": rows,
        }
