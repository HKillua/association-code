"""Explicit offline TSA-cTFER adaptation of the shared registered observations.

This module neither reads archives nor chooses splits. Callers supply training
records to fit moments and build positive pools; validation records are packed
with those frozen moments. The original candidate pairs and labels are retained.
"""
from __future__ import annotations

import numpy as np


TASKS = ("ITSA", "MSTSA")


def _observations(scene):
    values = np.asarray(scene["values"])
    observed = np.asarray(scene["observed"], dtype=bool)
    if values.ndim != 3 or values.shape[-1] != 4 or observed.shape != values.shape[:2]:
        raise ValueError("invalid registered scene values/observed shape")
    return values, observed


def fit_normalizer(scenes):
    """Population lon/lat moments over supplied training observed points only."""
    count = 0
    mean = np.zeros(2, dtype=np.float64)
    m2 = np.zeros(2, dtype=np.float64)
    scene_ids = []
    for scene in scenes:
        values, observed = _observations(scene)
        points = np.asarray(values[observed][:, [1, 0]], dtype=np.float64)
        if not np.isfinite(points).all():
            raise ValueError("nonfinite training longitude/latitude")
        scene_ids.append(str(scene["scene_id"]))
        if not len(points):
            continue
        # Combine centered moments to avoid cancellation around large degree offsets.
        chunk_mean = points.mean(axis=0)
        chunk_m2 = ((points - chunk_mean) ** 2).sum(axis=0)
        delta = chunk_mean - mean
        combined = count + len(points)
        m2 += chunk_m2 + delta ** 2 * (count * len(points) / combined)
        mean += delta * (len(points) / combined)
        count = combined
    if not count:
        raise ValueError("training split contains no observed longitude/latitude")
    std = np.sqrt(np.maximum(m2 / count, 0.0))
    constant = std < 1e-6
    std[constant] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist(), "feature_order": ["lon", "lat"],
            "observed_point_count": count, "training_scene_ids": scene_ids,
            "constant_features": np.flatnonzero(constant).tolist(),
            "policy": "training observed registered lon/lat; population std; std<1e-6 replaced by 1"}


def pack_scene(scene, normalizer, length=128, time_bin_seconds=20.0, time_bins=4096):
    """Pack every registered observation without adding or discarding information.

    Timestamps are scene-relative bins, not token indices. This is an explicit
    offline choice because the paper does not specify its timestamp vocabulary.
    All original metadata and large arrays are shallow references, except the
    packed mask. ``times`` retains each actual observation time, including after
    crop augmentation; padding values, timestamps and IDs are all zero.
    """
    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise ValueError("packing length must be an integer >= 2")
    if isinstance(time_bins, bool) or not isinstance(time_bins, int) or time_bins < 1:
        raise ValueError("time_bins must be a positive integer")
    if not np.isfinite(time_bin_seconds) or time_bin_seconds <= 0:
        raise ValueError("time_bin_seconds must be finite and positive")
    mean, std = (np.asarray(normalizer[key], dtype=np.float64) for key in ("mean", "std"))
    if (mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("invalid longitude/latitude normalizer")
    values, mask = _observations(scene)
    actual_times = np.asarray(scene["observation_times"], dtype=np.float64)
    grid = np.asarray(scene["grid"], dtype=np.float64)
    if actual_times.shape != mask.shape or grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
        raise ValueError("invalid registered scene timestamp shape or grid")
    n = len(values)
    positions = np.zeros((n, length, 2), dtype=np.float32)
    observed = np.zeros((n, length), dtype=bool)
    times = np.zeros((n, length), dtype=np.float64)
    time_ids = np.zeros((n, length), dtype=np.int64)
    clipped = 0
    for node in range(n):
        indices = np.flatnonzero(mask[node])
        if len(indices) < 2:
            raise ValueError("each valid track must contain at least two registered observations")
        if len(indices) > length:
            raise ValueError("packing would truncate registered observations")
        node_times = actual_times[node, indices]
        points = np.asarray(values[node, indices][:, [1, 0]], dtype=np.float64)
        if not np.isfinite(node_times).all() or not np.isfinite(points).all():
            raise ValueError("nonfinite observed coordinates or timestamps")
        order = np.argsort(node_times, kind="stable")
        node_times, points = node_times[order], points[order]
        if np.any(np.diff(node_times) <= 0):
            raise ValueError("actual observation times must be distinct and increasing")
        normalized = (points - mean) / std
        if not np.isfinite(normalized).all() or np.any(np.abs(normalized) > np.finfo(np.float32).max):
            raise ValueError("nonfinite normalized longitude/latitude")
        bins = np.floor((node_times - grid[0]) / time_bin_seconds)
        if not np.isfinite(bins).all():
            raise ValueError("nonfinite timestamp bin")
        clipped += int(np.count_nonzero((bins < 0) | (bins >= time_bins)))
        size = len(indices)
        positions[node, :size] = normalized
        observed[node, :size] = True
        times[node, :size] = node_times
        time_ids[node, :size] = np.clip(bins, 0, time_bins - 1).astype(np.int64)
    packed = dict(scene)
    packed.update(positions=positions, time_ids=time_ids, observed=observed, times=times,
                  packing_audit={"scene_id": str(scene["scene_id"]),
                           "length": length, "source_length": int(values.shape[1]),
                           "valid_points": int(mask.sum()), "clipped_time_count": clipped,
                           "time_bin_seconds": float(time_bin_seconds), "time_bins": time_bins,
                           "time_origin": float(grid[0]), "added_observations": 0,
                           "discarded_observations": 0})
    return packed


def build_positive_pools(scenes):
    """Group eligible positive pairs by global A-B identity for each task.

    Training eligibility follows the paper's duration/gap/overlap conditions.
    This does not change the full validation candidate list. One pair per global
    identity is sampled in a batch so additional same-identity pairs are never
    incorrectly used as contrastive negatives.
    """
    pools = {task: {} for task in TASKS}
    positive_count = 0
    pair_count = 0
    for scene_index, scene in enumerate(scenes):
        identities = np.asarray(scene["identities"], dtype=str)
        sources = np.asarray(scene["sources"], dtype=str)
        intervals = np.asarray(scene["intervals"], dtype=np.float64)
        pairs = np.asarray(scene["pairs"], dtype=np.int64)
        labels = np.asarray(scene["labels"])
        n = len(identities)
        if (sources.shape != (n,) or intervals.shape != (n, 2)
                or pairs.ndim != 2 or pairs.shape[1] != 2 or labels.shape != (len(pairs),)
                or not np.isfinite(intervals).all() or np.any(intervals[:, 1] < intervals[:, 0])
                or np.any((pairs < 0) | (pairs >= n))):
            raise ValueError("invalid positive-pool scene metadata")
        pair_count += len(pairs)
        # Candidate lists may have millions of negatives; Python only visits positives.
        positive_indices = np.flatnonzero(labels == 1)
        positive_count += len(positive_indices)
        for pair_index in positive_indices:
            left, right = map(int, pairs[pair_index])
            if left == right or identities[left] != identities[right]:
                raise ValueError("positive label is inconsistent with distinct A-B identity nodes")
            a, b = intervals[left], intervals[right]
            task = None
            if sources[left] == sources[right]:
                if a[0] > b[0]:
                    left, right, a, b = right, left, b, a
                if a[1] <= b[0] and b[0] - a[1] < 1200 and a[1] - a[0] > 120 and b[1] - b[0] > 120:
                    task = "ITSA"
            elif min(a[1], b[1]) - max(a[0], b[0]) > 120:
                task = "MSTSA"
            if task is not None:
                identity = str(identities[left])
                pools[task].setdefault(identity, []).append((scene_index, left, right))
    audit = {"scene_count": len(scenes), "candidate_pair_count": pair_count,
             "positive_pair_count": positive_count,
             "eligible_pair_counts": {task: sum(map(len, pools[task].values())) for task in TASKS},
             "eligible_identity_counts": {task: len(pools[task]) for task in TASKS},
             "sampling": "uniform global A-B identities without replacement, then uniform eligible pair"}
    return pools, audit


def _copy_tokens(scene, node, keep, arrays, destination):
    count = len(keep)
    for name in ("positions", "time_ids", "times"):
        arrays[name][destination, :count] = scene[name][node, keep]
    arrays["observed"][destination, :count] = True


def sample_batch(packed_scenes, pools, task, batch_pairs, rng, min_crop_fraction=0.5):
    """Sample B distinct identities, then return left[0:B], right[0:B]."""
    if task not in TASKS:
        raise ValueError("unknown TSA task " + str(task))
    if isinstance(batch_pairs, bool) or not isinstance(batch_pairs, int) or batch_pairs < 2:
        raise ValueError("batch_pairs must be an integer >= 2")
    if not np.isfinite(min_crop_fraction) or not 0 < min_crop_fraction <= 1:
        raise ValueError("min_crop_fraction must lie in (0, 1]")
    pool = pools[task]
    identities = sorted(pool)
    if len(identities) < batch_pairs:
        raise ValueError("%s requires %d distinct eligible identities; found %d" %
                         (task, batch_pairs, len(identities)))
    chosen = [identities[int(index)] for index in rng.choice(len(identities), size=batch_pairs, replace=False)]
    references = [pool[identity][int(rng.integers(len(pool[identity])))] for identity in chosen]
    length = packed_scenes[references[0][0]]["positions"].shape[1]
    arrays = {"positions": np.zeros((2 * batch_pairs, length, 2), dtype=np.float32),
              "time_ids": np.zeros((2 * batch_pairs, length), dtype=np.int64),
              "times": np.zeros((2 * batch_pairs, length), dtype=np.float64),
              "observed": np.zeros((2 * batch_pairs, length), dtype=bool)}
    stats = {"pair_count": batch_pairs, "cropped_pairs": 0, "fallback_pairs": 0,
             "input_points": 0, "output_points": 0}
    windows = []
    for pair_number, (scene_index, left, right) in enumerate(references):
        scene = packed_scenes[scene_index]
        if scene["positions"].shape[1:] != (length, 2):
            raise ValueError("packed scenes have inconsistent token lengths")
        left_keep = np.flatnonzero(scene["observed"][left])
        right_keep = np.flatnonzero(scene["observed"][right])
        left_times, right_times = scene["times"][left, left_keep], scene["times"][right, right_keep]
        if len(left_keep) < 2 or len(right_keep) < 2:
            raise ValueError("sampled track has fewer than two observed tokens")
        original_count = len(left_keep) + len(right_keep)
        fraction = float(rng.uniform(min_crop_fraction, 1.0))
        fallback = False
        if task == "ITSA":
            start = float(right_times[0])
            stop = start + fraction * float(right_times[-1] - start)
            cropped_right = right_keep[right_times <= stop]
            # With sparse registered points, keep at least two while retaining a prefix.
            if len(cropped_right) < 2:
                cropped_right = right_keep[:2]
                stop = float(right_times[1])
            right_keep = cropped_right
        else:
            overlap_start = float(max(left_times[0], right_times[0]))
            overlap_stop = float(min(left_times[-1], right_times[-1]))
            if overlap_stop <= overlap_start:
                raise ValueError("MSTSA sampled pair has no actual time overlap")
            duration = (overlap_stop - overlap_start) * fraction
            start = float(rng.uniform(overlap_start, overlap_stop - duration))
            stop = start + duration
            cropped_left = left_keep[(left_times >= start) & (left_times <= stop)]
            cropped_right = right_keep[(right_times >= start) & (right_times <= stop)]
            if len(cropped_left) < 2 or len(cropped_right) < 2:
                fallback = True
            else:
                left_keep, right_keep = cropped_left, cropped_right
        kept_count = len(left_keep) + len(right_keep)
        stats["input_points"] += original_count
        stats["output_points"] += kept_count
        stats["cropped_pairs"] += int(kept_count < original_count)
        stats["fallback_pairs"] += int(fallback)
        windows.append({"start": start, "stop": stop, "fraction": fraction, "fallback": fallback})
        _copy_tokens(scene, left, left_keep, arrays, pair_number)
        _copy_tokens(scene, right, right_keep, arrays, pair_number + batch_pairs)
    arrays.update(identities=np.asarray(chosen, dtype=str), pairs=np.asarray(references, dtype=np.int64),
                  task=task, augmentation=stats, crop_windows=windows)
    return arrays
