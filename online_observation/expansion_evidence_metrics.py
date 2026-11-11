"""Score frozen 4–8 km expansion evidence after replay."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from math import isfinite


POPULATIONS = ("all_open_events", "wide_events", "first_open_per_pair", "first_wide_per_pair")
TRUTH_GROUPS = ("true", "false_both_seen", "false_unseen_truth")
POSITION_BOUNDS = (50, 100, 200, 400, 800, 1200, 2000, 4000, 5000)
TIME_BOUNDS = (0, 10, 20, 40, 60, 120, 240)
FEATURE_BOUNDS = {
    "range_m": POSITION_BOUNDS,
    "past_drift_m": POSITION_BOUNDS,
    "held_residual_m": POSITION_BOUNDS,
    "witness_count": (0, 1, 2, 3, 4, 8),
    "witness_residual_m": POSITION_BOUNDS,
    "repeat_offset_drift_m": POSITION_BOUNDS,
    "repeat_both": (0, 1),
    "episode_age_s": TIME_BOUNDS,
    "singleton": (0, 1),
}
JOINT_SUPPORT = ("repeat_both", "singleton_and_repeat_both", "past_drift_le_400_and_repeat_both",
                 "repeat_offset_drift_le_400_and_repeat_both")
SHORT_REVISIT_SECONDS = 20
COUNT_FIELDS = (
    "edges", "first", "repeated", "short_revisit", "witness_true_edges", "witness_false_edges",
    *(f"{name}_{n}" for name in ("seen_truth_ends", "fresh_truth_ends", "open_truth_ends", "sparse_ends", "updated_ends")
      for n in range(3)),
    *(f"witness_truth_{status}" for status in ("missing", "all_true", "mixed", "all_false")),
)
_BIN_NAMES = {field: tuple(f"le_{value}" for value in bounds) + ("above",)
              for field, bounds in FEATURE_BOUNDS.items()}


def _empty():
    return {
        "events": 0,
        "populations": {
            population: {
                truth: {"counts": Counter(dict.fromkeys(COUNT_FIELDS, 0)),
                        "histograms": {field: Counter() for field in FEATURE_BOUNDS},
                        "joint_support": {condition: Counter() for condition in JOINT_SUPPORT}}
                for truth in TRUTH_GROUPS
            } for population in POPULATIONS
        },
    }


def _bin(field, value):
    if value is None:
        return "missing"
    if not isfinite(value) or value < 0:
        raise ValueError(f"invalid nonnegative expansion evidence: {field}")
    if field in ("singleton", "repeat_both") and value not in (0, 1):
        raise ValueError(f"invalid binary expansion evidence: {field}")
    return _BIN_NAMES[field][bisect_left(FEATURE_BOUNDS[field], value)]


def _joint_status(*states):
    if False in states:
        return "no"
    return "missing" if None in states else "yes"


def _joint_features(features):
    repeat = features.get("repeat_both")
    singleton = features.get("singleton")
    past = features.get("past_drift_m")
    drift = features.get("repeat_offset_drift_m")
    return {
        "repeat_both": _joint_status(repeat),
        "singleton_and_repeat_both": _joint_status(singleton, repeat),
        "past_drift_le_400_and_repeat_both": _joint_status(None if past is None else past <= 400, repeat),
        "repeat_offset_drift_le_400_and_repeat_both": _joint_status(None if drift is None else drift <= 400, repeat),
    }


def _record(group, stamp, previous, ends, bins, joints, witness_truth):
    counts = group["counts"]
    counts["edges"] += 1
    counts["first" if previous is None else "repeated"] += 1
    counts["short_revisit"] += previous is not None and stamp - previous <= SHORT_REVISIT_SECONDS
    for name, value in ends.items():
        counts[f"{name}_{value}"] += 1

    witness_count, true_count = witness_truth
    counts["witness_true_edges"] += true_count
    counts["witness_false_edges"] += witness_count - true_count
    status = ("missing" if not witness_count else "all_true" if true_count == witness_count else
              "mixed" if true_count else "all_false")
    counts[f"witness_truth_{status}"] += 1
    for field, bin_name in bins.items():
        group["histograms"][field][bin_name] += 1
    for condition, outcome in joints.items():
        group["joint_support"][condition][outcome] += 1


def score_expansion_evidence(frames, labels):
    """Grade candidate snapshots using only truth counterparts already arrived.

    Each frame supplies a stamp, updated and fresh fragment keys, and open
    edges. Each edge supplies its two keys, W nomination, recorded witnesses,
    and the feature mapping constructed before labels were read. Fragment keys
    and exact feature values are kept out of the returned result.
    """
    result = _empty()
    seen_keys = set()
    seen_sources = defaultdict(set)
    last_open, last_wide = {}, {}
    previous_stamp = None
    for frame in frames:
        stamp = frame.stamp
        if not isfinite(stamp) or previous_stamp is not None and stamp <= previous_stamp:
            raise ValueError("expansion evidence frames must use increasing finite times")
        previous_stamp = stamp
        result["events"] += 1
        updated, fresh = set(frame.updated), set(frame.fresh)
        seen_keys.update(updated)
        if not updated <= fresh <= seen_keys:
            raise ValueError("fresh keys must come from the received expansion evidence prefix")
        for key in updated:
            seen_sources[labels[key]].add(key[0])

        fresh_counts = Counter(key[0] for key in fresh)
        fresh_labels = defaultdict(set)
        for key in fresh:
            fresh_labels[key[0]].add(labels[key])
        open_truth_ends = {key for edge in frame.edges if labels[edge.pair[0]] == labels[edge.pair[1]]
                           for key in edge.pair}
        current = set()
        for edge in frame.edges:
            left, right = pair = tuple(sorted(edge.pair))
            if left[0] == right[0] or not {left, right} <= fresh or pair in current:
                raise ValueError("each open expansion edge must be fresh, cross-source, and unique in its frame")
            current.add(pair)
            left_label, right_label = labels[left], labels[right]
            history = int(right[0] in seen_sources[left_label]) + int(left[0] in seen_sources[right_label])
            truth = ("true" if left_label == right_label else "false_both_seen" if history == 2
                     else "false_unseen_truth")
            ends = dict(
                seen_truth_ends=history,
                fresh_truth_ends=int(left_label in fresh_labels[right[0]]) + int(right_label in fresh_labels[left[0]]),
                open_truth_ends=int(left in open_truth_ends) + int(right in open_truth_ends),
                sparse_ends=int(len(fresh) - fresh_counts[left[0]] <= 3)
                + int(len(fresh) - fresh_counts[right[0]] <= 3),
                updated_ends=int(left in updated) + int(right in updated),
            )
            features = edge.features
            bins = {field: _bin(field, features.get(field)) for field in FEATURE_BOUNDS}
            joints = _joint_features(features)
            witnesses = edge.witnesses
            witness_truth = len(witnesses), sum(labels[a] == labels[b] for a, b in witnesses)
            last = last_open.get(pair)
            groups = [("all_open_events", last)]
            if last is None:
                groups.append(("first_open_per_pair", None))
            last_open[pair] = stamp
            if edge.wide:
                last = last_wide.get(pair)
                groups.append(("wide_events", last))
                if last is None:
                    groups.append(("first_wide_per_pair", None))
                last_wide[pair] = stamp
            for population, previous in groups:
                group = result["populations"][population][truth]
                _record(group, stamp, previous, ends, bins, joints, witness_truth)
    return result


def pool_expansion_evidence(results):
    """Pool scene counters without collapsing fragment keys across scenes."""
    merged = _empty()
    for scene in results:
        merged["events"] += scene["events"]
        for population in POPULATIONS:
            for truth in TRUTH_GROUPS:
                target = merged["populations"][population][truth]
                source = scene["populations"][population][truth]
                target["counts"].update(source["counts"])
                for family in ("histograms", "joint_support"):
                    for name, counts in source[family].items():
                        target[family][name].update(counts)
    return merged
