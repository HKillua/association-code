"""Fixed-population edge metrics and paired component uncertainty (NumPy only).

Thresholds predict score >= threshold. ``None`` is the serializable all-reject
threshold, ordered above every finite score. No test-side threshold fitting is
performed implicitly by any evaluation function.
"""

from __future__ import annotations

import numpy as np


def _records(records):
    normalized = []
    seen = set()
    for record in records:
        scene_id = str(record["scene_id"])
        if scene_id in seen:
            raise ValueError("duplicate scene_id")
        seen.add(scene_id)
        scores = np.asarray(record["scores"], dtype=np.float64)
        labels = np.asarray(record["labels"])
        pairs = np.asarray(record["pairs"])
        sources = np.asarray(record["sources"])
        if scores.ndim != 1 or labels.shape != scores.shape or pairs.shape != (len(scores), 2):
            raise ValueError("record needs scores[E], labels[E], pairs[E,2]")
        if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("scores must be finite and labels binary")
        if sources.ndim != 1 or not np.issubdtype(pairs.dtype, np.integer):
            raise ValueError("sources must be a vector and pairs integer")
        if len(pairs) and ((pairs < 0).any() or (pairs >= len(sources)).any() or
                           (pairs[:, 0] >= pairs[:, 1]).any()):
            raise ValueError("pairs must satisfy 0 <= i < j < node count")
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError("duplicate pairs")
        normalized.append({**record, "scene_id": scene_id, "scores": scores,
                           "labels": labels.astype(bool), "pairs": pairs,
                           "sources": sources,
                           "component_id": str(record.get("component_id", scene_id))})
    return normalized


def predict(scores, threshold):
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if threshold is None:
        return np.zeros(values.shape, dtype=bool)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite or None (all reject)")
    return values >= threshold


def _counts(labels, predictions):
    tp = int(np.count_nonzero(labels & predictions))
    fp = int(np.count_nonzero(~labels & predictions))
    fn = int(np.count_nonzero(labels & ~predictions))
    tn = int(np.count_nonzero(~labels & ~predictions))
    denominator = 2 * tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1": 2 * tp / denominator if denominator else 1.0,
            "precision": precision, "recall": recall, "p_ca": recall,
            "p_fa": 1 - precision if precision is not None else None}


def _paper_metrics(active):
    """Scene averages; undefined denominators never silently remove a scene."""
    recalls = [row["recall"] for row in active if row["recall"] is not None]
    false_acceptances = [row["p_fa"] for row in active if row["p_fa"] is not None]
    defined_means = {
        "m_pca": float(np.mean(recalls)) if recalls else None,
        "m_pfa": float(np.mean(false_acceptances)) if false_acceptances else None,
    }
    weighted = [row for row in active if row["tp"] + row["fp"] > 0]
    prediction_count = sum(row["tp"] + row["fp"] for row in weighted)
    weighted_recall = None
    if prediction_count and all(row["recall"] is not None for row in weighted):
        weighted_recall = sum((row["tp"] + row["fp"]) * row["recall"]
                              for row in weighted) / prediction_count
    return {
        "unit": "fraction", "eligible_scene_count": len(active),
        "m_pca": defined_means["m_pca"] if len(recalls) == len(active) else None,
        "m_pfa": defined_means["m_pfa"] if len(false_acceptances) == len(active) else None,
        "mw_pca": weighted_recall,
        "mw_pfa": sum(row["fp"] for row in weighted) / prediction_count if prediction_count else None,
        "defined_scene_count": {"m_pca": len(recalls), "m_pfa": len(false_acceptances)},
        "defined_only_mean": defined_means,
        "predicted_positive_count": prediction_count,
        "prediction_weighted_scene_count": len(weighted),
        "definition": "eligible: pair_count > 0; m: equal scene weights; "
                      "PCA=TP/(TP+FN), PFA=FP/(TP+FP); mw: weights TP+FP; "
                      "any undefined positive-weight term makes the population metric null",
    }


def _aggregate(rows):
    active = [row for row in rows if row["pair_count"] > 0]
    tp, fp, fn, tn = (sum(row[key] for row in active) for key in ("tp", "fp", "fn", "tn"))
    denominator = 2 * tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    positives = [row["f1"] for row in active if row["positive_count"]]
    negatives = [row for row in active if not row["positive_count"]]
    return {"scene_count": len(active), "pair_count": sum(row["pair_count"] for row in active),
            "macro_f1": float(np.mean([row["f1"] for row in active])) if active else None,
            "positive_scene_macro_f1": float(np.mean(positives)) if positives else None,
            "positive_scene_count": len(positives), "negative_scene_count": len(negatives),
            "negative_scene_false_acceptance_rate":
                sum(row["fp"] > 0 for row in negatives) / len(negatives) if negatives else None,
            "micro_f1": 2 * tp / denominator if denominator else (1.0 if active else None),
            "micro_precision": precision, "micro_recall": recall,
            "p_ca": recall, "p_fa": 1 - precision if precision is not None else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "paper_metrics": _paper_metrics(active)}


def evaluate_scenes(records, threshold):
    """Return overall, ITSA/MSTSA and all-scene rows without prediction filtering."""
    records = _records(records)
    rows, same_rows, cross_rows = [], [], []
    for record in records:
        labels, pairs = record["labels"], record["pairs"]
        decisions = predict(record["scores"], threshold)
        same = record["sources"][pairs[:, 0]] == record["sources"][pairs[:, 1]]
        for mask, destination in ((np.ones(len(labels), dtype=bool), rows),
                                   (same, same_rows), (~same, cross_rows)):
            row = {"scene_id": record["scene_id"], "component_id": record["component_id"],
                   "node_count": len(record["sources"]), "pair_count": int(mask.sum()),
                   "positive_count": int(labels[mask].sum()), **_counts(labels[mask], decisions[mask])}
            if not row["pair_count"]:
                row["f1"] = None
            destination.append(row)
    return {**_aggregate(rows), "threshold": threshold, "per_scene": rows,
            "itsa": {**_aggregate(same_rows), "per_scene": same_rows},
            "mstsa": {**_aggregate(cross_rows), "per_scene": cross_rows},
            "coverage": {"all_scenes": len(records), "eligible_scenes": sum(bool(len(r["scores"])) for r in records),
                         "scenes_without_pairs": sum(not len(r["scores"]) for r in records)},
            "empty_f1_convention": "truth and prediction empty: 1; no eligible pairs: excluded from shared denominator"}


def select_threshold(records):
    """Exact validation macro-F1 sweep, O(E log E); ties prefer higher threshold.

    All equal-score edges are added together, so the sweep agrees exactly with
    >= semantics. The all-reject option wins any tie. No rounding/grid search.
    """
    records = _records(records)
    active = [record for record in records if len(record["scores"])]
    if not active:
        raise ValueError("threshold selection requires eligible validation pairs")
    scores = np.concatenate([record["scores"] for record in active])
    labels = np.concatenate([record["labels"] for record in active])
    scene = np.concatenate([np.full(len(record["scores"]), i, dtype=np.int64)
                            for i, record in enumerate(active)])
    truth_count = np.array([record["labels"].sum() for record in active], dtype=np.int64)
    initial_f1 = (truth_count == 0).astype(np.float64)
    # Group by scene then decreasing score. Cumulative TP/predicted-positive
    # counts determine each scene's F1 at every threshold it actually changes.
    # This is fully vectorized: a million distinct scores do not cause a
    # million Python/NumPy calls during every validation epoch.
    order = np.lexsort((-scores, scene))
    ordered_scores, ordered_labels, ordered_scenes = scores[order], labels[order], scene[order]
    starts = np.r_[0, np.flatnonzero((ordered_scenes[1:] != ordered_scenes[:-1]) |
                                     (ordered_scores[1:] != ordered_scores[:-1])) + 1]
    group_scenes, group_scores = ordered_scenes[starts], ordered_scores[starts]
    group_count = np.diff(np.r_[starts, len(scores)])
    group_positive = np.add.reduceat(ordered_labels.astype(np.int64), starts)
    scene_starts = np.r_[0, np.flatnonzero(group_scenes[1:] != group_scenes[:-1]) + 1]
    groups_per_scene = np.diff(np.r_[scene_starts, len(starts)])
    all_tp, all_predictions = np.cumsum(group_positive), np.cumsum(group_count)
    tp = all_tp - np.repeat(all_tp[scene_starts] - group_positive[scene_starts], groups_per_scene)
    prediction_count = all_predictions - np.repeat(
        all_predictions[scene_starts] - group_count[scene_starts], groups_per_scene)
    f1 = 2 * tp / (prediction_count + truth_count[group_scenes])
    previous_f1 = np.r_[0., f1[:-1]]
    previous_f1[scene_starts] = initial_f1[group_scenes[scene_starts]]
    delta_f1 = f1 - previous_f1
    unique_scores, score_index = np.unique(group_scores, return_inverse=True)
    changes = np.bincount(score_index, weights=delta_f1, minlength=len(unique_scores))[::-1]
    macro_values = np.r_[initial_f1.sum(), initial_f1.sum() + np.cumsum(changes)] / len(active)
    maximum = macro_values.max()
    # Numerically identical macro values retain the earlier (higher) threshold.
    best_index = int(np.flatnonzero(macro_values >= maximum - 1e-12)[0])
    best_threshold = None if best_index == 0 else float(unique_scores[::-1][best_index - 1])
    return {"threshold": best_threshold, "macro_f1": float(macro_values[best_index]),
            "candidate_count": int(len(unique_scores) + 1), "scene_count": len(active),
            "tie_break": "higher threshold, all-reject above all finite scores"}


def _components(n, pairs, positive):
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs[positive]:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(n)])


def grouping_diagnostics(record, threshold):
    """Common connected-component sensitivity probe, including singleton nodes.

    Mixed/split identities are inferred from the complete positive truth graph;
    callers using incomplete eligibility masks must not call them ground-truth
    identity counts. Closure pair errors are measured on the supplied pair set.
    """
    record = _records([record])[0]
    pairs, labels = record["pairs"], record["labels"]
    decisions = predict(record["scores"], threshold)
    n = len(record["sources"])
    groups = _components(n, pairs, decisions)
    truth = _components(n, pairs, labels)
    closure = groups[pairs[:, 0]] == groups[pairs[:, 1]]
    mixed = sum(len(np.unique(truth[groups == group])) > 1 for group in np.unique(groups))
    fragmented = sum(len(np.unique(groups[truth == group])) > 1 for group in np.unique(truth))
    return {"scene_id": record["scene_id"], "component_count": len(np.unique(groups)),
            "singleton_count": sum(np.count_nonzero(groups == g) == 1 for g in np.unique(groups)),
            "mixed_identity_components": int(mixed), "fragmented_truth_identities": int(fragmented),
            "false_bridge_edges": int(np.count_nonzero(decisions & ~labels)),
            "closure": _counts(labels, closure),
            "added_closure_fp": int(np.count_nonzero(closure & ~decisions & ~labels)),
            "groups": groups.tolist(), "complete_pair_graph": len(pairs) == n * (n - 1) // 2}


def _aligned(baseline_records, candidate_records):
    baseline, candidate = _records(baseline_records), _records(candidate_records)
    right = {r["scene_id"]: r for r in candidate}
    if {r["scene_id"] for r in baseline} != set(right):
        raise ValueError("paired comparisons require identical scene populations")
    result = []
    for left in baseline:
        other = right[left["scene_id"]]
        if (left["component_id"] != other["component_id"] or
                not np.array_equal(left["pairs"], other["pairs"]) or
                not np.array_equal(left["labels"], other["labels"]) or
                not np.array_equal(left["sources"], other["sources"])):
            raise ValueError("paired comparisons require identical pairs, labels, sources and components")
        result.append((left, other))
    return result


def compare_predictions(baseline_records, candidate_records, baseline_threshold, candidate_threshold):
    counts = {"wrong_to_right": 0, "right_to_wrong": 0, "corrected_fp": 0,
              "corrected_fn": 0, "introduced_fp": 0, "introduced_fn": 0}
    rows = []
    for base, candidate in _aligned(baseline_records, candidate_records):
        old = predict(base["scores"], baseline_threshold)
        new = predict(candidate["scores"], candidate_threshold)
        labels = base["labels"]
        fixed, broken = (old != labels) & (new == labels), (old == labels) & (new != labels)
        row = {"scene_id": base["scene_id"],
               "wrong_to_right": int(fixed.sum()), "right_to_wrong": int(broken.sum()),
               "corrected_fp": int(np.count_nonzero(fixed & ~labels)),
               "corrected_fn": int(np.count_nonzero(fixed & labels)),
               "introduced_fp": int(np.count_nonzero(broken & ~labels)),
               "introduced_fn": int(np.count_nonzero(broken & labels))}
        for key in counts:
            counts[key] += row[key]
        rows.append(row)
    return {**counts, "net_corrected_edges": counts["wrong_to_right"] - counts["right_to_wrong"],
            "per_scene": rows}


def paired_cluster_bootstrap(baseline_records, candidate_records, baseline_threshold,
                            candidate_threshold, n_bootstrap=2000, seed=0, min_clusters=10):
    """Scene-weighted macro difference, resampling shared-A components jointly.

    The same cluster multiplicities apply to both arms. A component with more
    scenes retains that weight in each replicate; edges are never resampled as
    independent observations. Fewer than min_clusters returns descriptive only.
    """
    if n_bootstrap < 1 or min_clusters < 1:
        raise ValueError("bootstrap count and min_clusters must be positive")
    baseline_records, candidate_records = list(baseline_records), list(candidate_records)
    if any("component_id" not in record or record["component_id"] is None
           for record in baseline_records + candidate_records):
        raise ValueError("bootstrap requires explicit shared-identity component_id for every scene")
    aligned = _aligned(baseline_records, candidate_records)
    groups = {}
    for base, candidate in aligned:
        if not len(base["scores"]):
            continue
        old = _counts(base["labels"], predict(base["scores"], baseline_threshold))["f1"]
        new = _counts(candidate["labels"], predict(candidate["scores"], candidate_threshold))["f1"]
        groups.setdefault(base["component_id"], []).append(new - old)
    group_ids = sorted(groups)
    sums = np.array([sum(groups[g]) for g in group_ids])
    sizes = np.array([len(groups[g]) for g in group_ids])
    effect = float(sums.sum() / sizes.sum()) if len(sizes) else None
    result = {"difference": effect, "difference_pp": 100 * effect if effect is not None else None,
              "cluster_count": len(groups), "scene_count": int(sizes.sum()),
              "seed": int(seed), "replicates": 0, "ci95": None, "ci95_pp": None,
              "status": "descriptive_insufficient_clusters", "per_cluster": [
                  {"component_id": g, "scene_count": int(sizes[i]),
                   "mean_difference": float(sums[i] / sizes[i])} for i, g in enumerate(group_ids)]}
    if len(groups) < min_clusters:
        return result
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for replicate in range(n_bootstrap):
        selected = rng.integers(0, len(groups), size=len(groups))
        draws[replicate] = sums[selected].sum() / sizes[selected].sum()
    ci = np.quantile(draws, [0.025, 0.975])
    result.update({"ci95": ci.tolist(), "ci95_pp": (100 * ci).tolist(),
                   "replicates": int(n_bootstrap), "status": "exploratory_cluster_bootstrap"})
    return result


def _interval_union_length(intervals):
    """Lebesgue length of observed-support envelopes, including disjoint gaps."""
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    if not ordered:
        return 0.0
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def third_node_diagnostics(records, threshold, max_group_details=200):
    """Post-hoc joint-error/support audit of *frozen calibrated A0 logits*.

    Records additionally require intervals[N,2], identities[N], node_ids[N].
    Each target edge selects the common k maximizing sigmoid(z_ik)*sigmoid(z_jk),
    with stable node-ID ties. Labels/identities are used only after this choice.
    Same-scene identity pairs define dependent reporting groups, not independent
    observations. No correlation coefficient or causal interpretation is inferred
    from K<=1 (or any other count). Interval union measures support *envelopes*,
    not actual measurement density or statistically independent observations.
    """
    if not isinstance(max_group_details, int) or max_group_details < 0:
        raise ValueError("max_group_details must be a nonnegative integer")
    records = _records(records)
    counter_keys = ("edge_count", "selected_count", "k_zero_count", "k_le_one_count",
                    "both_neighbor_wrong", "one_neighbor_wrong", "neither_neighbor_wrong",
                    "third_new_source_count", "positive_additional_interval_count")

    def empty():
        return {**{key: 0 for key in counter_keys}, "additional_interval_sum": 0.0}

    def add(destination, item):
        for key in counter_keys:
            destination[key] += item[key]
        destination["additional_interval_sum"] += item["additional_interval_sum"]

    def finish(item):
        result = dict(item)
        denominator = item["selected_count"]
        result["additional_interval_mean"] = item["additional_interval_sum"] / denominator if denominator else None
        result["both_neighbor_wrong_fraction"] = item["both_neighbor_wrong"] / denominator if denominator else None
        result["one_neighbor_wrong_fraction"] = item["one_neighbor_wrong"] / denominator if denominator else None
        result["neither_neighbor_wrong_fraction"] = item["neither_neighbor_wrong"] / denominator if denominator else None
        return result

    total = empty()
    by_label = {"positive": empty(), "negative": empty()}
    conditions, groups = {}, {}
    for record in records:
        n = len(record["sources"])
        intervals = np.asarray(record.get("intervals"), dtype=float)
        identities, node_ids = record.get("identities"), record.get("node_ids")
        if intervals.shape != (n, 2) or not np.isfinite(intervals).all() or (intervals[:, 1] < intervals[:, 0]).any():
            raise ValueError("third-node audit requires finite ordered intervals[N,2]")
        if identities is None or node_ids is None or len(identities) != n or len(node_ids) != n:
            raise ValueError("third-node audit requires identities[N] and node_ids[N]")
        stable_ids = [str(value) for value in node_ids]
        if len(set(stable_ids)) != n:
            raise ValueError("node_ids must be unique within a scene")
        scores = record["scores"]
        exp_negative = np.exp(-np.abs(scores))
        probabilities = np.where(scores >= 0, 1 / (1 + exp_negative),
                                  exp_negative / (1 + exp_negative))
        q = np.zeros((n, n))
        mask, truth, decisions = (np.zeros((n, n), dtype=bool) for _ in range(3))
        i_all, j_all = record["pairs"].T
        q[i_all, j_all] = q[j_all, i_all] = probabilities
        mask[i_all, j_all] = mask[j_all, i_all] = True
        truth[i_all, j_all] = truth[j_all, i_all] = record["labels"]
        edge_decisions = predict(scores, threshold)
        decisions[i_all, j_all] = decisions[j_all, i_all] = edge_decisions
        for (i, j), label in zip(record["pairs"], record["labels"]):
            label_name = "positive" if label else "negative"
            pair_source = "same_source" if record["sources"][i] == record["sources"][j] else "cross_source"
            identity_pair = tuple(sorted((str(identities[i]), str(identities[j]))))
            group_key = (record["scene_id"], *identity_pair)
            if group_key not in groups:
                groups[group_key] = {**empty(), "scene_id": record["scene_id"],
                                     "identity_pair": list(identity_pair),
                                     "target_positive_edges": 0, "target_negative_edges": 0}
            groups[group_key]["target_positive_edges" if label else "target_negative_edges"] += 1
            neighbors = np.flatnonzero(mask[i] & mask[j])
            item = empty()
            item["edge_count"] = 1
            item["k_zero_count"] = int(len(neighbors) == 0)
            item["k_le_one_count"] = int(len(neighbors) <= 1)
            third_source = "no_common_neighbor"
            if len(neighbors):
                support = q[i, neighbors] * q[j, neighbors]
                maximum = support.max()
                tied = neighbors[support == maximum]
                k = min(tied, key=lambda index: stable_ids[index])
                wrong = int(decisions[i, k] != truth[i, k]) + int(decisions[j, k] != truth[j, k])
                category = ("neither_neighbor_wrong", "one_neighbor_wrong", "both_neighbor_wrong")[wrong]
                new_source = bool(record["sources"][k] != record["sources"][i] and
                                  record["sources"][k] != record["sources"][j])
                extra = max(0.0, _interval_union_length(intervals[[i, j, k]]) -
                            _interval_union_length(intervals[[i, j]]))
                item.update({"selected_count": 1, category: 1,
                             "third_new_source_count": int(new_source),
                             "positive_additional_interval_count": int(extra > 0),
                             "additional_interval_sum": extra})
                third_source = "new_source" if new_source else "endpoint_source"
            condition_key = (label_name, pair_source, third_source)
            if condition_key not in conditions:
                conditions[condition_key] = {**empty(), "target_label": label_name,
                                              "pair_source": pair_source, "third_source": third_source}
            for destination in (total, by_label[label_name], groups[group_key], conditions[condition_key]):
                add(destination, item)
    ordered_groups = [finish(groups[key]) for key in sorted(groups)]
    selected_groups = [group for group in ordered_groups if group["selected_count"]]
    group_means = {}
    for key in ("both_neighbor_wrong_fraction", "one_neighbor_wrong_fraction",
                "neither_neighbor_wrong_fraction", "additional_interval_mean"):
        group_means[key] = float(np.mean([group[key] for group in selected_groups])) if selected_groups else None
    return {"threshold": threshold, "overall": finish(total),
            "by_target_label": {key: finish(value) for key, value in by_label.items()},
            "by_condition": [finish(conditions[key]) for key in sorted(conditions)],
            "identity_pair_group_count": len(groups), "selected_identity_pair_group_count": len(selected_groups),
            "identity_pair_macro": group_means,
            "identity_pair_groups": ordered_groups[:max_group_details],
            "omitted_group_details": max(0, len(groups) - max_group_details),
            "selection": "highest original sigmoid(z_ik)*sigmoid(z_jk), stable node_id tie; no truth selection",
            "group_unit": "scene plus unordered identity pair; edges within groups are dependent",
            "support_unit": "input interval units; union of temporal support envelopes, not measurement density",
            "inference": "descriptive joint-error counts; no independence, correlation or causality estimate"}


def brier_scores(records):
    """Brier error for records whose scores are calibrated logits, not distances."""
    records = _records(records)
    rows = []
    squared_sum, edge_count = 0.0, 0
    for record in records:
        if not len(record["scores"]):
            continue
        z = record["scores"]
        e = np.exp(-np.abs(z))
        q = np.where(z >= 0, 1 / (1 + e), e / (1 + e))
        errors = (q - record["labels"]) ** 2
        rows.append({"scene_id": record["scene_id"], "pair_count": len(errors),
                     "brier": float(errors.mean())})
        squared_sum += float(errors.sum())
        edge_count += len(errors)
    return {"macro_brier": float(np.mean([row["brier"] for row in rows])) if rows else None,
            "micro_brier": squared_sum / edge_count if edge_count else None,
            "scene_count": len(rows), "pair_count": edge_count, "per_scene": rows,
            "input_assumption": "scores are calibrated logits; this metric does not prove posterior calibration"}
