"""Observe fixed association costs and score their truth margins after replay."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import sqrt

from .continuous_tracker import FRESH_S, NULL_COST
from .nemot_replay import ARMS, Fragment


RESIDUAL_BOUNDS_M = (50, 100, 200, 400, 800, 1200, 2000, 3000, 4000, 5000)
MARGIN_BOUNDS_M = (-2000, -1000, -500, -200, -100, -50, 0, 50, 100, 200, 500, 1000, 2000)
_COST_BOUNDS = tuple(value ** 2 for value in RESIDUAL_BOUNDS_M)
_RESIDUAL_BIN_NAMES = tuple(f"le_{value}" for value in RESIDUAL_BOUNDS_M) + ("above",)
_MARGIN_BIN_NAMES = tuple(f"le_{value}" for value in MARGIN_BOUNDS_M) + ("above",)
_GROUPS = ("all", "first", "continuation", "fresh_0_3", "first_fresh_0_3", "continuation_fresh_0_3")
_ARM_PAIRS = tuple(combinations(ARMS, 2))
_ARM_METRICS = ("counts", "true_residual_m", "false_residual_m", "min_true_residual_m",
                "min_false_residual_m", "margin_m")


@dataclass(frozen=True)
class EdgeQuery:
    key: Fragment
    first: bool
    fresh_peer_count: int
    candidates: tuple[tuple[Fragment, tuple[float, ...]], ...]
    selected: tuple[tuple[Fragment, ...], ...]


@dataclass(frozen=True)
class EdgeFrame:
    stamp: float
    queries: tuple[EdgeQuery, ...]


class EdgeAuditRecorder:
    """Copy only updated-key queries and their chosen partners, without labels."""

    def __init__(self):
        self.frames: list[EdgeFrame] = []

    def __call__(self, stamp, updated, born, fresh, costs, memberships):
        by_source = Counter(key[0] for key in fresh)
        candidates = defaultdict(list)
        for edge, kf_cost in costs["kf"].items():
            left, right = edge
            if left not in updated and right not in updated:
                continue
            triplet = (kf_cost, *(costs[arm][edge] for arm in ARMS[1:]))
            if left in updated:
                candidates[left].append((right, triplet))
            if right in updated:
                candidates[right].append((left, triplet))

        selected = []
        for arm in ARMS:
            picked = {}
            for _, members in memberships[arm]:
                for key in members:
                    if key in updated:
                        picked[key] = tuple(peer for peer in members if peer[0] != key[0])
            selected.append(picked)

        queries = tuple(EdgeQuery(key, key in born, len(fresh) - by_source[key[0]],
                                  tuple(sorted(candidates[key])), tuple(picked[key] for picked in selected))
                        for key in sorted(updated))
        self.frames.append(EdgeFrame(stamp, queries))


def _empty_audit():
    return {
        group: {
            "support": Counter(),
            "arms": {arm: {name: Counter() for name in _ARM_METRICS} for arm in ARMS},
            "comparisons": {f"{before}_to_{after}": {"counts": Counter(), "margin_delta_m": Counter()}
                            for before, after in _ARM_PAIRS},
        } for group in _GROUPS
    }


def merge_edge_audits(scene_results):
    """Sum integer counters over scenes, retaining empty strata."""
    merged = _empty_audit()
    for scene in scene_results:
        for group, part in scene.items():
            target = merged[group]
            target["support"].update(part["support"])
            for family in ("arms", "comparisons"):
                for method, metrics in part[family].items():
                    for name, counts in metrics.items():
                        target[family][method][name].update(counts)
    return merged


def _cost_bin(squared):
    return _RESIDUAL_BIN_NAMES[bisect_left(_COST_BOUNDS, squared)]


def _margin_bin(distance):
    return _MARGIN_BIN_NAMES[bisect_left(MARGIN_BOUNDS_M, distance)]


def _minima(query, truth, index, single_false_unseen, labels):
    true_costs = [costs[index] for (_, costs), is_true in zip(query.candidates, truth, strict=True) if is_true]
    false_costs = [costs[index] for (_, costs), is_true in zip(query.candidates, truth, strict=True) if not is_true]
    selected = query.selected[index]
    identity = labels[query.key]
    any_true = any(labels[key] == identity for key in selected)
    any_false = any(labels[key] != identity for key in selected)
    outcome = "wrong" if any_false else "correct" if any_true else "unmatched"
    metrics = {name: Counter() for name in _ARM_METRICS}
    counts = metrics["counts"]
    counts[outcome] += 1
    if single_false_unseen:
        counts["single_false_unseen_accepted" if any_false else "single_false_unseen_unmatched"] += 1
    minimum_true = min(true_costs, default=None)
    minimum_false = min(false_costs, default=None)
    if minimum_true is not None:
        metrics["true_residual_m"].update(map(_cost_bin, true_costs))
        metrics["min_true_residual_m"][_cost_bin(minimum_true)] += 1
    if minimum_false is not None:
        metrics["false_residual_m"].update(map(_cost_bin, false_costs))
        metrics["min_false_residual_m"][_cost_bin(minimum_false)] += 1

    rank = None
    margin = None
    if minimum_true is not None and minimum_false is not None:
        rank = ("win" if minimum_true < minimum_false else "loss" if minimum_true > minimum_false else "tie")
        margin = sqrt(minimum_false) - sqrt(minimum_true)
        counts[f"true_rank_{rank}"] += 1
        metrics["margin_m"][_margin_bin(margin)] += 1
    if minimum_true is not None and not any_true:
        if minimum_true >= 2 * NULL_COST:
            counts["unselected_all_true_at_or_above_null"] += 1
            counts["unselected_all_true_above_null" if minimum_true > 2 * NULL_COST
                   else "unselected_best_true_at_null"] += 1
        else:
            counts["unselected_true_below_null"] += 1
            counts[f"unselected_true_below_null_{rank or 'no_false'}"] += 1
    return metrics, outcome, rank, margin, minimum_true, minimum_false


def _comparisons(query, diagnostics):
    for left, right in combinations(range(len(ARMS)), 2):
        _, before, rank_before, margin_before, true_before, false_before = diagnostics[left]
        _, after, rank_after, margin_after, true_after, false_after = diagnostics[right]
        counts = Counter({f"{before}_to_{after}": 1})
        counts["changed_selected_partner"] += query.selected[left] != query.selected[right]
        histogram = Counter()
        if margin_before is not None:
            delta = margin_after - margin_before
            counts[f"rank_{rank_before}_to_{rank_after}"] += 1
            counts["margin_increased" if delta > 0 else "margin_decreased" if delta < 0 else "margin_unchanged"] += 1
            counts["min_true_decreased"] += true_after < true_before
            counts["min_true_and_false_decreased"] += true_after < true_before and false_after < false_before
            counts["min_true_decreased_margin_decreased"] += true_after < true_before and delta < 0
            histogram[_margin_bin(delta)] += 1
        yield f"{ARMS[left]}_to_{ARMS[right]}", counts, histogram


def score_edge_audit(frames, labels):
    """Label frozen event queries using only opposite-source history available then."""
    result = _empty_audit()
    last_report = defaultdict(dict)
    for frame in frames:
        for query in frame.queries:
            last_report[labels[query.key]][query.key[0]] = frame.stamp
        for query in frame.queries:
            source = query.key[0]
            identity = labels[query.key]
            ages = [frame.stamp - last for other, last in last_report[identity].items() if other != source]
            truth = tuple(labels[key] == identity for key, _ in query.candidates)
            true_reachable = any(truth)
            truth_support = ("true_reachable" if true_reachable else "true_outside_graph" if any(age <= FRESH_S for age in ages)
                             else "true_stale_only" if ages else "true_never_seen")
            single_false_unseen = truth_support == "true_never_seen" and len(truth) == 1
            support = dict(queries=1, no_candidate=int(not truth), candidate_edges=len(truth),
                           true_edges=sum(truth), false_edges=len(truth) - sum(truth),
                           mixed_candidates=int(true_reachable and not all(truth)),
                           single_false_unseen=int(single_false_unseen))
            support[truth_support] = 1
            diagnostics = tuple(_minima(query, truth, index, single_false_unseen, labels)
                                for index in range(len(ARMS)))
            comparisons = tuple(_comparisons(query, diagnostics))
            phase = "first" if query.first else "continuation"
            groups = ("all", phase, "fresh_0_3", f"{phase}_fresh_0_3") if query.fresh_peer_count <= 3 else ("all", phase)
            for name in groups:
                group = result[name]
                group["support"].update(support)
                for arm, (metrics, *_) in zip(ARMS, diagnostics, strict=True):
                    for metric, counts in metrics.items():
                        group["arms"][arm][metric].update(counts)
                for pair, counts, histogram in comparisons:
                    group["comparisons"][pair]["counts"].update(counts)
                    group["comparisons"][pair]["margin_delta_m"].update(histogram)
    return result
