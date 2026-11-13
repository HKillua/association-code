"""Summarize fixed-graph global reassignment at original MTAD rejoin batches."""
from __future__ import annotations

from collections import Counter, defaultdict

from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics
from .hypothesis_gap_diagnostic import _direction
from .rejoin_next_report_summary import _focus
from .rejoin_track_history_summary import _cohorts, _distribution, _truth_split
from .rejoin_witness_squeeze import _truth_side


ARMS = ("global_actual", "global_masked", "global_fixed_incident", "global_full_alternative", "global_consensus",
        "global_abstain")
CONTROLS = ("original", "geometry_masked", "current", "native_masked", "pair_demeaned",
            "boundary_mask_assimilate", "boundary_full_current_mask", "boundary_robust")
SCOPES = ("focal_count", "retain_count", "release_count", "external_edges", "unmatched_endpoints")
VIEWS = ("actual", "masked", "fixed_mask")


def _comparison(targets, correct):
    outcomes = {}
    changes = {}

    def transitions(population, actions):
        pairs = Counter((row["decisions"]["current"] is True, action is True)
                        for row, action in zip(population, actions, strict=True))
        return dict(added=pairs[False, True], removed=pairs[True, False])

    for arm in (*CONTROLS, *ARMS):
        hits, errors = ([row["decisions"][arm] for row in group] for group in (targets, correct))
        outcomes[arm] = _metrics(hits.count(True), errors.count(True), len(hits), len(errors)) | dict(
            target_scored=sum(action is not None for action in hits),
            correct_scored=sum(action is not None for action in errors))
        if arm in ARMS:
            changes[arm] = dict(target=transitions(targets, hits), correct=transitions(correct, errors),
                                net_delta=outcomes[arm]["net"] - outcomes["current"]["net"])
    return dict(target=len(targets), correct=len(correct), arms=outcomes, changes_vs_current=changes)


def _compare(targets, correct):
    def available(row):
        return row["global_assignment"] is not None and row["decisions"]["current"] is not None

    return dict(full=_comparison(targets, correct), common_support=_comparison(
        [row for row in targets if available(row)], [row for row in correct if available(row)]))


def _local_direction(row):
    old = row["boundary_report"]["provenance"]["graph"]
    if _direction(old["factual_cached_margin_m2"]) != -1:
        raise ValueError("an original factual full rejoin did not have its frozen negative cached margin")
    return _direction(old["coordinate_mask_cached_margin_m2"])


def _pattern(view):
    if view["rejoined"] != bool(view["focal_count"]):
        raise ValueError("a rejoin flag does not match the original focal edge predicate")
    if view["retain_count"] == 2:
        return "full_retain"
    if view["rejoined"]:
        return "partial_rejoin"
    if view["release_count"] == 2:
        return "full_release"
    return "no_focal_other"


def _flatten(summary, prefix):
    for name, value in summary.items():
        scope = f"{prefix}:{name}"
        if isinstance(value, dict):
            yield from _flatten(value, scope)
        elif isinstance(value, (int, float)):
            yield scope, value


def _readouts(rows):
    direction_counts = {arm: Counter() for arm in ARMS}
    patterns, comparisons, labels, affected, graphs = (Counter() for _ in range(5))
    values = defaultdict(list)
    for row in rows:
        evidence = row["global_assignment"]
        actual, masked, fixed = (evidence[arm] for arm in VIEWS)
        actual_pattern, pattern, fixed_pattern = map(_pattern, (actual, masked, fixed))
        if actual_pattern != "full_retain":
            raise ValueError("an original public trigger was not a full factual rejoin")
        original_sign = _local_direction(row)
        new_sign = _direction(evidence["graph"]["masked_margin_m2"])
        if (original_sign != new_sign or _direction(evidence["graph"]["actual_margin_m2"]) != -1):
            raise ValueError("the same-clock mask changed the C55 cached local rank")
        local = evidence["local_masked"]
        local_pattern = _pattern(local)
        patterns[pattern] += 1
        patterns[f"{pattern}:{_truth_side(row) or 'mixed'}"] += 1
        patterns[f"fixed_mask:{fixed_pattern}"] += 1
        patterns[f"fixed_mask_to_masked:{fixed_pattern}->{pattern}"] += 1
        patterns[f"local_mask_to_global_mask:{local_pattern}->{pattern}"] += 1
        comparisons["focal_rejoin_removed"] += not masked["rejoined"]
        comparisons["focal_edges_reduced"] += masked["focal_count"] < actual["focal_count"]
        comparisons["full_retain_removed"] += masked["retain_count"] != 2
        comparisons["full_release_restored"] += masked["release_count"] == 2
        comparisons["external_edges_in_masked"] += bool(masked["external_edges"])
        comparisons["unmatched_endpoints_in_masked"] += bool(masked["unmatched_endpoints"])
        comparisons["fixed_mask_focal_rejoin_removed"] += not fixed["rejoined"]
        comparisons["fixed_mask_focal_edges_reduced"] += fixed["focal_count"] < actual["focal_count"]
        comparisons["fixed_mask_full_retain_removed"] += fixed["retain_count"] != 2
        comparisons["fixed_mask_full_release_restored"] += fixed["release_count"] == 2
        comparisons["masked_and_fixed_rejoin_removed"] += not (masked["rejoined"] or fixed["rejoined"])
        comparisons["masked_and_fixed_rejoin_differs"] += masked["rejoined"] != fixed["rejoined"]
        comparisons["masked_and_fixed_pattern_differs"] += pattern != fixed_pattern
        comparisons["local_and_global_rejoin_differs"] += local["rejoined"] != masked["rejoined"]
        comparisons["local_and_global_pattern_differs"] += local_pattern != pattern
        comparisons["current_shape_veto_and_masked_rejoin_removed"] += row["decisions"]["current"] is True and not masked["rejoined"]
        comparisons["current_shape_veto_only"] += row["decisions"]["current"] is True and masked["rejoined"]
        comparisons["masked_rejoin_removed_only"] += row["decisions"]["current"] is not True and not masked["rejoined"]
        for field in SCOPES:
            for scope in VIEWS:
                values[f"{scope}_{field}"].append(evidence[scope][field])
            comparisons[f"{field}:{actual[field]}->{masked[field]}"] += 1

        truth = _truth_side(row)
        expected = 1 if truth == "release" else -1 if truth == "retain" else None
        for arm in ARMS:
            vote = evidence["directions"][arm]
            direction_counts[arm][{1: "release", -1: "retain", 0: "abstain"}[vote]] += 1
            if vote and expected is not None:
                direction_counts[arm]["correct" if vote == expected else "wrong"] += 1
        for scope in ("graph", "affected", "affected_fixed_mask", "local_masked"):
            counts = graphs if scope == "graph" else affected
            for name, value in _flatten(evidence[scope], scope):
                values[name].append(value)
                counts[f"{name}:nonzero"] += bool(value)
        for name, value in _flatten(evidence["labels"], "labels"):
            labels[name] += value
            labels[f"{name}:windows"] += bool(value)
    return dict(patterns=dict(patterns), transitions=dict(comparisons), affected=dict(affected), graph=dict(graphs),
                labels=dict(labels), directions={arm: dict(counts) for arm, counts in direction_counts.items()},
                distributions={name: _distribution(samples) for name, samples in values.items()})


def _case(row):
    return dict(scene=row["scene"], component=row["component"], sparse=row["sparse"], start=row["row"]["start"],
                stop=row["row"]["stop"], truth=_truth_side(row), local_mask_direction=_local_direction(row),
                decisions=row["decisions"], global_assignment=row["global_assignment"])


def _cohort(rows, *, cases=False):
    targets, correct = _truth_split(rows)
    observed = [row for row in rows if row["global_assignment"] is not None]
    return dict(windows=len(rows), evidence_windows=len(observed), sparse=sum(row["sparse"] for row in rows),
                scenes=dict(Counter(row["scene"] for row in rows)),
                components=dict(Counter(row["component"] for row in rows)),
                truth=dict(Counter(_truth_side(row) or "mixed" for row in rows)),
                actions=_compare(targets, correct), **_readouts(observed)) | (
                    {"cases": list(map(_case, observed))} if cases else {})


def _gate(dual, sparse, components, true_case, evidence_rows):
    main = "global_consensus"
    dual_change = dual["full"]["changes_vs_current"][main]
    sparse_change = sparse["full"]["changes_vs_current"][main]
    differences = {name: values["dual"]["full"]["changes_vs_current"][main]["net_delta"]
                   for name, values in components.items()}
    positive = sum(value > 0 for value in differences.values())
    active = [row for row in evidence_rows if row["global_assignment"]["directions"][main] == 1]
    collateral = [row["global_assignment"]["labels"]["component_delta"] for row in active]
    false_added = sum(row["added_false"] for row in collateral)
    true_removed = sum(row["removed_true"] for row in collateral)
    checks = dict(keeps_unique_added_true=true_case["global_assignment"]["directions"][main] == 1,
                  no_new_dual_errors=dual_change["correct"]["added"] == 0,
                  no_new_sparse_errors=sparse_change["correct"]["added"] == 0,
                  dual_net_gain=dual_change["net_delta"] > 0,
                  sparse_noninferiority=sparse_change["net_delta"] >= 0,
                  positive_in_multiple_components=positive >= 2,
                  no_new_false_in_affected_component=false_added == 0,
                  no_removed_correct_in_affected_component=true_removed == 0)
    return dict(checks=checks, passes=all(checks.values()), positive_components=positive,
                component_net_changes=differences, active_full_release_windows=len(active),
                active_component_false_added=false_added, active_component_true_removed=true_removed)


def summarize(rows):
    """Verify original memberships and compare factual with masked global solves."""
    eligible = [row for row in rows if row.get("innovation") is not None]
    dual = [row for row in eligible if row.get("common") is not None]
    targets, correct = _truth_split(eligible)
    inherited = _cohorts(rows, eligible, dual, targets, correct)
    focus = _focus(inherited)
    if any((row["global_assignment"] is not None) != (row["boundary_report"] is not None) for row in rows):
        raise ValueError("a trigger was added to or lost from the C55 population")
    flipped = [row for row in dual if _local_direction(row) == 1]
    remaining = [row for row in dual if _local_direction(row) != 1]
    if (len(flipped), len(remaining)) != (13, 18):
        raise ValueError("the original 13/18 local ranking split changed")

    sparse = [row for row in rows if row["sparse"]]
    sparse_targets = [row for row in sparse if row["row"]["target_edges"] > 0]
    sparse_correct = [row for row in sparse if _primary_control(row["row"])]
    if (len(sparse_targets), len(sparse_correct)) != (28, 142):
        raise ValueError("the fixed sparse population changed")
    components = sorted({row["component"] for row in targets + correct})
    dual_targets, dual_correct = _truth_split(dual)
    by_component = {component: {name: _compare([r for r in t if r["component"] == component],
                                               [r for r in c if r["component"] == component])
                                for name, t, c in (("primary", sparse_targets, sparse_correct),
                                                   ("broad", targets, correct), ("dual", dual_targets, dual_correct))}
                    for component in components}
    primary, dual_results = _compare(sparse_targets, sparse_correct), _cohort(dual)
    detailed = focus | dict(local_flipped=flipped, local_unchanged=remaining)
    return dict(primary=primary, broad=_compare(targets, correct),
                broad_sparse=_compare([r for r in targets if r["sparse"]], [r for r in correct if r["sparse"]]),
                by_component=by_component, dual=dual_results,
                gate=_gate(dual_results["actions"], primary, by_component, focus["added_hit"][0], dual)) | {
                    name: _cohort(group, cases=name in detailed) for name, group in (inherited | detailed).items()}
