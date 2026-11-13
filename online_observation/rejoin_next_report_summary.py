"""Summarize matched next-report evidence on the frozen MTAD rejoin windows."""
from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from statistics import fmean

from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics
from .hypothesis_gap_diagnostic import _direction
from .rejoin_track_history_summary import _cohorts, _distribution, _truth_split
from .rejoin_witness_squeeze import _truth_side


SIDES = ("release", "retain")
ARMS = ("next_assimilate", "next_skip", "next_witness_mask", "next_robust", "next_abstain")
CONTROLS = ("original", "geometry_masked", "current", "native_masked", "native_confirmed", "pair_demeaned")
LOSS_FIELDS = ("own_loss_m2", "own_nominal_nll", "affected_edge_loss_m2")
MARGIN_FIELDS = ("actual_m2", "masked_m2", "increment_m2")


def _changes(rows, arm):
    changes = Counter((row["decisions"]["current"] is True, row["decisions"][arm] is True) for row in rows)
    return dict(added=changes[False, True], removed=changes[True, False])


def _comparison(targets, correct):
    actions = {}
    for arm in (*CONTROLS, *ARMS):
        hits, errors = ([row["decisions"][arm] for row in group] for group in (targets, correct))
        actions[arm] = _metrics(hits.count(True), errors.count(True), len(hits), len(errors)) | dict(
            target_scored=sum(value is not None for value in hits), correct_scored=sum(value is not None for value in errors))
    changes = {arm: dict(target=_changes(targets, arm), correct=_changes(correct, arm),
                         net_delta=actions[arm]["net"] - actions["current"]["net"]) for arm in ARMS}
    return dict(target=len(targets), correct=len(correct), arms=actions, changes_vs_current=changes)


def _matched(row, eligibility):
    evidence = row.get("next_report")
    return (evidence is not None and row["decisions"]["current"] is not None
            and row["decisions"]["geometry_masked"] is not None
            and any(branch[eligibility] for branch in evidence["branches"].values()))


def _compare(targets, correct):
    return dict(full=_comparison(targets, correct), **{
        name: _comparison([r for r in targets if _matched(r, qualification)],
                          [r for r in correct if _matched(r, qualification)])
        for name, qualification in (("common_support", "independent"), ("pre_boundary_support", "supported"))})


def _focus(fixed):
    errors = fixed["both_farther"] + fixed["false_closer"]
    by_mask = defaultdict(list)
    for row in errors:
        masked = _direction(row["history"]["rolling"]["masked_m2"])
        drivers = row["history"]["temporal"]["loo_driving_sides"]
        if not masked or (masked == -1) != bool(drivers):
            raise ValueError("the C53 split no longer matches its frozen report-mask dependence")
        by_mask[masked].append(row)
    true = fixed["added_hit"]
    if (len(by_mask[-1]), len(by_mask[1]), len(true)) != (5, 5, 1) or any(
        _direction(row["history"]["rolling"]["masked_m2"]) != 1
        or row["history"]["temporal"]["loo_driving_sides"] for row in true
    ):
        raise ValueError("the frozen C53 5/5/1 populations changed")
    return dict(new_report_driven=by_mask[-1], original_geometry_wrong=by_mask[1], added_hit=true)


def _losses(pairs, field):
    delta = [before - after for before, after in pairs]
    if field.endswith("_m2"):
        locations = {f"{arm}_rmse_m": sqrt(fmean(values)) for arm, values in zip(("assimilate", "skip"), zip(*pairs))}
        differences = Counter(map(_direction, delta))
    else:
        locations = {f"{arm}_mean": fmean(values) for arm, values in zip(("assimilate", "skip"), zip(*pairs))}
        differences = Counter((value > 0) - (value < 0) for value in delta)
    return dict(count=len(pairs), skip_improves=differences[1], skip_worsens=differences[-1],
                ties=differences[0], paired_assimilate_minus_skip=_distribution(delta), **locations)


def _witnesses(items):
    coverage, statuses, margins = Counter(), Counter(), Counter()
    numeric = defaultdict(list)
    for item, truth in items:
        coverage["comparisons"] += 1
        statuses[item["status"]] += 1
        for name in ("raw_after_focal_before_stop", "raw_after_floor_before_stop",
                     "cached_after_floor_before_stop", "novel_after_floor_before_stop"):
            coverage[f"{name}_reports"] += len(item[name])
            coverage[f"{name}_comparisons"] += bool(item[name])
        coverage["at_stop_comparisons"] += bool(item["at_stop"])
        if item["witness_stamp"] is not None:
            coverage["witness_selected"] += 1
            coverage["witness_in_boundary_shape"] += bool(item.get("in_boundary_shape"))
            numeric["witness_delay_s"].append(item["witness_delay_s"])
            numeric["lead_to_stop_s"].append(item["lead_to_stop_s"])
        scores = item["scores"]
        if scores is None:
            continue
        assimilate, skip = (scores[name] for name in ("assimilate", "skip"))
        for field in LOSS_FIELDS:
            if assimilate[field] is not None and skip[field] is not None:
                numeric[field].append((assimilate[field], skip[field]))
        if assimilate["actual_m2"] is None or skip["actual_m2"] is None:
            continue
        for field in MARGIN_FIELDS:
            directions = [_direction(score[field]) for score in (assimilate, skip)]
            margins[f"{field}:{directions[0]}->{directions[1]}"] += 1
            if truth is not None:
                for arm, vote in zip(("assimilate", "skip"), directions, strict=True):
                    correctness = "correct" if vote == truth else "wrong" if vote else "tie"
                    margins[f"{field}:{arm}:{correctness}"] += 1
    return dict(coverage=dict(coverage), statuses=dict(statuses), margins=dict(margins),
                paired_losses={key: _losses(values, key) for key, values in numeric.items() if key in LOSS_FIELDS},
                clocks={key: _distribution(values) for key, values in numeric.items() if key not in LOSS_FIELDS})


def _evidence(rows):
    witnesses = defaultdict(list)
    directions = {arm: Counter() for arm in ARMS}
    branch_counts = Counter()
    for row in rows:
        evidence = row["next_report"]
        truth = _truth_side(row)
        expected = 1 if truth == "release" else -1 if truth == "retain" else None
        for arm in ARMS:
            vote = evidence["directions"][arm]
            directions[arm][{1: "release", 0: "abstain", -1: "retain"}[vote]] += 1
            if expected is not None and vote:
                directions[arm]["correct" if expected == vote else "wrong"] += 1
        supported = sum(branch["supported"] for branch in evidence["branches"].values())
        independent = sum(branch["independent"] for branch in evidence["branches"].values())
        branch_counts[f"supported_focals:{supported}"] += 1
        branch_counts[f"independent_focals:{independent}"] += 1
        branch_counts["windows_with_matched_pre_boundary_support"] += bool(supported)
        branch_counts["windows_with_matched_independent_support"] += bool(independent)
        branch_counts["focal_pairs_excluded_by_boundary_shape"] += supported - independent
        for side in SIDES:
            focal = evidence["focals"][side]
            items = dict(same=focal["same"], earlier_same=focal["earlier_same"], **{
                f"opposite_{branch}": focal["opposite"][branch] for branch in SIDES})
            if items["same"]["novel_after_floor_before_stop"]:
                raise ValueError("a latest C51 report had a cache-excluded same-track successor")
            for role, item in items.items():
                if item is not None:
                    witnesses[role].append((item, expected))
                    witnesses[f"{side}:{role}"].append((item, expected))
        branch_counts["windows_with_same_track_raw_successor"] += any(
            focal["same"]["raw_after_focal_before_stop"] for focal in evidence["focals"].values())
        branch_counts["windows_with_same_track_cache_excluded_successor"] += any(
            focal["same"]["novel_after_floor_before_stop"] for focal in evidence["focals"].values())
        branch_counts["windows_with_same_track_at_stop"] += any(
            focal["same"]["at_stop"] for focal in evidence["focals"].values())
        branch_counts["windows_with_novel_opposite_witness"] += any(
            item["witness_stamp"] is not None for focal in evidence["focals"].values() for item in focal["opposite"].values())
        branch_counts["windows_with_earlier_same_track_witness"] += any(
            focal["earlier_same"] is not None and focal["earlier_same"]["witness_stamp"] is not None
            for focal in evidence["focals"].values())
    return dict(branch_coverage=dict(branch_counts), directions={arm: dict(counts) for arm, counts in directions.items()},
                witnesses={role: _witnesses(items) for role, items in witnesses.items()})


def _case(row):
    return dict(scene=row["scene"], component=row["component"], sparse=row["sparse"], start=row["row"]["start"],
                stop=row["row"]["stop"], truth=_truth_side(row), decisions=row["decisions"],
                masked_m2=row["history"]["rolling"]["masked_m2"],
                loo_driving_sides=row["history"]["temporal"]["loo_driving_sides"], next_report=row["next_report"])


def _cohort(rows, *, cases=False):
    targets, correct = _truth_split(rows)
    evidence = [row for row in rows if row["next_report"] is not None]
    return dict(windows=len(rows), evidence_windows=len(evidence), sparse=sum(row["sparse"] for row in rows),
                scenes=dict(Counter(row["scene"] for row in rows)), components=dict(Counter(row["component"] for row in rows)),
                truth=dict(Counter(_truth_side(row) or "mixed" for row in rows)), actions=_compare(targets, correct),
                **_evidence(evidence)) | ({"cases": list(map(_case, evidence))} if cases else {})


def _gate(dual, sparse, components, news_errors, true_case):
    main = "next_robust"
    dual_changes = dual["full"]["changes_vs_current"][main]
    sparse_changes = sparse["full"]["changes_vs_current"][main]
    differences = {name: values["dual"]["full"]["changes_vs_current"][main]["net_delta"]
                   for name, values in components.items()}
    positive = sum(value > 0 for value in differences.values())
    active_corrections = sum(row["decisions"]["pair_demeaned"] is True
                             and row["next_report"]["directions"][main] == -1 for row in news_errors)
    total_corrections = sum(row["decisions"]["pair_demeaned"] is True
                            and row["decisions"][main] is False for row in news_errors)
    checks = dict(keeps_unique_added_true=true_case["next_report"]["directions"][main] == 1,
                  corrects_a_new_report_error=active_corrections > 0,
                  no_new_dual_errors=dual_changes["correct"]["added"] == 0,
                  no_new_sparse_errors=sparse_changes["correct"]["added"] == 0,
                  dual_net_gain=dual_changes["net_delta"] > 0, sparse_noninferiority=sparse_changes["net_delta"] >= 0,
                  positive_in_multiple_components=positive >= 2)
    return dict(checks=checks, passes=all(checks.values()), positive_components=positive,
                component_net_changes=differences, news_errors_actively_corrected=active_corrections,
                news_errors_removed_including_fallback=total_corrections)


def summarize(rows):
    """Check frozen memberships and report both full-population and paired support actions."""
    eligible = [row for row in rows if row.get("innovation") is not None]
    dual = [row for row in eligible if row.get("common") is not None]
    target, correct = _truth_split(eligible)
    fixed = _cohorts(rows, eligible, dual, target, correct)
    focus = _focus(fixed)
    if any((row["next_report"] is not None) != (row["common"] is not None) for row in rows):
        raise ValueError("a frozen dual rejoin lost next-report accounting")
    sparse = [row for row in rows if row["sparse"]]
    sparse_target = [row for row in sparse if row["row"]["target_edges"] > 0]
    sparse_correct = [row for row in sparse if _primary_control(row["row"])]
    if (len(sparse_target), len(sparse_correct)) != (28, 142):
        raise ValueError("the fixed sparse population changed")

    components = sorted({row["component"] for row in target + correct})
    dual_target, dual_correct = _truth_split(dual)
    by_component = {component: {name: _compare([r for r in t if r["component"] == component],
                                               [r for r in c if r["component"] == component])
                                for name, t, c in (("primary", sparse_target, sparse_correct),
                                                   ("broad", target, correct), ("dual", dual_target, dual_correct))}
                    for component in components}
    primary, dual_result = _compare(sparse_target, sparse_correct), _cohort(dual)
    case_groups = {"new_report_driven", "original_geometry_wrong", "added_hit"}
    return dict(primary=primary, broad=_compare(target, correct),
                broad_sparse=_compare([r for r in target if r["sparse"]], [r for r in correct if r["sparse"]]),
                dual=dual_result, by_component=by_component,
                gate=_gate(dual_result["actions"], primary, by_component, focus["new_report_driven"], focus["added_hit"][0])) | {
                    name: _cohort(group, cases=name in case_groups) for name, group in (fixed | focus).items()}
