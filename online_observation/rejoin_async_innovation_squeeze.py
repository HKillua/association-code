"""Summarize cache-gap origins and native-time rejoin counterfactuals."""
from __future__ import annotations

from collections import Counter

from .async_report_scores import reported_direction
from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics
from .hypothesis_gap_diagnostic import _direction
from .rejoin_witness_squeeze import _truth_side, _truthward


ARMS = ("original", "geometry_masked", "current", "native_masked", "native_actual",
        "native_increment", "native_confirmed")
CONTROLS = ("current", "geometry_masked")
SIDES = ("release", "retain")


def _range(values):
    return [min(values), max(values)] if values else None


def _compare(target, correct):
    def count(rows, arm):
        return sum(row["decisions"].get(arm) is True for row in rows)

    def changes(rows):
        pairs = Counter((row["decisions"].get("current"), row["decisions"].get("native_confirmed")) for row in rows)
        return dict(added=pairs[False, True] + pairs[None, True], removed=pairs[True, False] + pairs[True, None])

    return dict(target=len(target), correct=len(correct), arms={
        arm: _metrics(count(target, arm), count(correct, arm), len(target), len(correct)) | dict(
            target_scored=sum(row["decisions"].get(arm) is not None for row in target),
            correct_scored=sum(row["decisions"].get(arm) is not None for row in correct))
        for arm in ARMS}, target_changes=changes(target), correct_changes=changes(correct))


def _scorable(row):
    return row.get("innovation") is not None and row["innovation"]["rolling"]["support_count"] == 2


def _comparison(target, correct):
    def available(rows):
        return [row for row in rows if _scorable(row) and all(row["decisions"][arm] is not None for arm in CONTROLS)]

    return dict(full=_compare(target, correct), common_support=_compare(available(target), available(correct)))


def _lineage(rows):
    violations = [(side, gap) for row in rows for side, role in row["lineage"]["roles"].items()
                  for gap in role["violations"]]
    counts, role_window_counts, per_window_max, removed_window_reports = Counter(), Counter(), [], []
    for row in rows:
        roles = row["lineage"]["roles"]
        active = [gap for role in roles.values() for gap in role["violations"]]
        if not active:
            continue
        counts["+".join(sorted({gap["kind"] for gap in active}))] += 1
        role_window_counts["+".join(sorted(side for side, role in roles.items() if role["violations"]))] += 1
        per_window_max.append(max(gap["span_s"] for gap in active))
        removed_window_reports.append(sum(len(gap["removed_times"]) for gap in active))

    return dict(raw_status=dict(Counter(row["lineage"]["raw_status"] for row in rows)),
                retained_status=dict(Counter(row["lineage"]["retained_status"] for row in rows)),
                kind_by_window=dict(counts), roles_by_window=dict(role_window_counts),
                kind_by_gap=dict(Counter(gap["kind"] for _, gap in violations)),
                removed_by_gap=dict(Counter(len(gap["removed_times"]) for _, gap in violations)),
                removed_by_window=dict(Counter(removed_window_reports)),
                max_gap_s=_range(per_window_max), native_step_s=_range([step for _, gap in violations
                                                                      for step in gap["native_steps_s"]]))


def _evidence(rows, arm):
    counts, directions, status, ages, compare_truth = Counter(), Counter(), Counter(), [], Counter()
    for row in rows:
        truth = _truth_side(row)
        sign = {"release": 1, "retain": -1}.get(truth)
        innovation = row["innovation"]
        reports = [r for r in innovation["reports"].values() if r is not None]
        result = innovation[arm]
        counts[f"support_{result['support_count']}"] += 1
        for report in reports:
            status[report[arm]["status"]] += 1
            ages.append(report["age_to_stop_s"])
            for role in report["refreshed_roles"]:
                counts[f"rolling_refreshed_{role}"] += 1
        if result["support_count"] != 2:
            continue
        actual, masked, increment = (_direction(result[key]) for key in ("actual_m2", "masked_m2", "increment_m2"))
        individual = [_direction(r[arm]["increment_m2"]) for r in reports]
        confirmed = row["confirmation"][arm]
        directions[f"increments:{individual[0]},{individual[1]}"] += 1
        counts.update(same_nonzero_increments=individual[0] == individual[1] != 0,
                      actual_vs_mask_changed=actual != masked,
                      raw_vs_common_increment_changed=increment != reported_direction(result["common_increment"]),
                      registered_vs_unregistered_actual_changed=actual != _direction(result["unregistered_actual_m2"]))
        if sign is None:
            continue
        for name, vote in (("actual", actual), ("masked", masked), ("increment", increment),
                           ("common_increment", reported_direction(result["common_increment"])),
                           ("confirmed", confirmed), ("boundary_masked", _direction(row["features"]["geometry_masked_m2"]))):
            compare_truth[f"{name}_{'correct' if vote == sign else 'wrong' if vote else 'zero'}"] += 1
    return dict(counts=dict(counts), directions=dict(directions), status=dict(status),
                selected_age_at_rejoin_s=_range(ages), truth_votes=dict(compare_truth))


def _cohort(rows):
    target = [row for row in rows if _truth_side(row) == "release"]
    correct = [row for row in rows if _truth_side(row) == "retain"]
    joint = [row for row in rows if _scorable(row) and row["innovation"]["onset_origin"]["support_count"] == 2]
    return dict(windows=len(rows), sparse=sum(row["sparse"] for row in rows),
                truth=dict(Counter(_truth_side(row) or "mixed" for row in rows)),
                scenes=dict(Counter(row["scene"] for row in rows)),
                components=dict(Counter(row["component"] for row in rows)),
                lineage=_lineage(rows), actions=_comparison(target, correct),
                rolling=_evidence(rows, "rolling"), onset_origin=_evidence(rows, "onset_origin"),
                same_prior_support=dict(windows=len(joint), rolling=_evidence(joint, "rolling"),
                                        onset_origin=_evidence(joint, "onset_origin")))


def summarize(rows):
    eligible = [row for row in rows if row.get("innovation") is not None]
    target = [row for row in eligible if _truth_side(row) == "release"]
    correct = [row for row in eligible if _truth_side(row) == "retain"]
    sparse = [row for row in rows if row["sparse"]]
    primary_target = [row for row in sparse if row["row"]["target_edges"] > 0]
    primary_correct = [row for row in sparse if _primary_control(row["row"])]
    opportunities = [row for row in eligible if row["lineage"]["novel_roles"]]
    both = [row for row in opportunities if len(row["lineage"]["novel_roles"]) == 2]
    single = [row for row in opportunities if len(row["lineage"]["novel_roles"]) == 1]
    if (len(opportunities), len(both), len(primary_target), len(primary_correct), len(target), len(correct)) != (
            71, 31, 28, 142, 67, 308):
        raise ValueError("an original evaluation or opportunity population changed")
    farther, closer = _truthward(target + correct)
    components = sorted({row["component"] for row in target + correct})

    def subset(items, component):
        return [row for row in items if row["component"] == component]

    return dict(primary=_comparison(primary_target, primary_correct), broad=_comparison(target, correct),
                broad_sparse=_comparison([row for row in target if row["sparse"]],
                                         [row for row in correct if row["sparse"]]),
                by_component={c: dict(primary=_comparison(subset(primary_target, c), subset(primary_correct, c)),
                                      broad=_comparison(subset(target, c), subset(correct, c))) for c in components},
                all=_cohort(eligible), opportunities=_cohort(opportunities), dual=_cohort(both), single=_cohort(single),
                farther=_cohort(farther), closer=_cohort(closer), correct_rejoins=_cohort(correct),
                current_false_vetoes=_cohort([row for row in correct if row["decisions"]["current"] is True]),
                mixed=_cohort([row for row in eligible if _truth_side(row) is None]))
