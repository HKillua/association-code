"""Summarize causal track-history contrasts on frozen MTAD rejoin windows."""
from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from statistics import fmean, median

from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics
from .hypothesis_gap_diagnostic import _direction
from .rejoin_witness_squeeze import _truth_side


SIDES = ("release", "retain")
ROLES = ("own", *SIDES)
CONTROLS = ("original", "geometry_masked", "current", "native_masked", "native_confirmed", "pair_demeaned")
ARMS = ("history_confirmed", "history_support_only", "history_masked", "history_own_corrected",
        "history_opponents_corrected", "history_both_corrected")
SCORE_VIEWS = ("rolling", "own_corrected", "opponents_corrected", "both_corrected")


def _compare(targets, correct):
    def values(population, arm):
        return [row["decisions"].get(arm, row["decisions"]["current"]) for row in population]

    def changes(population, outcomes):
        pairs = Counter((row["decisions"]["current"] is True, value is True)
                        for row, value in zip(population, outcomes, strict=True))
        return dict(added=pairs[False, True], removed=pairs[True, False])

    counts, changes_by_arm = {}, {}
    for arm in (*CONTROLS, *ARMS):
        t_values, c_values = values(targets, arm), values(correct, arm)
        counts[arm] = _metrics(sum(v is True for v in t_values), sum(v is True for v in c_values),
                               len(targets), len(correct)) | dict(
            target_scored=sum(v is not None for v in t_values), correct_scored=sum(v is not None for v in c_values))
        if arm in ARMS:
            changes_by_arm[arm] = dict(target=changes(targets, t_values), correct=changes(correct, c_values),
                                       net_delta=counts[arm]["net"] - counts["current"]["net"])
    main = changes_by_arm["history_confirmed"]
    return dict(target=len(targets), correct=len(correct), arms=counts,
                changes_vs_current=changes_by_arm,
                target_changes=main["target"], correct_changes=main["correct"])


def _history_support(row):
    return row.get("history") is not None and row["history"]["temporal"]["status"] == "supported"


def _comparison(targets, correct):
    def same_support(population, condition):
        return [row for row in population if all(row["decisions"].get(key) is not None for key in
                                                  ("current", "geometry_masked")) and condition(row)]

    def compare_support(condition):
        return _compare(same_support(targets, condition), same_support(correct, condition))

    return dict(full=_compare(targets, correct), dual_support=compare_support(lambda row: row.get("common") is not None),
                common_support=compare_support(_history_support), eligible_support=compare_support(
                    lambda row: _history_support(row) and row["history"]["temporal"]["eligible"]))


def _distribution(values):
    if not values:
        return dict(count=0, minimum=None, median=None, maximum=None, mean=None)
    return dict(count=len(values), minimum=min(values), median=median(values), maximum=max(values), mean=fmean(values))


def _truth_split(rows):
    return tuple([row for row in rows if _truth_side(row) == side] for side in SIDES)


def _cohorts(rows, eligible, dual, targets, correct):
    original_adds = [row for row in dual if row["decisions"]["native_confirmed"] is True
                     and row["decisions"]["current"] is not True]
    retained = [row for row in original_adds if row["confirmed"]["demeaned"] == 1]
    farther = [row for row in retained if _truth_side(row) == "retain"
               and _direction(row["innovation"]["rolling"]["release_change_m2"]) > 0]
    closer = [row for row in retained if _truth_side(row) == "retain"
              and _direction(row["innovation"]["rolling"]["release_change_m2"]) < 0]
    added = [row for row in retained if _truth_side(row) == "release"]
    removed = [row for row in original_adds if _truth_side(row) == "retain" and row["confirmed"]["demeaned"] != 1]
    new_demeaned = [row for row in dual if _truth_side(row) == "retain"
                    and row["confirmed"]["raw"] != 1 and row["confirmed"]["demeaned"] == 1]
    new_demeaned_action = [row for row in new_demeaned if row["decisions"]["native_confirmed"] is not True
                           and row["decisions"]["pair_demeaned"] is True]
    if tuple(map(len, (rows, eligible, dual, targets, correct, farther, closer, added, removed, new_demeaned,
                       new_demeaned_action))) != (482, 416, 31, 67, 308, 7, 3, 1, 1, 6, 5):
        raise ValueError("a frozen C52 population changed")
    focused_errors = {id(row) for row in farther + closer}
    return dict(both_farther=farther, false_closer=closer, added_hit=added, removed_wrong=removed,
                new_demeaned_wrong=new_demeaned, new_demeaned_wrong_actions=new_demeaned_action,
                other_correct_rejoins=[row for row in dual if _truth_side(row) == "retain" and id(row) not in focused_errors],
                correct_rejoins=correct, current_false_vetoes=[row for row in correct if row["decisions"]["current"] is True],
                mixed=[row for row in eligible if _truth_side(row) is None])


def _temporal(rows):
    statuses, counts, patterns, votes = (Counter() for _ in range(4))
    losses = defaultdict(list)
    for row in rows:
        data = row["history"]["temporal"]
        truth = _truth_side(row)
        sign = 1 if truth == "release" else -1 if truth == "retain" else None
        statuses[data["status"]] += 1
        counts[f"reason:{data['reason']}"] += 1
        counts["after_onset_preceding_reports"] += data.get("after_onset_count", 0)
        counts["latest_both_after_onset"] += all(data["latest_after_onset"].values())
        counts["eligible"] += data["eligible"]
        for name in ("loo_driving_sides", "isolated_innovation_sides"):
            affected = data.get(name, ())
            counts[f"{name}:any"] += bool(affected)
            counts[f"{name}:both"] += len(affected) == 2
            for side in affected:
                counts[f"{name}:{side}"] += 1
        counts["single_mask_sensitive_with_isolated_innovation"] += bool(
            set(data["loo_driving_sides"]) & set(data["isolated_innovation_sides"]))
        for prefix, label in (("", "mask"), ("raw_", "raw_mask")):
            for side in SIDES:
                value = data[f"{prefix}loo_actual_directions"][side]
                votes[f"{label}_{side}:{value}"] += 1
                if sign is not None:
                    votes[f"{label}_{side}_{'correct' if value == sign else 'wrong' if value else 'zero'}"] += 1
                losses[f"{label}_{side}"].append(data[f"{prefix}loo_actual_m2"][side])
        if data["status"] != "supported":
            continue
        demeaned = row["confirmed"]["demeaned"]
        for name in ("actual_directions", "increment_directions", "masked_directions"):
            previous = [d for side in SIDES for d in data[name][side]]
            pattern = ",".join(map(str, previous))
            patterns[f"{name}:{pattern}"] += 1
            same = len(set(previous)) == 1
            counts[f"{name}_unanimous"] += same and bool(previous[0])
            counts[f"{name}_contains_opposition"] += 1 in previous and -1 in previous
            counts[f"{name}_contains_zero"] += 0 in previous
            if sign is not None and same:
                votes[f"{name}_{'correct' if previous[0] == sign else 'wrong' if previous[0] else 'zero'}"] += 1
            counts[f"{name}_agree_with_demeaned"] += bool(demeaned) and same and previous[0] == demeaned
    return dict(statuses=dict(statuses), counts=dict(counts), patterns=dict(patterns), votes=dict(votes),
                single_mask_margins_m2={label: _distribution(data) for label, data in losses.items()})


def _decision_evidence(rows):
    votes = defaultdict(Counter)
    for row in rows:
        truth = _truth_side(row)
        expected = 1 if truth == "release" else -1 if truth == "retain" else None
        for arm in ARMS:
            direction = row["history_decisions"].get(arm)
            status = "unavailable" if direction is None else "release" if direction > 0 else "retain" if direction < 0 else "abstain"
            votes[arm][status] += 1
            if expected is not None and direction:
                votes[arm]["correct" if direction == expected else "wrong"] += 1
    return {arm: dict(counts) for arm, counts in votes.items()}


def _loss_summary(pairs):
    if not pairs:
        return dict(count=0, original_rmse_m=None, corrected_rmse_m=None, improves=0, worsens=0, ties=0)
    original, corrected = zip(*pairs)
    signs = Counter(_direction(before - after) for before, after in pairs)
    return dict(count=len(pairs), original_rmse_m=sqrt(fmean(original)), corrected_rmse_m=sqrt(fmean(corrected)),
                improves=signs[1], worsens=signs[-1], ties=signs[0])


def _track_evidence(rows):
    history_status, preceding_status, present_status, repeats = (Counter() for _ in range(4))
    history_losses = defaultdict(list)
    current_losses = []
    terms = defaultdict(list)
    for row in rows:
        for report_side in SIDES:
            report = row["history"]["reports"][report_side]
            present_status[f"{report_side}:{report['status']}"] += 1
            for prior in report["preceding"]:
                preceding_status[prior["status"]] += 1
                preceding_status[f"{report_side}:{prior['status']}"] += 1
            for role in ROLES:
                evidence = report["history"][role]
                status = evidence["status"]
                history_status[status] += 1
                history_status[f"{role}:{status}"] += 1
                if evidence["holdout_loss_m2"] is not None:
                    history_losses[role].append((evidence["holdout_baseline_m2"], evidence["holdout_loss_m2"]))
            if report["own_corrected_loss_m2"] is not None:
                current_losses.append((report["own_loss_m2"], report["own_corrected_loss_m2"]))
                repeat = report["decomposition"]
                for name in ("persistent_increment_m2", "remainder_increment_m2"):
                    terms[name].append(repeat[name])
                repeats["persistent_dominates"] += abs(repeat["persistent_increment_m2"]) > abs(repeat["remainder_increment_m2"])
                repeats["remainder_dominates"] += abs(repeat["persistent_increment_m2"]) < abs(repeat["remainder_increment_m2"])
        differential = row["history"].get("differential")
        if differential is not None:
            repeats["differential_supported_windows"] += 1
            for name in ("persistent_increment_m2", "remainder_increment_m2", "total_increment_m2"):
                terms[f"pair_{name}"].append(differential[name])
                if name != "total_increment_m2":
                    terms[f"pair_absolute_{name}"].append(abs(differential[name]))
            repeats["pair_persistent_dominates"] += abs(differential["persistent_increment_m2"]) > abs(differential["remainder_increment_m2"])
            repeats["pair_remainder_dominates"] += abs(differential["persistent_increment_m2"]) < abs(differential["remainder_increment_m2"])
            repeats[f"pair_term_signs:{_direction(differential['persistent_increment_m2'])},{_direction(differential['remainder_increment_m2'])}"] += 1
    return dict(history_status=dict(history_status), preceding_status=dict(preceding_status),
                present_status=dict(present_status), repeated_innovation=dict(repeats),
                history_holdout={role: _loss_summary(history_losses[role]) for role in ROLES},
                current_holdout=_loss_summary(current_losses), increment_terms={k: _distribution(v) for k, v in terms.items()})


def _score_evidence(rows):
    counts = {view: Counter() for view in SCORE_VIEWS}
    for row in rows:
        truth = _truth_side(row)
        expected = 1 if truth == "release" else -1 if truth == "retain" else None
        for view in SCORE_VIEWS:
            score = row["history"][view]
            support = score["support_count"]
            counts[view][f"support:{support}"] += 1
            if support != 2:
                continue
            for field in ("actual_m2", "masked_m2", "increment_m2"):
                direction = _direction(score[field])
                counts[view][f"{field}:{direction}"] += 1
                if expected is not None:
                    counts[view][f"{field}_{'correct' if direction == expected else 'wrong' if direction else 'zero'}"] += 1
            signs = (_direction(score[f"{side}_change_m2"]) for side in SIDES)
            counts[view][f"edge_changes:{','.join(map(str, signs))}"] += 1
    return {view: dict(values) for view, values in counts.items()}


def _case(row):
    return dict(scene=row["scene"], component=row["component"], sparse=row["sparse"], start=row["row"]["start"],
                stop=row["row"]["stop"], truth=_truth_side(row),
                old_release_change_m2=row["innovation"]["rolling"]["release_change_m2"],
                old_confirmation=row["confirmed"], history=row["history"],
                history_decisions=row["history_decisions"], decisions=row["decisions"])


def _cohort(rows, *, include_cases=False):
    usable = [row for row in rows if row.get("history") is not None]
    target, correct = _truth_split(rows)
    result = dict(windows=len(rows), history_windows=len(usable), sparse=sum(row["sparse"] for row in rows),
                  truth=dict(Counter(_truth_side(row) or "mixed" for row in rows)),
                  scenes=dict(Counter(row["scene"] for row in rows)), components=dict(Counter(row["component"] for row in rows)),
                  actions=_comparison(target, correct), temporal=_temporal(usable), decision_evidence=_decision_evidence(usable),
                  track_evidence=_track_evidence(usable), score_evidence=_score_evidence(usable))
    return result | ({"cases": list(map(_case, usable))} if include_cases else {})


def _gate(dual, sparse, components, true_case):
    main = "history_confirmed"
    differences = {name: values["dual"]["full"]["changes_vs_current"][main]["net_delta"]
                   for name, values in components.items()}
    positive_components = sum(value > 0 for value in differences.values())
    dual_full, sparse_full = dual["full"], sparse["full"]
    dual_common = dual["common_support"]["arms"]
    checks = dict(keeps_unique_added_true=true_case.get("history_decisions", {}).get(main) == 1,
                  no_new_dual_errors=dual_full["correct_changes"]["added"] == 0,
                  no_new_sparse_errors=sparse_full["correct_changes"]["added"] == 0,
                  dual_net_gain=dual_full["changes_vs_current"][main]["net_delta"] > 0,
                  sparse_noninferiority=sparse_full["changes_vs_current"][main]["net_delta"] >= 0,
                  positive_in_multiple_components=positive_components >= 2)
    return dict(checks=checks, passes=all(checks.values()), positive_components=positive_components,
                component_net_changes=differences,
                common_support_net_vs_current=dual_common[main]["net"] - dual_common["current"]["net"],
                common_support_net_vs_boundary_masked=dual_common[main]["net"] - dual_common["geometry_masked"]["net"],
                common_support_net_vs_native_masked=dual_common[main]["net"] - dual_common["native_masked"]["net"])


def summarize(rows):
    """Reproduce fixed populations, then compare track history on common support."""
    eligible = [row for row in rows if row.get("innovation") is not None]
    dual = [row for row in eligible if row.get("common") is not None]
    target, correct = _truth_split(eligible)
    sparse = [row for row in rows if row["sparse"]]
    sparse_target = [row for row in sparse if row["row"]["target_edges"] > 0]
    sparse_correct = [row for row in sparse if _primary_control(row["row"])]
    if (len(sparse_target), len(sparse_correct)) != (28, 142):
        raise ValueError("a frozen C51 sparse evaluation population changed")
    fixed = _cohorts(rows, eligible, dual, target, correct)

    def for_component(items, component):
        return [row for row in items if row["component"] == component]

    components = sorted({row["component"] for row in target + correct})
    dual_target, dual_correct = _truth_split(dual)
    by_component = {}
    for component in components:
        by_component[component] = {name: _comparison(for_component(t, component), for_component(c, component))
                                   for name, t, c in (("primary", sparse_target, sparse_correct),
                                                      ("broad", target, correct), ("dual", dual_target, dual_correct))}
    primary = _comparison(sparse_target, sparse_correct)
    dual_result = _cohort(dual)
    return dict(primary=primary, broad=_comparison(target, correct),
                broad_sparse=_comparison([row for row in target if row["sparse"]], [row for row in correct if row["sparse"]]),
                by_component=by_component, dual=dual_result, gate=_gate(dual_result["actions"], primary, by_component,
                                                                      fixed["added_hit"][0])) | {
        name: _cohort(group, include_cases=name in ("both_farther", "false_closer", "added_hit", "removed_wrong"))
        for name, group in fixed.items()}
