"""Summarize fixed MTAD rejoin scores and external innovation controls."""
from __future__ import annotations

from collections import Counter
from math import hypot, sqrt
from statistics import fmean, median

from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics
from .hypothesis_gap_diagnostic import _direction
from .interval_assignment import _dot
from .rejoin_witness_squeeze import _truth_side


SIDES = ("release", "retain")
SCORE_VIEWS = ("raw", "demeaned", "external", "equal_shrink", "shared_external")
ACTION_ARMS = ("pair_demeaned", "donor_confirmed", "matched_shrink", "donor_support_only",
               "donor_qualified", "donor_shared", "abstain")
CONTROLS = ("original", "geometry_masked", "current", "native_masked", "native_actual",
            "native_increment", "native_confirmed")
LOSS_FIELDS = ("innovation_squared_m2", "external_raw_residual_m2", "external_applied_residual_m2",
               "shrink_residual_m2", "demeaned_residual_m2")


def _strict_support(row):
    checks = row.get("qualification") or {}
    return all(checks.get(name) for name in ("supported_both", "distinct_events", "distinct_tracks"))


def _qualified_support(row):
    return _strict_support(row) and row["qualification"]["reduces_both"]


def _compare(target, correct):
    arms = (*CONTROLS, *ACTION_ARMS)

    def action(row, arm):
        decisions = row["decisions"]
        return decisions.get(arm, decisions["current"]) if arm in ACTION_ARMS else decisions.get(arm)

    def count(population, arm):
        return sum(action(row, arm) is True for row in population)

    def changed(population, arm):
        pairs = Counter((row["decisions"]["current"] is True, action(row, arm) is True) for row in population)
        return dict(added=pairs[False, True], removed=pairs[True, False])

    metrics = {arm: _metrics(count(target, arm), count(correct, arm), len(target), len(correct)) | dict(
        target_scored=sum(action(row, arm) is not None for row in target),
        correct_scored=sum(action(row, arm) is not None for row in correct)) for arm in arms}
    changes = {arm: dict(target=changed(target, arm), correct=changed(correct, arm),
                         net_delta=metrics[arm]["net"] - metrics["current"]["net"])
               for arm in ACTION_ARMS}
    main = changes["donor_qualified"]
    return dict(target=len(target), correct=len(correct), arms=metrics, changes_vs_current=changes,
                target_changes=main["target"], correct_changes=main["correct"])


def _comparison(target, correct):
    def common(population, support):
        return [row for row in population if row.get("common") is not None and support(row)
                and row["decisions"]["current"] is not None and row["decisions"]["geometry_masked"] is not None]

    return dict(full=_compare(target, correct),
                dual_support=_compare(common(target, lambda _: True), common(correct, lambda _: True)),
                strict_donor_support=_compare(common(target, _strict_support), common(correct, _strict_support)),
                qualified_donor_support=_compare(common(target, _qualified_support), common(correct, _qualified_support)))


def _distribution(values):
    if not values:
        return dict(count=0, minimum=None, median=None, maximum=None, mean=None)
    return dict(count=len(values), minimum=min(values), median=median(values), maximum=max(values), mean=fmean(values))


def _cosine(first, second):
    norm = hypot(*first) * hypot(*second)
    return max(-1.0, min(1.0, _dot(first, second) / norm)) if norm else None


def _magnitudes(common):
    detail = common["decomposition"]
    pair = [common["reports"][side] for side in SIDES]
    result = {key: detail[key] for key in ("common_increment_m2", "differential_increment_m2")}
    result.update({key: hypot(*detail[f"{key}_m"]) for key in
                   ("mean_innovation", "differential_innovation", "mean_axis", "differential_axis")})
    result["absolute_common_increment_m2"] = abs(detail["common_increment_m2"])
    result["absolute_differential_increment_m2"] = abs(detail["differential_increment_m2"])
    for name, field in (("axis_cosine", "axis_m"), ("innovation_cosine", "innovation_m"), ("donor_cosine", "donor_m")):
        first, second = (item[field] for item in pair)
        result[name] = _cosine(first, second) if first is not None and second is not None else None
    return result


def _losses(rows):
    pairs, raw_improves, applied_improves, applied_beats_shrink = [], 0, 0, 0
    for row in rows:
        for report in row["common"]["reports"].values():
            if report is None or report["external_raw_residual_m2"] is None:
                continue
            pairs.append(report)
            raw_improves += _direction(report["external_raw_reduction_m2"]) > 0
            applied_improves += _direction(report["external_applied_reduction_m2"]) > 0
            applied_beats_shrink += _direction(report["shrink_residual_m2"] - report["external_applied_residual_m2"]) > 0

    def loss(field):
        values = [record[field] for record in pairs]
        return dict(mean_m2=fmean(values), rmse_m=sqrt(fmean(values))) if values else dict(mean_m2=None, rmse_m=None)

    return dict(reports=len(pairs), losses={field: loss(field) for field in LOSS_FIELDS},
                raw_improves=raw_improves, applied_improves=applied_improves,
                applied_beats_equal_shrink=applied_beats_shrink)


def _signs(score):
    return {side: _direction(score[f"{side}_change_m2"]) for side in SIDES}


def _score_summary(rows, arm):
    votes, per_report_changes, grouped_changes = Counter(), Counter(), Counter()
    for row in rows:
        truth = _truth_side(row)
        score = row["common"][arm]
        confirmed = row["confirmed"][arm]
        for report in score["reports"].values():
            if report is not None:
                signs = _signs(report)
                per_report_changes[f"{signs['release']},{signs['retain']}"] += 1
                if truth is not None:
                    false_side = "retain" if truth == "release" else "release"
                    per_report_changes[f"truth_order:{signs[truth]},{signs[false_side]}"] += 1
        if score["support_count"] < 2:
            continue
        signs = _signs(score)
        grouped_changes[f"{signs['release']},{signs['retain']}"] += 1
        for label, vote in (("actual", _direction(score["actual_m2"])),
                            ("increment", _direction(score["increment_m2"])), ("confirmed", confirmed)):
            votes[f"{label}:{vote}"] += 1
            if truth is not None:
                target = 1 if truth == "release" else -1
                votes[f"{label}_{'correct' if vote == target else 'wrong' if vote else 'zero'}"] += 1
    return dict(votes=dict(votes), per_report_change_signs=dict(per_report_changes),
                pair_change_signs=dict(grouped_changes))


def _record_identity(row, counts):
    identity = row.get("donor_identity")
    if not identity:
        return
    candidates = {side: len(row["witnesses"][side]["donors"]) for side in SIDES}
    for key in ("within_group", "within_pair"):
        minimum = 1 if key == "within_group" else 2
        for side, value in identity[key].items():
            if candidates[side] >= minimum:
                counts[f"{key}_reports"] += 1
                counts[f"{key}_overlapping_reports"] += value > 0
                counts[f"{key}_overlapping_identities"] += value
    if all(candidates.values()):
        counts["across_reports_windows"] += 1
        counts["across_reports_overlapping_windows"] += identity["across_reports"] > 0
        counts["across_reports_overlapping_identities"] += identity["across_reports"]


def _mechanism(rows):
    dual = [row for row in rows if row.get("common") is not None]
    statuses, qualifications, geometry, raw_demeaned, identities, donor_cover = (Counter() for _ in range(6))
    values = {}
    for row in dual:
        common = row["common"]
        for name, value in _magnitudes(common).items():
            if value is not None:
                values.setdefault(name, []).append(value)
        detail = common["decomposition"]
        common_part = detail["common_increment_m2"]
        differential = detail["differential_increment_m2"]
        geometry[f"component_sign:{_direction(common_part)},{_direction(differential)}"] += 1
        geometry["common_dominates"] += abs(common_part) > abs(differential)
        geometry["differential_dominates"] += abs(differential) > abs(common_part)
        geometry["same_magnitude"] += abs(differential) == abs(common_part)
        if common["donor_dot_m2"] is not None:
            geometry[f"donor_dot_sign:{_direction(common['donor_dot_m2'])}"] += 1
        raw, demeaned = common["raw"], common["demeaned"]
        for name in ("actual_m2", "increment_m2"):
            raw_demeaned[f"{name}:{_direction(raw[name])}->{_direction(demeaned[name])}"] += 1
        raw_demeaned[f"confirmation:{row['confirmed']['raw']}->{row['confirmed']['demeaned']}"] += 1
        for report_side in SIDES:
            original_sign = _direction(raw["reports"][report_side]["increment_m2"])
            residual_sign = _direction(demeaned["reports"][report_side]["increment_m2"])
            raw_demeaned[f"report_{report_side}_increment:{original_sign}->{residual_sign}"] += 1
        checks = row["qualification"]
        for key in ("supported_both", "distinct_events", "distinct_tracks", "reduces_both"):
            qualifications[key] += bool(checks[key])
        qualifications["strict_support"] += _strict_support(row)
        qualifications["qualified_support"] += _qualified_support(row)
        qualifications["same_event_on_both_reports"] += bool(checks["supported_both"] and not checks["distinct_events"])
        qualifications["same_track_on_both_reports"] += bool(checks["supported_both"] and not checks["distinct_tracks"])
        selected_roles = [side for side in SIDES if common["reports"][side]["donor_m"] is not None]
        donor_cover["+".join(selected_roles) or "neither"] += 1
        for side, witness in row["witnesses"].items():
            statuses[witness["status"]] += 1
            statuses[f"{side}:{witness['status']}"] += 1
            statuses["same_batch_external_reports"] += witness["same_batch_count"]
            values.setdefault("near_eligible_count", []).append(witness["near_count"])
            for donor in (witness.get("donors") or ()):
                values.setdefault("donor_prior_age_s", []).append(donor["prior_age_s"])
                values.setdefault("donor_distance_m", []).append(donor["distance_m"])
        _record_identity(row, identities)

    strict = [row for row in dual if _strict_support(row)]
    return dict(dual_windows=len(dual), witness_status=dict(statuses), donor_cover_by_window=dict(donor_cover),
                qualification=dict(qualifications),
                donor_identity=dict(identities), geometry=dict(geometry),
                magnitudes={key: _distribution(data) for key, data in values.items()},
                raw_to_demeaned=dict(raw_demeaned), scores={arm: _score_summary(dual, arm) for arm in SCORE_VIEWS},
                donor_loss=_losses(dual), strict_donor_loss=_losses(strict))


def _public_case(row):
    witnesses = {side: dict(status=item["status"], near_count=item["near_count"],
                           same_batch_count=item["same_batch_count"], donors=[
                               {key: donor[key] for key in ("stamp", "prior_age_s", "distance_m", "vector_m")}
                               for donor in (item.get("donors") or ())])
                 for side, item in row["witnesses"].items()}
    return dict(scene=row["scene"], component=row["component"], sparse=row["sparse"],
                start=row["row"]["start"], stop=row["row"]["stop"], truth_side=_truth_side(row),
                common=row["common"], witnesses=witnesses, qualification=row["qualification"],
                donor_identity=row.get("donor_identity"), confirmed=row["confirmed"], decisions=row["decisions"])


def _cohort(rows, cases=False):
    target = [row for row in rows if _truth_side(row) == "release"]
    correct = [row for row in rows if _truth_side(row) == "retain"]
    output = dict(windows=len(rows), sparse=sum(row["sparse"] for row in rows),
                  truth=dict(Counter(_truth_side(row) or "mixed" for row in rows)),
                  scenes=dict(Counter(row["scene"] for row in rows)),
                  components=dict(Counter(row["component"] for row in rows)),
                  mechanism=_mechanism(rows), actions=_comparison(target, correct))
    return output | ({"cases": list(map(_public_case, rows))} if cases else {})


def summarize(rows):
    """Join previously fixed populations to pre-label causal donor readouts."""
    eligible = [row for row in rows if row.get("innovation") is not None]
    dual = [row for row in eligible if row["innovation"]["rolling"]["support_count"] == 2]
    target = [row for row in eligible if _truth_side(row) == "release"]
    correct = [row for row in eligible if _truth_side(row) == "retain"]
    sparse = [row for row in rows if row["sparse"]]
    primary_target = [row for row in sparse if row["row"]["target_edges"] > 0]
    primary_correct = [row for row in sparse if _primary_control(row["row"])]
    if not all(row.get("common") is not None for row in dual):
        raise ValueError("a C51 dual readout was not carried forward")
    changed = [row for row in dual if row["decisions"]["native_confirmed"] is True
               and row["decisions"]["current"] is not True]
    both_farther = [row for row in changed if _truth_side(row) == "retain"
                    and _direction(row["innovation"]["rolling"]["release_change_m2"]) > 0]
    false_closer = [row for row in changed if _truth_side(row) == "retain"
                    and _direction(row["innovation"]["rolling"]["release_change_m2"]) < 0]
    added_hit = [row for row in changed if _truth_side(row) == "release"]
    if (len(rows), len(eligible), len(dual), len(target), len(correct), len(primary_target), len(primary_correct),
        len(both_farther), len(false_closer), len(added_hit)) != (482, 416, 31, 67, 308, 28, 142, 8, 3, 1):
        raise ValueError("a frozen C51 evaluation or mechanism population changed")

    def component_subset(population, component):
        return [row for row in population if row["component"] == component]

    changed_correct = {id(row) for row in both_farther + false_closer}
    components = sorted({row["component"] for row in target + correct})
    return dict(primary=_comparison(primary_target, primary_correct), broad=_comparison(target, correct),
                broad_sparse=_comparison([row for row in target if row["sparse"]],
                                         [row for row in correct if row["sparse"]]),
                by_component={component: dict(
                    primary=_comparison(component_subset(primary_target, component), component_subset(primary_correct, component)),
                    broad=_comparison(component_subset(target, component), component_subset(correct, component)),
                ) for component in components},
                dual=_cohort(dual), both_farther=_cohort(both_farther, cases=True),
                false_closer=_cohort(false_closer, cases=True), added_hit=_cohort(added_hit, cases=True),
                correct_rejoins=_cohort(correct),
                current_false_vetoes=_cohort([row for row in correct if row["decisions"]["current"] is True]),
                other_correct_rejoins=_cohort([row for row in correct if id(row) not in changed_correct]),
                mixed=_cohort([row for row in eligible if _truth_side(row) is None]))
