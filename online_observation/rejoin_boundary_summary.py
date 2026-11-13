"""Compare fixed, same-batch MTAD trigger evidence and downstream actions."""
from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from statistics import fmean

from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics
from .hypothesis_gap_diagnostic import _direction
from .rejoin_next_report_summary import _focus
from .rejoin_track_history_summary import _cohorts, _distribution, _truth_split
from .rejoin_witness_squeeze import _truth_side
from .relative_motion_gap import _same


PATHS = ("assimilate", "skip_release", "skip_retain", "skip_both")
ARMS = (*(f"boundary_{path}" for path in PATHS), "boundary_mask_assimilate", "boundary_mask_skip_both",
        "boundary_confirm_assimilate", "boundary_confirm_skip_both", "boundary_robust",
        "boundary_shape_only_mask", "boundary_full_current_mask", "boundary_abstain")
CONTROLS = ("original", "geometry_masked", "current", "native_masked", "native_confirmed", "pair_demeaned")
MARGIN_FIELDS = ("actual_m2", "masked_m2", "increment_m2")


def _changes(rows, arm):
    pairs = Counter((row["decisions"]["current"] is True, row["decisions"][arm] is True) for row in rows)
    return dict(added=pairs[False, True], removed=pairs[True, False])


def _comparison(targets, correct):
    actions = {}
    for arm in (*CONTROLS, *ARMS):
        hits, errors = ([row["decisions"][arm] for row in group] for group in (targets, correct))
        actions[arm] = _metrics(hits.count(True), errors.count(True), len(hits), len(errors)) | dict(
            target_scored=sum(value is not None for value in hits), correct_scored=sum(value is not None for value in errors))
    changes = {arm: dict(target=_changes(targets, arm), correct=_changes(correct, arm),
                         net_delta=actions[arm]["net"] - actions["current"]["net"]) for arm in ARMS}
    return dict(target=len(targets), correct=len(correct), arms=actions, changes_vs_current=changes)


def _compare(targets, correct):
    def matched(row):
        return (row["boundary_report"] is not None and row["boundary_report"]["common_support"]
                and row["decisions"]["current"] is not None and row["decisions"]["geometry_masked"] is not None)

    return dict(full=_comparison(targets, correct), common_support=_comparison(
        [row for row in targets if matched(row)], [row for row in correct if matched(row)]))


def _losses(pairs, *, squared):
    if not pairs:
        return dict(count=0, improves=0, worsens=0, ties=0, paired_delta=_distribution([]))
    before, after = zip(*pairs, strict=True)
    delta = [initial - updated for initial, updated in pairs]
    if squared:
        directions = Counter(map(_direction, delta))
        estimates = dict(assimilate_rmse_m=sqrt(fmean(before)), variant_rmse_m=sqrt(fmean(after)))
    else:
        directions = Counter((value > 0) - (value < 0) for value in delta)
        estimates = dict(assimilate_mean=fmean(before), variant_mean=fmean(after))
    return dict(count=len(pairs), improves=directions[1], worsens=directions[-1], ties=directions[0],
                paired_delta=_distribution(delta), **estimates)


def _paired_scores(rows):
    output = {}
    for arm in PATHS[1:]:
        pairs = defaultdict(list)
        transitions = Counter()
        effects = defaultdict(list)
        mechanisms = Counter()
        counted = 0
        for row in rows:
            scores = row["boundary_report"]["predictions"]["scores"]
            before, after = scores["assimilate"], scores[arm]
            if before["status"] != "supported" or after["status"] != "supported":
                continue
            counted += 1
            side = _truth_side(row)
            for scope, first, second in (("group", before, after), ("trigger", before["local"], after["local"])):
                effect = {name: second[name] - first[name] for name in MARGIN_FIELDS}
                if not _same(effect["actual_m2"], effect["masked_m2"] + effect["increment_m2"]):
                    raise ValueError("the omission effects do not close against the same-time mask")
                for name, difference in effect.items():
                    effects[f"{scope}_{name}"].append(difference)
                for name in MARGIN_FIELDS:
                    old_vote, new_vote = (_direction(score[name]) for score in (first, second))
                    transitions[f"{scope}_{name}:{old_vote}->{new_vote}"] += 1
                    if side is not None:
                        truth = 1 if side == "release" else -1
                        for path, vote in (("assimilate", old_vote), ("variant", new_vote)):
                            correctness = "correct" if vote == truth else "wrong" if vote else "tie"
                            transitions[f"{scope}_{name}:{path}:{correctness}"] += 1
                if side is not None:
                    truth = 1 if side == "release" else -1
                    if _direction(first["actual_m2"]) != truth and _direction(second["actual_m2"]) == truth:
                        for view, score in (("assimilate", first), ("variant", second)):
                            mask_truth = _direction(score["masked_m2"]) == truth
                            increment_truth = _direction(score["increment_m2"]) == truth
                            mechanisms[f"{scope}:recovered:{view}:mask_{int(mask_truth)}:increment_{int(increment_truth)}"] += 1
            if side is None:
                continue
            other = "retain" if side == "release" else "release"
            for role, branch in (("true", side), ("false", other)):
                pairs[f"{role}_incident_m2"].append(tuple(score["edges"][branch]["trigger_m2"] for score in (before, after)))
                pairs[f"{role}_companion_m2"].append(tuple(score["edges"][branch]["companion_m2"] for score in (before, after)))
                pairs[f"{role}_nominal_nll"].append(tuple(score["nominal_nll"][branch] for score in (before, after)))
        output[arm] = dict(windows=counted, signs=dict(transitions),
                           correction_sources=dict(mechanisms),
                           margin_effects_variant_minus_assimilate_m2={name: _distribution(values)
                                                                      for name, values in effects.items()},
                           paired_losses={name: _losses(value, squared=name.endswith("_m2")) for name, value in pairs.items()})
    return output


def _provenance(rows):
    graph, shape = Counter(), Counter()
    geometry = defaultdict(list)
    for row in rows:
        provenance = row["boundary_report"]["provenance"]
        candidate, current = provenance["graph"], provenance["shape"]
        for key in ("trigger_coordinate_read_by_gate", "trigger_coordinate_read_by_kf_cost", "retain_selected",
                    "release_selected", "focal_selected", "previous_retain_selected"):
            graph[key] += bool(candidate[key])
        for key in ("peers", "peers_same_batch", "incident_costs_changed", "fixed_incident_costs_changed"):
            geometry[key].append(candidate[key])
            graph[f"{key}:nonzero"] += bool(candidate[key])
        for scope in ("incident", "fixed_incident", "incident_previous_to_actual", "fixed_incident_previous_to_actual"):
            for gate, changes in candidate[scope].items():
                for key, value in changes.items():
                    graph[f"{scope}:{gate}:{key}"] += value
                    graph[f"{scope}:{gate}:{key}:nonzero"] += bool(value)
        for key in ("factual_cached_margin_m2", "coordinate_mask_cached_margin_m2"):
            if candidate[key] is not None:
                geometry[key].append(candidate[key])
        for key in ("trigger_coordinate_read_by_geometry", "trigger_in_shape_provenance", "current_action",
                    "shape_only_mask_action", "geometry_only_mask_action", "full_coordinate_mask_action"):
            shape[key] += current[key] is True
            shape[f"{key}:unavailable"] += current[key] is None
        shape[f"status:{current['status']}"] += 1
        for key in ("shape_only_mask_action", "geometry_only_mask_action", "full_coordinate_mask_action"):
            shape[f"{key}:differs_from_current"] += current[key] != current["current_action"]
    return dict(graph=dict(graph), shape=dict(shape), graph_distributions={name: _distribution(values)
                                                                         for name, values in geometry.items()})


def _evidence(rows):
    statuses = Counter()
    directions = {arm: Counter() for arm in ARMS}
    clocks = defaultdict(list)
    for row in rows:
        evidence = row["boundary_report"]
        predictions = evidence["predictions"]
        statuses["all_paths_supported"] += evidence["common_support"]
        statuses[predictions["status"]] += 1
        for arm, score in predictions["scores"].items():
            statuses[f"{arm}:{score['status']}"] += 1
        for side, focus in predictions["focals"].items():
            statuses[f"focal_{side}:{focus['status']}"] += 1
            statuses[f"focal_{side}_with_intervening_update"] += bool(focus["intervening_stamps"])
            clocks[f"focal_{side}_age_s"].append(predictions["stamp"] - focus["stamp"])
        truth = _truth_side(row)
        expected = 1 if truth == "release" else -1 if truth == "retain" else None
        for arm, vote in evidence["directions"].items():
            directions[arm][{1: "release", 0: "abstain", -1: "retain"}[vote]] += 1
            if expected is not None and vote:
                directions[arm]["correct" if vote == expected else "wrong"] += 1

    common = [row for row in rows if row["boundary_report"]["common_support"]]
    return dict(statuses=dict(statuses), directions={arm: dict(values) for arm, values in directions.items()},
                clocks={name: _distribution(values) for name, values in clocks.items()},
                provenance=_provenance(rows),
                natural_paired_scores=_paired_scores(rows), common_paired_scores=_paired_scores(common))


def _case(row):
    return dict(scene=row["scene"], component=row["component"], sparse=row["sparse"], start=row["row"]["start"],
                stop=row["row"]["stop"], truth=_truth_side(row), decisions=row["decisions"],
                masked_m2=row["history"]["rolling"]["masked_m2"],
                loo_driving_sides=row["history"]["temporal"]["loo_driving_sides"],
                boundary_report=row["boundary_report"])


def _cohort(rows, *, cases=False):
    targets, correct = _truth_split(rows)
    evidence = [row for row in rows if row["boundary_report"] is not None]
    return dict(windows=len(rows), evidence_windows=len(evidence), sparse=sum(row["sparse"] for row in rows),
                scenes=dict(Counter(row["scene"] for row in rows)), components=dict(Counter(row["component"] for row in rows)),
                truth=dict(Counter(_truth_side(row) or "mixed" for row in rows)), actions=_compare(targets, correct),
                **_evidence(evidence)) | ({"cases": list(map(_case, evidence))} if cases else {})


def _gate(dual, sparse, components, news_errors, true_case):
    main = "boundary_robust"
    dual_changes = dual["full"]["changes_vs_current"][main]
    sparse_changes = sparse["full"]["changes_vs_current"][main]
    differences = {name: values["dual"]["full"]["changes_vs_current"][main]["net_delta"]
                   for name, values in components.items()}
    positive = sum(value > 0 for value in differences.values())
    active_corrections = sum(row["boundary_report"]["directions"][main] == -1 for row in news_errors)
    total_corrections = sum(row["decisions"][main] is False for row in news_errors)
    checks = dict(keeps_unique_added_true=true_case["boundary_report"]["directions"][main] == 1,
                  corrects_a_new_report_error=active_corrections > 0,
                  no_new_dual_errors=dual_changes["correct"]["added"] == 0,
                  no_new_sparse_errors=sparse_changes["correct"]["added"] == 0,
                  dual_net_gain=dual_changes["net_delta"] > 0, sparse_noninferiority=sparse_changes["net_delta"] >= 0,
                  positive_in_multiple_components=positive >= 2)
    return dict(checks=checks, passes=all(checks.values()), positive_components=positive,
                component_net_changes=differences, news_errors_actively_corrected=active_corrections,
                news_errors_removed_including_fallback=total_corrections)


def summarize(rows):
    """Check frozen cohorts and score same-batch contrasts on matched support."""
    eligible = [row for row in rows if row.get("innovation") is not None]
    dual = [row for row in eligible if row.get("common") is not None]
    target, correct = _truth_split(eligible)
    fixed = _cohorts(rows, eligible, dual, target, correct)
    focus = _focus(fixed)
    if any((row["boundary_report"] is not None) != (row["next_report"] is not None) for row in rows):
        raise ValueError("a frozen trigger was added to or lost from the evaluation population")
    sparse = [row for row in rows if row["sparse"]]
    sparse_target = [row for row in sparse if row["row"]["target_edges"] > 0]
    sparse_correct = [row for row in sparse if _primary_control(row["row"])]
    if (len(sparse_target), len(sparse_correct)) != (28, 142):
        raise ValueError("the fixed sparse population changed")

    components = sorted({row["component"] for row in target + correct})
    dual_target, dual_correct = _truth_split(dual)
    by_component = {component: {name: _compare([row for row in t if row["component"] == component],
                                               [row for row in c if row["component"] == component])
                                for name, t, c in (("primary", sparse_target, sparse_correct),
                                                   ("broad", target, correct), ("dual", dual_target, dual_correct))}
                    for component in components}
    primary, dual_result = _compare(sparse_target, sparse_correct), _cohort(dual)
    return dict(primary=primary, broad=_compare(target, correct),
                broad_sparse=_compare([r for r in target if r["sparse"]], [r for r in correct if r["sparse"]]),
                dual=dual_result, by_component=by_component,
                gate=_gate(dual_result["actions"], primary, by_component, focus["new_report_driven"], focus["added_hit"][0])) | {
                    name: _cohort(group, cases=name in focus) for name, group in (fixed | focus).items()}
