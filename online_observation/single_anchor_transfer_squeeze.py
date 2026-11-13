"""Summarize fixed-window cross-target anchor transfer and assignment margin effects."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import sqrt
from pathlib import Path
from statistics import fmean, median

from .async_report_scores import reported_direction
from .async_report_squeeze import _primary_control


ARMS = ("raw", "masked", "raw_agreement", "always", "always_masked", "corroborated", "hold")
MOMENTS = ("onset", "later_first", "later_last", "latest", "boundary")


def _mean(values):
    return fmean(values) if values else None


def _median(values):
    return median(values) if values else None


def _warning(snapshot, arm):
    score = snapshot["scores"].get(arm) if snapshot else None
    return score is not None and reported_direction(score) > 0


def _supported(snapshot):
    return bool(snapshot and snapshot.get("anchor") is not None)


def _clean(snapshot):
    return _supported(snapshot) and snapshot.get("anchor_correct") is True and snapshot.get("anchor_overlap") is False


def _entry_counts(entries, arm):
    active = [(row, snapshot) for _, _, row, snapshot in entries if _warning(snapshot, arm)]
    leads = [row["stop"] - snapshot["stamp"] for row, snapshot in active]
    if any(lead < 0 for lead in leads):
        raise ValueError("a report occurs beyond the original window")
    return dict(warnings=len(active), warning_edges=sum(row["target_edges"] or row["focal"] for row, _ in active),
                warning_rate=len(active) / len(entries) if entries else None,
                median_lead_s=_median(leads),
                lead_edge_seconds=sum(lead * (row["target_edges"] or row["focal"])
                                      for lead, (row, _) in zip(leads, active, strict=True)))


def _changes(entries, arm, reference):
    changes = Counter((_warning(sample, reference), _warning(sample, arm)) for _, _, _, sample in entries)
    return dict(added=changes[False, True], removed=changes[True, False], inherited=changes[True, True],
                stayed_clear=changes[False, False])


def _decomposition(entries):
    counts = Counter()
    true_squared, corrected_squared = [], []
    true_costs, false_costs, truth_margin_deltas, equal_margin_deltas = [], [], [], []
    linear_effects, quadratic_effects = [], []
    for _, _, _, snapshot in entries:
        if not _supported(snapshot):
            continue
        true_side = snapshot.get("true_side")
        if true_side not in ("release", "retain"):
            counts["unknown_true_side"] += 1
            continue
        terms = snapshot.get("decomposition")
        if not terms:
            raise ValueError("a scored anchor is missing its candidate residual decomposition")
        false_side = "retain" if true_side == "release" else "release"
        truth, false = terms[true_side], terms[false_side]
        true_squared.append(truth["raw_squared"])
        corrected_squared.append(truth["adjusted_squared"])
        true_reduction = truth["raw_squared"] - truth["adjusted_squared"]
        false_reduction = false["raw_squared"] - false["adjusted_squared"]
        true_cost, false_cost = -truth["delta_cost"], -false["delta_cost"]
        true_costs.append(true_cost)
        false_costs.append(false_cost)
        native_delta = true_cost - false_cost
        common_delta = true_reduction - false_reduction
        truth_margin_deltas.append(native_delta)
        equal_margin_deltas.append(common_delta)
        linear_effects.append(false["linear"] - truth["linear"])
        quadratic_effects.append(false["quadratic"] - truth["quadratic"])
        true_improves = true_reduction > 1e-6
        false_gains_more = native_delta < -1e-9
        equal_false_gains_more = common_delta < -1e-6
        wrong_candidate_release = true_side == "retain"
        raw_wrong = _warning(snapshot, "raw") == wrong_candidate_release
        corrected_wrong = _warning(snapshot, "always") == wrong_candidate_release
        equal = snapshot["equal_variance"]
        equal_raw_wrong = (reported_direction(equal["raw"]) > 0) == wrong_candidate_release
        equal_corrected_wrong = (reported_direction(equal["always"]) > 0) == wrong_candidate_release
        counts.update({
            "truth_residual_improved": int(true_improves),
            "false_residual_improved": int(false_reduction > 1e-6),
            "false_gained_more_native": int(false_gains_more),
            "false_gained_more_equal": int(equal_false_gains_more),
            "truth_improved_false_gained_more_native": int(true_improves and false_gains_more),
            "truth_improved_false_gained_more_equal": int(true_improves and equal_false_gains_more),
            "native_only_margin_deterioration": int(false_gains_more and not equal_false_gains_more),
            "equal_only_margin_deterioration": int(equal_false_gains_more and not false_gains_more),
            "new_wrong_choice": int(not raw_wrong and corrected_wrong),
            "truth_improved_new_wrong_choice": int(true_improves and not raw_wrong and corrected_wrong),
            "inherited_wrong_choice": int(raw_wrong and corrected_wrong),
            "truth_improved_inherited_wrong_choice": int(true_improves and raw_wrong and corrected_wrong),
            "fixed_wrong_choice": int(raw_wrong and not corrected_wrong),
            "new_wrong_choice_equal": int(not equal_raw_wrong and equal_corrected_wrong),
            "fixed_wrong_choice_equal": int(equal_raw_wrong and not equal_corrected_wrong),
            "new_native_wrong_equal_right": int(not raw_wrong and corrected_wrong and not equal_corrected_wrong),
            "corrected_native_wrong_equal_right": int(corrected_wrong and not equal_corrected_wrong),
            "corrected_equal_wrong_native_right": int(equal_corrected_wrong and not corrected_wrong),
        })
    return dict(n=len(true_squared), counts=dict(counts),
                raw_truth_rmse=sqrt(fmean(true_squared)) if true_squared else None,
                corrected_truth_rmse=sqrt(fmean(corrected_squared)) if corrected_squared else None,
                mean_true_cost_reduction=_mean(true_costs), mean_false_cost_reduction=_mean(false_costs),
                mean_truth_margin_delta=_mean(truth_margin_deltas),
                mean_truth_margin_linear=_mean(linear_effects), mean_truth_margin_quadratic=_mean(quadratic_effects),
                mean_equal_squared_margin_delta=_mean(equal_margin_deltas))


def _cohort(entries, *, first=False):
    anchors = [entry for entry in entries if _supported(entry[3])]
    clean = [entry for entry in anchors if _clean(entry[3])]
    base = dict(windows=len(entries), snapshots=sum(sample is not None for _, _, _, sample in entries),
                anchored=len(anchors), clean_anchored=len(clean),
                anchor_wrong=sum(sample.get("anchor_correct") is False for _, _, _, sample in anchors),
                anchor_overlaps=sum(sample.get("anchor_overlap") is True for _, _, _, sample in anchors),
                median_anchor_age_s=_median([sample["anchor_age"] for _, _, _, sample in anchors]),
                reasons=dict(Counter((sample.get("reason") or "unspecified") if sample else "missing_snapshot"
                                     for _, _, _, sample in entries)),
                full={arm: _entry_counts(entries, arm) for arm in ARMS},
                matched={arm: _entry_counts(anchors, arm) for arm in ARMS},
                clean_matched={arm: _entry_counts(clean, arm) for arm in ARMS},
                matched_changes={arm: _changes(anchors, arm, "raw") for arm in ("always", "corroborated")},
                corroboration_vs_unanchored_agreement=_changes(anchors, "corroborated", "raw_agreement"),
                decomposition=_decomposition(anchors), clean_decomposition=_decomposition(clean))
    if first:
        leads = [row["stop"] - row["first_corroborated"] for _, _, row, _ in entries
                 if row.get("first_corroborated") is not None]
        if any(lead <= 0 for lead in leads):
            raise ValueError("first corroborated warning must occur strictly before original rejoin")
        base["first_corroborated"] = dict(warnings=len(leads), median_lead_s=_median(leads))
    return base


def _metrics(hit, false, nt, nc):
    return dict(hits=hit, false_vetoes=false, net=hit-false, target_windows=nt, correct_windows=nc,
                hit_rate=hit/nt if nt else None, false_veto_rate=false/nc if nc else None,
                balanced_accuracy=(hit/nt+1-false/nc)/2 if nt and nc else None)


def _comparison(target, correct, moments=MOMENTS):
    stages = {}
    for moment in moments:
        target_summary, correct_summary = (
            _cohort([(alias, scene, row, row.get(moment)) for alias, scene, row in cohort], first=moment == "latest")
            for cohort in (target, correct)
        )
        comparisons = {}
        for name, population in (("full", "windows"), ("matched", "anchored"), ("clean_matched", "clean_anchored")):
            comparisons[name] = {arm: _metrics(target_summary[name][arm]["warnings"], correct_summary[name][arm]["warnings"],
                                                target_summary[population], correct_summary[population]) for arm in ARMS}
        stages[moment] = dict(target=target_summary, correct_rejoin=correct_summary, comparison=comparisons)
    return dict(target_windows=len(target), correct_windows=len(correct),
                target_components=len({scene["component"] for _, scene, _ in target}),
                correct_components=len({scene["component"] for _, scene, _ in correct}), moments=stages)


def _unit_key(unit, alias, scene, snapshot):
    component = scene["component"]
    if unit == "component":
        return (component,)
    if snapshot is None:
        return None
    anchor, target = snapshot.get("anchor"), snapshot.get("target_identity")
    if unit == "anchor_pair" and anchor is not None:
        return component, alias, str(anchor)
    if unit == "target_identity" and target is not None:
        return component, alias, str(target)
    if unit == "anchor_target" and anchor is not None and target is not None:
        return component, alias, str(anchor), str(target)
    return None


def _weighting(entries, unit):
    groups = defaultdict(list)
    for entry in entries:
        key = _unit_key(unit, entry[0], entry[1], entry[3])
        if key is not None:
            groups[key].append(entry)
    rows = {}
    for key, group in sorted(groups.items()):
        rows["/".join(key)] = dict(
            windows=len(group), anchored=sum(_supported(sample) for _, _, _, sample in group),
            warning_rates={arm: _entry_counts(group, arm)["warning_rate"] for arm in ARMS},
            decomposition=_decomposition(group),
            anchor_physical_identities=len({sample["anchor_identity"] for _, _, _, sample in group
                                            if sample and sample.get("anchor_identity") is not None}),
            target_physical_identities=len({sample["target_identity"] for _, _, _, sample in group
                                            if sample and sample.get("target_identity") is not None}),
        )
    mean_rates = {arm: _mean([row["warning_rates"][arm] for row in rows.values()]) for arm in ARMS}
    results = [row["decomposition"] for row in rows.values() if row["decomposition"]["n"]]
    errors = {arm: sqrt(fmean(row[arm]**2 for row in results)) if results else None
              for arm in ("raw_truth_rmse", "corrected_truth_rmse")}
    return dict(units=len(rows), with_truth_units=len(results), covered_windows=sum(row["windows"] for row in rows.values()),
                mean_warning_rates=mean_rates, equal_unit_errors=errors, details=rows)


def _deduplicated(target, correct):
    populations = [[(alias, scene, row, row.get("latest")) for alias, scene, row in cohort] for cohort in (target, correct)]
    result = {}
    for unit in ("anchor_pair", "target_identity", "anchor_target", "component"):
        hit, veto = (_weighting(entries, unit) for entries in populations)
        result[unit] = dict(target=hit, correct_rejoin=veto)
        result[unit]["balanced_accuracy"] = {
            arm: (hit["mean_warning_rates"][arm]+1-veto["mean_warning_rates"][arm])/2
            if hit["units"] and veto["units"] else None for arm in ARMS}
    return result


def summarize(summary):
    entries = [(alias, scene, row) for alias, scene in summary["scenes"].items() for row in scene["rows"]]
    sparse = [entry for entry in entries if entry[1]["sparse"]]
    target = [entry for entry in sparse if entry[2]["target_edges"] > 0]
    correct = [entry for entry in sparse if _primary_control(entry[2])]
    switch_target = [entry for entry in target if entry[2]["switch_source"] is True]
    switch_correct = [entry for entry in correct if entry[2]["switch_source"] is True]
    components = {scene["component"] for _, scene, _ in switch_target}
    same = [entry for entry in switch_correct if entry[1]["component"] in components]
    late = ("latest", "boundary")
    return dict(
        window_population=dict(all=len(entries), sparse=len(sparse)),
        primary=_comparison(switch_target, switch_correct),
        same_components=_comparison(switch_target, same, moments=late),
        all_rejoin=_comparison(target, correct, moments=late),
        no_switch=_comparison([entry for entry in target if entry[2]["switch_source"] is False],
                              [entry for entry in correct if entry[2]["switch_source"] is False], moments=late),
        by_component={component: _comparison([entry for entry in switch_target if entry[1]["component"] == component],
                                             [entry for entry in switch_correct if entry[1]["component"] == component],
                                             moments=late)
                      for component in sorted(components | {scene["component"] for _, scene, _ in switch_correct})},
        deduplicated_latest=dict(primary=_deduplicated(switch_target, switch_correct),
                                 same_components=_deduplicated(switch_target, same)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(json.loads(args.summary.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
