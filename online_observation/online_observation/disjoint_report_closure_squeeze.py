"""Summarize frozen two-report closure opportunities and paired assignment decisions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from .async_report_scores import reported_direction
from .async_report_squeeze import _primary_control
from .hypothesis_gap_diagnostic import _direction as closure_direction


KINDS = ("disjoint", "repeat", "reciprocal_release", "reciprocal_retain")
SINGLE_ARMS = ("raw", "masked", "corroborated", "hold")
PAIR_ARMS = ("actual", "masked", "earlier_masked", "both_masked", "two_votes", "new_support", "both_add")
CONTROLS = ("raw", "masked", "corroborated", "hold")


def _median(values):
    return median(values) if values else None


def _score(snapshot, pair, arm):
    if arm.startswith("single_"):
        control = arm.removeprefix("single_")
        if control == "hold":
            return -1
        return reported_direction(snapshot["single"][control]) if snapshot is not None else None
    if pair is None:
        return None
    if arm == "two_votes":
        return 1 if pair[arm] else -1
    if arm in ("new_support", "both_add"):
        positive = closure_direction(pair["actual"]) > 0 and closure_direction(pair["delta"]) > 0
        if arm == "both_add":
            positive = positive and closure_direction(pair["actual"]-pair["earlier_masked"]) > 0
        return 1 if positive else -1
    return closure_direction(pair[arm])


def _counts(entries, arm):
    signs, truth = Counter(), Counter()
    leads = []
    for row, snapshot, pair in entries:
        decision = _score(snapshot, pair, arm)
        signs[decision] += 1
        if (row["true_release"], row["true_retain"]) in ((2, 0), (0, 2)):
            expected = 1 if row["true_release"] == 2 else -1
            truth["unavailable" if decision is None else "tied" if decision == 0 else
                  "correct" if decision == expected else "wrong"] += 1
        if decision == 1:
            lead = row["stop"]-snapshot["stamp"]
            if lead < 0:
                raise ValueError("closure used a report later than the original rejoin")
            leads.append((lead, row["target_edges"] or row["focal"]))
    return dict(warnings=signs[1], retain=signs[-1], ties=signs[0], unavailable=signs[None],
                warning_edges=sum(edges for _, edges in leads),
                median_lead_s=_median([lead for lead, _ in leads]),
                lead_edge_seconds=sum(lead*edges for lead, edges in leads), pure_truth=dict(truth))


def _changes(entries, arm, control):
    counts = Counter((_score(snapshot, pair, control) == 1, _score(snapshot, pair, arm) == 1)
                     for _, snapshot, pair in entries)
    return dict(added=counts[False, True], removed=counts[True, False],
                inherited=counts[True, True], stayed_clear=counts[False, False])


def _support(entries):
    usable = [(row, snapshot, pair) for row, snapshot, pair in entries if pair is not None]
    lags = [pair["lag_s"] for _, _, pair in usable]
    windows = [row["stop"]-row["start"] for row, _, _ in usable]
    reuse = Counter()
    for _, _, pair in usable:
        raw = pair["reused_older_report"]
        if isinstance(raw, dict):
            reuse.update({side: int(bool(value)) for side, value in raw.items()})
            reuse["any"] += any(raw.values())
        else:
            reuse["any"] += bool(raw)
    return dict(available=len(usable), median_lag_s=_median(lags), min_lag_s=min(lags, default=None),
                max_lag_s=max(lags, default=None), median_window_s=_median(windows),
                reused_older_report=dict(reuse),
                median_opposite_ages_s={side: [
                    _median([pair["ages"][side][index] for _, _, pair in usable if "ages" in pair])
                    for index in (0, 1)] for side in ("release", "retain")})


def _available_stages(rows, kind):
    counts = Counter((bool(_select(row, "first_eligible", kind)),
                      bool((boundary := _select(row, "boundary", kind)) and kind in boundary["pairs"]))
                     for _, _, row in rows)
    return dict(strict_only=counts[True, False], strict_and_boundary=counts[True, True],
                boundary_only=counts[False, True], neither=counts[False, False])


def _select(row, moment, kind):
    reports = row["reports"]
    stop = row["stop"]
    if moment == "latest":
        return next((report for report in reversed(reports) if report["stamp"] < stop), None)
    if moment == "boundary":
        return next((report for report in reversed(reports) if report["stamp"] == stop), None)
    return next((report for report in reports if report["stamp"] < stop and kind in report["pairs"]), None)


def _cohort(rows, moment, kind):
    entries = []
    for _, _, row in rows:
        snapshot = _select(row, moment, kind)
        entries.append((row, snapshot, snapshot["pairs"].get(kind) if snapshot is not None else None))
    matched = [entry for entry in entries if entry[2] is not None]
    arms = (*PAIR_ARMS, *(f"single_{arm}" for arm in SINGLE_ARMS))
    controls = (*PAIR_ARMS[1:5], *(f"single_{arm}" for arm in CONTROLS))
    comparisons = {f"{arm}_vs_{control}": _changes(matched, arm, control)
                   for arm in ("actual", "new_support", "both_add") for control in controls}
    return dict(windows=len(entries), scored_reports=sum(sample is not None for _, sample, _ in entries),
                **_support(entries), full={arm: _counts(entries, arm) for arm in arms},
                matched={arm: _counts(matched, arm) for arm in arms}, paired=comparisons)


def _first_warning(rows, kind):
    result = {}
    for arm in PAIR_ARMS:
        samples = []
        for _, _, row in rows:
            hit = next((report for report in row["reports"] if report["stamp"] < row["stop"]
                        and _score(report, report["pairs"].get(kind), arm) == 1), None)
            if hit is not None:
                samples.append(row["stop"]-hit["stamp"])
        result[arm] = dict(warnings=len(samples), median_lead_s=_median(samples),
                           min_lead_s=min(samples, default=None), max_lead_s=max(samples, default=None))
    return result


def _metrics(hit, false, n_target, n_correct):
    tpr, fpr = (hit/n_target if n_target else None), (false/n_correct if n_correct else None)
    return dict(hits=hit, false_vetoes=false, net=hit-false, target_windows=n_target, correct_windows=n_correct,
                hit_rate=tpr, false_veto_rate=fpr, balanced_accuracy=(tpr+1-fpr)/2
                if tpr is not None and fpr is not None else None)


def _comparison(target, correct):
    moments = {}
    for moment in ("latest", "boundary", "first_eligible"):
        kinds = {}
        for kind in KINDS:
            hit, false = (_cohort(population, moment, kind) for population in (target, correct))
            if moment == "first_eligible":
                hit["first_warning"], false["first_warning"] = (_first_warning(population, kind)
                                                               for population in (target, correct))
                hit["availability_stages"], false["availability_stages"] = (_available_stages(population, kind)
                                                                            for population in (target, correct))
            counts = {mode: {arm: _metrics(hit[mode][arm]["warnings"], false[mode][arm]["warnings"],
                                           hit[denominator], false[denominator]) for arm in hit[mode]}
                      for mode, denominator in (("full", "windows"), ("matched", "available"))}
            kinds[kind] = dict(target=hit, correct_rejoin=false, **counts)
        moments[moment] = kinds
    return dict(target_windows=len(target), target_focal_edges=sum(row["target_edges"] for _, _, row in target),
                correct_windows=len(correct), target_components=len({scene["component"] for _, scene, _ in target}),
                correct_components=len({scene["component"] for _, scene, _ in correct}), moments=moments)


def _supply(entries):
    output = {}
    for kind in KINDS:
        output[kind] = {}
        for moment in ("latest", "boundary", "first_eligible"):
            picked = [(row, _select(row, moment, kind)) for _, _, row in entries]
            output[kind][moment] = _support([(row, snapshot, snapshot["pairs"].get(kind) if snapshot else None)
                                             for row, snapshot in picked])
        counts = Counter()
        for _, _, row in entries:
            for snapshot in row["reports"]:
                pair = snapshot["pairs"].get(kind)
                if pair is not None:
                    phase = "strict" if snapshot["stamp"] < row["stop"] else "boundary"
                    counts[phase] += 1
                    reuse = pair["reused_older_report"]
                    counts[f"{phase}_reuse"] += any(reuse.values()) if isinstance(reuse, dict) else bool(reuse)
        output[kind]["report_counts"] = dict(counts)
        output[kind]["availability_stages"] = _available_stages(entries, kind)
    return dict(windows=len(entries), kinds=output)


def summarize(summary):
    rows = [(alias, scene, row) for alias, scene in summary["scenes"].items() for row in scene["rows"]]
    sparse = [entry for entry in rows if entry[1]["sparse"]]
    target = [entry for entry in sparse if entry[2]["target_edges"] > 0]
    correct = [entry for entry in sparse if _primary_control(entry[2])]
    switched_target = [entry for entry in target if entry[2]["switch_source"] is True]
    switched_correct = [entry for entry in correct if entry[2]["switch_source"] is True]
    components = {scene["component"] for _, scene, _ in target}
    same = [entry for entry in correct if entry[1]["component"] in components]
    return dict(primary=_comparison(target, correct), source_switch=_comparison(switched_target, switched_correct),
                same_components=_comparison(target, same), same_components_switch=_comparison(
                    switched_target, [entry for entry in switched_correct if entry[1]["component"] in components]),
                no_source_switch=_comparison([entry for entry in target if entry[2]["switch_source"] is False],
                                             [entry for entry in correct if entry[2]["switch_source"] is False]),
                by_component={component: _comparison([entry for entry in target if entry[1]["component"] == component],
                                                     [entry for entry in correct if entry[1]["component"] == component])
                              for component in sorted(components)},
                all_windows=_supply(rows), sparse_windows=_supply(sparse))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = summarize(json.loads(args.summary.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
