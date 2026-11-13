"""Summarize fixed-clock forecast deletion and report-coordinate controls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from .async_report_scores import reported_direction
from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics
from .hypothesis_gap_diagnostic import _direction


KINDS = ("combined", "reciprocal_release", "reciprocal_retain")
PAIR_FIELDS = {
    "original_actual": ("original", "actual"),
    "original_masked": ("original", "masked"),
    "earlier_masked": ("original", "earlier_masked"),
    "both_masked": ("original", "both_masked"),
    "removed_actual": ("omitted", "actual"),
    "removed_masked": ("omitted", "masked"),
    "removed_and_earlier_masked": ("omitted", "earlier_masked"),
    "removed_and_both_masked": ("omitted", "both_masked"),
}
SUPPORT_ARMS = {
    "original_new_support": ("original_actual", "original_masked"),
    "earlier_masked_new_support": ("earlier_masked", "both_masked"),
    "removed_new_support": ("removed_actual", "removed_masked"),
    "removed_and_earlier_masked_new_support": ("removed_and_earlier_masked", "removed_and_both_masked"),
}
SINGLE_FIELDS = ("raw", "masked", "corroborated", "omitted", "omitted_masked")
ARMS = (*PAIR_FIELDS, *SUPPORT_ARMS, *(f"single_{field}" for field in SINGLE_FIELDS), "hold")
PAIRED = {
    "removed_actual": ("original_actual", "earlier_masked", "removed_masked"),
    "removed_new_support": ("original_new_support", "earlier_masked_new_support", "removed_masked"),
    "removed_and_earlier_masked": ("earlier_masked", "removed_actual"),
    "removed_and_earlier_masked_new_support": ("earlier_masked_new_support", "removed_new_support"),
    "single_omitted": ("single_raw", "single_masked", "single_omitted_masked"),
}


def _pick(report, kind):
    if kind != "combined":
        return report["pairs"].get(kind)
    return max(report["pairs"].values(), default=None,
               key=lambda item: (item["original"]["older_stamp"], item["original"]["kind"]))


def _pick_moment(row, moment, kind):
    stop = row["stop"]
    reports = row["reports"]
    if moment == "latest":
        return next((sample for sample in reversed(reports) if sample["stamp"] < stop), None)
    if moment == "boundary":
        return next((sample for sample in reversed(reports) if sample["stamp"] == stop), None)
    return next((sample for sample in reports if sample["stamp"] < stop and _pick(sample, kind)), None)


def _omitted(pair):
    removal = pair["removal"]
    return removal["omitted"] if removal["status"] == "supported" else None


def _values(report, pair):
    values = dict.fromkeys(ARMS)
    values["hold"] = -1
    if report is not None:
        for field in ("raw", "masked", "corroborated"):
            values[f"single_{field}"] = report["single"][field]
    if pair is None:
        return values
    pair_views = {"original": pair["original"], "omitted": _omitted(pair)}
    for arm, (view, field) in PAIR_FIELDS.items():
        source = pair_views[view]
        if source is not None:
            values[arm] = source[field]
    for arm, (actual, masked) in SUPPORT_ARMS.items():
        if values[actual] is not None:
            values[arm] = int(_direction(values[actual]) > 0 and _direction(values[actual]-values[masked]) > 0)
    if pair_views["omitted"] is not None:
        values["single_omitted"] = pair["removal"].get("single_omitted")
        values["single_omitted_masked"] = pair["removal"].get("single_omitted_masked")
    return values


def _positive(arm, value):
    return value is not None and (reported_direction(value) if arm.startswith("single_") else _direction(value)) > 0


def _expected(row):
    true_sides = row["true_release"], row["true_retain"]
    return 1 if true_sides == (2, 0) else -1 if true_sides == (0, 2) else 0


def _median(values):
    return median(values) if values else None


def _arm_counts(entries, arm):
    warnings = [(row["stop"]-report["stamp"], row["target_edges"] or row["focal"])
                for row, report, _, values in entries if _positive(arm, values[arm])]
    if any(lead < 0 for lead, _ in warnings):
        raise ValueError("a warning depends on a report after the original window")
    return dict(warnings=len(warnings), scored=sum(values[arm] is not None for _, _, _, values in entries),
                warning_edges=sum(edges for _, edges in warnings), median_lead_s=_median([lead for lead, _ in warnings]),
                lead_edge_seconds=sum(lead*edges for lead, edges in warnings))


def _changes(entries, arm, control):
    counts = Counter((_positive(control, values[control]), _positive(arm, values[arm]))
                     for _, _, _, values in entries)
    return dict(added=counts[False, True], removed=counts[True, False],
                inherited=counts[True, True], stayed_clear=counts[False, False])


def _gain(entries):
    signs = Counter()
    by_truth = {name: Counter() for name in ("release_true", "retain_true", "mixed_or_ambiguous")}
    absolute_deltas, late_gains, deletion_gains = [], [], []
    for row, _, pair, _ in entries:
        omitted = _omitted(pair)
        original = pair["original"]["delta"]
        after = omitted["delta"]
        signs[f"{_direction(original)}→{_direction(after)}"] += 1
        expect = _expected(row)
        group = "release_true" if expect > 0 else "retain_true" if expect < 0 else "mixed_or_ambiguous"
        by_truth[group].update(readouts=1, original_release_gain=int(_direction(original) > 0),
                               deleted_release_gain=int(_direction(after) > 0),
                               original_toward_truth=int(expect*_direction(original) > 0),
                               deleted_toward_truth=int(expect*_direction(after) > 0),
                               new_toward_truth=int(expect*_direction(original) <= 0 < expect*_direction(after)),
                               lost_toward_truth=int(expect*_direction(after) <= 0 < expect*_direction(original)))
        absolute_deltas.append(abs(after-original))
        late_gains.append(original)
        deletion_gains.append(after)
    return dict(sign_transition=dict(signs), by_truth={name: dict(count) for name, count in by_truth.items()},
                median_absolute_change_m2=_median(absolute_deltas), median_original_m2=_median(late_gains),
                median_removed_m2=_median(deletion_gains))


def _cohort(rows, moment, kind):
    entries = []
    for row in rows:
        report = _pick_moment(row, moment, kind)
        pair = _pick(report, kind) if report is not None else None
        entries.append((row, report, pair, _values(report, pair)))
    available = [entry for entry in entries if entry[2] is not None]
    matched = [entry for entry in available if _omitted(entry[2]) is not None]
    return dict(windows=len(entries), available=len(available), supported=len(matched),
                status=dict(Counter(pair["removal"]["status"] for _, _, pair, _ in available)),
                full={arm: _arm_counts(entries, arm) for arm in ARMS},
                matched={arm: _arm_counts(matched, arm) for arm in ARMS},
                paired={f"{arm}_vs_{control}": _changes(matched, arm, control)
                        for arm, controls in PAIRED.items() for control in controls},
                late_evidence=_gain(matched))


def _earliest(rows, kind):
    output = {}
    for arm in ("original_new_support", "removed_new_support", "removed_and_earlier_masked_new_support"):
        leads = []
        for row in rows:
            for report in row["reports"]:
                if report["stamp"] >= row["stop"]:
                    break
                if _positive(arm, _values(report, _pick(report, kind))[arm]):
                    leads.append(row["stop"]-report["stamp"])
                    break
        output[arm] = dict(warnings=len(leads), median_lead_s=_median(leads))
    return output


def _comparison(target, correct):
    moments = {}
    for moment in ("latest", "boundary", "first_eligible"):
        moments[moment] = {}
        for kind in KINDS:
            a, b = (_cohort(rows, moment, kind) for rows in (target, correct))
            counts = {mode: {arm: _metrics(a[mode][arm]["warnings"], b[mode][arm]["warnings"],
                                           a[denominator], b[denominator]) for arm in ARMS}
                      for mode, denominator in (("full", "windows"), ("matched", "supported"))}
            moments[moment][kind] = dict(target=a, correct_rejoin=b, **counts)
    return dict(target_windows=len(target), correct_windows=len(correct),
                target_focal_edges=sum(row["target_edges"] for row in target), moments=moments,
                earliest={kind: dict(target=_earliest(target, kind), correct_rejoin=_earliest(correct, kind))
                          for kind in KINDS})


def _census(rows):
    outcomes = Counter()
    diagnostics = {}
    for phase in ("strict", "boundary"):
        for kind in KINDS:
            entries = []
            for row in rows:
                for report in row["reports"]:
                    if (report["stamp"] < row["stop"]) != (phase == "strict"):
                        continue
                    pairs = report["pairs"].values() if kind == "combined" else [_pick(report, kind)]
                    for pair in pairs:
                        if pair is not None:
                            entries.append((row, report, pair, _values(report, pair)))
            matched = [entry for entry in entries if _omitted(entry[2]) is not None]
            status = Counter(pair["removal"]["status"] for _, _, pair, _ in entries)
            diagnostics[f"{phase}_{kind}"] = dict(readouts=len(entries), supported=len(matched), status=dict(status),
                                                   late_evidence=_gain(matched),
                                                   pure_truth=dict(Counter(str(_expected(row)) for row, _, _, _ in entries)))
            if kind == "combined":
                outcomes[phase] = len(entries)
    return dict(windows=len(rows), readouts=dict(outcomes), strata=diagnostics)


def summarize(summary):
    entries = [(scene, row) for scene in summary["scenes"].values() for row in scene["rows"]]
    sparse = [entry for entry in entries if entry[0]["sparse"]]
    target = [entry for entry in sparse if entry[1]["target_edges"] > 0]
    correct = [entry for entry in sparse if _primary_control(entry[1])]
    components = {scene["component"] for scene, _ in target}
    switched_target = [entry for entry in target if entry[1]["switch_source"] is True]
    switched_correct = [entry for entry in correct if entry[1]["switch_source"] is True]

    def rows(population):
        return [row for _, row in population]

    def by_component(population, selected):
        return rows([entry for entry in population if entry[0]["component"] in selected])

    return dict(primary=_comparison(rows(target), rows(correct)),
                source_switch=_comparison(rows(switched_target), rows(switched_correct)),
                same_components=_comparison(rows(target), by_component(correct, components)),
                same_components_switch=_comparison(rows(switched_target), by_component(switched_correct, components)),
                by_component={component: _comparison(by_component(target, {component}), by_component(correct, {component}))
                              for component in sorted(components)},
                pair_census={name: _census(rows(population)) for name, population in (
                    ("all", entries), ("sparse", sparse), ("target", target), ("correct_rejoin", correct))})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = summarize(json.loads(args.summary.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
