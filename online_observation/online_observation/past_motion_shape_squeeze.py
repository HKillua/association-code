"""Compare past motion shape and fixed-clock controls in frozen MTAD windows."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from .async_report_squeeze import _primary_control
from .disjoint_report_closure_squeeze import _metrics
from .hypothesis_gap_diagnostic import _direction


MOMENTS = ("latest", "onset", "boundary")
SHAPES = ("net", "change", "two_block", "recent")
ARMS = ("position_actual", "position_masked", "hold", *SHAPES,
        *(f"{name}_masked" for name in SHAPES),
        "two_block_geometry", "net_geometry", "two_block_new_support")
COMPARISONS = {
    "two_block": ("net", "recent", "change", "position_actual", "two_block_masked"),
    "two_block_geometry": ("net_geometry", "position_actual", "position_masked"),
    "two_block_new_support": ("two_block", "two_block_masked", "position_actual"),
}


def _select(row, moment):
    reports = row["reports"]
    if moment == "latest":
        return next((report for report in reversed(reports) if report["stamp"] < row["stop"]), None)
    stamp = row["start"] if moment == "onset" else row["stop"]
    return next((report for report in reports if report["stamp"] == stamp), None)


def _positive(value):
    return value is not None and _direction(value) > 0


def _values(report):
    values = dict.fromkeys(ARMS) | {"hold": -1}
    if report is None:
        return values
    geometry = report["geometry"]
    values.update(position_actual=geometry["actual"], position_masked=geometry["masked"])
    shape = report["shape"]
    if shape["actual"] is None:
        return values
    actual, masked = shape["actual"], shape["masked"]
    for name in SHAPES:
        values[name], values[f"{name}_masked"] = actual[name], masked[name]
    values.update(two_block_geometry=int(_positive(actual["two_block"]) and _positive(geometry["actual"])),
                  net_geometry=int(_positive(actual["net"]) and _positive(geometry["actual"])),
                  two_block_new_support=int(_positive(actual["two_block"])
                                            and _positive(actual["two_block"]-masked["two_block"])))
    return values


def _median(values):
    return median(values) if values else None


def _stats(entries, arm):
    warnings = [(row["stop"]-report["stamp"], row["target_edges"] or row["focal"])
                for row, report, values in entries if _positive(values[arm])]
    if any(lead < 0 for lead, _ in warnings):
        raise ValueError("warning outside the original fixed window")
    return dict(warnings=len(warnings), scored=sum(values[arm] is not None for _, _, values in entries),
                warning_edges=sum(edges for _, edges in warnings),
                median_lead_s=_median([lead for lead, _ in warnings]),
                lead_edge_seconds=sum(lead*edges for lead, edges in warnings))


def _changes(entries, arm, control):
    counts = Counter((_positive(values[control]), _positive(values[arm])) for _, _, values in entries)
    return dict(added=counts[False, True], removed=counts[True, False],
                inherited=counts[True, True], stayed_clear=counts[False, False])


def _support(entries):
    reports = [report for _, report, _ in entries if report is not None]
    shapes = [report["shape"] for report in reports if report["shape"]["actual"] is not None]
    return dict(reports=len(reports), status=dict(Counter(report["shape"]["status"] for report in reports)),
                supported=len(shapes), current_used=sum(bool(shape["current_used"]) for shape in shapes),
                post_start_used=sum(bool(shape["post_start_used"]) for shape in shapes),
                end_after_start=sum(bool(shape["end_after_start"]) for shape in shapes),
                median_age_of_end_s=_median([report["stamp"]-report["shape"]["end"] for report in reports
                                            if report["shape"]["actual"] is not None]),
                median_availability_lag_s=_median([shape["available_at"]-shape["end"] for shape in shapes]))


def _cohort(rows, moment):
    entries = [(row, report, _values(report)) for row in rows for report in (_select(row, moment),)]
    matched = [entry for entry in entries if entry[2]["two_block"] is not None]
    return dict(windows=len(entries), **_support(entries),
                full={arm: _stats(entries, arm) for arm in ARMS},
                matched={arm: _stats(matched, arm) for arm in ARMS},
                paired={f"{arm}_vs_{control}": _changes(matched, arm, control)
                        for arm, controls in COMPARISONS.items() for control in controls})


def _earliest(rows):
    leads = {arm: [] for arm in ARMS}
    for row in rows:
        pending = set(ARMS)
        for report in row["reports"]:
            if report["stamp"] >= row["stop"]:
                break
            values = _values(report)
            warned = {arm for arm in pending if _positive(values[arm])}
            for arm in warned:
                leads[arm].append(row["stop"]-report["stamp"])
            pending.difference_update(warned)
    return {arm: dict(warnings=len(values), median_lead_s=_median(values)) for arm, values in leads.items()}


def _comparison(target, correct):
    moments = {}
    for moment in MOMENTS:
        a, b = _cohort(target, moment), _cohort(correct, moment)
        counts = {mode: {arm: _metrics(a[mode][arm]["warnings"], b[mode][arm]["warnings"],
                                       a[denominator], b[denominator]) for arm in ARMS}
                  for mode, denominator in (("full", "windows"), ("matched", "supported"))}
        moments[moment] = dict(target=a, correct_rejoin=b, **counts)
    return dict(target_windows=len(target), correct_windows=len(correct),
                target_focal_edges=sum(row["target_edges"] for row in target), moments=moments,
                earliest=dict(target=_earliest(target), correct_rejoin=_earliest(correct)))


def _census(rows):
    phases = {phase: [] for phase in ("strict", "boundary")}
    for row in rows:
        for report in row["reports"]:
            phase = "strict" if report["stamp"] < row["stop"] else "boundary"
            phases[phase].append((row, report, None))
    return {phase: _support(entries) for phase, entries in phases.items()}


def summarize(summary):
    entries = [(scene["component"], row) for scene in summary["scenes"].values() for row in scene["rows"]]
    sparse = [(scene["component"], row) for scene in summary["scenes"].values() if scene["sparse"]
              for row in scene["rows"]]
    target = [(component, row) for component, row in sparse if row["target_edges"] > 0]
    correct = [(component, row) for component, row in sparse if _primary_control(row)]
    components = {component for component, _ in target}

    def rows(population, selected=None, switched=False):
        return [row for component, row in population
                if (selected is None or component in selected) and (not switched or row["switch_source"] is True)]

    return dict(primary=_comparison(rows(target), rows(correct)),
                same_components=_comparison(rows(target), rows(correct, components)),
                source_switch=_comparison(rows(target, switched=True), rows(correct, switched=True)),
                same_components_switch=_comparison(rows(target, switched=True), rows(correct, components, True)),
                by_component={component: _comparison(rows(target, {component}), rows(correct, {component}))
                              for component in sorted(components)},
                support_census={name: _census(rows(population)) for name, population in (
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
