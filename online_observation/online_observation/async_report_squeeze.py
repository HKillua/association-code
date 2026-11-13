"""Aggregate frozen asynchronous report diagnostics by switch, edge, and component."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean, median

from .async_report_scores import reported_direction
from .hypothesis_gap_diagnostic import _direction as position_direction


METRICS = ("position", "euclidean", "common", "mahalanobis", "likelihood")
SCORE_ARMS = tuple(f"{metric}_{view}" for metric in METRICS for view in ("actual", "masked"))
ARMS = (*SCORE_ARMS, "hold_retain", "hold_release")
MOMENTS = ("onset", "later_first", "later_last", "latest", "boundary")
CONTRASTS = (
    *((f"{metric}_actual", f"{metric}_masked") for metric in METRICS),
    ("euclidean_actual", "position_actual"),
    ("common_actual", "euclidean_actual"),
    ("mahalanobis_actual", "common_actual"),
    ("likelihood_actual", "mahalanobis_actual"),
    ("likelihood_actual", "euclidean_actual"),
    ("likelihood_actual", "position_actual"),
    ("likelihood_actual", "hold_retain"),
    ("likelihood_actual", "hold_release"),
)
CHI2_2_P95 = 5.991464547107979
AGE_BUCKETS = ("le10", "gt10_le30", "gt30")


def _median(samples):
    return median(samples) if samples else None


def _direction(snapshot, arm):
    if arm == "hold_retain":
        return -1
    if arm == "hold_release":
        return 1
    if snapshot is None or snapshot.get(arm) is None:
        return None
    measure = position_direction if arm.startswith(("position_", "euclidean_")) else reported_direction
    return measure(snapshot[arm])


def _warning_leads(rows, arm, weight):
    leads = []
    for row in rows:
        first = row["start"] if arm == "hold_release" else row["first_warning"].get(arm)
        if first is not None:
            lead = row["stop"] - first
            if lead <= 0:
                raise ValueError("first-warning summaries require a strictly pre-rejoin event")
            leads.append((lead, row[weight]))
    return leads


def _moment(rows, moment, weight):
    snapshots = [row.get(moment) for row in rows]
    n_available = sum(snapshot is not None for snapshot in snapshots)
    comparisons = {}
    decisions = {arm: [_direction(snapshot, arm) for snapshot in snapshots] for arm in ARMS}
    arms = {}
    for arm, values in decisions.items():
        leads = [(row["stop"] - snapshot["stamp"], row[weight])
                 for row, snapshot, value in zip(rows, snapshots, values, strict=True)
                 if snapshot is not None and value == 1]
        arms[arm] = dict(warnings=values.count(1), ties=values.count(0), unavailable=values.count(None),
                         warning_edges=sum(row[weight] for row, value in zip(rows, values, strict=True)
                                           if value == 1),
                         warning_lead_median_s=_median([lead for lead, _ in leads]),
                         warning_lead_edge_s=sum(lead * edges for lead, edges in leads))
    for arm, control in CONTRASTS:
        gained, lost = [], []
        for row, proposed, reference in zip(rows, decisions[arm], decisions[control], strict=True):
            if proposed == 1 and reference != 1:
                gained.append(row[weight])
            elif proposed != 1 and reference == 1:
                lost.append(row[weight])
        comparisons[f"{arm}_vs_{control}"] = dict(
            gain=len(gained), loss=len(lost), net=len(gained)-len(lost),
            edge_gain=sum(gained), edge_loss=sum(lost), edge_net=sum(gained)-sum(lost),
        )
    return dict(available=n_available, arms=arms, paired_warning_changes=comparisons)


def _cohort(rows, weight="focal"):
    first = {}
    for arm in ARMS:
        weighted = _warning_leads(rows, arm, weight)
        leads = [lead for lead, _ in weighted]
        first[arm] = dict(warnings=len(leads), median_lead_s=_median(leads),
                          min_lead_s=min(leads, default=None), max_lead_s=max(leads, default=None),
                          warning_edges=sum(edges for _, edges in weighted),
                          warning_lead_edge_s=sum(lead * edges for lead, edges in weighted))
    return dict(windows=len(rows), focal_edges=sum(row[weight] for row in rows),
                pre_rejoin_group_report_windows=sum(row["group_batches"] > 0 for row in rows),
                pre_rejoin_quiet_windows=sum(row["quiet_batches"] > 0 for row in rows),
                pre_rejoin_no_frame_windows=sum(row["window_batches"] == 0 for row in rows),
                median_window_s=_median([row["stop"]-row["start"] for row in rows]),
                first_warning=first,
                **{moment: _moment(rows, moment, weight) for moment in MOMENTS})


def _correct_rejoin(row):
    return (row["end"] == "kf_rejoin" and row["late_support"]
            and row["focal"] == row["focal_true"] == 2
            and row["true_release"] == 0 and row["true_retain"] == 2)


def _primary_control(row):
    return _correct_rejoin(row) and row["full_rejoin"] and row["boundary_reports"] == 1


def _metric_pair(target, control):
    pairs = {}
    for moment in MOMENTS:
        pairs[moment] = {}
        for arm in ARMS:
            hit = target[moment]["arms"][arm]["warnings"]
            false_alarm = control[moment]["arms"][arm]["warnings"]
            tpr = hit / target["windows"] if target["windows"] else None
            fpr = false_alarm / control["windows"] if control["windows"] else None
            balanced = (tpr + (1-fpr)) / 2 if tpr is not None and fpr is not None else None
            pairs[moment][arm] = dict(hit=hit, false_alarm=false_alarm, hit_rate=tpr, false_alarm_rate=fpr,
                                      correct_rejoin_preserved=control["windows"]-false_alarm,
                                      balanced_accuracy=balanced)
    return pairs


def _comparison(target, controls):
    target_counts = _cohort(target, "target_edges")
    control_counts = _cohort(controls)
    return dict(target=target_counts, correct_rejoin=control_counts,
                discrimination=_metric_pair(target_counts, control_counts))


def _without_registration(row):
    def original_coordinates(snapshot):
        if snapshot is None or snapshot.get("unregistered") is None:
            return snapshot
        return snapshot["unregistered"] | {"stamp": snapshot["stamp"]}

    first_warning = {}
    for snapshot in row.get("batches", ()):
        native = original_coordinates(snapshot)
        if native["stamp"] >= row["stop"]:
            break
        for arm in SCORE_ARMS:
            if _direction(native, arm) == 1:
                first_warning.setdefault(arm, native["stamp"])
    return row | {moment: original_coordinates(row.get(moment)) for moment in MOMENTS} | {
        "first_warning": first_warning,
    }


def _age_bucket(age):
    return AGE_BUCKETS[0] if age <= 10 else AGE_BUCKETS[1] if age <= 30 else AGE_BUCKETS[2]


def _nis_stats(values):
    values = sorted(values)
    n = len(values)
    above = sum(value > CHI2_2_P95 for value in values)
    return dict(reports=n, mean=fmean(values) if n else None, median=_median(values),
                empirical_p95=values[(95*n+99)//100-1] if n else None,
                above_chi2_2_p95=above, fraction_above_chi2_2_p95=above/n if n else None)


def _source(report):
    if "source" in report:
        return str(report["source"])
    if "source_slot" in report:
        return str(report["source_slot"])
    key = report.get("key")
    return str(key[0]) if isinstance(key, (tuple, list)) and key else "unknown"


def _source_of_snapshot(row, moment):
    snapshot = row.get(moment)
    if snapshot is None:
        return "no_score"
    sources = {_source(report) for report in snapshot["reports"]}
    return next(iter(sources)) if len(sources) == 1 else "mixed"


def _add_nis(samples, kind, alias, stamp, key, age, squared, variance, source):
    identity = (alias, stamp, str(key)) if key is not None else (alias, stamp, age, squared, variance, source)
    value = squared / variance
    samples[kind].setdefault(identity, (_age_bucket(age), source, value))


def _calibration(scene_rows):
    samples = defaultdict(dict)
    for alias, _, row in scene_rows:
        batches = row.get("batches", (row.get(moment) for moment in MOMENTS))
        for snapshot in batches:
            if snapshot is None:
                continue
            stamp = snapshot["stamp"]
            for report in snapshot["reports"]:
                source = _source(report)
                key = report.get("key")
                _add_nis(samples, "self", alias, stamp, key, report["age"], report["self_squared"],
                         report["self_variance"], source)
                for side in ("release", "retain"):
                    if row[f"true_{side}"] != 2:
                        continue
                    partner = report.get(f"{side}_key")
                    pair = (key, partner) if key is not None and partner is not None else None
                    _add_nis(samples, "cross_true", alias, stamp, pair, report[f"{side}_age"],
                             report[f"{side}_squared"], report[f"{side}_variance"], source)
    return dict(chi2_2_p95=CHI2_2_P95,
                population="all scored native reports in each frozen window, including rejoin batch",
                by_kind={kind: {bucket: _nis_stats([value for age, _, value in samples[kind].values()
                                                   if age == bucket]) for bucket in AGE_BUCKETS}
                         for kind in ("self", "cross_true")},
                by_source={source: {kind: {bucket: _nis_stats([
                    value for age, slot, value in samples[kind].values() if age == bucket and slot == source
                ]) for bucket in AGE_BUCKETS} for kind in ("self", "cross_true")}
                           for source in sorted({source for data in samples.values()
                                                 for _, source, _ in data.values()})})


def summarize(summary):
    """Summarize the label-joined table without changing the frozen score traces."""
    entries = [(alias, scene, row) for alias, scene in summary["scenes"].items() for row in scene["rows"]]
    sparse = [entry for entry in entries if entry[1]["sparse"]]
    target = [entry for entry in sparse if entry[2]["target_edges"] > 0]
    controls = [entry for entry in sparse if _primary_control(entry[2])]
    target_components = {scene["component"] for _, scene, _ in target}
    control_components = {scene["component"] for _, scene, _ in controls}

    def rows(entries):
        return [row for _, _, row in entries]

    return dict(
        all=_cohort(rows(entries)), sparse=_cohort(rows(sparse)),
        primary=_comparison(rows(target), rows(controls)),
        target_components=len(target_components), correct_rejoin_components=len(control_components),
        same_components=_comparison(rows(target), rows([entry for entry in controls
                                                        if entry[1]["component"] in target_components])),
        by_component={component: _comparison(
            rows([entry for entry in target if entry[1]["component"] == component]),
            rows([entry for entry in controls if entry[1]["component"] == component]),
        ) for component in sorted(target_components | control_components)},
        by_registration={name: _comparison(
            rows([entry for entry in target if entry[1]["registration"] == fitted]),
            rows([entry for entry in controls if entry[1]["registration"] == fitted]),
        ) for name, fitted in (("fitted", True), ("abstained", False))},
        without_registration=_comparison([_without_registration(row) for _, _, row in target],
                                         [_without_registration(row) for _, _, row in controls]),
        by_reporting_source={moment: {source: _comparison(
            rows([entry for entry in target if _source_of_snapshot(entry[2], moment) == source]),
            rows([entry for entry in controls if _source_of_snapshot(entry[2], moment) == source]),
        ) for source in sorted({_source_of_snapshot(entry[2], moment) for entry in target + controls})}
                             for moment in ("latest", "boundary")},
        wider_correct_rejoin={name: _cohort([row for _, _, row in sparse if _correct_rejoin(row)
                                             and bool(row["full_rejoin"]) == full
                                             and bool(row["boundary_reports"]) == reported])
                              for name, full, reported in (("full_quiet", True, False),
                                                           ("partial_reporting", False, True),
                                                           ("partial_quiet", False, False))},
        wider_comparison=_comparison(rows(target), [row for _, _, row in sparse if _correct_rejoin(row)]),
        calibration={"all": _calibration(entries), "sparse": _calibration(sparse),
                     "target": _calibration(target), "correct_rejoin": _calibration(controls)},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(json.loads(args.summary.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
