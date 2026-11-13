"""Summarize the frozen cross-source residual and historical-anchor readouts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import hypot, log, sqrt
from pathlib import Path
from statistics import fmean, median

from .async_report_scores import reported_direction
from .async_report_squeeze import CHI2_2_P95, _primary_control
from .cross_source_calibration import decompose


MOMENTS = ("onset", "later_first", "later_last", "latest", "boundary")
TIERS = ("graph", "motion")
FIT_ARMS = ("mean", "mean_masked", "scale", "both", "both_masked")


def _average(values):
    return fmean(values) if values else None


def _middle(values):
    return median(values) if values else None


def _squared(vector):
    return sum(value * value for value in vector)


def _subtract(vector, center):
    return tuple(value - mean for value, mean in zip(vector, center, strict=True))


def _multiply(vector, scale):
    return tuple(value * scale for value in vector)


def _decomposition(vectors, pair_ids):
    if not vectors:
        return dict(window=decompose(()), equal_pair=decompose(()), distinct_pairs=0)
    groups = defaultdict(list)
    for vector, pair in zip(vectors, pair_ids, strict=True):
        groups[pair].append(vector)
    by_pair = [decompose(group) for group in groups.values()]
    equal_pair = decompose(row["mean"] for row in by_pair)
    equal_pair["mean_squared"] = fmean(row["mean_squared"] for row in by_pair)
    equal_pair["centered_spread"] += fmean(row["centered_spread"] for row in by_pair)
    return dict(window=decompose(vectors), equal_pair=equal_pair, distinct_pairs=len(groups))


def _tail(values):
    count = len(values)
    exceed = sum(value > CHI2_2_P95 for value in values)
    return dict(n=count, mean=_average(values), median=_middle(values), above_p95=exceed,
                fraction_above_p95=exceed / count if count else None)


def _phase_entries(entries, moment):
    return [(alias, scene, row, row[moment]) for alias, scene, row in entries if row.get(moment) is not None]


def _residual(entries):
    cross, disparity, own, white, white_own, pair_ids, variances, ages = ([] for _ in range(8))
    for index, (alias, _, _, snapshot) in enumerate(entries):
        side = snapshot["true_side"]
        if side is None:
            continue
        sign = 1 if str(snapshot["source"]) == "0" else -1
        variance = snapshot["variance"][side]
        value = _multiply(snapshot["cross"][side], sign)
        own_value = _multiply(snapshot["self"], sign)
        cross.append(value)
        own.append(own_value)
        disparity.append(_multiply(snapshot["pred_delta"][side], sign))
        white.append(_multiply(value, 1 / sqrt(variance)))
        white_own.append(_multiply(own_value, 1 / sqrt(snapshot["self_variance"])))
        ages.append(snapshot["age"][side])
        variances.append(variance)
        # Anonymous pair IDs are scene-local. Unmapped pairs are kept as distinct rows.
        pair = snapshot.get("true_pair")
        pair_ids.append((alias, pair) if pair is not None else (alias, None, index))

    nominal = _decomposition(cross, pair_ids)
    whitened = _decomposition(white, pair_ids)
    center = nominal["window"]["mean"]
    white_center = whitened["window"]["mean"]
    return dict(cross=nominal, whitened_cross=whitened,
                prediction_disparity=_decomposition(disparity, pair_ids),
                own=_decomposition(own, pair_ids), whitened_own=_decomposition(white_own, pair_ids),
                median_true_horizon_s=_middle(ages),
                median_nominal_sigma_m=_middle([sqrt(value) for value in variances]),
                median_cross_norm_m=_middle([hypot(*value) for value in cross]),
                median_own_norm_m=_middle([hypot(*value) for value in own]),
                raw_nis=_tail([_squared(value) for value in white]),
                raw_centered_nis=_tail([_squared(_subtract(value, center)) / variance
                                        for value, variance in zip(cross, variances, strict=True)]),
                whitened_centered_nis=_tail([_squared(_subtract(value, white_center)) for value in white]),
                self_nis=_tail([_squared(value) for value in white_own]))


def _age_bucket(snapshot):
    side = snapshot["true_side"]
    age = snapshot["age"][side]
    return "le10" if age <= 10 else "gt10_le30" if age <= 30 else "gt30"


def _residual_strata(entries):
    by_source, by_age, by_component_source = defaultdict(list), defaultdict(list), defaultdict(list)
    for entry in entries:
        _, scene, _, snapshot = entry
        if snapshot["true_side"] is None:
            continue
        source = str(snapshot["source"])
        by_source[source].append(entry)
        by_age[_age_bucket(snapshot)].append(entry)
        by_component_source[(scene["component"], source)].append(entry)
    return dict(total=_residual(entries), by_source={key: _residual(rows) for key, rows in sorted(by_source.items())},
                by_age={key: _residual(rows) for key, rows in sorted(by_age.items())},
                by_component_source={f"{component}:{source}": _residual(rows)
                                     for (component, source), rows in sorted(by_component_source.items())})


def _positive(score):
    return score is not None and reported_direction(score) == 1


def _warning_row(entries, scores):
    warned = [(row, snapshot) for (_, _, row, snapshot), score in zip(entries, scores, strict=True)
              if _positive(score)]
    leads = [(row["stop"] - snapshot["stamp"], row["target_edges"] or row["focal"]) for row, snapshot in warned]
    return dict(available=sum(score is not None for score in scores), warnings=len(warned),
                warning_edges=sum(edges for _, edges in leads),
                lead_median_s=_middle([lead for lead, _ in leads]),
                lead_edge_seconds=sum(lead * edges for lead, edges in leads))


def _truth_metrics(entries, fits, arm):
    raw_squared, calibrated_squared, raw_nis, calibrated_nis = [], [], [], []
    raw_nll, calibrated_nll = [], []
    for (_, _, _, snapshot), fit in zip(entries, fits, strict=True):
        side = snapshot["true_side"]
        if side is None:
            continue
        vector, variance = snapshot["cross"][side], snapshot["variance"][side]
        centered = _subtract(vector, fit["mean"]) if arm in ("mean", "both") else vector
        calibrated_variance = variance * fit["scale"] if arm in ("scale", "both") else variance
        native_sq, calibrated_sq = _squared(vector), _squared(centered)
        raw_squared.append(native_sq)
        calibrated_squared.append(calibrated_sq)
        raw_nis.append(native_sq / variance)
        calibrated_nis.append(calibrated_sq / calibrated_variance)
        raw_nll.append(raw_nis[-1] + 2 * log(variance))
        calibrated_nll.append(calibrated_nis[-1] + 2 * log(calibrated_variance))
    return dict(n=len(raw_squared), raw_mean_squared=_average(raw_squared), calibrated_mean_squared=_average(calibrated_squared),
                raw_nis=_tail(raw_nis), calibrated_nis=_tail(calibrated_nis),
                raw_nll=_average(raw_nll), calibrated_nll=_average(calibrated_nll),
                squared_error_better=sum(calibrated < raw for raw, calibrated in zip(raw_squared, calibrated_squared, strict=True)),
                squared_error_worse=sum(calibrated > raw for raw, calibrated in zip(raw_squared, calibrated_squared, strict=True)),
                nll_better=sum(calibrated < raw for raw, calibrated in zip(raw_nll, calibrated_nll, strict=True)),
                nll_worse=sum(calibrated > raw for raw, calibrated in zip(raw_nll, calibrated_nll, strict=True)))


def _anchors(entries, tier):
    fits = [snapshot["anchors"][tier] for _, _, _, snapshot in entries]
    counts = Counter(fit["n"] for fit in fits)
    recent_counts = Counter(fit["recent_n"] for fit in fits)
    result = dict(windows=len(entries), histogram=dict(sorted(counts.items())), recent_histogram=dict(sorted(recent_counts.items())),
                  at_least={n: sum(fit["n"] >= n for fit in fits) for n in (1, 2, 3)},
                  recent_at_least={n: sum(fit["recent_n"] >= n for fit in fits) for n in (1, 2, 3)},
                  anchors=sum(fit["n"] for fit in fits),
                  audit={name: dict(anchors=sum(fit[name] for fit in fits), windows=sum(fit[name] > 0 for fit in fits))
                         for name in ("wrong", "overlap", "duplicate")},
                  clean_mean_windows=sum(fit["mean"] is not None and not (fit["wrong"] or fit["overlap"] or fit["duplicate"])
                                         for fit in fits),
                  clean_scale_windows=sum(fit["scale"] is not None and not (fit["wrong"] or fit["overlap"] or fit["duplicate"])
                                          for fit in fits),
                  mean_norm_median=_middle([hypot(*fit["mean"]) for fit in fits if fit["mean"] is not None]),
                  variance_multiplier_median=_middle([fit["scale"] for fit in fits if fit["scale"] is not None]),
                  arms={})
    for arm in FIT_ARMS:
        scores = [fit["scores"][arm] for fit in fits]
        raw_arm = "masked" if arm.endswith("masked") else "raw"
        raw = [snapshot["scores"][raw_arm] for _, _, _, snapshot in entries]
        supported = [index for index, score in enumerate(scores) if score is not None]
        changed_up = sum(_positive(scores[i]) and not _positive(raw[i]) for i in supported)
        changed_down = sum(not _positive(scores[i]) and _positive(raw[i]) for i in supported)
        supported_entries = [entries[i] for i in supported]
        result["arms"][arm] = dict(matched=_warning_row(supported_entries, [scores[i] for i in supported]),
                                    matched_reference=_warning_row(supported_entries, [raw[i] for i in supported]),
                                    paired_warning_gain=changed_up, paired_warning_loss=changed_down,
                                    fallback_raw=_warning_row(entries, [score if score is not None else baseline
                                                                        for score, baseline in zip(scores, raw, strict=True)]),
                                    fallback_retain=_warning_row(entries, scores))
        if not arm.endswith("masked"):
            result["arms"][arm]["held_out_truth"] = _truth_metrics(supported_entries, [fits[i] for i in supported], arm)
    return result


def _phase_summary(entries, *, detail):
    residual = _residual_strata(entries) if detail else {"total": _residual(entries)}
    return dict(available=len(entries), residual=residual,
                base={arm: _warning_row(entries, [snapshot["scores"][arm] for _, _, _, snapshot in entries])
                      for arm in ("raw", "masked")},
                anchors={tier: _anchors(entries, tier) for tier in TIERS})


def _counts(target, correct, n_target, n_correct):
    tpr = target / n_target if n_target else None
    fpr = correct / n_correct if n_correct else None
    balanced = (tpr + 1 - fpr) / 2 if tpr is not None and fpr is not None else None
    return dict(hits=target, false_vetoes=correct, target_total=n_target, correct_total=n_correct,
                hit_rate=tpr, false_veto_rate=fpr, balanced_accuracy=balanced, net=target - correct)


def _contrast(target, correct, n_target, n_correct):
    result = {arm: _counts(target["base"][arm]["warnings"], correct["base"][arm]["warnings"], n_target, n_correct)
              for arm in ("raw", "masked")}
    result["hold_retain"] = _counts(0, 0, n_target, n_correct)
    for tier in TIERS:
        result[tier] = {}
        for arm in FIT_ARMS:
            a, b = (group["anchors"][tier]["arms"][arm] for group in (target, correct))
            result[tier][arm] = {mode: _counts(a[mode]["warnings"], b[mode]["warnings"],
                                              a[mode]["available"] if mode.startswith("matched") else n_target,
                                              b[mode]["available"] if mode.startswith("matched") else n_correct)
                                 for mode in ("matched", "matched_reference", "fallback_raw", "fallback_retain")}
    return result


def _comparison(target, correct, *, moments=MOMENTS, detail=False):
    phases = {}
    for moment in moments:
        a, b = (_phase_summary(_phase_entries(rows, moment), detail=detail) for rows in (target, correct))
        phases[moment] = dict(target=a, correct_rejoin=b, comparison=_contrast(a, b, len(target), len(correct)))
    return dict(target_windows=len(target), correct_windows=len(correct),
                target_focal_edges=sum(row["target_edges"] for _, _, row in target),
                target_components=len({scene["component"] for _, scene, _ in target}),
                correct_components=len({scene["component"] for _, scene, _ in correct}), moments=phases)


def summarize(summary):
    """Compare calibrated and uncalibrated reads without updating any score."""
    all_entries = [(alias, scene, row) for alias, scene in summary["scenes"].items() for row in scene["rows"]]
    sparse = [entry for entry in all_entries if entry[1]["sparse"]]
    target = [entry for entry in sparse if entry[2]["target_edges"] > 0]
    correct = [entry for entry in sparse if _primary_control(entry[2])]
    switch_target = [entry for entry in target if entry[2]["switch_source"] is True]
    switch_correct = [entry for entry in correct if entry[2]["switch_source"] is True]
    target_components = {scene["component"] for _, scene, _ in switch_target}
    late = ("latest", "boundary")

    return dict(primary=_comparison(switch_target, switch_correct, detail=True),
                same_components=_comparison(switch_target, [entry for entry in switch_correct
                                                            if entry[1]["component"] in target_components], moments=late),
                no_switch=_comparison([entry for entry in target if entry[2]["switch_source"] is False],
                                      [entry for entry in correct if entry[2]["switch_source"] is False], moments=late),
                all_rejoin=_comparison(target, correct, moments=late),
                all_windows={moment: _phase_summary(_phase_entries(all_entries, moment), detail=False) for moment in late},
                sparse_windows={moment: _phase_summary(_phase_entries(sparse, moment), detail=False) for moment in late},
                window_population=dict(all=len(all_entries), sparse=len(sparse)),
                by_component={component: _comparison([entry for entry in switch_target if entry[1]["component"] == component],
                                                     [entry for entry in switch_correct if entry[1]["component"] == component],
                                                     moments=late)
                              for component in sorted(target_components | {scene["component"] for _, scene, _ in switch_correct})},
                by_same_identity={name: _comparison([entry for entry in switch_target if entry[2]["same_identity"] is same],
                                                     [entry for entry in switch_correct if entry[2]["same_identity"] is same], moments=late)
                                  for name, same in (("same", True), ("different", False))},
                by_latest_source={source: _comparison([entry for entry in switch_target if str(entry[2]["latest"]["source"]) == source],
                                                       [entry for entry in switch_correct if str(entry[2]["latest"]["source"]) == source], moments=late)
                                  for source in ("0", "1")})


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
