"""Evaluate same-coverage rejection on frozen Level1 T_core fragment births."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
import json
import math
from pathlib import Path
from time import perf_counter

from .evaluation_data import read_labels
from .measurements import read_archive_rows
from .positions import event_batches
from .registration_probe import BIRTH_AGE_S, FIT_STOP_S, SCENES, birth_predictions, estimate_translation


GATE_M = 4000
COVERAGES = (50, 80)
COHORTS = ("competitive", "all_unique")
ARMS = ("distance", "ratio")
FALSE_REASONS = ("true_in_gate", "true_only_outside_gate", "no_fresh_true")

# E-ON5's corrected 4 km counts from the saved registration result.
FROZEN_FIELDS = (
    "births", "candidate_pairs", "true_opportunity", "true_in_gate", "false_in_gate",
    "true_and_false_in_gate", "no_candidate_in_gate", "nearest_exact_ties",
    "unique_correct_matches", "unique_false_matches",
    "unique_fp_true_in_gate", "unique_fp_true_only_outside_gate", "unique_fp_no_fresh_true",
)
FROZEN_COUNTS = {
    "0": (49, 1849, 39, 39, 38, 32, 4, 0, 32, 13, 7, 0, 6),
    "3": (219, 28590, 169, 168, 166, 130, 15, 0, 158, 46, 10, 0, 36),
    "7": (11, 58, 8, 8, 1, 1, 3, 0, 8, 0, 0, 0, 0),
}

Fragment = tuple[str, str]


@dataclass(frozen=True)
class Birth:
    fresh: tuple[Fragment, ...]
    inside: tuple[tuple[Fragment, float], ...]
    winner: Fragment | None
    distance: float | None
    ratio: float | None


def prepare_births(predictions):
    """Compute geometric winners and scores without using fragment keys as ranks."""
    births = {}
    for key, candidates in predictions:
        fresh = tuple(other for other, _, _ in candidates)
        if key in births or len(set(fresh)) != len(fresh):
            raise ValueError("duplicate birth or fresh candidate")
        if any(not math.isfinite(distance) or distance < 0 for _, _, distance in candidates):
            raise ValueError("invalid corrected candidate distance")
        inside = tuple(sorted(((other, distance) for other, _, distance in candidates if distance <= GATE_M),
                              key=lambda pair: pair[1]))
        winner = distance = ratio = None
        if inside and (len(inside) == 1 or inside[0][1] < inside[1][1]):
            winner, distance = inside[0]
            second = inside[1][1] if len(inside) > 1 else GATE_M
            ratio = distance / second
        births[key] = Birth(fresh, inside, winner, distance, ratio)
    return births


def acceptance_weights(scores, k):
    """Allocate k places with uniform expected weights within the cutoff tie."""
    if not 0 <= k <= len(scores):
        raise ValueError("invalid acceptance count")
    weights = {}
    remaining = k
    for _, tied in groupby(sorted(scores.items(), key=lambda entry: entry[1]), key=lambda entry: entry[1]):
        if remaining == 0:
            break
        group = tuple(key for key, _ in tied)
        slots = min(remaining, len(group))
        weight = Fraction(slots, len(group))
        weights.update(dict.fromkeys(group, weight))
        remaining -= slots
    if sum(weights.values()) != k:
        raise RuntimeError("acceptance weights do not sum to the requested count")
    return weights


def freeze_rankings(births):
    selections = {}
    for cohort in COHORTS:
        eligible = {key: birth for key, birth in births.items()
                    if birth.winner is not None and (cohort == "all_unique" or len(birth.inside) >= 2)}
        scores = {arm: {key: getattr(birth, arm) for key, birth in eligible.items()} for arm in ARMS}
        selections[cohort] = {}
        for q in COVERAGES:
            k = q * len(eligible) // 100
            selections[cohort][q] = (eligible.keys(), k, {arm: acceptance_weights(scores[arm], k) for arm in ARMS})
    return selections


def _label_births(births, labels):
    population = Counter(dict.fromkeys((*FROZEN_FIELDS, "single_candidate", "multiple_candidates",
                                        "true_only_outside_gate", "no_fresh_true",
                                        "second_rank_has_winner_identity"), 0))
    by_gate_count = {}
    outcomes = {}
    for key, birth in births.items():
        if key not in labels or any(other not in labels for other in birth.fresh):
            raise ValueError("unmapped measurement fragment in post-replay labels")
        identity = labels[key]
        true_opportunity = any(labels[other] == identity for other in birth.fresh)
        inside_truth = [labels[other] == identity for other, _ in birth.inside]
        count = len(inside_truth)
        true_in_gate = any(inside_truth)
        false_in_gate = not all(inside_truth)
        reason = FALSE_REASONS[0] if true_in_gate else FALSE_REASONS[1] if true_opportunity else FALSE_REASONS[2]
        correct = birth.winner is not None and labels[birth.winner] == identity
        wrong = birth.winner is not None and not correct
        tied = count > 0 and birth.winner is None
        second_shares_identity = birth.winner is not None and count >= 2 and any(
            labels[other] == labels[birth.winner] for other, distance in birth.inside[1:]
            if distance == birth.inside[1][1]
        )
        counts = dict(
            births=1, candidate_pairs=len(birth.fresh), true_opportunity=true_opportunity,
            true_in_gate=true_in_gate, false_in_gate=false_in_gate,
            true_and_false_in_gate=true_in_gate and false_in_gate,
            no_candidate_in_gate=count == 0, single_candidate=count == 1, multiple_candidates=count >= 2,
            nearest_exact_ties=tied, unique_correct_matches=correct, unique_false_matches=wrong,
            true_only_outside_gate=reason == FALSE_REASONS[1], no_fresh_true=reason == FALSE_REASONS[2],
            second_rank_has_winner_identity=second_shares_identity,
        )
        counts.update({f"unique_fp_{kind}": wrong and reason == kind for kind in FALSE_REASONS})
        counts = {field: int(value) for field, value in counts.items()}
        population.update(counts)
        by_gate_count.setdefault(count, Counter()).update(counts)
        outcomes[key] = (count, correct, reason)
    if (population["births"] != sum(population[field] for field in (
            "no_candidate_in_gate", "nearest_exact_ties", "unique_correct_matches", "unique_false_matches"))
            or population["births"] != sum(population[reason] for reason in FALSE_REASONS)):
        raise RuntimeError("inconsistent birth population or opportunity counts")
    return population, by_gate_count, outcomes


def _tally():
    fields = ("eligible", "accepted", "tp", "fp", *(f"fp_{reason}" for reason in FALSE_REASONS))
    return dict(arms={arm: Counter(dict.fromkeys(fields, 0)) for arm in ARMS},
                changes=Counter(dict.fromkeys(("gained_tp", "gained_fp", "lost_tp", "lost_fp"), 0)))


def _tally_birth(tally, correct, reason, distance, ratio):
    for arm, weight in zip(ARMS, (distance, ratio)):
        counts = tally["arms"][arm]
        counts["eligible"] += 1
        counts["accepted"] += weight
        if correct:
            counts["tp"] += weight
        else:
            counts["fp"] += weight
            counts[f"fp_{reason}"] += weight
    truth = "tp" if correct else "fp"
    tally["changes"][f"gained_{truth}"] += max(0, ratio - distance)
    tally["changes"][f"lost_{truth}"] += max(0, distance - ratio)


def _evaluate(outcomes, selection):
    eligible, k, weights = selection
    strata = {}
    for key in eligible:
        count, correct, reason = outcomes[key]
        _tally_birth(strata.setdefault(count, _tally()), correct, reason,
                     weights["distance"].get(key, 0), weights["ratio"].get(key, 0))
    return _summarize(len(eligible), k, strata)


def _merge_tallies(tallies):
    merged = _tally()
    for tally in tallies:
        for arm in ARMS:
            merged["arms"][arm].update(tally["arms"][arm])
        merged["changes"].update(tally["changes"])
    return merged


def _risk_summary(tally):
    risks = {}
    for arm, counts in tally["arms"].items():
        if (counts["accepted"] != counts["tp"] + counts["fp"]
                or counts["fp"] != sum(counts[f"fp_{reason}"] for reason in FALSE_REASONS)
                or not 0 <= counts["accepted"] <= counts["eligible"]):
            raise RuntimeError("inconsistent acceptance or error counts")
        risks[arm] = Fraction(counts["fp"], counts["accepted"]) if counts["accepted"] else None
    distance, ratio = (risks[arm] for arm in ARMS)
    return dict(tally, risk=risks, risk_delta_pp=100 * (ratio - distance) if None not in (distance, ratio) else None)


def _summarize(n, k, strata):
    tally = _merge_tallies(strata.values())
    for arm in ARMS:
        counts = tally["arms"][arm]
        if (counts["eligible"], counts["accepted"]) != (n, k):
            raise RuntimeError("paired acceptance counts differ from their prespecified denominators")
    for truth in ("tp", "fp"):
        difference = tally["arms"]["ratio"][truth] - tally["arms"]["distance"][truth]
        if difference != tally["changes"][f"gained_{truth}"] - tally["changes"][f"lost_{truth}"]:
            raise RuntimeError("paired selection changes do not reconcile")
    return dict(_risk_summary(tally), eligible=n, k=k, coverage=Fraction(k, n) if n else None,
                same_acceptance_weights=not any(tally["changes"].values()),
                by_gate_count={count: _risk_summary(strata[count]) for count in sorted(strata)})


def evaluate_scene(births, rankings, labels, frozen_counts):
    population, by_count, outcomes = _label_births(births, labels)
    observed = tuple(population[field] for field in FROZEN_FIELDS)
    if observed != frozen_counts:
        raise ValueError(f"E-ON5 counts differ: expected {dict(zip(FROZEN_FIELDS, frozen_counts))}, "
                         f"observed {dict(zip(FROZEN_FIELDS, observed))}")
    cohorts = {
        cohort: {q: _evaluate(outcomes, selections[q]) for q in COVERAGES}
        for cohort, selections in rankings.items()
    }
    cohort_sizes = dict(competitive=population["multiple_candidates"] - population["nearest_exact_ties"],
                        all_unique=population["unique_correct_matches"] + population["unique_false_matches"])
    if any(report["eligible"] != cohort_sizes[cohort] for cohort, reports in cohorts.items()
           for report in reports.values()):
        raise RuntimeError("cohort counts differ from the birth population")
    return dict(population=population, population_by_gate_count=dict(sorted(by_count.items())), cohorts=cohorts)


def _pool(reports):
    counts = sorted({count for report in reports for count in report["by_gate_count"]})
    strata = {count: _merge_tallies(report["by_gate_count"][count] for report in reports
                                  if count in report["by_gate_count"]) for count in counts}
    return _summarize(sum(report["eligible"] for report in reports), sum(report["k"] for report in reports), strata)


def aggregate(scenes):
    population = Counter()
    by_count = {}
    for report in scenes.values():
        population.update(report["population"])
        for count, counts in report["population_by_gate_count"].items():
            by_count.setdefault(count, Counter()).update(counts)
    cohorts = {cohort: {} for cohort in COHORTS}
    for cohort in COHORTS:
        for q in COVERAGES:
            reports = {scene: report["cohorts"][cohort][q] for scene, report in scenes.items()}
            pooled = _pool(reports.values())
            leave_one_out = {}
            for omitted in reports:
                other = [report for scene, report in reports.items() if scene != omitted]
                k = sum(report["k"] for report in other)
                fp = {arm: sum(report["arms"][arm]["fp"] for report in other) for arm in ARMS}
                delta = 100 * Fraction(fp["ratio"] - fp["distance"], k) if k else None
                direction = None if delta is None else "lower" if delta < 0 else "higher" if delta > 0 else "equal"
                leave_one_out[omitted] = dict(k=k, fp=fp, risk_delta_pp=delta, direction=direction)
            deltas = [report["risk_delta_pp"] for report in leave_one_out.values()
                      if report["risk_delta_pp"] is not None]
            cohorts[cohort][q] = dict(pooled, leave_one_scene_out=leave_one_out,
                                       leave_one_scene_out_delta_pp_range=[min(deltas), max(deltas)] if deltas else None)
    return dict(population=population, population_by_gate_count=dict(sorted(by_count.items())), cohorts=cohorts)


def qualify(scenes, pooled):
    checks = {}
    for q in COVERAGES:
        primary = pooled["cohorts"]["competitive"][q]
        secondary = pooled["cohorts"]["all_unique"][q]["risk_delta_pp"]
        deltas = [scene["cohorts"]["competitive"][q]["risk_delta_pp"] for scene in scenes.values()]
        supported = [delta for delta in deltas if delta is not None]
        checks[q] = dict(
            scenes_with_acceptance=len(supported), scenes_with_nonincreasing_risk=sum(delta <= 0 for delta in supported),
            max_scene_increase_at_most_5_pp=bool(supported) and max(supported) <= 5,
            pooled_improvement_at_least_5_pp=primary["risk_delta_pp"] is not None and primary["risk_delta_pp"] <= -5,
            all_unique_nonincreasing=secondary is not None and secondary <= 0,
            same_acceptance_weights=primary["same_acceptance_weights"],
        )
    if any(check["scenes_with_acceptance"] < 2 for check in checks.values()):
        status = "insufficient_scene_support"
    elif all(check["same_acceptance_weights"] for check in checks.values()):
        status = "no_acceptance_change"
    elif all(check["scenes_with_nonincreasing_risk"] >= 2 and check["max_scene_increase_at_most_5_pp"]
             and check["pooled_improvement_at_least_5_pp"] and check["all_unique_nonincreasing"]
             for check in checks.values()):
        status = "qualified"
    else:
        status = "not_qualified"
    return dict(status=status, by_coverage_pct=checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs=3, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    began = perf_counter()

    prepared = {}
    for scene in SCENES:
        events = list(event_batches(read_archive_rows(args.archive, args.manifest, scene)))
        fit = estimate_translation(events)
        births = prepare_births(birth_predictions(events, fit))
        geometry_counts = dict(births=len(births), candidate_pairs=sum(len(birth.fresh) for birth in births.values()),
                               no_candidate_in_gate=sum(not birth.inside for birth in births.values()),
                               nearest_exact_ties=sum(bool(birth.inside) and birth.winner is None
                                                      for birth in births.values()))
        expected = dict(zip(FROZEN_FIELDS, FROZEN_COUNTS[scene]))
        if any(count != expected[field] for field, count in geometry_counts.items()):
            raise ValueError(f"scene {scene} has different E-ON5 geometric counts: {geometry_counts}")
        prepared[scene] = (fit, births, freeze_rankings(births))

    scenes = {}
    for scene, (fit, births, rankings) in prepared.items():
        labels = read_labels(args.archive, args.manifest, scene)
        scenes[scene] = evaluate_scene(births, rankings, labels, FROZEN_COUNTS[scene])
        scenes[scene]["fit"] = dict(status=fit["status"], reason=fit["reason"])

    pooled = aggregate(scenes)
    summary = dict(
        scene_ids=SCENES, scenes=scenes, aggregate=pooled, qualification=qualify(scenes, pooled),
        clock="ideal_measurement_time", fit_stop_seconds=FIT_STOP_S, birth_max_age_seconds=BIRTH_AGE_S,
        gate_metres=GATE_M, coverage_pct=COVERAGES,
        population="directed_cross_source_fragment_births_strictly_after_fit",
        nearest_tie_policy="abstain on exact ties", cutoff_tie_policy="uniform expected weights on exact ties",
        change_definition="positive per-birth differences in marginal acceptance weights",
        wall_seconds=perf_counter() - began,
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False,
                  default=lambda value: int(value) if value.denominator == 1 else float(value))
        output.write("\n")


if __name__ == "__main__":
    main()
