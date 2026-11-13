"""Compare label-blind activity with geometric rejection at Level1 births."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from time import perf_counter

from .evaluation_data import read_labels
from .measurements import read_archive_rows
from .positions import event_batches, position_step
from .registration_probe import (
    BIRTH_AGE_S, FIT_STOP_S, SCENES, birth_predictions, estimate_translation, registration_distances,
)
from .rejection_probe import (
    FROZEN_COUNTS, FROZEN_FIELDS, GATE_M, Birth, Fragment, _label_births, acceptance_weights, prepare_births,
)
from .stale_history_probe import FROZEN_ABSENCE_BY_SCENE


MAX_HISTORY_AGE_S = 180
NEAR_M = 1000
COVERAGES = (90, 80)
ACTIVITY = ("peer_local", "peer_local_excluding_fresh", "peer_global", "same_local")
ARMS = ("distance", "ratio", *ACTIVITY)
CAUSAL_ARMS = ("distance_1km", "activity_gate")
GEOMETRIES = ("near_single", "near_multiple", "far_single", "far_multiple", "forced_reject")
WRONG_GROUPS = ("wrong_never_seen", "wrong_prior_seen_stale", "wrong_true_in_gate", "wrong_true_only_outside_gate")
GROUPS = ("correct", *WRONG_GROUPS, "forced_reject")


@dataclass(frozen=True)
class Forecast:
    birth: Birth
    stale_peers: frozenset[Fragment]
    geometry: str
    activity: dict[str, int]


def _count_activity(key, position, birth, history, fit):
    counts = dict.fromkeys(ACTIVITY, 0)
    fresh = frozenset(birth.fresh)
    for other, other_position in history.items():
        if other[0] == key[0]:
            counts["same_local"] += position_step(position, other_position)[0] <= GATE_M
        elif other != birth.winner:
            counts["peer_global"] += 1
            if registration_distances(key, position, other, other_position, fit)[1] <= GATE_M:
                counts["peer_local"] += 1
                counts["peer_local_excluding_fresh"] += other not in fresh
    return counts


def predict(events, fit):
    """Freeze the E-ON5 selection and the last row in the strictly older snapshot."""
    events = tuple(events)
    births = prepare_births(birth_predictions(events, fit))
    forecasts, history, seen = {}, {}, set()
    lagged_events = iter(events)
    lagged = next(lagged_events, None)
    for stamp, positions in events:
        born = positions.keys() - seen
        seen.update(positions)
        if stamp <= FIT_STOP_S or not born:
            continue
        while lagged is not None and stamp - lagged[0] > BIRTH_AGE_S:
            old_stamp, old_positions = lagged
            history.update({key: (old_stamp, position) for key, position in old_positions.items()})
            lagged = next(lagged_events, None)
        bounded = {key: position for key, (old_stamp, position) in history.items()
                   if stamp - old_stamp <= MAX_HISTORY_AGE_S}
        for key in born:
            birth = births[key]
            geometry = "forced_reject"
            if birth.winner is not None:
                distance = "near" if birth.distance <= NEAR_M else "far"
                multiplicity = "single" if len(birth.inside) == 1 else "multiple"
                geometry = f"{distance}_{multiplicity}"
            stale_peers = frozenset(peer for peer in seen if peer[0] != key[0]).difference(birth.fresh)
            forecasts[key] = Forecast(birth, stale_peers, geometry,
                                      _count_activity(key, positions[key], birth, bounded, fit))
    return forecasts


def freeze_decisions(forecasts):
    matched = {q: {arm: {} for arm in ARMS} for q in COVERAGES}
    causal = {arm: {} for arm in CAUSAL_ARMS}
    for geometry in GEOMETRIES[:-1]:
        stratum = {key: f for key, f in forecasts.items() if f.geometry == geometry}
        scores = {arm: {key: getattr(f.birth, arm) for key, f in stratum.items()} for arm in ARMS[:2]}
        for arm in ACTIVITY:
            scores[arm] = {key: (-f.activity[arm], f.birth.distance) for key, f in stratum.items()}
        for q in COVERAGES:
            k = q * len(stratum) // 100
            for arm in ARMS:
                matched[q][arm].update(acceptance_weights(scores[arm], k))
        for key, f in stratum.items():
            causal["distance_1km"][key] = int(f.birth.distance <= NEAR_M)
            causal["activity_gate"][key] = int(not (f.activity["peer_global"] > 0 and f.activity["peer_local"] == 0))
    return matched, causal


def _check_geometry(scene, forecasts, frozen):
    births = [f.birth for f in forecasts.values()]
    observed = dict(births=len(births), candidate_pairs=sum(len(birth.fresh) for birth in births),
                    no_candidate_in_gate=sum(not birth.inside for birth in births),
                    nearest_exact_ties=sum(bool(birth.inside) and birth.winner is None for birth in births))
    if any(value != frozen[field] for field, value in observed.items()):
        raise ValueError(f"scene {scene} has different E-ON5 geometric counts: {observed}")


def score(scene, forecasts, decisions, labels, frozen, expected_absence):
    """Grade fixed predictions and retain only source and aggregate-ready fields."""
    population, _, outcomes = _label_births({key: f.birth for key, f in forecasts.items()}, labels)
    if any(population[field] != frozen[field] for field in FROZEN_FIELDS):
        raise ValueError(f"scene {scene} has different E-ON5 labelled counts: {dict(population)}")
    matched, causal = decisions
    rows = []
    for key, f in forecasts.items():
        if not f.stale_peers <= labels.keys():
            raise ValueError("unmapped historical fragment in post-replay labels")
        _, correct, reason = outcomes[key]
        if f.birth.winner is None:
            group = "forced_reject"
        elif correct:
            group = "correct"
        elif reason == "no_fresh_true":
            group = ("wrong_prior_seen_stale" if any(labels[peer] == labels[key] for peer in f.stale_peers)
                     else "wrong_never_seen")
        else:
            group = f"wrong_{reason}"
        rows.append(dict(source=key[0], geometry=f.geometry, group=group, activity=f.activity,
                         matched={q: {arm: matched[q][arm].get(key, 0) for arm in ARMS} for q in COVERAGES},
                         causal={arm: causal[arm].get(key, 0) for arm in CAUSAL_ARMS}))
    groups = Counter(row["group"] for row in rows)
    observed = {name: groups[f"wrong_{name}"] for name in expected_absence}
    if observed != expected_absence:
        raise ValueError(f"scene {scene} has different E-ON35 absence counts: {observed}")
    return rows


def _rates(rows, groups, weights):
    accepted_by_group = dict.fromkeys(GROUPS, 0)
    for row, weight in zip(rows, weights, strict=True):
        accepted_by_group[row["group"]] += weight
    if accepted_by_group["forced_reject"]:
        raise RuntimeError("forced rejection was accepted")
    correct = accepted_by_group["correct"]
    wrong = sum(accepted_by_group[group] for group in WRONG_GROUPS)
    accepted = correct + wrong
    eligible = len(rows) - groups["forced_reject"]
    rejected_by_group = {group: groups[group] - accepted_by_group[group] for group in GROUPS}
    return dict(
        accepted=accepted, correct=correct, wrong=wrong, rejected=len(rows) - accepted,
        target_wrong_rejected=rejected_by_group["wrong_never_seen"], correct_lost=rejected_by_group["correct"],
        other_wrong_rejected=sum(rejected_by_group[group] for group in WRONG_GROUPS[1:]),
        accepted_by_group=accepted_by_group, rejected_by_group=rejected_by_group,
        coverage=Fraction(accepted, len(rows)) if rows else None,
        coverage_of_eligible=Fraction(accepted, eligible) if eligible else None,
        correct_per_birth=Fraction(correct, len(rows)) if rows else None,
        wrong_per_birth=Fraction(wrong, len(rows)) if rows else None,
        wrong_per_accepted=Fraction(wrong, accepted) if accepted else None,
    )


def _summarize(rows):
    groups = Counter(dict.fromkeys(GROUPS, 0))
    groups.update(row["group"] for row in rows)
    eligible = len(rows) - groups["forced_reject"]
    return dict(
        population=dict(births=len(rows), eligible=eligible, rankable_support=eligible > 0, groups=groups),
        matched={q: {arm: _rates(rows, groups, (row["matched"][q][arm] for row in rows)) for arm in ARMS}
                 for q in COVERAGES},
        causal={arm: _rates(rows, groups, (row["causal"][arm] for row in rows)) for arm in CAUSAL_ARMS},
        activity={arm: dict(nonzero=sum(row["activity"][arm] > 0 for row in rows),
                            total_keys=sum(row["activity"][arm] for row in rows)) for arm in ACTIVITY},
    )


def summarize(rows):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    return dict(_summarize(rows), by_source={source: _summarize(items) for source, items in sorted(by_source.items())},
                by_geometry={geometry: _summarize([row for row in rows if row["geometry"] == geometry])
                             for geometry in GEOMETRIES})


def _check_budgets(scene, report):
    for geometry, strata in report["by_geometry"].items():
        for q in COVERAGES:
            expected = q * strata["population"]["eligible"] // 100
            if any(arm["accepted"] != expected for arm in strata["matched"][q].values()):
                raise RuntimeError(f"scene {scene} {geometry} {q}% has unequal or incorrect acceptance counts")


def qualify(scenes, aggregate):
    main = aggregate["matched"][90]
    local = main["peer_local"]
    target = local["target_wrong_rejected"]
    scene_delta = {
        scene: report["matched"][90]["peer_local"]["target_wrong_rejected"]
        - report["matched"][90]["distance"]["target_wrong_rejected"]
        for scene, report in scenes.items() if report["population"]["groups"]["wrong_never_seen"]
    }
    controls = [main[arm]["target_wrong_rejected"] for arm in ("peer_global", "same_local")]
    criteria = dict(
        target_rejected_at_least_14=target >= 14,
        correct_lost_at_most_10=local["correct_lost"] <= 10,
        at_least_3_more_target_rejections_than_each_geometry=all(
            target - main[arm]["target_wrong_rejected"] >= 3 for arm in ARMS[:2]),
        correct_retention_at_least_each_geometry=all(local["correct"] >= main[arm]["correct"] for arm in ARMS[:2]),
        no_worse_than_activity_controls_and_strictly_better_than_one=target >= max(controls) and target > min(controls),
        no_target_regression_by_scene=all(delta >= 0 for delta in scene_delta.values()),
    )
    relative = all(criteria.values())
    causal = aggregate["causal"]["activity_gate"]
    causal_pass = causal["target_wrong_rejected"] >= 14 and causal["correct_lost"] <= 10
    status = ("qualified_for_held_scene" if relative and causal_pass else
              "relative_qualified_only" if relative else "not_qualified")
    return dict(status=status, relative_pass=relative, causal_pass=causal_pass,
                relative_criteria=criteria, scene_target_rejection_delta_vs_distance=scene_delta)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs=3, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    began = perf_counter()

    frozen = {scene: dict(zip(FROZEN_FIELDS, counts)) for scene, counts in FROZEN_COUNTS.items()}
    prepared = {}
    for scene in SCENES:
        events = list(event_batches(read_archive_rows(args.archive, args.manifest, scene, required_split="T_core")))
        fit = estimate_translation(events)
        forecasts = predict(events, fit)
        _check_geometry(scene, forecasts, frozen[scene])
        prepared[scene] = forecasts, freeze_decisions(forecasts), dict(status=fit["status"], reason=fit["reason"])

    scenes, all_scored = {}, []
    for scene, (forecasts, decisions, fit) in prepared.items():
        labels = read_labels(args.archive, args.manifest, scene)
        scored = score(scene, forecasts, decisions, labels, frozen[scene], FROZEN_ABSENCE_BY_SCENE[scene])
        scenes[scene] = dict(summarize(scored), fit=fit)
        _check_budgets(scene, scenes[scene])
        all_scored.extend(scored)

    aggregate = summarize(all_scored)
    summary = dict(
        scene_ids=SCENES, scenes=scenes, aggregate=aggregate, qualification=qualify(scenes, aggregate),
        clock="ideal_measurement_time", fit_stop_seconds=FIT_STOP_S, birth_max_age_seconds=BIRTH_AGE_S,
        history_age_seconds=dict(min_exclusive=BIRTH_AGE_S, max_inclusive=MAX_HISTORY_AGE_S),
        gate_metres=GATE_M, near_stratum_metres=NEAR_M, coverage_pct=COVERAGES,
        matched_population="scene and geometry strata of original unique winners",
        source_grouping="query source", cutoff_tie_policy="uniform expected weights on exact ties",
        wall_seconds=perf_counter() - began,
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False,
                  default=lambda value: int(value) if value.denominator == 1 else float(value))
        output.write("\n")


if __name__ == "__main__":
    main()
