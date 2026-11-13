"""Compare six causal association decisions at birth and the next local report."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from .evaluation_data import read_labels
from .measurements import read_archive_rows
from .positions import event_batches
from .registration_probe import BIRTH_AGE_S, FIT_STOP_S, SCENES, estimate_translation, registration_distances
from .rejection_probe import FROZEN_COUNTS, FROZEN_FIELDS, GATE_M, Fragment


WAIT_S = 30
END_S = 7200
ARMS = ("birth", "expiry", "query", "peer_freshness", "peer_position", "open")
COMPARISONS = (("birth", "expiry"), ("expiry", "query"), ("query", "peer_freshness"),
               ("peer_freshness", "peer_position"), ("peer_position", "open"),
               ("expiry", "open"), ("birth", "open"))
OUTCOMES = ("correct", "wrong", "reject")
GROUPS = ("correct", "wrong_true_in_gate", "wrong_true_only_outside_gate", "wrong_no_fresh_true", "reject")
ENTRANTS = ("renewed", "unseen")


@dataclass(frozen=True)
class Selection:
    fresh: frozenset[Fragment]
    in_gate: frozenset[Fragment]
    winner: Fragment | None


@dataclass(frozen=True)
class Prediction:
    disposition: str
    selections: dict[str, Selection]
    stale_at_birth: frozenset[Fragment]
    wait: float | None = None
    grid_wait: int | None = None
    entrants: dict[str, dict[Fragment, float]] | None = None


def select(key, position, candidates, fit):
    inside = {}
    for peer, peer_position in candidates.items():
        _, distance = registration_distances(key, position, peer, peer_position, fit)
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("invalid corrected distance")
        if distance <= GATE_M:
            inside[peer] = distance
    best = min(inside.values(), default=math.inf)
    winners = [peer for peer, distance in inside.items() if distance == best]
    return Selection(frozenset(candidates), frozenset(inside), winners[0] if len(winners) == 1 else None)


def _confirm(key, t0, t1, position, old_position, original, birth, latest, arrivals, fit):
    fresh = {peer: point for peer, (stamp, point) in latest.items()
             if peer[0] != key[0] and t1 - stamp <= BIRTH_AGE_S}
    expiry = {peer: point for peer, (stamp, point) in original.items() if t1 - stamp <= BIRTH_AGE_S}
    renewed_times = {peer: point for peer, (_, point) in original.items() if peer in fresh}
    peer_positions = {peer: fresh[peer] for peer in renewed_times}
    selections = dict(birth.selections,
                      expiry=select(key, old_position, expiry, fit),
                      query=select(key, position, expiry, fit),
                      peer_freshness=select(key, position, renewed_times, fit),
                      peer_position=select(key, position, peer_positions, fit),
                      open=select(key, position, fresh, fit))
    entrants = {group: {} for group in ENTRANTS}
    for peer in fresh.keys() - original.keys():
        group = "renewed" if peer in birth.stale_at_birth else "unseen"
        times = arrivals[peer]
        entrants[group][peer] = times[bisect_right(times, t0)] - t0
    return Prediction("returned", selections, birth.stale_at_birth,
                      t1 - t0, math.ceil(t1) - math.ceil(t0), entrants)


def predict(events, fit):
    """Freeze each birth and next-report snapshot before any label is supplied."""
    latest, pending, predictions = {}, {}, {}
    arrivals = defaultdict(list)
    for stamp, positions in events:
        born = positions.keys() - latest.keys()
        latest.update({key: (stamp, position) for key, position in positions.items()})
        for key in positions:
            arrivals[key].append(stamp)
        for key in pending.keys() & positions.keys():
            t0, old_position, original = pending.pop(key)
            if stamp <= END_S and stamp - t0 <= WAIT_S:
                predictions[key] = _confirm(key, t0, stamp, positions[key], old_position, original,
                                            predictions[key], latest, arrivals, fit)
        if stamp <= FIT_STOP_S:
            continue
        for key in born:
            original = {peer: entry for peer, entry in latest.items()
                        if peer[0] != key[0] and stamp - entry[0] <= BIRTH_AGE_S}
            stale = frozenset(peer for peer in latest if peer[0] != key[0] and peer not in original)
            first = select(key, positions[key], {peer: entry[1] for peer, entry in original.items()}, fit)
            disposition = "right_censored" if stamp + WAIT_S > END_S else "timeout"
            predictions[key] = Prediction(disposition, {"birth": first}, stale)
            if stamp < END_S:
                pending[key] = (stamp, positions[key], original)
    return predictions


def _check_birth_geometry(scene, predictions, frozen):
    first = [prediction.selections["birth"] for prediction in predictions.values()]
    counts = dict(births=len(first), candidate_pairs=sum(len(arm.fresh) for arm in first),
                  no_candidate_in_gate=sum(not arm.in_gate for arm in first),
                  nearest_exact_ties=sum(bool(arm.in_gate) and arm.winner is None for arm in first))
    if any(value != frozen[name] for name, value in counts.items()):
        raise ValueError(f"scene {scene} has different E-ON5 geometric counts: {counts}")


def _grade(selection, truth):
    in_gate = len(selection.in_gate & truth)
    fresh = len(selection.fresh & truth)
    outcome = "reject" if selection.winner is None else "correct" if selection.winner in truth else "wrong"
    return dict(outcome=outcome, candidate_pairs=len(selection.fresh), in_gate_pairs=len(selection.in_gate),
                true_opportunity=int(fresh > 0), true_in_gate=int(in_gate > 0),
                false_in_gate=int(len(selection.in_gate) > in_gate), no_candidate_in_gate=int(not selection.in_gate),
                nearest_exact_ties=int(bool(selection.in_gate) and selection.winner is None))


def score(predictions, labels):
    """Attach truth after decisions and return identity-free records for tallying."""
    by_identity = defaultdict(set)
    for key, identity in labels.items():
        by_identity[identity].add(key)
    scored = []
    for key, prediction in predictions.items():
        if (key not in labels or not prediction.stale_at_birth <= labels.keys()
                or any(not arm.fresh <= labels.keys() for arm in prediction.selections.values())):
            raise ValueError("unmapped fragment in post-replay labels")
        truth = by_identity[labels[key]]
        arms = {arm: _grade(selection, truth) for arm, selection in prediction.selections.items()}
        first = arms["birth"]
        group = first["outcome"]
        if group == "wrong":
            group = ("wrong_true_in_gate" if first["true_in_gate"] else
                     "wrong_true_only_outside_gate" if first["true_opportunity"] else "wrong_no_fresh_true")
        absence = None
        if group == "wrong_no_fresh_true":
            absence = "prior_seen_stale" if prediction.stale_at_birth & truth else "never_seen"

        entrants = {}
        for kind, keys in (prediction.entrants or {}).items():
            selection = prediction.selections["open"]
            true_keys = keys.keys() & truth
            entrants[kind] = dict(
                keys=len(keys), true_keys=len(true_keys), in_gate_keys=len(keys.keys() & selection.in_gate),
                true_in_gate_keys=len(true_keys & selection.in_gate),
                selected_true=int(selection.winner in true_keys),
                selected_wrong=int(selection.winner in keys and selection.winner not in truth),
                first_true_wait=min((keys[peer] for peer in true_keys), default=None),
            )
        scored.append(dict(disposition=prediction.disposition, wait=prediction.wait, grid_wait=prediction.grid_wait,
                           group=group, absence=absence, arms=arms, entrants=entrants))
    return scored


def _arm_counts(rows, arm):
    selected = [row["arms"][arm] for row in rows if arm in row["arms"]]
    counts = {outcome: sum(result["outcome"] == outcome for result in selected) for outcome in OUTCOMES}
    counts["no_decision"] = len(rows) - len(selected)
    for field in ("candidate_pairs", "in_gate_pairs", "true_opportunity", "true_in_gate", "false_in_gate",
                  "no_candidate_in_gate", "nearest_exact_ties"):
        counts[field] = sum(result[field] for result in selected)
    if sum(counts[outcome] for outcome in OUTCOMES) + counts["no_decision"] != len(rows):
        raise RuntimeError("inconsistent arm population")
    return counts


def _check_birth_truth(scene, scored, frozen):
    birth = _arm_counts(scored, "birth")
    counts = dict(true_opportunity=birth["true_opportunity"], true_in_gate=birth["true_in_gate"],
                  false_in_gate=birth["false_in_gate"], unique_correct_matches=birth["correct"],
                  unique_false_matches=birth["wrong"], true_and_false_in_gate=sum(
                      bool(row["arms"]["birth"]["true_in_gate"] and row["arms"]["birth"]["false_in_gate"])
                      for row in scored))
    for group in GROUPS[1:4]:
        counts[group.replace("wrong_", "unique_fp_")] = sum(row["group"] == group for row in scored)
    if (any(value != frozen[name] for name, value in counts.items())
            or birth["reject"] != frozen["no_candidate_in_gate"] + frozen["nearest_exact_ties"]):
        raise ValueError(f"scene {scene} has different E-ON5 labelled counts: {counts}")


def _population(rows):
    return dict(births=len(rows), **{status: sum(row["disposition"] == status for row in rows)
                                    for status in ("returned", "timeout", "right_censored")})


def _distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return dict(n=0, min=None, median=None, p95=None, max=None)
    return dict(n=len(values), min=values[0], median=median(values),
                p95=values[math.ceil(.95 * len(values)) - 1], max=values[-1])


def _paired_comparison(rows, before, after):
    transitions = {origin: dict.fromkeys(OUTCOMES, 0) for origin in OUTCOMES}
    for row in rows:
        transitions[row["arms"][before]["outcome"]][row["arms"][after]["outcome"]] += 1
    delta = {outcome: sum(transitions[other][outcome] - transitions[outcome][other] for other in OUTCOMES)
             for outcome in OUTCOMES} if rows else None
    return dict(n=len(rows), transitions=transitions, delta=delta)


def _entrant_counts(rows, kind):
    entrants = [row["entrants"][kind] for row in rows if kind in row["entrants"]]
    return dict(
        births_with_keys=sum(item["keys"] > 0 for item in entrants),
        births_with_true_keys=sum(item["true_keys"] > 0 for item in entrants),
        births_with_true_in_gate_keys=sum(item["true_in_gate_keys"] > 0 for item in entrants),
        **{field: sum(item[field] for item in entrants) for field in (
            "keys", "true_keys", "in_gate_keys", "true_in_gate_keys", "selected_true", "selected_wrong")},
        first_true_report_after_birth_seconds=_distribution(item["first_true_wait"] for item in entrants),
    )


def summarize(rows):
    paired = [row for row in rows if row["disposition"] == "returned"]
    population = _population(rows)
    if population["births"] != population["returned"] + population["timeout"] + population["right_censored"]:
        raise RuntimeError("inconsistent next-report population")
    whole = {arm: _arm_counts(rows, arm) for arm in ARMS}
    if (whole["birth"]["no_decision"] != 0
            or any(whole[arm]["no_decision"] != len(rows) - len(paired) for arm in ARMS[1:])):
        raise RuntimeError("next-report arms do not share the same decision population")
    absence = {}
    for kind in ("prior_seen_stale", "never_seen"):
        missing = [row for row in rows if row["absence"] == kind]
        opened = [row for row in missing if "open" in row["arms"]]
        absence[kind] = dict(
            population=_population(missing), open=_arm_counts(missing, "open"),
            entrants={group: _entrant_counts(opened, group) for group in ENTRANTS},
            first_recovered_true_report_after_birth_seconds=_distribution(
                min((item["first_true_wait"] for item in row["entrants"].values()
                     if item["first_true_wait"] is not None), default=None) for row in opened),
        )
    return dict(
        population=population, whole=whole,
        paired=dict(n=len(paired), arms={arm: _arm_counts(paired, arm) for arm in ARMS},
                    comparisons={f"{after}_minus_{before}": _paired_comparison(paired, before, after)
                                 for before, after in COMPARISONS}),
        birth_groups={group: {arm: _arm_counts([row for row in rows if row["group"] == group], arm) for arm in ARMS}
                      for group in GROUPS},
        wrong_no_fresh_at_birth=absence,
        open_entrants={kind: _entrant_counts(paired, kind) for kind in ENTRANTS},
        waiting=dict(seconds=_distribution(row["wait"] for row in paired),
                     grid_seconds=_distribution(row["grid_wait"] for row in paired)),
    )


def qualify(scenes, aggregate):
    changes = {scene: report["paired"]["comparisons"]["open_minus_expiry"]["delta"]
               for scene, report in scenes.items()}
    supported = sum(change is not None for change in changes.values())
    nonworsening = sum(change is not None and change["correct"] >= 0 and change["wrong"] <= 0
                       for change in changes.values())
    delta = aggregate["paired"]["comparisons"]["open_minus_expiry"]["delta"]
    if supported < 2:
        status = "insufficient_scene_support"
    elif delta["correct"] >= 5 and delta["wrong"] <= 0 and nonworsening >= 2:
        status = "qualified"
    elif delta["correct"] < 0 and delta["wrong"] < 0:
        status = "coverage_loss"
    else:
        status = "not_qualified"
    return dict(status=status, scenes_with_returns=supported, nonworsening_scenes=nonworsening,
                delta=delta, scene_deltas=changes)


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
    predictions, fits = {}, {}
    for scene in SCENES:
        events = list(event_batches(read_archive_rows(args.archive, args.manifest, scene)))
        fit = estimate_translation(events)
        predictions[scene] = predict(events, fit)
        _check_birth_geometry(scene, predictions[scene], frozen[scene])
        fits[scene] = dict(status=fit["status"], reason=fit["reason"])

    scenes, all_scored = {}, []
    for scene in SCENES:
        scored = score(predictions[scene], read_labels(args.archive, args.manifest, scene))
        _check_birth_truth(scene, scored, frozen[scene])
        scenes[scene] = dict(summarize(scored), fit=fits[scene])
        all_scored.extend(scored)

    aggregate = summarize(all_scored)
    summary = dict(
        scene_ids=SCENES, scenes=scenes, aggregate=aggregate, qualification=qualify(scenes, aggregate),
        clock="ideal_measurement_time", fit_stop_seconds=FIT_STOP_S, birth_max_age_seconds=BIRTH_AGE_S,
        gate_metres=GATE_M, wait_limit_seconds=WAIT_S, administrative_end_seconds=END_S,
        grid_seconds_definition="ceil(t1) - ceil(t0), using t1 snapshot",
        p95_definition="nearest rank", wall_seconds=perf_counter() - began,
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
