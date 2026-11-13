"""Diagnose stale cross-source retrieval at the original Level1 birth event."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import perf_counter

from .confirmation_probe import (
    GROUPS, OUTCOMES, Selection, _check_birth_geometry, _check_birth_truth, _distribution,
    _grade, _paired_comparison,
)
from .evaluation_data import read_labels
from .measurements import read_archive_rows
from .positions import event_batches, position_step
from .registration_probe import (
    BIRTH_AGE_S, FIT_STOP_S, SCENES, estimate_translation, registration_distances,
)
from .rejection_probe import FROZEN_COUNTS, FROZEN_FIELDS, GATE_M, Fragment
from .stream import _advance_position


PRIMARY_ARM = "hold_600"
MOTION_ARM = "motion_600"
MOTION_SECONDS = 60
MAX_MOTION_GAP_SECONDS = 60
ARM_AGES = {"birth": BIRTH_AGE_S, "hold_120": 120, "hold_300": 300,
            PRIMARY_ARM: 600, "hold_all": None, MOTION_ARM: 600}
FROZEN_ABSENCE = {"prior_seen_stale": 14, "never_seen": 28}
FROZEN_ABSENCE_BY_SCENE = {
    "0": {"prior_seen_stale": 1, "never_seen": 5},
    "3": {"prior_seen_stale": 13, "never_seen": 23},
    "7": {"prior_seen_stale": 0, "never_seen": 0},
}
LEVELS = ("age_qualified_true", "true_in_gate", "true_strictly_beats_birth_winner",
          "unique_true_winner", "top_tie_includes_true", "new_false_outranks_rescuable_true",
          "new_false_ties_rescuable_true")
DISTRIBUTIONS = ("youngest_true_age_seconds", "minimum_true_distance_metres",
                 "youngest_in_gate_true_age_seconds")
ARM_COUNTERS = ("candidate_pairs", "in_gate_pairs", "true_opportunity", "true_in_gate", "false_in_gate",
                "no_candidate_in_gate", "nearest_exact_ties", "stale_candidate_pairs",
                "selected_stale", "projected_candidate_pairs", "selected_projected",
                "wrong_to_different_wrong_key_from_birth")


@dataclass(frozen=True)
class Edge:
    age: float
    held: float
    projected: float | None

    def distance(self, arm):
        return self.projected if arm == MOTION_ARM and self.projected is not None else self.held


@dataclass(frozen=True)
class Birth:
    edges: dict[Fragment, Edge]
    selections: dict[str, Selection]
    nearest: dict[str, frozenset[Fragment]]


def _project(last, previous):
    stamp, position = last
    earlier_stamp, earlier_position = previous
    gap = stamp - earlier_stamp
    if not 0 < gap <= MAX_MOTION_GAP_SECONDS:
        return None
    distance, reverse = position_step(position, earlier_position)
    return _advance_position(*position, distance / gap, (reverse + 180) % 360, MOTION_SECONDS)


def _select(edges, arm):
    max_age = ARM_AGES[arm]
    distances = {peer: edge.distance(arm) for peer, edge in edges.items()
                 if max_age is None or edge.age <= max_age}
    inside = {peer: distance for peer, distance in distances.items() if distance <= GATE_M}
    best = min(inside.values(), default=math.inf)
    nearest = frozenset(peer for peer, distance in inside.items() if distance == best)
    winner = next(iter(nearest)) if len(nearest) == 1 else None
    return Selection(frozenset(distances), frozenset(inside), winner), nearest


def _corrected(key, position, peer, point, fit):
    _, distance = registration_distances(key, position, peer, point, fit)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("invalid corrected distance")
    return distance


def predict(events, fit):
    """Freeze all candidate scores and selections at the end of each birth batch."""
    latest, previous, predictions = {}, {}, {}
    for stamp, positions in events:
        born = positions.keys() - latest.keys()
        for key, position in positions.items():
            if key in latest:
                previous[key] = latest[key]
            latest[key] = stamp, position
        if stamp <= FIT_STOP_S or not born:
            continue

        projected = {key: _project(entry, previous[key]) for key, entry in latest.items()
                     if key in previous and BIRTH_AGE_S < stamp - entry[0] <= ARM_AGES[MOTION_ARM]}
        for key in born:
            position = positions[key]
            edges = {}
            for peer, (seen, point) in latest.items():
                if key[0] == peer[0]:
                    continue
                held_distance = _corrected(key, position, peer, point, fit)
                motion_point = projected.get(peer)
                motion_distance = None if motion_point is None else _corrected(
                    key, position, peer, motion_point, fit)
                edges[peer] = Edge(stamp - seen, held_distance, motion_distance)
            selections, nearest = {}, {}
            for arm in ARM_AGES:
                selections[arm], nearest[arm] = _select(edges, arm)
            predictions[key] = Birth(edges, selections, nearest)
    return predictions


def _target_levels(birth, arm, truth):
    selection = birth.selections[arm]
    eligible = selection.fresh & truth
    inside = selection.in_gate & truth
    nearest = birth.nearest[arm]
    incumbent_distance = birth.edges[birth.selections["birth"].winner].held
    best_true = min((birth.edges[peer].distance(arm) for peer in inside), default=math.inf)
    beats_birth = best_true < incumbent_distance
    false_stale_top = nearest - truth - birth.selections["birth"].fresh
    return dict(
        age_qualified_true=bool(eligible), true_in_gate=bool(inside),
        true_strictly_beats_birth_winner=beats_birth, unique_true_winner=selection.winner in truth,
        top_tie_includes_true=len(nearest) > 1 and bool(nearest & truth),
        new_false_outranks_rescuable_true=beats_birth and bool(false_stale_top) and not nearest & truth,
        new_false_ties_rescuable_true=beats_birth and bool(false_stale_top) and bool(nearest & truth),
        youngest_true_age_seconds=min((birth.edges[peer].age for peer in eligible), default=None),
        minimum_true_distance_metres=min(
            (birth.edges[peer].distance(arm) for peer in eligible), default=None),
        youngest_in_gate_true_age_seconds=min((birth.edges[peer].age for peer in inside), default=None),
    )


def score(predictions, labels):
    """Grade frozen choices and discard fragment keys before forming the report."""
    by_identity = defaultdict(set)
    for key, identity in labels.items():
        by_identity[identity].add(key)
    rows = []
    for key, birth in predictions.items():
        if key not in labels or not birth.edges.keys() <= labels.keys():
            raise ValueError("unmapped fragment in post-replay labels")
        truth = by_identity[labels[key]]
        arms = {arm: _grade(selection, truth) for arm, selection in birth.selections.items()}
        baseline = arms["birth"]
        baseline_selection = birth.selections["birth"]
        group = baseline["outcome"]
        absence = None
        if group == "wrong":
            group = ("wrong_true_in_gate" if baseline["true_in_gate"] else
                     "wrong_true_only_outside_gate" if baseline["true_opportunity"] else "wrong_no_fresh_true")
            if group == "wrong_no_fresh_true":
                absence = "prior_seen_stale" if birth.edges.keys() & truth else "never_seen"

        for arm, selection in birth.selections.items():
            stale = selection.fresh - baseline_selection.fresh
            projected = {peer for peer in stale if arm == MOTION_ARM and birth.edges[peer].projected is not None}
            changed_wrong = (baseline["outcome"] == arms[arm]["outcome"] == "wrong"
                             and selection.winner != baseline_selection.winner)
            arms[arm].update(stale_candidate_pairs=len(stale), selected_stale=int(selection.winner in stale),
                             projected_candidate_pairs=len(projected),
                             selected_projected=int(selection.winner in projected),
                             wrong_to_different_wrong_key_from_birth=int(changed_wrong))
        diagnostics = {arm: _target_levels(birth, arm, truth) for arm in ARM_AGES} if (
            absence == "prior_seen_stale") else None
        rows.append(dict(group=group, absence=absence, arms=arms, diagnostics=diagnostics))
    return rows


def _arm_summary(rows, arm):
    counts = {outcome: sum(row["arms"][arm]["outcome"] == outcome for row in rows) for outcome in OUTCOMES}
    counts.update({field: sum(row["arms"][arm][field] for row in rows) for field in ARM_COUNTERS})
    counts["added_candidate_pairs_from_birth"] = sum(
        row["arms"][arm]["candidate_pairs"] - row["arms"]["birth"]["candidate_pairs"] for row in rows)
    counts["candidate_pairs_per_birth"] = _distribution(row["arms"][arm]["candidate_pairs"] for row in rows)
    return counts


def _comparisons(rows):
    pairs = [("birth", arm) for arm in ARM_AGES if arm != "birth"] + [(PRIMARY_ARM, MOTION_ARM)]
    return {f"{after}_minus_{before}": _paired_comparison(rows, before, after) for before, after in pairs}


def summarize(rows):
    by_group = {group: [row for row in rows if row["group"] == group] for group in GROUPS}
    by_group.update({group: [row for row in rows if row["absence"] == group] for group in FROZEN_ABSENCE})
    target = by_group["prior_seen_stale"]
    diagnostics = {}
    for arm in ARM_AGES:
        entries = [row["diagnostics"][arm] for row in target]
        diagnostics[arm] = {field: sum(item[field] for item in entries) for field in LEVELS}
        diagnostics[arm].update({field: _distribution(item[field] for item in entries)
                                 for field in DISTRIBUTIONS})
    return dict(
        births=len(rows), arms={arm: _arm_summary(rows, arm) for arm in ARM_AGES}, comparisons=_comparisons(rows),
        groups={group: dict(births=len(members), arms={arm: _arm_summary(members, arm) for arm in ARM_AGES})
                for group, members in by_group.items()},
        stale_true_diagnostics=diagnostics,
    )


def qualify(aggregate):
    comparison = aggregate["comparisons"][f"{PRIMARY_ARM}_minus_birth"]
    recovered = aggregate["stale_true_diagnostics"][PRIMARY_ARM]["unique_true_winner"]
    correct_to_wrong = comparison["transitions"]["correct"]["wrong"]
    delta = comparison["delta"]
    checks = dict(recovered_at_least_3=recovered >= 3, net_correct_at_least_3=delta["correct"] >= 3,
                  wrong_not_increased=delta["wrong"] <= 0, correct_to_wrong_at_most_1=correct_to_wrong <= 1)
    return dict(status="qualified" if all(checks.values()) else "not_qualified", checks=checks,
                recovered_prior_seen_stale=recovered, delta=delta, correct_to_wrong=correct_to_wrong)


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
        events = list(event_batches(read_archive_rows(args.archive, args.manifest, scene, required_split="T_core")))
        fit = estimate_translation(events)
        predictions[scene] = predict(events, fit)
        _check_birth_geometry(scene, predictions[scene], frozen[scene])
        fits[scene] = dict(status=fit["status"], reason=fit["reason"])

    scenes, all_scored = {}, []
    for scene in SCENES:
        scored = score(predictions[scene], read_labels(args.archive, args.manifest, scene))
        _check_birth_truth(scene, scored, frozen[scene])
        scenes[scene] = dict(summarize(scored), fit=fits[scene])
        observed = {kind: scenes[scene]["groups"][kind]["births"] for kind in FROZEN_ABSENCE}
        if observed != FROZEN_ABSENCE_BY_SCENE[scene]:
            raise ValueError(f"scene {scene} E-ON35 absence counts differ: {observed}")
        all_scored.extend(scored)

    aggregate = summarize(all_scored)
    observed_absence = {kind: aggregate["groups"][kind]["births"] for kind in FROZEN_ABSENCE}
    if observed_absence != FROZEN_ABSENCE:
        raise ValueError(f"E-ON35 absence counts differ: expected {FROZEN_ABSENCE}, observed {observed_absence}")
    summary = dict(
        scene_ids=SCENES, scenes=scenes, aggregate=aggregate, qualification=qualify(aggregate),
        clock="ideal_measurement_time", fit_stop_seconds=FIT_STOP_S, gate_metres=GATE_M,
        primary_arm=PRIMARY_ARM, arm_max_age_seconds=ARM_AGES, motion_advance_seconds=MOTION_SECONDS,
        motion_max_input_gap_seconds=MAX_MOTION_GAP_SECONDS, p95_definition="nearest rank",
        wall_seconds=perf_counter() - began,
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
