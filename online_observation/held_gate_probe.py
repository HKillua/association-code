"""Compare frozen 1 km and 4 km birth gates on 32 held Level1 scenes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import perf_counter

from .evaluation_data import read_labels
from .measurements import read_archive_rows
from .positions import event_batches
from .registration_probe import (
    BIRTH_AGE_S, FIT_STOP_S, GATES_M, SCENES as DEVELOPMENT_SCENES,
    birth_predictions, estimate_translation,
)
from .rejection_probe import Fragment


SCENES = (
    "474", "563", "758", "775", "816", "865", "1093", "1098", "1151", "1390", "1534",
    "1769", "1906", "1936", "2331", "2464", "2626", "2670", "2811", "2966", "3065", "3084",
    "3443", "3621", "3624", "4243", "4456", "4579", "4695", "4718", "4741", "4811",
)
METHODS = ("raw", "corrected")
ARMS = tuple(f"{method}|{gate}" for method in METHODS for gate in GATES_M)
OUTCOMES = ("correct", "wrong", "reject")
OPPORTUNITIES = ("fresh", "stale", "never_observed")
REJECTION_REASONS = ("no_candidate", "distance", "tie")
POOL_SIZES = ("0-3", "4-15", "16+")


@dataclass(frozen=True)
class Choice:
    in_gate: frozenset[Fragment]
    winner: Fragment | None
    reject_reason: str | None


@dataclass(frozen=True)
class Birth:
    fresh: frozenset[Fragment]
    choices: dict[str, Choice]
    fitted_pairs: int


@dataclass(frozen=True)
class Outcome:
    decision: str
    truth_in_gate: bool
    reject_reason: str | None
    candidates_in_gate: int


@dataclass(frozen=True)
class ScoredBirth:
    source: str
    candidate_pairs: int
    fitted_pairs: int
    opportunity: str
    arms: dict[str, Outcome]


def _choose(distances, gate):
    within = {key: distance for key, distance in distances.items() if distance <= gate}
    if not within:
        return Choice(frozenset(), None, "distance" if distances else "no_candidate")
    nearest = min(within.values())
    winners = [key for key, distance in within.items() if distance == nearest]
    if len(winners) != 1:
        return Choice(frozenset(within), None, "tie")
    return Choice(frozenset(within), winners[0], None)


def predict(events, fit):
    """Record causal first reports and freeze four decisions on each birth."""
    events = tuple(events)
    first_seen = {}
    for stamp, positions in events:
        for key in positions:
            first_seen.setdefault(key, stamp)

    births = {}
    calibrated_sources = set(fit["source_order"]) if fit["translation_east_north_m"] is not None else set()
    for key, pairs in birth_predictions(events, fit):
        choices = {}
        for method, column in zip(METHODS, (1, 2), strict=True):
            distances = {pair[0]: pair[column] for pair in pairs}
            if any(not math.isfinite(distance) or distance < 0 for distance in distances.values()):
                raise ValueError("invalid candidate distance")
            narrow, wide = (_choose(distances, gate) for gate in GATES_M)
            if narrow.winner is not None and narrow.winner != wide.winner:
                raise RuntimeError("the narrow gate changed the accepted winner")
            choices[f"{method}|1000"] = narrow
            choices[f"{method}|4000"] = wide
        fitted_pairs = sum(key[0] in calibrated_sources and peer[0] in calibrated_sources for peer, *_ in pairs)
        births[key] = Birth(frozenset(peer for peer, *_ in pairs), choices, fitted_pairs)
    return first_seen, births


def score(first_seen, births, labels):
    """Score frozen decisions against cross-source labels available by the birth."""
    if not first_seen.keys() <= labels.keys():
        raise ValueError("unmapped measurement fragment in post-replay labels")
    earliest = defaultdict(dict)
    for key, stamp in first_seen.items():
        by_source = earliest[labels[key]]
        by_source[key[0]] = min(stamp, by_source.get(key[0], stamp))

    rows = []
    for key, birth in births.items():
        source = key[0]
        identity = labels[key]
        truth = frozenset(peer for peer in birth.fresh if labels[peer] == identity)
        previously_seen = any(peer_source != source and stamp <= first_seen[key]
                              for peer_source, stamp in earliest[identity].items())
        opportunity = "fresh" if truth else "stale" if previously_seen else "never_observed"
        arms = {}
        for arm, choice in birth.choices.items():
            outcome = ("reject" if choice.winner is None else
                       "correct" if choice.winner in truth else "wrong")
            arms[arm] = Outcome(outcome, bool(choice.in_gate & truth), choice.reject_reason, len(choice.in_gate))
        rows.append(ScoredBirth(source, len(birth.fresh), birth.fitted_pairs, opportunity, arms))
    return rows


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _arm_summary(rows, arm, fresh_truth):
    outcomes = Counter()
    reject_reasons = dict.fromkeys(REJECTION_REASONS, 0)
    wrong_by_opportunity = dict.fromkeys(OPPORTUNITIES, 0)
    wrong_by_in_gate_multiplicity = dict.fromkeys(("single", "multiple"), 0)
    no_history_wrong_by_in_gate_multiplicity = dict.fromkeys(("single", "multiple"), 0)
    true_in_gate = wrong_with_true_in_gate = reject_with_true_in_gate = 0
    for row in rows:
        result = row.arms[arm]
        outcomes[result.decision] += 1
        if result.reject_reason is not None:
            reject_reasons[result.reject_reason] += 1
        if result.decision == "wrong":
            wrong_by_opportunity[row.opportunity] += 1
            multiplicity = "single" if result.candidates_in_gate == 1 else "multiple"
            wrong_by_in_gate_multiplicity[multiplicity] += 1
            no_history_wrong_by_in_gate_multiplicity[multiplicity] += row.opportunity == "never_observed"
        true_in_gate += result.truth_in_gate
        wrong_with_true_in_gate += result.truth_in_gate and result.decision == "wrong"
        reject_with_true_in_gate += result.truth_in_gate and result.decision == "reject"

    correct, wrong, reject = (outcomes[outcome] for outcome in OUTCOMES)
    accepted = correct + wrong
    if (sum(reject_reasons.values()) != reject or
            correct + wrong_with_true_in_gate + reject_with_true_in_gate != true_in_gate):
        raise RuntimeError("inconsistent rejection reasons or true candidate outcomes")
    return dict(
        correct=correct, wrong=wrong, reject=reject, accepted=accepted, true_in_gate=true_in_gate,
        wrong_with_true_in_gate=wrong_with_true_in_gate, reject_with_true_in_gate=reject_with_true_in_gate,
        fresh_true_outside_gate=fresh_truth - true_in_gate,
        missed_fresh_truth=fresh_truth - correct, reject_reasons=reject_reasons,
        wrong_by_opportunity=wrong_by_opportunity,
        wrong_by_in_gate_multiplicity=wrong_by_in_gate_multiplicity,
        no_history_wrong_by_in_gate_multiplicity=no_history_wrong_by_in_gate_multiplicity,
        rates=dict(coverage=_rate(accepted, len(rows)), correct_per_birth=_rate(correct, len(rows)),
                   wrong_per_birth=_rate(wrong, len(rows)), wrong_per_accepted=_rate(wrong, accepted),
                   correct_per_fresh_truth=_rate(correct, fresh_truth)),
    )


def _transitions(rows, method):
    matrix = {outcome: dict.fromkeys(OUTCOMES, 0) for outcome in OUTCOMES}
    never_observed_wrong_4 = never_observed_wrong_1 = 0
    for row in rows:
        wide = row.arms[f"{method}|4000"].decision
        narrow = row.arms[f"{method}|1000"].decision
        if narrow != wide and narrow != "reject":
            raise RuntimeError("the narrow gate introduced a new acceptance")
        matrix[wide][narrow] += 1
        if row.opportunity == "never_observed":
            never_observed_wrong_4 += wide == "wrong"
            never_observed_wrong_1 += narrow == "wrong"
    return dict(
        matrix=matrix, true_accepts_lost=matrix["correct"]["reject"],
        false_accepts_removed=matrix["wrong"]["reject"],
        no_history_false_accepts_4000=never_observed_wrong_4,
        no_history_false_accepts_removed=never_observed_wrong_4 - never_observed_wrong_1,
        no_history_false_accepts_remaining_1000=never_observed_wrong_1,
    )


def _summarize(rows):
    counts = Counter(row.opportunity for row in rows)
    candidate_pairs = sum(row.candidate_pairs for row in rows)
    fitted_pairs = sum(row.fitted_pairs for row in rows)
    return dict(
        births=len(rows), opportunities={kind: counts[kind] for kind in OPPORTUNITIES},
        candidate_pairs=candidate_pairs, fitted_candidate_pairs=fitted_pairs,
        fallback_candidate_pairs=candidate_pairs - fitted_pairs,
        arms={arm: _arm_summary(rows, arm, counts["fresh"]) for arm in ARMS},
        transitions_4000_to_1000={method: _transitions(rows, method) for method in METHODS},
    )


def summarize(rows):
    by_source = defaultdict(list)
    by_pool_size = {label: [] for label in POOL_SIZES}
    for row in rows:
        by_source[row.source].append(row)
        index = 0 if row.candidate_pairs <= 3 else 1 if row.candidate_pairs <= 15 else 2
        by_pool_size[POOL_SIZES[index]].append(row)
    return dict(
        _summarize(rows),
        by_source={source: _summarize(items) for source, items in sorted(by_source.items())},
        by_pool_size={label: _summarize(items) for label, items in by_pool_size.items()},
    )


def identity_overlap(held, development):
    held_a = {identity.rsplit("-", 1)[0] for identity in held}
    development_a = {identity.rsplit("-", 1)[0] for identity in development}
    return {
        label: dict(held_unique=len(left), development_unique=len(right), shared=len(left & right))
        for label, left, right in (("A", held_a, development_a), ("A-B", held, development))
    }


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
        events = list(event_batches(read_archive_rows(args.archive, args.manifest, scene, required_split="T_core")))
        fit = estimate_translation(events)
        first_seen, births = predict(events, fit)
        prepared[scene] = fit, first_seen, births

    scenes, all_rows, held_identities = {}, [], set()
    fit_groups = {"fitted": [], "raw_fallback": []}
    for scene, (fit, first_seen, births) in prepared.items():
        labels = read_labels(args.archive, args.manifest, scene)
        held_identities.update(labels.values())
        rows = score(first_seen, births, labels)
        mode = "fitted" if fit["status"] == "fitted" else "raw_fallback"
        scenes[scene] = dict(summarize(rows), fit=fit, correction_mode=mode)
        fit_groups[mode].append((scene, rows))
        all_rows.extend(rows)
    development_identities = {identity for scene in DEVELOPMENT_SCENES
                              for identity in read_labels(args.archive, args.manifest, scene).values()}

    summary = dict(
        scene_ids=SCENES, scenes=scenes, aggregate=summarize(all_rows),
        by_fit_status={mode: dict(summarize([row for _, rows in members for row in rows]),
                                 scene_ids=[scene for scene, _ in members],
                                 scenes_with_births=sum(bool(rows) for _, rows in members))
                       for mode, members in fit_groups.items()},
        identity_overlap_with_development=dict(
            identity_overlap(held_identities, development_identities), development_scene_ids=DEVELOPMENT_SCENES,
            scope="all identity labels in the named scenes",
        ),
        clock="ideal_measurement_time", fit_stop_seconds=FIT_STOP_S, birth_max_age_seconds=BIRTH_AGE_S,
        gate_metres=GATES_M, tie_policy="abstain on exact nearest ties",
        history_definition="opposite-source first report time at or before the birth",
        wall_seconds=perf_counter() - began,
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
