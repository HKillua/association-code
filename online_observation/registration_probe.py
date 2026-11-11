"""Qualify a frozen relative position correction on Level1 T_core births."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

from .association_probe import SCENES
from .evaluation_data import read_labels
from .measurements import read_archive_rows
from .positions import event_batches, position_step


FIT_STOP_S = 600
FIT_STEP_S = 30
FIT_AGE_S = 20
FIT_RADIUS_M = 8000
BIN_WIDTH_M = 500
MIN_KEYS_PER_SOURCE = 3
MIN_EPOCHS = 3
BIRTH_AGE_S = 60
GATES_M = (1000, 4000)


def _displacement(left, right):
    distance, bearing = position_step(left, right)
    azimuth = math.radians(bearing)
    return distance, distance * math.sin(azimuth), distance * math.cos(azimuth)


def _bin_index(component):
    """Round to the nearest bin centre, with half-bin ties away from zero."""
    return int(math.copysign(math.floor(abs(component) / BIN_WIDTH_M + 0.5), component))


def _peak_evidence(index, votes):
    return dict(
        centre_east_north_m=[coordinate * BIN_WIDTH_M for coordinate in index],
        votes=len(votes),
        distinct_epochs=len({vote[2] for vote in votes}),
        distinct_fragments_per_source=[len({vote[side] for vote in votes}) for side in (3, 4)],
    )


def estimate_translation(events):
    """Fit source 1 minus source 0 from unlabeled reports through 600 s."""
    prefix = [(stamp, positions) for stamp, positions in events if 0 <= stamp <= FIT_STOP_S]
    sources = sorted({key[0] for _, positions in prefix for key in positions})
    audit = dict(
        source_order=sources, status="abstained", reason=None, translation_east_north_m=None,
        eligible_vectors=0, epochs_with_votes=0, occupied_bins=0, peak_count=0,
        next_lower_count=0, tied_peak_bins=0, peak_evidence=[],
    )
    if len(sources) != 2:
        return dict(audit, reason="expected_two_prefix_sources")

    pending = iter(prefix)
    event = next(pending, None)
    latest = {}
    bins = defaultdict(list)
    for epoch in range(FIT_STEP_S, FIT_STOP_S + 1, FIT_STEP_S):
        while event is not None and event[0] <= epoch:
            stamp, positions = event
            latest.update({key: (stamp, position) for key, position in positions.items()})
            event = next(pending, None)
        recent = {source: [] for source in sources}
        for key, (stamp, position) in latest.items():
            if epoch - stamp <= FIT_AGE_S:
                recent[key[0]].append((key, position))
        before = audit["eligible_vectors"]
        for left_key, left in recent[sources[0]]:
            for right_key, right in recent[sources[1]]:
                distance, east, north = _displacement(left, right)
                if distance <= FIT_RADIUS_M:
                    index = (_bin_index(east), _bin_index(north))
                    bins[index].append((east, north, epoch, left_key, right_key))
                    audit["eligible_vectors"] += 1
        audit["epochs_with_votes"] += audit["eligible_vectors"] > before

    if not bins:
        return dict(audit, reason="no_eligible_vectors")
    count = max(map(len, bins.values()))
    peaks = sorted(index for index, votes in bins.items() if len(votes) == count)
    audit.update(
        occupied_bins=len(bins), peak_count=count, tied_peak_bins=len(peaks) if len(peaks) > 1 else 0,
        next_lower_count=max((len(votes) for votes in bins.values() if len(votes) < count), default=0),
        peak_evidence=[_peak_evidence(index, bins[index]) for index in peaks],
    )
    if len(peaks) != 1:
        return dict(audit, reason="tied_peak_counts")
    peak = audit["peak_evidence"][0]
    if (min(peak["distinct_fragments_per_source"]) < MIN_KEYS_PER_SOURCE
            or peak["distinct_epochs"] < MIN_EPOCHS):
        return dict(audit, reason="insufficient_peak_support")
    votes = bins[peaks[0]]
    return dict(audit, status="fitted", translation_east_north_m=[
        median(vote[0] for vote in votes), median(vote[1] for vote in votes),
    ])


def birth_predictions(events, fit):
    """Generate paired distances for the same causal cross-source candidates."""
    source_order = fit["source_order"]
    translation = fit["translation_east_north_m"]
    latest = {}
    for stamp, positions in events:
        births = positions.keys() - latest.keys()
        latest.update({key: (stamp, position) for key, position in positions.items()})
        if stamp <= FIT_STOP_S or not births:
            continue
        recent = {key: position for key, (seen, position) in latest.items() if stamp - seen <= BIRTH_AGE_S}
        for key in births:
            pairs = []
            for other_key, other in recent.items():
                if key[0] == other_key[0]:
                    continue
                # Always orient the vector by source, regardless of which fragment was born.
                if key[0] < other_key[0]:
                    ordered_sources, left, right = [key[0], other_key[0]], positions[key], other
                else:
                    ordered_sources, left, right = [other_key[0], key[0]], other, positions[key]
                raw, east, north = _displacement(left, right)
                corrected = raw
                if translation is not None and ordered_sources == source_order:
                    corrected = math.hypot(east - translation[0], north - translation[1])
                pairs.append((other_key, raw, corrected))
            yield key, pairs


_COUNTERS = (
    "births", "candidate_pairs", "true_opportunity", "true_in_gate", "false_in_gate",
    "true_and_false_in_gate", "no_candidate_in_gate", "nearest_all_true", "nearest_all_false",
    "nearest_exact_ties", "nearest_mixed_ties", "unique_correct_matches", "unique_false_matches",
    "wrong_nearest_with_true_in_gate", "wrong_nearest_without_opportunity",
)


def evaluate_births(predictions, labels):
    results = {
        f"{method}|{gate}": Counter(dict.fromkeys(_COUNTERS, 0))
        for method in ("raw", "corrected") for gate in GATES_M
    }
    for key, pairs in predictions:
        if key not in labels or any(other not in labels for other, *_ in pairs):
            raise ValueError("unmapped measurement fragment in post-replay labels")
        candidates = [(labels[other] == labels[key], raw, corrected) for other, raw, corrected in pairs]
        opportunity = any(truth for truth, *_ in candidates)
        for method, distance_column in (("raw", 1), ("corrected", 2)):
            for gate in GATES_M:
                result = results[f"{method}|{gate}"]
                result["births"] += 1
                result["candidate_pairs"] += len(candidates)
                result["true_opportunity"] += opportunity
                within = [(entry[distance_column], entry[0]) for entry in candidates
                          if entry[distance_column] <= gate]
                true_in_gate = any(truth for _, truth in within)
                false_in_gate = any(not truth for _, truth in within)
                result["true_in_gate"] += true_in_gate
                result["false_in_gate"] += false_in_gate
                result["true_and_false_in_gate"] += true_in_gate and false_in_gate
                if not within:
                    result["no_candidate_in_gate"] += 1
                    continue
                minimum = min(distance for distance, _ in within)
                winners = [truth for distance, truth in within if distance == minimum]
                tied = len(winners) > 1
                result["nearest_exact_ties"] += tied
                if all(winners):
                    result["nearest_all_true"] += 1
                    result["unique_correct_matches"] += not tied
                elif any(winners):
                    result["nearest_mixed_ties"] += 1
                else:
                    result["nearest_all_false"] += 1
                    result["unique_false_matches"] += not tied
                    result["wrong_nearest_with_true_in_gate"] += true_in_gate
                    result["wrong_nearest_without_opportunity"] += not opportunity
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    began = perf_counter()

    predictions_by_scene, scenes = {}, {}
    for scene in SCENES:
        events = list(event_batches(read_archive_rows(args.archive, args.manifest, scene)))
        fit = estimate_translation(events)
        predictions_by_scene[scene] = list(birth_predictions(events, fit))
        scenes[scene] = dict(fit=fit, correction_mode="shift" if fit["status"] == "fitted" else "raw_fallback")

    for scene, predictions in predictions_by_scene.items():
        scenes[scene]["births"] = evaluate_births(predictions, read_labels(args.archive, args.manifest, scene))

    summary = dict(
        scene_ids=SCENES, scenes=scenes, clock="ideal_measurement_time",
        population="directed_cross_source_fragment_births_strictly_after_fit",
        fit_start_seconds=0, fit_stop_seconds=FIT_STOP_S, fit_epoch_step_seconds=FIT_STEP_S,
        fit_max_age_seconds=FIT_AGE_S, fit_max_vector_metres=FIT_RADIUS_M, fit_bin_width_metres=BIN_WIDTH_M,
        fit_min_fragments_per_source=MIN_KEYS_PER_SOURCE, fit_min_epochs=MIN_EPOCHS,
        birth_max_age_seconds=BIRTH_AGE_S, birth_gates_metres=GATES_M,
        orientation="source_order[0] to source_order[1]", tie_policy="abstain on exact ties",
        wall_seconds=perf_counter() - began,
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
