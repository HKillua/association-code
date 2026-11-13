"""Diagnose position-only fragment births before implementing an online decoder."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

from .evaluation_data import read_labels
from .measurements import read_archive_rows
from .positions import event_batches, position_step
from .stream import _advance_position


SCENES = ("0", "3", "7")
GATES_M = (250, 1000, 4000)
MAX_AGES_S = (60, 600)


def birth_candidates(rows):
    """Compute causal physical distances before exposing target labels."""
    latest = {}
    for stamp, positions in event_batches(rows):
        births = positions.keys() - latest.keys()
        if births:
            # Every same-time report is available at this event. The source of
            # a candidate controls capacity; the batch number has no rank.
            visible = latest | {key: (stamp, position, 0., 0.) for key, position in positions.items()}
            recent = {}
            for key, (time, position, speed, course) in visible.items():
                age = stamp - time
                if age <= max(MAX_AGES_S):
                    projected = _advance_position(*position, speed, course, age) if age and speed else position
                    recent[key] = (age, position, projected)
            for key in births:
                position = positions[key]
                yield key, [
                    (candidate, age, position_step(position, hold)[0], position_step(position, projected)[0])
                    for candidate, (age, hold, projected) in recent.items() if candidate != key
                ]
        for key, position in positions.items():
            if key in latest:
                last_time, last_position, _, _ = latest[key]
                distance, reverse = position_step(position, last_position)
                latest[key] = (stamp, position, distance / (stamp - last_time), (reverse + 180) % 360)
            else:
                latest[key] = (stamp, position, 0., 0.)


def evaluate_births(predictions, labels):
    counters = defaultdict(Counter)
    for key, candidates in predictions:
        if key not in labels or any(candidate not in labels for candidate, *_ in candidates):
            raise ValueError("unmapped measurement fragment in post-replay labels")
        for relation in ("same_source", "cross_source"):
            same_source = relation == "same_source"
            related = [
                (age, hold, secant, labels[candidate] == labels[key])
                for candidate, age, hold, secant in candidates
                if (key[0] == candidate[0]) == same_source
            ]
            for horizon in MAX_AGES_S:
                eligible = [item for item in related if item[0] <= horizon]
                opportunity = any(item[-1] for item in eligible)
                for method, index in (("hold", 1), ("secant", 2)):
                    for gate in GATES_M:
                        result = counters[relation, horizon, method, gate]
                        result["births"] += 1
                        result["true_opportunity"] += opportunity
                        result["true_opportunity_age_zero"] += any(
                            age == 0 and truth for age, _, _, truth in eligible
                        )
                        within = [(item[index], item[-1]) for item in eligible if item[index] <= gate]
                        true_within = any(truth for _, truth in within)
                        result["true_in_gate"] += true_within
                        result["true_and_false_in_gate"] += true_within and any(not truth for _, truth in within)
                        if not within:
                            result["no_candidate_in_gate"] += 1
                            continue
                        best = min(distance for distance, _ in within)
                        winners = [truth for distance, truth in within if distance == best]
                        if all(winners):
                            result["nearest_all_true"] += 1
                        elif any(winners):
                            result["nearest_mixed_tie"] += 1
                        else:
                            result["nearest_all_false"] += 1
                            result["wrong_nearest_with_true_in_gate"] += true_within
                            result["wrong_nearest_without_opportunity"] += not opportunity
    return counters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    began = perf_counter()
    scenes, pooled = {}, defaultdict(Counter)
    for scene in SCENES:
        predictions = list(birth_candidates(read_archive_rows(args.archive, args.manifest, scene)))
        labels = read_labels(args.archive, args.manifest, scene)
        evaluated = evaluate_births(predictions, labels)
        scenes[scene] = {"|".join(map(str, key)): dict(value) for key, value in evaluated.items()}
        for key, values in evaluated.items():
            pooled[key].update(values)
    summary = dict(
        scenes=scenes, pooled={"|".join(map(str, key)): dict(value) for key, value in pooled.items()},
        scene_ids=SCENES, max_ages_seconds=MAX_AGES_S, gates_metres=GATES_M,
        clock="ideal_measurement_time", population="directed_fragment_births",
        wall_seconds=perf_counter() - began,
    )
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
