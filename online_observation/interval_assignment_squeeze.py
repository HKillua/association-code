"""Decompose long wrong intervals from already frozen replay outputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import pickle

from .evaluation_data import read_labels
from .interval_assignment_experiment import COMPARISONS, METHODS, _pair_spans


def _long_origins(reference, candidate):
    counts = Counter()
    seconds = {kind: [] for kind in ("original_long", "original_short", "outside_original_wrong")}
    for pair, spans in candidate.items():
        for start, stop, censored in spans:
            if stop - start <= 60:
                continue
            short, long = [], []
            for prev_start, prev_stop, _ in reference.get(pair, ()):
                overlap = min(stop, prev_stop) - max(start, prev_start)
                if overlap > 0:
                    (long if prev_stop - prev_start > 60 else short).append(overlap)
            short_seconds = math.fsum(short)
            long_seconds = math.fsum(long)
            outside = stop - start - short_seconds - long_seconds
            if outside < -1e-8:
                raise ValueError("overlapping reference intervals for the same edge")
            seconds["original_long"].append(long_seconds)
            seconds["original_short"].append(short_seconds)
            seconds["outside_original_wrong"].append(max(0.0, outside))
            if long:
                counts["overlaps_long"] += 1
            elif short:
                counts["overlaps_only_short"] += 1
                counts["overlaps_multiple_short"] += len(short) > 1
            else:
                counts["no_positive_overlap"] += 1
            counts["right_censored"] += censored
    return {"episodes": dict(counts), "seconds": {kind: math.fsum(values) for kind, values in seconds.items()}}


def _distinct_outcomes(actions, true):
    masks = defaultdict(set)
    for action in actions:
        masks[action.removed, action.added].add(action.complete)
    mask_names = {frozenset({True}): "complete", frozenset({False}): "incomplete",
                  frozenset({False, True}): "both"}
    return dict(Counter(f"{sum(map(true, removed))}_to_{sum(map(true, added))}_"
                        f"{mask_names[frozenset(present)]}"
                        for (removed, added), present in masks.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = json.loads((args.frozen_run / "summary.json").read_text())
    rows = []
    for scene, entry in summary["scenes"].items():
        # Only use the private snapshots produced by interval_assignment_experiment.
        with gzip.open(args.frozen_run / "private_frozen" / f"{scene}.pkl.gz", "rb") as stream:
            result, actions = pickle.load(stream)
        labels = read_labels(args.archive, args.manifest, scene)

        def true(edge):
            return labels[edge[0]] == labels[edge[1]]

        wrong = {arm: {edge: spans for edge, spans in _pair_spans(result.native_reports, result.runs[arm]).items()
                       if not true(edge)} for arm in METHODS}
        origins = {}
        for before, after in COMPARISONS:
            record = _long_origins(wrong[before], wrong[after])
            expected = entry["arms"][after]["wrong"]
            actual = sum(record["episodes"].get(kind, 0) for kind in
                         ("overlaps_long", "overlaps_only_short", "no_positive_overlap"))
            if actual != expected["over60_episodes"] or not math.isclose(
                math.fsum(record["seconds"].values()), expected["over60_seconds"], abs_tol=1e-7,
            ):
                raise ValueError("post hoc long interval decomposition disagrees with frozen summary")
            origins[f"{after}_vs_{before}"] = record
        outcomes = {arm: _distinct_outcomes(choices, true) for arm, choices in actions.items()}
        if any(sum(values.values()) != entry["arms"][arm]["actions"]["all"]["distinct_actions"]
               for arm, values in outcomes.items()):
            raise ValueError("post hoc action counts disagree with frozen summary")
        rows.append(dict(component=entry["component"], sparse=entry["sparse"],
                         long_interval_origins=origins, distinct_action_outcomes=outcomes))
    with args.output.open("x") as stream:
        json.dump(dict(phase=summary["phase"], scenes=rows), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
