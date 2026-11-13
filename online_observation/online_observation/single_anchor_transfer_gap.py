"""Audit single-anchor transfer on the original frozen online assignment windows."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .async_report_scores import reported_direction
from .cross_source_calibration_gap import STAGES
from .evaluation_data import read_labels
from .interval_assignment_experiment import _roster
from .nemot_experiment import _write_json, read_roster
from .single_anchor_transfer import score_snapshot


def _freeze(rows):
    return [[(sample, score_snapshot(sample)) for sample in window] for window in rows]


def _public(sample, result, labels, anchors, identities):
    reporter = labels[sample["reporter"]]
    alternatives = [side for side, partner in sample["partners"].items() if labels[partner] == reporter]
    group_identities = {labels[key] for key in sample["keys"]}
    anchor = result["anchor"]
    outcome = {name: result[name] for name in ("scores", "reason", "decomposition", "equal_variance", "margin_delta")}
    outcome.update(stamp=sample["stamp"], true_side=alternatives[0] if len(alternatives) == 1 else None,
                   target_identity=str(identities.setdefault(reporter, len(identities))), anchor=None,
                   anchor_identity=None, anchor_age=None, anchor_correct=None, anchor_overlap=None,
                   available_anchors=sample["anchors"]["graph"]["n"],
                   recent_anchors=sample["anchors"]["graph"]["recent_n"])
    if anchor is not None:
        first, second = map(labels.__getitem__, anchor.edge)
        outcome.update(anchor=str(anchors.setdefault(anchor.edge, len(anchors))),
                       anchor_identity=str(identities.setdefault(first, len(identities))) if first == second else None,
                       anchor_age=sample["stamp"]-anchor.stamp, anchor_correct=first == second,
                       anchor_overlap=bool(group_identities.intersection((first, second))))
    return outcome


def _evaluate(frozen, previous, labels):
    anchors, identities = {}, {}
    output = []
    for window, old in zip(frozen, previous, strict=True):
        indexed = {sample["stamp"]: (sample, result) for sample, result in window}
        if len(indexed) != len(window):
            raise ValueError("a frozen single-report event is repeated inside a window")
        row = {key: value for key, value in old.items() if key not in STAGES}
        for stage in STAGES:
            baseline = old[stage]
            if baseline is None:
                row[stage] = None
                continue
            sample, result = indexed[baseline["stamp"]]
            row[stage] = public = _public(sample, result, labels, anchors, identities)
            if (public["true_side"] != baseline["true_side"] or public["scores"]["raw"] != baseline["scores"]["raw"]
                    or public["scores"]["masked"] != baseline["scores"]["masked"]
                    or (public["anchor"] is not None) != bool(baseline["anchors"]["graph"]["recent_n"])):
                raise ValueError("single-anchor readout differs from the frozen C-ON41 control or availability")
        row["first_corroborated"] = next((sample["stamp"] for sample, result in window if sample["stamp"] < row["stop"]
                                         and reported_direction(result["scores"]["corroborated"]) > 0), None)
        output.append(row)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    scenes = _roster(args.manifest)["splits"]["development"]
    if scenes != read_roster(args.manifest)["splits"]["held"]:
        raise ValueError("the diagnostic must use the original 28 scenes")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    destination = args.output_dir / "private_frozen"
    destination.mkdir(mode=0o700)
    began = perf_counter()
    total_windows = total_batches = 0
    for index, _ in enumerate(scenes, 1):
        name = f"scene_{index:02d}.pkl.gz"
        with gzip.open(args.calibration_run / "private_frozen" / name, "rb") as stream:
            windows = _freeze(pickle.load(stream))
        with gzip.open(destination / name, "xb", compresslevel=1) as stream:
            pickle.dump(windows, stream, protocol=pickle.HIGHEST_PROTOCOL)
        total_windows += len(windows)
        total_batches += sum(map(len, windows))
        print(f"Frozen {index}/{len(scenes)}, {len(windows)} groups", flush=True)

    print("All single-anchor scores and abstentions frozen. Reading evaluation labels.", flush=True)
    previous = json.loads((args.calibration_run / "summary.json").read_text(encoding="utf-8"))
    output = {}
    for index, scene in enumerate(scenes, 1):
        alias = f"scene_{index:02d}"
        old = previous["scenes"][alias]
        with gzip.open(destination / f"{alias}.pkl.gz", "rb") as stream:
            frozen = pickle.load(stream)
        rows = _evaluate(frozen, old["rows"], read_labels(args.archive, args.manifest, scene))
        output[alias] = {key: old[key] for key in ("component", "sparse", "registration")} | {"rows": rows}
    _write_json(args.output_dir / "summary.json", dict(scenes=output, clock=previous["clock"],
                candidate_graph_unchanged=True, publications_unchanged=True, training_updates=0,
                frozen_windows=total_windows, frozen_batches=total_batches, wall_seconds=perf_counter()-began))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
