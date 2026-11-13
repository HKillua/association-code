"""Trace whether two arrived reports supply a new frozen assignment contrast."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .async_report_scores import reported_direction
from .disjoint_report_closure import score_pair, select_pairs
from .interval_assignment_experiment import _roster
from .nemot_experiment import _write_json, read_roster


def _freeze(windows):
    captured = []
    for window in windows:
        history = []
        reports = []
        for sample, single in window:
            if history and history[-1][0]["stamp"] >= sample["stamp"]:
                raise ValueError("frozen group reports must be in strict native order")
            pairs = {}
            for kind, earlier in select_pairs((old for old, _ in history), sample).items():
                earlier_scores = next(scores for old, scores in reversed(history) if old is earlier)
                pair = score_pair(earlier, sample)
                pairs[kind] = pair | dict(
                    earlier_stamp=earlier["stamp"],
                    two_votes=all(reported_direction(values["raw"]) > 0
                                  for values in (earlier_scores, single["scores"])),
                    ages={side: (earlier["age"][side], sample["age"][side]) for side in ("release", "retain")},
                )
            reports.append(dict(stamp=sample["stamp"], single=single["scores"], pairs=pairs))
            history.append((sample, single["scores"]))
        captured.append(reports)
    return captured


def _join(windows, old_rows):
    output = []
    fields = ("start", "stop", "end", "late_support", "full_rejoin", "boundary_reports", "focal",
              "focal_true", "true_release", "true_retain", "target_edges", "switch_source")
    for reports, old in zip(windows, old_rows, strict=True):
        if any(not old["start"] <= item["stamp"] <= old["stop"] for item in reports):
            raise ValueError("a score was assigned outside the original window")
        early = [item for item in reports if item["stamp"] < old["stop"]]
        boundary = reports[-1] if reports and reports[-1]["stamp"] == old["stop"] else None
        for item, moment in ((early[-1] if early else None, "latest"), (boundary, "boundary")):
            expected = old[moment]
            if ((item is None) != (expected is None) or
                    (item is not None and (item["stamp"] != expected["stamp"] or item["single"] != expected["scores"]))):
                raise ValueError("the C-ON42 same-time comparator changed")
        output.append({field: old[field] for field in fields} | {"reports": reports})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    scenes = _roster(args.manifest)["splits"]["development"]
    if scenes != read_roster(args.manifest)["splits"]["held"]:
        raise ValueError("the diagnostic requires the original 28 scenes")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    destination = args.output_dir / "frozen"
    destination.mkdir(mode=0o700)
    began = perf_counter()
    windows = reports = 0
    for index, _ in enumerate(scenes, 1):
        name = f"scene_{index:02d}.pkl.gz"
        with gzip.open(args.anchor_run / "private_frozen" / name, "rb") as stream:
            captured = _freeze(pickle.load(stream))
        with gzip.open(destination / name, "xb", compresslevel=1) as stream:
            pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)
        windows += len(captured)
        reports += sum(map(len, captured))
        print(f"Frozen {index}/{len(scenes)}, {len(captured)} groups", flush=True)
    print("All report-pair scores frozen. Joining previous anonymous evaluation.", flush=True)
    previous = json.loads((args.anchor_run / "summary.json").read_text(encoding="utf-8"))
    if (windows, reports) != (previous["frozen_windows"], previous["frozen_batches"]):
        raise ValueError("the number of frozen groups or reports changed")
    output = {}
    for index, _ in enumerate(scenes, 1):
        alias = f"scene_{index:02d}"
        old = previous["scenes"][alias]
        with gzip.open(destination / f"{alias}.pkl.gz", "rb") as stream:
            captured = pickle.load(stream)
        output[alias] = {key: old[key] for key in ("component", "sparse")} | {"rows": _join(captured, old["rows"])}
    _write_json(args.output_dir / "summary.json", dict(scenes=output, clock=previous["clock"],
                candidate_graph_unchanged=True, publications_unchanged=True, training_updates=0,
                frozen_windows=windows, frozen_batches=reports, wall_seconds=perf_counter()-began))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
