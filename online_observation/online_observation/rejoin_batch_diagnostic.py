"""Audit same-batch rejoin decisions on the frozen MTAD position replay."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .hypothesis_gap_diagnostic import _direction
from .nemot_experiment import _write_json
from .rejoin_batch_squeeze import summarize


ARMS = ("original", "geometry", "geometry_masked", "net_geometry", "two_block_geometry",
        "masked_shape_geometry", "cached_shape_geometry")


def _report_at(reports, stop):
    early, boundary = None, None
    last = float("-inf")
    for report in reports:
        stamp = report["stamp"]
        if not last < stamp <= stop:
            raise ValueError("a fixed report is out of order or past the window")
        if stamp == stop:
            boundary = report
        else:
            early = report
        last = stamp
    return early, boundary


def _shape(report, view, feature):
    if report is None or report["shape"][view] is None:
        return None
    shape = report["shape"]
    if shape["available_at"] > report["stamp"]:
        raise ValueError("the shape uses a report after its native time")
    return shape[view][feature]


def _capture(reports):
    """Freeze the last report and its predecessor before joining trigger metadata."""
    if not reports:
        return None
    stop = reports[-1]["stamp"]
    earlier, current = _report_at(reports, stop)
    geometry = _direction(current["geometry"]["actual"]) > 0
    cached = _shape(earlier, "actual", "two_block")
    scores = (("net_geometry", _shape(current, "actual", "net")),
              ("two_block_geometry", _shape(current, "actual", "two_block")),
              ("masked_shape_geometry", _shape(current, "masked", "two_block")),
              ("cached_shape_geometry", cached))
    decisions = {name: None if score is None else geometry and _direction(score) > 0 for name, score in scores}
    return dict(stamp=stop, decisions=dict(original=False, geometry=geometry,
                geometry_masked=_direction(current["geometry"]["masked"]) > 0) | decisions,
                current_used=current["shape"]["current_used"],
                end_after_start=current["shape"]["end_after_start"],
                post_start_used=current["shape"]["post_start_used"],
                cache_stamp=earlier["stamp"] if earlier else None,
                shape_end=current["shape"]["end"], cache_end=earlier["shape"]["end"] if earlier else None)


def _read_frozen(path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    frozen = args.output_dir / "frozen"
    frozen.mkdir(mode=0o700)
    began = perf_counter()
    windows = batches = 0
    for index in range(1, 29):
        filename = f"scene_{index:02d}.pkl.gz"
        records = _read_frozen(args.shape_run / "frozen" / filename)
        # Capture hypotheses for the last report without knowing outcomes, then
        # verify its trigger against the separate frozen replay clock below.
        captures = [_capture(reports) for reports in records]
        with gzip.open(frozen / filename, "xb", compresslevel=1) as stream:
            pickle.dump(captures, stream, protocol=pickle.HIGHEST_PROTOCOL)
        windows += len(records)
        batches += sum(map(len, records))
    print("All candidate decisions frozen. Joining original trigger and anonymous evaluation.", flush=True)
    previous = json.loads((args.shape_run / "summary.json").read_text(encoding="utf-8"))
    if (windows, batches, len(previous["scenes"])) != (482, 1477, 28):
        raise ValueError("the original fixed population changed")
    rows, trigger = [], Counter()
    for alias, scene in previous["scenes"].items():
        captures = _read_frozen(frozen / f"{alias}.pkl.gz")
        for row, capture in zip(scene["rows"], captures, strict=True):
            eligible = row["end"] == "kf_rejoin" and row["full_rejoin"] and row["boundary_reports"] == 1
            if not eligible or capture is None or capture["stamp"] != row["stop"]:
                rows.append((scene, row, dict.fromkeys(ARMS)))
                if eligible:
                    trigger["missing_current_readout"] += 1
                continue
            latest, boundary = _report_at(row["reports"], row["stop"])
            if boundary is None or (latest["stamp"] if latest else None) != capture["cache_stamp"]:
                raise ValueError("an evaluated decision used a different clock")
            decisions = capture["decisions"]
            if any(value is not None for name, value in decisions.items() if name.startswith("cached_")):
                if not capture["cache_stamp"] < capture["stamp"]:
                    raise ValueError("the cached shape must predate the current batch")
            if not capture["current_used"] and decisions["two_block_geometry"] != decisions["masked_shape_geometry"]:
                raise ValueError("current-coordinate masking changed an unsupported coordinate")
            trigger.update(eligible=1, current_used=int(capture["current_used"]),
                           post_start_used=int(capture["post_start_used"]),
                           end_after_start=int(capture["end_after_start"]),
                           shape_flip=int(decisions["two_block_geometry"] != decisions["masked_shape_geometry"]))
            rows.append((scene, row, decisions))
    comparisons = summarize(rows)
    primary = comparisons["primary"]
    if (primary["target_windows"], primary["correct_windows"]) != (28, 142):
        raise ValueError("the original sparse target or control cohort changed")
    _write_json(args.output_dir / "summary.json", dict(
        clock=previous["clock"], original_publications_unchanged=True, original_candidate_graph_unchanged=True,
        new_measurement_reads=0, training_updates=0, external_dataset_reads=0,
        wall_seconds=perf_counter()-began, frozen_windows=windows, frozen_batches=batches,
        trigger=dict(trigger), **comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
