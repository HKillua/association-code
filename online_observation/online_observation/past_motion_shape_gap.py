"""Audit common-past motion changes within fixed online assignment windows."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_tracker import Plane
from .hypothesis_gap_diagnostic import Switch, _load_frozen, _position_readouts, _update_priors
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import _registration_shifts
from .past_motion_shape import score_past_motion
from .registration_probe import FIT_STOP_S
from .relative_motion_gap import _decompose, _same, _sub


def _capture(events, original, gaps, anchor_windows):
    by_stamp = defaultdict(list)
    captured = [[] for _ in gaps]
    keys = set()
    for index, (gap, window) in enumerate(zip(gaps, anchor_windows, strict=True)):
        switch = Switch(**gap["switch"])
        keys.update(switch.keys)
        for sample, control in window:
            stamp = sample["stamp"]
            if (not gap["start"] <= stamp <= gap["stop"] or frozenset(sample["keys"]) != switch.keys
                    or any(tuple(sorted((sample["reporter"], partner))) not in getattr(switch, side)
                           for side, partner in sample["partners"].items())):
                raise ValueError("historical shape readout does not match the original legal alternatives")
            by_stamp[stamp].append((index, gap["start"], switch, sample, control["scores"]))

    plane = Plane.from_first_batch(events[0][1].values())
    shifts = _registration_shifts(original.fit)
    priors, histories = {}, defaultdict(list)
    for (stamp, batch), native in zip(events, original.native_reports, strict=True):
        if (stamp, tuple(sorted(batch))) != native:
            raise ValueError("report timestamps or keys differ from the fixed graph replay")
        points = {key: plane.project(value) for key, value in batch.items() if key in keys}
        for key, point in points.items():
            histories[key].append((stamp, point))
        phase = shifts if stamp > FIT_STOP_S else {}
        for index, start, switch, sample, original_score in by_stamp.pop(stamp, ()):
            view = _position_readouts(switch.keys, stamp, points, priors, phase)
            if view is None or view[2] != {sample["reporter"]}:
                raise ValueError("the original single-report batch did not reproduce")
            masked, actual, new = view
            reporter = sample["reporter"]
            checks = list(zip(sample["self"], _sub(actual[reporter], masked[reporter]), strict=True))
            for side, partner in sample["partners"].items():
                checks.extend(zip(sample["cross"][side], _sub(actual[reporter], masked[partner]), strict=True))
                checks.extend(zip(sample["pred_delta"][side], _sub(masked[reporter], masked[partner]), strict=True))
            if not all(_same(saved, replayed) for saved, replayed in checks):
                raise ValueError("the original reported or masked positions did not reproduce")
            geometry = _decompose(switch, masked, actual, new)
            shape = score_past_motion(sample, switch, histories)
            provenance = shape.get("provenance")
            if provenance and any(t > stamp for times in provenance.values() for t in times):
                raise ValueError("a historical bracket used a later report")
            shape["end_after_start"] = shape["end"] is not None and shape["end"] > start
            shape["post_start_used"] = bool(provenance and any(
                t > start for times in provenance.values() for t in times))
            captured[index].append(dict(stamp=stamp, single=original_score,
                                        geometry={name: geometry[name] for name in ("actual", "masked")},
                                        shape=shape))
        _update_priors(priors, stamp, points)
    if by_stamp or [len(window) for window in captured] != [len(window) for window in anchor_windows]:
        raise ValueError("the frozen single-report population changed")
    return captured


def _join(captured, previous):
    rows = []
    for current, old in zip(captured, previous, strict=True):
        reports = []
        for item, baseline in zip(current, old["reports"], strict=True):
            if (item["stamp"], item["single"]) != (baseline["stamp"], baseline["single"]):
                raise ValueError("the frozen report times or comparison arms changed")
            shape = {key: value for key, value in item["shape"].items() if key != "provenance"}
            reports.append(item | {"shape": shape})
        rows.append(old | {"reports": reports})
    return rows


def _read(path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gap-run", type=Path, required=True)
    parser.add_argument("--anchor-run", type=Path, required=True)
    parser.add_argument("--closure-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    scenes = _roster(args.manifest)["splits"]["development"]
    if len(scenes) != 28 or scenes != read_roster(args.manifest)["splits"]["held"]:
        raise ValueError("the historical motion diagnostic requires the original 28 scenes")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    frozen = args.output_dir / "frozen"
    frozen.mkdir(mode=0o700)
    began = perf_counter()
    windows = reports = 0
    for index, scene in enumerate(scenes, 1):
        name = f"scene_{index:02d}.pkl.gz"
        original, gaps = _load_frozen(args.gap_run / "private_frozen" / name)
        captured = _capture(_events(args.archive, args.manifest, scene), original, gaps,
                            _read(args.anchor_run / "private_frozen" / name))
        with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
            pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)
        windows += len(captured)
        reports += sum(map(len, captured))
        print(f"Frozen {index}/{len(scenes)}, {len(captured)} groups", flush=True)
    print("All history scores frozen. Joining the previous anonymous evaluation.", flush=True)
    old = json.loads((args.closure_run / "summary.json").read_text(encoding="utf-8"))
    if (windows, reports) != (old["frozen_windows"], old["frozen_batches"]):
        raise ValueError("the original comparison population changed")
    output = {}
    for index in range(1, len(scenes)+1):
        alias = f"scene_{index:02d}"
        previous = old["scenes"][alias]
        output[alias] = {key: previous[key] for key in ("component", "sparse")} | {
            "rows": _join(_read(frozen / f"{alias}.pkl.gz"), previous["rows"])}
    _write_json(args.output_dir / "summary.json", dict(scenes=output, clock=old["clock"], frozen_windows=windows,
                frozen_batches=reports, publications_unchanged=True, candidate_graph_unchanged=True,
                training_updates=0, wall_seconds=perf_counter()-began))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
