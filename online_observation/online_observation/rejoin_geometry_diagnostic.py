"""Audit the scale and meaning of the fixed MTAD rejoin geometry increment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .async_report_scores import reported_direction
from .hypothesis_gap_diagnostic import Switch, _direction, _load_frozen
from .nemot_experiment import _write_json
from .rejoin_batch_diagnostic import _read_frozen, _report_at
from .rejoin_geometry_increment import score_increment
from .rejoin_geometry_squeeze import summarize
from .relative_motion_gap import _same


CONTROLS = ("original", "current", "cache", "union", "geometry", "geometry_masked")
ARMS = (*CONTROLS, "raw_positive", "nominal_gaussian", "nominal_correction")


def _prior_ages(native_reports, requested):
    last = {}
    selected = {}
    for stamp, keys in native_reports:
        if stamp in requested:
            selected[stamp] = {key: stamp-last[key] for key in requested[stamp] if key in last}
        last.update(dict.fromkeys(keys, stamp))
    return selected


def _features(sample, boundary, ages):
    geometry = boundary["geometry"]
    output = score_increment(sample, geometry)
    reporter_age = ages[sample["reporter"]]
    if reporter_age <= 0 or any(
            not _same(ages[partner], sample["age"][side]) for side, partner in sample["partners"].items()):
        raise ValueError("the prior report ages did not reproduce on the original clock")
    return output | dict(geometry_m2=geometry["actual"], geometry_masked_m2=geometry["masked"],
                         reporter_age_s=reporter_age, release_age_s=sample["age"]["release"],
                         retain_age_s=sample["age"]["retain"])


def _actions(base, features):
    actions = {arm: base[arm] for arm in CONTROLS}
    raw = _direction(features["increment_m2"]) > 0
    gaussian = (reported_direction(features["gaussian_actual"]) > 0
                and reported_direction(features["gaussian_increment"]) > 0)
    correction = gaussian and _direction(-features["release_change_m2"]) > 0
    for arm, vote in (("raw_positive", raw), ("nominal_gaussian", gaussian), ("nominal_correction", correction)):
        actions[arm] = (None if base["current"] is None else
                        base["current"] or (base["cache"] is True and vote))
    return actions


def _capture(original, gaps, anchor, shapes, cached):
    requested = defaultdict(set)
    candidates = []
    for gap, reports, shape, old in zip(gaps, anchor, shapes, cached, strict=True):
        stamp = gap["stop"]
        _, boundary = _report_at(shape, stamp)
        if old is None:
            candidates.append(None)
            continue
        switch = Switch(**gap["switch"])
        at_boundary = [sample for sample, _ in reports if sample["stamp"] == stamp]
        if (len(at_boundary) != 1 or boundary is None or gap["end"] != "kf_rejoin"
                or (gap["start"], stamp) != (old["start"], old["stop"])):
            raise ValueError("a rejoin record did not reproduce on the fixed windows")
        sample, = at_boundary
        if (set(sample["keys"]) != switch.keys or any(
                tuple(sorted((sample["reporter"], partner))) not in getattr(switch, side)
                for side, partner in sample["partners"].items())):
            raise ValueError("the fixed two matchings or reporter changed")
        requested[stamp].update((sample["reporter"], *sample["partners"].values()))
        candidates.append((stamp, sample, boundary, old))

    ages = _prior_ages(original.native_reports, requested)
    output = []
    for candidate in candidates:
        if candidate is None:
            output.append(None)
            continue
        stamp, sample, boundary, old = candidate
        features = _features(sample, boundary, ages[stamp])
        saved = old["features"]
        if saved is not None and not _same(saved["geometry_increment_m2"], features["increment_m2"]):
            raise ValueError("the C-ON47 unnormalized geometric increment changed")
        output.append(dict(stamp=stamp, features=features, decisions=_actions(old["decisions"], features)))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gap-run", "anchor-run", "shape-run", "cache-run", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-frozen", type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    began = perf_counter()
    filenames = tuple(f"scene_{index:02d}.pkl.gz" for index in range(1, 29))
    frozen = args.resume_frozen or args.output_dir / "frozen"
    if args.resume_frozen is None:
        frozen.mkdir(mode=0o700)
        for name in filenames:
            original, gaps = _load_frozen(args.gap_run / "private_frozen" / name)
            captured = _capture(original, gaps, _read_frozen(args.anchor_run / "private_frozen" / name),
                                _read_frozen(args.shape_run / "frozen" / name),
                                _read_frozen(args.cache_run / "frozen" / name))
            with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
                pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)
    if {path.name for path in frozen.glob("*.pkl.gz")} != set(filenames):
        raise ValueError("the replay requires all 28 frozen scenes, including empty ones")
    print("All geometric evidence and actions frozen. Joining anonymous evaluation.", flush=True)
    previous = json.loads((args.cache_run / "rows.json").read_text(encoding="utf-8"))
    by_scene = defaultdict(list)
    for row in previous:
        by_scene[row["scene"]].append(row)
    output = []
    aliases = {name.removesuffix(".pkl.gz") for name in filenames}
    if not by_scene.keys() <= aliases:
        raise ValueError("a joined row has a scene outside the frozen roster")
    for name in filenames:
        alias = name.removesuffix(".pkl.gz")
        captured = _read_frozen(frozen / name)
        for old, data in zip(by_scene.get(alias, ()), captured, strict=True):
            if old["decisions"]["original"] is None:
                output.append(old | dict(features=None, decisions=dict.fromkeys(ARMS)))
                continue
            if data is None or data["stamp"] != old["row"]["stop"] or any(
                    data["decisions"][arm] != old["decisions"][arm] for arm in CONTROLS):
                raise ValueError("the original causal trigger or controls changed")
            output.append(old | dict(features=data["features"], decisions=data["decisions"]))
    if (len(output), sum(x["features"] is not None for x in output)) != (482, 416):
        raise ValueError("the original evaluation population changed")
    comparisons = summarize(output)
    _write_json(args.output_dir / "rows.json", output)
    _write_json(args.output_dir / "summary.json", dict(
        clock="ideal_native_measurement_time", frozen_scenes=len(filenames), frozen_windows=len(output), eligible_rejoins=416,
        resumed_frozen=args.resume_frozen is not None,
        new_measurement_reads=0, external_dataset_reads=0, training_updates=0,
        original_candidate_graph_unchanged=True, original_publications_unchanged=True,
        wall_seconds=perf_counter()-began, **comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
