"""Audit the causal support of cached motion at frozen MTAD rejoin batches."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_tracker import FRESH_S
from .hypothesis_gap_diagnostic import Switch, _direction, _load_frozen
from .nemot_experiment import _write_json
from .rejoin_batch_diagnostic import _capture as original_decisions, _read_frozen, _report_at
from .rejoin_cache_squeeze import ARMS, summarize
from .rejoin_persistence import RejoinPersistence


def _support_keys(shape):
    return frozenset((key, stamp) for key, stamps in shape["provenance"].items() for stamp in stamps)


def _qualify(current, cache, conditions):
    if current is None:
        return None
    return current or (cache is True and all(conditions))


def _features(earlier, boundary, history, switch, start, stop):
    shape = earlier["shape"]
    if not shape["end"] <= shape["available_at"] <= earlier["stamp"] < stop:
        raise ValueError("the cached support clocks are not causal")
    now = boundary["shape"]
    spans = history.at(switch, stop)
    kf, i22 = spans["kf_release_s"], spans["i22_retain_s"]
    if min(kf, i22) + 1e-6 < stop-start or spans["kf_retain_s"] != 0:
        raise ValueError("the strict-pre-rejoin state disagrees with the original risk window")
    cache_support = _support_keys(shape)
    current_support = _support_keys(now) if now["actual"] is not None else None
    geometry = boundary["geometry"]
    cache_end_age = stop-shape["end"]
    covers_end = kf > 0 and cache_end_age <= kf+1e-6
    return dict(
        cache_age_s=stop-earlier["stamp"], support_age_s=stop-shape["available_at"],
        cache_end_age_s=cache_end_age, gap_duration_s=stop-start,
        current_end_age_s=stop-now["end"] if now["end"] is not None else None,
        geometry_m2=geometry["actual"], geometry_masked_m2=geometry["masked"],
        geometry_increment_m2=geometry["actual"]-geometry["masked"],
        cache_geometry_m2=earlier["geometry"]["actual"],
        cached_shape_m2=shape["actual"]["two_block"],
        current_shape_m2=now["actual"]["two_block"] if now["actual"] is not None else None,
        current_masked_shape_m2=now["masked"]["two_block"] if now["masked"] is not None else None,
        release_at_cache_s=history.duration("kf", switch.release, earlier["stamp"]),
        retain_at_cache_s=history.duration("i22", switch.retain, earlier["stamp"]),
        original_retain_before_gap_s=history.duration("kf", switch.retain, start),
        cache_support_reuse=current_support == cache_support if current_support is not None else None,
        new_support_points=len(current_support-cache_support) if current_support is not None else None,
        recent=cache_end_age <= FRESH_S, cache_end_within_release=covers_end,
        continuous_from_cache_end=covers_end and kf+1e-6 >= i22,
        geometry_stable=all(_direction(geometry[view]) > 0 for view in ("actual", "masked")),
        **spans), cache_support


def _capture(reports, gap, history, original, tokens):
    start, stop = gap["start"], gap["stop"]
    earlier, boundary = _report_at(reports, stop)
    if gap["end"] != "kf_rejoin" or boundary is None:
        return None
    observed = original_decisions(reports)
    if observed != original:
        raise ValueError("the C-ON46 decision does not reproduce")
    base = observed["decisions"]
    current, cache = base["two_block_geometry"], base["cached_shape_geometry"]
    decisions = dict(original=False, geometry=base["geometry"], geometry_masked=base["geometry_masked"],
                     current=current, cache=cache, union=_qualify(current, cache, (True,)))
    features = None
    if cache is not None:
        switch = Switch(**gap["switch"])
        features, support = _features(earlier, boundary, history, switch, start, stop)
        features["cache_id"] = str(tokens.setdefault((switch, support), len(tokens)))
    for arm, condition in (("age", ("recent",)), ("persistence", ("continuous_from_cache_end",)),
                           ("geometry_agreement", ("geometry_stable",)),
                           ("qualified", ("recent", "continuous_from_cache_end", "geometry_stable"))):
        decisions[arm] = _qualify(current, cache, (features[key] for key in condition) if features else (False,))
    return dict(start=start, stop=stop, features=features, decisions=decisions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-run", type=Path, required=True)
    parser.add_argument("--shape-run", type=Path, required=True)
    parser.add_argument("--rejoin-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    frozen = args.output_dir / "frozen"
    frozen.mkdir(mode=0o700)
    began = perf_counter()
    for index in range(1, 29):
        name = f"scene_{index:02d}.pkl.gz"
        result, gaps = _load_frozen(args.gap_run / "private_frozen" / name)
        history = RejoinPersistence(result)
        inputs = (_read_frozen(args.shape_run / "frozen" / name), gaps,
                  _read_frozen(args.rejoin_run / "frozen" / name))
        tokens = {}
        captured = [_capture(reports, gap, history, original, tokens)
                    for reports, gap, original in zip(*inputs, strict=True)]
        with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
            pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print("All cache evidence and candidate actions frozen. Joining anonymous evaluation.", flush=True)
    previous = json.loads((args.shape_run / "summary.json").read_text(encoding="utf-8"))
    original = json.loads((args.rejoin_run / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for alias, scene in previous["scenes"].items():
        captured = _read_frozen(frozen / f"{alias}.pkl.gz")
        for old, data in zip(scene["rows"], captured, strict=True):
            eligible = old["end"] == "kf_rejoin" and old["full_rejoin"] and old["boundary_reports"] == 1
            if data is None or not eligible:
                data = dict(features=None, decisions=dict.fromkeys(ARMS))
            elif (data["start"], data["stop"]) != (old["start"], old["stop"]):
                raise ValueError("the cached decision does not match the original window")
            row = {key: value for key, value in old.items() if key != "reports"}
            rows.append(dict(scene=alias, component=scene["component"], sparse=scene["sparse"],
                             row=row, features=data["features"], decisions=data["decisions"]))
    if (len(rows), sum(entry["decisions"]["original"] is not None for entry in rows)) != (482, 416):
        raise ValueError("the original rejoin population changed")
    comparisons = summarize(rows)
    for arm, old_arm in (("current", "two_block_geometry"), ("cache", "cached_shape_geometry"),
                         ("original", "original"), ("geometry", "geometry"), ("geometry_masked", "geometry_masked")):
        if comparisons["primary"]["full"][arm] != original["primary"]["full"][old_arm]:
            raise ValueError("an original C-ON46 arm changed")
    _write_json(args.output_dir / "rows.json", rows)
    _write_json(args.output_dir / "summary.json", dict(clock=previous["clock"], frozen_windows=len(rows),
                candidate_graph_unchanged=True, publications_unchanged=True, external_dataset_reads=0,
                new_measurement_reads=0, training_updates=0, wall_seconds=perf_counter()-began,
                **comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
