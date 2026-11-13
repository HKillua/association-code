"""Audit cache-disjoint fresh displacement at the frozen MTAD rejoin batches."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_tracker import Plane
from .hypothesis_gap_diagnostic import Switch, _direction, _load_frozen
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .rejoin_batch_diagnostic import _read_frozen, _report_at
from .rejoin_fresh_displacement import score_fresh_displacement
from .rejoin_fresh_displacement_squeeze import summarize
from .relative_motion_gap import _same


def _attribution(fresh):
    observed = fresh["decached"]
    masked = fresh["post_onset_masked"]
    if not observed["qualified"]:
        return dict(result="no_new", release=False, retain=False, mismatch_change_m2=None)
    if masked["status"] != "supported":
        return dict(result="only_support", release=False, retain=False, mismatch_change_m2=None)
    changes = {side: observed["mismatch_m2"][side] - masked["mismatch_m2"][side]
               for side in ("release", "retain")}
    direction = _direction(observed["margin_m2"])
    release, retain = changes["release"], changes["retain"]
    return dict(result="paired", mismatch_change_m2=changes,
                release=direction > 0 and _direction(retain) > 0 and _direction(retain - release) > 0,
                retain=direction < 0 and _direction(release) > 0 and _direction(release - retain) > 0)


def _actions(current, fresh, attribution):
    def override(arm, valid):
        direction = _direction(arm["margin_m2"]) if valid else 0
        return direction > 0 if direction else current

    original, decached = fresh["unfiltered"], fresh["decached"]
    return dict(unfiltered_rank=override(original, original["status"] == "supported"),
                fresh_rank=override(decached, decached["qualified"]),
                fresh_attributed=True if attribution["release"] else False if attribution["retain"] else current,
                fresh_guard=False if current is True and attribution["retain"] else current)


def _readout(sample, histories, previous, onset, base):
    shape = previous["shape"] if previous is not None else None
    provenance = shape["provenance"] if shape is not None and shape["actual"] is not None else None
    if provenance is not None and any(t > previous["stamp"] for times in provenance.values() for t in times):
        raise ValueError("old shape support follows its own report")
    fresh = score_fresh_displacement(sample, histories, provenance, onset)
    if fresh["cache_verified"]:
        cache_status = "verified"
    else:
        cache_status = "no_scored_cache" if shape is None or shape["actual"] is None else "missing_provenance"

    original = fresh["unfiltered"]
    old = base["features"]["witness"]
    for role, age in original["ages_s"].items():
        previous_age = old["ages_s"][role]
        if age != previous_age and (age is None or previous_age is None or not _same(age, previous_age)):
            raise ValueError("the original strict-prior native ages changed")
    if old["fresh"] and old["interval"] is not None:
        if (original["status"] != "supported"
                or not _same(original["margin_m2"], old["interval"]["displacement_margin_m2"])):
            raise ValueError("unfiltered displacement disagrees with the frozen witness")
    decached = fresh["decached"]
    if decached["brackets"] is not None:
        for side, key in sample["partners"].items():
            for pair in decached["brackets"][side]:
                if set(pair) & set(provenance[key]):
                    raise ValueError("a decached bracket contains an old cache report")
    if any(fresh[name]["available_at"] is not None and fresh[name]["available_at"] >= sample["stamp"]
           for name in ("unfiltered", "decached", "post_onset_masked")):
        raise ValueError("a displacement uses the decision batch or a future report")
    attribution = _attribution(fresh)
    decisions = base["decisions"] | _actions(base["decisions"]["current"], fresh, attribution)
    return dict(stamp=sample["stamp"], fresh=fresh, cache_status=cache_status,
                attribution=attribution, decisions=decisions)


def _capture(events, original, gaps, anchors, shapes, old):
    requests, keys = defaultdict(list), set()
    captured = [None] * len(gaps)
    for index, (gap, anchor, shape, base) in enumerate(zip(gaps, anchors, shapes, old, strict=True)):
        if base is None:
            continue
        stamp = gap["stop"]
        previous, boundary = _report_at(shape, stamp)
        samples = [sample for sample, _ in anchor if sample["stamp"] == stamp]
        if len(samples) != 1 or boundary is None or stamp != base["stamp"] or gap["end"] != "kf_rejoin":
            raise ValueError("the frozen single-report rejoin changed")
        sample, = samples
        switch = Switch(**gap["switch"])
        if set(sample["keys"]) != switch.keys or any(
                tuple(sorted((sample["reporter"], partner))) not in getattr(switch, side)
                for side, partner in sample["partners"].items()):
            raise ValueError("the frozen candidate graph changed")
        keys.update((sample["reporter"], *sample["partners"].values()))
        requests[stamp].append((index, sample, previous, gap["start"], base))

    histories = defaultdict(list)
    plane = Plane.from_first_batch(events[0][1].values())
    for (stamp, batch), native in zip(events, original.native_reports, strict=True):
        if (stamp, tuple(sorted(batch))) != native:
            raise ValueError("the original report clock or track keys changed")
        for index, sample, previous, onset, base in requests.pop(stamp, ()):
            if sample["reporter"] not in batch:
                raise ValueError("the decision batch has no original triggering report")
            captured[index] = _readout(sample, histories, previous, onset, base)
        for key, point in batch.items():
            if key in keys:
                histories[key].append((stamp, plane.project(point)))
    if requests:
        raise ValueError("some rejoin windows did not appear in the native stream")
    return captured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    for name in ("manifest", "gap-run", "anchor-run", "shape-run", "witness-run", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    scenes = _roster(args.manifest)["splits"]["development"]
    if len(scenes) != 28 or scenes != read_roster(args.manifest)["splits"]["held"]:
        raise ValueError("the original MTAD scene roster changed")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    frozen = args.output_dir / "frozen"
    frozen.mkdir(mode=0o700)
    started = perf_counter()
    for index, scene in enumerate(scenes, 1):
        name = f"scene_{index:02d}.pkl.gz"
        original, gaps = _load_frozen(args.gap_run / "private_frozen" / name)
        captured = _capture(_events(args.archive, args.manifest, scene), original, gaps,
                            _read_frozen(args.anchor_run / "private_frozen" / name),
                            _read_frozen(args.shape_run / "frozen" / name),
                            _read_frozen(args.witness_run / "frozen" / name))
        with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
            pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)

    print("All displacement scores and decisions frozen. Joining anonymous evaluation.", flush=True)
    by_scene = defaultdict(list)
    for row in json.loads((args.witness_run / "rows.json").read_text(encoding="utf-8")):
        by_scene[row["scene"]].append(row)
    rows = []
    for index in range(1, len(scenes) + 1):
        alias = f"scene_{index:02d}"
        for previous, captured in zip(by_scene.pop(alias, ()), _read_frozen(frozen / f"{alias}.pkl.gz"), strict=True):
            if (captured is None) != (previous["features"] is None):
                raise ValueError("the original legal readout set changed")
            if captured is None:
                rows.append(previous | {"fresh": None})
                continue
            if (captured["stamp"] != previous["row"]["stop"]
                    or any(captured["decisions"][arm] != vote for arm, vote in previous["decisions"].items())):
                raise ValueError("an original rejoin control or decision time changed")
            rows.append(previous | {key: captured[key] for key in ("fresh", "attribution", "decisions", "cache_status")})
    if by_scene or (len(rows), sum(row["fresh"] is not None for row in rows)) != (482, 416):
        raise ValueError("the fixed population changed")
    comparisons = summarize(rows)
    _write_json(args.output_dir / "rows.json", rows)
    _write_json(args.output_dir / "summary.json", dict(clock="ideal_native_measurement_time", frozen_scenes=len(scenes),
                publications_unchanged=True, candidate_graph_unchanged=True,
                training_updates=0, external_dataset_reads=0, wall_seconds=perf_counter()-started, **comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
