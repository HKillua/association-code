"""Audit strictly prior opposite-source witnesses at original MTAD rejoins."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_tracker import FRESH_S, Plane
from .hypothesis_gap_diagnostic import Switch, _direction, _load_frozen
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .rejoin_batch_diagnostic import _read_frozen, _report_at
from .rejoin_witness import score_reverse_witness
from .rejoin_witness_squeeze import summarize
from .relative_motion_gap import _same


CONTROLS = ("original", "current", "cache", "union", "geometry", "geometry_masked", "raw_positive", "nominal_gaussian")
ARMS = (*CONTROLS, "witness_position", "witness_motion", "witness_exclusion")


def _opposite_support(provenance, keys):
    return {(key, stamp) for key in keys for stamp in provenance.get(key, ())}


def _witness(sample, histories, cache, current, start, features):
    stamp = sample["stamp"]
    readout = score_reverse_witness(sample, histories)
    ages = readout["ages_s"]
    if any(not _same(ages[role], features[f"{role}_age_s"]) for role in ("reporter", "release", "retain")):
        raise ValueError("the original last-report ages did not reproduce")
    endpoint, interval = readout["endpoint"], readout["interval"]
    fresh = all(0 < age <= FRESH_S for age in ages.values())
    both = fresh and endpoint is not None and endpoint["both_partners_after_query"]
    if interval is not None:
        provenance = interval["provenance"]
    elif endpoint is not None:
        provenance = endpoint["provenance"]
    else:
        provenance = {}
    if any(t >= stamp for times in provenance.values() for t in times):
        raise ValueError("an opposite witness used a current or future report")
    keys = sample["partners"].values()
    used = _opposite_support(provenance, keys)

    def overlap(report):
        if report is None or not report["shape"]["provenance"]:
            return None
        return len(used & _opposite_support(report["shape"]["provenance"], keys))

    position_direction = _direction(endpoint["margin_m2"]) if both else 0
    motion_direction = _direction(interval["displacement_margin_m2"]) if both and interval else 0
    exclusion = dict.fromkeys(("release", "retain"), False)
    if both and interval is not None:
        growth = interval["residual_change_m2"]
        for side, other, sign in (("release", "retain", 1), ("retain", "release", -1)):
            exclusion[side] = (position_direction == sign and _direction(growth[other]) > 0
                               and _direction(growth[other] - growth[side]) > 0)
    public = {key: value for key, value in readout.items() if key not in ("endpoint", "interval")}
    if endpoint is not None:
        endpoint = {key: value for key, value in endpoint.items() if key != "provenance"}
    if interval is not None:
        interval = {key: value for key, value in interval.items() if key != "provenance"}
        interval["earlier"] = {key: value for key, value in interval["earlier"].items() if key != "provenance"}
    return public | dict(endpoint=endpoint, interval=interval, fresh=fresh,
                         positive_side={1: "release", -1: "retain"}.get(position_direction),
                         motion_side={1: "release", -1: "retain"}.get(motion_direction),
                         release_exclusion=exclusion["release"], retain_exclusion=exclusion["retain"],
                         opposite_report_count=len(used), post_onset_opposite_reports=sum(t > start for _, t in used),
                         opposite_overlap_cache=overlap(cache), opposite_overlap_current=overlap(current)), provenance


def _capture(events, original, gaps, anchors, shapes, geometry):
    requests = defaultdict(list)
    keys = set()
    captured = [None] * len(gaps)
    for index, (gap, anchor, shape, base) in enumerate(zip(gaps, anchors, shapes, geometry, strict=True)):
        if base is None:
            continue
        stamp = gap["stop"]
        previous, boundary = _report_at(shape, stamp)
        samples = [sample for sample, _ in anchor if sample["stamp"] == stamp]
        if (len(samples) != 1 or boundary is None or stamp != base["stamp"]
                or gap["end"] != "kf_rejoin"):
            raise ValueError("the original single-report boundary did not reproduce")
        sample, = samples
        switch = Switch(**gap["switch"])
        if set(sample["keys"]) != switch.keys or any(
                tuple(sorted((sample["reporter"], partner))) not in getattr(switch, side)
                for side, partner in sample["partners"].items()):
            raise ValueError("an original four-end group changed")
        keys.update((sample["reporter"], *sample["partners"].values()))
        requests[stamp].append((index, sample, previous, boundary, base, gap["start"]))

    histories = defaultdict(list)
    plane = Plane.from_first_batch(events[0][1].values())
    for (stamp, batch), native in zip(events, original.native_reports, strict=True):
        if (stamp, tuple(sorted(batch))) != native:
            raise ValueError("the raw report clock or participant keys changed")
        for index, sample, cache, current, base, start in requests.pop(stamp, ()):
            if sample["reporter"] not in batch:
                raise ValueError("the original current report is missing")
            features = base["features"]
            witness, provenance = _witness(sample, histories, cache, current, start, features)
            decisions = {arm: base["decisions"][arm] for arm in CONTROLS}
            current_vote = decisions["current"]
            cache_only = current_vote is False and decisions["cache"] is True
            for arm, vote in (("witness_position", witness["positive_side"] == "release"),
                              ("witness_motion", witness["motion_side"] == "release"),
                              ("witness_exclusion", witness["release_exclusion"])):
                decisions[arm] = None if current_vote is None else current_vote or (cache_only and vote)
            captured[index] = dict(stamp=stamp, decisions=decisions, provenance=provenance,
                                   features=features | dict(cache_only=cache_only, witness=witness))
        for key, point in batch.items():
            if key in keys:
                histories[key].append((stamp, plane.project(point)))
    if requests:
        raise ValueError("some fixed windows are outside the original native clock")
    return captured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    for name in ("manifest", "gap-run", "anchor-run", "shape-run", "geometry-frozen", "geometry-run", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    scenes = _roster(args.manifest)["splits"]["development"]
    if len(scenes) != 28 or scenes != read_roster(args.manifest)["splits"]["held"]:
        raise ValueError("the original MTAD scene roster changed")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    frozen = args.output_dir / "frozen"
    frozen.mkdir(mode=0o700)
    began = perf_counter()
    for index, scene in enumerate(scenes, 1):
        name = f"scene_{index:02d}.pkl.gz"
        original, gaps = _load_frozen(args.gap_run / "private_frozen" / name)
        captured = _capture(_events(args.archive, args.manifest, scene), original, gaps,
                            _read_frozen(args.anchor_run / "private_frozen" / name),
                            _read_frozen(args.shape_run / "frozen" / name),
                            _read_frozen(args.geometry_frozen / name))
        with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
            pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print("All strict-prior witnesses and decisions frozen. Joining anonymous evaluation.", flush=True)
    old = json.loads((args.geometry_run / "rows.json").read_text(encoding="utf-8"))
    by_scene = defaultdict(list)
    for item in old:
        by_scene[item["scene"]].append(item)
    output = []
    for index in range(1, len(scenes)+1):
        alias = f"scene_{index:02d}"
        for before, item in zip(by_scene.pop(alias, ()), _read_frozen(frozen / f"{alias}.pkl.gz"), strict=True):
            if before["features"] is None:
                output.append(before | {"decisions": dict.fromkeys(ARMS)})
                continue
            if (item is None or item["stamp"] != before["row"]["stop"]
                    or any(before["decisions"][arm] != item["decisions"][arm] for arm in CONTROLS)):
                raise ValueError("an original control changed")
            output.append(before | {key: item[key] for key in ("features", "decisions")})
    if by_scene or (len(output), sum(x["features"] is not None for x in output)) != (482, 416):
        raise ValueError("the frozen rejoin cohort changed")
    comparisons = summarize(output)
    _write_json(args.output_dir / "rows.json", output)
    _write_json(args.output_dir / "summary.json", dict(
        clock="ideal_native_measurement_time", frozen_scenes=len(scenes), frozen_windows=len(output),
        eligible_rejoins=416, training_updates=0, external_dataset_reads=0,
        candidate_graph_unchanged=True, publications_unchanged=True,
        wall_seconds=perf_counter()-began, **comparisons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
