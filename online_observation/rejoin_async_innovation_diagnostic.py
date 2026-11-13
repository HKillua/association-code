"""Audit native-time innovations and cache-induced gaps on fixed MTAD rejoins."""
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
from .rejoin_async_gap_lineage import trace_gap_lineage
from .rejoin_async_innovation import score_report_innovations
from .rejoin_async_innovation_squeeze import summarize
from .rejoin_batch_diagnostic import _read_frozen, _report_at


SIDES = ("release", "retain")
NEW_ARMS = ("native_masked", "native_actual", "native_increment", "native_confirmed")


def _confirmed(scores, arm):
    group = scores[arm]
    if group["support_count"] != 2:
        return 0
    direction = _direction(group["actual_m2"])
    if not direction or any(_direction(report[arm]["increment_m2"]) != direction
                            for report in scores["reports"].values()):
        return 0
    release, retain = (group[f"{side}_change_m2"] for side in SIDES)
    rejected = retain if direction > 0 else release
    return direction if _direction(rejected) > 0 and _direction(direction * (retain - release)) > 0 else 0


def _actions(current, scores):
    main = scores["rolling"]
    directions = [_direction(main[name]) if main["support_count"] == 2 else 0
                  for name in ("masked_m2", "actual_m2", "increment_m2")]
    directions.append(_confirmed(scores, "rolling"))
    return {arm: direction > 0 if direction else current for arm, direction in zip(NEW_ARMS, directions, strict=True)}


def _readout(sample, switch, histories, previous, onset, old, fit):
    shape = previous["shape"] if previous is not None else None
    provenance = shape["provenance"] if shape is not None and shape["actual"] is not None else None
    if provenance is not None and any(t > previous["stamp"] for times in provenance.values() for t in times):
        raise ValueError("old cache contains a report after its own time")
    lineage = trace_gap_lineage(sample, histories, provenance, onset)
    base = old["fresh"]
    if (lineage["left"], lineage["right"], lineage["cache_verified"], lineage["raw_status"],
        lineage["retained_status"]) != (base["left"], base["right"], base["cache_verified"],
                                       base["unfiltered"]["status"], base["decached"]["status"]):
        raise ValueError("the C50 fixed interval or eligibility changed")
    novel = dict.fromkeys(SIDES)
    for side in SIDES:
        role = lineage["roles"][side]
        counts = base["opportunities"][side]
        times = role["novel_post_onset"]
        if (role["old_cache_count"], len(role["raw_post_onset"]), None if times is None else len(times)) != (
                counts["removed_cache"], counts["post_onset"], counts["post_onset_outside_cache"]):
            raise ValueError("the C50 opportunity set changed")
        if times:
            novel[side] = sample["partners"][side], times[-1]
    scores = score_report_innovations(switch, histories, novel, onset, sample["stamp"], fit)
    for report in scores["reports"].values():
        if report is None:
            continue
        for arm in ("rolling", "onset_origin"):
            if any(t is not None and t >= report["stamp"] for t in report[arm]["prior_stamps"].values()):
                raise ValueError("a report reused its own batch in a prediction")
        if any(t is not None and t > onset for t in report["onset_origin"]["prior_stamps"].values()):
            raise ValueError("an onset prior uses reports after onset")
    decisions = old["decisions"] | _actions(old["decisions"]["current"], scores)
    scores["reports"] = {side: None if report is None else {k: v for k, v in report.items()
                                                           if k not in ("key", "partners")}
                         for side, report in scores["reports"].items()}
    return dict(stamp=sample["stamp"], lineage=lineage, innovation=scores, decisions=decisions,
                confirmation={arm: _confirmed(scores, arm) for arm in ("rolling", "onset_origin")})


def _capture(events, original, gaps, anchors, shapes, frozen):
    requests, keys = defaultdict(list), set()
    captured = [None] * len(gaps)
    for index, (gap, anchor, shapes_at_gap, old) in enumerate(zip(gaps, anchors, shapes, frozen, strict=True)):
        if old is None:
            continue
        stamp = gap["stop"]
        previous, boundary = _report_at(shapes_at_gap, stamp)
        samples = [sample for sample, _ in anchor if sample["stamp"] == stamp]
        if len(samples) != 1 or boundary is None or stamp != old["stamp"] or gap["end"] != "kf_rejoin":
            raise ValueError("a frozen single-report rejoin changed")
        sample, = samples
        switch = Switch(**gap["switch"])
        if set(sample["keys"]) != switch.keys or any(
                tuple(sorted((sample["reporter"], partner))) not in getattr(switch, side)
                for side, partner in sample["partners"].items()):
            raise ValueError("the frozen candidate graph changed")
        keys.update(switch.keys)
        requests[stamp].append((index, sample, switch, previous, gap["start"], old))
    histories = defaultdict(list)
    plane = Plane.from_first_batch(events[0][1].values())
    for (stamp, batch), native in zip(events, original.native_reports, strict=True):
        if (stamp, tuple(sorted(batch))) != native:
            raise ValueError("the original report clock or track keys changed")
        for index, sample, switch, previous, onset, old in requests.pop(stamp, ()):
            if sample["reporter"] not in batch:
                raise ValueError("the decision batch has no original triggering report")
            captured[index] = _readout(sample, switch, histories, previous, onset, old, original.fit)
        for key, point in batch.items():
            if key in keys:
                histories[key].append((stamp, plane.project(point)))
    if requests:
        raise ValueError("a fixed window did not appear in the native stream")
    return captured


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    for name in ("manifest", "gap-run", "anchor-run", "shape-run", "fresh-run", "output-dir"):
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
                            _read_frozen(args.fresh_run / "frozen" / name))
        with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
            pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)

    print("All gap and innovation evidence frozen. Joining anonymous evaluation.", flush=True)
    by_scene = defaultdict(list)
    for row in json.loads((args.fresh_run / "rows.json").read_text(encoding="utf-8")):
        by_scene[row["scene"]].append(row)
    rows = []
    for index in range(1, len(scenes) + 1):
        alias = f"scene_{index:02d}"
        for previous, captured in zip(by_scene.pop(alias, ()), _read_frozen(frozen / f"{alias}.pkl.gz"), strict=True):
            if previous["fresh"] is None:
                rows.append(previous | {"innovation": None})
                continue
            if (captured is None or captured["stamp"] != previous["row"]["stop"]
                    or any(captured["decisions"][arm] != vote for arm, vote in previous["decisions"].items())):
                raise ValueError("a prior control or decision time changed")
            rows.append(previous | {key: captured[key] for key in ("lineage", "innovation", "confirmation", "decisions")})
    if by_scene or (len(rows), sum(row["innovation"] is not None for row in rows)) != (482, 416):
        raise ValueError("the fixed population changed")
    summary = summarize(rows)
    _write_json(args.output_dir / "rows.json", rows)
    _write_json(args.output_dir / "summary.json", dict(clock="ideal_native_measurement_time", frozen_scenes=len(scenes),
                frozen_windows=len(rows), legal_rejoins=416, publications_unchanged=True, candidate_graph_unchanged=True,
                training_updates=0, external_dataset_reads=0, wall_seconds=perf_counter()-started, **summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
