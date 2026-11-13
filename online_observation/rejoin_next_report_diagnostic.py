"""Compare one-update counterfactuals on later reports in the frozen MTAD replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import pickle
from statistics import fmean
from time import perf_counter

from .continuous_tracker import Plane
from .hypothesis_gap_diagnostic import Switch, _direction, _load_frozen
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import _other
from .rejoin_batch_diagnostic import _read_frozen, _report_at
from .rejoin_next_report import score_next_report
from .rejoin_next_report_summary import summarize


SIDES = ("release", "retain")
PATHS = ("assimilate", "skip")
ACTIVE_ARMS = ("next_assimilate", "next_skip", "next_witness_mask", "next_robust")
ARMS = (*ACTIVE_ARMS, "next_abstain")


def _unanimous(votes):
    first, *others = votes
    return first if all(vote == first for vote in others) else 0


def _branch(opposite):
    if any(report["status"] != "supported" for report in opposite.values()):
        return dict(supported=False, independent=False, scores=None, directions=None)
    scores = {arm: {field: fmean(report["scores"][arm][field] for report in opposite.values())
                    for field in ("actual_m2", "masked_m2", "increment_m2")} for arm in PATHS}
    use, skip = (scores[arm] for arm in PATHS)
    votes = dict(next_assimilate=_direction(use["actual_m2"]), next_skip=_direction(skip["actual_m2"]),
                 next_witness_mask=_direction(use["masked_m2"]),
                 next_robust=_unanimous([_direction(scores[arm][field]) for arm in PATHS
                                         for field in ("actual_m2", "increment_m2")]))
    return dict(supported=True, independent=not any(r["in_boundary_shape"] for r in opposite.values()),
                scores=scores, directions=votes)


def _readout(switch, histories, selections, cache, boundary, onset, stop, fit, old):
    floor = max(stamp for _, stamp in selections.values())

    def score(focal, witness, after=None):
        result = score_next_report(switch, histories, focal, witness, cache[witness], stop, fit, after=after)
        return result | {"in_boundary_shape": result["witness_stamp"] in boundary.get(witness, ())}

    focals, branches = {}, {}
    for side, (key, stamp) in selections.items():
        earlier = old["lineage"]["roles"][side]["novel_post_onset"]
        actual = tuple(t for t, _ in histories[key] if onset < t < stop and t not in cache[key])
        if actual != earlier or not actual or actual[-1] != stamp:
            raise ValueError("the complete C51 cache-excluded selection changed")
        same = score((key, stamp), key)
        if same["witness_stamp"] is not None:
            raise ValueError("a latest cache-excluded report unexpectedly has a cache-excluded successor")
        first = score((key, actual[0]), key) if len(actual) >= 2 else None
        if first is not None and first["witness_stamp"] != actual[1]:
            raise ValueError("the earlier-probe witness is not the second cache-excluded event")
        partners = {role: _other(next(edge for edge in getattr(switch, role) if key in edge), key) for role in SIDES}
        opposite = {role: score((key, stamp), partner, floor) for role, partner in partners.items()}
        focals[side] = dict(same=same, earlier_same=first, opposite=opposite, focal_stamp=stamp,
                            focal_in_boundary_shape=stamp in boundary.get(key, ()))
        branches[side] = _branch(opposite)

    independent = [row["directions"] for row in branches.values() if row["independent"]]
    directions = {arm: _unanimous([row[arm] for row in independent]) if independent else 0 for arm in ACTIVE_ARMS}
    return dict(focals=focals, branches=branches, directions=directions | {"next_abstain": 0})


def _capture(events, original, gaps, anchors, shapes, async_rows, history_rows):
    requests, keys = defaultdict(list), set()
    captured = [None] * len(gaps)
    for index, (gap, anchor, shape, old, history) in enumerate(zip(gaps, anchors, shapes, async_rows, history_rows,
                                                                 strict=True)):
        if history is None:
            continue
        stop = gap["stop"]
        samples = [sample for sample, _ in anchor if sample["stamp"] == stop]
        previous, boundary = _report_at(shape, stop)
        if (len(samples) != 1 or previous is None or boundary is None or old is None or old["stamp"] != stop
                or history["stamp"] != stop or gap["end"] != "kf_rejoin"):
            raise ValueError("a C53 window lost its original input")
        sample, = samples
        switch = Switch(**gap["switch"])
        cache = previous["shape"]["provenance"]
        boundary_cache = boundary["shape"]["provenance"] or {}
        if (set(sample["keys"]) != switch.keys or cache is None or set(cache) != switch.keys
                or (boundary_cache and set(boundary_cache) != switch.keys)
                or any(t > previous["stamp"] for stamps in cache.values() for t in stamps)
                or any(t > stop for stamps in boundary_cache.values() for t in stamps)):
            raise ValueError("the fixed candidate or shape provenance changed")
        selections = {side: (sample["partners"][side], old["innovation"]["reports"][side]["stamp"])
                      for side in SIDES}
        for side, (key, stamp) in selections.items():
            if (not gap["start"] < stamp < stop or stamp != history["history"]["reports"][side]["stamp"]
                    or tuple(sorted((sample["reporter"], key))) not in getattr(switch, side)):
                raise ValueError("the C53 selected report or fixed edge changed")
        keys.update(switch.keys)
        requests[stop].append((index, switch, selections, cache, boundary_cache, gap["start"], old))

    histories = defaultdict(list)
    plane = Plane.from_first_batch(events[0][1].values())
    for (stamp, batch), native in zip(events, original.native_reports, strict=True):
        if (stamp, tuple(sorted(batch))) != native:
            raise ValueError("the original native report clock or track keys changed")
        for key in keys.intersection(batch):
            histories[key].append((stamp, plane.project(batch[key])))
        # The current batch is visible for the exact-stop coverage count only.
        for index, switch, selections, cache, boundary, onset, old in requests.pop(stamp, ()):
            result = _readout(switch, histories, selections, cache, boundary, onset, stamp, original.fit, old)
            captured[index] = dict(stamp=stamp, next_report=result)
    if requests:
        raise ValueError("a C53 decision batch was missing from the original stream")
    return captured


def _join(history_run, frozen, num_scenes):
    by_scene = defaultdict(list)
    for row in json.loads((history_run / "rows.json").read_text(encoding="utf-8")):
        by_scene[row["scene"]].append(row)
    rows = []
    for index in range(1, num_scenes + 1):
        alias = f"scene_{index:02d}"
        captures = _read_frozen(frozen / f"{alias}.pkl.gz")
        for old, capture in zip(by_scene.pop(alias, ()), captures, strict=True):
            result = None
            if old["history"] is not None:
                if capture is None or capture["stamp"] != old["row"]["stop"]:
                    raise ValueError("the original eligible rejoin or decision clock changed")
                result = capture["next_report"]
            votes = result["directions"] if result is not None else dict.fromkeys(ARMS, 0)
            decisions = old["decisions"] | {name: vote > 0 if vote else old["decisions"]["current"]
                                             for name, vote in votes.items()}
            rows.append(old | dict(next_report=result, decisions=decisions))
    if by_scene or (len(rows), sum(row["innovation"] is not None for row in rows),
                    sum(row["next_report"] is not None for row in rows)) != (482, 416, 31):
        raise ValueError("the original MTAD evaluation population changed")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    for name in ("manifest", "gap-run", "anchor-run", "shape-run", "async-run", "history-run", "output-dir"):
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
                            _read_frozen(args.async_run / "frozen" / name),
                            _read_frozen(args.history_run / "frozen" / name))
        with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
            pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print("All next-report evidence and actions frozen. Joining anonymous evaluation.", flush=True)
    rows = _join(args.history_run, frozen, len(scenes))
    summary = summarize(rows)
    _write_json(args.output_dir / "rows.json", rows)
    _write_json(args.output_dir / "summary.json", dict(clock="ideal_native_measurement_time", frozen_scenes=len(scenes),
                frozen_windows=len(rows), legal_rejoins=416, publications_unchanged=True, candidate_graph_unchanged=True,
                training_updates=0, external_dataset_reads=0, wall_seconds=perf_counter()-started, **summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
