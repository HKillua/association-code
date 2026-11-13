"""Use the original MTAD trigger to compare strictly earlier forecast paths."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_tracker import Plane
from .hypothesis_gap_diagnostic import Switch, _direction, _load_frozen, _update_priors
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import _registration_shifts
from .registration_probe import FIT_STOP_S
from .rejoin_batch_diagnostic import _read_frozen, _report_at
from .rejoin_boundary_provenance import audit_boundary
from .rejoin_boundary_report import score_boundary
from .rejoin_boundary_summary import summarize
from .relative_motion_gap import _same


SIDES = ("release", "retain")
PATHS = ("assimilate", "skip_release", "skip_retain", "skip_both")
ARMS = (*(f"boundary_{path}" for path in PATHS), "boundary_mask_assimilate", "boundary_mask_skip_both",
        "boundary_confirm_assimilate", "boundary_confirm_skip_both", "boundary_robust",
        "boundary_shape_only_mask", "boundary_full_current_mask", "boundary_abstain")


def _agree(values):
    first, *rest = values
    return first if all(value == first for value in rest) else 0


def _confirmation(path):
    return _agree([_direction(path[key]) for key in ("actual_m2", "increment_m2")]
                  + [_direction(path["local"]["actual_m2"])])


def _directions(predictions, provenance):
    scores = predictions["scores"]
    votes = dict.fromkeys(ARMS, 0)
    if predictions["status"] == "supported":
        for name, score in scores.items():
            votes[f"boundary_{name}"] = _direction(score["actual_m2"])
        for name in ("assimilate", "skip_both"):
            votes[f"boundary_mask_{name}"] = _direction(scores[name]["masked_m2"])
            votes[f"boundary_confirm_{name}"] = _confirmation(scores[name])
        votes["boundary_robust"] = _agree([_confirmation(scores[arm]) for arm in PATHS])
    for arm, field in (("boundary_shape_only_mask", "shape_only_mask_action"),
                       ("boundary_full_current_mask", "full_coordinate_mask_action")):
        action = provenance["shape"][field]
        votes[arm] = (1 if action else -1) if action is not None else 0
    return votes


def _readout(switch, histories, selections, sample, fit, priors, points, frame, previous, focal, shape):
    stop, trigger = sample["stamp"], sample["reporter"]
    predictions = score_boundary(switch, histories, selections, trigger, stop, fit)
    shifts = _registration_shifts(fit) if stop > FIT_STOP_S else {}
    provenance = audit_boundary(switch, trigger, stop, priors, points, shifts, frame, previous, focal, shape)
    saved = provenance["shape"]
    factual = predictions["scores"]["assimilate"]
    if (factual["status"] != "supported" or not _same(factual["actual_m2"], saved["geometry_actual_m2"])
            or not _same(factual["masked_m2"], saved["geometry_masked_m2"])):
        raise ValueError("the assimilated strict-prior geometry did not reproduce C45")
    for path in predictions["scores"].values():
        if any(time is not None and time >= stop for time in path["prior_stamps"].values()):
            raise ValueError("a forecast used a current or later report")
        if path["status"] == "supported" and not _same(path["increment_m2"], path["local"]["increment_m2"]):
            raise ValueError("masking the trigger changed a nonincident edge")
    for side in SIDES:
        edge = provenance["graph"]["fixed_edges"][side]
        if (not _same(factual["edges"][side]["trigger_m2"], edge["kf_actual_m2"])
                or not _same(factual["edges"][side]["trigger_masked_m2"], edge["kf_masked_m2"])):
            raise ValueError("the original incident candidate and causal report scores disagree")
    return dict(predictions=predictions, provenance=provenance, common_support=predictions["status"] == "supported",
                directions=_directions(predictions, provenance))


def _capture(events, graph, original, gaps, anchors, shapes, async_rows, next_rows):
    requests, keys = defaultdict(list), set()
    captured = [None] * len(gaps)
    for index, (gap, anchor, reports, earlier, next_report) in enumerate(zip(gaps, anchors, shapes, async_rows,
                                                                           next_rows, strict=True)):
        if next_report is None:
            continue
        stop = gap["stop"]
        samples = [sample for sample, _ in anchor if sample["stamp"] == stop]
        _, shape = _report_at(reports, stop)
        if (len(samples) != 1 or shape is None or earlier is None or earlier["stamp"] != stop
                or next_report["stamp"] != stop or gap["end"] != "kf_rejoin"):
            raise ValueError("a C54 trigger lost its fixed inputs")
        sample, = samples
        switch = Switch(**gap["switch"])
        if set(sample["keys"]) != switch.keys:
            raise ValueError("the original two switch matchings changed")
        selections = {side: (sample["partners"][side], earlier["innovation"]["reports"][side]["stamp"])
                      for side in SIDES}
        for side, (key, time) in selections.items():
            previous = next_report["next_report"]["focals"][side]
            if (not gap["start"] < time < stop or time != previous["focal_stamp"]
                    or tuple(sorted((sample["reporter"], key))) not in getattr(switch, side)):
                raise ValueError("an original focal report, time or edge changed")
        keys.update(switch.keys)
        requests[stop].append((index, switch, selections, sample, gap["focal"], shape))
    if not requests:
        return captured

    plane = Plane.from_first_batch(events[0][1].values())
    priors, histories = {}, defaultdict(list)
    previous_frame = None
    for (stamp, batch), native in zip(events, original.native_reports, strict=True):
        frame = pickle.load(graph)
        if (stamp, tuple(sorted(batch))) != native or frame[:2] != native:
            raise ValueError("raw, frozen graph and original native report clocks or keys changed")
        points = {key: plane.project(value) for key, value in batch.items()}
        for key in keys.intersection(points):
            histories[key].append((stamp, points[key]))
        for index, switch, selections, sample, focal, shape in requests.pop(stamp, ()):
            if previous_frame is None:
                raise ValueError("a rejoin must have an earlier graph frame")
            result = _readout(switch, histories, selections, sample, original.fit, priors, points, frame,
                              previous_frame, focal, shape)
            captured[index] = dict(stamp=stamp, boundary_report=result)
        _update_priors(priors, stamp, points)
        previous_frame = frame
    terminal = pickle.load(graph)
    if (requests or terminal.native_reports != original.native_reports or terminal.fit != original.fit
            or terminal.runs["kf"] != original.runs["kf"] or graph.read(1)):
        raise ValueError("the original graph trace did not reproduce the fixed replay")
    return captured


def _join(next_run, frozen, num_scenes):
    by_scene = defaultdict(list)
    for row in json.loads((next_run / "rows.json").read_text(encoding="utf-8")):
        by_scene[row["scene"]].append(row)
    rows = []
    for index in range(1, num_scenes + 1):
        alias = f"scene_{index:02d}"
        captures = _read_frozen(frozen / f"{alias}.pkl.gz")
        for old, captured in zip(by_scene.pop(alias, ()), captures, strict=True):
            result = None
            current = old["decisions"]["current"]
            if old["next_report"] is not None:
                if captured is None or captured["stamp"] != old["row"]["stop"]:
                    raise ValueError("a C54 original eligible rejoin or decision clock changed")
                result = captured["boundary_report"]
                graph, shape = (result["provenance"][view] for view in ("graph", "shape"))
                if (not graph["retain_selected"] or shape["current_action"] != current
                        or (_direction(shape["geometry_masked_m2"]) > 0) != old["decisions"]["geometry_masked"]):
                    raise ValueError("an original graph selection, shape action or geometry mask changed")
            directions = result["directions"] if result is not None else dict.fromkeys(ARMS, 0)
            decisions = old["decisions"] | {arm: vote > 0 if vote else current for arm, vote in directions.items()}
            rows.append(old | dict(boundary_report=result, decisions=decisions))
    if by_scene or (len(rows), sum(row["innovation"] is not None for row in rows),
                    sum(row["boundary_report"] is not None for row in rows)) != (482, 416, 31):
        raise ValueError("the original MTAD evaluation population changed")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    for name in ("manifest", "graph-run", "gap-run", "anchor-run", "shape-run", "async-run", "next-run", "output-dir"):
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
        with gzip.open(args.graph_run / "private_frozen" / name, "rb") as graph:
            captures = _capture(_events(args.archive, args.manifest, scene), graph, original, gaps,
                                _read_frozen(args.anchor_run / "private_frozen" / name),
                                _read_frozen(args.shape_run / "frozen" / name),
                                _read_frozen(args.async_run / "frozen" / name),
                                _read_frozen(args.next_run / "frozen" / name))
        with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
            pickle.dump(captures, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print("All boundary evidence and actions frozen. Joining anonymous evaluation.", flush=True)
    rows = _join(args.next_run, frozen, len(scenes))
    summary = summarize(rows)
    _write_json(args.output_dir / "rows.json", rows)
    _write_json(args.output_dir / "summary.json", dict(clock="ideal_native_measurement_time", frozen_scenes=len(scenes),
                frozen_windows=len(rows), legal_rejoins=416, publications_unchanged=True, candidate_graph_unchanged=True,
                training_updates=0, external_dataset_reads=0, wall_seconds=perf_counter()-started, **summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
