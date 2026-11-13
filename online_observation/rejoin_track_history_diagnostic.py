"""Audit consecutive native track reports before the frozen MTAD rejoin."""
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
from .interval_assignment import _dot
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .rejoin_batch_diagnostic import _read_frozen
from .rejoin_common_innovation_diagnostic import _confirmed
from .rejoin_track_history import score_track_history
from .rejoin_track_history_summary import summarize
from .relative_motion_gap import _same, _sub


SIDES = ("release", "retain")
CORRECTED = ("own_corrected", "opponents_corrected", "both_corrected")
NEW_ARMS = ("history_confirmed", "history_support_only", "history_masked",
            *(f"history_{name}" for name in CORRECTED))


def _mask_one(common, masked_side):
    vectors = {side: (0., 0.) if side == masked_side else common["reports"][side]["innovation_m"]
               for side in SIDES}
    mean = tuple(fmean(axis) for axis in zip(*vectors.values(), strict=True))
    masks = common["raw"]["reports"]
    return {name: fmean(masks[side]["masked_m2"] + 2*_dot(common["reports"][side]["axis_m"],
                                                        _sub(vectors[side], center)) for side in SIDES)
            for name, center in (("raw", (0., 0.)), ("demeaned", mean))}


def _differential(history, common):
    parts = {side: history["reports"][side]["decomposition"] for side in SIDES}
    if any(part is None for part in parts.values()):
        return None
    axis = common["decomposition"]["differential_axis_m"]
    persistent, remainder = ({side: sign*_dot(parts[side][field], axis)
                              for side, sign in zip(SIDES, (1, -1), strict=True)}
                             for field in ("persistent_m", "remainder_m"))
    total = common["decomposition"]["differential_increment_m2"]
    if not _same(sum(persistent.values()) + sum(remainder.values()), total):
        raise ValueError("the historical and remainder terms did not reconstruct the C52 differential")
    return dict(persistent_increment_m2=sum(persistent.values()), remainder_increment_m2=sum(remainder.values()),
                total_increment_m2=total, persistent_by_report_m2=persistent, remainder_by_report_m2=remainder)


def _temporal(history, common, direction):
    previous = {side: history["reports"][side]["preceding"] for side in SIDES}
    points = [point for pair in previous.values() for point in pair]
    supported = len(points) == 4 and all(p is not None and p["status"] == "supported" for p in points)
    after_onset = {side: bool(pair and pair[-1] is not None and pair[-1]["after_onset"])
                   for side, pair in previous.items()}
    eligible = supported and all(after_onset.values())

    def signs(field):
        return {side: [_direction(p[field]) if p is not None and p["status"] == "supported" else None for p in pair]
                for side, pair in previous.items()}

    actual, increment, masked = (signs(f"{field}_m2") for field in ("actual", "increment", "masked"))
    masks = {side: _mask_one(common, side) for side in SIDES}
    loo = {side: row["demeaned"] for side, row in masks.items()}
    raw_loo = {side: row["raw"] for side, row in masks.items()}
    loo_directions = {side: _direction(value) for side, value in loo.items()}
    mask_vote = _direction(common["demeaned"]["masked_m2"])
    driving = [side for side, sign in loo_directions.items() if direction and sign != direction]
    isolated = []
    for side, pair in previous.items():
        if not all(p is not None and p["status"] == "supported" for p in pair):
            continue
        current = common["raw"]["reports"][side]["increment_m2"]
        preceding = [p["increment_m2"] for p in pair]
        if (_direction(current) and all(_direction(value) == -_direction(current) for value in preceding)
                and abs(current) > max(map(abs, preceding))):
            isolated.append(side)

    if not supported:
        reason = "missing"
    elif not eligible:
        reason = "older_pre_onset"
    elif not direction:
        reason = "no_demeaned_signal"
    elif any(value != direction for scores in (actual, increment) for pair in scores.values() for value in pair):
        reason = "prior_disagrees"
    elif driving:
        reason = "one_report_sensitive"
    else:
        reason = "confirmed"

    return dict(status="supported" if supported else "missing", reason=reason, eligible=eligible,
                actual_directions=actual, increment_directions=increment, masked_directions=masked,
                after_onset_count=sum(bool(p and p["after_onset"]) for p in points), latest_after_onset=after_onset,
                loo_actual_directions=loo_directions, loo_actual_m2=loo,
                raw_loo_actual_directions={side: _direction(value) for side, value in raw_loo.items()},
                raw_loo_actual_m2=raw_loo, loo_driving_sides=driving, isolated_innovation_sides=isolated,
                masked_vote=mask_vote if eligible else 0)


def _readout(switch, histories, novel, onset, fit, earlier, common_capture):
    history = score_track_history(switch, histories, novel, onset, fit)
    common = common_capture["common"]
    for side, report in history["reports"].items():
        old = earlier["innovation"]["reports"][side]["rolling"]
        current = report["rolling"]
        if (current["status"] != "supported" or current["prior_stamps"] != old["prior_stamps"]
                or any(not _same(current[key], value) for key, value in common["raw"]["reports"][side].items())):
            raise ValueError("a frozen C51/C52 report geometry, mask or clock changed")
        for p in report["preceding"]:
            if p is not None and (not p["stamp"] < report["stamp"] or any(
                    time is not None and time >= p["stamp"] for time in p["prior_stamps"].values())):
                raise ValueError("a preceding report reused its own batch or a future report")

    direction = common_capture["confirmed"]["demeaned"]
    temporal = _temporal(history, common, direction)
    history.update(temporal=temporal, differential=_differential(history, common))
    actions = {f"history_{arm}": _confirmed({**history[arm], "reports": {
        side: report[arm] for side, report in history["reports"].items()}}) for arm in CORRECTED}
    actions.update(history_confirmed=direction if temporal["reason"] == "confirmed" else 0,
                   history_support_only=direction if temporal["eligible"] else 0,
                   history_masked=temporal["masked_vote"])
    return history, actions


def _capture(events, original, gaps, anchors, previous, common):
    requests, keys = defaultdict(list), set()
    captured = [None] * len(gaps)
    for index, (gap, anchor, earlier, common_capture) in enumerate(zip(gaps, anchors, previous, common, strict=True)):
        if common_capture is None:
            continue
        stamp = gap["stop"]
        samples = [sample for sample, _ in anchor if sample["stamp"] == stamp]
        if (len(samples) != 1 or gap["end"] != "kf_rejoin" or earlier is None
                or earlier["stamp"] != stamp or common_capture["stamp"] != stamp):
            raise ValueError("a frozen rejoin lost its original inputs")
        switch = Switch(**gap["switch"])
        sample, = samples
        if set(sample["keys"]) != switch.keys or tuple(sorted(switch.keys)) != common_capture["group"]:
            raise ValueError("the frozen candidate group changed")
        novel = {side: (sample["partners"][side], earlier["innovation"]["reports"][side]["stamp"]) for side in SIDES}
        if any(not gap["start"] < time < stamp or time != earlier["lineage"]["roles"][side]["novel_post_onset"][-1]
               for side, (_, time) in novel.items()):
            raise ValueError("a selected report lost its original cache exclusion or timing")
        keys.update(switch.keys)
        requests[stamp].append((index, switch, novel, gap["start"], earlier, common_capture))

    histories = defaultdict(list)
    plane = Plane.from_first_batch(events[0][1].values())
    for (stamp, batch), native in zip(events, original.native_reports, strict=True):
        if (stamp, tuple(sorted(batch))) != native:
            raise ValueError("the original native report clock or track keys changed")
        for index, switch, novel, onset, earlier, common_capture in requests.pop(stamp, ()):
            history, actions = _readout(switch, histories, novel, onset, original.fit, earlier, common_capture)
            captured[index] = dict(stamp=stamp, history=history, history_decisions=actions)
        for key in keys.intersection(batch):
            histories[key].append((stamp, plane.project(batch[key])))
    if requests:
        raise ValueError("a frozen rejoin was absent from the native report stream")
    return captured


def _join(common_run, frozen, num_scenes):
    by_scene = defaultdict(list)
    for row in json.loads((common_run / "rows.json").read_text(encoding="utf-8")):
        by_scene[row["scene"]].append(row)
    rows = []
    for index in range(1, num_scenes + 1):
        alias = f"scene_{index:02d}"
        captured = _read_frozen(frozen / f"{alias}.pkl.gz")
        for old, evidence in zip(by_scene.pop(alias, ()), captured, strict=True):
            history, actions = None, dict.fromkeys(NEW_ARMS, 0)
            if old["common"] is not None:
                if evidence is None or evidence["stamp"] != old["row"]["stop"]:
                    raise ValueError("the original eligible rejoin or decision clock changed")
                history, actions = evidence["history"], evidence["history_decisions"]
            decisions = old["decisions"] | {name: vote > 0 if vote else old["decisions"]["current"]
                                             for name, vote in actions.items()}
            rows.append(old | dict(history=history, history_decisions=actions, decisions=decisions))
    if by_scene or (len(rows), sum(r["innovation"] is not None for r in rows),
                    sum(r["history"] is not None for r in rows)) != (482, 416, 31):
        raise ValueError("the original MTAD evaluation population changed")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    for name in ("manifest", "gap-run", "anchor-run", "async-run", "common-run", "output-dir"):
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
                            _read_frozen(args.async_run / "frozen" / name),
                            _read_frozen(args.common_run / "frozen" / name))
        with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
            pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print("All track-history evidence and actions frozen. Joining anonymous evaluation.", flush=True)
    rows = _join(args.common_run, frozen, len(scenes))
    summary = summarize(rows)
    _write_json(args.output_dir / "rows.json", rows)
    _write_json(args.output_dir / "summary.json", dict(clock="ideal_native_measurement_time", frozen_scenes=len(scenes),
                frozen_windows=len(rows), legal_rejoins=416, publications_unchanged=True, candidate_graph_unchanged=True,
                training_updates=0, external_dataset_reads=0, wall_seconds=perf_counter()-started, **summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
