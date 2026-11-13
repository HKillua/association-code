"""Audit shared innovations and held-out witnesses on frozen MTAD rejoins."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from math import sqrt
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_tracker import Plane, RADIUS_M
from .evaluation_data import read_labels
from .hypothesis_gap_diagnostic import Switch, _direction, _load_frozen
from .interval_assignment import _dot
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import _add, _other, _registration_shifts
from .registration_probe import FIT_STOP_S
from .rejoin_batch_diagnostic import _read_frozen, _report_at
from .rejoin_common_innovation import compare_common_innovations
from .rejoin_common_innovation_squeeze import ACTION_ARMS, summarize
from .relative_motion_gap import _same, _sub
from .shared_innovation_gap import InnovationHistory


SIDES = ("release", "retain")
COMMON_ARMS = ("raw", "demeaned", "external", "equal_shrink", "shared_external")


def _witness(bank, key, stamp, origin, group, points):
    near = [row for row in bank.candidates(key[0], stamp, origin, group) if row[0] <= RADIUS_M**2]
    selected = near[:2]
    status = "too_few_neighbors"
    mean = None
    if len(selected) == 2:
        first, second = (row[2].vector for row in selected)
        status = "neighbors_disagree"
        if _dot(first, second) > 0:
            status = "supported"
            mean = tuple((a+b)/2 for a, b in zip(first, second, strict=True))
    return dict(status=status, near_count=len(near), same_batch_count=sum(k[0] == key[0] and k not in group
                                                                        for k in points),
                donors=[dict(key=k, stamp=r.stamp, prior_age_s=r.prior_age,
                             distance_m=sqrt(distance), vector_m=r.vector) for distance, k, r in selected]), mean


def _geometry(bank, switch, key, stamp, points, fit, prior):
    partners = {side: _other(next(edge for edge in getattr(switch, side) if key in edge), key) for side in SIDES}
    keys = dict(own=key, **partners)
    if any(bank.priors[k].stamp != prior[role] or prior[role] >= stamp for role, k in keys.items()):
        raise ValueError("a C51 strict-prior timestamp changed")
    predicted = {role: bank.priors[k].position(stamp, True) for role, k in keys.items()}
    shifts = _registration_shifts(fit) if stamp > FIT_STOP_S else {}
    shift = _sub(shifts.get(key[0], (0., 0.)), shifts.get(partners["release"][0], (0., 0.)))
    return dict(actual=_add(points[key], shift), masked=_add(predicted["own"], shift),
                release=predicted["release"], retain=predicted["retain"]), predicted["own"]


def _confirmed(arm):
    if arm["support_count"] != 2:
        return 0
    direction = _direction(arm["actual_m2"])
    if not direction or any(_direction(report["increment_m2"]) != direction for report in arm["reports"].values()):
        return 0
    release, retain = (arm[f"{side}_change_m2"] for side in SIDES)
    rejected = retain if direction > 0 else release
    return direction if _direction(rejected) > 0 and _direction(direction*(retain-release)) > 0 else 0


def _combine(parts, base, switch):
    geometry = {side: parts[side][0] for side in SIDES}
    witnesses = {side: parts[side][1] for side in SIDES}
    means = {side: parts[side][2] for side in SIDES}
    common = compare_common_innovations(geometry, means)
    for side in SIDES:
        previous = base["innovation"]["reports"][side]["rolling"]
        if any(not _same(value, previous[key]) for key, value in common["raw"]["reports"][side].items()):
            raise ValueError("the C51 raw or masked score changed")
    decomposition = common["decomposition"]
    if not _same(decomposition["common_increment_m2"] + decomposition["differential_increment_m2"],
                 common["raw"]["increment_m2"]):
        raise ValueError("the common and differential increments do not reconstruct the raw margin")
    confirmed = {arm: _confirmed(common[arm]) for arm in COMMON_ARMS}
    if confirmed["raw"] != base["confirmation"]["rolling"]:
        raise ValueError("the C51 action readout changed")

    supported_both = all(means[side] is not None for side in SIDES)
    first, second = (witnesses[side]["donors"] for side in SIDES)
    distinct_tracks = supported_both and not ({r["key"] for r in first} & {r["key"] for r in second})
    distinct_events = supported_both and not ({(r["key"], r["stamp"]) for r in first}
                                             & {(r["key"], r["stamp"]) for r in second})
    reduces_both = supported_both and all(_direction(row["external_raw_reduction_m2"]) > 0
                                          for row in common["reports"].values())
    qualified = distinct_tracks and distinct_events and reduces_both
    current = base["decisions"]["current"]

    def action(direction):
        return direction > 0 if direction else current

    decisions = base["decisions"] | dict(
        pair_demeaned=action(confirmed["demeaned"]), donor_confirmed=action(confirmed["external"]),
        matched_shrink=action(confirmed["equal_shrink"]) if qualified else current,
        donor_support_only=base["decisions"]["native_confirmed"] if qualified else current,
        donor_qualified=action(confirmed["external"]) if qualified else current,
        donor_shared=action(confirmed["shared_external"]), abstain=current)
    return dict(stamp=base["stamp"], common=common, witnesses=witnesses, confirmed=confirmed,
                qualification=dict(supported_both=supported_both, distinct_tracks=distinct_tracks,
                                   distinct_events=distinct_events, reduces_both=reduces_both),
                decisions=decisions, group=tuple(sorted(switch.keys)))


def _capture(events, original, gaps, anchors, shapes, old):
    requests = defaultdict(list)
    active = {}
    for index, (gap, anchors_at_gap, shapes_at_gap, base) in enumerate(zip(gaps, anchors, shapes, old, strict=True)):
        if base is None or base["innovation"]["rolling"]["support_count"] != 2:
            continue
        stamp = gap["stop"]
        samples = [sample for sample, _ in anchors_at_gap if sample["stamp"] == stamp]
        previous, boundary = _report_at(shapes_at_gap, stamp)
        if len(samples) != 1 or boundary is None or previous is None or base["stamp"] != stamp:
            raise ValueError("a dual support window lost its frozen inputs")
        sample, = samples
        switch = Switch(**gap["switch"])
        if set(sample["keys"]) != switch.keys or gap["end"] != "kf_rejoin":
            raise ValueError("a frozen candidate group changed")
        provenance = previous["shape"]["provenance"]
        active[index] = (switch, base, {})
        for side, report in base["innovation"]["reports"].items():
            key, time = sample["partners"][side], report["stamp"]
            if (not gap["start"] < time < stamp or time != base["lineage"]["roles"][side]["novel_post_onset"][-1]
                    or time in provenance[key]):
                raise ValueError("a selected report is not the frozen cache-excluded input")
            requests[time].append((index, side, key, report["rolling"]["prior_stamps"]))

    captured = [None] * len(gaps)
    if not requests:
        return captured
    bank = InnovationHistory()
    plane = Plane.from_first_batch(events[0][1].values())
    for (stamp, observations), native in zip(events, original.native_reports, strict=True):
        if (stamp, tuple(sorted(observations))) != native:
            raise ValueError("the original report clock or track keys changed")
        points = {key: plane.project(point) for key, point in observations.items()}
        for index, side, key, prior in requests.pop(stamp, ()):
            switch, base, parts = active[index]
            geometry, origin = _geometry(bank, switch, key, stamp, points, original.fit, prior)
            witness, mean = _witness(bank, key, stamp, origin, switch.keys, points)
            parts[side] = geometry, witness, mean
            if len(parts) == 2:
                captured[index] = _combine(parts, base, switch)
        bank.ingest(stamp, points)
    if requests:
        raise ValueError("a dual report was absent from the original clock")
    return captured


def _anonymous(capture, labels):
    group = {labels[key] for key in capture["group"]}
    source = {side: [labels[row["key"]] for row in witness["donors"]]
              for side, witness in capture["witnesses"].items()}
    identity = dict(within_group={side: len(set(identities) & group) for side, identities in source.items()},
                    within_pair={side: len(identities) - len(set(identities)) for side, identities in source.items()},
                    across_reports=len(set(source["release"]) & set(source["retain"])))
    witnesses = {side: witness | {"donors": [{k: v for k, v in row.items() if k != "key"} for row in witness["donors"]]}
                 for side, witness in capture["witnesses"].items()}
    return {k: v for k, v in capture.items() if k not in ("group", "stamp")} | dict(witnesses=witnesses,
                                                                                    donor_identity=identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    for name in ("manifest", "gap-run", "anchor-run", "shape-run", "async-run", "output-dir"):
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
                            _read_frozen(args.async_run / "frozen" / name))
        with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
            pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print("All common-innovation evidence and actions frozen. Joining anonymous evaluation.", flush=True)
    by_scene = defaultdict(list)
    for row in json.loads((args.async_run / "rows.json").read_text(encoding="utf-8")):
        by_scene[row["scene"]].append(row)
    rows = []
    for index, scene in enumerate(scenes, 1):
        alias = f"scene_{index:02d}"
        captures = _read_frozen(frozen / f"{alias}.pkl.gz")
        labels = read_labels(args.archive, args.manifest, scene) if any(c is not None for c in captures) else {}
        for previous, capture in zip(by_scene.pop(alias, ()), captures, strict=True):
            eligible = previous["innovation"] is not None and previous["innovation"]["rolling"]["support_count"] == 2
            if not eligible:
                fallback = dict.fromkeys((*ACTION_ARMS, "abstain"), previous["decisions"]["current"])
                rows.append(previous | {"common": None, "decisions": previous["decisions"] | fallback})
                continue
            if (capture is None or capture["stamp"] != previous["row"]["stop"]
                    or any(capture["decisions"][arm] != vote for arm, vote in previous["decisions"].items())):
                raise ValueError("an original decision or time changed")
            rows.append(previous | _anonymous(capture, labels))
    if by_scene or (len(rows), sum(r["innovation"] is not None for r in rows),
                    sum(r["common"] is not None for r in rows)) != (482, 416, 31):
        raise ValueError("the fixed evaluation population changed")
    summary = summarize(rows)
    _write_json(args.output_dir / "rows.json", rows)
    _write_json(args.output_dir / "summary.json", dict(clock="ideal_native_measurement_time", frozen_scenes=len(scenes),
                frozen_windows=len(rows), legal_rejoins=416, publications_unchanged=True, candidate_graph_unchanged=True,
                training_updates=0, external_dataset_reads=0, wall_seconds=perf_counter()-started, **summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
