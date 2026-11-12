"""Test the effect of newly arrived reports during a frozen assignment disagreement."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import gzip
import json
import math
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_tracker import FRESH_S, MEMORY_S, LocalTrack, Plane
from .evaluation_data import read_labels
from .interval_assignment import PAIR_NULL_COST
from .interval_assignment_experiment import _pair_spans, _roster
from .nemot_episodes import _published
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import Edge, _add, _distance_sq, _registration_shifts
from .registration_probe import FIT_STOP_S


WINDOW_SECONDS = 60.0
COST_ROUNDOFF_M2 = 1e-6


@dataclass(frozen=True)
class Switch:
    release: tuple[Edge, Edge]
    retain: tuple[Edge, Edge]

    @property
    def edges(self):
        return (*self.release, *self.retain)

    @property
    def keys(self):
        return frozenset(key for edge in self.release for key in edge)


@dataclass
class Gap:
    switch: Switch
    start: float
    focal: tuple[Edge, ...]
    old_group_selected: bool
    old_group_legal: bool
    trigger_outside: bool
    trigger_registration: bool
    prior_cached: float | None
    counts: Counter = field(default_factory=Counter)
    onset: dict | None = None
    first_new: dict | None = None
    first_quiet: dict | None = None
    stop: float = 0.0
    end: str = ""


def _legal(edges, costs):
    return all(costs.get(edge, math.inf) < PAIR_NULL_COST for edge in edges)


def _switch(focal, kf_owner, interval_owner):
    left, right = (kf_owner.get(key) for key in focal)
    if left is None or right is None or left == right:
        return None
    release = tuple(sorted((left, right)))
    keys = {key for edge in release for key in edge}
    retain = {interval_owner.get(key) for key in keys}
    if None in retain or len(retain) != 2 or {key for edge in retain for key in edge} != keys:
        return None
    return Switch(release, tuple(sorted(retain)))


def _margin(switch, cost):
    return math.fsum([*(cost(edge) for edge in switch.retain),
                      *(-cost(edge) for edge in switch.release)])


def _direction(value):
    if value > COST_ROUNDOFF_M2:
        return 1
    if value < -COST_ROUNDOFF_M2:
        return -1
    return 0


def _measure(gap, stamp, points, priors, shifts, costs):
    keys = gap.switch.keys
    if any(key not in priors or not 0 < stamp - priors[key].stamp <= FRESH_S for key in keys):
        return None
    masked = {key: _add(priors[key].position(stamp, True), shifts.get(key[0], (0., 0.))) for key in keys}
    new = keys.intersection(points)
    seen = masked | {key: _add(points[key], shifts.get(key[0], (0., 0.))) for key in new}
    old_margin = _margin(gap.switch, lambda edge: _distance_sq(masked[edge[0]], masked[edge[1]]))
    new_margin = _margin(gap.switch, lambda edge: _distance_sq(seen[edge[0]], seen[edge[1]]))
    if not new and new_margin != old_margin:
        raise RuntimeError("the no-report contrast changed the score")
    return dict(delay=stamp-gap.start, reports=len(new), sources=len({key[0] for key in new}),
                focal_reports=len(new.intersection(key for edge in gap.focal for key in edge)),
                cached_m2=_margin(gap.switch, costs.__getitem__),
                masked_m2=old_margin, actual_m2=new_margin, increment_m2=new_margin-old_margin)


def _outside_report(keys, updated, costs):
    """Check for a report elsewhere in the same eligible candidate component."""
    outside = updated - keys
    if not outside:
        return False
    neighbors = defaultdict(set)
    for (left, right), cost in costs.items():
        if cost < PAIR_NULL_COST:
            neighbors[left].add(right)
            neighbors[right].add(left)
    pending = list(keys)
    reached = set(keys)
    for key in pending:
        new = neighbors[key] - reached
        if outside.intersection(new):
            return True
        reached.update(new)
        pending.extend(new)
    return False


def _update_priors(priors, stamp, points):
    for key, point in points.items():
        old = priors.get(key)
        priors[key] = (old.updated(stamp, point) if old is not None and stamp-old.stamp <= MEMORY_S
                       else LocalTrack(stamp, point, point))


def _collect(events, graph_stream, result):
    if not events or len(events) != len(result.native_reports):
        raise ValueError("measurements and frozen replay have different lengths")
    plane = Plane.from_first_batch(events[0][1].values())
    shifts = _registration_shifts(result.fit)
    priors, active = {}, {}
    closed = []
    counts = Counter()
    previous_kf, previous_interval = set(), set()
    previous_costs = {}
    current_interval = ()
    interval_changes = iter(result.runs["interval_swap"])
    next_change = next(interval_changes, None)
    previous_stamp = -math.inf

    for (stamp, observations), frozen_report in zip(events, result.native_reports, strict=True):
        t, reports, born, fresh_keys, candidate_items, selected = pickle.load(graph_stream)
        updated = set(observations)
        if (t, reports) != frozen_report or t != stamp or tuple(sorted(updated)) != reports:
            raise ValueError("native report timestamp or keys disagree across inputs")
        if next_change is not None and next_change[0] == stamp:
            current_interval = next_change[1]
            next_change = next(interval_changes, None)
        fresh = set(fresh_keys)
        if set(born) != updated - priors.keys() or fresh != updated | {
            key for key, prior in priors.items() if stamp - prior.stamp <= FRESH_S
        }:
            raise ValueError("frozen birth or freshness differs from arrived reports")
        kf = set(selected)
        interval = _published(current_interval, fresh)
        costs = dict(candidate_items)
        points = {key: plane.project(point) for key, point in observations.items()}
        registration = previous_stamp <= FIT_STOP_S < stamp
        departed = (previous_kf & previous_interval & interval) - kf
        kf_owner = {key: edge for edge in kf for key in edge} if departed else {}
        interval_owner = {key: edge for edge in interval for key in edge} if departed else {}

        for switch, gap in tuple(active.items()):
            expiry = min(priors[key].stamp + FRESH_S for key in switch.keys)
            deadline = gap.start + WINDOW_SECONDS
            if min(expiry, deadline) < stamp:
                gap.stop = min(expiry, deadline)
                gap.end = "expiry" if expiry < deadline else "window"
            elif not _legal(switch.edges, costs):
                gap.stop, gap.end = stamp, "graph_loss"
            elif any(edge in kf for edge in gap.focal):
                gap.stop, gap.end = stamp, "kf_rejoin"
            elif any(edge not in interval for edge in gap.focal):
                gap.stop, gap.end = stamp, "interval_withdrawal"
            elif not (set(switch.release) <= kf and set(switch.retain) <= interval):
                gap.stop, gap.end = stamp, "other_assignment"
            if gap.end:
                closed.append(active.pop(switch))

        entries = defaultdict(list)
        for edge in sorted(departed):
            counts["focal_departures"] += 1
            if any(stamp - priors[key].stamp > FRESH_S for key in edge):
                counts["focal_expired_before_event"] += 1
                continue
            switch = _switch(edge, kf_owner, interval_owner)
            if switch is None or not _legal(switch.edges, costs):
                counts["focal_without_legal_switch"] += 1
                continue
            entries[switch].append(edge)

        for switch, focal in entries.items():
            if switch in active:
                raise RuntimeError("a continuing group cannot start a second disagreement")
            if _direction(_margin(switch, costs.__getitem__)) < 0:
                raise RuntimeError("the frozen KF solution loses to a legal local switch")
            old_legal = _legal(switch.edges, previous_costs)
            cached = _margin(switch, previous_costs.__getitem__) if old_legal else None
            active[switch] = Gap(switch, stamp, tuple(focal),
                                 set(switch.retain) <= previous_kf,
                                 old_legal, _outside_report(switch.keys, updated, costs), registration, cached)

        for gap in active.values():
            keys = gap.switch.keys
            new = keys.intersection(updated)
            gap.counts["events"] += 1
            gap.counts["new_events" if new else "quiet_events"] += 1
            gap.counts["group_reports"] += len(new)
            gap.counts["focal_reports"] += len(new.intersection(key for edge in gap.focal for key in edge))
            if (gap.first_new if new else gap.first_quiet) is not None:
                continue
            sample = _measure(gap, stamp, points, priors, shifts if stamp > FIT_STOP_S else {}, costs)
            if stamp == gap.start:
                gap.onset = sample
            if sample is None:
                gap.counts["unavailable_prior_events"] += 1
            elif new:
                gap.first_new = sample
            else:
                sample["outside_report"] = (_outside_report(keys, updated, costs)
                                            or _outside_report(keys, updated, previous_costs))
                sample["registration"] = registration
                gap.first_quiet = sample

        _update_priors(priors, stamp, points)
        counts.update(events=1, reports=len(reports), candidate_edge_events=len(costs))
        previous_kf, previous_interval, previous_costs, previous_stamp = kf, interval, costs, stamp

    for gap in active.values():
        gap.stop, gap.end = stamp, "scene_end"
        closed.append(gap)
    original = pickle.load(graph_stream)
    if (next_change is not None or original.native_reports != result.native_reports
            or original.runs["kf"] != result.runs["kf"] or original.fit != result.fit
            or counts["candidate_edge_events"] != result.diagnostics["fixed_candidate_edge_events"]
            or counts["reports"] != result.diagnostics["reports"] or graph_stream.read(1)):
        raise ValueError("frozen graph trace differs from the interval baseline")
    return [asdict(gap) | {"counts": dict(gap.counts)} for gap in closed], dict(counts)


def _load_frozen(path):
    with gzip.open(path, "rb") as stream:
        result, gaps = pickle.load(stream)
    for gap in gaps:
        counts = gap["counts"]
        if isinstance(counts, Counter):
            # The first capture used asdict on a Counter, which rebuilt it
            # from pairs and made (name, value) into the stored keys.
            fixed = {}
            for (name, number), copies in counts.items():
                if not isinstance(name, str) or not isinstance(number, int) or copies != 1 or name in fixed:
                    raise ValueError("unrecognized frozen counter encoding")
                fixed[name] = number
            gap["counts"] = fixed
    return result, gaps


def _bridges(result, true, by_focal):
    baseline = _pair_spans(result.native_reports, result.runs["kf"])
    interval = _pair_spans(result.native_reports, result.runs["interval_swap"])
    output = []
    for edge, intervals in interval.items():
        if true(edge):
            continue
        reference = baseline.get(edge, ())
        for start, stop, _ in intervals:
            if stop-start <= 60:
                continue
            overlapping = [(a, b) for a, b, _ in reference if max(start, a) < min(stop, b)]
            only_short = all(b-a <= 60 for a, b in overlapping)
            for (a, b), (c, d) in zip(overlapping, overlapping[1:]):
                if b >= c or max(b-a, d-c) > 60:
                    continue
                row = dict(only_short=only_short, gap_seconds=c-b)
                gap = by_focal.get((edge, b))
                if gap is not None:
                    row.update(covered_seconds=max(0., min(c, gap["stop"])-b),
                               stop_reason=gap["end"], first_new_delay=None,
                               group_had_new=bool(gap["counts"].get("new_events", 0)))
                    evidence = gap["first_new"]
                    if evidence is not None and b+evidence["delay"] < c:
                        row["first_new_delay"] = evidence["delay"]
                        row["new_favors_release"] = _direction(evidence["increment_m2"]) > 0
                        row["new_flips_to_release"] = (_direction(evidence["masked_m2"]) < 0
                                                        and _direction(evidence["actual_m2"]) > 0)
                        row["remaining_at_new"] = max(0., min(c, gap["stop"])-b-evidence["delay"])
                output.append(row)
    return output


def _score(result, gaps, labels):
    def true(edge):
        return labels[edge[0]] == labels[edge[1]]

    by_focal = {(edge, gap["start"]): gap for gap in gaps for edge in gap["focal"]}
    if len(by_focal) != sum(len(gap["focal"]) for gap in gaps):
        raise RuntimeError("the same focal edge and start was collected twice")
    public = []
    for gap in sorted(gaps, key=lambda item: (item["start"], item["focal"])):
        release = sum(map(true, gap["switch"]["release"]))
        retain = sum(map(true, gap["switch"]["retain"]))
        public.append(dict(duration=gap["stop"]-gap["start"], end=gap["end"],
                           focal_true=sum(map(true, gap["focal"])), focal_count=len(gap["focal"]),
                           true_release=release, true_retain=retain, old_group_selected=gap["old_group_selected"],
                           old_group_legal=gap["old_group_legal"], prior_cached_m2=gap["prior_cached"],
                           trigger_outside=gap["trigger_outside"], trigger_registration=gap["trigger_registration"],
                           counts=gap["counts"], onset=gap["onset"], first_new=gap["first_new"],
                           first_quiet=gap["first_quiet"]))
    return public, _bridges(result, true, by_focal)


def _run(args):
    roster = _roster(args.manifest)
    scenes = roster["splits"]["development"]
    if scenes != read_roster(args.manifest)["splits"]["held"]:
        raise ValueError("historical graphs and interval development used different scenes")
    meta = json.loads((args.interval_run / "summary.json").read_text())
    if meta["phase"] != "development" or set(scenes) != meta["scenes"].keys():
        raise ValueError("this diagnostic requires the original 28-scene interval development run")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    frozen = (args.resume_frozen or args.output_dir) / "private_frozen"
    if args.resume_frozen is None:
        frozen.mkdir(mode=0o700)
    elif {path.name for path in frozen.glob("*.pkl.gz")} != {f"scene_{i:02d}.pkl.gz" for i in range(1, len(scenes)+1)}:
        raise ValueError("resume requires all 28 frozen contrasts")
    started = perf_counter()
    entries = {}
    for i, scene in enumerate(scenes, 1):
        alias = f"scene_{i:02d}"
        with gzip.open(args.interval_run / "private_frozen" / f"{scene}.pkl.gz", "rb") as stream:
            result, _ = pickle.load(stream)
        if args.resume_frozen is None:
            with gzip.open(args.graph_run / "private_frozen" / f"{alias}.pkl.gz", "rb") as stream:
                gaps, counts = _collect(_events(args.archive, args.manifest, scene), stream, result)
            with gzip.open(frozen / f"{alias}.pkl.gz", "xb", compresslevel=1) as stream:
                pickle.dump((result, gaps), stream, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            captured, gaps = _load_frozen(frozen / f"{alias}.pkl.gz")
            if (captured.native_reports != result.native_reports or captured.runs != result.runs
                    or captured.fit != result.fit or captured.diagnostics != result.diagnostics):
                raise ValueError("captured contrast does not match the frozen interval run")
            counts = dict(events=result.diagnostics["events"], reports=result.diagnostics["reports"],
                          candidate_edge_events=result.diagnostics["fixed_candidate_edge_events"])
        entry = meta["scenes"][scene]
        entries[alias] = dict(component=entry["component"], sparse=entry["sparse"], counts=counts,
                              focal_windows=sum(len(gap["focal"]) for gap in gaps))
        stage = "Checked" if args.resume_frozen else "Frozen"
        print(f"{stage} {i}/{len(scenes)}, {counts['events']} events", flush=True)

    print("All report contrasts frozen. Reading evaluation identities.", flush=True)
    for i, scene in enumerate(scenes, 1):
        alias = f"scene_{i:02d}"
        result, gaps = _load_frozen(frozen / f"{alias}.pkl.gz")
        rows, bridges = _score(result, gaps, read_labels(args.archive, args.manifest, scene))
        entries[alias].update(gaps=rows, bridges=bridges)
    _write_json(args.output_dir / "summary.json", dict(scenes=entries, clock="ideal_native_measurement_time",
               window_seconds=WINDOW_SECONDS, graph_unchanged=True, publications_unchanged=True,
               training_updates=0, resumed_frozen_capture=args.resume_frozen is not None,
               wall_seconds=perf_counter()-started))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    capture = parser.add_mutually_exclusive_group(required=True)
    capture.add_argument("--graph-run", type=Path)
    capture.add_argument("--resume-frozen", type=Path)
    parser.add_argument("--interval-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    _run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
