"""Diagnose fixed-graph assignment reversals with causal four-fragment reports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import gzip
import math
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_tracker import Plane
from .evaluation_data import read_labels
from .hypothesis_gap_diagnostic import (
    Gap, Switch, _direction, _legal, _load_frozen, _margin, _position_readouts, _update_priors,
)
from .interval_assignment import MIN_SPAN_SECONDS, _bracketed, _coordinates, _dot
from .interval_assignment_experiment import _pair_spans, _roster
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import _distance_sq


def _sub(left, right):
    return left[0] - right[0], left[1] - right[1]


def _separations(switch, points):
    (a, b), (c, d) = switch.release
    if a[0] != c[0] or b[0] != d[0] or a[0] == b[0] or set(switch.retain) != {(a, d), (c, b)}:
        raise ValueError("relative motion requires two competing two-source perfect matchings")
    return _sub(points[a], points[c]), _sub(points[b], points[d])


def _same(left, right):
    return math.isclose(left, right, abs_tol=1e-5, rel_tol=1e-10)


def _decompose(switch, masked, seen, new):
    u, v = _separations(switch, masked)
    u1, v1 = _separations(switch, seen)
    du, dv = _sub(u1, u), _sub(v1, v)
    base, actual = 2*_dot(u, v), 2*_dot(u1, v1)
    terms = (2*_dot(du, v), 2*_dot(u, dv), 2*_dot(du, dv))
    if not (_same(base, _margin(switch, lambda e: _distance_sq(masked[e[0]], masked[e[1]])))
            and _same(actual, _margin(switch, lambda e: _distance_sq(seen[e[0]], seen[e[1]])))
            and _same(actual-base, math.fsum(terms))):
        raise RuntimeError("the signed separation terms do not reproduce the group margin")

    sources = sorted({key[0] for key in new})
    sample = dict(masked=base, actual=actual, increment=actual-base,
                  terms=terms, reports=len(new), sources=len(sources),
                  prior_separation=[math.hypot(*p) for p in (u, v)],
                  actual_separation=[math.hypot(*p) for p in (u1, v1)])
    if len(sources) == 1:
        slot = int(sources[0] != switch.release[0][0][0])
        old, delta = (u, v)[slot], (du, dv)[slot]
        norm2 = _dot(old, old)
        sample["updated_source_slot"] = slot
        if norm2 > 0:
            axial = _dot(old, delta) / norm2
            sample.update(axial_only=base*(1+axial), transverse_only=actual-base*axial,
                          axis_reversed=axial < -1, relative_perturbation=math.hypot(*delta)/math.sqrt(norm2))
    if not new and (base != actual or any(terms)):
        raise RuntimeError("quiet frame changed the same-time score")
    return sample


def _motion(switch, histories, stamp, masked, new):
    keys = switch.keys
    if any(len(histories[key]) < 2 for key in keys):
        return None
    stop = min(histories[key][-1][0] for key in keys)
    start = stop - MIN_SPAN_SECONDS
    if any(histories[key][0][0] > start or histories[key][-1][0] > stamp for key in keys):
        return None
    support = {key: _bracketed(histories[key], start, stop) for key in keys}
    if any(not track for track in support.values()):
        return None
    updated_support = {key for key in new if support[key][-1][0] == stamp}
    observed, forecast = ({}, {}), ({}, {})
    for key, track in support.items():
        observed[0][key], observed[1][key] = _coordinates(track, (start, stop), (0., 0.))
        replaced = [*track[:-1], (stamp, masked[key])] if key in updated_support else track
        forecast[0][key], forecast[1][key] = _coordinates(replaced, (start, stop), (0., 0.))

    def score(points):
        u0, v0 = _separations(switch, points[0])
        u1, v1 = _separations(switch, points[1])
        du, dv = _sub(u1, u0), _sub(v1, v0)
        return 2*_dot(u1, v1), 2*_dot(du, dv), [math.hypot(*vec) for vec in (du, dv)]

    endpoint, displacement, motion_norms = score(observed)
    mask_endpoint, mask_displacement, _ = score(forecast)
    if not updated_support and (endpoint != mask_endpoint or displacement != mask_displacement):
        raise RuntimeError("a window with no new support changed when masked")
    return dict(start=start, stop=stop, support={key: tuple(t for t, _ in track) for key, track in support.items()},
                updated_reports=len(updated_support), updated_sources=len({key[0] for key in updated_support}),
                endpoint=endpoint, displacement=displacement, mask_endpoint=mask_endpoint,
                mask_displacement=mask_displacement, motion_norms=motion_norms)


@dataclass
class Trace:
    gap: Gap
    counts: Counter = field(default_factory=Counter)
    onset: dict | None = None
    quiet: dict | None = None
    first_new: dict | None = None
    previous: dict | None = None
    first_cross: tuple[dict, dict] | None = None
    first_same: tuple[dict, dict] | None = None
    motion_onset: dict | None = None
    motion_later: dict | None = None
    motion_first: dict | None = None
    motion_disjoint: dict | None = None

    def observe(self, stamp, points, priors, histories, costs):
        gap = self.gap
        self.counts["events"] += 1
        new = gap.switch.keys.intersection(points)
        self.counts["new" if new else "quiet"] += 1
        if not _legal(gap.switch.edges, costs):
            raise ValueError("an active frozen group lost candidate eligibility")
        if not new and self.quiet is not None and stamp != gap.start:
            return
        positions = _position_readouts(gap.switch.keys, stamp, points, priors, {})
        if positions is None:
            self.counts["no_prior"] += 1
            if new:
                self.previous = None
            return
        masked, seen, new = positions
        point = _decompose(gap.switch, masked, seen, new) | {"delay": stamp-gap.start}
        if stamp == gap.start:
            self.onset = point
        if not new:
            if self.quiet is None:
                self.quiet = point
        else:
            self._new_point(point)

        # A quiet event needs no second displacement readout.
        if not new and stamp != gap.start:
            return
        motion = _motion(gap.switch, histories, stamp, masked, new)
        if motion is None:
            self.counts["no_common_window"] += 1
            return
        motion["delay"] = stamp-gap.start
        if stamp == gap.start:
            self.motion_onset = motion
        if not motion["updated_reports"]:
            self.counts["old_support_only"] += 1
            return
        self.counts["new_motion"] += 1
        if stamp > gap.start and self.motion_later is None:
            self.motion_later = motion
        if self.motion_first is None:
            self.motion_first = motion
        elif self.motion_disjoint is None and all(
            old[-1] < motion["support"][key][0] for key, old in self.motion_first["support"].items()
        ):
            self.motion_disjoint = motion

    def _new_point(self, point):
        if self.first_new is None:
            self.first_new = point
        previous = self.previous
        if previous is not None and previous["sources"] == point["sources"] == 1:
            same = previous["updated_source_slot"] == point["updated_source_slot"]
            if same and self.first_same is None:
                self.first_same = previous, point
            if not same and self.first_cross is None:
                self.first_cross = previous, point
        self.previous = point

    def public(self):
        def timing(sample):
            if sample is None:
                return None
            start = self.gap.start
            return {key: value for key, value in sample.items() if key not in ("support", "start", "stop")} | {
                "left_offset": sample["start"]-start, "right_offset": sample["stop"]-start,
                "age_of_right_endpoint": start+sample["delay"]-sample["stop"],
                "earliest_support_offset": min(ts[0] for ts in sample["support"].values())-start,
            }

        return dict(duration=self.gap.stop-self.gap.start, counts=dict(self.counts), onset=self.onset,
                    quiet=self.quiet, first_new=self.first_new, first_cross=self.first_cross,
                    first_same=self.first_same, motion_onset=timing(self.motion_onset),
                    motion_later=timing(self.motion_later), motion_disjoint=timing(self.motion_disjoint))


def _capture(events, graph_stream, result, gaps):
    traces = [Trace(Gap(**(row | {"switch": Switch(**row["switch"])}))) for row in gaps]
    starts = defaultdict(list)
    for trace in traces:
        starts[trace.gap.start].append(trace)
    active = []
    plane = Plane.from_first_batch(events[0][1].values())
    priors, histories = {}, defaultdict(list)
    counts = Counter()
    for (stamp, observations), native in zip(events, result.native_reports, strict=True):
        t, keys, _, _, edges, _ = pickle.load(graph_stream)
        if (t, keys) != native or (stamp, tuple(sorted(observations))) != native:
            raise ValueError("native reports do not match both frozen runs")
        points = {key: plane.project(point) for key, point in observations.items()}
        for key, point in points.items():
            histories[key].append((stamp, point))
        active = [trace for trace in active if stamp < trace.gap.stop]
        active.extend(starts.pop(stamp, ()))
        if active:
            costs = dict(edges)
            for trace in active:
                trace.observe(stamp, points, priors, histories, costs)
        _update_priors(priors, stamp, points)
        counts.update(events=1, reports=len(keys), candidate_edges=len(edges))
    original = pickle.load(graph_stream)
    if (starts or original.native_reports != result.native_reports or original.runs["kf"] != result.runs["kf"]
            or original.fit != result.fit or graph_stream.read(1)):
        raise ValueError("gap and original graph snapshots disagree")
    if counts["candidate_edges"] != result.diagnostics["fixed_candidate_edge_events"]:
        raise ValueError("frozen graph size changed")
    for trace in traces:
        gap = trace.gap
        for expected, found in ((gap.onset, trace.onset), (gap.first_new, trace.first_new)):
            if expected is None or gap.start+expected["delay"] >= gap.stop > gap.start:
                continue
            if (found is None or found["delay"] != expected["delay"]
                    or not _same(found["masked"], expected["masked_m2"])
                    or not _same(found["actual"], expected["actual_m2"])):
                raise ValueError("the equal-time position control differs from C-ON37")
    return traces, dict(counts)


def _positive(value):
    return _direction(value) == 1


def _point_release(sample):
    return _positive(sample["actual"]) and _positive(sample["increment"])


def _decisions(row):
    decisions = {}
    for key in ("first_cross", "first_same"):
        pair = row[key]
        if pair is not None:
            first, second = pair
            decisions[key] = dict(delay=second["delay"], retained=False, masked=_positive(second["masked"]),
                                  point=_positive(second["actual"]), single=_point_release(second),
                                  consecutive=_point_release(first) and _point_release(second))
    for key in ("motion_later", "motion_onset"):
        sample = row[key]
        if sample is not None and sample["updated_reports"]:
            point = _positive(sample["endpoint"])
            motion = _positive(sample["displacement"])
            decisions[key] = dict(delay=sample["delay"], retained=False, point=point, motion=motion,
                                  agreement=point and motion,
                                  masked=_positive(sample["mask_endpoint"]) and _positive(sample["mask_displacement"]))
    return decisions


def _target_bridges(result, rows, by_focal, true):
    kf = _pair_spans(result.native_reports, result.runs["kf"])
    i22 = _pair_spans(result.native_reports, result.runs["interval_swap"])
    output = []
    for edge, spans in i22.items():
        if true(edge):
            continue
        for start, stop, _ in spans:
            if stop-start <= 60:
                continue
            originals = [(a, b) for a, b, _ in kf.get(edge, ()) if max(start, a) < min(stop, b)]
            if any(b-a > 60 for a, b in originals):
                continue
            for (_, left), (right, _) in zip(originals, originals[1:]):
                if right <= left:
                    continue
                index = by_focal.get((edge, left))
                trace = dict(duration=right-left, row=index)
                if index is not None:
                    row = rows[index]
                    trace["remaining"] = {name: max(0., min(right-left, row["duration"])-d["delay"])
                                          for name, d in row["decisions"].items()}
                output.append(trace)
    return output


def _score(result, gaps, traces, labels):
    def true(edge):
        return labels[edge[0]] == labels[edge[1]]

    rows, index = [], {}
    for i, (gap, row) in enumerate(zip(gaps, traces, strict=True)):
        for edge in gap["focal"]:
            index[edge, gap["start"]] = i
        rows.append(row | dict(focal=len(gap["focal"]), focal_true=sum(map(true, gap["focal"])),
                               true_release=sum(map(true, gap["switch"]["release"])),
                               true_retain=sum(map(true, gap["switch"]["retain"])),
                               decisions=_decisions(row)))
    return rows, _target_bridges(result, rows, index, true)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-run", type=Path, required=True)
    parser.add_argument("--gap-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    roster = _roster(args.manifest)
    scenes = roster["splits"]["development"]
    if scenes != read_roster(args.manifest)["splits"]["held"]:
        raise ValueError("the graph must use the original 28-scene roster")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    frozen = args.output_dir / "private_frozen"
    frozen.mkdir(mode=0o700)
    began = perf_counter()
    entries = {}
    for i, scene in enumerate(scenes, 1):
        alias = f"scene_{i:02d}"
        result, gaps = _load_frozen(args.gap_run / "private_frozen" / f"{alias}.pkl.gz")
        with gzip.open(args.graph_run / "private_frozen" / f"{alias}.pkl.gz", "rb") as graph:
            traces, counts = _capture(_events(args.archive, args.manifest, scene), graph, result, gaps)
        with gzip.open(frozen / f"{alias}.pkl.gz", "xb", compresslevel=1) as stream:
            # Keep private support identities for an independent provenance audit.
            payload = [trace.public() | {"supports": {
                name: getattr(trace, name)["support"] for name in
                ("motion_onset", "motion_first", "motion_later", "motion_disjoint") if getattr(trace, name)
            }} for trace in traces]
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        entries[alias] = dict(component=roster["components"][scene],
                              sparse=2*result.diagnostics["first_peers_le3"] >= result.diagnostics["first_total"] > 0,
                              counts=counts)
        print(f"Frozen {i}/{len(scenes)}, {len(traces)} groups", flush=True)

    print("All motion and report contrasts frozen. Reading evaluation identities.", flush=True)
    for i, scene in enumerate(scenes, 1):
        alias = f"scene_{i:02d}"
        result, gaps = _load_frozen(args.gap_run / "private_frozen" / f"{alias}.pkl.gz")
        with gzip.open(frozen / f"{alias}.pkl.gz", "rb") as stream:
            payload = pickle.load(stream)
        for row in payload:
            row.pop("supports")
        rows, bridges = _score(result, gaps, payload, read_labels(args.archive, args.manifest, scene))
        entries[alias].update(rows=rows, bridges=bridges)
    _write_json(args.output_dir / "summary.json", dict(scenes=entries, clock="ideal_native_measurement_time",
                window_seconds=MIN_SPAN_SECONDS, publications_unchanged=True, candidate_graph_unchanged=True,
                training_updates=0, wall_seconds=perf_counter()-began))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
