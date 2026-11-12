"""Audit timing and prior same-source innovations on frozen assignment disagreements."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_tracker import FRESH_S, RADIUS_M, Plane
from .evaluation_data import read_labels
from .hypothesis_gap_diagnostic import Gap, Switch, _direction, _load_frozen, _margin, _position_readouts, _update_priors
from .interval_assignment import _dot
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import _distance_sq
from .relative_motion_gap import _same, _sub, _target_bridges
from .support_gap_audit import audit_support


@dataclass(frozen=True)
class Innovation:
    stamp: float
    vector: tuple[float, float]
    prior_age: float


class InnovationHistory:
    def __init__(self):
        self.priors = {}
        self.streak = Counter()
        self.latest = defaultdict(dict)

    def ingest(self, stamp, points):
        for key, point in points.items():
            old = self.priors.get(key)
            age = stamp-old.stamp if old is not None else math.inf
            records = self.latest[key[0]]
            if 0 < age <= FRESH_S:
                if self.streak[key] >= 2:
                    records[key] = Innovation(stamp, _sub(point, old.position(stamp, True)), age)
                self.streak[key] += 1
            else:
                self.streak[key] = 1
                records.pop(key, None)
        _update_priors(self.priors, stamp, points)

    def candidates(self, source, stamp, origin, exclude):
        candidates = []
        for key, record in self.latest[source].items():
            if key in exclude or not 0 < stamp-record.stamp <= FRESH_S:
                continue
            if self.priors[key].stamp != record.stamp:
                raise ValueError("a neighbor innovation is not from its latest report")
            distance = _distance_sq(origin, self.priors[key].position(stamp, True))
            candidates.append((distance, key, record))
        return sorted(candidates)


def _norm2(vector):
    return _dot(vector, vector)


def _average(left, right):
    return (left[0]+right[0])/2, (left[1]+right[1])/2


def _margin_for(switch, positions):
    return _margin(switch, lambda edge: _distance_sq(positions[edge[0]], positions[edge[1]]))


def _matched_far(far, near):
    if len(far) < len(near):
        return []
    remaining = list(far)
    matched = []
    for _, _, local in near:
        chosen = min(remaining, key=lambda c: (abs(c[2].stamp-local.stamp), c[0], c[1]))
        remaining.remove(chosen)
        matched.append(chosen)
    return matched


def _sample(gap, stamp, points, bank):
    keys = gap.switch.keys
    new = keys.intersection(points)
    if len(new) != 1:
        raise ValueError("shared innovation diagnostic requires one reported endpoint")
    target, = new
    same_batch = sum(key[0] == target[0] and key not in keys for key in points)
    result = dict(delay=stamp-gap.start, same_batch_external=same_batch, target=target)
    readouts = _position_readouts(keys, stamp, points, bank.priors, {})
    if readouts is None:
        return result | {"status": "no_group_prior"}
    masked, actual, _ = readouts
    innovation = _sub(actual[target], masked[target])
    baseline_sq = _norm2(innovation)
    raw = _margin_for(gap.switch, actual)
    prior = _margin_for(gap.switch, masked)
    candidates = bank.candidates(target[0], stamp, masked[target], keys)
    near = [item for item in candidates if item[0] <= RADIUS_M**2]
    result.update(near=len(near), raw=raw, masked=prior, raw_sq=baseline_sq,
                  target_prior_age=stamp-bank.priors[target].stamp)
    if len(near) < 2:
        return result | {"status": "too_few_neighbors"}
    witnesses = near[:2]
    a, b = (record.vector for _, _, record in witnesses)
    result.update(witnesses=tuple(key for _, key, _ in witnesses),
                  witness_stamps=tuple(record.stamp for _, _, record in witnesses),
                  witness_prior_ages=tuple(record.prior_age for _, _, record in witnesses),
                  witness_distances=tuple(math.sqrt(distance) for distance, _, _ in witnesses),
                  witness_dot=_dot(a, b))
    if result["witness_dot"] <= 0:
        return result | {"status": "neighbors_disagree"}

    common = _average(a, b)
    magnitude = math.sqrt(_norm2(common))
    target_magnitude = math.sqrt(baseline_sq)
    applied = min(magnitude, target_magnitude)
    corrected = tuple(value*applied/magnitude for value in common)
    attenuated = (tuple(value*applied/target_magnitude for value in innovation)
                  if target_magnitude > 0 else (0., 0.))
    shifted = actual | {key: _sub(actual[key], corrected) for key in keys if key[0] == target[0]}
    if not _same(_margin_for(gap.switch, shifted), raw):
        raise RuntimeError("same-source common shift changed an equal-cardinality score")
    result.update(status="supported", raw_neighbor_sq=_norm2(_sub(innovation, common)),
                  applied_neighbor_sq=_norm2(_sub(innovation, corrected)), applied_norm=applied,
                  neighbor=_margin_for(gap.switch, actual | {target: _sub(actual[target], corrected)}),
                  shrink=_margin_for(gap.switch, actual | {target: _sub(actual[target], attenuated)}))

    far = _matched_far([item for item in candidates if item[0] > RADIUS_M**2], witnesses)
    if far:
        far_a, far_b = (record.vector for _, _, record in far)
        far_mean = _average(far_a, far_b)
        result.update(far_sq=_norm2(_sub(innovation, far_mean)),
                      far_witnesses=tuple(key for _, key, _ in far),
                      far_stamps=tuple(record.stamp for _, _, record in far),
                      far_dot=_dot(far_a, far_b))
    return result


def _capture(events, result, frozen_gaps):
    gaps = [Gap(**(row | {"switch": Switch(**row["switch"])})) for row in frozen_gaps]
    starts = defaultdict(list)
    for index, gap in enumerate(gaps):
        starts[gap.start].append(index)
    rows = [dict(onset=None, later_first=None, later_supported=None, counts=Counter()) for _ in gaps]
    active = []
    plane = Plane.from_first_batch(events[0][1].values())
    bank = InnovationHistory()
    for (stamp, observations), native in zip(events, result.native_reports, strict=True):
        if (stamp, tuple(sorted(observations))) != native:
            raise ValueError("observations no longer match the frozen native clock")
        points = {key: plane.project(point) for key, point in observations.items()}
        active = [index for index in active if stamp < gaps[index].stop]
        active.extend(starts.pop(stamp, ()))
        for index in active:
            gap = gaps[index]
            row = rows[index]
            count = len(gap.switch.keys.intersection(points))
            is_onset = stamp == gap.start
            row["counts"]["single" if count == 1 else "quiet" if count == 0 else "multiple"] += 1
            if count != 1:
                if is_onset:
                    row["onset"] = dict(status="quiet" if not count else "multiple", delay=0.)
                continue
            if not is_onset and row["later_supported"] is not None:
                continue
            sample = _sample(gap, stamp, points, bank)
            if is_onset:
                row["onset"] = sample
                available = sample["status"] != "no_group_prior"
                if available != (gap.onset is not None) or (available and not (
                    _same(sample["raw"], gap.onset["actual_m2"])
                    and _same(sample["masked"], gap.onset["masked_m2"])
                )):
                    raise ValueError("same-time onset contrast differs from the frozen position readout")
            elif row["later_first"] is None:
                row["later_first"] = sample
            if not is_onset and sample["status"] == "supported":
                row["later_supported"] = sample
        bank.ingest(stamp, points)
    if starts:
        raise ValueError("one or more frozen starts have no native event")
    return [row | {"counts": dict(row["counts"])} for row in rows]


def _score(result, gaps, neighbors, support, labels):
    def true(edge):
        return labels[edge[0]] == labels[edge[1]]

    rows, index = [], {}
    for i, (gap, row, timing) in enumerate(zip(gaps, neighbors, support, strict=True)):
        for edge in gap["focal"]:
            index[edge, gap["start"]] = i
        public = dict(duration=gap["stop"]-gap["start"], support=timing,
                      focal=len(gap["focal"]), focal_true=sum(map(true, gap["focal"])),
                      true_release=sum(map(true, gap["switch"]["release"])),
                      true_retain=sum(map(true, gap["switch"]["retain"])), counts=row["counts"], decisions={})
        for name in ("onset", "later_first", "later_supported"):
            sample = row[name]
            if sample is None:
                public[name] = None
                continue
            public[name] = {k: v for k, v in sample.items() if k not in
                            ("target", "witnesses", "far_witnesses", "witness_stamps", "far_stamps")}
            if sample["status"] != "supported":
                continue
            witnesses = [labels[key] for key in sample["witnesses"]]
            group_labels = {labels[key] for pair in gap["switch"]["release"] for key in pair}
            public[name].update(witnesses_same_identity=len(set(witnesses)) != len(witnesses),
                                witness_in_group_identity=bool(set(witnesses) & group_labels))
            public["decisions"][name] = {"delay": sample["delay"], "retain": False} | {
                arm: _direction(sample[arm]) > 0 for arm in ("raw", "masked", "neighbor", "shrink")
            }
        rows.append(public)
    return rows, _target_bridges(result, rows, index, true)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-run", type=Path, required=True)
    parser.add_argument("--gap-run", type=Path, required=True)
    parser.add_argument("--motion-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    roster = _roster(args.manifest)
    scenes = roster["splits"]["development"]
    if scenes != read_roster(args.manifest)["splits"]["held"]:
        raise ValueError("the graph must use the original 28-scene roster")
    prior = json.loads((args.motion_run / "summary.json").read_text())["scenes"]
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    frozen = args.output_dir / "private_frozen"
    frozen.mkdir(mode=0o700)
    entries = {}
    began = perf_counter()
    for i, scene in enumerate(scenes, 1):
        alias = f"scene_{i:02d}"
        result, gaps = _load_frozen(args.gap_run / "private_frozen" / f"{alias}.pkl.gz")
        graph = args.graph_run / "private_frozen" / f"{alias}.pkl.gz"
        support = audit_support(result, gaps, graph)
        for row, earlier in zip(support, prior[alias]["rows"], strict=True):
            got = row["first"]["within"]["comparable_bracket"]
            motion = earlier["motion_later"]
            if ((got is None) != (motion is None) or
                    (got is not None and (got["delay"] != motion["delay"]
                                          or got["common_right"]-row["start"] != motion["right_offset"]))):
                raise ValueError("four-end support audit disagrees with C-ON38")
        neighbors = _capture(_events(args.archive, args.manifest, scene), result, gaps)
        with gzip.open(frozen / f"{alias}.pkl.gz", "xb", compresslevel=1) as stream:
            pickle.dump((support, neighbors), stream, protocol=pickle.HIGHEST_PROTOCOL)
        entries[alias] = dict(component=roster["components"][scene], sparse=prior[alias]["sparse"],
                              events=result.diagnostics["events"], reports=result.diagnostics["reports"],
                              candidate_edge_events=result.diagnostics["fixed_candidate_edge_events"])
        print(f"Frozen {i}/{len(scenes)}, {len(gaps)} groups", flush=True)

    print("All support and neighbor contrasts frozen. Reading evaluation identities.", flush=True)
    for i, scene in enumerate(scenes, 1):
        alias = f"scene_{i:02d}"
        result, gaps = _load_frozen(args.gap_run / "private_frozen" / f"{alias}.pkl.gz")
        with gzip.open(frozen / f"{alias}.pkl.gz", "rb") as stream:
            support, neighbors = pickle.load(stream)
        rows, bridges = _score(result, gaps, neighbors, support, read_labels(args.archive, args.manifest, scene))
        entries[alias].update(rows=rows, bridges=bridges)
    _write_json(args.output_dir / "summary.json", dict(scenes=entries, clock="ideal_native_measurement_time",
                publications_unchanged=True, candidate_graph_unchanged=True,
                training_updates=0, wall_seconds=perf_counter()-began))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
