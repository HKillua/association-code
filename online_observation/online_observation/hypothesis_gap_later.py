"""Read later report evidence inside already frozen assignment disagreements."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_tracker import Plane
from .evaluation_data import read_labels
from .hypothesis_gap_diagnostic import Gap, Switch, _bridges, _legal, _load_frozen, _measure, _update_priors
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import _registration_shifts
from .registration_probe import FIT_STOP_S


def _capture_later(events, graph_stream, result, gaps):
    starts = defaultdict(list)
    groups = [Gap(**(gap | {"switch": Switch(**gap["switch"])})) for gap in gaps]
    for i, gap in enumerate(groups):
        if gap.counts.get("new_events", 0) > bool(gap.onset and gap.onset["reports"]):
            starts[gap.start].append(i)
    samples = [dict(first_later=None, first_other_source=None) for _ in groups]
    if not starts:
        return samples
    if len(events) != len(result.native_reports):
        raise ValueError("native reports and frozen later windows differ")
    plane = Plane.from_first_batch(events[0][1].values())
    shifts = _registration_shifts(result.fit)
    priors, active = {}, {}
    for (stamp, observations), original in zip(events, result.native_reports, strict=True):
        t, reports, _, _, candidate_items, _ = pickle.load(graph_stream)
        if (stamp, tuple(sorted(observations))) != original or (t, reports) != original:
            raise ValueError("late report trace changed relative to the frozen graph")
        points = {key: plane.project(point) for key, point in observations.items()}
        current_costs = None
        for i, trigger_sources in tuple(active.items()):
            gap = groups[i]
            if stamp >= gap.stop:
                del active[i]
                continue
            sources = {key[0] for key in gap.switch.keys & points.keys()}
            if not sources:
                continue
            row = samples[i]
            other_source = bool(trigger_sources and sources - trigger_sources)
            if row["first_later"] is not None and not other_source:
                continue
            if current_costs is None:
                current_costs = dict(candidate_items)
            if not _legal(gap.switch.edges, current_costs):
                raise ValueError("a frozen later comparison has an illegal edge")
            sample = _measure(gap, stamp, points, priors, shifts if stamp > FIT_STOP_S else {}, current_costs)
            if sample is None:
                continue
            sample["other_than_trigger_source"] = other_source
            sample["remaining_seconds"] = gap.stop - stamp
            if row["first_later"] is None:
                row["first_later"] = sample
            if other_source:
                row["first_other_source"] = sample
                del active[i]

        for i in starts.get(stamp, ()):
            active[i] = {key[0] for key in groups[i].switch.keys & points.keys()}
        _update_priors(priors, stamp, points)
    return samples


def _later_score(result, gaps, samples, labels):
    def true(edge):
        return labels[edge[0]] == labels[edge[1]]

    rows = []
    for gap, sample in zip(gaps, samples, strict=True):
        rows.append(dict(focal_count=len(gap["focal"]), focal_true=sum(map(true, gap["focal"])),
                         true_retain=sum(map(true, gap["switch"]["retain"])),
                         true_release=sum(map(true, gap["switch"]["release"])), **sample))
    bridges = {}
    for kind in ("first_later", "first_other_source"):
        index = {(edge, gap["start"]): gap | {"first_new": sample[kind]}
                 for gap, sample in zip(gaps, samples, strict=True) for edge in gap["focal"]}
        bridges[kind] = _bridges(result, true, index)
    return rows, bridges


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
        raise ValueError("the graph must use the original frozen development roster")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    destination = args.output_dir / "private_frozen"
    destination.mkdir(mode=0o700)
    began = perf_counter()
    entries = {}
    for i, scene in enumerate(scenes, 1):
        alias = f"scene_{i:02d}"
        result, gaps = _load_frozen(args.gap_run / "private_frozen" / f"{alias}.pkl.gz")
        has_later = any(gap["counts"].get("new_events", 0) > bool(gap["onset"] and gap["onset"]["reports"])
                        for gap in gaps)
        if has_later:
            with gzip.open(args.graph_run / "private_frozen" / f"{alias}.pkl.gz", "rb") as graph:
                samples = _capture_later(_events(args.archive, args.manifest, scene), graph, result, gaps)
        else:
            samples = [dict(first_later=None, first_other_source=None) for _ in gaps]
        with gzip.open(destination / f"{alias}.pkl.gz", "xb", compresslevel=1) as stream:
            pickle.dump((samples, has_later), stream, protocol=pickle.HIGHEST_PROTOCOL)
        entries[alias] = dict(component=roster["components"][scene],
                             sparse=2*result.diagnostics["first_peers_le3"] >= result.diagnostics["first_total"] > 0)
        print(f"Frozen {i}/{len(scenes)}", flush=True)

    print("All later contrasts frozen. Reading evaluation identities.", flush=True)
    for i, scene in enumerate(scenes, 1):
        alias = f"scene_{i:02d}"
        result, gaps = _load_frozen(args.gap_run / "private_frozen" / f"{alias}.pkl.gz")
        with gzip.open(destination / f"{alias}.pkl.gz", "rb") as stream:
            samples, has_later = pickle.load(stream)
        rows, bridges = _later_score(result, gaps, samples, read_labels(args.archive, args.manifest, scene))
        entries[alias].update(rows=rows, bridges=bridges, measurement_reads=has_later)
    _write_json(args.output_dir / "summary.json", dict(scenes=entries, clock="ideal_native_measurement_time",
               wall_seconds=perf_counter()-began))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
