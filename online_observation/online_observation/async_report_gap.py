"""Replay frozen assignment windows with causal, asynchronous report scores."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .async_report_scores import reported_direction, score_batch
from .continuous_tracker import Plane
from .evaluation_data import read_labels
from .hypothesis_gap_diagnostic import Switch, _direction, _legal, _load_frozen, _update_priors
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import _registration_shifts
from .registration_probe import FIT_STOP_S
from .relative_motion_gap import _same, _target_bridges


READOUTS = tuple(f"{score}_{mode}" for score in
                 ("position", "euclidean", "common", "mahalanobis", "likelihood")
                 for mode in ("actual", "masked"))


def _positive(arm, value):
    direction = _direction if arm.startswith(("position_", "euclidean_")) else reported_direction
    return direction(value) > 0


def _replay(events, native_result, graph, gaps):
    starts = defaultdict(list)
    switches = [Switch(**gap["switch"]) for gap in gaps]
    for index, gap in enumerate(gaps):
        starts[gap["start"]].append(index)
    traces = [dict(batches=[], counts=Counter(), boundary_reports=None, full_rejoin=False) for _ in gaps]
    active = []
    priors = {}
    plane = Plane.from_first_batch(events[0][1].values())
    shifts = _registration_shifts(native_result.fit)
    candidate_events = 0
    for (stamp, observations), native in zip(events, native_result.native_reports, strict=True):
        graph_stamp, keys, _, _, candidates, selected = pickle.load(graph)
        if (stamp, tuple(sorted(observations))) != native or (graph_stamp, keys) != native:
            raise ValueError("replay and frozen graph have different native clocks or report keys")
        candidate_events += len(candidates)
        active = [index for index in active if stamp <= gaps[index]["stop"]]
        active.extend(starts.pop(stamp, ()))
        points = {key: plane.project(point) for key, point in observations.items()}
        registration = shifts if stamp > FIT_STOP_S else {}
        if active:
            costs = dict(candidates)
            for index in active:
                gap, switch, trace = gaps[index], switches[index], traces[index]
                is_boundary = stamp == gap["stop"]
                group_reports = len(switch.keys.intersection(points))
                if is_boundary:
                    trace["boundary_reports"] = group_reports
                    trace["full_rejoin"] = set(switch.retain) <= set(selected)
                    if gap["end"] != "kf_rejoin":
                        continue
                elif stamp > gap["start"]:
                    trace["counts"].update(window=1, reported=int(group_reports > 0), quiet=int(group_reports == 0))
                if not _legal(switch.edges, costs):
                    raise ValueError("original group has an ineligible comparison edge")
                batch = score_batch(switch, stamp, points, priors, registration)
                if batch is None:
                    if group_reports and not is_boundary:
                        trace["counts"]["prior_unavailable"] += 1
                    continue
                batch["stamp"] = stamp
                batch["registration"] = bool(registration)
                batch["unregistered"] = (score_batch(switch, stamp, points, priors, {}) if registration else None)
                if stamp == gap["start"] and gap["onset"] is not None and not (
                    _same(batch["position_actual"], gap["onset"]["actual_m2"])
                    and _same(batch["position_masked"], gap["onset"]["masked_m2"])
                ):
                    raise ValueError("onset endpoint readout differs from the frozen causal control")
                trace["batches"].append(batch)
        _update_priors(priors, stamp, points)
    original = pickle.load(graph)
    if (starts or candidate_events != native_result.diagnostics["fixed_candidate_edge_events"]
            or original.native_reports != native_result.native_reports
            or original.runs["kf"] != native_result.runs["kf"] or original.fit != native_result.fit
            or graph.read(1)):
        raise ValueError("original graph or native candidate count differs from the frozen replay")
    return traces


def _public_batch(batch, tokens, sources):
    if batch is None:
        return None
    reports = []
    for report in batch["reports"]:
        clean = dict(report)
        clean["source"] = sources.setdefault(report["key"][0], len(sources))
        for name in ("key", "release_key", "retain_key"):
            clean[name] = tokens.setdefault(report[name], len(tokens))
        reports.append(clean)
    result = batch | {"reports": reports}
    if batch.get("unregistered") is not None:
        result["unregistered"] = _public_batch(batch["unregistered"], tokens, sources)
    return result


def _is_late(timing):
    first = timing["first"]
    return first["within"]["refreshed_bracket"] is None and any(
        first[zone]["refreshed_bracket"] is not None for zone in ("boundary", "beyond")
    )


def _evaluate(result, gaps, traces, timing, labels, previous):
    def true(edge):
        return labels[edge[0]] == labels[edge[1]]

    rows, by_focal = [], {}
    tokens, sources = {}, {}
    for index, (gap, trace, support) in enumerate(zip(gaps, traces, timing, strict=True)):
        start, stop, end = (gap[name] for name in ("start", "stop", "end"))
        if (start, stop, end) != tuple(support[name] for name in ("start", "stop", "end")):
            raise ValueError("timing contract no longer matches the frozen risk window")
        for edge in gap["focal"]:
            by_focal[edge, start] = index
        batches = [_public_batch(batch, tokens, sources) for batch in trace["batches"]]
        early = [batch for batch in batches if batch["stamp"] < stop]
        later = [batch for batch in early if batch["stamp"] > start]
        first_warning = {}
        for batch in early:
            for arm in READOUTS:
                if _positive(arm, batch[arm]):
                    first_warning.setdefault(arm, batch["stamp"])
        counts = trace["counts"]
        row = dict(start=start, stop=stop, end=end, duration=stop-start, late_support=_is_late(support),
                   full_rejoin=trace["full_rejoin"], boundary_reports=trace["boundary_reports"],
                   focal=len(gap["focal"]), focal_true=sum(map(true, gap["focal"])),
                   true_release=sum(map(true, gap["switch"]["release"])),
                   true_retain=sum(map(true, gap["switch"]["retain"])), target_edges=0,
                   window_batches=counts["window"], group_batches=counts["reported"],
                   quiet_batches=counts["quiet"], prior_unavailable=counts["prior_unavailable"],
                   first_warning=first_warning, batches=batches, decisions={},
                   onset=next((batch for batch in early if batch["stamp"] == start), None),
                   latest=early[-1] if early else None, later_first=later[0] if later else None,
                   later_last=later[-1] if later else None,
                   boundary=next((batch for batch in batches if batch["stamp"] == stop), None))
        rows.append(row)
    bridges = _target_bridges(result, rows, by_focal, true)
    if [(bridge["row"], bridge["duration"]) for bridge in bridges] != [
        (bridge["row"], bridge["duration"]) for bridge in previous["bridges"]
    ]:
        raise ValueError("wrong-chain focal population differs from C-ON39")
    for bridge in bridges:
        if bridge["row"] is not None:
            row = rows[bridge["row"]]
            if row["end"] == "kf_rejoin" and row["late_support"] and bridge["duration"] == row["duration"]:
                row["target_edges"] += 1
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-run", type=Path, required=True)
    parser.add_argument("--gap-run", type=Path, required=True)
    parser.add_argument("--timing-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    roster = _roster(args.manifest)
    scenes = roster["splits"]["development"]
    if scenes != read_roster(args.manifest)["splits"]["held"]:
        raise ValueError("the diagnostic must use the original 28-scene roster")
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    frozen = args.output_dir / "private_frozen"
    frozen.mkdir(mode=0o700)
    began = perf_counter()
    entries = {}
    for index, scene in enumerate(scenes, 1):
        alias = f"scene_{index:02d}"
        archive_name = f"{alias}.pkl.gz"
        result, gaps = _load_frozen(args.gap_run / "private_frozen" / archive_name)
        with gzip.open(args.graph_run / "private_frozen" / archive_name, "rb") as graph:
            traces = _replay(_events(args.archive, args.manifest, scene), result, graph, gaps)
        with gzip.open(frozen / archive_name, "xb", compresslevel=1) as stream:
            pickle.dump(traces, stream, protocol=pickle.HIGHEST_PROTOCOL)
        entries[alias] = dict(component=roster["components"][scene], registration=bool(_registration_shifts(result.fit)),
                              sparse=2*result.diagnostics["first_peers_le3"] >= result.diagnostics["first_total"] > 0,
                              events=result.diagnostics["events"], reports=result.diagnostics["reports"],
                              candidate_edge_events=result.diagnostics["fixed_candidate_edge_events"])
        print(f"Frozen {index}/{len(scenes)}, {len(traces)} groups", flush=True)

    print("All report scores frozen. Reading evaluation identities.", flush=True)
    previous = json.loads((args.timing_run / "summary.json").read_text())["scenes"]
    for index, scene in enumerate(scenes, 1):
        alias = f"scene_{index:02d}"
        archive_name = f"{alias}.pkl.gz"
        result, gaps = _load_frozen(args.gap_run / "private_frozen" / archive_name)
        with gzip.open(frozen / archive_name, "rb") as stream:
            traces = pickle.load(stream)
        with gzip.open(args.timing_run / "private_frozen" / archive_name, "rb") as stream:
            timing, _ = pickle.load(stream)
        rows = _evaluate(result, gaps, traces, timing, read_labels(args.archive, args.manifest, scene), previous[alias])
        entries[alias].update(rows=rows)
    payload = dict(scenes=entries, clock="ideal_native_measurement_time", publications_unchanged=True,
                   candidate_graph_unchanged=True, training_updates=0, wall_seconds=perf_counter()-began)
    _write_json(args.output_dir / "summary.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
