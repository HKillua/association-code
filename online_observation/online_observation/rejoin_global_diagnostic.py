"""Re-solve the original MTAD association graph at each frozen rejoin batch."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

from .continuous_tracker import Plane
from .evaluation_data import read_labels
from .hypothesis_gap_diagnostic import Switch, _direction, _load_frozen, _update_priors
from .interval_assignment_experiment import _roster
from .nemot_experiment import _events, _write_json, read_roster
from .nemot_replay import _registration_shifts
from .registration_probe import FIT_STOP_S
from .rejoin_batch_diagnostic import _read_frozen
from .rejoin_global_assignment import score_global
from .rejoin_global_summary import ARMS, summarize
from .relative_motion_gap import _same


VIEWS = ("actual", "masked", "fixed_mask")


def _directions(result, local_flip):
    actual, masked, fixed = (result[arm] for arm in VIEWS)
    full_retain = masked["retain_count"] == 2
    full_release = masked["release_count"] == 2
    return dict(global_actual=-1 if actual["rejoined"] else 1,
                global_masked=-1 if masked["rejoined"] else 1,
                global_fixed_incident=-1 if fixed["rejoined"] else 1,
                global_full_alternative=-1 if full_retain else 1 if full_release else 0,
                global_consensus=-1 if full_retain else 1 if full_release and local_flip else 0,
                global_abstain=0)


def _capture(events, graph, original, gaps, anchors, boundary_rows):
    requests = defaultdict(list)
    captured = [None] * len(gaps)
    for index, (gap, anchor, boundary) in enumerate(zip(gaps, anchors, boundary_rows, strict=True)):
        if boundary is None:
            continue
        stop = gap["stop"]
        samples = [sample for sample, _ in anchor if sample["stamp"] == stop]
        if len(samples) != 1 or boundary["stamp"] != stop or gap["end"] != "kf_rejoin":
            raise ValueError("the original boundary lost its trigger or its clock")
        sample, = samples
        switch = Switch(**gap["switch"])
        if set(sample["keys"]) != switch.keys or sample["reporter"] not in switch.keys or any(
                tuple(sorted((sample["reporter"], sample["partners"][side]))) not in getattr(switch, side)
                for side in ("release", "retain")):
            raise ValueError("an original fixed candidate edge changed")
        requests[stop].append((index, switch, gap["focal"], sample["reporter"], boundary["boundary_report"]))
    if not requests:
        return captured

    plane = Plane.from_first_batch(events[0][1].values())
    registered = _registration_shifts(original.fit)
    priors = {}
    for (stamp, batch), native in zip(events, original.native_reports, strict=True):
        frame = pickle.load(graph)
        if (stamp, tuple(sorted(batch))) != native or frame[:2] != native:
            raise ValueError("the raw, original and graph report times or keys disagree")
        points = {key: plane.project(point) for key, point in batch.items()}
        shifts = registered if stamp > FIT_STOP_S else {}
        for index, switch, focal, trigger, boundary in requests.pop(stamp, ()):
            result = score_global(switch, focal, trigger, frame, priors, points, shifts)
            old_graph = boundary["provenance"]["graph"]
            current_graph = result["graph"]
            if (not _same(current_graph["actual_margin_m2"], old_graph["factual_cached_margin_m2"])
                    or not _same(current_graph["masked_margin_m2"], old_graph["coordinate_mask_cached_margin_m2"])
                    or (result["actual"]["retain_count"] == 2) != old_graph["retain_selected"]):
                raise ValueError("C55 local rank or original rejoin selection changed")
            local_flip = _direction(current_graph["actual_margin_m2"]) == -1 and _direction(
                current_graph["masked_margin_m2"]) == 1
            current_graph["local_flip"] = local_flip
            result["directions"] = _directions(result, local_flip)
            captured[index] = dict(stamp=stamp, switch_keys=tuple(sorted(switch.keys)), global_assignment=result)
        _update_priors(priors, stamp, points)
    terminal = pickle.load(graph)
    if (requests or terminal.native_reports != original.native_reports or terminal.fit != original.fit
            or terminal.runs["kf"] != original.runs["kf"] or graph.read(1)):
        raise ValueError("the full graph trace did not reproduce the frozen original replay")
    return captured


def _anonymous(capture, labels):
    result = capture["global_assignment"]
    pairs = {arm: set(result[f"selected_{arm}"]) for arm in VIEWS}
    keys, component = (set(group) for group in (capture["switch_keys"], result["component_keys"]))

    def kind(edge):
        left, right = (labels.get(key) for key in edge)
        return "unknown" if left is None or right is None else "true" if left == right else "false"

    def counts(edges):
        value = Counter(map(kind, edges))
        return {name: value[name] for name in ("true", "false", "unknown")}

    def by_scope(edges):
        return dict(touching_switch={edge for edge in edges if not keys.isdisjoint(edge)},
                    component={edge for edge in edges if set(edge) <= component})

    def delta(before, after):
        return {f"{action}_{name}": value for action, edges in (("added", after - before), ("removed", before - after))
                for name, value in counts(edges).items()}

    scopes = {arm: by_scope(edges) for arm, edges in pairs.items()}
    anonymous = {arm: {name: counts(edges) for name, edges in scoped.items()} for arm, scoped in scopes.items()}
    anonymous.update(component_delta=delta(scopes["actual"]["component"], scopes["masked"]["component"]),
                     switch_delta=delta(scopes["actual"]["touching_switch"], scopes["masked"]["touching_switch"]),
                     fixed_mask_component_delta=delta(scopes["actual"]["component"], scopes["fixed_mask"]["component"]))
    for arm in ("masked", "fixed_mask"):
        if any(not set(edge) <= component for edge in pairs["actual"] ^ pairs[arm]):
            raise ValueError("a changed edge escaped the original affected component")
    private_fields = {"component_keys", *(f"selected_{arm}" for arm in VIEWS)}
    return {key: value for key, value in result.items() if key not in private_fields} | {"labels": anonymous}


def _join(boundary_run, frozen, scenes, archives, manifest):
    by_scene = defaultdict(list)
    for row in json.loads((boundary_run / "rows.json").read_text(encoding="utf-8")):
        by_scene[row["scene"]].append(row)
    rows = []
    for index, scene in enumerate(scenes, 1):
        alias = f"scene_{index:02d}"
        captures = _read_frozen(frozen / f"{alias}.pkl.gz")
        previous = by_scene.pop(alias, ())
        labels = read_labels(archives, manifest, scene) if any(row["boundary_report"] is not None for row in previous) else {}
        for old, capture in zip(previous, captures, strict=True):
            result = None
            if old["boundary_report"] is not None:
                if capture is None or capture["stamp"] != old["row"]["stop"]:
                    raise ValueError("a C55 legal rejoin lost its frozen clock")
                result = _anonymous(capture, labels)
                if result["actual"]["retain_count"] != 2:
                    raise ValueError("an original public full rejoin did not reproduce")
            directions = result["directions"] if result is not None else dict.fromkeys(ARMS, 0)
            current = old["decisions"]["current"]
            decisions = old["decisions"] | {arm: vote > 0 if vote else current for arm, vote in directions.items()}
            rows.append(old | dict(global_assignment=result, decisions=decisions))
    if by_scene or (len(rows), sum(row["innovation"] is not None for row in rows),
                    sum(row["global_assignment"] is not None for row in rows)) != (482, 416, 31):
        raise ValueError("the original MTAD evaluation population changed")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs=3, required=True)
    for name in ("manifest", "graph-run", "gap-run", "anchor-run", "boundary-run", "output-dir"):
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
                                _read_frozen(args.boundary_run / "frozen" / name))
        with gzip.open(frozen / name, "xb", compresslevel=1) as stream:
            pickle.dump(captures, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print("All global assignments and actions frozen. Joining anonymous evaluation.", flush=True)
    rows = _join(args.boundary_run, frozen, scenes, args.archive, args.manifest)
    summary = summarize(rows)
    _write_json(args.output_dir / "rows.json", rows)
    _write_json(args.output_dir / "summary.json", dict(clock="ideal_native_measurement_time", frozen_scenes=len(scenes),
                frozen_windows=len(rows), legal_rejoins=416, publications_unchanged=True, candidate_graph_unchanged=True,
                training_updates=0, external_dataset_reads=0, wall_seconds=perf_counter()-started, **summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
