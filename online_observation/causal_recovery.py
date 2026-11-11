"""Replay causal additions to the original geometry association."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
from itertools import product
import json
from pathlib import Path
import pickle
import shutil
from time import perf_counter

import torch

from .causal_anchor_support import AnchorVote, fit_causal_anchor
from .causal_common_motion import compare_common_motion
from .causal_recovery_metrics import pool_recovery, score_recovery
from .continuous_assignment import solve_global
from .continuous_tracker import NULL_COST, RADIUS_M, Plane
from .evaluation_data import read_labels
from .nemot_experiment import _events, _write_json
from .nemot_replay import Edge, Fragment, _registration_shifts, replay
from .registration_probe import FIT_RADIUS_M, FIT_STOP_S, estimate_translation


DEVELOPMENT = ("731", "4060", "1598")
EXPECTED_OUTSIDE = {"731": 1946, "4060": 1349, "1598": 529}
ROSTER_PATH = Path(__file__).with_name("causal_recovery_roster.json")
GATE_SQ = RADIUS_M**2
COARSE_SQ = FIT_RADIUS_M**2


@dataclass(frozen=True)
class Evidence:
    pair: Edge
    open: bool
    motion_status: str
    motion_supported: bool
    motion_unique: bool
    anchor_status: str
    witnesses: tuple[Edge, ...]
    anchor_in_gate: bool
    single_in_gate: bool


@dataclass(frozen=True)
class Frame:
    stamp: float
    updated: tuple[Fragment, ...]
    born: tuple[Fragment, ...]
    fresh: tuple[Fragment, ...]
    geometry_edges: tuple[Edge, ...]
    geometry_links: tuple[Edge, ...]
    additions: dict[str, tuple[Edge, ...]]
    evidence: tuple[Evidence, ...]


def _minus(a, b):
    return a[0] - b[0], a[1] - b[1]


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _length_sq(point):
    return point[0] ** 2 + point[1] ** 2


def _assign(costs, null):
    if not costs:
        return ()
    left = sorted({a for a, _ in costs})
    right = sorted({b for _, b in costs})
    matrix = [[costs.get((a, b)) for b in right] for a in left]
    result = solve_global(matrix, [null] * len(left), [null] * len(right))
    return tuple((left[i], right[j]) for i, j in result.pairs)


class RecoveryTrace:
    """Observe the common geometry without updating its decision state."""

    def __init__(self, fit):
        self.plane = None
        self.shifts = _registration_shifts(fit)
        self.histories = defaultdict(list)
        self.motion_cache = {}
        self.frames = []

    def observed_events(self, events):
        for stamp, positions in events:
            if self.plane is None:
                self.plane = Plane.from_first_batch(positions.values())
            for key, point in positions.items():
                self.histories[key].append((stamp, self.plane.project(point)))
            yield stamp, positions

    def _motion(self, pair, stamp):
        left, right = (self.histories[key] for key in pair)
        report_stamps = left[-1][0], right[-1][0]
        entry = self.motion_cache.get(pair)
        if entry is None or entry[0] != report_stamps:
            entry = report_stamps, compare_common_motion(left, right, stamp)
            self.motion_cache[pair] = entry
        return entry[1]

    def __call__(self, stamp, updated, born, fresh, costs, memberships):
        geometric = costs["kf"]
        geometry_links = tuple(sorted(members for _, members in memberships["kf"] if len(members) == 2))
        occupied = {key for pair in geometry_links for key in pair}
        by_source = defaultdict(list)
        for key in sorted(fresh):
            by_source[key[0]].append(key)
        shifts = self.shifts if stamp > FIT_STOP_S else {}
        points = {key: self.histories[key][-1][1] for key in fresh}
        aligned = {key: _add(point, shifts.get(key[0], (0., 0.))) for key, point in points.items()}
        raw_offsets, witness_pairs, ring = {}, [], {}
        if len(by_source) == 2:
            left, right = (by_source[source] for source in sorted(by_source))
            for a, b in product(left, right):
                pair = a, b
                raw = _minus(points[b], points[a])
                if _length_sq(raw) <= COARSE_SQ:
                    witness_pairs.append(pair)
                if pair in geometric:
                    continue
                squared = _length_sq(_minus(aligned[b], aligned[a]))
                if GATE_SQ < squared <= COARSE_SQ:
                    ring[pair] = squared
                    raw_offsets[pair] = raw

        additions = {"wide": (), "anchor": (), "motion": ()}
        evidence = []
        if ring:
            open_ring = {pair: cost for pair, cost in ring.items() if occupied.isdisjoint(pair)}
            additions["wide"] = _assign(open_ring, COARSE_SQ / 2)
            motion = {pair: self._motion(pair, stamp) for pair in ring}
            qualified = [pair for pair in open_ring if motion[pair].qualified]
            degree = Counter(key for pair in qualified for key in pair)
            additions["motion"] = tuple(sorted(pair for pair in qualified if all(degree[key] == 1 for key in pair)))
            motion_unique = set(additions["motion"])

            votes = []
            for pair in witness_pairs:
                common = self._motion(pair, stamp)
                if common.qualified:
                    offsets = tuple(_minus(b, a) for a, b in zip(common.left_points, common.right_points, strict=True))
                    votes.append(AnchorVote(*pair, offsets))
            anchor_costs = {}
            for pair in ring:
                anchor = fit_causal_anchor(votes, *pair)
                raw = raw_offsets[pair]
                corrected = _length_sq(_minus(raw, anchor.translation)) if anchor.translation is not None else None
                single = anchor.single_translation
                in_gate = corrected is not None and corrected <= GATE_SQ
                if in_gate and pair in open_ring:
                    anchor_costs[pair] = corrected
                evidence.append(Evidence(pair, pair in open_ring, motion[pair].status, motion[pair].supported,
                                         pair in motion_unique, anchor.status, anchor.witnesses, in_gate,
                                         single is not None and _length_sq(_minus(raw, single)) <= GATE_SQ))
            additions["anchor"] = _assign(anchor_costs, NULL_COST)
        self.frames.append(Frame(stamp, tuple(sorted(updated)), tuple(sorted(born)), tuple(sorted(fresh)),
                                 tuple(sorted(geometric)), geometry_links, additions, tuple(evidence)))


def capture_scene(events):
    sources = {key[0] for _, batch in events for key in batch}
    if len(sources) != 2:
        raise ValueError("the frozen Level1 experiment requires exactly two sources per scene")
    fit = estimate_translation(events)
    observer = RecoveryTrace(fit)
    replay(observer.observed_events(events), lambda context, stamp, keys: {}, fit=fit, observer=observer)
    return observer.frames, {key: fit[key] for key in ("status", "reason")}


def _members(manifest, phase):
    source = json.loads(manifest.read_text(encoding="utf-8"))
    components = source["components"]
    if phase == "development":
        members = DEVELOPMENT
        proxies = {}
    else:
        frozen = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
        members = tuple(frozen["splits"]["held"])
        proxies = frozen["proxy_stratum"]
        previous = json.loads(ROSTER_PATH.with_name("nemot_roster.json").read_text(encoding="utf-8"))
        blocked = set(previous["components"].values()) | {components[scene] for scene in DEVELOPMENT}
        if (set(members) & set(DEVELOPMENT) or any(components[scene] in blocked for scene in members)
                or any(components[scene] != frozen["components"][scene] for scene in members)):
            raise ValueError("held recovery roster violates component isolation")
    if len(set(members)) != len(members) or not set(members) <= set(source["splits"]["T_core"]):
        raise ValueError("recovery roster must contain unique T_core scenes")
    return members, components, proxies


def _run(archive, manifest, phase, destination):
    members, components, proxies = _members(manifest, phase)
    destination.mkdir(mode=0o700, exist_ok=False)
    spool = destination / "private_frozen"
    spool.mkdir(mode=0o700)
    began = perf_counter()
    scenes = {}
    for scene in members:
        frames, fit = capture_scene(_events(archive, manifest, scene))
        with gzip.open(spool / f"{scene}.pkl.gz", "xb", compresslevel=1) as stream:
            pickle.dump(frames, stream, protocol=pickle.HIGHEST_PROTOCOL)
        by_first_pool = [len({peer for peer in frame.fresh if peer[0] != key[0]})
                         for frame in frames for key in frame.born]
        sparse = 2 * sum(size <= 3 for size in by_first_pool) >= len(by_first_pool) > 0
        scenes[scene] = dict(component=components[scene], sparse=sparse, fit=fit,
                             proxy_stratum=proxies.get(scene), events=len(frames))
        print(f"Frozen {len(scenes)}/{len(members)} scene {scene}, {len(frames)} native events", flush=True)
    print("All recovery decisions frozen. Reading A-B for scoring.", flush=True)
    for scene, detail in scenes.items():
        with gzip.open(spool / f"{scene}.pkl.gz", "rb") as stream:
            frames = pickle.load(stream)
        detail["metrics"] = score_recovery(frames, read_labels(archive, manifest, scene))
    strata = {"all": list(scenes.values()), "sparse": [s for s in scenes.values() if s["sparse"]],
              "sparse_abstained": [s for s in scenes.values() if s["sparse"] and s["fit"]["status"] == "abstained"],
              "sparse_fitted": [s for s in scenes.values() if s["sparse"] and s["fit"]["status"] == "fitted"],
              "other": [s for s in scenes.values() if not s["sparse"]]}
    parity = phase != "development" or all(
        detail["metrics"]["queries"]["all"]["outside"] == EXPECTED_OUTSIDE[scene] for scene, detail in scenes.items())
    _write_json(destination / "summary.json", dict(
        phase=phase, scenes=scenes, expected_development_outside=EXPECTED_OUTSIDE if phase == "development" else None,
        development_outside_parity=parity if phase == "development" else None,
        strata={name: dict(scene_count=len(parts), metrics=pool_recovery([s["metrics"] for s in parts]))
                for name, parts in strata.items() if parts},
        clock="ideal_native_measurement_time", training_updates=0, wall_seconds=perf_counter() - began,
    ))
    if parity:
        shutil.rmtree(spool)
    return parity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs=3, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "held"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return 0 if _run(args.archive, args.manifest, args.phase, args.output_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
