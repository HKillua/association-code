"""Audit causal neighbor evidence for sparse cross-source first reports."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import gzip
from itertools import combinations
from math import cos, hypot, radians
from pathlib import Path
import pickle
import shutil
from time import perf_counter

import torch

from .causal_common_motion import SEGMENT_SECONDS, compare_common_motion
from .causal_recovery import capture_scene
from .causal_recovery_metrics import COHORTS, pool_recovery, score_recovery
from .evaluation_data import read_labels
from .expansion_confirmation import ConfirmationState, ExpansionRecorder, _members, _strata
from .neighbor_audit import pool_neighbor_audit, score_neighbor_audit
from .neighbor_witness import NeighborEvidence, NeighborWitnessIndex
from .nemot_experiment import _events, _write_json
from .nemot_replay import Edge


ROSTER = Path(__file__).with_name("neighbor_witness_roster.json")
BASELINES = ("geometry", "wide", "witness_confirm")
NEIGHBOR_COHORTS = (*COHORTS, "first_sparse", "outside_sparse")
MIN_PEER_DISPLACEMENT_M = 120
MAX_PEER_SPEED_DIFFERENCE_M_S = 2
MIN_PEER_DIRECTION_COSINE = cos(radians(15))


@dataclass(frozen=True)
class ProbeFrame:
    ring: dict[Edge, NeighborEvidence]
    writes: dict[Edge, NeighborEvidence]
    parallel: dict[Edge, bool | None]


def _parallel_motion(first, second, stamp):
    common = compare_common_motion(first, second, stamp)
    if not common.supported:
        return None
    left, right = common.left_points, common.right_points
    ax, ay = left[-1][0] - left[0][0], left[-1][1] - left[0][1]
    bx, by = right[-1][0] - right[0][0], right[-1][1] - right[0][1]
    speed_a, speed_b = hypot(ax, ay), hypot(bx, by)
    if min(speed_a, speed_b) < MIN_PEER_DISPLACEMENT_M:
        return False
    return (abs(speed_a - speed_b) <= MAX_PEER_SPEED_DIFFERENCE_M_S * (2 * SEGMENT_SECONDS)
            and ax * bx + ay * by >= MIN_PEER_DIRECTION_COSINE * speed_a * speed_b)


class NeighborRecorder:
    def __init__(self):
        self.legacy = ExpansionRecorder()
        self.frames: list[ProbeFrame] = []

    def __call__(self, frame, histories, ring, motion):
        new_samples = self.legacy(frame, histories, ring, motion)
        opened = [edge.pair for edge in self.legacy.frames[-1].edges]
        index = NeighborWitnessIndex(frame.stamp, frame.fresh, histories) if opened or new_samples else None
        ring_evidence = {pair: index.lookup(pair) for pair in opened}
        write_evidence = {pair: index.lookup(pair) for pair, _ in new_samples}

        born = set(frame.born)
        by_birth = defaultdict(list)
        for left, right in opened:
            if left in born:
                by_birth[left].append(right)
            if right in born:
                by_birth[right].append(left)
        peer_pairs = {pair for peers in by_birth.values() for pair in combinations(sorted(peers), 2)}
        parallel = {pair: _parallel_motion(histories[pair[0]], histories[pair[1]], frame.stamp)
                    for pair in sorted(peer_pairs)}
        self.frames.append(ProbeFrame(ring_evidence, write_evidence, parallel))


def _run(archive, manifest, phase, destination):
    members, components, proxies = _members(manifest, phase, ROSTER)
    destination.mkdir(mode=0o700, exist_ok=False)
    spool = destination / "private_frozen"
    spool.mkdir(mode=0o700)
    began = perf_counter()
    scenes = {}
    references = ("geometry", "wide", "witness_confirm")
    for scene in members:
        recorder = NeighborRecorder()
        frames, fit = capture_scene(_events(archive, manifest, scene), expansion_observer=recorder)
        confirmation = ConfirmationState()
        expansions = recorder.legacy.frames
        frames = [replace(frame, additions={**frame.additions, **confirmation.decide(expansion)})
                  for frame, expansion in zip(frames, expansions, strict=True)]
        with gzip.open(spool / f"{scene}.pkl.gz", "xb", compresslevel=1) as stream:
            pickle.dump((frames, expansions, recorder.frames), stream, protocol=pickle.HIGHEST_PROTOCOL)
        first_pools = [sum(peer[0] != key[0] for peer in frame.fresh) for frame in frames for key in frame.born]
        sparse = 2 * sum(size <= 3 for size in first_pools) >= len(first_pools) > 0
        scenes[scene] = dict(component=components[scene], sparse=sparse, fit=fit, proxy_stratum=proxies[scene],
                             events=len(frames))
        print(f"Frozen {len(scenes)}/{len(members)} scene {scene}, {len(frames)} native events", flush=True)
    print("All neighbor evidence and decisions frozen. Reading A-B for scoring.", flush=True)
    for scene, detail in scenes.items():
        with gzip.open(spool / f"{scene}.pkl.gz", "rb") as stream:
            frames, expansions, probes = pickle.load(stream)
        labels = read_labels(archive, manifest, scene)
        detail["recovery"] = score_recovery(frames, labels, arms=BASELINES, references=references,
                                           cohorts=NEIGHBOR_COHORTS)
        detail["audit"] = score_neighbor_audit(frames, expansions, probes, labels, arms=BASELINES)
    _write_json(destination / "summary.json", dict(
        phase=phase, scenes=scenes, clock="ideal_native_measurement_time", training_updates=0,
        strata={name: dict(scene_count=len(parts), component_count=len({s["component"] for s in parts}),
                           recovery=pool_recovery((s["recovery"] for s in parts), arms=BASELINES,
                                                  references=references, cohorts=NEIGHBOR_COHORTS),
                           audit=pool_neighbor_audit(s["audit"] for s in parts))
                for name, parts in _strata(scenes).items() if parts}, wall_seconds=perf_counter() - began,
    ))
    shutil.rmtree(spool)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs=3, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "held"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    _run(args.archive, args.manifest, args.phase, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
