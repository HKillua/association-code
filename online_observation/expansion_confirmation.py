"""Capture fixed-ring evidence before reading MTAD identity labels."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import gzip
import json
from math import hypot, sqrt
from pathlib import Path
import pickle
import shutil
from time import perf_counter

import torch

from .causal_recovery import capture_scene
from .causal_recovery_metrics import ARMS, pool_recovery, score_recovery
from .evaluation_data import read_labels
from .expansion_evidence import HistoricalGeometryWitnesses, common_offset_evidence, current_common_offset
from .expansion_evidence_metrics import pool_expansion_evidence, score_expansion_evidence
from .nemot_experiment import _events, _write_json
from .nemot_replay import Edge, Fragment


ROSTER = Path(__file__).with_name("expansion_confirmation_roster.json")
CONFIRMATION_ARMS = ("dual_report", "witness_veto", "witness_confirm")
HELD_ARMS = ("geometry", "wide", *CONFIRMATION_ARMS)
CONFIRMATION_LIMIT_SECONDS = 60
WITNESS_RESIDUAL_METRES = 400


@dataclass(frozen=True)
class ExpansionEdge:
    pair: Edge
    wide: bool
    features: dict[str, float | int | None]
    witnesses: tuple[Edge, ...]
    witness_offsets: tuple[tuple[Edge, tuple[float, float]], ...] = ()


@dataclass(frozen=True)
class ExpansionFrame:
    stamp: float
    updated: tuple[Fragment, ...]
    fresh: tuple[Fragment, ...]
    edges: tuple[ExpansionEdge, ...]


@dataclass(frozen=True)
class Nomination:
    stamp: float
    report_stamps: tuple[float, float]
    raw_offset: tuple[float, float]


class ExpansionRecorder:
    def __init__(self):
        self.witnesses = HistoricalGeometryWitnesses()
        self.nominations: dict[Edge, Nomination] = {}
        self.frames: list[ExpansionFrame] = []

    def __call__(self, frame, histories, ring, motion):
        wide = set(frame.additions["wide"])
        opened = [item.pair for item in frame.evidence if item.open]
        degrees = Counter(key for pair in opened for key in pair)
        nominations = {}
        edges = []
        for pair in opened:
            left, right = (histories[key] for key in pair)
            current_offset = current_common_offset(left, right, frame.stamp)
            witness = self.witnesses.lookup(pair, frame.stamp, current_offset)
            own_history = common_offset_evidence(motion[pair])
            repeat_both, episode_age, repeat_drift = 0, None, None
            if pair in wide:
                (left_stamp, (lx, ly)), (right_stamp, (rx, ry)) = left[-1], right[-1]
                raw_offset = (rx - lx, ry - ly)
                origin = self.nominations.get(pair)
                if origin is None:
                    origin = Nomination(frame.stamp, (left_stamp, right_stamp), raw_offset)
                nominations[pair] = origin
                episode_age = frame.stamp - origin.stamp
                repeat_both = int(left_stamp > origin.report_stamps[0] and right_stamp > origin.report_stamps[1])
                if repeat_both:
                    repeat_drift = hypot(raw_offset[0] - origin.raw_offset[0], raw_offset[1] - origin.raw_offset[1])
            edges.append(ExpansionEdge(pair, pair in wide, dict(
                range_m=sqrt(ring[pair]), past_drift_m=own_history.past_drift_m,
                held_residual_m=own_history.held_residual_m, witness_count=witness.count,
                witness_residual_m=witness.residual_m, repeat_offset_drift_m=repeat_drift,
                repeat_both=repeat_both, episode_age_s=episode_age,
                singleton=int(all(degrees[key] == 1 for key in pair)),
            ), witness.witnesses, witness.offsets))
        self.nominations = nominations
        self.frames.append(ExpansionFrame(frame.stamp, frame.updated, frame.fresh, tuple(edges)))
        return self.witnesses.observe(frame.stamp, frame.geometry_links, histories)


class ConfirmationState:
    """Publish only eligible W nominations and keep acceptance within each run."""

    def __init__(self):
        self.confirmed = {arm: set() for arm in CONFIRMATION_ARMS}

    def decide(self, frame):
        current = {arm: set() for arm in CONFIRMATION_ARMS}
        for edge in frame.edges:
            if not edge.wide:
                continue
            features = edge.features
            residual = features["witness_residual_m"]
            repeat = bool(features["repeat_both"])
            compatible = residual is not None and residual <= WITNESS_RESIDUAL_METRES
            allowed = residual is None or compatible
            within_limit = features["episode_age_s"] <= CONFIRMATION_LIMIT_SECONDS
            eligibility = {
                "dual_report": repeat,
                "witness_veto": repeat and allowed,
                "witness_confirm": compatible or (repeat and allowed),
            }
            for arm, eligible in eligibility.items():
                if edge.pair in self.confirmed[arm] or (within_limit and eligible):
                    current[arm].add(edge.pair)
        self.confirmed = current
        return {arm: tuple(sorted(pairs)) for arm, pairs in current.items()}


def _members(manifest, phase, roster_path=ROSTER):
    inventory = json.loads(manifest.read_text(encoding="utf-8"))
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    if roster["source_split"] != "T_core" or phase not in roster["splits"]:
        raise ValueError(f"no frozen T_core {phase} roster")
    components = inventory["components"]
    dev = set(roster["splits"]["development"])
    members = tuple(roster["splits"][phase])
    selected_components = {components[scene] for scene in members}
    counts = roster["counts"][phase]
    if (len(set(members)) != len(members) or not set(members) <= set(inventory["splits"]["T_core"])
            or len(members) != counts["scenes"] or len(selected_components) != counts["components"]
            or any(components[scene] != roster["components"][scene] for scene in members)):
        raise ValueError("invalid frozen expansion roster")
    if phase == "held":
        exclusions = roster["exclusions"]
        if (selected_components & ({components[scene] for scene in dev}
                                   | set(exclusions["blocked_nemot_and_diagnostic_components"]))
                or set(members) & set(exclusions["historical_scene_ids"])):
            raise ValueError("held expansion roster violates component or scene isolation")
    return members, components, roster["proxy_stratum"]


def _strata(scenes):
    parts = list(scenes.values())
    return {
        "all": parts,
        "sparse": [s for s in parts if s["sparse"]],
        "sparse_abstained": [s for s in parts if s["sparse"] and s["fit"]["status"] == "abstained"],
        "other": [s for s in parts if not s["sparse"]],
    }


def _run(archive, manifest, phase, destination):
    members, components, proxies = _members(manifest, phase)
    held = phase == "held"
    arms = HELD_ARMS if held else ARMS
    references = ("geometry", "wide", "dual_report", "witness_veto") if held else ("geometry",)
    destination.mkdir(mode=0o700, exist_ok=False)
    spool = destination / "private_frozen"
    spool.mkdir(mode=0o700)
    began = perf_counter()
    scenes = {}
    for scene in members:
        recorder = ExpansionRecorder()
        frames, fit = capture_scene(_events(archive, manifest, scene), expansion_observer=recorder)
        if held:
            confirmation = ConfirmationState()
            frames = [replace(frame, additions={**frame.additions, **confirmation.decide(expansion)})
                      for frame, expansion in zip(frames, recorder.frames, strict=True)]
        with gzip.open(spool / f"{scene}.pkl.gz", "xb", compresslevel=1) as stream:
            pickle.dump((frames, recorder.frames), stream, protocol=pickle.HIGHEST_PROTOCOL)
        first_pools = [sum(peer[0] != key[0] for peer in frame.fresh) for frame in frames for key in frame.born]
        sparse = 2 * sum(size <= 3 for size in first_pools) >= len(first_pools) > 0
        scenes[scene] = dict(component=components[scene], sparse=sparse, fit=fit,
                             proxy_stratum=proxies[scene], events=len(frames))
        print(f"Frozen {len(scenes)}/{len(members)} scene {scene}, {len(frames)} native events", flush=True)
    print("All expansion evidence frozen. Reading A-B for scoring.", flush=True)
    for scene, detail in scenes.items():
        with gzip.open(spool / f"{scene}.pkl.gz", "rb") as stream:
            frames, expansion = pickle.load(stream)
        labels = read_labels(archive, manifest, scene)
        detail["recovery"] = score_recovery(frames, labels, arms=arms, references=references)
        selections = ({arm: frame.additions[arm] for arm in HELD_ARMS[1:]} for frame in frames) if held else None
        detail["evidence"] = score_expansion_evidence(expansion, labels, selections=selections)
    _write_json(destination / "summary.json", dict(
        phase=phase, scenes=scenes, clock="ideal_native_measurement_time", training_updates=0,
        strata={name: dict(scene_count=len(parts), component_count=len({s["component"] for s in parts}),
                           recovery=pool_recovery((s["recovery"] for s in parts), arms=arms, references=references),
                           evidence=pool_expansion_evidence(s["evidence"] for s in parts))
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
