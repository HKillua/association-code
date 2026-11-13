"""Audit association margins and error episodes from the frozen NEMOT replay."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import pickle
import shutil
from time import perf_counter

import torch

from .continuous_metrics import evaluate_publications
from .evaluation_data import read_labels
from .nemot_edge_diagnostics import EdgeAuditRecorder, merge_edge_audits, score_edge_audit
from .nemot_episodes import pool_episodes, score_episodes
from .nemot_experiment import LearnedPrediction, _events, _write_json, read_roster
from .nemot_motion import NEMOTMotion
from .nemot_replay import ARMS, publications, replay, score_native_births


def _load_frozen(artifacts, scenes):
    reference = json.loads((artifacts / "summary.json").read_text(encoding="utf-8"))
    if tuple(reference["arm_names"]) != ARMS or set(reference["scenes"]) != set(scenes):
        raise ValueError("the prior replay must contain the original 28 scenes and three motion arms")
    models = {}
    for arm in ARMS[1:]:
        model = NEMOTMotion(use_neighbors=arm == "neighbors")
        state = torch.load(artifacts / f"{arm}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False)
        models[arm] = model.eval()
    return reference, models


def _matches_json(observed, expected):
    # The stored summary has string keys for integral histogram bins.
    return json.loads(json.dumps(observed)) == expected


def _score_scene(snapshot, frames, labels, stop, previous):
    episodes = score_episodes(snapshot, labels, stop=stop)
    native = score_native_births(snapshot, labels)
    parity = {"geometry": snapshot.diagnostics == previous["geometry"]}
    for arm in ARMS:
        result = evaluate_publications(snapshot.native_reports, publications(snapshot, arm, stop), labels)
        if episodes["arms"][arm]["pairs"] != result["pairs"]["cross_source"]:
            raise RuntimeError(f"{arm} error episodes do not reconcile with cross-source pair-seconds")
        old = previous["arms"][arm]
        parity[arm] = dict(
            pair_seconds=result["pairs"] == old["pairs"], identity_and_coverage=result["counts"] == old["counts"],
            native_births=_matches_json(native[arm]["births"], old["native_births"]),
            native_candidates=_matches_json(native[arm]["candidates"], old["native_candidates"]),
            first_choice_comparisons=native[arm].get("first_choice_comparisons", {}) == old["first_choice_comparisons"],
        )
    return dict(parity=parity, edges=score_edge_audit(frames, labels), episodes=episodes)


def _run(archive, manifest, artifacts, destination):
    roster = read_roster(manifest)
    members = roster["splits"]["held"]
    reference, models = _load_frozen(artifacts, members)
    destination.mkdir(mode=0o700, exist_ok=False)
    spool = destination / "private_frozen"
    spool.mkdir(mode=0o700)
    began = perf_counter()
    scenes = {}
    for scene in members:
        events = _events(archive, manifest, scene)
        recorder = EdgeAuditRecorder()
        snapshot = replay(events, LearnedPrediction(models), observer=recorder)
        with gzip.open(spool / f"{scene}.pkl.gz", "xb", compresslevel=1) as stream:
            pickle.dump((snapshot, recorder.frames), stream, protocol=pickle.HIGHEST_PROTOCOL)
        births = snapshot.diagnostics["first_total"]
        sparse = births > 0 and 2 * snapshot.diagnostics["first_peers_le3"] >= births
        scenes[scene] = dict(component=roster["components"][scene], sparse=sparse,
                             native_events=len(events), stop=math.ceil(events[-1][0]),
                             fit={key: snapshot.fit[key] for key in ("status", "reason")})
        print(f"Frozen {len(scenes)}/{len(members)} scene {scene}, {len(events)} native events", flush=True)

    print("All association traces frozen. Reading A-B for post-replay scoring.", flush=True)
    for scene, detail in scenes.items():
        with gzip.open(spool / f"{scene}.pkl.gz", "rb") as stream:
            snapshot, frames = pickle.load(stream)
        labels = read_labels(archive, manifest, scene)
        previous = reference["scenes"][scene]
        detail.update(_score_scene(snapshot, frames, labels, detail["stop"], previous))
        detail["parity"]["population"] = all(detail[key] == previous[key]
                                              for key in ("sparse", "native_events", "stop", "component", "fit"))

    strata = {"all": list(scenes.values()), "sparse": [s for s in scenes.values() if s["sparse"]],
              "other": [s for s in scenes.values() if not s["sparse"]],
              "sparse_fitted": [s for s in scenes.values() if s["sparse"] and s["fit"]["status"] == "fitted"],
              "sparse_abstained": [s for s in scenes.values() if s["sparse"] and s["fit"]["status"] == "abstained"]}
    aggregate = {name: dict(scene_count=len(parts), edges=merge_edge_audits([s["edges"] for s in parts]),
                            episodes=pool_episodes([s["episodes"] for s in parts]))
                 for name, parts in strata.items() if parts}
    parity = all(detail["parity"]["geometry"] and detail["parity"]["population"]
                 and all(all(detail["parity"][arm].values()) for arm in ARMS) for detail in scenes.values())
    _write_json(destination / "summary.json", dict(
        arm_names=ARMS, scenes=scenes, strata=aggregate, prior_replay_parity=parity,
        checkpoint_source=str(artifacts), training_updates=0, clock="ideal_native_measurement_time",
        error_episode_grid_seconds=1, wall_seconds=perf_counter() - began,
    ))
    if parity:
        shutil.rmtree(spool)
    return parity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs=3, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return 0 if _run(args.archive, args.manifest, args.frozen_run, args.output_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
