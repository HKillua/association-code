"""Fit and replay the frozen component-disjoint online motion study."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import pickle
import platform
import shutil
from statistics import median
from time import perf_counter, perf_counter_ns

import numpy as np
import torch

from .continuous_factorial import _histogram_quantiles, _pool, _pool_births, _sum_dicts
from .continuous_metrics import evaluate_publications
from .evaluation_data import read_labels
from .measurements import read_archive_rows
from .nemot_inputs import motion_batch, motion_examples
from .nemot_motion import fit_motion, predict_residuals
from .nemot_replay import ARMS, publications, replay, score_native_births
from .positions import event_batches


ROSTER_PATH = Path(__file__).with_name("nemot_roster.json")
ROLES = ("train", "select", "held")
BUDGET = dict(epochs=6, batch_size=128, learning_rate=.001, weight_decay=.0001, seed=27)


def read_roster(manifest):
    roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    inventory = json.loads(manifest.read_text(encoding="utf-8"))
    eligible = set(inventory["splits"]["T_core"])
    seen = set()
    for role in ROLES:
        scenes = roster["splits"][role]
        components = {roster["components"][scene] for scene in scenes}
        if (len(scenes) != roster["counts"][role]["scenes"] or len(scenes) != len(set(scenes))
                or len(components) != roster["counts"][role]["components"]
                or not set(scenes) <= eligible or seen & components
                or any(inventory["components"][scene] != roster["components"][scene] for scene in scenes)):
            raise ValueError(f"frozen roster violates {role} scene or shared-A component isolation")
        seen.update(components)
    return roster


def _events(archive, manifest, scene):
    events = tuple(event_batches(read_archive_rows(archive, manifest, scene, required_split="T_core")))
    if not events or events[0][0] < 0:
        raise ValueError(f"scene {scene} requires a nonempty nonnegative native report clock")
    return events


def _training_batches(archive, manifest, roster):
    batches = {}
    counts = {}
    for role in ("train", "select"):
        examples = []
        counts[role] = {}
        members = roster["splits"][role]
        for index, scene in enumerate(members, start=1):
            sampled = motion_examples(scene, _events(archive, manifest, scene))
            counts[role][scene] = len(sampled)
            examples.extend(sampled)
            if index % 8 == 0 or index == len(members):
                print(f"Prepared {role} {index}/{len(members)} scenes, {len(examples)} forecast targets", flush=True)
        batches[role] = motion_batch(examples)
    return batches, counts


def _errors(batch, models):
    result = {arm: [] for arm in ARMS}
    for start in range(0, len(batch), BUDGET["batch_size"]):
        block = batch.take(slice(start, start + BUDGET["batch_size"]))
        target_offset = block.target - block.center
        result["kf"].extend(torch.linalg.vector_norm(target_offset, dim=1).tolist())
        for arm, model in models.items():
            residual = predict_residuals(model, block).double()
            result[arm].extend(torch.linalg.vector_norm(residual - target_offset, dim=1).tolist())
    return result


def _error_summary(errors):
    return {arm: dict(n=len(values), mean_m=float(np.mean(values)) if values else None,
                      median_m=median(values) if values else None,
                      p95_m=sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None)
            for arm, values in errors.items()}


def _latencies(samples):
    if not samples:
        return dict(n=0, p50_ms=None, p95_ms=None, p99_ms=None, max_ms=None)
    ordered = sorted(samples)
    quantiles = {f"p{int(100 * p)}_ms": ordered[math.ceil(p * len(ordered)) - 1] / 1e6
                 for p in (.50, .95, .99)}
    return dict(n=len(ordered), **quantiles, max_ms=ordered[-1] / 1e6)


class LearnedPrediction:
    """Compute both learned arms on identical causal request features."""

    def __init__(self, models):
        self.models = models
        self.timing_ns = {arm: array("Q") for arm in models}
        self.diagnostics = Counter()

    def __call__(self, context, stamp, keys):
        result = {arm: {} for arm in self.models}
        eligible, examples = [], []
        self.diagnostics["stale_endpoint_queries"] += len(keys)
        for key in keys:
            example = context.example(key, stamp)
            if len(example.history) >= 2 and example.horizon > 0:
                eligible.append(key)
                examples.append(example)
                self.diagnostics["with_any_neighbor"] += any(item.neighbors for item in example.history)
                self.diagnostics["with_anchor_neighbor"] += bool(example.history[-1].neighbors)
        self.diagnostics["learned_endpoint_queries"] += len(eligible)
        if not eligible:
            return result
        batch = motion_batch(examples)
        for arm, model in self.models.items():
            began = perf_counter_ns()
            values = predict_residuals(model, batch).tolist()
            self.timing_ns[arm].append(perf_counter_ns() - began)
            result[arm].update(zip(eligible, map(tuple, values), strict=True))
        return result


def _write_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2)
        stream.write("\n")


def _train(archive, manifest, roster, destination):
    batches, counts = _training_batches(archive, manifest, roster)
    models, histories = {}, {}
    for arm in ARMS[1:]:
        print(f"Training {arm} on {len(batches['train'])} targets", flush=True)
        model, history = fit_motion(batches["train"], batches["select"], use_neighbors=arm == "neighbors", **BUDGET)
        models[arm] = model
        histories[arm] = history
        with (destination / f"{arm}.pt").open("xb") as stream:
            torch.save(model.state_dict(), stream)
        print(f"Selected {arm} epoch {history['selected_epoch']}, MAE {history['selected_mae_m']:.3f} m", flush=True)
    _write_json(destination / "training.json", dict(budget=BUDGET, scenes=counts, histories=histories,
                                                    selected_forecasts=_error_summary(_errors(batches["select"], models))))
    return models


def _pool_native(parts):
    pooled = _pool_births(parts)
    delays = Counter()
    for part in parts:
        delays.update(part["delay_milliseconds_histogram"])
    pooled["delay_milliseconds_histogram"] = dict(delays)
    pooled["delay_seconds_quantiles_ms_rounding"] = _histogram_quantiles(delays, (.50, .95), scale=1000)
    return pooled


def _pool_candidates(parts):
    strata = ("by_first_fresh_peer_count", "by_first_gate_peer_count")
    totals = _sum_dicts({key: value for key, value in part.items() if key not in strata} for part in parts)
    for name in strata:
        totals[name] = {group: _sum_dicts(part[name][group] for part in parts) for group in ("0-3", "4+")}
    return totals


def _timing_summary(timing):
    return dict(event_all_arms=_latencies(timing["event_service_ns"]),
                paired_model_requests=_latencies(timing["two_model_prediction_ns"]),
                cost_and_assignment={arm: _latencies(values)
                                     for arm, values in timing["arm_cost_and_assignment_ns"].items()})


def _combine(scenes, forecasts):
    combined = {}
    for arm in ARMS:
        scored = [scene["arms"][arm] for scene in scenes]
        results = _pool(scored)
        births = [score["native_births"] for score in scored]
        results["native_births"] = _pool_native(births)
        results["native_births"]["by_opportunity"] = {
            group: _pool_native([birth["by_opportunity"][group] for birth in births])
            for group in births[0]["by_opportunity"]}
        results["native_candidates"] = _pool_candidates([score["native_candidates"] for score in scored])
        results["first_choice_comparisons"] = {
            base: _sum_dicts(score["first_choice_comparisons"][base] for score in scored)
            for base in scored[0]["first_choice_comparisons"]}
        results["forecast"] = _error_summary({arm: [x for scene in scenes for x in forecasts[scene["scene"]][arm]]})[arm]
        combined[arm] = results
    return combined


def _run(archive, manifest, destination):
    roster = read_roster(manifest)
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    began = perf_counter()
    models = _train(archive, manifest, roster, destination)
    spool = destination / "private_frozen"
    spool.mkdir(mode=0o700)
    scenes, forecasts = {}, {}
    prediction_ns = {arm: array("Q") for arm in models}
    for scene in roster["splits"]["held"]:
        events = _events(archive, manifest, scene)
        predictor = LearnedPrediction(models)
        snapshot = replay(events, predictor)
        with gzip.open(spool / f"{scene}.pkl.gz", "xb", compresslevel=1) as stream:
            pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
        peers = snapshot.diagnostics["first_peers_le3"]
        births = snapshot.diagnostics["first_total"]
        forecasts[scene] = _errors(motion_batch(motion_examples(scene, events)), models)
        scenes[scene] = dict(scene=scene, component=roster["components"][scene], sparse=2 * peers >= births > 0,
                             native_events=len(events), stop=math.ceil(events[-1][0]),
                             fit={key: snapshot.fit[key] for key in ("status", "reason")},
                             forecast=_error_summary(forecasts[scene]), geometry=snapshot.diagnostics,
                             replay_timing=_timing_summary(snapshot.timing), learned_requests=dict(predictor.diagnostics),
                             model_timing={arm: _latencies(values) for arm, values in predictor.timing_ns.items()},
                             arms={})
        for arm, values in predictor.timing_ns.items():
            prediction_ns[arm].extend(values)
        print(f"Frozen held {len(scenes)}/{len(roster['splits']['held'])} scene {scene}, {len(events)} native events",
              flush=True)

    print("All held predictions frozen. Reading A-B only for scoring.", flush=True)
    for scene, detail in scenes.items():
        with gzip.open(spool / f"{scene}.pkl.gz", "rb") as stream:
            snapshot = pickle.load(stream)
        labels = read_labels(archive, manifest, scene)
        native = score_native_births(snapshot, labels)
        for arm in ARMS:
            scores = evaluate_publications(snapshot.native_reports, publications(snapshot, arm, detail["stop"]), labels)
            scores.update(diagnostics={}, native_births=native[arm]["births"],
                          native_candidates=native[arm]["candidates"],
                          first_choice_comparisons=native[arm].get("first_choice_comparisons", {}))
            detail["arms"][arm] = scores
    strata = {"all": list(scenes.values()), "sparse": [x for x in scenes.values() if x["sparse"]],
              "other": [x for x in scenes.values() if not x["sparse"]]}
    _write_json(destination / "summary.json", dict(
        arm_names=ARMS, scenes=scenes, strata={name: dict(scene_count=len(parts),
            aggregate=_combine(parts, forecasts) if parts else None) for name, parts in strata.items()},
        timing_model_only={arm: _latencies(values) for arm, values in prediction_ns.items()},
        clock="ideal_native_measurement_time", scoring_grid_seconds=1, runtime=dict(
            python=platform.python_version(), torch=str(torch.__version__), threads=torch.get_num_threads()),
        wall_seconds=perf_counter() - began,
    ))
    shutil.rmtree(spool)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", nargs=3, type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    _run(args.archive, args.manifest, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
