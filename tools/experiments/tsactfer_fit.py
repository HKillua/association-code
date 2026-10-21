"""One bounded TSA-cTFER offline fit; T_core training, full V_select selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from association.utils.config import load_config, validate_manifest_config
from association.data.archive import MTADArchive
from association.data.legacy_dataset import load_split_from_archive
from association.evaluation.association import evaluate_scenes, select_threshold
from association.training.common import _update
from association.utils import runtime
from baselines.tsactfer import TSAConfig, TSActfer, symmetric_info_nce
from baselines.tsactfer_data import (build_positive_pools, fit_normalizer,
                                    pack_scene, sample_batch)


def validate_tsa_config(config):
    if config["scope"] != "TSA-cTFER-offline-independent-adaptation-v1":
        raise ValueError("Unknown TSA adaptation scope")
    model = TSAConfig(**config["model"])
    training, data = config["training"], config["data"]
    for name in ("updates", "warmup_updates", "pair_batch_size", "validate_every_updates",
                 "inference_batch_tracks"):
        value = training[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if (training["updates"] < 2 or training["updates"] % 2 or
            training["warmup_updates"] >= training["updates"] or training["pair_batch_size"] < 2):
        raise ValueError("Need even updates, shorter warmup and at least two pairs")
    if training["validate_every_updates"] % 2:
        raise ValueError("Validate only after both task heads have been updated")
    for name in ("learning_rate", "temperature", "gradient_clip", "max_hours"):
        value = training[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{name} must be positive and finite")
    if training["max_hours"] > 6:
        raise ValueError("This entry is limited to the approved six-hour TSA allocation")
    if (isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or
            config["seed"] < 0):
        raise ValueError("seed must be a nonnegative integer")
    for name in ("time_bin_seconds", "min_crop_fraction"):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"data.{name} must be positive and finite")
    if data["min_crop_fraction"] > 1:
        raise ValueError("min_crop_fraction cannot exceed one")
    return model


def learning_rate(update, training):
    warmup, total = training["warmup_updates"], training["updates"]
    if update <= warmup:
        multiplier = update / warmup
    else:
        # Half cosine on (warmup, total]; final update has zero LR, explicitly.
        multiplier = .5 * (1 + math.cos(math.pi * (update - warmup) / (total - warmup)))
    return training["learning_rate"] * multiplier


def _inputs(batch, device):
    return tuple(torch.as_tensor(batch[name], device=device) for name in
                 ("positions", "time_ids", "observed"))


def paper_scope_coverage(scenes):
    counts = {task: {"candidates": 0, "positives": 0, "paper_scope_candidates": 0,
                     "paper_scope_positives": 0} for task in ("itsa", "mstsa")}
    for scene in scenes:
        pairs = np.asarray(scene["pairs"])
        left, right = pairs.T
        intervals = np.asarray(scene["intervals"])
        starts, ends = intervals[:, 0], intervals[:, 1]
        same = np.asarray(scene["sources"])[left] == np.asarray(scene["sources"])[right]
        gap = np.maximum(starts[left], starts[right]) - np.minimum(ends[left], ends[right])
        duration = ends - starts
        positive = np.asarray(scene["labels"], dtype=bool)
        itsa = (gap >= 0) & (gap < 1200) & (duration[left] > 120) & (duration[right] > 120)
        mstsa = -gap > 120
        for task, mask, eligible in (("itsa", same, itsa), ("mstsa", ~same, mstsa)):
            for key, value in (("candidates", mask), ("positives", mask & positive),
                               ("paper_scope_candidates", mask & eligible),
                               ("paper_scope_positives", mask & eligible & positive)):
                counts[task][key] += int(value.sum())
    return counts


@torch.no_grad()
def score_scenes(model, scenes, device, batch_tracks, check=lambda: None):
    """Encode each node once; route every original pair without filtering."""
    model.eval()
    records = []
    for scene in scenes:
        check()
        n = len(scene["positions"])
        outputs = {task: [] for task in ("itsa", "mstsa")}
        for start in range(0, n, batch_tracks):
            check()
            batch = {name: scene[name][start:start + batch_tracks] for name in
                     ("positions", "time_ids", "observed")}
            pooled = model.encode(*_inputs(batch, device))
            for task in outputs:
                outputs[task].append(F.normalize(model.project(pooled, task), dim=-1).cpu().numpy())
        vectors = {task: np.concatenate(parts) if parts else np.empty((0, model.config.d_model), np.float32)
                   for task, parts in outputs.items()}
        pairs = np.asarray(scene["pairs"])
        scores = np.empty(len(pairs), np.float32)
        same = np.asarray(scene["sources"])[pairs[:, 0]] == np.asarray(scene["sources"])[pairs[:, 1]]
        for task, mask in (("itsa", same), ("mstsa", ~same)):
            selected = pairs[mask]
            scores[mask] = np.einsum("ij,ij->i", vectors[task][selected[:, 0]], vectors[task][selected[:, 1]])
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite TSA pair score")
        records.append({key: scene[key] for key in ("scene_id", "component_id", "node_ids", "labels",
                                                    "pairs", "sources", "identities", "intervals")})
        records[-1]["scores"] = scores
    check()
    return records


def train(archive_path, manifest, protocol_config, config, device, output):
    model_config = validate_tsa_config(config)
    validate_manifest_config(manifest, protocol_config)
    if model_config.length < protocol_config["model"]["length"]:
        raise ValueError("TSA slots cannot truncate the common registered input")
    device = torch.device(device)
    training = config["training"]
    binding = {"scope": config["scope"], "seed": config["seed"]}
    run = runtime.Run(output, {"kind": config["scope"], "binding": binding,
                              "config": config, "provenance": runtime.provenance(device)}, training["max_hours"])
    history, phases = [], {}
    try:
        runtime.seed_everything(config["seed"])
        rng = np.random.default_rng(config["seed"])
        started = time.monotonic()
        with MTADArchive(archive_path) as archive:
            core = load_split_from_archive(archive, manifest, "T_core", protocol_config["model"]["length"])
            run.check()
            validation = load_split_from_archive(archive, manifest, "V_select", protocol_config["model"]["length"])
            members = list(archive.members_read)
        run.check()
        normalizer = fit_normalizer(core)
        pools, pool_audit = build_positive_pools(core)
        packing = {"length": model_config.length, "time_bins": model_config.time_bins,
                   "time_bin_seconds": config["data"]["time_bin_seconds"]}
        train_scenes = [pack_scene(scene, normalizer, **packing) for scene in core]
        valid_scenes = [pack_scene(scene, normalizer, **packing) for scene in validation]
        audit = {"training_pools": pool_audit, "normalizer": normalizer,
                 "training_splits": ["T_core"], "selection_split": "V_select", "test_opened": False,
                 "training_scene_ids": [s["scene_id"] for s in core],
                 "validation_scene_ids": [s["scene_id"] for s in validation],
                 "V_select_scope_coverage": paper_scope_coverage(validation), "archive_members_read": members,
                 "packing": packing, "registered_length": protocol_config["model"]["length"],
                 "packing_audits": {"T_core": [s["packing_audit"] for s in train_scenes],
                                    "V_select": [s["packing_audit"] for s in valid_scenes]}}
        runtime.write_json(run.path / "data_audit.json", audit)
        for task in ("ITSA", "MSTSA"):
            if len(pools[task]) < training["pair_batch_size"]:
                raise ValueError(f"{task} has {len(pools[task])} eligible identities; "
                                 f"requires {training['pair_batch_size']} without batch reduction")
        run.check()
        phases["data"] = time.monotonic() - started
        model = TSActfer(model_config).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                     betas=(.9, .999), eps=1e-6, weight_decay=0.)
        best = None
        checkpoint = run.path / "checkpoint.pt"
        phases.update(training=0., validation=0.)
        for update in range(1, training["updates"] + 1):
            run.check()
            start = time.monotonic()
            task = "ITSA" if update % 2 else "MSTSA"
            batch = sample_batch(train_scenes, pools, task, training["pair_batch_size"], rng,
                                 min_crop_fraction=config["data"]["min_crop_fraction"])
            lr = learning_rate(update, training)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            embeddings = model(*_inputs(batch, device), task=task.lower())
            loss = symmetric_info_nce(embeddings, training["temperature"])
            loss_value, grad = _update(loss, optimizer, model.parameters(), training["gradient_clip"])
            elapsed = time.monotonic() - start
            phases["training"] += elapsed
            row = {"update": update, "task": task, "loss": loss_value, "gradient_norm": grad,
                   "learning_rate": lr, "pair_count": len(batch["identities"]),
                   "seconds": elapsed, "augmentation": batch["augmentation"]}
            history.append(row)
            run.check()
            if update % training["validate_every_updates"] == 0 or update == training["updates"]:
                start = time.monotonic()
                records = score_scenes(model, valid_scenes, device, training["inference_batch_tracks"], run.check)
                selection = select_threshold(records)
                row["selection"] = selection
                if best is None or selection["macro_f1"] > best["validation"]["macro_f1"] + 1e-12:
                    best = {"update": update, "threshold": selection["threshold"],
                            "validation": evaluate_scenes(records, selection["threshold"])}
                    payload = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                               "model_config": asdict(model_config), "normalizer": normalizer,
                               "binding": binding, "selection": best,
                               "config": config, "protocol_config": protocol_config}
                    temporary = checkpoint.with_suffix(".pt.tmp")
                    torch.save(payload, temporary)
                    temporary.replace(checkpoint)
                phases["validation"] += time.monotonic() - start
                print(f"TSA update={update} loss={loss_value:.6f} V_macro_F1={selection['macro_f1']:.6f}", flush=True)
            if update % 10 == 0 or "selection" in row:
                runtime.write_json(run.path / "history.json", history)
                runtime.write_json(run.path / "progress.json", {"completed_updates": update,
                                   "selected": best, "phase_seconds": phases})
        run.check()
        start = time.monotonic()
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if any(payload["binding"].get(key) != value for key, value in binding.items()):
            raise ValueError("TSA checkpoint binding mismatch")
        if (payload["model_config"] != asdict(model_config) or payload["config"] != config
                or payload["protocol_config"] != protocol_config or payload["normalizer"] != normalizer):
            raise ValueError("TSA checkpoint configuration/normalizer mismatch")
        model.load_state_dict(payload["state_dict"])
        records = score_scenes(model, valid_scenes, device, training["inference_batch_tracks"], run.check)
        measured = evaluate_scenes(records, best["threshold"])
        if measured != best["validation"]:
            raise ValueError("Reloaded checkpoint does not reproduce selected validation metrics")
        cache_binding = {**binding, "split": "V_select"}
        runtime.save_records(run.path / "V_select", records, cache_binding)
        phases["cache"] = time.monotonic() - start
        summary = {**binding,
                   "selected_update": best["update"], "threshold": best["threshold"], "validation": measured,
                   "training_splits": ["T_core"], "selection_split": "V_select", "test_opened": False,
                   "completed_updates": training["updates"], "updates_per_task": training["updates"] // 2,
                   "nonzero_lr_updates": sum(row["learning_rate"] > 0 for row in history),
                   "parameter_count": parameter_count,
                   "cache_binding": cache_binding, "normalizer": normalizer, "phase_seconds": phases,
                   "V_select_scope_coverage": audit["V_select_scope_coverage"],
                   "interpretation": "Exploratory validation-selected offline adaptation; not original online "
                                     "TSA-cTFER, not held-out test, no convergence or novelty claim.",
                   "conformance_document": "docs/tsactfer_adaptation.md"}
        runtime.write_json(run.path / "summary.json", summary)
        runtime.write_json(run.path / "history.json", history)
        run.check()
        run.finish(artifacts=["summary.json", "checkpoint.pt", "history.json", "data_audit.json", "V_select/index.json"],
                   phase_seconds=phases)
        return summary
    except BaseException as error:
        runtime.write_json(run.path / "history.json", history)
        run.finish("FAILED", failure_type=type(error).__name__, failure=str(error), phase_seconds=phases)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol-config", default="configs/first-fit-v2.json")
    parser.add_argument("--tsa-config", default="configs/tsactfer-first-fit.json")
    parser.add_argument("--device", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    def terminated(signum, frame):
        raise TimeoutError("SIGTERM received; retain partial artifacts without claiming completion")

    previous = signal.signal(signal.SIGTERM, terminated)
    try:
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
        train(args.archive, runtime.read_json(args.manifest), load_config(args.protocol_config),
              runtime.read_json(args.tsa_config), runtime.device_from_string(args.device), args.output)
        print(f"COMPLETED TSA offline fit {Path(args.output).resolve()}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"FAILED {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
