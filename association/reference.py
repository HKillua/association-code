"""Bounded TCM/TAM reference training, immutable checkpoints and raw caches.

Only T_core (A0) or T_core+T_head (A_full) can update parameters. V_select
selects an epoch and threshold; this module never opens T_test measurements.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import validate_manifest
from .config import validate_config, validate_manifest_config
from .dataset import load_split
from .metrics import evaluate_scenes, select_threshold
from .models import AssociationModel, ModelConfig
from . import runtime


def fit_normalizer(prepared_scenes):
    """Population moments over training observed grid points, never padding."""
    total = None
    squared = None
    count = 0
    scene_ids = []
    for scene in prepared_scenes:
        values = np.asarray(scene["values"], dtype=np.float64)
        observed = np.asarray(scene["observed"], dtype=bool)
        if values.ndim != 3 or observed.shape != values.shape[:2]:
            raise ValueError("invalid normalized-scene input shape")
        valid = values[observed]
        if not np.isfinite(valid).all():
            raise ValueError("nonfinite training observations")
        if total is None:
            total = np.zeros(values.shape[-1], dtype=np.float64)
            squared = np.zeros_like(total)
        total += valid.sum(axis=0)
        squared += (valid * valid).sum(axis=0)
        count += len(valid)
        scene_ids.append(str(scene["scene_id"]))
    if not count:
        raise ValueError("training split contains no observed model inputs")
    mean = total / count
    variance = np.maximum(squared / count - mean * mean, 0.0)
    std = np.sqrt(variance)
    constant = std < 1e-6
    std[constant] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist(), "observed_point_count": count,
            "training_scene_ids": scene_ids, "constant_features": np.flatnonzero(constant).tolist(),
            "policy": "training observed registered points; population variance; std<1e-6 replaced by 1"}


def normalize_scene(scene, normalizer):
    values = np.asarray(scene["values"], dtype=np.float32)
    observed = np.asarray(scene["observed"], dtype=bool)
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    if mean.shape != (values.shape[-1],) or std.shape != mean.shape:
        raise ValueError("normalizer feature dimension mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("normalizer must have finite means and positive finite scales")
    # Missing values never participate in arithmetic or feature statistics.
    normalized = np.zeros_like(values)
    normalized[observed] = (values[observed] - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("normalization produced nonfinite values")
    return normalized, observed


def score_reference(model, normalizer, prepared_scenes, device, batch_tracks=64,
                    pair_batch_size=1024, check=None):
    """Raw symmetric CE logits / negative distances, with bounded pair memory."""
    if batch_tracks < 1 or pair_batch_size < 1:
        raise ValueError("score batch sizes must be positive")
    device = torch.device(device)
    model.eval()
    records = []
    with torch.inference_mode():
        for scene in prepared_scenes:
            if check:
                check()
            values, observed = normalize_scene(scene, normalizer)
            embeddings = []
            for offset in range(0, len(values), batch_tracks):
                if check:
                    check()
                encoded = model.encode_tracks(
                    torch.as_tensor(values[offset:offset + batch_tracks], device=device),
                    torch.as_tensor(observed[offset:offset + batch_tracks], device=device),
                )
                if not torch.isfinite(encoded).all():
                    raise FloatingPointError("nonfinite reference embeddings")
                embeddings.append(encoded)
            embedded = torch.cat(embeddings) if embeddings else torch.empty(
                0, model.embedding_dim, device=device)
            pairs = np.asarray(scene["pairs"], dtype=np.int64).reshape(-1, 2)
            chunks = []
            for offset in range(0, len(pairs), pair_batch_size):
                if check:
                    check()
                scores = model.score_pairs(embedded, pairs[offset:offset + pair_batch_size])
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("nonfinite reference scores")
                chunks.append(scores.cpu().numpy())
            scores = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
            record = {key: scene[key] for key in ["scene_id", "component_id", "node_ids",
                                                   "identities", "sources", "intervals", "labels"]}
            record.update(scores=scores, pairs=pairs)
            records.append(record)
    return records


def _binding(manifest, config, role, branch, seed):
    return {"manifest_sha256": manifest["manifest_sha256"], "config_hash": runtime.object_hash(config),
            "code_fingerprint": runtime.code_fingerprint(), "role": role, "branch": branch, "seed": int(seed)}


def cache_binding(summary, split):
    return {key: summary[key] for key in ["manifest_sha256", "config_hash", "code_fingerprint",
                                         "role", "branch", "seed", "checkpoint_sha256"]} | {"split": split}


def _check_configuration(config, role, branch):
    config = validate_config(config)
    if role not in ("A0", "A_full") or branch not in ("ce", "contrastive"):
        raise ValueError("reference requires role A0/A_full and branch ce/contrastive")
    model_config = ModelConfig(**config["model"])
    ref = config["reference"]
    return model_config, ref


def _set_epoch_lr(optimizer, ref, epoch, total_epochs):
    """Constant LR for first warmup_epochs, then half-cosine to zero.

    'warmup' retains the existing config key; this is NOT a linear warmup.
    Each independently optimized stage starts its own recorded schedule.
    """
    plateau = min(int(ref.get("warmup_epochs", 20)), total_epochs)
    if total_epochs == 1 or epoch < plateau:
        factor = 1.0
    else:
        progress = (epoch - plateau + 1) / max(total_epochs - plateau, 1)
        factor = 0.5 * (1 + math.cos(math.pi * progress))
    lr = float(ref["learning_rate"]) * factor
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _optimizer(parameters, ref):
    return torch.optim.Adam(parameters, lr=ref["learning_rate"], betas=(0.85, 0.99),
                            eps=1e-6, weight_decay=ref.get("weight_decay", 0.0))


def _update(loss, optimizer, parameters, clip):
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm.detach())


def _save_checkpoint(path, model, model_config, normalizer, binding, selection):
    payload = {"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
               "model_config": asdict(model_config), "normalizer": normalizer,
               "binding": binding, "selection": selection}
    temp = Path(path).with_suffix(".pt.tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _coverage(scenes):
    return [{"scene_id": scene["scene_id"], "node_count": len(scene["values"]),
             "pair_count": len(scene["pairs"]), "audit": scene["audit"]} for scene in scenes]


def train_reference(archive_path, manifest, config, role, branch, seed, device, output):
    """Train one bounded reference fit; output must not exist, including failed runs."""
    model_config, ref = _check_configuration(config, role, branch)
    if ref.get("schedule", "legacy") == "staged_updates_v2":
        if ref["purpose"] != "fit":
            raise ValueError("probe configuration requires probe-training; it is not an eligible reference")
        from .staged import train_staged_reference
        return train_staged_reference(archive_path, manifest, config, role, branch, seed, device, output)
    validate_manifest_config(manifest, config)
    device = torch.device(device)
    binding = _binding(manifest, config, role, branch, seed)
    run = runtime.Run(output, {**binding, "kind": "association_reference", "config": config,
                              "provenance": runtime.provenance(device)}, ref["max_hours"])
    history = []
    phase_seconds = {}
    try:
        runtime.seed_everything(seed)
        rng = np.random.default_rng(seed)
        start = time.monotonic()
        core = load_split(archive_path, manifest, "T_core", model_config.length)
        train_scenes = list(core)
        head = None
        if role == "A_full":
            head = load_split(archive_path, manifest, "T_head", model_config.length)
            train_scenes += head
        validation = load_split(archive_path, manifest, "V_select", model_config.length)
        if not train_scenes or not any(len(scene["pairs"]) for scene in train_scenes):
            raise ValueError("reference training requires eligible training association pairs")
        if not validation or not any(len(scene["pairs"]) for scene in validation):
            raise ValueError("reference selection requires eligible validation association pairs")
        normalizer = fit_normalizer(train_scenes)
        normalized = [normalize_scene(scene, normalizer) for scene in train_scenes]
        train_track_indices = [(s, t) for s, scene in enumerate(train_scenes)
                               for t in range(len(scene["values"]))]
        if not any(int(mask.sum()) >= 2 for _, masks in normalized for mask in masks):
            raise ValueError("TCM pretraining has no observed temporal transitions")
        run.check()
        phase_seconds["preparation"] = time.monotonic() - start
        model = AssociationModel(model_config, branch).to(device)
        model.input_embedding.requires_grad_(False)
        parameters = list(model.completion_parameters())
        optimizer = _optimizer(parameters, ref)
        clip = float(ref.get("gradient_clip", 1.0))
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")
        start = time.monotonic()
        for epoch in range(ref["tcm_epochs"]):
            run.check()
            model.train()
            lr = _set_epoch_lr(optimizer, ref, epoch, ref["tcm_epochs"])
            order = rng.permutation(len(train_track_indices))
            losses, steps = [], 0
            for offset in range(0, len(order), ref["batch_tracks"]):
                run.check()
                selected = [train_track_indices[i] for i in order[offset:offset + ref["batch_tracks"]]]
                x = torch.as_tensor(np.stack([normalized[s][0][t] for s, t in selected]), device=device)
                mask = torch.as_tensor(np.stack([normalized[s][1][t] for s, t in selected]), device=device)
                loss = model.completion_loss(x, mask)
                value, norm = _update(loss, optimizer, parameters, clip)
                losses.append(value)
                steps += 1
            history.append({"phase": "TCM", "epoch": epoch + 1, "loss": float(np.mean(losses)),
                            "steps": steps, "learning_rate": lr})
            runtime.write_json(run.path / "history.json", history)
        phase_seconds["tcm"] = time.monotonic() - start
        joint = bool(ref.get("joint_completion", True))
        model.set_completion_trainable(joint, include_embedding=True)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = _optimizer(parameters, ref)
        best = None
        checkpoint = run.path / "checkpoint.pt"
        start = time.monotonic()
        for epoch in range(ref["tam_epochs"]):
            run.check()
            model.train()
            lr = _set_epoch_lr(optimizer, ref, epoch, ref["tam_epochs"])
            losses, association_losses, steps = [], [], 0
            for scene_index in rng.permutation(len(train_scenes)):
                scene = train_scenes[scene_index]
                pairs = np.asarray(scene["pairs"], dtype=np.int64).reshape(-1, 2)
                pair_order = rng.permutation(len(pairs))
                values, masks = normalized[scene_index]
                for offset in range(0, len(pairs), ref["pair_batch_size"]):
                    run.check()
                    selected = pair_order[offset:offset + ref["pair_batch_size"]]
                    nodes, inverse = np.unique(pairs[selected], return_inverse=True)
                    local_pairs = inverse.reshape(-1, 2)
                    x = torch.as_tensor(values[nodes], device=device)
                    observed = torch.as_tensor(masks[nodes], device=device)
                    # Chunk the encoder without detaching: all association gradients
                    # still reach the exact shared encoder/completion parameters.
                    embeddings = torch.cat([model.encode_tracks(x[k:k + ref["batch_tracks"]],
                                                                observed[k:k + ref["batch_tracks"]])
                                            for k in range(0, len(nodes), ref["batch_tracks"])])
                    scores = model.score_pairs(embeddings, local_pairs)
                    labels = torch.as_tensor(np.asarray(scene["labels"])[selected], device=device)
                    association_loss = model.association_loss(scores, labels)
                    loss = association_loss
                    if joint and ref.get("completion_weight", 0.1):
                        loss = loss + ref.get("completion_weight", 0.1) * model.completion_loss(x, observed)
                    value, norm = _update(loss, optimizer, parameters, clip)
                    losses.append(value)
                    association_losses.append(float(association_loss.detach()))
                    steps += 1
            records = score_reference(model, normalizer, validation, device, ref["batch_tracks"], check=run.check)
            selection = select_threshold(records)
            metric = selection["macro_f1"]
            # Keep the earlier epoch on a validation-score tie. Thresholds within
            # an epoch independently use the higher-threshold tie rule.
            if best is None or metric > best["validation_macro_f1"] + 1e-12:
                best = {"epoch": epoch + 1, "threshold": selection["threshold"],
                        "validation_macro_f1": metric, "validation": evaluate_scenes(records, selection["threshold"])}
                _save_checkpoint(checkpoint, model, model_config, normalizer, binding, best)
            history.append({"phase": "TAM", "epoch": epoch + 1, "loss": float(np.mean(losses)),
                            "association_loss": float(np.mean(association_losses)), "steps": steps,
                            "learning_rate": lr, "selection": selection})
            runtime.write_json(run.path / "history.json", history)
            run.check()
        phase_seconds["tam_and_validation"] = time.monotonic() - start
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        checkpoint_sha = runtime.digest(checkpoint)
        summary = {**binding, "checkpoint_sha256": checkpoint_sha, "model_config": asdict(model_config),
                   "threshold": best["threshold"], "validation": best["validation"], "selected_epoch": best["epoch"],
                   "normalizer": normalizer, "training_scene_ids": [scene["scene_id"] for scene in train_scenes],
                   "training_splits": ["T_core"] + (["T_head"] if role == "A_full" else []),
                   "selection_split": "V_select", "test_opened": False, "history_file": "history.json",
                   "joint_completion": joint, "completion_weight": ref.get("completion_weight", 0.1) if joint else 0.0,
                   "lr_schedule": "per-stage constant first warmup_epochs, then half-cosine to zero; not linear warmup",
                   "reference_status": "documented independent association reference; convergence not automatically established"}
        start = time.monotonic()
        if head is None:
            head = load_split(archive_path, manifest, "T_head", model_config.length)
        for split, scenes in (("T_head", head), ("V_select", validation)):
            run.check()
            records = score_reference(model, normalizer, scenes, device, ref["batch_tracks"], check=run.check)
            runtime.save_records(run.path / split, records, cache_binding(summary, split))
            run.check()
        phase_seconds["cache"] = time.monotonic() - start
        summary["phase_seconds"] = phase_seconds
        summary["cache_bindings"] = {split: cache_binding(summary, split) for split in ("T_head", "V_select")}
        summary["coverage"] = {"training": _coverage(train_scenes), "V_select": _coverage(validation),
                               "T_head": _coverage(head)}
        runtime.write_json(run.path / "summary.json", summary)
        run.check()
        run.finish(summary_sha256=runtime.digest(run.path / "summary.json"), checkpoint_sha256=checkpoint_sha,
                   phase_seconds=phase_seconds)
        return summary
    except BaseException as error:
        run.finish("FAILED", failure_type=type(error).__name__, failure=str(error), phase_seconds=phase_seconds)
        raise


def load_reference(path, manifest, config, role, branch, seed, device):
    """Reject incomplete, modified, stale-code/config/manifest references."""
    path = Path(path)
    model_config, ref = _check_configuration(config, role, branch)
    if ref.get("schedule", "legacy") == "staged_updates_v2" and ref["purpose"] != "fit":
        raise ValueError("probe configuration cannot load an eligible reference")
    validate_manifest_config(manifest, config)
    run = runtime.read_json(path / "run.json")
    if run.get("status") != "COMPLETED":
        raise ValueError("reference is not a completed run")
    if run.get("summary_sha256") != runtime.digest(path / "summary.json"):
        raise ValueError("reference summary checksum mismatch")
    summary = runtime.read_json(path / "summary.json")
    expected = _binding(manifest, config, role, branch, seed)
    if any(summary.get(key) != value or run.get(key) != value for key, value in expected.items()):
        raise ValueError("reference role/branch/seed, manifest, config, or code fingerprint mismatch")
    checksum = runtime.digest(path / "checkpoint.pt")
    if summary.get("checkpoint_sha256") != checksum or run.get("checkpoint_sha256") != checksum:
        raise ValueError("reference checkpoint checksum mismatch")
    payload = torch.load(path / "checkpoint.pt", map_location=torch.device(device), weights_only=True)
    if payload["binding"] != expected or payload["model_config"] != asdict(model_config):
        raise ValueError("checkpoint metadata differs from reference provenance")
    if payload["normalizer"] != summary["normalizer"] or payload["selection"]["threshold"] != summary["threshold"]:
        raise ValueError("checkpoint selection or normalizer differs from reference summary")
    model = AssociationModel(model_config, branch).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("checkpoint contains nonfinite parameters")
    return model, payload["normalizer"], summary
