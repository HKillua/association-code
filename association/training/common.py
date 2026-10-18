"""Shared normalization, scoring and optimizer steps for the active trainers."""
from __future__ import annotations
from functools import wraps
import numpy as np
import torch

def fit_normalizer(prepared_scenes):
    """Population moments over training observed grid points, never padding."""
    prepared_scenes = list(prepared_scenes)
    if any(scene.get('input_adapter') == 'native_time_v1' for scene in prepared_scenes):
        from association.data.timed import fit_timed_normalizer
        return fit_timed_normalizer(prepared_scenes)
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
    if scene.get('input_adapter') == 'native_time_v1':
        from association.data.timed import pack_timed_scene
        return pack_timed_scene(scene, normalizer)
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


def stable_evaluation_parameters(function):
    """Use one parameter-grad state for validation across frozen/joint stages.

    Some inference kernels choose different FP32 paths from requires_grad even
    inside inference_mode. A threshold chosen on a boundary score must not
    drift merely because its checkpoint was restored in a different stage.
    Restore each flag afterwards so validation cannot change training freezes.
    """
    @wraps(function)
    def evaluate(model, *args, **kwargs):
        flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
        try:
            model.requires_grad_(False)
            return function(model, *args, **kwargs)
        finally:
            for parameter, enabled in flags:
                parameter.requires_grad_(enabled)
    return evaluate


@stable_evaluation_parameters
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
