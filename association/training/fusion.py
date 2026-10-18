"""One effective-batch TFM/joint update; no sampling, data access, or fit launch.

The caller supplies already selected pairs and positive-pair supervision in
their original order. Microbatches change memory use, not loss denominators.
This is training machinery, not a choice of native-data or selection policy.
"""
from __future__ import annotations

import math

import torch


def fusion_training_parameters(model, fusion, objective, *, stage):
    """Set explicit stage trainability/modes and return optimizer parameters.

    Build a fresh stage optimizer from this return value. Frozen TFM training
    freezes the entire association model; joint training updates all modules,
    including the four uncertainty weights. No parameters are updated here.
    """
    if stage not in ("TFM_FROZEN", "JOINT"):
        raise ValueError("stage must be TFM_FROZEN or JOINT")
    if (model.config.length, model.config.d_model) != (fusion.config.length, fusion.config.d_model):
        raise ValueError("TCM and TFM length/dimension must agree")
    joint = stage == "JOINT"
    model.requires_grad_(joint)
    model.set_completion_trainable(joint, include_embedding=True)
    model.train(joint)
    fusion.requires_grad_(True).train()
    objective.requires_grad_(True).train()
    for module in (model, fusion, objective):
        module.zero_grad(set_to_none=True)
    return [p for module in (model, fusion, objective) for p in module.parameters() if p.requires_grad]


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def fusion_training_step(model, fusion, objective, optimizer, *, stage, values,
                         observed, pairs, labels, target, valid, pair_microbatch,
                         track_microbatch, completion_weight, gradient_clip, check=None):
    """Update once with TFM loss, or TAM + weighted TCM + TFM losses.

    ``values`` are normalized observation-only tracks. ``target/valid`` address
    positive rows of ``pairs`` in order, retaining repeated draws; they must
    come from the explicit positive-pair target assembler. No reference states
    enter completion or association. A missing effective-batch feature fails
    without resampling or partially stepping an optimizer.

    Joint pairs share one TCM forward with TAM/TFM within each chunk. Shared
    endpoints across chunks are recomputed, freeing each graph after backward.
    With dropout enabled this changes the random-mask realization compared to
    one large batch; exact-gradient comparisons therefore use dropout=0.
    ``track_microbatch`` limits only auxiliary TCM loss forwards. The pair path
    completes up to ``2 * pair_microbatch`` endpoints at once. Optional ``check``
    lets the run's existing deadline abort between chunks and before a step.

    TFM_FROZEN lazily reuses deterministic eval completions within this update.
    Detached entries live on CPU, bounded by this batch's unique endpoints;
    the device only receives the current chunk. No cache survives the call.
    Fusion chunks/dropout draws and the effective-batch objective are unchanged.
    Missing-only completion batches can differ in floating-point rounding from
    recomputing full chunks; this does not promise cross-layout bitwise parity.
    """
    _positive_integer(pair_microbatch, "pair_microbatch")
    _positive_integer(track_microbatch, "track_microbatch")
    for value, name in ((completion_weight, "completion_weight"), (gradient_clip, "gradient_clip")):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if gradient_clip == 0:
        raise ValueError("gradient_clip must be positive")
    if stage == "TFM_FROZEN" and completion_weight != 0:
        raise ValueError("TFM_FROZEN requires completion_weight=0")
    parameters = fusion_training_parameters(model, fusion, objective, stage=stage)
    optimizer_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if (len(optimizer_parameters) != len(parameters)
            or {id(p) for p in optimizer_parameters} != {id(p) for p in parameters}):
        raise ValueError("optimizer must contain exactly this stage's model, fusion and loss parameters")
    model._validate_inputs(values, observed)
    anchor = next(model.parameters())
    if values.dtype != anchor.dtype or values.device != anchor.device:
        raise ValueError("observations must use the model dtype and device")
    if next(fusion.parameters()).device != values.device or next(fusion.parameters()).dtype != values.dtype:
        raise ValueError("TFM must use the observation dtype and device")
    if objective.log_variance.device != values.device or objective.log_variance.dtype != values.dtype:
        raise ValueError("FusionLoss must use the observation dtype and device")
    pairs = torch.as_tensor(pairs, device=values.device)
    labels = torch.as_tensor(labels, device=values.device)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype not in (torch.int32, torch.int64):
        raise ValueError("pairs must be an integer [E,2] array")
    if not len(pairs) or (pairs < 0).any() or (pairs >= len(values)).any() or (pairs[:, 0] == pairs[:, 1]).any():
        raise ValueError("a nonempty batch of valid non-self pairs is required")
    if labels.shape != (len(pairs),) or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must be binary and correspond to each pair")
    positive = labels == 1
    if stage == "TFM_FROZEN" and not positive.all():
        raise ValueError("TFM_FROZEN receives only true positive pairs")
    positive_count = int(positive.sum())
    shape = (positive_count, model.config.length, 4)
    if target.shape != shape or valid.shape != shape or valid.dtype != torch.bool:
        raise ValueError("targets require [positive_pairs,length,4] states and boolean masks")
    if (target.device != values.device or valid.device != values.device or target.dtype != values.dtype):
        raise ValueError("targets/masks must use the observation device and targets its dtype")
    counts = valid.sum(dim=(0, 1))
    if (counts == 0).any():
        raise ValueError("no valid supervision for at least one effective-batch feature; do not resample")
    if not torch.isfinite(target[valid]).all() or (target[..., 2][valid[..., 2]] < 0).any():
        raise ValueError("valid targets must be finite with nonnegative native speed")
    pair_targets = positive.long().cumsum(0) - 1
    endpoints = torch.unique(pairs)
    transition_counts = (observed[endpoints].sum(1) - 1).clamp_min(0) * (2 * model.config.d_model)
    total_transitions = int(transition_counts.sum())
    report = {"stage": stage, "pair_count": len(pairs), "positive_pairs": positive_count,
              "unique_endpoints": len(endpoints), "completion_target_elements": total_transitions,
              "fusion_valid_counts": counts.tolist(), "pair_microbatch": pair_microbatch,
              "track_microbatch": track_microbatch, "association_loss": 0.0,
              "completion_loss": 0.0, "fusion_loss": 0.0,
              "fusion_component_losses": dict.fromkeys(objective.feature_names, 0.0),
              "microbatches": 0}
    frozen_cache = {} if stage == "TFM_FROZEN" else None
    if frozen_cache is not None:
        report["completion_cache"] = dict(scope="step_local_cpu", hits=0, misses=0,
            forward_calls=0, max_forward_tracks=0, peak_entries=0, peak_bytes=0)
    optimizer.zero_grad(set_to_none=True)
    try:
        if stage == "JOINT" and completion_weight and total_transitions:
            for start in range(0, len(endpoints), track_microbatch):
                if check:
                    check()
                nodes = endpoints[start:start + track_microbatch]
                elements = int(transition_counts[start:start + track_microbatch].sum())
                if not elements:
                    continue  # No temporal target exists, not a rejected sampled pair.
                contribution = model.completion_loss(values[nodes], observed[nodes]) * (elements / total_transitions)
                (completion_weight * contribution).backward()
                report["completion_loss"] += float(contribution.detach())
        regularized = False
        for start in range(0, len(pairs), pair_microbatch):
            if check:
                check()
            end = min(start + pair_microbatch, len(pairs))
            nodes, inverse = torch.unique(pairs[start:end], return_inverse=True)
            local_pairs = inverse.reshape(-1, 2)
            if stage == "TFM_FROZEN":
                keys = nodes.tolist()
                missing = [key for key in keys if key not in frozen_cache]
                stats = report["completion_cache"]
                stats["hits"] += len(keys) - len(missing)
                stats["misses"] += len(missing)
                if missing:
                    with torch.no_grad():
                        fresh = model.complete(values[missing], observed[missing])
                    for index, key in enumerate(missing):
                        frozen_cache[key] = fresh[index].detach().to(device="cpu", copy=True)
                    stats["forward_calls"] += 1
                    stats["max_forward_tracks"] = max(stats["max_forward_tracks"], len(missing))
                    stats["peak_entries"] = len(frozen_cache)
                    stats["peak_bytes"] = len(frozen_cache) * fresh[0].numel() * fresh.element_size()
                    del fresh  # Do not retain device storage via cached tensor views.
                completed = torch.stack([frozen_cache[key] for key in keys]).to(values.device)
            else:
                completed = model.complete(values[nodes], observed[nodes])
            loss = None
            if stage == "JOINT":
                embedded = model.tam_encoder(completed).flatten(1)
                scores = model.score_pairs(embedded, local_pairs)
                loss = model.association_loss(scores, labels[start:end]) * ((end - start) / len(pairs))
                report["association_loss"] += float(loss.detach())
            local_positive = positive[start:end]
            if local_positive.any():
                rows = pair_targets[start:end][local_positive]
                states = fusion.forward_pairs(completed, local_pairs[local_positive])["states"]
                result = objective(states, target[rows], valid[rows], normalization_counts=counts,
                                   regularizer_weight=0.0 if regularized else 1.0)
                regularized = True
                loss = result["total"] if loss is None else loss + result["total"]
                report["fusion_loss"] += float(result["total"].detach())
                for name, contribution in result["losses"].items():
                    report["fusion_component_losses"][name] += float(contribution.detach())
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("missing or nonfinite microbatch objective")
            loss.backward()
            report["microbatches"] += 1
        if check:
            check()
        norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
        optimizer.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        if frozen_cache is not None:
            frozen_cache.clear()
    report["gradient_norm"] = float(norm)
    report["loss"] = (report["association_loss"] + completion_weight * report["completion_loss"]
                      + report["fusion_loss"])
    return report
