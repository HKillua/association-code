"""Train frozen-score contextual controls using T_head and V_select only.

No reference encoder is trained here. All arms share one monotone calibrator,
the same scene-balanced losses, and validation-only threshold/epoch selection.
Artifacts are immutable completed runs; partial/failed runs cannot be loaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .context import (HEAD_DIMENSIONS, FeatureScaler, MonotoneCalibrator,
                      ResidualHead, context_features, fixed_rule, head_features,
                      scene_pool_features)
from .metrics import _records, evaluate_scenes, select_threshold
from .runtime import (Run, digest, jsonable, object_hash, provenance, read_json,
                      seed_everything, semantic_binding, write_json)


HEAD_KINDS = ("B", "B_shuffle", "PairMLP", "EndpointMarginal", "ScenePool")


def _seed(seed, *parts):
    payload = ":".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**32)


def _configuration(config):
    source = config.get("heads", config)
    result = {"epochs": int(source.get("epochs", 40)),
              "learning_rate": float(source.get("learning_rate", .001)),
              "weight_decay": float(source.get("weight_decay", .0001)),
              "deltas": [float(delta) for delta in source.get("deltas", [.5, 1., 2.])],
              "max_hours": float(source.get("max_hours", 3)),
              "calibration_steps": int(source.get("calibration_steps", 200)),
              "calibration_learning_rate": .01}
    if not 1 <= result["epochs"] <= 40 or result["calibration_steps"] < 1:
        raise ValueError("head epochs must be 1..40 and calibration_steps positive")
    if not 0 < result["max_hours"] <= 3:
        raise ValueError("head run cap must be positive and at most 3 hours")
    if (not np.isfinite(result["learning_rate"]) or result["learning_rate"] <= 0 or
            not np.isfinite(result["weight_decay"]) or result["weight_decay"] < 0):
        raise ValueError("head optimizer settings are invalid")
    if (not result["deltas"] or not np.isfinite(result["deltas"]).all() or
            min(result["deltas"]) <= 0 or len(set(result["deltas"])) != len(result["deltas"])):
        raise ValueError("deltas must be distinct positive finite values")
    return result


def _validate_split_records(head_records, val_records):
    head, val = _records(head_records), _records(val_records)
    for split, records in (("T_head", head), ("V_select", val)):
        if not records or not any(len(record["scores"]) for record in records):
            raise ValueError(f"{split} has no eligible pairs")
        if any("component_id" not in original or original["component_id"] is None
               for original in (head_records if split == "T_head" else val_records)):
            raise ValueError("explicit shared-identity components are required")
    if {record["scene_id"] for record in head} & {record["scene_id"] for record in val}:
        raise ValueError("T_head and V_select share scene IDs")
    if {record["component_id"] for record in head} & {record["component_id"] for record in val}:
        raise ValueError("T_head and V_select share identity components")
    labels = np.concatenate([record["labels"] for record in head])
    if not labels.any() or labels.all():
        raise ValueError("T_head needs both positive and negative pairs before fitting heads")
    return head, val


def _copy_with_scores(record, scores):
    result = dict(record)
    result["scores"] = np.asarray(scores, dtype=np.float64)
    if result["scores"].shape != np.asarray(record["scores"]).shape or not np.isfinite(result["scores"]).all():
        raise ValueError("predicted scores must preserve shape and be finite")
    return result


def _calibrate(calibrator, records, device):
    calibrated = []
    with torch.no_grad():
        for record in records:
            raw = torch.as_tensor(record["scores"], dtype=torch.float32, device=device)
            calibrated.append(_copy_with_scores(record, calibrator(raw).cpu().numpy()))
    return calibrated


def _graph(record):
    n = len(record["sources"])
    matrix, mask = np.zeros((n, n)), np.zeros((n, n), dtype=bool)
    i, j = record["pairs"].T
    matrix[i, j] = matrix[j, i] = record["scores"]
    mask[i, j] = mask[j, i] = True
    return matrix, mask


def _features(record, kind, seed, split, epoch=None):
    matrix, mask = _graph(record)
    is_shuffle = kind == "B_shuffle"
    context = context_features(matrix, mask, shuffle=is_shuffle,
                               seed=_seed(seed, record["scene_id"], split, "fixed" if epoch is None else epoch))
    # Context returns canonical lexicographic pairs; cache records may use another
    # valid ordering. Restore their original order instead of relabeling features.
    positions = {tuple(pair): i for i, pair in enumerate(context["pairs"])}
    order = np.array([positions[tuple(pair)] for pair in record["pairs"]], dtype=np.int64)
    pooled = scene_pool_features(matrix, mask) if kind == "ScenePool" else None
    selected = head_features(kind, context, pooled)[order]
    return selected, context["counts"][order]


def _fixed_predictions(records, delta):
    predictions = []
    for record in records:
        matrix, mask = _graph(record)
        context = context_features(matrix, mask)
        residuals = fixed_rule(context, delta)
        positions = {tuple(pair): i for i, pair in enumerate(context["pairs"])}
        correction = np.array([residuals[positions[tuple(pair)]] for pair in record["pairs"]])
        predictions.append(_copy_with_scores(record, record["scores"] + correction))
    return predictions


def _head_predictions(head, scaler, records, kind, seed, split, device, packs=None):
    predictions = []
    head.eval()
    with torch.no_grad():
        for index, record in enumerate(records):
            features, counts = _features(record, kind, seed, split) if packs is None else packs[index]
            tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            residual = head(scaler(tensor), counts).cpu().numpy()
            predictions.append(_copy_with_scores(record, record["scores"] + residual))
    return predictions


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _check_gradients(module):
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in module.parameters()):
        raise ValueError("nonfinite gradient; preserving failed run")


def train_heads(head_records, val_records, config, seed, device, output, binding):
    """Fit on raw A0-cache records; return ``selection[arm]`` (no test inputs).

    T_head losses weight each eligible scene equally and each of its edges
    equally. Calibration takes one aggregate optimizer step per configured step;
    residual heads take one optimizer step per scene per epoch. All arms/deltas
    receive the same full epoch budget, with independent freshly initialized
    heads. T_head shuffle varies by epoch; V_select and inference are fixed.
    """
    configuration = _configuration(config)
    device = torch.device(device)
    seed = int(seed)
    binding = jsonable(binding)
    run = Run(output, {"kind": "context-heads", "schema": 1, "binding": binding,
                       "seed": seed, "config": configuration,
                       "config_sha256": object_hash(configuration),
                       "provenance": provenance(device)}, max_hours=configuration["max_hours"])
    try:
        head_records, val_records = list(head_records), list(val_records)
        head_records, val_records = _validate_split_records(head_records, val_records)
        active = [record for record in head_records if len(record["scores"])]
        seed_everything(seed)
        calibrator = MonotoneCalibrator().to(device)
        optimizer = torch.optim.Adam(calibrator.parameters(), lr=.01)
        calibration_curve = []
        for step in range(configuration["calibration_steps"]):
            run.check()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for record in active:
                run.check()
                raw = torch.as_tensor(record["scores"], dtype=torch.float32, device=device)
                labels = torch.as_tensor(record["labels"], dtype=torch.float32, device=device)
                loss = F.binary_cross_entropy_with_logits(calibrator(raw), labels)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite calibration loss")
                (loss / len(active)).backward()
                total_loss += loss.detach().item() / len(active)
            _check_gradients(calibrator)
            optimizer.step()
            calibration_curve.append({"step": step + 1, "scene_mean_bce": total_loss})
        calibrator.eval().requires_grad_(False)
        calibrated_head = _calibrate(calibrator, head_records, device)
        calibrated_val = _calibrate(calibrator, val_records, device)
        run.check()
        if not any((_features(record, "B", seed, "T_head")[1] > 0).any()
                   for record in calibrated_head):
            raise ValueError("T_head has no common third-node opportunities; cannot fit contextual heads")
        selection = {"A_prime": {"threshold": select_threshold(calibrated_val)["threshold"],
                                  "delta": 0., "epoch": 0, "parameter_count": 0}}
        selection["A_prime"]["validation"] = evaluate_scenes(calibrated_val, selection["A_prime"]["threshold"])
        saved_heads = {}
        for kind in HEAD_KINDS:
            run.check()
            train_packs = [_features(record, kind, seed, "T_head") for record in calibrated_head]
            val_packs = [_features(record, kind, seed, "V_select") for record in calibrated_val]
            scaler = FeatureScaler(HEAD_DIMENSIONS[kind][0]).to(device)
            scaler.fit(np.concatenate([pack[0] for pack in train_packs]), split="T_head")
            scaler.eval()
            best, best_state, trials = None, None, []
            for delta in configuration["deltas"]:
                # Reset independently for every delta, with the same initial
                # random hidden weights within an arm so delta is the only change.
                seed_everything(_seed(seed, "initialize", kind))
                head = ResidualHead(kind, delta).to(device)
                optimizer = torch.optim.Adam(head.parameters(), lr=configuration["learning_rate"],
                                             weight_decay=configuration["weight_decay"])
                curve = []
                for epoch in range(1, configuration["epochs"] + 1):
                    run.check()
                    head.train()
                    order = np.random.default_rng(_seed(seed, "scene-order", epoch)).permutation(len(calibrated_head))
                    loss_sum, scene_count = 0.0, 0
                    for index in order:
                        run.check()
                        record = calibrated_head[index]
                        if not len(record["scores"]):
                            continue
                        if kind == "B_shuffle":
                            features, counts = _features(record, kind, seed, "T_head", epoch)
                        else:
                            features, counts = train_packs[index]
                        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
                        z = torch.as_tensor(record["scores"], dtype=torch.float32, device=device)
                        labels = torch.as_tensor(record["labels"], dtype=torch.float32, device=device)
                        optimizer.zero_grad(set_to_none=True)
                        logits = z + head(scaler(tensor), counts)
                        loss = F.binary_cross_entropy_with_logits(logits, labels)
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite contextual training loss")
                        loss.backward()
                        _check_gradients(head)
                        optimizer.step()
                        loss_sum += loss.detach().item()
                        scene_count += 1
                    predictions = _head_predictions(head, scaler, calibrated_val, kind, seed, "V_select", device, val_packs)
                    threshold = select_threshold(predictions)
                    entry = {"epoch": epoch, "threshold": threshold["threshold"],
                             "macro_f1": threshold["macro_f1"], "scene_mean_training_bce": loss_sum / scene_count}
                    curve.append(entry)
                    run.check()
                    if best is None or entry["macro_f1"] > best["validation"]["macro_f1"] + 1e-12:
                        best = {"threshold": entry["threshold"], "delta": delta, "epoch": epoch,
                                "parameter_count": head.parameter_count,
                                "validation": evaluate_scenes(predictions, entry["threshold"])}
                        best_state = _cpu_state(head)
                trials.append({"delta": delta, "validation_curve": curve})
            best["trials"] = trials
            best["selection_tie_rule"] = "first delta in fixed config order, then earliest epoch"
            selection[kind] = best
            saved_heads[kind] = {"head": best_state, "scaler": _cpu_state(scaler), "delta": best["delta"]}
            # Persist selection progress so budget failure retains useful evidence
            # but load_heads still refuses the unfinished run.
            write_json(run.path / "selection-progress.json", selection)
        fixed_trials, best_fixed = [], None
        for delta in configuration["deltas"]:
            run.check()
            predictions = _fixed_predictions(calibrated_val, delta)
            threshold = select_threshold(predictions)
            trial = {"delta": delta, "threshold": threshold["threshold"], "macro_f1": threshold["macro_f1"]}
            fixed_trials.append(trial)
            if best_fixed is None or trial["macro_f1"] > best_fixed["validation"]["macro_f1"] + 1e-12:
                best_fixed = {"threshold": trial["threshold"], "delta": delta, "epoch": 0,
                              "parameter_count": 0,
                              "validation": evaluate_scenes(predictions, trial["threshold"])}
        best_fixed["trials"] = fixed_trials
        selection["FixedRule"] = best_fixed
        document = {"schema": 1, "binding": binding, "seed": seed, "config": configuration,
                    "config_sha256": object_hash(configuration), "arms": selection,
                    "calibration": {"slope": calibrator.slope.item(), "bias": calibrator.bias.item(),
                                    "curve": calibration_curve, "fit_split": "T_head",
                                    "loss": "mean of per-scene mean edge BCE"},
                    "training_scene_ids": [r["scene_id"] for r in head_records],
                    "validation_scene_ids": [r["scene_id"] for r in val_records],
                    "training_component_ids": sorted({r["component_id"] for r in head_records}),
                    "validation_component_ids": sorted({r["component_id"] for r in val_records}),
                    "zero_neighbor_policy": "all residual arms preserve calibrated logit exactly",
                    "shuffle_policy": "T_head epoch-specific; V_select and T_test fixed per seed/scene/split"}
        torch.save({"schema": 1, "state_dict": _cpu_state(calibrator)}, run.path / "calibrator.pt")
        torch.save({"schema": 1, "heads": saved_heads}, run.path / "heads.pt")
        write_json(run.path / "selection.json", document)
        run.check()
        run.finish(artifacts={name: digest(run.path / name) for name in ("calibrator.pt", "heads.pt", "selection.json")})
        return selection
    except BaseException as error:
        run.finish("FAILED", error_type=type(error).__name__, error=str(error))
        raise


def load_heads(path, device, expected_binding=None):
    """Load complete runs with matching experiment content and model shapes."""
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    if path.is_file() and path.name != "heads.pt":
        raise ValueError("load_heads accepts a run directory or its heads.pt")
    metadata = read_json(directory / "run.json")
    if metadata.get("status") != "COMPLETED" or metadata.get("kind") != "context-heads":
        raise ValueError("only a completed contextual-head run can be loaded")
    selection = read_json(directory / "selection.json")
    binding = semantic_binding(selection["binding"])
    if binding != semantic_binding(metadata["binding"]) or (expected_binding is not None and binding != semantic_binding(jsonable(expected_binding))):
        raise ValueError("head provenance binding mismatch")
    if selection["config"] != metadata["config"]:
        raise ValueError("head configuration mismatch")
    if selection["seed"] != metadata["seed"]:
        raise ValueError("head seed mismatch")
    device = torch.device(device)
    calibration_payload = torch.load(directory / "calibrator.pt", map_location="cpu", weights_only=True)
    head_payload = torch.load(directory / "heads.pt", map_location="cpu", weights_only=True)
    if calibration_payload.get("schema") != 1 or head_payload.get("schema") != 1:
        raise ValueError("unsupported head checkpoint schema")
    calibrator = MonotoneCalibrator().to(device)
    calibrator.load_state_dict(calibration_payload["state_dict"])
    calibrator.eval().requires_grad_(False)
    heads = {}
    for kind in HEAD_KINDS:
        state = head_payload["heads"][kind]
        if state["delta"] != selection["arms"][kind]["delta"]:
            raise ValueError("head residual bound disagrees with selection")
        module = ResidualHead(kind, state["delta"]).to(device)
        module.load_state_dict(state["head"])
        scaler = FeatureScaler(HEAD_DIMENSIONS[kind][0]).to(device)
        scaler.load_state_dict(state["scaler"])
        if not scaler.fitted.item():
            raise ValueError("unfitted feature scaler")
        module.eval().requires_grad_(False)
        scaler.eval()
        heads[kind] = {"head": module, "scaler": scaler}
    modules = [calibrator] + [item[name] for item in heads.values() for name in ("head", "scaler")]
    if any(not torch.isfinite(value).all() for module in modules for value in module.state_dict().values()):
        raise FloatingPointError("head checkpoint contains nonfinite values")
    return {"calibrator": calibrator, "heads": heads, "selection": selection["arms"],
            "binding": selection["binding"], "seed": selection["seed"], "device": device,
            "metadata": selection}


def predict_heads(bundle, records, seed, split="T_test"):
    """Predict all arms from raw A0 scores without fitting or changing thresholds."""
    if int(seed) != bundle["seed"]:
        raise ValueError("inference seed must match the frozen head-run seed")
    if split not in ("T_head", "V_select", "T_test"):
        raise ValueError("split must be T_head, V_select or T_test")
    normalized = _records(records)
    if split == "T_test":
        seen_scenes = set(bundle["metadata"]["training_scene_ids"]) | set(bundle["metadata"]["validation_scene_ids"])
        seen_components = set(bundle["metadata"]["training_component_ids"]) | set(bundle["metadata"]["validation_component_ids"])
        if any(record["scene_id"] in seen_scenes or record["component_id"] in seen_components for record in normalized):
            raise ValueError("held-out prediction overlaps head training or selection scenes/components")
    calibrated = _calibrate(bundle["calibrator"], normalized, bundle["device"])
    predictions = {"A_prime": calibrated}
    for kind in HEAD_KINDS:
        components = bundle["heads"][kind]
        predictions[kind] = _head_predictions(components["head"], components["scaler"], calibrated,
                                              kind, int(seed), split, bundle["device"])
    predictions["FixedRule"] = _fixed_predictions(calibrated, bundle["selection"]["FixedRule"]["delta"])
    components = bundle["heads"]["B"]
    predictions["B_inference_shuffle"] = _head_predictions(
        components["head"], components["scaler"], calibrated, "B_shuffle", int(seed), split, bundle["device"])
    return predictions
