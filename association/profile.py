"""Bounded, explicitly ineligible reference throughput and memory probe.

This exercises real TAM/optional-TCM gradients on training inputs. It produces
no reference checkpoint, validation selection, or experimental effect claim.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import runtime
from .config import validate_manifest_config
from .dataset import load_split
from .models import AssociationModel
from .reference import (_binding, _check_configuration, _optimizer, _update,
                        fit_normalizer, normalize_scene)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(device):
    """Read process allocator counters without sampling or synchronizing CUDA."""
    return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None}


def profile_reference(archive_path, manifest, config, role, branch, seed, device, output,
                      steps=200, max_minutes=30):
    """Profile <=200 actual optimization steps on the largest training scene.

    A0 sees T_core, A_full sees T_core+T_head. Scene selection uses valid node
    count only; ties use numeric scene ID. Pair order is seed-controlled, uses
    every pair before repeating, and applies no label-dependent subsampling.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 200:
        raise ValueError("profile steps must be an integer in [1, 200]")
    if not math.isfinite(max_minutes) or not 0 < max_minutes <= 30:
        raise ValueError("profile max_minutes must be in (0, 30]")
    model_config, ref = _check_configuration(config, role, branch)
    validate_manifest_config(manifest, config)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("profile requires explicit cpu or cuda device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profile requested but CUDA is unavailable; no CPU fallback")
    binding = _binding(manifest, config, role, branch, seed)
    run = runtime.Run(output, {**binding, "kind": "reference_profile_not_eligible",
                              "eligible_reference": False, "config": config,
                              "provenance": runtime.provenance(device)}, max_minutes / 60)
    report = {**binding, "status": "RUNNING", "eligible_reference": False,
              "requested_steps": steps, "completed_steps": 0, "max_minutes": float(max_minutes),
              "device": str(device), "gpu_measured": device.type == "cuda",
              "test_measurements_read": False, "validation_measurements_read": False,
              "checkpoint_written": False, "step_seconds": [], "losses": [], "gradient_norms": [],
              "step_phase_seconds": [], "step_positive_pair_counts": [],
              "phase": "data_loading", "phase_seconds": {}, "elapsed_seconds": 0.0,
              "peak_allocated_bytes": None, "peak_reserved_bytes": None,
              "memory_measurement_scope": "process allocator peak since profile start; not whole-device occupancy",
              "progress_note": "Last persisted snapshot only; interrupted RUNNING output is incomplete and may omit the active step.",
              "step_timing_note": "Wall times with CUDA synchronization at each phase boundary; includes instrumentation overhead and is not directly comparable to probes without these boundaries.",
              "warmup_steps": min(5, steps), "initialization": "fresh seeded model; no TCM pretraining",
              "scope": "execution-cost and gradient smoke only; not a trained association reference"}
    phase_started = time.perf_counter()

    def persist_progress(next_phase=None):
        nonlocal phase_started
        now = time.perf_counter()
        report["phase_seconds"][report["phase"]] = now - phase_started
        report["elapsed_seconds"] = time.monotonic() - run.started
        try:
            report.update(_memory_snapshot(device))
            report["memory_snapshot_completed_steps"] = report["completed_steps"]
            report["memory_snapshot_elapsed_seconds"] = report["elapsed_seconds"]
        except Exception as error:
            # A failed telemetry read must not hide the original training error
            # or discard a previously persisted allocator peak.
            report["memory_measurement_error"] = f"{type(error).__name__}: {error}"
        if next_phase is not None:
            report["phase"] = next_phase
            phase_started = now
            report["phase_seconds"][next_phase] = 0.0
        runtime.write_json(run.path / "profile.json", report)

    runtime.write_json(run.path / "profile.json", report)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        runtime.seed_everything(seed)
        rng = np.random.default_rng(seed)
        scenes = load_split(archive_path, manifest, "T_core", model_config.length)
        if role == "A_full":
            scenes += load_split(archive_path, manifest, "T_head", model_config.length)
        persist_progress("preparation")
        run.check()
        if not scenes or not any(len(scene["pairs"]) for scene in scenes):
            raise ValueError("profiling requires eligible training pairs")
        normalizer = fit_normalizer(scenes)
        chosen = min(scenes, key=lambda scene: (-len(scene["values"]), int(scene["scene_id"])))
        if not len(chosen["pairs"]):
            raise ValueError("largest valid training scene has no pairs")
        values, masks = normalize_scene(chosen, normalizer)
        pairs = np.asarray(chosen["pairs"], dtype=np.int64).reshape(-1, 2)
        all_labels = np.asarray(chosen["labels"], dtype=np.float32)
        train_nodes = sum(len(scene["values"]) for scene in scenes)
        train_pairs = sum(len(scene["pairs"]) for scene in scenes)
        tam_steps_per_epoch = sum(math.ceil(len(scene["pairs"]) / ref["pair_batch_size"]) for scene in scenes)
        tcm_steps_per_epoch = math.ceil(train_nodes / ref["batch_tracks"])
        report.update({"training_scene_ids": [scene["scene_id"] for scene in scenes],
                       "normalizer": normalizer, "selected_scene_id": chosen["scene_id"],
                       "selected_scene_node_count": len(values), "selected_scene_pair_count": len(pairs),
                       "scene_selection": "maximum valid node count; numeric scene ID ties; training only",
                       "training_scene_count": len(scenes), "training_node_count": train_nodes,
                       "training_pair_count": train_pairs, "tam_steps_per_epoch": tam_steps_per_epoch,
                       "tcm_steps_per_epoch": tcm_steps_per_epoch,
                       "planned_tam_steps": tam_steps_per_epoch * ref["tam_epochs"],
                       "planned_tcm_steps": tcm_steps_per_epoch * ref["tcm_epochs"],
                       "pair_batch_size": ref["pair_batch_size"], "batch_tracks": ref["batch_tracks"]})
        persist_progress("model_initialization")
        model = AssociationModel(model_config, branch).to(device)
        joint = bool(ref.get("joint_completion", True))
        model.set_completion_trainable(joint, include_embedding=True)
        model.train()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        before = [parameter.detach().cpu().clone() for parameter in parameters]
        optimizer = _optimizer(parameters, ref)
        clip = float(ref.get("gradient_clip", 1.0))
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")
        report.update(joint_completion=joint,
                      completion_weight=ref.get("completion_weight", 0.1) if joint else 0.0,
                      trainable_parameter_count=sum(parameter.numel() for parameter in parameters))
        if device.type == "cuda":
            _synchronize(device)
        order = rng.permutation(len(pairs))
        cursor = 0
        repeated_passes = 0
        persist_progress("optimization")
        run.check()
        for step in range(steps):
            run.check()
            if cursor >= len(order):
                order = rng.permutation(len(pairs))
                cursor = 0
                repeated_passes += 1
            selected = order[cursor:cursor + ref["pair_batch_size"]]
            cursor += len(selected)
            _synchronize(device)
            start = time.perf_counter()
            nodes, inverse = np.unique(pairs[selected], return_inverse=True)
            x = torch.as_tensor(values[nodes], device=device)
            observed = torch.as_tensor(masks[nodes], device=device)
            _synchronize(device)
            batch_ready = time.perf_counter()
            embeddings = torch.cat([
                model.encode_tracks(x[k:k + ref["batch_tracks"]], observed[k:k + ref["batch_tracks"]])
                for k in range(0, len(nodes), ref["batch_tracks"])
            ])
            _synchronize(device)
            encoded = time.perf_counter()
            scores = model.score_pairs(embeddings, inverse.reshape(-1, 2))
            labels = torch.as_tensor(all_labels[selected], device=device)
            loss = model.association_loss(scores, labels)
            _synchronize(device)
            association_ready = time.perf_counter()
            if joint and ref.get("completion_weight", 0.1):
                loss = loss + ref.get("completion_weight", 0.1) * model.completion_loss(x, observed)
            _synchronize(device)
            completion_ready = time.perf_counter()
            value, norm = _update(loss, optimizer, parameters, clip)
            _synchronize(device)
            updated = time.perf_counter()
            elapsed = updated - start
            report["step_seconds"].append(elapsed)
            report["step_phase_seconds"].append({
                "batch_preparation": batch_ready - start, "encode_tracks": encoded - batch_ready,
                "association_loss": association_ready - encoded,
                "completion_auxiliary": completion_ready - association_ready,
                "backward_optimizer": updated - completion_ready})
            report["losses"].append(value)
            report["gradient_norms"].append(norm)
            report["completed_steps"] = step + 1
            report.setdefault("step_pair_counts", []).append(len(selected))
            report.setdefault("step_unique_tracks", []).append(len(nodes))
            report["step_positive_pair_counts"].append(int(np.count_nonzero(all_labels[selected] == 1)))
            persist_progress()
            run.check()
        persist_progress("finalization")
        if not all(torch.isfinite(parameter).all() for parameter in parameters):
            raise FloatingPointError("profile optimizer produced nonfinite parameters")
        changed = [not torch.equal(initial, parameter.detach().cpu())
                   for initial, parameter in zip(before, parameters)]
        steady = np.asarray(report["step_seconds"][report["warmup_steps"]:], dtype=float)
        report.update(parameters_changed=any(changed), changed_parameter_tensor_count=sum(changed),
                      repeated_scene_passes=repeated_passes, measured_steady_steps=len(steady),
                      mean_all_step_seconds=float(np.mean(report["step_seconds"])),
                      mean_steady_step_seconds=float(steady.mean()) if len(steady) else None,
                      p50_steady_step_seconds=float(np.median(steady)) if len(steady) else None,
                      p95_steady_step_seconds=float(np.quantile(steady, .95)) if len(steady) else None,
                      gpu_measurement_note="actual synchronized CUDA operations" if device.type == "cuda"
                      else "CPU only; no GPU speed or CUDA-memory claim")
        tam_estimate = float(steady.mean()) * report["planned_tam_steps"] if len(steady) else None
        report["cost_projection"] = {
            "status": "TAM_training_only_extrapolation" if len(steady) else "warmup_only_no_projection",
            "tam_training_only_seconds": tam_estimate,
            "tam_training_only_hours": tam_estimate / 3600 if tam_estimate is not None else None,
            "tcm_training_seconds": None, "validation_seconds": None, "cache_seconds": None,
            "full_reference_seconds": None,
            "reference_cap_hours": ref["max_hours"],
            "tam_estimate_exceeds_total_cap": tam_estimate > ref["max_hours"] * 3600 if tam_estimate is not None else None,
            "limitations": "Largest-node scene and shuffled pair mix may differ from other scenes. This covers only TAM optimization, excluding TCM pretraining, validation, caching and input preparation. It is neither a mathematical lower bound nor permission to extend the budget.",
        }
        run.check()
        report["status"] = "COMPLETED"
        persist_progress()
        run.finish(profile_sha256=runtime.digest(run.path / "profile.json"),
                   eligible_reference=False, completed_steps=report["completed_steps"])
        return report
    except BaseException as error:
        report.update(status="FAILED", failure_type=type(error).__name__, failure=str(error))
        persist_progress()
        run.finish("FAILED", failure_type=type(error).__name__, failure=str(error),
                   eligible_reference=False, completed_steps=report["completed_steps"],
                   profile_sha256=runtime.digest(run.path / "profile.json"))
        raise
