"""Fit existing CDF1 heads on one completed A0/CE/seed20 score cache.

This validation-only entry point never opens an archive or a test score cache.
It retains the reference's original full config and the existing head schedule.
The enclosing handover supplies the total deadline, including input loading.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.config import load_config
from association.heads import load_heads, predict_heads, train_heads
from association.metrics import compare_predictions, evaluate_scenes, predict
from association.pipeline import _head_binding
from association.reference import cache_binding, load_reference
from association.runtime import (Run, device_from_string, digest, load_records,
                                 provenance, read_json, save_records, semantic_binding,
                                 validate_record_split, write_json)


ARM_ORDER = ("A0", "A_prime", "B", "EndpointMarginal", "PairMLP", "ScenePool",
             "FixedRule", "B_shuffle")


def _edge_examples(baseline, candidate, old_threshold, new_threshold, limit=5):
    """First changed edges in saved scene/pair order, not cherry-picked gains."""
    examples = {"corrected": [], "introduced_errors": []}
    candidates = {r["scene_id"]: r for r in candidate}
    for old in baseline:
        new = candidates[old["scene_id"]]
        labels = np.asarray(old["labels"], dtype=bool)
        before, after = predict(old["scores"], old_threshold), predict(new["scores"], new_threshold)
        for kind, mask in (("corrected", (before != labels) & (after == labels)),
                           ("introduced_errors", (before == labels) & (after != labels))):
            for edge in np.flatnonzero(mask)[:max(0, limit - len(examples[kind]))]:
                left, right = old["pairs"][edge]
                examples[kind].append({"scene_id": old["scene_id"], "pair": [int(left), int(right)],
                    "node_ids": [old["node_ids"][left], old["node_ids"][right]],
                    "label": int(labels[edge]), "old_prediction": bool(before[edge]),
                    "new_prediction": bool(after[edge]), "old_score": float(old["scores"][edge]),
                    "new_score": float(new["scores"][edge])})
    return examples


def _markdown(report):
    rows = ["# CDF1 exploratory validation comparison", "",
            "All rows use the same V_select scenes and candidates. Thresholds and head "
            "epochs were selected on this set; these are not held-out test results.", "",
            "| Arm | Macro F1 | Micro F1 | Precision | Recall | Same-source F1 | Cross-source F1 |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm in ARM_ORDER:
        metrics = report["arms"][arm]["validation"]
        values = [metrics[key] for key in ("macro_f1", "micro_f1", "micro_precision", "micro_recall")]
        values += [metrics["itsa"]["macro_f1"], metrics["mstsa"]["macro_f1"]]
        rows.append("| " + arm + " | " + " | ".join(
            "undefined" if value is None else f"{value:.6f}" for value in values) + " |")
    rows += ["", "| Comparison | Macro F1 difference |", "| --- | --- |"]
    for name, difference in report["differences"].items():
        value = difference["macro_f1"]
        rows.append(f"| {name} | {'undefined' if value is None else f'{value:+.6f}'} |")
    rows += ["", "| Baseline → CDF1 | Wrong → right | Right → wrong | Net corrections |",
             "| --- | ---: | ---: | ---: |"]
    for baseline, counts in report["error_changes"].items():
        rows.append(f"| {baseline} → B | {counts['wrong_to_right']} | {counts['right_to_wrong']} | "
                    f"{counts['net_corrected_edges']} |")
    rows += ["", "A_full and independent literature baselines (including TSA-cTFER) are not "
             "included by this entry point. They remain separate comparisons. These numbers "
             "alone do not establish novelty, a successful innovation, or superiority to TMTAFF.", ""]
    return "\n".join(rows)


def run_validation(reference_path, manifest_path, config_path, device, output):
    """Reuse completed reference artifacts, train all existing arms, save a table."""
    reference_path = Path(reference_path).resolve()
    device = torch.device(device)
    manifest = read_json(manifest_path)
    config = load_config(config_path)
    run = Run(output, {"kind": "cdf1-validation-only", "seed": 20,
                       "provenance": provenance(device),
                       "reference_path": str(reference_path)})
    try:
        # Check completion and actual configuration/split/model content, without SHA gates.
        # The encoder is loaded only on CPU, never rerun or trained.
        model, normalizer, summary = load_reference(
            reference_path, manifest, config, "A0", "ce", 20, "cpu")
        del model, normalizer
        records = {}
        for split in ("T_head", "V_select"):
            binding = cache_binding(summary, split)
            if semantic_binding(summary["cache_bindings"][split]) != semantic_binding(binding):
                raise ValueError(f"Reference {split} cache binding is inconsistent")
            records[split] = load_records(reference_path / split, binding)
            validate_record_split(records[split], manifest, split)
        reference_validation = evaluate_scenes(records["V_select"], summary["threshold"])
        if reference_validation != summary["validation"]:
            raise ValueError("Reference V_select cache metrics differ from selected validation")
        binding = _head_binding(manifest, config, summary, 20)
        started = time.monotonic()
        choices = train_heads(records["T_head"], records["V_select"], config,
                              20, device, run.path / "heads", binding)
        training_seconds = time.monotonic() - started
        arms = {"A0": {"threshold": summary["threshold"],
                        "validation": reference_validation},
                **choices}
        started = time.monotonic()
        bundle = load_heads(run.path / "heads", device, expected_binding=binding)
        corrected = predict_heads(bundle, records["V_select"], 20, split="V_select")
        predictions = {"A0": records["V_select"], **{arm: corrected[arm] for arm in ARM_ORDER if arm != "A0"}}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.monotonic() - started
        changes = {}
        for arm in ARM_ORDER:
            if evaluate_scenes(predictions[arm], arms[arm]["threshold"]) != arms[arm]["validation"]:
                raise ValueError(f"Reloaded {arm} does not reproduce selected validation metrics")
            if arm != "A0":
                save_records(run.path / "V_select" / arm, predictions[arm],
                             {"split": "V_select", "arm": arm, "seed": 20})
            if arm != "B":
                changes[arm] = compare_predictions(predictions[arm], predictions["B"],
                                                   arms[arm]["threshold"], arms["B"]["threshold"])
        differences = {}
        for baseline in ARM_ORDER:
            if baseline == "B":
                continue
            left, right = arms["B"]["validation"], arms[baseline]["validation"]
            differences[f"B_minus_{baseline}"] = {
                key: None if left[key] is None or right[key] is None else left[key] - right[key]
                for key in ("macro_f1", "micro_f1", "micro_precision", "micro_recall")}
        report = {
            "schema": 1, "scope": "exploratory_validation_only", "seed": 20,
            "binding": binding, "arms": arms, "differences": differences,
            "error_changes": changes,
            "A0_to_B_examples": _edge_examples(predictions["A0"], predictions["B"],
                                               arms["A0"]["threshold"], arms["B"]["threshold"]),
            "example_policy": "First up to five corrected and five newly wrong edges in cache order",
            "phase_seconds": {"head_training": training_seconds, "selected_heads_inference": inference_seconds},
            "training_split": "T_head", "selection_split": "V_select",
            "reference_training_splits": summary["training_splits"],
            "reference_summary_validation": summary["validation"],
            "test_opened": False,
            "head_config": config["heads"],
            "reference_artifacts": {name: digest(reference_path / name) for name in
                                    ("run.json", "summary.json", "checkpoint.pt",
                                     "T_head/index.json", "V_select/index.json")},
            "missing_comparisons": ["A_full", "independent literature baseline including TSA-cTFER"],
            "interpretation": "Validation selected thresholds, epochs and deltas. Differences are "
                              "descriptive, not unbiased generalization estimates or novelty evidence. "
                              "A0 metrics are recomputed from the same frozen V_select cache used by heads."
        }
        write_json(run.path / "validation.json", report)
        (run.path / "validation.md").write_text(_markdown(report), encoding="utf-8")
        run.finish(artifacts={name: digest(run.path / name) for name in
                              ("validation.json", "validation.md", "heads/run.json",
                               "heads/selection.json")})
        return report
    except BaseException as error:
        run.finish("FAILED", error_type=type(error).__name__, error=str(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Completed A0/CE/seed20 run directory")
    parser.add_argument("--manifest", required=True, help="Original immutable manifest.json")
    parser.add_argument("--config", default="configs/first-fit-v2.json", help="Original reference config")
    parser.add_argument("--device", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", required=True, help="New directory; existing output is refused")
    args = parser.parse_args(argv)
    try:
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
        device = device_from_string(args.device)
        run_validation(args.reference, args.manifest, args.config, device, args.output)
        print(f"COMPLETED validation-only {Path(args.output).resolve()}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"FAILED {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
