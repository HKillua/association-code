"""Fit existing CDF1 heads on one completed A0/CE/seed20 score cache.

This validation-only entry point never opens an archive or a test score cache.
It retains the reference's original full config and the existing head schedule.
The enclosing handover supplies the total deadline, including input loading.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from association.config import load_config
from association.heads import train_heads
from association.metrics import evaluate_scenes
from association.pipeline import _head_binding
from association.reference import cache_binding, load_reference
from association.runtime import (Run, device_from_string, digest, load_records,
                                 provenance, read_json, write_json)


ARM_ORDER = ("A0", "A_prime", "B", "EndpointMarginal", "PairMLP", "ScenePool",
             "FixedRule", "B_shuffle")


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
    rows += ["", "A_full and independent literature baselines (including TSA-cTFER) are not "
             "included by this entry point. They remain separate comparisons. These numbers "
             "alone do not establish novelty, a successful innovation, or superiority to TMTAFF.", ""]
    return "\n".join(rows)


def run_validation(reference_path, manifest_path, config_path, device, output):
    """Reuse verified reference artifacts, train all existing arms, save a table."""
    reference_path = Path(reference_path).resolve()
    device = torch.device(device)
    manifest = read_json(manifest_path)
    config = load_config(config_path)
    run = Run(output, {"kind": "cdf1-validation-only", "seed": 20,
                       "provenance": provenance(device),
                       "reference_path": str(reference_path)})
    try:
        # This verifies completion, config/code/split bindings and checkpoint hashes.
        # The encoder is loaded only on CPU for verification, never rerun or trained.
        model, normalizer, summary = load_reference(
            reference_path, manifest, config, "A0", "ce", 20, "cpu")
        del model, normalizer
        records = {}
        for split in ("T_head", "V_select"):
            binding = cache_binding(summary, split)
            if summary["cache_bindings"][split] != binding:
                raise ValueError(f"Reference {split} cache binding is inconsistent")
            records[split] = load_records(reference_path / split, binding)
        binding = _head_binding(manifest, config, summary, 20)
        choices = train_heads(records["T_head"], records["V_select"], config,
                              20, device, run.path / "heads", binding)
        arms = {"A0": {"threshold": summary["threshold"],
                        "validation": evaluate_scenes(records["V_select"], summary["threshold"])},
                **choices}
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
