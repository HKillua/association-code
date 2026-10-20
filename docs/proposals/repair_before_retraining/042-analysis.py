#!/usr/bin/env python3
"""Recompute 042 diagnostics from returned distances, without model inference.

Run from any directory. Only the NPZ distances/noise and recorded metadata are
read. Cross-track embedding variance is aggregated from returned per-repeat
statistics; the underlying embeddings were not returned and cannot be audited.
"""
from pathlib import Path
import gzip
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "HandOver/results/TAM-MODE-CHECK-20261020-042"
MODES = ("eval", "train", "train_all_dropout_zero")


def read_json(name):
    with gzip.open(RESULT / name, "rt") as handle:
        return json.load(handle)


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "presentations": int(values.size),
        "mean": float(values.mean()),
        "std_population": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, .95)),
        "max": float(values.max()),
    }


def auc(distances, labels):
    """Exact rank AUC: smaller distance predicts positive; ties count one half."""
    difference = distances[labels == 1, None] - distances[labels == 0][None, :]
    return float(np.mean((difference < 0) + .5 * (difference == 0)))


def confusion(distances, labels, threshold):
    prediction = distances <= threshold
    return {name: int(np.sum(condition)) for name, condition in {
        "tp": prediction & (labels == 1), "fp": prediction & (labels == 0),
        "fn": ~prediction & (labels == 1), "tn": ~prediction & (labels == 0),
    }.items()}


def main():
    summary = read_json("summary.json.gz")
    run = read_json("run.json.gz")
    preparation = read_json("preparation.json.gz")
    margin = float(summary["effective_margin"])
    threshold = -float(run["source_selection"]["threshold"])
    # This is a restricted, indistinguishable constant-distance comparator,
    # not a lower bound on all learned models or a theorem about dropout.
    equal_distance_optimum = margin - .5
    equal_distance_loss = margin / 4 - 1 / 16
    samples = summary["samples"]
    report = {
        "scope": "Four fixed scenes, 32 unique selected pairs, MC8; not full-V performance, "
                 "a training replay, or an estimate of independent sample size.",
        "loss_formula": "0.5*y*d + 0.5*(1-y)*max(m-d,0)^2; balanced = "
                        "0.5*mean(positive term) + 0.5*mean(negative term)",
        "numeric_policy": "All independent arithmetic uses float64 on returned float32 distances. "
                          "Source loss reporting used float32 and can differ by rounding.",
        "effective_margin": margin,
        "original_distance_acceptance_threshold": threshold,
        "restricted_equal_distance_comparator": {
            "formula": "min over d>=0 of 0.25*(d+max(m-d,0)^2)",
            "optimal_distance": equal_distance_optimum,
            "optimal_balanced_loss": equal_distance_loss,
            "scope": "Only the hypothetical identical fixed distance for both classes; "
                     "not a global model lower bound or a proved dropout noise floor.",
        },
        "metadata_checks": {
            "completed_zero_update": run["status"] == "COMPLETED" and run["optimizer_updates"] == 0,
            "source_selected_update_600": run["source_selection"]["update"] == 600,
            "reported_restoration_all_true": all(summary["state_restoration"].values()),
            "reported_no_reference_or_test": not run["reference_read"] and not run["test_opened"]
                and preparation["reference_members_read"] == 0 and not preparation["test_opened"],
            "reported_no_reselection": not run["threshold_reselected"]
                and not run["margin_recalibrated"] and not run["normalizer_refit"],
        },
        "partitions": {},
        "per_sample": [],
    }
    source_mean_max_difference = 0.
    source_loss_max_difference = 0.
    with np.load(RESULT / "distances.npz", allow_pickle=False) as raw:
        for i, sample in enumerate(samples):
            assert np.array_equal(raw[f"sample_{i}_labels"], sample["labels"])
            assert np.array_equal(raw[f"sample_{i}_pairs"], sample["local_pairs"])
            deterministic = raw[f"eval_sample_{i}_distances"]
            without_dropout = raw[f"train_all_dropout_zero_sample_{i}_distances"]
            train_distance = raw[f"train_sample_{i}_distances"].astype(np.float64)
            report["per_sample"].append({
                "sample_index": i, "split": sample["split"], "scene_id": sample["scene_id"],
                "unique_pair_count": len(raw[f"sample_{i}_labels"]),
                "node_occurrence_count": len(sample["node_ids"]),
                "eval_identical_across_repeats": bool(np.all(deterministic == deterministic[0])),
                "eval_equals_train_dropout_zero": bool(np.array_equal(deterministic, without_dropout)),
                "max_absolute_eval_vs_train_zero_difference": float(np.max(abs(
                    deterministic.astype(np.float64) - without_dropout))),
                "train_all_presentation_std": float(train_distance.std()),
                "train_per_pair_repeat_std_mean": float(train_distance.std(axis=0).mean()),
                "train_per_repeat_pair_mean_std": float(train_distance.mean(axis=1).std()),
            })
        partitions = {
            "T_core": [i for i, sample in enumerate(samples) if sample["split"] == "T_core"],
            "V_select": [i for i, sample in enumerate(samples) if sample["split"] == "V_select"],
            "combined": list(range(len(samples))),
        }
        for split, indices in partitions.items():
            labels = np.concatenate([raw[f"sample_{i}_labels"] for i in indices])
            part = {
                "scene_ids": [samples[i]["scene_id"] for i in indices],
                "unique_pairs": len(labels),
                "positive_unique_pairs": int(sum(labels == 1)),
                "negative_unique_pairs": int(sum(labels == 0)),
                "modes": {},
            }
            for mode in MODES:
                distance = np.concatenate([raw[f"{mode}_sample_{i}_distances"]
                    for i in indices], axis=1).astype(np.float64)
                noise = np.concatenate([raw[f"{mode}_sample_{i}_same_track_noise"]
                    for i in indices], axis=1).astype(np.float64)
                variance = np.concatenate([raw[f"{mode}_sample_{i}_cross_track_variance"]
                    for i in indices]).astype(np.float64)
                positive, negative = distance[:, labels == 1], distance[:, labels == 0]
                positive_term = .5 * positive
                negative_term = .5 * np.maximum(margin - negative, 0.) ** 2
                positive_contribution = .5 * float(positive_term.mean())
                negative_contribution = .5 * float(negative_term.mean())
                loss = positive_contribution + negative_contribution
                item = {
                    "positive_distance": distribution(positive),
                    "negative_distance": distribution(negative),
                    "same_track_independent_forward_distance": distribution(noise),
                    "cross_track_variance": {
                        **distribution(variance),
                        "provenance": "Aggregate of returned per-repeat mean coordinate variances, "
                                      "equal sample/repeat weights; raw embeddings unavailable.",
                    },
                    "positive_over_negative_mean_distance": float(positive.mean() / negative.mean()),
                    "positive_over_same_track_noise_mean": float(positive.mean() / noise.mean())
                        if noise.mean() else None,
                    "negative_over_same_track_noise_mean": float(negative.mean() / noise.mean())
                        if noise.mean() else None,
                    "positive_conditional_cl_term": distribution(positive_term),
                    "negative_conditional_cl_term": distribution(negative_term),
                    "positive_balanced_loss_contribution": positive_contribution,
                    "negative_balanced_loss_contribution": negative_contribution,
                    "balanced_loss_float64": loss,
                    "balanced_loss_over_restricted_equal_distance_optimum": loss / equal_distance_loss,
                    "negative_inside_margin_numerator": int(np.sum(negative < margin)),
                    "negative_inside_margin_denominator_presentations": int(negative.size),
                    "negative_inside_margin_fraction": float(np.mean(negative < margin)),
                    "per_repeat_auc_selected_pairs": [auc(row, labels) for row in distance],
                    "per_pair_repeat_std_mean": float(distance.std(axis=0).mean()),
                }
                part["modes"][mode] = item
                if split != "combined":
                    source = summary["modes"][mode]["by_split"][split]
                    for key, value in [("positive_distance", positive),
                                       ("negative_distance", negative), ("same_track_noise", noise)]:
                        source_mean_max_difference = max(source_mean_max_difference,
                            abs(float(value.mean()) - source[key]["mean"]))
                    source_loss_max_difference = max(source_loss_max_difference,
                        abs(loss - source["class_balanced_loss"]))
            controlled = np.concatenate([raw[f"eval_sample_{i}_distances"][0]
                for i in indices]).astype(np.float64)
            fast = np.concatenate([raw[f"eval_default_fastpath_sample_{i}_distances"]
                for i in indices]).astype(np.float64)
            old_order = np.sign(controlled[:, None] - controlled[None, :])
            new_order = np.sign(fast[:, None] - fast[None, :])
            upper = np.triu(np.ones(old_order.shape, dtype=bool), k=1)
            changed = (controlled <= threshold) != (fast <= threshold)
            part["fastpath"] = {
                "max_absolute_distance_difference": float(abs(fast - controlled).max()),
                "max_relative_to_controlled_distance": float((abs(fast - controlled) / controlled).max()),
                "auc_controlled": auc(controlled, labels),
                "auc_default_fastpath": auc(fast, labels),
                "changed_pair_order_comparisons": int(np.sum((old_order != new_order) & upper)),
                "pair_order_comparison_denominator": int(upper.sum()),
                "original_threshold_flips": int(changed.sum()),
                "controlled_original_threshold_confusion": confusion(controlled, labels, threshold),
                "fastpath_original_threshold_confusion": confusion(fast, labels, threshold),
                "scope": "Only selected pair scores, no inference or threshold reselection. "
                         "Small absolute changes can change near-zero score rankings.",
            }
            part["train_vs_eval"] = {
                "positive_mean_distance_ratio": part["modes"]["train"]["positive_distance"]["mean"]
                    / part["modes"]["eval"]["positive_distance"]["mean"],
                "negative_mean_distance_ratio": part["modes"]["train"]["negative_distance"]["mean"]
                    / part["modes"]["eval"]["negative_distance"]["mean"],
                "mean_reported_cross_variance_ratio": part["modes"]["train"]["cross_track_variance"]["mean"]
                    / part["modes"]["eval"]["cross_track_variance"]["mean"],
            }
            report["partitions"][split] = part
    report["source_summary_crosschecks"] = {
        "max_raw_distance_or_noise_mean_difference": source_mean_max_difference,
        "max_float64_loss_difference_from_source_float32": source_loss_max_difference,
    }
    assert all(report["metadata_checks"].values())
    assert all(x["eval_identical_across_repeats"] and x["eval_equals_train_dropout_zero"]
               for x in report["per_sample"])
    assert source_mean_max_difference < 1e-12
    assert source_loss_max_difference < .001
    destination = Path(__file__).with_name("042-audit.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {destination.relative_to(ROOT)}")
    for split, part in report["partitions"].items():
        train = part["modes"]["train"]
        print(split, "train_balanced_loss", train["balanced_loss_float64"],
              "negative_inside_margin", train["negative_inside_margin_numerator"],
              "/", train["negative_inside_margin_denominator_presentations"],
              "fastpath_auc", part["fastpath"]["auc_controlled"], "->",
              part["fastpath"]["auc_default_fastpath"],
              "threshold_flips", part["fastpath"]["original_threshold_flips"])


if __name__ == "__main__":
    main()
