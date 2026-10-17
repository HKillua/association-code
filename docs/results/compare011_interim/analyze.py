"""CPU analysis of A0/TSA/A_full caches and CDF summary arithmetic; no fitting."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from association.runtime import load_records
from association.metrics import evaluate_scenes, compare_predictions, paired_cluster_bootstrap
from tools.experiments.tsactfer_fit import paper_scope_coverage

SOURCE = ROOT / "HandOver/results/COMPARE-20261017-011"
OUTPUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def compact(metrics):
    keys = ("scene_count", "pair_count", "macro_f1", "micro_f1", "micro_precision",
            "micro_recall", "tp", "fp", "fn", "tn", "negative_scene_count",
            "negative_scene_false_acceptance_rate")
    result = {key: metrics[key] for key in keys}
    result["fdr"] = metrics["p_fa"]
    result["fpr"] = metrics["fp"] / (metrics["fp"] + metrics["tn"])
    return result


def scope_mask(record):
    left, right = record["pairs"].T
    starts, ends = record["intervals"].T
    duration = ends - starts
    same = record["sources"][left] == record["sources"][right]
    gap = np.maximum(starts[left], starts[right]) - np.minimum(ends[left], ends[right])
    return ((same & (gap >= 0) & (gap < 1200) & (duration[left] > 120)
             & (duration[right] > 120)) | (~same & (-gap > 120)))


def subset(records, masks):
    return [dict(record, **{key: record[key][mask] for key in ("pairs", "labels", "scores")})
            for record, mask in zip(records, masks)]


def rank_metrics(records):
    """Global ROC-AUC with half-credit ties and threshold-group average precision."""
    scores = np.concatenate([row["scores"] for row in records])
    labels = np.concatenate([row["labels"] for row in records]).astype(bool)
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=labels, minlength=len(unique))
    negatives = counts - positives
    auc = np.sum(positives * (np.cumsum(negatives) - 0.5 * negatives)) / (labels.sum() * (~labels).sum())
    tp, predicted = np.cumsum(positives[::-1]), np.cumsum(counts[::-1])
    ap = np.sum((tp / predicted) * positives[::-1]) / labels.sum()
    return {"roc_auc": float(auc), "average_precision": float(ap),
            "positive_prevalence": float(labels.mean()), "distinct_scores": len(unique),
            "tied_score_groups": int((counts > 1).sum()),
            "max_tie_group_size": int(counts.max()),
            "scope": "Global full-V_select ranking of already selected models; no threshold fitting"}


def audit_cdf_summary(report, selection, reference_metrics):
    """Check returned counts, not unreturned CDF scores or individual edges."""
    assert report["arms"]["A0"]["validation"] == reference_metrics
    assert report["reference_summary_validation"] == reference_metrics
    assert all(report["arms"][name] == arm for name, arm in selection["arms"].items())
    reference_rows = {row["scene_id"]: row for row in reference_metrics["per_scene"]}
    counts = ("tp", "fp", "fn", "tn")
    rows_by_arm = {}
    for name, arm in report["arms"].items():
        metrics = arm["validation"]
        rows = {row["scene_id"]: row for row in metrics["per_scene"]}
        assert len(rows) == len(metrics["per_scene"]) and rows.keys() == reference_rows.keys(), name
        rows_by_arm[name] = rows
        for sid, row in rows.items():
            assert all(isinstance(row[k], int) and row[k] >= 0 for k in counts), (name, sid)
            assert sum(row[k] for k in counts) == row["pair_count"], (name, sid)
            assert row["tp"] + row["fn"] == row["positive_count"], (name, sid)
            for key in ("component_id", "node_count", "pair_count", "positive_count"):
                assert row[key] == reference_rows[sid][key], (name, sid, key)
            denominator = 2 * row["tp"] + row["fp"] + row["fn"]
            expected_f1 = 2 * row["tp"] / denominator if denominator else 1.0
            assert np.isclose(row["f1"], expected_f1, rtol=0, atol=1e-12), (name, sid)
        totals = {key: sum(row[key] for row in rows.values()) for key in counts + ("pair_count",)}
        assert all(metrics[key] == value for key, value in totals.items()), name
        assert metrics["scene_count"] == len(rows), name
        expected = {
            "macro_f1": np.mean([row["f1"] for row in rows.values()]),
            "micro_f1": 2 * totals["tp"] / (2 * totals["tp"] + totals["fp"] + totals["fn"]),
            "micro_precision": totals["tp"] / (totals["tp"] + totals["fp"]),
            "micro_recall": totals["tp"] / (totals["tp"] + totals["fn"]),
        }
        assert all(np.isclose(metrics[key], value, rtol=0, atol=1e-12)
                   for key, value in expected.items()), name
    change_keys = ("wrong_to_right", "right_to_wrong", "corrected_fp", "corrected_fn",
                   "introduced_fp", "introduced_fn")
    for baseline, changes in report["error_changes"].items():
        rows = {row["scene_id"]: row for row in changes["per_scene"]}
        assert len(rows) == len(changes["per_scene"]) and rows.keys() == reference_rows.keys(), baseline
        assert all(changes[k] == sum(row[k] for row in rows.values()) for k in change_keys), baseline
        for sid, row in rows.items():
            old, new = rows_by_arm[baseline][sid], rows_by_arm["B"][sid]
            assert all(isinstance(row[k], int) and row[k] >= 0 for k in change_keys), (baseline, sid)
            assert row["wrong_to_right"] == row["corrected_fp"] + row["corrected_fn"]
            assert row["right_to_wrong"] == row["introduced_fp"] + row["introduced_fn"]
            assert row["introduced_fp"] - row["corrected_fp"] == new["fp"] - old["fp"]
            assert row["introduced_fn"] - row["corrected_fn"] == new["fn"] - old["fn"]
            assert row["corrected_fp"] <= old["fp"] and row["corrected_fn"] <= old["fn"]
            assert row["introduced_fp"] <= old["tn"] and row["introduced_fn"] <= old["tp"]
        old = report["arms"][baseline]["validation"]
        new = report["arms"]["B"]["validation"]
        assert changes["net_corrected_edges"] == changes["wrong_to_right"] - changes["right_to_wrong"]
        assert changes["net_corrected_edges"] == old["fp"] + old["fn"] - new["fp"] - new["fn"]
    return {
        "evidence_level": "Returned per-scene/count summaries only; CDF score caches not returned",
        "all_arm_summary_arithmetic_checked": True,
        "same_reported_scene_component_node_pair_and_positive_counts_as_A0": True,
        "individual_CDF_edges_scores_and_common_paper_scope_recomputed": False,
        "full": {name: compact(arm["validation"]) for name, arm in report["arms"].items()},
        "error_change_direction": "baseline -> B (CDF1); negative net means B introduces more errors",
        "error_changes_baseline_to_B": {name: {k: v for k, v in row.items() if k != "per_scene"}
                                         for name, row in report["error_changes"].items()},
        "phase_seconds": report["phase_seconds"],
        "test_opened_reported": report["test_opened"],
        "missing": ["Seven V_select/<arm> score-cache directories, needed for independent edge verification",
                    "CDF/TSA common applicability subset and paired score diagnostics"],
    }


def analyze():
    summary = {
        "A0": read(SOURCE / "reference010/summary_metrics.json"),
        "TSA": read(SOURCE / "tsa/files/summary.json"),
        "A_full": read(SOURCE / "afull/files/summary_metrics.json"),
    }
    records = {
        "A0": load_records(SOURCE / "reference010/V_select"),
        "TSA": load_records(SOURCE / "tsa/files/V_select"),
        "A_full": load_records(SOURCE / "afull/files/V_select"),
    }
    for name in ("TSA", "A_full"):
        assert len(records["A0"]) == len(records[name]) == 64
        for left, right in zip(records["A0"], records[name]):
            for key in ("scene_id", "component_id", "node_ids"):
                assert left[key] == right[key], (name, key)
            for key in ("pairs", "labels", "identities", "sources", "intervals"):
                assert np.array_equal(left[key], right[key]), (name, key)
    metrics = {name: evaluate_scenes(rows, summary[name]["threshold"])
               for name, rows in records.items()}
    for name in metrics:
        assert metrics[name] == summary[name]["validation"], name
    coverage = paper_scope_coverage(records["A0"])
    assert coverage == summary["TSA"]["V_select_scope_coverage"]
    masks = [scope_mask(row) for row in records["A0"]]
    restricted = {name: evaluate_scenes(subset(rows, masks), summary[name]["threshold"])
                  for name, rows in records.items()}
    changes = compare_predictions(records["A0"], records["TSA"],
                                  summary["A0"]["threshold"], summary["TSA"]["threshold"])
    afull_changes = compare_predictions(records["A0"], records["A_full"],
                                        summary["A0"]["threshold"], summary["A_full"]["threshold"])
    differences = np.array([b["f1"] - a["f1"] for a, b in
                           zip(metrics["A0"]["per_scene"], metrics["TSA"]["per_scene"])])
    cdf = audit_cdf_summary(read(SOURCE / "cdf/files/validation.json"),
                            read(SOURCE / "cdf/files/heads/selection.json"), metrics["A0"])
    afull_run = read(SOURCE / "afull/files/run.json")
    afull_progress = read(SOURCE / "afull/files/progress.json")
    afull_history = read(SOURCE / "afull/files/history.json")
    afull_selections = [{"phase": row["phase"], "update": row["update"], **row["selection"]}
                        for row in afull_history if "selection" in row]
    assert afull_run["status"] == afull_progress["status"] == "COMPLETED"
    assert (afull_run["role"], afull_run["branch"], afull_run["seed"]) == ("A_full", "ce", 20)
    assert afull_run["eligible_reference"] and summary["A_full"]["eligible_reference"]
    assert summary["A_full"]["test_opened"] is False and afull_progress["test_measurements_read"] is False
    assert (SOURCE / "afull/exit_code.txt").read_text().strip() == "0"
    assert float((SOURCE / "afull/elapsed_seconds.txt").read_text()) <= 10800
    assert afull_run["config"] == read(ROOT / "configs/first-fit-v2.json")
    assert [(row["phase"], row["completed_updates"]) for row in afull_progress["stages"]] == [
        ("TCM", 100), ("TAM_FROZEN", 100), ("JOINT", 20)]
    assert len(afull_history) == afull_progress["completed_updates"] == 220
    cdf_training = read(SOURCE / "cdf/files/heads/selection.json")["training_scene_ids"]
    assert summary["A_full"]["training_splits"] == ["T_core", "T_head"]
    assert summary["A_full"]["training_scene_ids"] == summary["A0"]["training_scene_ids"] + cdf_training
    assert summary["A_full"]["normalizer"]["training_scene_ids"] == summary["A_full"]["training_scene_ids"]
    assert not set(summary["A_full"]["training_scene_ids"]) & {r["scene_id"] for r in records["A_full"]}
    afull_unique_scores = len(np.unique(np.concatenate([r["scores"] for r in records["A_full"]])))
    assert afull_selections[-1]["candidate_count"] == afull_unique_scores + 1
    assert afull_selections[-1]["threshold"] == summary["A_full"]["threshold"]
    afull_training = {}
    for phase in ("TCM", "TAM_FROZEN", "JOINT"):
        rows = [row for row in afull_history if row["phase"] == phase]
        loss_key = "loss" if phase == "TCM" else "association_loss"
        losses = np.array([row[loss_key] for row in rows])
        gradients = np.array([row["gradient_norm"] for row in rows])
        assert np.isfinite(losses).all() and np.isfinite(gradients).all()
        afull_training[phase] = {"updates": len(rows), "loss_field": loss_key,
            "first_10_loss_mean": float(losses[:10].mean()), "last_10_loss_mean": float(losses[-10:].mean()),
            "all_gradient_norms_nonzero": bool((gradients > 0).all()),
            "gradient_norm_min": float(gradients.min()), "gradient_norm_max": float(gradients.max()),
            "gradient_norm_above_clip_1_count": int((gradients > 1).sum())}
    report = {
        "scope": "011 jobs completed: A0/TSA/A_full scores verified; CDF summary arithmetic checked, score caches pending",
        "sources": {name: str(path.relative_to(ROOT)) for name, path in
                    (("A0", SOURCE / "reference010"), ("TSA", SOURCE / "tsa/files"),
                     ("A_full", SOURCE / "afull/files"), ("CDF_summary", SOURCE / "cdf/files"))},
        "cdf_summary_audit": cdf,
        "afull_evidence": {
            "status": "COMPLETED; command, exit code 0, run/progress/history, summary subset and all validation scores returned",
            "history_progress_run_summary_subset_and_scores_returned": True,
            "outer_elapsed_seconds": float((SOURCE / "afull/elapsed_seconds.txt").read_text()),
            "inner_elapsed_seconds": afull_run["elapsed_seconds"],
            "phase_seconds": summary["A_full"]["phase_seconds"],
            "training_scene_count": len(summary["A_full"]["training_scene_ids"]),
            "association_updates": summary["A_full"]["selected_association_update"],
            "history_selections": afull_selections,
            "training_diagnostics": afull_training,
            "final_unique_scores": afull_unique_scores,
            "actual_pair_count": metrics["A_full"]["pair_count"],
            "candidate_count_definition": "select_threshold: number of unique scores plus one all-reject threshold, not pair_count",
            "interpretation": "History confirms 29001/28985 are threshold candidate counts; final 28984 unique scores plus all-reject coexist with all 29008 pairs. No dropped-edge evidence.",
        },
        "reproduces_both_reported_validation_dictionaries": True,
        "reproduces_all_three_reported_validation_dictionaries": True,
        "thresholds": {name: row["threshold"] for name, row in summary.items()},
        "full": {name: compact(row) for name, row in metrics.items()},
        "by_source": {name: {group: compact(row[group]) for group in ("itsa", "mstsa")}
                      for name, row in metrics.items()},
        "common_paper_scope_original_thresholds": {name: compact(row) for name, row in restricted.items()},
        "paper_scope_coverage": coverage,
        "error_changes_A0_to_TSA": changes,
        "error_changes_A0_to_A_full": afull_changes,
        "scene_f1_TSA_win_tie_loss": [int((differences > 1e-12).sum()),
                                      int((np.abs(differences) <= 1e-12).sum()),
                                      int((differences < -1e-12).sum())],
        "cluster_uncertainty": paired_cluster_bootstrap(records["A0"], records["TSA"],
            summary["A0"]["threshold"], summary["TSA"]["threshold"], n_bootstrap=2000,
            seed=20261017, min_clusters=10),
        "all_accept": compact(evaluate_scenes(records["A0"],
                              min(float(r["scores"].min()) for r in records["A0"]))),
        "score_quantiles": {name: {str(label): np.quantile(np.concatenate([
            row["scores"][row["labels"] == label] for row in rows]), [0, .1, .5, .9, 1]).tolist()
            for label in (0, 1)} for name, rows in records.items()},
        "score_rank_metrics": {name: rank_metrics(rows) for name, rows in records.items()},
        "limits": ["V_select selected weights and thresholds; interval is descriptive, not independent test inference",
                   "Single seed, unequal optimization schedules and validation selection counts",
                   "TSA is offline adaptation, not original online system; native checkpoints remain on server",
                   "No zero-positive scenes; no T_test read; no new training or threshold fitting",
                   "CDF counts are internally consistent but its seven score caches are absent; no independent edge/common-subset recomputation",
                   "A_full used a larger pool with the same short update count; worse results do not establish a causal effect of more data or adequate training"],
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("error_changes_A0_to_TSA", "error_changes_A0_to_A_full",
                                     "cdf_summary_audit", "cluster_uncertainty", "score_quantiles")},
                     ensure_ascii=False, indent=2))
    print("error changes", {key: value for key, value in changes.items() if key != "per_scene"})
    print("cluster uncertainty", {key: value for key, value in report["cluster_uncertainty"].items()
                                  if key != "per_cluster"})


if __name__ == "__main__":
    analyze()
