"""Read-only CPU analysis of returned A0/TSA validation caches; no fitting."""
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


def analyze():
    summary = {
        "A0": read(SOURCE / "reference010/summary_metrics.json"),
        "TSA": read(SOURCE / "tsa/files/summary.json"),
    }
    records = {
        "A0": load_records(SOURCE / "reference010/V_select"),
        "TSA": load_records(SOURCE / "tsa/files/V_select"),
    }
    assert len(records["A0"]) == len(records["TSA"]) == 64
    for left, right in zip(records["A0"], records["TSA"]):
        for key in ("scene_id", "component_id", "node_ids"):
            assert left[key] == right[key], key
        for key in ("pairs", "labels", "identities", "sources", "intervals"):
            assert np.array_equal(left[key], right[key]), key
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
    differences = np.array([b["f1"] - a["f1"] for a, b in
                           zip(metrics["A0"]["per_scene"], metrics["TSA"]["per_scene"])])
    report = {
        "scope": "Interim, single-seed validation-selected comparison; CDF1/A_full not returned",
        "sources": {name: str(path.relative_to(ROOT)) for name, path in
                    (("A0", SOURCE / "reference010"), ("TSA", SOURCE / "tsa/files"))},
        "reproduces_both_reported_validation_dictionaries": True,
        "thresholds": {name: row["threshold"] for name, row in summary.items()},
        "full": {name: compact(row) for name, row in metrics.items()},
        "by_source": {name: {group: compact(row[group]) for group in ("itsa", "mstsa")}
                      for name, row in metrics.items()},
        "common_paper_scope_original_thresholds": {name: compact(row) for name, row in restricted.items()},
        "paper_scope_coverage": coverage,
        "error_changes_A0_to_TSA": changes,
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
        "limits": ["V_select selected weights and thresholds; interval is descriptive, not independent test inference",
                   "Single seed, unequal optimization schedules and validation selection counts",
                   "TSA is offline adaptation, not original online system; native checkpoints remain on server",
                   "No zero-positive scenes; no T_test read; no new training or threshold fitting"],
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("error_changes_A0_to_TSA", "cluster_uncertainty", "score_quantiles")},
                     ensure_ascii=False, indent=2))
    print("error changes", {key: value for key, value in changes.items() if key != "per_scene"})
    print("cluster uncertainty", {key: value for key, value in report["cluster_uncertainty"].items()
                                  if key != "per_cluster"})


if __name__ == "__main__":
    analyze()
