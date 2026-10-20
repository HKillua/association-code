"""Audit 035 returned association caches, retaining every recorded threshold.

CPU array arithmetic only: no model, original archive, training, new threshold
selection or test access. 020 has aggregates only; no paired claims for 020.
"""
from collections import Counter
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from association.evaluation.association import evaluate_scenes, grouping_diagnostics

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "HandOver/results/BASELINE-TAM-FROM033-20261020-035"
OLD = ROOT / "HandOver/results/STAGEWISE-COMPLETION-20261019-028"
METRICS = ("scene_count", "pair_count", "macro_f1", "micro_f1", "micro_precision",
           "micro_recall", "tp", "fp", "fn", "tn", "p_fa")


def read(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as handle:
        return json.load(handle)


def caches(path):
    out = {}
    with tarfile.open(path) as archive:
        for name in archive.getnames():
            if not name.endswith("/index.json"):
                continue
            index = json.load(archive.extractfile(name))
            records = []
            for entry in index["records"]:
                member = str(Path(name).parent / entry["file"])
                with np.load(io.BytesIO(archive.extractfile(member).read())) as arrays:
                    record = {key: arrays[key] for key in arrays.files}
                record.update(scene_id=entry["scene_id"], component_id=entry["component_id"],
                              node_ids=np.asarray(entry["node_ids"]))
                record["labels"] = record["labels"].astype(bool)
                assert np.isfinite(record["scores"]).all()
                n = len(record["sources"])
                assert len(record["pairs"]) == n * (n - 1) // 2
                if "identities" in record:
                    pairs = record["pairs"]
                    assert np.array_equal(record["labels"],
                        record["identities"][pairs[:, 0]] == record["identities"][pairs[:, 1]])
                records.append(record)
            out[index["binding"].get("update", 1000)] = (records, index["binding"])
    return out


def auc(labels, scores):
    # Mann–Whitney ROC AUC, average ranks for exact ties.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - counts + 1 + ends) / 2
    p, n = int(labels.sum()), int((~labels).sum())
    return float((ranks[inverse][labels].sum() - p * (p + 1) / 2) / (p * n)) if p and n else None


def quantiles(values):
    return dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"),
                    map(float, np.quantile(values, [0, .05, .25, .5, .75, .95, 1]))))


def describe(records, threshold, per_scene=False):
    metrics = evaluate_scenes(records, threshold)
    labels = np.concatenate([r["labels"] for r in records])
    scores = np.concatenate([r["scores"] for r in records])
    scene_auc = [auc(r["labels"], r["scores"]) for r in records]
    valid_auc = [value for value in scene_auc if value is not None]
    result = {key: metrics[key] for key in METRICS}
    result.update(threshold=threshold, pooled_roc_auc=auc(labels, scores),
                  scene_macro_roc_auc=float(np.mean(valid_auc)) if valid_auc else None,
                  auc_defined_scene_count=len(valid_auc),
                  false_positive_rate=metrics["fp"] / (metrics["fp"] + metrics["tn"]),
                  positive_prior=float(labels.mean()),
                  score_quantiles={"positive": quantiles(scores[labels]),
                                   "negative": quantiles(scores[~labels])},
                  unique_score_count=int(len(np.unique(scores))),
                  exact_zero_score_count=int(np.count_nonzero(scores == 0)),
                  paper_metrics=metrics["paper_metrics"])
    for group in ("itsa", "mstsa"):
        result[group] = {key: metrics[group][key] for key in METRICS}
        masks = [r["sources"][r["pairs"][:, 0]] == r["sources"][r["pairs"][:, 1]]
                 for r in records]
        masks = masks if group == "itsa" else [~mask for mask in masks]
        result[group]["pooled_roc_auc"] = auc(
            np.concatenate([r["labels"][m] for r, m in zip(records, masks)]),
            np.concatenate([r["scores"][m] for r, m in zip(records, masks)]))
    if per_scene:
        result["per_scene"] = [{**row, "roc_auc": value}
                               for row, value in zip(metrics["per_scene"], scene_auc)]
    return result


def same_population(left, right):
    other = {r["scene_id"]: r for r in right}
    assert {r["scene_id"] for r in left} == set(other)
    for row in left:
        match = other[row["scene_id"]]
        for key in ("labels", "pairs", "sources", "node_ids"):
            assert np.array_equal(row[key], match[key]), (row["scene_id"], key)


def paired(reference, reference_threshold, candidate, candidate_threshold):
    """Count error changes when replacing the reference by 035."""
    same_population(reference, candidate)
    other = {r["scene_id"]: r for r in candidate}
    totals = Counter()
    scene_changes = Counter()
    for row in reference:
        match = other[row["scene_id"]]
        truth = row["labels"]
        a = row["scores"] >= reference_threshold
        b = match["scores"] >= candidate_threshold
        totals["corrected_errors"] += int(np.count_nonzero((a != truth) & (b == truth)))
        totals["new_errors"] += int(np.count_nonzero((a == truth) & (b != truth)))
        totals["new_fp"] += int(np.count_nonzero(~truth & ~a & b))
        totals["removed_fp"] += int(np.count_nonzero(~truth & a & ~b))
        totals["new_fn"] += int(np.count_nonzero(truth & a & ~b))
        totals["removed_fn"] += int(np.count_nonzero(truth & ~a & b))
        fa = evaluate_scenes([row], reference_threshold)["macro_f1"]
        fb = evaluate_scenes([match], candidate_threshold)["macro_f1"]
        scene_changes["better" if fb > fa + 1e-12 else "worse" if fb < fa - 1e-12 else "equal"] += 1
    totals["net_additional_errors"] = totals["new_errors"] - totals["corrected_errors"]
    return dict(direction="comparator -> selected035", **totals, scene_f1_changes=dict(scene_changes))


run, summary, history = (read(SOURCE / name) for name in
                         ("run.json.gz", "summary.json.gz", "history.json.gz"))
status, request = read(SOURCE / "status.json"), read(SOURCE / "request.json")
config, coverage = read(SOURCE / "actual_config.json"), read(SOURCE / "coverage.json.gz")
assert status["status"] == run["status"] == "COMPLETED"
assert status["returncode"] == int((SOURCE / "exit_code.txt").read_text()) == 0
assert run["phase"] == "tam" and run["training_stages"] == ["TAM_FROZEN"]
assert run["config"] == config["training"]
assert run["initialization"]["source_tcm_updates"] == 4000
assert not run["test_opened"] and not summary["test_opened"]
assert len(history) == 1000 and [r["update"] for r in history] == list(range(1, 1001))
assert all(r["stage"] == "TAM_FROZEN" and r["pair_count"] == 64
           and r["positive_pairs"] == 32 for r in history)
assert all(np.isfinite([r["loss"], r["gradient_norm"], r["seconds"]]).all() for r in history)
assert config["training"]["updates"] == dict(TCM=0, TAM_FROZEN=1000, TFM_FROZEN=0, JOINT=0)
assert summary["evaluation"]["inference_reused"] and not summary["evaluation"]["physical_fusion_evaluated"]
source_summary = read(ROOT / "HandOver/results/BASELINE-TCM-COVERAGE-20261020-033/summary.json.gz")
assert summary["normalizer"] == source_summary["normalizer"]

records_by_update = caches(SOURCE / "association-scores.tar.gz")
assert sorted(records_by_update) == list(range(100, 1001, 100))
selected_update = summary["selection"]["update"]
selected_threshold = summary["selection"]["threshold"]
selected_records = records_by_update[selected_update][0]
curve = []
for row in summary["validation_score_records"]:
    update, selection = row["update"], row["selection"]
    records, binding = records_by_update[update]
    same_population(selected_records, records)
    assert binding["threshold"] == selection["threshold"]
    actual = describe(records, selection["threshold"])
    for key in METRICS:
        assert np.isclose(actual[key], row["association"][key], atol=1e-12, rtol=1e-12), (update, key)
    for group in ("itsa", "mstsa"):
        for key in METRICS:
            assert np.isclose(actual[group][key], row["association"][group][key], atol=1e-12, rtol=1e-12)
    per100 = history[update - 100:update]
    curve.append(dict(update=update, **{k: actual[k] for k in METRICS},
        threshold=selection["threshold"], pooled_roc_auc=actual["pooled_roc_auc"],
        scene_macro_roc_auc=actual["scene_macro_roc_auc"], score_quantiles=actual["score_quantiles"],
        loss_mean=float(np.mean([r["loss"] for r in per100])),
        gradient_norm_mean=float(np.mean([r["gradient_norm"] for r in per100])),
        gradient_clip_fraction=float(np.mean([r["gradient_norm"] > 1 for r in per100])),
        training_seconds=sum(r["seconds"] for r in per100),
        validation_seconds=history[update - 1]["validation_seconds"]))
assert selected_update == max(curve, key=lambda r: r["macro_f1"])["update"] == 600
selected = describe(selected_records, selected_threshold, per_scene=True)
for key in METRICS:
    assert np.isclose(selected[key], summary["evaluation"]["association"][key], atol=1e-12, rtol=1e-12)

cal = run["margin_calibration"]
distances = np.concatenate([s["distances"] for s in cal["samples"]])
labels = np.concatenate([s["labels"] for s in cal["samples"]]).astype(bool)
assert len(cal["samples"]) == cal["completed_batches"] == 64
assert len(distances) == 4096 and int(labels.sum()) == 2048
assert np.isfinite(distances).all()
margin = float(np.median(distances[~labels]))
assert margin == run["effective_margin"] == cal["effective_margin"]
selected_distances = -np.concatenate([r["scores"] for r in selected_records]).astype(np.float64)
selected_labels = np.concatenate([r["labels"] for r in selected_records])
selected_hinge = np.maximum(np.float32(margin) - selected_distances[~selected_labels], 0.)
mode_discrepancy = dict(
    eval_positive_distance_mean=float(selected_distances[selected_labels].mean()),
    eval_negative_distance_mean=float(selected_distances[~selected_labels].mean()),
    eval_negative_hinge_active_fraction=float(np.mean(selected_hinge > 0)),
    balanced_eval_cl_from_cached_scores=float(.25 * (
        selected_distances[selected_labels].mean() + np.square(selected_hinge).mean())),
    corresponding_updates_501_600_train_loss_mean=float(np.mean([r["loss"] for r in history[500:600]])),
    scope="FP32 cached eval scores replayed in float64; equal class weights and training margin. "
          "Different train/V populations and modes; does not isolate dropout or establish a code bug.")

prior = read(HERE / "association-distance-local-check.json")
with np.load(HERE / "association-distance-local-records.npz") as data:
    control = [dict(scene_id=str(sid), **{
        key: data[f"{sid}_{key}"] for key in ("scores", "labels", "pairs", "sources", "node_ids")})
        for sid in data["scene_ids"]]
candidate_records, _ = caches(OLD / "021/identity-records.tar.gz")[1000]
candidate_run = read(OLD / "021/run.json.gz")
candidate_threshold = candidate_run["identity_selection"]["threshold"]
same_population(selected_records, candidate_records)
same_population(selected_records, control)
old_run, old_history = read(OLD / "020/run.json.gz"), read(OLD / "020/history.json.gz")
old_selections = [r for r in old_history if "selection" in r]
old_selected = max(old_selections, key=lambda r: r["selection"]["macro_f1"])
old_metadata = json.loads(subprocess.check_output(["git", "show",
    "37f3dd5:HandOver/results/STAGEWISE-COMPLETION-20261019-028/020/020-before-stop-checkpoint-metadata.json"],
    cwd=ROOT))
old_validation = old_metadata["020"]["selected_snapshot"]["validation"]
for a, b in zip(sorted(old_validation["per_scene"], key=lambda r: r["scene_id"]),
                sorted(selected["per_scene"], key=lambda r: r["scene_id"])):
    for key in ("scene_id", "node_count", "pair_count", "positive_count"):
        assert a[key] == b[key]

# Graph closure is a sensitivity diagnostic, not an executed fusion evaluation.
graph = [grouping_diagnostics(r, selected_threshold) for r in selected_records]
grouping = {key: int(sum(r[key] for r in graph)) for key in
            ("component_count", "singleton_count", "mixed_identity_components",
             "fragmented_truth_identities", "false_bridge_edges", "added_closure_fp")}
grouping["truth_identity_count"] = sum(len(np.unique(r["identities"])) for r in selected_records)
grouping["closure_counts"] = {key: sum(r["closure"][key] for r in graph) for key in ("tp", "fp", "fn", "tn")}
grouping["scope"] = "connected-component closure of same selected pair decisions; no fusion model run"

# Actual visible intervals, without excluding any pair or selecting a new threshold.
temporal = {}
for name, lo, hi in (("overlap", -1, 0), ("gap_0_300", 0, 300),
                     ("gap_300_600", 300, 600), ("gap_600_1800", 600, 1800),
                     ("gap_above_1800", 1800, np.inf)):
    subset = []
    for record in selected_records:
        pairs, intervals = record["pairs"], record["intervals"]
        gap = np.maximum(0., np.maximum(intervals[pairs[:, 0], 0], intervals[pairs[:, 1], 0])
                         - np.minimum(intervals[pairs[:, 0], 1], intervals[pairs[:, 1], 1]))
        mask = (gap > lo) & (gap <= hi)
        subset.append({**record, **{key: record[key][mask] for key in ("scores", "labels", "pairs")}})
    cell = evaluate_scenes(subset, selected_threshold)
    temporal[name] = {key: cell[key] for key in METRICS}
    temporal[name]["pooled_roc_auc"] = auc(np.concatenate([r["labels"] for r in subset]),
                                        np.concatenate([r["scores"] for r in subset]))

result = dict(
    source=str(SOURCE.relative_to(ROOT)), command=request["argv"], scope="V_select development",
    execution=dict(returncode=status["returncode"], status=run["status"],
        launcher_seconds=status["elapsed_seconds"], payload_seconds=run["elapsed_seconds"],
        training_seconds=sum(r["seconds"] for r in history),
        calibration_seconds=cal["seconds"],
        validation_seconds=sum(r["seconds"] for r in run["validation_timings"]),
        final_cache_metrics_seconds=run["final_evaluation_seconds"],
        test_opened=False, initialization=run["initialization"],
        checkpoint_cursor=run["checkpoint_cursor"], cursor=run["cursor"],
        time_policy=request["time_policy"], physical_fusion_evaluated=False),
    config=config,
    population=dict(scenes=len(selected_records), nodes=sum(len(r["sources"]) for r in selected_records),
        pairs=selected["pair_count"], positives=selected["tp"] + selected["fn"],
        components=len({r["component_id"] for r in selected_records}),
        complete_i_lt_j=True, ten_caches_aligned=True, aligned_with_position_control_and_021=True,
        old020_per_scene_counts_aligned=True),
    calibration=dict(seed=cal["seed"], batch_count=64, pair_presentations=len(distances),
        positive_presentations=int(labels.sum()), negative_presentations=int((~labels).sum()),
        scope=cal["scope"], negative_median_recomputed=margin,
        positive_distance_quantiles=quantiles(distances[labels]),
        negative_distance_quantiles=quantiles(distances[~labels]),
        old020_effective_margin=old_run["effective_margin"]),
    coverage={k: v for k, v in coverage["TAM_FROZEN"].items() if not isinstance(v, (list, dict))},
    selected_update=selected_update, selected=selected, curve=curve,
    temporal_subsets=temporal, grouping=grouping,
    train_eval_scale_observation=mode_discrepancy,
    comparators=dict(
        old020=dict(update=old_selected["update"], tcm_updates=1000,
            threshold=old_selected["selection"]["threshold"],
            **{key: old_validation[key] for key in METRICS},
            limitation="aggregate only; no paired score/AUC verification"),
        position_control=describe(control, prior["selection"]["threshold"]),
        candidate021=describe(candidate_records, candidate_threshold)),
    paired_vs_position_control=paired(control, prior["selection"]["threshold"], selected_records, selected_threshold),
    paired_vs_021=paired(candidate_records, candidate_threshold, selected_records, selected_threshold),
    checks=dict(history_1000_finite=True, all_ten_cache_metrics_match=True,
        negative_median_matches=True, selected_best_of_original_ten=True,
        source033_normalizer_equal=True,
        threshold_reselected=False, model_constructed=False, raw_archive_read=False))
output = HERE / "035-audit.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(selected_update=selected_update,
    selected={key: selected[key] for key in (*METRICS, "pooled_roc_auc", "scene_macro_roc_auc")},
    grouping=grouping, paired_vs_position_control=result["paired_vs_position_control"],
    paired_vs_021=result["paired_vs_021"]), ensure_ascii=False, indent=2))
