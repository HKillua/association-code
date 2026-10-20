"""Audit returned circular-TCM diagnostics; no model, ZIP, or GPU access.

033 and 037 have different frozen feature spaces. Pool each endpoint against
its own state-hold control; compare only population, coverage, and measured cost
across endpoints. Reuses the existing report arithmetic, not its conclusions.
"""
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("audit033", HERE / "033-analysis.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
read, at, pooled, compare = prior.read, prior.at, prior.pooled, prior.compare
OLD = ROOT / "HandOver/results/BASELINE-TCM-COVERAGE-20261020-033"
NEW = ROOT / "HandOver/results/CIRCULAR-TCM-20261020-037"
COUNT_KEYS = ("query_count", "evaluable_query_count",
              "missing_reference_query_count", "feature_elements")


def close(a, b):
    assert a == b or (a is not None and b is not None and
                      math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10)), (a, b)


def outcome(row, path):
    m = at(row, path)["feature_mse"]
    h = at(row, (path[0], "state_hold", *path[2:]))["feature_mse"]
    if m is None or h is None:
        return "not_evaluable"
    return "better" if m < h else "worse" if m > h else "equal"


def heterogeneity(rows):
    paths = {
        "one_step": ("one_step", "model", "all"),
        "query": ("query", "model", "regions", "all"),
        "before_support": ("query", "model", "regions", "before_support"),
        "after_support": ("query", "model", "regions", "after_support"),
    }
    scenes = sorted({r["scene_id"] for r in rows})
    by_scene = {sid: compare([r for r in rows if r["scene_id"] == sid])
                for sid in scenes}
    by_source = {}
    for source in sorted({r["node_id"].split(":")[1] for r in rows}):
        part = [r for r in rows if r["node_id"].split(":")[1] == source]
        by_source[source] = {
            "fragments": len(part), "metrics": compare(part),
            "fragment_counts": {k: dict(Counter(outcome(r, p) for r in part))
                                for k, p in paths.items()},
        }
    all_metrics = compare(rows)
    for path, value in all_metrics.items():
        for method in ("model", "state_hold"):
            for key in (*COUNT_KEYS, "squared_error_sum"):
                close(sum(v["metrics"][path][method][key] for v in by_source.values()),
                      value[method][key])
    query = {sid: v["query/regions/all"] for sid, v in by_scene.items()}
    means = {m: statistics.mean(v[m]["feature_mse"] for v in query.values())
             for m in ("model", "state_hold")}
    equal = [r for r in rows if outcome(r, paths["query"]) == "equal"]
    fail = sorted(
        (dict(scene_id=sid, model_mse=v["model"]["feature_mse"],
              state_hold_mse=v["state_hold"]["feature_mse"],
              model_minus_hold_sse=v["model"]["squared_error_sum"] -
              v["state_hold"]["squared_error_sum"]) for sid, v in query.items()
         if v["model"]["feature_mse"] > v["state_hold"]["feature_mse"]),
        key=lambda x: x["model_minus_hold_sse"], reverse=True)
    return {
        "fragment_counts": {k: dict(Counter(outcome(r, p) for r in rows))
                            for k, p in paths.items()},
        "one_step_query_joint": [
            dict(one_step=a, query=b, fragment_count=n) for (a, b), n in sorted(
                Counter((outcome(r, paths["one_step"]), outcome(r, paths["query"]))
                        for r in rows).items())],
        "equal_query_without_evaluable_outside": sum(
            sum(r["query"]["model"]["regions"][k]["evaluable_query_count"]
                for k in ("before_support", "after_support")) == 0 for r in equal),
        "by_source": by_source,
        "scene_counts": {
            label: dict(Counter(
                "better" if v[key]["model"]["feature_mse"] <
                v[key]["state_hold"]["feature_mse"] else
                "worse" if v[key]["model"]["feature_mse"] >
                v[key]["state_hold"]["feature_mse"] else "equal"
                for v in by_scene.values()))
            for label, key in (("one_step", "one_step/all"),
                               ("query", "query/regions/all"))},
        "equally_weighted_scene_query": {
            **means, "model_vs_hold_reduction": 1 - means["model"] / means["state_hold"],
            "median_scene_reduction": statistics.median(
                v["model_vs_hold_reduction"] for v in query.values())},
        "failure_scenes": fail,
        "per_scene": {
            sid: {name: {
                "model_mse": value[key]["model"]["feature_mse"],
                "state_hold_mse": value[key]["state_hold"]["feature_mse"],
                "evaluable_queries": value[key]["model"]["evaluable_query_count"],
                "model_vs_hold_reduction": value[key]["model_vs_hold_reduction"],
            } for name, key in (("one_step", "one_step/all"), ("query", "query/regions/all"),
                                ("outside_support", "query/regions/outside_support"))}
            for sid, value in by_scene.items()},
    }


def endpoint(path, status_name):
    run, summary, history, report = (read(path / f"{name}.json.gz") for name in
                                    ("run", "summary", "history", "completion-evaluation"))
    status, request = read(path / status_name), read(path / "request.json")
    assert status["status"] == run["status"] == "COMPLETED"
    assert status["returncode"] == int((path / "exit_code.txt").read_text()) == 0
    assert run["phase"] == "tcm" and run["training_stages"] == ["TCM"]
    assert run["config"]["updates"] == dict(TCM=4000, TAM_FROZEN=0, TFM_FROZEN=0, JOINT=0)
    assert [(h["stage"], h["update"]) for h in history] == [("TCM", i) for i in range(1, 4001)]
    assert run["cursor"] == dict(stage=1, update=0, completed_updates=4000)
    assert run["checkpoint_cursor"] == dict(stage=0, update=4000, completed_updates=4000)
    assert not run["selection_performed"] and not run["validation_timings"]
    assert run["margin_calibration"] is None and run["effective_margin"] == 1
    assert run.get("initialization") is None and run["resume_from"] is None
    assert all(not d["test_opened"] for d in (run, summary, report))
    assert all(math.isfinite(h[k]) for h in history for k in ("loss", "gradient_norm", "seconds"))
    assert all(h["track_presentations"] == 64 for h in history)
    rows = report["per_fragment"]
    assert len({r["node_id"] for r in rows}) == len(rows) == report["fragment_count"] == 1635
    assert len({r["scene_id"] for r in rows}) == report["scene_count"] == 64
    for row in rows:
        assert row["node_id"].split(":")[0] == row["scene_id"]
        for p in prior.metric_paths():
            v = at(row, p)
            assert v["query_count"] == v["evaluable_query_count"] + v["missing_reference_query_count"]
            assert v["feature_elements"] == 128 * v["evaluable_query_count"]
            assert math.isfinite(v["squared_error_sum"]) and v["squared_error_sum"] >= 0
            close(v["feature_mse"], v["squared_error_sum"] / v["feature_elements"]
                  if v["feature_elements"] else None)
        for method in ("model", "state_hold"):
            one = row["one_step"][method]
            assert one["all"]["query_count"] == 2 * (row["native_count"] - 1)
            for k in (*COUNT_KEYS, "squared_error_sum"):
                close(one["all"][k], one["forward"][k] + one["reverse"][k])
            q = row["query"][method]
            for k in (*COUNT_KEYS, "squared_error_sum"):
                close(q["regions"]["all"][k], sum(q["regions"][r][k] for r in
                      ("inside_support", "before_support", "after_support")))
                close(q["regions"]["all"][k], sum(v[k] for v in q["input_distance_metrics"].values()))
        assert row["query"]["model"]["regions"]["inside_support"] == row["query"]["state_hold"]["regions"]["inside_support"]
    for p in prior.metric_paths():
        for key, v in pooled(rows, p).items():
            close(v, at(report, p)[key])
    assert summary["completion_evaluation"] == "completion-evaluation.json"
    assert run["final_evaluation_status"] == "COMPLETED"
    close(run["final_evaluation_seconds"], summary["evaluation_seconds"])
    updates = sum(h["seconds"] for h in history)
    coverage = summary["coverage"]["TCM"]
    assert coverage["track_presentations"] == sum(h["track_presentations"] for h in history) == 256000
    timing = dict(launcher_seconds=status["elapsed_seconds"], trainer_seconds=run["elapsed_seconds"],
                  update_seconds=updates, mean_update_seconds=updates / len(history),
                  median_update_seconds=statistics.median(h["seconds"] for h in history),
                  evaluation_seconds=run["final_evaluation_seconds"],
                  other_trainer_seconds=run["elapsed_seconds"] - updates - run["final_evaluation_seconds"])
    windows = {
        f"{lo}-{hi}": dict(mean_loss=statistics.mean(h["loss"] for h in history[lo-1:hi]),
                           mean_update_seconds=statistics.mean(h["seconds"] for h in history[lo-1:hi]))
        for lo, hi in ((1, 100), (1, 1000), (1001, 2000), (2001, 3000), (3001, 4000), (3901, 4000))}
    return dict(run=run, summary=summary, report=report, status=status, request=request,
                timing=timing, training_windows=windows,
                clipped_steps=sum(h["gradient_norm"] > run["config"]["gradient_clip"] for h in history),
                gradient_max=max(h["gradient_norm"] for h in history))


def main():
    old, new = endpoint(OLD, "launcher-status.json"), endpoint(NEW, "status.json")
    old_rows = {r["node_id"]: r for r in old["report"]["per_fragment"]}
    new_rows = {r["node_id"]: r for r in new["report"]["per_fragment"]}
    assert old_rows.keys() == new_rows.keys()
    for key, r in old_rows.items():
        other = new_rows[key]
        for field in ("scene_id", "node_id", "node_index", "identity", "native_count",
                      "reference_declared", "has_complete_reference"):
            assert r[field] == other[field]
        for p in prior.metric_paths():
            for field in COUNT_KEYS:
                assert at(r, p)[field] == at(other, p)[field]
    assert old["summary"]["normalizer"] == new["summary"]["normalizer"]
    assert old["summary"]["coverage"] == new["summary"]["coverage"]
    assert new["summary"]["coverage"] == read(NEW / "coverage.json.gz")
    actual = read(NEW / "actual_tcm_config.json")
    assert actual == read(ROOT / "configs/tmtaff-circular-tcm-037.json")
    assert actual["model"] == new["run"]["requested_model_config"]
    assert actual["training"] == new["run"]["config"]
    assert actual["model"]["heading_representation"] == "circular_sincos"
    assert new["request"]["time_policy"] == "advisory" and new["request"]["deadline_monotonic"] is None
    progress = read(NEW / "evaluation-progress.json")
    assert (progress["completed_scenes"], progress["total_scenes"],
            progress["completed_fragments"]) == (64, 64, 1635)
    argv = new["request"]["argv"]
    assert argv[argv.index("--phase")+1] == "tcm"
    assert argv[argv.index("--config")+1].endswith("/configs/tmtaff-circular-tcm-037.json")
    assert argv[argv.index("--cpu-threads")+1] == "4"
    flat_diffs = {}
    def flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from flatten(v, key)
            else:
                yield key, v
    a, b = dict(flatten(read(OLD / "actual_config.json"))), dict(flatten(actual))
    for k in sorted(a.keys() | b.keys()):
        if a.get(k) != b.get(k):
            flat_diffs[k] = dict(old033=a.get(k), new037=b.get(k))
    coverage = new["summary"]["coverage"]["TCM"]
    out = {
        "scope": "Returned per-fragment SSE/count audit only; no weights, ZIP, inference, or training.",
        "verification": {
            "actual_command": argv, "completed_updates": 4000, "history_contiguous_and_finite": True,
            "exit_code": 0, "phase": "tcm", "TAM_TFM_JOINT_updates": 0,
            "selection_performed": False, "test_opened": False,
            "all_24_metric_groups_reaggregated": True,
            "per_fragment_direction_region_distance_additivity_checked": True,
            "population_metadata_and_denominators_equal_to033": True,
            "raw_normalizer_and_complete_coverage_equal_to033": True,
            "checkpoint_loaded_locally": False, "actual_config_matches": True,
            "config_differences": flat_diffs,
        },
        "population": {"scenes": 64, "fragments": 1635,
                       "one_step_queries": 88832, "natural_queries": 104640,
                       "complete_reference_queries": 81207, "missing_reference_queries": 23433},
        "training": {"scenes": len(coverage["scenes"]), "unique_tracks": len(coverage["tracks"]),
                     "track_presentations": coverage["track_presentations"],
                     "windows037": new["training_windows"],
                     "clipped_steps037": new["clipped_steps"], "gradient_max037": new["gradient_max"]},
        "timing": {"old033": old["timing"], "new037": new["timing"],
                   "observed_relative_change": {k: new["timing"][k]/old["timing"][k]-1
                                                for k in ("launcher_seconds", "update_seconds", "evaluation_seconds")},
                   "interpretation": "Same timing definitions only; hardware/occupancy were not controlled, so changes are not isolated representation costs."},
        "full_V_037_in_own_space": compare(new["report"]["per_fragment"]),
        "full_V_033_in_own_space_reference_only": compare(old["report"]["per_fragment"]),
        "heterogeneity037": heterogeneity(new["report"]["per_fragment"]),
        "limits": [
            "The two representations define different latent targets and geometries; neither raw MSE nor model/hold ratios prove cross-representation superiority.",
            "Inside-support output equals observation interpolation; it is not learned completion.",
            "One-step supervision uses native observations; natural query supervision uses reference. Their disagreement alone does not isolate recursive error.",
            "Source IDs, scenes, and fragments are descriptive correlated samples, not independent repeats.",
            "No physical decoding, association scores, fusion, checkpoint tensor check, or T_test evaluation occurred locally.",
            "We reaggregate returned fragment SSE; no pointwise predictions were returned for independent residual reconstruction.",
        ],
    }
    for name, key in (("one_step_queries", ("one_step", "model", "all")),
                      ("natural_queries", ("query", "model", "regions", "all"))):
        assert out["population"][name] == at(new["report"], key)["query_count"]
    assert out["population"]["complete_reference_queries"] == new["report"]["query"]["model"]["regions"]["all"]["evaluable_query_count"]
    assert out["population"]["missing_reference_queries"] == new["report"]["query"]["model"]["regions"]["all"]["missing_reference_query_count"]
    (HERE / "037-audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(timing=out["timing"], training=out["training"],
                         fragment_counts=out["heterogeneity037"]["fragment_counts"],
                         scene_counts=out["heterogeneity037"]["scene_counts"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
