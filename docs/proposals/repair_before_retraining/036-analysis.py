"""Recompute the frozen-TCM feedback diagnostic from returned per-target SSE.

No model forward, original archive access, selection or training.
"""
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "HandOver/results/TCM-NATIVE-FEEDBACK-20261020-036"
METHODS = ("oracle_feedback", "free_feedback", "state_hold")


def read(name):
    with gzip.open(SOURCE / name, "rt") as handle:
        return json.load(handle)


raw, run = read("reference-feedback.json.gz"), read("run.json.gz")
status = json.loads((SOURCE / "status.json").read_text())
assert status["status"] == run["status"] == "COMPLETED"
assert status["returncode"] == int((SOURCE / "exit_code.txt").read_text()) == 0
assert run["reference_cursor"] == dict(stage=0, update=4000, completed_updates=4000)
assert run["optimizer_updates"] == 0 and not run["reference_members_read"]
assert not any(run[key] for key in ("test_opened", "normalizer_refit",
                                    "margin_recalibrated", "threshold_reselected"))
assert raw["fragment_count"] == len(raw["per_fragment"]) == run["fragment_count"] == 86
assert len({row["node_id"] for row in raw["per_fragment"]}) == 86
assert raw["scene_count"] == len(raw["per_scene"]) == 4
assert raw["first_step_check"] == dict(direction_count=164, all_identical=True,
                                      max_abs_difference=0.)
flat = []
for fragment in raw["per_fragment"]:
    for direction, row in fragment["directions"].items():
        assert row["target_count"] == min(3, fragment["native_count"] - 2)
        assert row["prefix_count"] + row["target_count"] == fragment["native_count"]
        for step in row["per_step"]:
            assert all(step["metrics"][method]["target_count"] == 1
                       and step["metrics"][method]["feature_elements"] == 128 for method in METHODS)
            assert math.isclose(step["horizon_seconds"],
                                abs(step["target_seconds"] - row["prefix_endpoint_seconds"]))
            if step["step"] == 1:
                assert step["metrics"]["oracle_feedback"] == step["metrics"]["free_feedback"]
                assert step["feedback_difference"]["squared_error_sum"] == 0
            flat.append(dict(node_id=fragment["node_id"], scene_id=fragment["scene_id"],
                             source=fragment["source"], direction=direction,
                             eligible_h3=row["target_count"] == 3, **step))


def summarize(rows):
    metrics = {}
    for method in METHODS:
        sse = math.fsum(row["metrics"][method]["squared_error_sum"] for row in rows)
        elements = sum(row["metrics"][method]["feature_elements"] for row in rows)
        metrics[method] = dict(sse=sse, elements=elements, mse=sse/elements if elements else None)
    oracle, free, hold = (metrics[method]["mse"] for method in METHODS)
    by_node = defaultdict(list)
    for row in rows:
        by_node[row["node_id"]].append(row)
    counts, failed = {}, {}
    for left, right in (("free_feedback", "oracle_feedback"),
                        ("free_feedback", "state_hold"), ("oracle_feedback", "state_hold")):
        change = Counter()
        failed[left + "_vs_" + right] = set()
        for node, entries in by_node.items():
            delta = math.fsum(row["metrics"][left]["squared_error_sum"]
                             - row["metrics"][right]["squared_error_sum"] for row in entries)
            label = "better" if delta < -1e-12 else "worse" if delta > 1e-12 else "equal"
            change[label] += 1
            if label == "worse":
                failed[left + "_vs_" + right].add(node)
        counts[left + "_vs_" + right] = dict(change)
    horizons = [row["horizon_seconds"] for row in rows]
    return dict(target_count=len(rows), fragment_count=len(by_node), metrics=metrics,
                free_relative_to_oracle=free/oracle-1 if oracle else None,
                oracle_reduction_relative_to_free=1-oracle/free if free else None,
                free_reduction_relative_to_hold=1-free/hold if hold else None,
                fragment_comparisons=counts,
                both_feedback_routes_worse_than_hold=len(
                    failed["free_feedback_vs_state_hold"] & failed["oracle_feedback_vs_state_hold"]),
                horizon_seconds=dict(min=min(horizons), median=statistics.median(horizons),
                                     max=max(horizons)) if horizons else None)


result = dict(
    source=str(SOURCE.relative_to(ROOT) / "reference-feedback.json.gz"),
    scope="four exposed V scenes; same frozen 033 endpoint; native targets, not reference or full-query evaluation",
    execution=dict(payload_elapsed_seconds=run["elapsed_seconds"],
                   launcher_elapsed_seconds=status["elapsed_seconds"], optimizer_updates=0,
                   test_opened=False, reference_members_read=[]),
    population=dict(fragments=86, scenes=raw["per_scene"],
                    statuses=dict(Counter(row["status"] for row in raw["per_fragment"]))),
    first_step_check=raw["first_step_check"],
    all_steps=summarize(flat),
    h2_h3=summarize([row for row in flat if row["step"] > 1]),
    per_step={str(h): summarize([row for row in flat if row["step"] == h]) for h in (1, 2, 3)},
    same_h3_eligible_population={
        str(h): summarize([row for row in flat if row["step"] == h and row["eligible_h3"]])
        for h in (1, 2, 3)},
    by_scene={}, by_direction={}, by_source={},
)
for field, key in (("scene_id", "by_scene"), ("direction", "by_direction"), ("source", "by_source")):
    for value in sorted({row[field] for row in flat}):
        selected = [row for row in flat if row[field] == value]
        result[key][value] = dict(
            h2_h3=summarize([row for row in selected if row["step"] > 1]),
            per_step={str(h): summarize([row for row in selected if row["step"] == h])
                      for h in (1, 2, 3)})

# Check every reported direction/step aggregate against independently pooled rows.
for direction in ("all", "forward", "reverse"):
    selected = flat if direction == "all" else [row for row in flat if row["direction"] == direction]
    for h in (None, 1, 2, 3):
        pooled = summarize(selected if h is None else [row for row in selected if row["step"] == h])
        cell = raw["aggregate"][direction]
        saved = cell["all_steps"] if h is None else cell["per_step"][str(h)]
        for method in METHODS:
            actual, expected = pooled["metrics"][method], saved["metrics"][method]
            assert pooled["target_count"] == expected["target_count"]
            assert actual["elements"] == expected["feature_elements"]
            assert math.isclose(actual["sse"], expected["squared_error_sum"], rel_tol=1e-12)
            assert math.isclose(actual["mse"], expected["feature_mse"], rel_tol=1e-12)
result["aggregate_reconstruction_passed"] = True
output = Path(__file__).with_suffix(".json")
output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(output)
for step, cell in result["per_step"].items():
    print(step, cell["target_count"],
          {name: cell["metrics"][name]["mse"] for name in METHODS},
          "free/oracle-1", cell["free_relative_to_oracle"])
