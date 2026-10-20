"""Recompute the returned 038 layer diagnostics; no model/data inference."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "HandOver/results/CANDIDATE-DYNAMICS-20261020-038"
LAYERS = ["decoder_input_centered_rms"] + [
    f"decoder_layer_{i}_centered_rms" for i in range(1, 7)
]


def summary(values):
    a = np.asarray(values, dtype=float)
    assert len(a) and np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                min=float(a.min()), max=float(a.max()))


def group(rows):
    layers = np.array([[r["metrics"][k] for k in LAYERS] for r in rows])
    assert np.all(layers > 0)
    return dict(
        fragments=len(rows),
        layers={key: summary(layers[:, i]) for i, key in enumerate(LAYERS)},
        final_over_input=summary(layers[:, -1] / layers[:, 0]),
        input_over_final=summary(layers[:, 0] / layers[:, -1]),
        ratio_of_means=float(layers[:, -1].mean() / layers[:, 0].mean()),
        successive_ratios={f"layer_{i}_over_previous": summary(layers[:, i] / layers[:, i-1])
                           for i in range(1, 7)},
        all_six_layers_decrease=int(np.all(np.diff(layers, axis=1) < 0, axis=1).sum()),
        query_counts=dict(
            total=sum(r["query_count"] for r in rows),
            before=sum(r["before_support_query_count"] for r in rows),
            inside=sum(r["inside_support_query_count"] for r in rows),
            after=sum(r["after_support_query_count"] for r in rows)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(gzip.decompress((SOURCE / "candidate-dynamics.json.gz").read_bytes()))
    run = json.loads(gzip.decompress((SOURCE / "run.json.gz").read_bytes()))
    status = json.loads((SOURCE / "status.json").read_text())
    request = json.loads((SOURCE / "request.json").read_text())
    rows = report["per_fragment"]
    assert run["status"] == status["status"] == "COMPLETED"
    assert int((SOURCE / "exit_code.txt").read_text()) == status["returncode"] == 0
    assert run["device"] == "cpu" and run["cpu_threads"] == 4
    assert not run["reference_members_read"] and not run["test_opened"]
    assert run["optimizer_updates"] == report["checks"]["optimizer_updates"] == 0
    assert run["fragment_count"] == report["fragment_count"] == report["completion_forward_count"] == len(rows) == 86
    assert run["scene_ids"] == [r["scene_id"] for r in report["per_scene"]]
    assert len({r["node_id"] for r in rows}) == 86
    assert all(r["query_count"] == 64 for r in rows)
    assert all(r["before_support_query_count"] + r["inside_support_query_count"]
               + r["after_support_query_count"] == r["query_count"] for r in rows)
    # Audit every reported aggregate against its returned fragment-level values.
    largest_aggregate_error = 0.
    for key, recorded in report["aggregate"].items():
        values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
        computed = summary(values)
        assert recorded["count"] == computed["n"]
        for name in ("mean", "median", "min", "max"):
            error = abs(recorded[name] - computed[name])
            largest_aggregate_error = max(largest_aggregate_error, error)
            np.testing.assert_allclose(recorded[name], computed[name], rtol=1e-12, atol=1e-12)
    for row in rows:
        m = row["metrics"]
        assert m["query_reconstruction_max_abs"] == m["physical_reconstruction_max_abs"] == 0
        assert m["final_hidden_centered_rms"] == m[LAYERS[-1]]
    d_model = 128  # Actual 034 config; not fitted from result.
    heads = {}
    for attribute in ("latitude", "longitude", "speed", "heading"):
        projected_errors = []
        residual_errors = []
        for row in rows:
            m = row["metrics"]
            expected = (m["final_hidden_centered_rms"] * np.sqrt(d_model)
                        * m[f"{attribute}_head_row_norm"]
                        * m[f"{attribute}_head_normalized_dynamic_projection"])
            projected_errors.append(abs(expected - m[f"{attribute}_head_projected_rms"]))
            residual_errors.append(abs(expected - m[f"{attribute}_residual_std"]))
        heads[attribute] = dict(
            row_norm=summary([r["metrics"][f"{attribute}_head_row_norm"] for r in rows]),
            normalized_projection=summary([
                r["metrics"][f"{attribute}_head_normalized_dynamic_projection"] for r in rows]),
            residual_std=summary([r["metrics"][f"{attribute}_residual_std"] for r in rows]),
            projected_formula_max_abs_error=max(projected_errors),
            fp32_residual_std_vs_float64_projection_max_abs_error=max(residual_errors))
    ratios = [(r["metrics"][LAYERS[-1]] / r["metrics"][LAYERS[0]], r["node_id"]) for r in rows]
    result = dict(
        source=str(SOURCE.relative_to(ROOT)), experiment="038", checkpoint_experiment="034",
        scope="Returned FP32 CPU self-forward summaries only; no new inference, target, training or T_test",
        execution=dict(exit_code=0, cpu_threads=4, updates=0, reference_members_read=[],
                       observation_member_count=len(run["observation_members_read"]),
                       status=status, request=request, payload_elapsed_seconds=run["elapsed_seconds"],
                       evaluation_elapsed_seconds=report["elapsed_seconds"]),
        aggregate_max_abs_recompute_error=largest_aggregate_error,
        overall=group(rows),
        per_scene={sid: group([r for r in rows if r["scene_id"] == sid]) for sid in run["scene_ids"]},
        by_source={sid: group([r for r in rows if r["source"] == sid])
                   for sid in sorted({r["source"] for r in rows})},
        native_count=summary([r["native_count"] for r in rows]),
        n2_fragments=sum(r["native_count"] == 2 for r in rows),
        ratio_extremes=dict(smallest=sorted(ratios)[:3], largest=sorted(ratios)[-3:]),
        heads=heads,
        remove_query_residual={key: value for key, value in report["aggregate"].items()
                               if key.startswith("remove_query_residual_")},
        interpretation=[
            "All six layer boundaries attenuate within-fragment query variation in every measured fragment.",
            "RMS amplitude loss is not a percent of information, explained variance or prediction accuracy.",
            "Layer summaries do not isolate self-attention, cross-attention, FFN or LayerNorm.",
            "Head normalized projection is a norm-normalized scalar projection, not retained information.",
            "One trained endpoint, four reused development scenes; not independent seeds or full-V layer evidence.",
            "038 uses true unwrapped pre-physical residuals; 034 full-V posthoc circular approximate-base analysis is complementary, not numerically identical.",
        ])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(overall=result["overall"], heads=heads), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
