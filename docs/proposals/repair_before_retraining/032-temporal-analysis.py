"""Read-only post-hoc temporal ablation of saved 032 predictions; no model forward.

All 64 scenes, 1635 fragments and 81207 valid queries retained. This is not
a trained alternative. Frozen predictions average over all 64 queries without
reading target values/masks. Required residual variation is oracle diagnostic.
"""
import io
import json
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "HandOver/results/COOPERATIVE-SPEED-20261020-032/completion-predictions.tar.gz"


def distance(a, b):
    la1, lo1, la2, lo2 = [np.deg2rad(v) for v in (a[..., 0], a[..., 1], b[..., 0], b[..., 1])]
    h = np.sin((la1-la2)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo1-lo2)/2)**2
    return 6371000*2*np.arctan2(np.sqrt(np.clip(h, 0, 1)), np.sqrt(np.clip(1-h, 0, 1)))


rows, sums = [], np.zeros(6)
with tarfile.open(ARCHIVE) as tf:
    for member in tf.getmembers():
        if not member.name.endswith(".npz"):
            continue
        a = np.load(io.BytesIO(tf.extractfile(member).read()))
        p, b, y, masks = a["prediction"].astype(float), a["position_base"].astype(float), a["target"], a["target_valid"]
        residual, required = p[..., :2]-b[..., :2], y[..., :2]-b[..., :2]
        residual[..., 1] = (residual[..., 1]+180) % 360-180
        required[..., 1] = (required[..., 1]+180) % 360-180
        frozen = b[..., :2]+residual.mean(axis=1, keepdims=True)
        frozen[..., 1] = (frozen[..., 1]+180) % 360-180
        lp = np.log(p[..., 2]+1e-6)
        frozen_lp = np.broadcast_to(lp.mean(axis=1, keepdims=True), lp.shape)
        v, s = masks[..., 0] & masks[..., 1], masks[..., 2]
        sums += [distance(p[v], y[v]).sum(), distance(frozen[v], y[v]).sum(), v.sum(),
                 np.square(lp[s]-np.log(y[..., 2][s]+1e-6)).sum(),
                 np.square(frozen_lp[s]-np.log(y[..., 2][s]+1e-6)).sum(), s.sum()]
        tau = a["query_seconds"]/20
        angles = tau[:, None]*np.exp(np.arange(0, 128, 2)*(-np.log(10000)/128))
        pe = np.stack((np.sin(angles), np.cos(angles)), axis=-1).reshape(len(tau), 128)
        for i in range(len(p)):
            # Local tangent conversion only for temporal spread, not MAE.
            scale = np.array([1, np.cos(np.deg2rad(y[i, v[i], 0].mean()))])*6371000*np.pi/180
            rows.append(dict(
                source=str(a["sources"][i]),
                query_pe_std=float(np.sqrt(np.mean(np.var(pe[v[i]], axis=0)))),
                predicted_position_residual_time_std_m=float(np.linalg.norm(np.std(residual[i, v[i]]*scale, axis=0))),
                required_position_residual_time_std_m=float(np.linalg.norm(np.std(required[i, v[i]]*scale, axis=0))),
                predicted_log_speed_time_std=float(np.std(lp[i, s[i]])),
                target_log_speed_time_std=float(np.std(np.log(y[i, s[i], 2]+1e-6))),
            ))
result = {}
for source in ("all", "9001", "9002"):
    subset = [r for r in rows if source == "all" or r["source"] == source]
    result[source] = dict(fragment_count=len(subset), statistics={
        k: dict(median=float(np.median([r[k] for r in subset])),
                p90=float(np.percentile([r[k] for r in subset], 90)))
        for k in rows[0] if k != "source"
    })
result["pooled"] = dict(
    original_position_mae=sums[0]/sums[2],
    time_constant_residual_position_mae=sums[1]/sums[2],
    position_count=int(sums[2]), original_log_speed_mse=sums[3]/sums[5],
    time_constant_log_speed_mse=sums[4]/sums[5], speed_count=int(sums[5]))
assert len(rows) == 1635 and int(sums[2]) == int(sums[5]) == 81207
result["scope"] = "Saved 032 predictions only, all V; no inference/training. Means use all Q64 without target/mask."
result["interpretation"] = [
    "Position temporal spread is a local tangent approximation; MAE uses spherical distance.",
    "Target-required residual is an oracle diagnostic, not a predictor.",
    "Temporal spreads use the same valid query subset per fragment.",
    "Query sinusoidal input variation is not a measurement of trained decoder hidden states.",
    "Negligible output temporal contribution does not isolate decoder, head or objective causes."]
target = Path(__file__).with_name("032-temporal-audit.json")
target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
print(json.dumps(result["pooled"], ensure_ascii=False))
