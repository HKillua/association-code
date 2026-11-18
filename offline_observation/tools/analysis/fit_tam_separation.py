"""094F: evaluate original TAM endpoints on a fixed fit-side natural pair population."""

from pathlib import Path

import numpy as np
import torch
from association.data.archive import MTADArchive
from association.data.legacy_dataset import validate_archive
from association.data.timed import prepare_timed_scene
from association.training.common import normalize_scene
from association.training.tmtaff import load_complete_snapshot
from tools.analysis import association_endpoint_qualification as parent
from tools.analysis.association_frozen_pr_audit import read_json, require, score_groups
from tools.analysis.held_fusion_population import _canonical_tracks
from tools.analysis.offline_baseline_restart import digest, write
from tools.analysis.saved_replay_runtime import execute

RUN = "LEVEL1-COMPLETION-VALUE-TAM-20261024-094{}-attempt-1"


def measurements(archive, sid, motion, config):
    with np.load(motion, allow_pickle=False) as data:
        nodes, src, query = (data[k] for k in ("node_ids", "sources", "query_seconds"))
    tracks = _canonical_tracks(archive._rows(sid, "measurements"), sid, nodes, src)
    for track in tracks:
        track["identity_id"] = track["node_id"]
    intervals = np.asarray(
        [[t["canonical_times"][0], t["canonical_times"][-1]] for t in tracks]
    ).reshape(-1, 2)
    native = dict(
        scene_id=sid,
        split="T_core",
        tracks=tracks,
        node_ids=nodes,
        sources=src,
        identities=nodes,
        intervals=intervals,
    )
    return prepare_timed_scene(
        native,
        query_times=query,
        **{k: config[k] for k in ("length", "max_observations", "time_unit_seconds")},
    )


@torch.no_grad()
def predict(model, values, observed, pairs):
    model.requires_grad_(False).eval()
    parameter = next(model.parameters())
    embeds = []
    for start in range(0, len(values), 4):
        x = torch.as_tensor(
            values[start : start + 4], dtype=parameter.dtype, device=parameter.device
        )
        v = torch.as_tensor(observed[start : start + 4], device=parameter.device)
        embeds.append(model.tam_encoder(model.complete(x, v)).flatten(1))
    if not len(pairs):
        return np.empty(0, np.float32)
    embeddings = torch.cat(embeds)
    require(bool(torch.isfinite(embeddings).all()), "nonfinite embedding")
    scores = np.concatenate(
        [
            model.score_pairs(embeddings, pairs[i : i + 1024]).cpu().numpy()
            for i in range(0, len(pairs), 1024)
        ]
    )
    require(np.isfinite(scores).all(), "nonfinite score")
    return scores


def ranking(scores, truth):
    pos = int(truth.sum())
    neg = len(truth) - pos
    result = dict(
        positive=pos,
        negative=neg,
        prior=pos / len(truth) if len(truth) else None,
        unique_scores=len(np.unique(scores)),
        auc=None,
        ap=None,
    )
    for name, sub in [("positive", truth), ("negative", ~truth)]:
        result[name + "_distance_p05_p50_p95"] = (
            np.quantile(-scores[sub], [0.05, 0.5, 0.95]).tolist() if sub.any() else None
        )
    if pos and neg:
        _, n, tp = score_groups(scores, truth)
        fp = n - tp
        result["auc"] = float(
            np.sum(
                np.diff(fp, prepend=0) * (tp + np.r_[0, tp[:-1]]) / (2.0 * pos * neg)
            )
        )
        result["ap"] = float(np.sum(np.diff(tp, prepend=0) * (tp / n)) / pos)
    return result


def stats(rows):
    out = {}
    for rel in ("cross", "same"):
        slices = [r["cross"] if rel == "cross" else ~r["cross"] for r in rows]
        truth = np.concatenate(
            [r["truth"][take] for r, take in zip(rows, slices, strict=True)]
        )
        out[rel] = {
            a: ranking(
                np.concatenate(
                    [r[a][take] for r, take in zip(rows, slices, strict=True)]
                ),
                truth,
            )
            for a in "AB"
        }
    return out


def run(runs, out):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    snapshots, contracts, bindings = {}, {}, {}
    manifest_path = Path("configs/data/mtad-level1-scene3000.json")
    manifest, archives, sampled, _, _ = parent.bind_population(runs, manifest_path)
    scene_ids = list(sampled.values())[:8]
    private = out / "private"
    private.mkdir(mode=0o700)
    for arm, policy in [("A", "learned"), ("B", "state_hold")]:
        root = runs / RUN.format(arm)
        p = root / "checkpoint.pt"
        raw = torch.load(p, map_location="cpu", weights_only=True)
        require(
            raw["cursor"]["completed_updates"] == 4000
            and raw["completion_policy"] == policy,
            "terminal contract",
        )
        metadata = (
            "node_ids",
            "sources",
            "grid",
            "intervals",
            "pair_count",
            "observed_counts",
            "native_observed_counts",
            "split",
        )
        contracts[arm] = {
            row["scene_id"]: {k: row[k] for k in metadata}
            for row in raw["data_contract"][0]
            if row["scene_id"] in scene_ids
        }
        del raw
        snap = load_complete_snapshot(p, selected=False, device="cuda:0")
        snapshots[arm] = snap
        history = read_json(root / "history.json")
        reported = {
            str(sid) for record in history for sid in record["sampling"]["scene_ids"]
        }
        bindings[arm] = dict(
            checkpoint_sha256=digest(p),
            history_sha256=digest(root / "history.json"),
            logged_train_scenes=[sid for sid in scene_ids if sid in reported],
            end_batch=history[-1]["tam_train_batch"],
            history_rows=len(history),
        )
    require(
        contracts["A"] == contracts["B"] and len(contracts["A"]) == 8,
        "paired contracts",
    )
    normalizer = snapshots["A"]["normalizer"]
    require(
        normalizer == snapshots["B"]["normalizer"]
        and "physical_units" not in normalizer,
        "original 094 units",
    )
    data = runs / parent.DATA / "private"
    hashes = {}
    with MTADArchive(archives) as archive:
        validate_archive(archive, manifest)
        for sid in scene_ids:
            scene = measurements(
                archive,
                sid,
                data / "motion/T_core" / f"scene-{sid}.npz",
                normalizer["adapter_config"],
            )
            c = contracts["A"][sid]
            require(c["split"] == "T_core", "wrong role")
            for k in ("node_ids", "grid", "sources", "intervals"):
                np.testing.assert_array_equal(scene[k], c[k])
            np.testing.assert_array_equal(
                scene["observed"].sum(1), c["observed_counts"]
            )
            np.testing.assert_array_equal(
                scene["native_observed"].sum(1), c["native_observed_counts"]
            )
            pairs = np.column_stack(np.triu_indices(len(scene["node_ids"]), 1))
            require(len(pairs) == c["pair_count"], "pair roster")
            values, visible = normalize_scene(scene, normalizer)
            scores = {
                a: predict(snapshots[a]["model"], values, visible, pairs) for a in "AB"
            }
            path = private / f"{sid}.npz"
            np.savez_compressed(
                path,
                nodes=scene["node_ids"],
                query=scene["grid"],
                src=scene["sources"],
                pairs=pairs,
                **scores,
            )
            hashes[sid] = digest(path)
        require(
            [(r["scene_id"], r["kind"]) for r in archive.members_read]
            == [(sid, "measurements") for sid in scene_ids],
            "input members",
        )
    write(out / "freeze.json", dict(scenes=scene_ids, hashes=hashes, bindings=bindings))
    samples = []
    for sid in scene_ids:
        path = private / f"{sid}.npz"
        require(digest(path) == hashes[sid], "score freeze")
        with np.load(path, allow_pickle=False) as z:
            ids = parent.read_identity(
                data / "evaluation/T_core" / f"scene-{sid}.npz", z["nodes"], z["query"]
            )
            i, j = z["pairs"].T
            samples.append(
                dict(
                    scene=sid,
                    cross=z["src"][i] != z["src"][j],
                    truth=ids[i] == ids[j],
                    A=z["A"],
                    B=z["B"],
                )
            )
    largest = max(samples, key=lambda r: len(r["truth"]))["scene"]
    full = stats(samples)
    reduced = stats([r for r in samples if r["scene"] != largest])
    write(
        out / "summary.json",
        dict(
            full=full,
            without_largest=reduced,
            largest=largest,
            scenes={r["scene"]: stats([r]) for r in samples},
            bindings=bindings,
            cuda_peak_bytes=torch.cuda.max_memory_allocated(),
            gpu=torch.cuda.get_device_name(0),
        ),
    )


if __name__ == "__main__":
    execute(run)
