"""152R: independently align the saved peer predictions and audit actual training exposure."""

from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch

from tools.analysis.association_frozen_pr_audit import read_json, require
from tools.analysis.offline_baseline_restart import digest, write
from tools.analysis.saved_replay_runtime import execute

DATA = "LEVEL1-FUSION-PEER-FULL-DATA-20261030-151-attempt-1"
WS = "LEVEL1-STATIC-PARTITION-PILOT-20261029-114-attempt-1"
BASE = "LEVEL1-SOURCE-RISK-FULL-BENCHMARK-20261030-147-attempt-1"
PEERS = {
    "A": "LEVEL1-FUSION-PEER-FULL-PTFM-20261030-152A",
    "B": "LEVEL1-FUSION-PEER-FULL-DEEPAF-20261030-152B",
}
BASELINE = "WS_SHARED_Q_HARD"
MEAN = "WS_MEAN"
CORE = "WS_PTFM_CORE"
PLUS = "WS_PTFM_PLUS_MEAN"
DEEP = "WS_DEEPAF_PV"
METHODS = (BASELINE, MEAN, CORE, PLUS, DEEP)
ATTRS = ("latitude", "longitude", "speed", "heading")


def cost_and_training(runs, arm, train):
    stem = PEERS[arm]
    root = runs / (stem + "-attempt-1")
    launch = runs / (stem + "-launch-attempt-1")
    science = read_json(root / "run.json")
    manager = read_json(launch / "status.json")
    req = read_json(launch / "request.json")
    epochs, batch, updates = (150, 32, 427800) if arm == "A" else (20, 16, 80800)
    require(
        science["exit_code"] == 0
        and manager["returncode"] == 0
        and science["completed_V_scenes"] == 300
        and science["completed_epochs"] == epochs
        and science["optimizer_updates"] == updates,
        "original endpoint totals",
    )
    require(
        science["data_run"] == str(runs / DATA)
        and science["partition_run"] == str(runs / WS)
        and science["qhard_run"] == str(runs / BASE),
        "original inputs",
    )
    epoch_file = read_json(root / "epoch-summary.json")
    require(len(epoch_file) == epochs and len(train) == 2100, "original epoch count")
    generator = np.random.default_rng(20)
    ids = list(train)
    groups = [len(train[s]["groups"]) for s in ids]
    require(sum(groups) == 46126, "group pool")
    cursor = 0
    gradients = []
    epoch_results = []
    journal = root / "training.jsonl"
    with journal.open() as records:
        for epoch in range(1, epochs + 1):
            losses = []
            counts = np.zeros(4, np.int64)
            seen = 0
            for i in generator.permutation(len(ids)):
                order = generator.permutation(groups[i])
                for start in range(0, len(order), batch):
                    row = json.loads(next(records))
                    want = order[start : start + batch]
                    cursor += 1
                    require(
                        row["epoch"] == epoch
                        and row["update"] == cursor
                        and str(row["scene_id"]) == ids[i]
                        and row["groups"] == len(want)
                        and row["group_indices"] == want.tolist(),
                        "actual group order/exposure",
                    )
                    require(
                        np.isfinite(row["loss"]) and np.isfinite(row["gradient_norm"]),
                        "nonfinite training log",
                    )
                    counts += row["valid_supervision_counts"]
                    losses.append(row["loss"])
                    gradients.append(row["gradient_norm"])
                    seen += len(want)
            orig = epoch_file[epoch - 1]
            require(
                orig["epoch"] == epoch
                and orig["updates"] == len(losses)
                and orig["groups"] == seen == 46126
                and counts.tolist() == orig["valid_supervision_counts"],
                "epoch exposure/supervision",
            )
            np.testing.assert_allclose(np.mean(losses), orig["loss_mean"], rtol=1e-12)
            epoch_results.append(
                dict(
                    epoch=epoch,
                    updates=len(losses),
                    groups=seen,
                    supervision=counts.tolist(),
                    loss_mean=float(np.mean(losses)),
                    seconds=orig["elapsed_seconds"],
                )
            )
        require(
            not records.read().strip() and cursor == updates, "trailing/missing updates"
        )
    ck = root / "private/last-epoch.pt"
    weights = torch.load(ck, map_location="cpu", weights_only=True)
    require(
        weights["epoch"] == epochs
        and weights["updates"] == cursor
        and weights["method"] == science["method"]
        and weights["state_speed_unit"] == "mps",
        "checkpoint endpoint",
    )
    model = weights["model"]
    norm = weights["normalizer"]
    require(
        all(torch.isfinite(t).all().item() for t in model.values()),
        "checkpoint weights",
    )
    if arm == "A":
        require(
            norm["split"] == "T_core"
            and norm["truth_used"] is False
            and norm["count"] > 0
            and norm["speed_unit"] == "mps",
            "normalizer",
        )
        for key in ("mean", "scale"):
            np.testing.assert_array_equal(norm[key], model[key].numpy())
    else:
        require(norm is None, "unexpected normalizer")
    probes = {
        name: read_json(root / ("T-probe-" + name + ".json"))
        for name in ("u0", "final")
    }
    evidence = {
        p.name: digest(p)
        for p in (
            ck,
            journal,
            root / "epoch-summary.json",
            root / "run.json",
            root / "T-probe-u0.json",
            root / "T-probe-final.json",
        )
    }
    return dict(
        epochs=epoch_results,
        updates=cursor,
        group_presentations=epochs * sum(groups),
        groups=sum(groups),
        gradient_min=float(min(gradients)),
        gradient_max=float(max(gradients)),
        normalizer=norm,
        supervision_columns=science["supervision_count_columns"],
        probes=probes,
        sha256=evidence,
        train_seconds=sum(r["seconds"] for r in epoch_results),
        science_seconds=science["elapsed_seconds"],
        managed_seconds=manager["elapsed_seconds"],
        gpu_peak_bytes=science["cuda_peak_allocated_bytes"],
        rss_peak_bytes=science["peak_rss_bytes"],
        command=req["argv"],
        cwd=req["cwd"],
        managed=manager["status"],
        timeout=manager["timed_out"],
        advisory=manager["time_advisory_issued"],
    )


def position_errors(pred, truth):
    a, b = np.deg2rad(pred[..., 0]), np.deg2rad(truth[..., 0])
    dlon = np.deg2rad((truth[..., 1] - pred[..., 1] + 180) % 360 - 180)
    h = np.clip(
        np.sin((b - a) / 2) ** 2 + np.cos(a) * np.cos(b) * np.sin(dlon / 2) ** 2,
        0.0,
        1.0,
    )
    return 12742000.0 * np.arctan2(np.sqrt(h), np.sqrt(1.0 - h))


def radial(values):
    v = values[np.isfinite(values)]
    if not len(v):
        return dict(n=0, mae=None, rmse=None, p95=None, p99=None)
    return dict(
        n=len(v),
        mae=float(v.mean()),
        rmse=float(np.sqrt(np.mean(v * v))),
        p95=float(np.quantile(v, 0.95)),
        p99=float(np.quantile(v, 0.99)),
    )


def scalar(values):
    v = values[np.isfinite(values)]
    return dict(
        n=len(v),
        me=float(v.mean()) if len(v) else None,
        mae=float(np.abs(v).mean()) if len(v) else None,
        sd=float(v.std(ddof=1)) if len(v) > 1 else None,
    )


def paper_values(residual):
    result = {}
    for i, k in enumerate(ATTRS):
        factor = (1e4, 1e4, 3600 / 1852, 1)[i]
        per = [scalar(row[np.isfinite(row)] * factor) for row in residual[..., i]]
        result[k] = {
            s: float(np.mean(values))
            if (values := [v[s] for v in per if v[s] is not None])
            else None
            for s in ("me", "mae", "sd")
        }
    return result


def match_close(x, y, maxima, key):
    if x is None or y is None:
        require(x is None and y is None, "null mismatch " + key)
    else:
        diff = abs(x - y)
        maxima[key] = max(maxima.get(key, 0.0), float(diff))
        require(np.isclose(x, y, atol=1e-6, rtol=1e-9), "archived metric " + key)


def compare_saved_artifact(
    z,
    name,
    expected,
    groups,
    indices,
    raw,
    group_valid,
    query_target,
    truth,
    valid,
    pred,
    mask,
):
    prefix = name + "_"

    def same(k, value):
        np.testing.assert_array_equal(z[prefix + k], value, err_msg=name + " " + k)

    require(str(z[prefix + "state_speed_unit"]) == "mps", "saved unit")
    same("node_group_indices", expected)
    same("group_offsets", np.r_[0, np.cumsum([len(g) for g in groups])])
    same("group_node_indices", np.concatenate(groups) if groups else np.empty(0, int))
    same("target_ids", query_target)
    same("aligned_group_indices", indices)
    same("group_states", raw)
    same("group_valid", group_valid)
    for k, x in [
        ("aligned_truth", truth),
        ("aligned_truth_valid", valid),
        ("aligned_prediction", pred),
        ("aligned_prediction_valid", mask),
    ]:
        np.testing.assert_array_equal(
            z[prefix + k][: len(truth)], x, err_msg=name + " " + k
        )


def inspect_scene(runs, sid, entry, maximum):
    roots = {a: runs / (p + "-attempt-1") for a, p in PEERS.items()}
    membership = np.asarray(
        read_json(runs / WS / "private" / f"scene-{sid}.json.gz")["memberships"]["WS"],
        int,
    )
    groups = [
        np.flatnonzero(membership == g) for g in dict.fromkeys(membership.tolist())
    ]
    with ExitStack() as stack:

        def load(p):
            return stack.enter_context(np.load(p, allow_pickle=False))

        motion = load(Path(entry["motion"]))
        ev = load(Path(entry["evaluation"]))
        np.testing.assert_array_equal(motion["node_ids"], ev["node_ids"])
        np.testing.assert_array_equal(motion["query_seconds"], ev["query_seconds"])
        require(
            len(motion["query_seconds"]) == 64
            and str(motion["state_speed_unit"]) == "mps"
            and len(membership) == len(ev["node_ids"]),
            "Q64 and units",
        )
        targets, ids, truth, valid = (
            ev[k] for k in ("target_ids", "identities", "target_states", "target_valid")
        )
        registry = {k: i for i, k in enumerate(targets.tolist())}
        first = np.full(len(targets), -1, int)
        dupes = []
        for g, members in enumerate(groups):
            t = registry[ids[members[0]]]
            if first[t] == -1:
                first[t] = g
            else:
                dupes.append(g)
        indices = np.r_[first, np.asarray(dupes, int)]
        available = first >= 0
        states = {
            k: load(p / "private" / f"states-V_select-{sid}.npz")
            for k, p in roots.items()
        }
        original = load(runs / BASE / "private" / f"states-{sid}.npz")
        zraw = {
            k: load(p / "private" / f"raw-V_select-{sid}.npz") for k, p in roots.items()
        }
        public = {
            k: read_json(p / "scenes" / f"{sid}.json.gz") for k, p in roots.items()
        }
        original_row = read_json(runs / BASE / "scenes" / f"scene-{sid}.json.gz")
        raw = {
            BASELINE: original[BASELINE + "_group_states"],
            MEAN: zraw["A"][MEAN],
            CORE: zraw["A"][CORE],
            PLUS: zraw["A"][PLUS],
            DEEP: zraw["B"][DEEP],
        }
        for a in "AB":
            np.testing.assert_array_equal(zraw[a][BASELINE], raw[BASELINE])
            np.testing.assert_array_equal(zraw[a][MEAN], raw[MEAN])
        np.testing.assert_array_equal(raw[CORE], raw[PLUS])
        core_mask = zraw["A"]["core_valid"]
        ctrl_delta = np.abs(raw[DEEP][..., 2:] - raw[MEAN][..., 2:])
        fallback_delta = np.abs(raw[PLUS][~core_mask] - raw[MEAN][~core_mask])
        control = dict(
            deep_nonidentical=int(np.count_nonzero(ctrl_delta)),
            deep_max=float(np.max(ctrl_delta, initial=0.0)),
            fallback_nonidentical=int(np.count_nonzero(fallback_delta)),
            fallback_max=float(np.max(fallback_delta, initial=0.0)),
        )
        np.testing.assert_array_equal(
            raw[BASELINE][..., 2:], original[MEAN + "_group_states"][..., 2:]
        )
        result = {}
        for name in METHODS:
            m = raw[name]
            gv = core_mask if name == CORE else np.ones_like(m, bool)
            pred = np.full_like(truth, np.nan, dtype=float)
            mask = np.zeros_like(valid)
            pred[available] = m[first[available]]
            mask[available] = gv[first[available]]
            sources = (
                [original, states["A"], states["B"]]
                if name == BASELINE
                else [states["A"], states["B"]]
                if name == MEAN
                else [states["B" if name == DEEP else "A"]]
            )
            for z in sources:
                compare_saved_artifact(
                    z,
                    name,
                    membership,
                    groups,
                    indices,
                    m,
                    gv,
                    targets,
                    truth,
                    valid,
                    pred,
                    mask,
                )
            use = mask & valid
            residual = np.where(use, truth - pred, np.nan)
            pos = np.where(use[..., :2].all(-1), position_errors(pred, truth), np.nan)
            pos_stats = radial(pos)
            records = [public["B" if name == DEEP else "A"]]
            if name == BASELINE:
                records = [original_row, public["A"], public["B"]]
            if name == MEAN:
                records = [public["A"], public["B"]]
            for rec in records:
                require(
                    str(rec["scene_id"]) == sid
                    and rec["component_id"] == entry["component_id"],
                    "scene record binding",
                )
                archived = rec["methods"][name]
                old = archived["position"]["all"]
                require(
                    old["intersection_count"] == pos_stats["n"]
                    and old["available_target_count"]
                    == int(valid[..., :2].all(-1).sum()),
                    "scene position population",
                )
                for field in ("mae", "rmse", "p95", "p99"):
                    match_close(
                        pos_stats[field], old[field], maximum, "position_" + field
                    )
                pop = archived["population"]
                require(
                    pop["predicted_group_count"] == len(groups)
                    and pop["duplicate_prediction_group_count"] == len(dupes)
                    and pop["target_identity_count"] == len(targets)
                    and rec["methods"][name]["unmatched_target_count"]
                    == int((first < 0).sum()),
                    "output population",
                )
            result[name] = dict(
                pos=pos.ravel(),
                res=residual.reshape(-1, 4),
                paper=paper_values(residual),
            )
        return dict(
            id=sid,
            component=entry["component_id"],
            targets=len(targets),
            groups=len(groups),
            duplicates=len(dupes),
            missed=int((first < 0).sum()),
            den=valid.sum((0, 1)),
            pos_den=int(valid[..., :2].all(-1).sum()),
            methods=result,
            control=control,
        )


def summary(rows):
    den = np.sum([r["den"] for r in rows], axis=0)
    pden = sum(r["pos_den"] for r in rows)
    methods = {}
    positions = {}
    for name in METHODS:
        parts = [r["methods"][name] for r in rows]
        pos = np.concatenate([p["pos"] for p in parts])
        positions[name] = pos
        residual = np.concatenate([p["res"] for p in parts])
        attrs = {}
        for i, attr in enumerate(ATTRS):
            s = scalar(residual[:, i])
            attrs[attr] = {
                **s,
                "available": int(den[i]),
                "coverage": s["n"] / den[i] if den[i] else None,
            }

        def circle(v):
            return (v + 180.0) % 360.0 - 180.0

        paper = {
            a: {
                key: float(np.mean(vals))
                if (
                    vals := [
                        p["paper"][a][key]
                        for p in parts
                        if p["paper"][a][key] is not None
                    ]
                )
                else None
                for key in ("me", "mae", "sd")
            }
            for a in ATTRS
        }
        ps = radial(pos)
        methods[name] = dict(
            position={
                **ps,
                "available": pden,
                "coverage": ps["n"] / pden if pden else None,
            },
            attrs=attrs,
            circular_heading=scalar(circle(residual[:, 3])),
            short_longitude=scalar(circle(residual[:, 1])),
            paper=paper,
        )
    pairs = {}
    for name in (MEAN, CORE, PLUS, DEEP):
        mask = np.isfinite(positions[name]) & np.isfinite(positions[BASELINE])
        delta = positions[name][mask] - positions[BASELINE][mask]
        pairs[name] = dict(
            candidate=radial(positions[name][mask]),
            original147=radial(positions[BASELINE][mask]),
            harm_gt_1km=int((delta > 1000).sum()),
            gain_gt_1km=int((delta < -1000).sum()),
            mean_delta_m=float(delta.mean()) if len(delta) else None,
        )
    outside = np.isfinite(positions[PLUS]) & ~np.isfinite(positions[CORE])
    pairs["outside_core"] = {
        n: radial(positions[n][outside]) for n in (BASELINE, MEAN, PLUS, DEEP)
    }
    control = {
        k: (
            max(r["control"][k] for r in rows)
            if k.endswith("max")
            else sum(r["control"][k] for r in rows)
        )
        for k in rows[0]["control"]
    }
    return dict(
        scenes=len(rows),
        population={
            k: sum(r[k] for r in rows)
            for k in ("targets", "groups", "duplicates", "missed")
        },
        methods=methods,
        pairs=pairs,
        controls=control,
    )


def run(runs, out):
    torch.set_num_threads(1)
    data = read_json(runs / DATA / "private/index.json")
    original = read_json(runs / BASE / "summary.json")
    ids = list(data["V_select"])
    require(
        ids == list(map(str, original["all_components"]["scene_ids"]))
        and len(ids) == 300,
        "original 147 scene order",
    )
    for arm, stem in PEERS.items():
        root = runs / (stem + "-attempt-1")
        orig = read_json(root / "summary.json")
        require(
            ids == list(map(str, orig["all_components"]["scene_ids"])),
            "original peer scene order",
        )
        write(
            out / f"training-{arm}.json", cost_and_training(runs, arm, data["T_core"])
        )
    rows = []
    maximum = {}
    for sid in ids:
        rows.append(inspect_scene(runs, sid, data["V_select"][sid], maximum))
    bycomp = Counter(r["component"] for r in rows)
    largest = bycomp.most_common(1)[0][0]
    require(
        len(bycomp) == 38 and largest == original["largest_component_id"],
        "component population",
    )
    full = summary(rows)
    physical_names = {"me": "ME", "mae": "MAE", "sd": "residual_SD"}
    for arm, stem in PEERS.items():
        orig = read_json(runs / (stem + "-attempt-1") / "summary.json")
        for name in (
            (BASELINE, MEAN, CORE, PLUS) if arm == "A" else (BASELINE, MEAN, DEEP)
        ):
            paper = orig["physical"][name]["paper_fusion_metrics"]["attributes"]
            for attr in ATTRS:
                for k, source in physical_names.items():
                    match_close(
                        full["methods"][name]["paper"][attr][k],
                        paper[attr]["scene_equal_target_mean"][source],
                        maximum,
                        "paper_" + attr + "_" + k,
                    )
                require(
                    full["methods"][name]["attrs"][attr]["n"]
                    == paper[attr]["evaluated_points"],
                    "attribute population",
                )
    write(
        out / "scenes.json.gz",
        [dict(scene=r["id"], component=r["component"], **summary([r])) for r in rows],
    )
    write(
        out / "components.json.gz",
        {c: summary([r for r in rows if r["component"] == c]) for c in bycomp},
    )
    write(
        out / "summary.json",
        dict(
            full=full,
            without_largest=summary([r for r in rows if r["component"] != largest]),
            largest=largest,
            components=len(bycomp),
            archive_max_abs_differences=maximum,
        ),
    )


if __name__ == "__main__":
    execute(run)
