"""C151S: replay a complete frozen block swap and measure identifiable supply loss."""

from collections import Counter, defaultdict

import numpy as np

from tools.analysis import association_endpoint_qualification as parent
from tools.analysis.association_position_smoothing import require, verify_hash
from tools.analysis.association_frozen_pr_audit import read_json
from tools.analysis.offline_baseline_assignment import _components, _minimum_cost
from tools.analysis.offline_baseline_assignment_run import partition
from tools.analysis.offline_baseline_core import ACCEPT_METRES
from tools.analysis.offline_baseline_restart import digest, write

SOURCE = "LEVEL1-OFFLINE-ASSOCIATION-POSITION-SMOOTHING-C151P-20261118-attempt-1"
FROZEN = "7ba70380763d2e33dc2da76591f96af4215bd79365ec744883ce0cc5105703b7"
SUMMARY = "b730f9ec9ad833becaefc6c22f99d01b028acf26c7bb1b4bb674be7458d2094b"
SCENES = "1738bc80cca466eb1a95caef32df816c29fb234b148cac3225dcf8021d17b542"


def links(arr):
    return {tuple(map(int, item)) for item in arr}


def suppliers(membership, sources, support, identities=None):
    neighbors = (membership[:, None] == membership) & (sources[:, None] != sources)
    if identities is not None:
        neighbors &= identities[:, None] == identities
    return neighbors.astype(np.int32) @ support.astype(np.int32)


def feasible_partners(left, right, edges, quota, changed):
    """Forcing a pair uses its endpoints. The residual graph must realize K-1."""
    feasible = defaultdict(list)
    for i, j, _ in edges:
        if i not in changed and j not in changed:
            continue
        residual = [(u, v, 0) for u, v, _ in edges if u != i and v != j]
        _, matching, _ = _minimum_cost(
            left, right, residual, exact_count=0, stop_increment=1
        )
        if len(matching) >= quota - 1:
            feasible[i].append(j)
            feasible[j].append(i)
    return feasible


def groups_for(saved):
    return (
        np.split(saved["chain_nodes"], saved["chain_offsets"][1:-1])
        if len(saved["nodes"])
        else []
    )


def input_decisions(saved, nearest):
    chains = groups_for(saved)
    src, mask = saved["sources"], saved["support"]
    raw, proposed = links(saved["RAW_links"]), links(saved["SMOOTH_links"])
    graph = {
        tuple(e): d
        for e, d in zip(saved["candidate_pairs"], saved["RAW_cost"], strict=True)
        if d <= ACCEPT_METRES
    }
    old_count = suppliers(saved["RAW_membership"], src, mask)
    new_count = suppliers(saved["SMOOTH_membership"], src, mask)
    alarm = ~mask & (old_count > 0) & (new_count == 0)
    first_seconds = saved["query"][0] + nearest[:, 0]
    last_seconds = saved["query"][-1] - nearest[:, -1]
    chosen = set(raw)
    blocks, detail = [], []
    for index, (_, left, right, edges) in enumerate(
        _components(tuple(saved["chain_sources"]), graph)
    ):
        edge_set = {tuple(sorted((i, j))) for i, j, _ in edges}
        prior, next_ = raw & edge_set, proposed & edge_set
        require(len(prior) == len(next_), "block quota changed")
        if prior == next_:
            continue
        chain_ids = left + right
        nodes = np.sort(np.concatenate([chains[k] for k in chain_ids])).astype(int)
        veto = bool(alarm[nodes].any())
        if not veto:
            chosen.difference_update(prior)
            chosen.update(next_)
        changed = {i for edge in prior ^ next_ for i in edge}
        possible = feasible_partners(left, right, edges, len(prior), changed)
        old_partner = {i: j for e in prior for i, j in (e, e[::-1])}
        new_partner = {i: j for e in next_ for i, j in (e, e[::-1])}
        names = {k: f"L{a + 1}" for a, k in enumerate(left)}
        names.update({k: f"R{a + 1}" for a, k in enumerate(right)})

        def pair_names(pairs):
            return [sorted((names[i], names[j])) for i, j in sorted(pairs)]

        block = dict(
            block=index,
            nodes=nodes.tolist(),
            veto=veto,
            quota=len(prior),
            removed=pair_names(prior - next_),
            added=pair_names(next_ - prior),
            visible_lost=int(alarm[nodes].sum()),
        )
        for k in chain_ids:
            old, new = old_partner.get(k), new_partner.get(k)
            if old == new:
                continue
            other = [t for t in possible[k] if t != old]
            counts = np.zeros(mask.shape[1], np.int64)
            for t in other:
                counts += mask[chains[t]].any(axis=0)
            donors = chains[old] if old is not None else []
            new_donors = chains[new] if new is not None else []
            for node in chains[k]:
                for q in np.flatnonzero(
                    ~mask[node] & ((old_count[node] > 0) | (new_count[node] > 0))
                ):
                    d0 = [int(d) for d in donors if mask[d, q]]
                    d1 = [int(d) for d in new_donors if mask[d, q]]
                    require(
                        len(d0) == old_count[node, q] and len(d1) == new_count[node, q],
                        "supply lineage",
                    )
                    detail.append(
                        dict(
                            block=index,
                            node=int(node),
                            q=int(q),
                            query_seconds=float(saved["query"][q]),
                            old_edge=[names[k], names.get(old)],
                            new_edge=[names[k], names.get(new)],
                            old_donors=d0,
                            new_donors=d1,
                            alternative_chains=int(counts[q]),
                            receiver_age_s=float(nearest[node, q]),
                            receiver_first_s=float(first_seconds[node]),
                            receiver_last_s=float(last_seconds[node]),
                            receiver_side="before"
                            if saved["query"][q] < first_seconds[node]
                            else "after",
                            old_donor_age_s=min(
                                (float(nearest[d, q]) for d in d0), default=None
                            ),
                            new_donor_age_s=min(
                                (float(nearest[d, q]) for d in d1), default=None
                            ),
                        )
                    )
        blocks.append(block)
    guarded = partition(chains, sorted(chosen), len(src))
    return guarded, blocks, detail


def measures(members, src, mask, ids, raw_same):
    row, col = np.triu_indices(len(src), 1)
    subset = src[row] != src[col]
    truth = ids[row] == ids[col]
    match = members[row] == members[col]
    any_ = suppliers(members, src, mask) > 0
    same = suppliers(members, src, mask, ids) > 0
    _, anchors = np.unique(members, return_index=True)
    reps = len(set(ids[anchors]))
    return dict(
        tp=int((subset & truth & match).sum()),
        fp=int((subset & ~truth & match).sum()),
        fn=int((subset & truth & ~match).sum()),
        represented=reps,
        duplicates=len(anchors) - reps,
        any_all=int(any_.sum()),
        same_missing=int((same & ~mask).sum()),
        gross_lost=int((~mask & raw_same & ~same).sum()),
        restored=int((~mask & ~raw_same & same).sum()),
    )


def evaluate(saved, guarded, blocks, detail, ids, published):
    mask, src = saved["support"], saved["sources"]
    old, new = saved["RAW_membership"], saved["SMOOTH_membership"]
    before = suppliers(old, src, mask, ids) > 0
    arms = {
        key: measures(m, src, mask, ids, before)
        for key, m in [("RAW", old), ("SMOOTH", new), ("GUARD", guarded)]
    }
    for key in ("RAW", "SMOOTH"):
        for metric in ("tp", "fn", "fp"):
            require(
                arms[key][metric] == published["arms"][key]["MSTSA"][metric],
                "parent MSTSA",
            )
        for metric in ("represented", "duplicates"):
            require(
                arms[key][metric] == published["arms"][key][metric], "parent population"
            )
    require(
        arms["SMOOTH"]["gross_lost"] == published["supply"]["same_ship_missing_lost"]
        and arms["SMOOTH"]["restored"]
        == published["supply"]["same_ship_missing_restored"],
        "parent supply",
    )
    traces = defaultdict(list)
    for item in detail:
        traces[item["block"]].append(item)
    result = []
    for info in blocks:
        nodes = np.array(info["nodes"])
        base = measures(old[nodes], src[nodes], mask[nodes], ids[nodes], before[nodes])
        alt = measures(new[nodes], src[nodes], mask[nodes], ids[nodes], before[nodes])
        counts = Counter()
        ages = defaultdict(list)
        for d in traces[info["block"]]:
            label = ids[d["node"]]
            was = any(ids[j] == label for j in d["old_donors"])
            now = any(ids[j] == label for j in d["new_donors"])
            if was and not now:
                kind = "same_to_other" if d["new_donors"] else "same_to_none"
            elif d["old_donors"] and not d["new_donors"]:
                kind = "other_to_none"
            else:
                continue
            counts[kind] += 1
            counts[kind + "_no_alternative"] += int(d["alternative_chains"] == 0)
            ages[kind + "_receiver"].append(d["receiver_age_s"])
            ages[kind + "_old_donor"].append(d["old_donor_age_s"])
        require(
            counts["same_to_none"] + counts["same_to_other"] == alt["gross_lost"]
            and counts["same_to_none"] + counts["other_to_none"]
            == info["visible_lost"],
            "transition cross-check",
        )
        fn, fp = alt["fn"] - base["fn"], alt["fp"] - base["fp"]
        result.append(
            {
                **{k: v for k, v in info.items() if k != "nodes"},
                "delta_fn": fn,
                "delta_fp": fp,
                "lost": alt["gross_lost"],
                "gained": alt["restored"],
                "safe_gain": bool(
                    not alt["gross_lost"] and fn <= 0 and fp <= 0 and min(fn, fp) < 0
                ),
                "transitions": dict(counts),
                "seconds_p50": {k: float(np.median(v)) for k, v in ages.items()},
            }
        )
    require(
        sum(b["lost"] for b in result) == arms["SMOOTH"]["gross_lost"]
        and sum(b["visible_lost"] for b in result)
        == published["supply"]["any_missing_lost"],
        "blocks partition losses",
    )
    return arms, result


def aggregate(rows):
    arms = {}
    for name in ("RAW", "SMOOTH", "GUARD"):
        arms[name] = {
            k: sum(r["arms"][name][k] for r in rows) for k in rows[0]["arms"][name]
        }
    all_blocks = [b for r in rows for b in r["blocks"]]
    raw, guard = arms["RAW"], arms["GUARD"]
    changed = sum(any(not b["veto"] for b in r["blocks"]) for r in rows)
    return dict(
        scenes=len(rows),
        arms=arms,
        proposed_blocks=len(all_blocks),
        accepted_blocks=sum(not b["veto"] for b in all_blocks),
        safe_beneficial=sum(b["safe_gain"] for b in all_blocks),
        safe_beneficial_vetoed=sum(b["safe_gain"] and b["veto"] for b in all_blocks),
        transitions=dict(
            sum((Counter(b["transitions"]) for b in all_blocks), Counter())
        ),
        changed_scenes=changed,
        guard_pass=guard["fn"] < raw["fn"]
        and guard["fp"] < raw["fp"]
        and guard["gross_lost"] == 0
        and guard["represented"] >= raw["represented"]
        and guard["any_all"] >= raw["any_all"],
    )


def run(runs, out):
    private = out / "private"
    private.mkdir(mode=0o700)
    src = runs / SOURCE
    for name, sha in [
        ("freeze.json", FROZEN),
        ("summary.json", SUMMARY),
        ("scenes.json.gz", SCENES),
    ]:
        verify_hash(src / name, sha)
    frozen, summary, published = [
        read_json(src / k) for k in ("freeze.json", "summary.json", "scenes.json.gz")
    ]
    data = runs / parent.DATA / "private"
    prior = read_json(
        runs
        / ("LEVEL1-OFFLINE-ENDPOINT-QUALIFICATION-C148-20261116-attempt-1")
        / "freeze.json"
    )
    verify_hash(
        runs
        / "LEVEL1-OFFLINE-ENDPOINT-QUALIFICATION-C148-20261116-attempt-1"
        / "freeze.json",
        frozen["source_freeze_sha256"],
    )
    hashes = {}
    for component, sid in frozen["components"].items():
        old = src / "private" / f"{sid}.npz"
        motion = data / "motion/T_core" / f"scene-{sid}.npz"
        verify_hash(old, frozen["prediction_sha256"][sid])
        verify_hash(motion, prior["motion_sha256"][sid])
        with (
            np.load(old, allow_pickle=False) as saved,
            np.load(motion, allow_pickle=False) as m,
        ):
            for a, b in [
                ("nodes", "node_ids"),
                ("query", "query_seconds"),
                ("support", "support"),
            ]:
                np.testing.assert_array_equal(saved[a], m[b])
            guarded, blocks, detail = input_decisions(saved, m["nearest_seconds"])
        fp = private / f"{sid}.json.gz"
        write(fp, dict(guarded=guarded.tolist(), blocks=blocks, detail=detail))
        hashes[sid] = digest(fp)
    require(len(hashes) == 76, "incorrect fit population")
    write(out / "freeze.json", dict(decisions_sha256=hashes, source_sha256=FROZEN))
    rows = []
    for published_row in published:
        sid = published_row["scene_id"]
        fp = private / f"{sid}.json.gz"
        verify_hash(fp, hashes[sid])
        frozen_row = read_json(fp)
        truth = data / "evaluation/T_core" / f"scene-{sid}.npz"
        verify_hash(truth, summary["evaluation_sha256"][sid])
        with np.load(src / "private" / f"{sid}.npz", allow_pickle=False) as saved:
            ids = parent.read_identity(truth, saved["nodes"], saved["query"])
            arms, blocks = evaluate(
                saved,
                np.asarray(frozen_row["guarded"]),
                frozen_row["blocks"],
                frozen_row["detail"],
                ids,
                published_row,
            )
        rows.append(
            dict(
                scene_id=sid,
                component=published_row["component"],
                arms=arms,
                blocks=blocks,
            )
        )
    full = aggregate(rows)
    reduced = aggregate(
        [r for r in rows if r["component"] != summary["largest_fit_component"]]
    )
    require(full["scenes"] == 76 and reduced["scenes"] == 75, "largest component")
    write(out / "scenes.json.gz", rows)
    write(
        out / "summary.json",
        dict(
            full=full,
            without_largest=reduced,
            qualifies=full["guard_pass"]
            and reduced["guard_pass"]
            and reduced["changed_scenes"] >= 2,
        ),
    )


if __name__ == "__main__":
    from tools.analysis.saved_replay_runtime import execute

    execute(run)
