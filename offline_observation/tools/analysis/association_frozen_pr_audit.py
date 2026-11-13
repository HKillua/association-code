"""Read frozen 056/114/094 V_select arrays and print anonymous PR diagnostics."""

import argparse
import gzip
import json
import os
import resource
import time
from pathlib import Path

import numpy as np


RUNS = {
    '056': 'LEVEL1-ASSOCIATION-20261022-056-attempt-1',
    '114': 'LEVEL1-STATIC-PARTITION-PILOT-20261029-114-attempt-1',
    **{f'094{arm}': f'LEVEL1-COMPLETION-VALUE-TAM-20261024-094{arm}-attempt-1' for arm in 'AB'},
}
RECALL_PCT = {'itsa': (50, 80, 90, 95), 'mstsa': (50, 80, 90, 95)}
COUNTS = ('tp', 'fp', 'fn', 'tn')


def read_json(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def summarize(scene, truth, chosen, n_scenes):
    tally = lambda mask: np.bincount(scene[mask], minlength=n_scenes)
    tp, fp, fn, tn = [tally(mask) for mask in (truth & chosen, ~truth & chosen, truth & ~chosen, ~truth & ~chosen)]
    positive, predicted = tp + fn, tp + fp
    defined_recall, defined_pfa = positive > 0, predicted > 0
    recall = np.divide(tp, positive, out=np.zeros(n_scenes), where=defined_recall)
    pfa = np.divide(fp, predicted, out=np.zeros(n_scenes), where=defined_pfa)
    active = positive + fp + tn > 0
    return dict(
        **{name: int(count.sum()) for name, count in zip(COUNTS, (tp, fp, fn, tn))},
        micro_recall=divide(tp.sum(), positive.sum()), precision=divide(tp.sum(), predicted.sum()),
        pfa=divide(fp.sum(), predicted.sum()), negative_false_acceptance=divide(fp.sum(), (fp + tn).sum()),
        paper_m_pca=divide(recall.sum(), active.sum()) if np.array_equal(active, defined_recall) else None,
        paper_m_pfa=divide(pfa.sum(), active.sum()) if np.array_equal(active, defined_pfa) else None,
        paper_mw_pca=divide(recall @ predicted, predicted.sum()) if np.all(~defined_pfa | defined_recall) else None,
        paper_mw_pfa=divide(fp.sum(), predicted.sum()),
        defined_recall_scenes=int(defined_recall.sum()), defined_pfa_scenes=int(defined_pfa.sum()),
        active_scenes=int(active.sum()),
    )


def score_groups(scores, truth):
    order = np.argsort(scores)[::-1]
    ranked = scores[order]
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]) + 1, len(ranked)]
    cumulative_tp = np.cumsum(truth[order])[ends - 1]
    return ranked[ends - 1], ends, cumulative_tp


def ranking(scores, truth, scene, slices, scope, actual_threshold):
    thresholds, predicted, tp = score_groups(scores, truth)
    fp = predicted - tp
    positive = int(tp[-1])
    negative = int(fp[-1])
    added_tp = np.diff(tp, prepend=0)
    added_fp = np.diff(fp, prepend=0)
    pfa = fp / predicted

    def result_at(index):
        threshold = thresholds[index]
        prior = predicted[index - 1] if index else 0
        return dict(threshold=float(threshold), tie_size=int(predicted[index] - prior),
                    tie_tp=int(added_tp[index]), tie_fp=int(added_fp[index]),
                    **summarize(scene, truth, scores >= threshold, len(slices)))

    points = {}
    for percent in RECALL_PCT[scope]:
        first = int(np.searchsorted(tp, (percent * positive + 99) // 100))
        best_pfa = first + int(np.argmin(pfa[first:]))
        points[str(percent)] = dict(first_whole_tie=result_at(first), minimum_pfa_at_or_above=result_at(best_pfa))

    # Each scene gets its own label-aware cutoff solely to diagnose ranking.
    local_chosen = np.zeros(len(scores), dtype=bool)
    local_ap, local_auc = [], []
    for section in slices:
        local_scores, local_truth = scores[section], truth[section]
        values, pred, pos = score_groups(local_scores, local_truth)
        neg = pred - pos
        require(pos[-1] > 0 and neg[-1] > 0, 'local scope lacks a class')
        d_pos = np.diff(pos, prepend=0)
        d_neg = np.diff(neg, prepend=0)
        local_ap.append(float(d_pos @ (pos / pred) / pos[-1]))
        local_auc.append(float(d_pos @ (neg[-1] - neg + .5 * d_neg) / (pos[-1] * neg[-1])))
        cutoff = values[np.searchsorted(pos, (90 * int(pos[-1]) + 99) // 100)]
        local_chosen[section] = local_scores >= cutoff
    return dict(
        positive_count=positive, negative_count=negative, score_min=float(thresholds[-1]),
        score_max=float(thresholds[0]), distinct_scores=len(thresholds),
        max_score_tie=int(np.diff(predicted, prepend=0).max()),
        pooled_ap=float(added_tp @ (tp / predicted) / positive),
        pooled_auc=float(added_tp @ (negative - fp + .5 * added_fp) / (positive * negative)),
        local_ap_scene_mean=float(np.mean(local_ap)), local_auc_scene_mean=float(np.mean(local_auc)),
        actual_threshold=dict(threshold=actual_threshold,
                              **summarize(scene, truth, scores >= actual_threshold, len(slices))),
        fixed_micro_recall=points,
        each_scene_90pct_oracle=summarize(scene, truth, local_chosen, len(slices)),
    )


def transitions(scene, truth, raw, decoded, n_scenes):
    removed, added = raw & ~decoded, ~raw & decoded
    return dict(
        removed_true=int(np.count_nonzero(removed & truth)), removed_false=int(np.count_nonzero(removed & ~truth)),
        added_true=int(np.count_nonzero(added & truth)), added_false=int(np.count_nonzero(added & ~truth)),
        retained_true=int(np.count_nonzero(raw & decoded & truth)), retained_false=int(np.count_nonzero(raw & decoded & ~truth)),
        decoded=summarize(scene, truth, decoded, n_scenes),
    )


def check_counts(actual, expected):
    require(all(actual[key] == expected[key] for key in COUNTS), 'frozen endpoint confusion mismatch')


def run(runs, manifest_path):
    started = time.monotonic()
    paths = {name: runs / relative for name, relative in RUNS.items()}
    manifest = read_json(manifest_path)
    ids = manifest['splits']['V_select']
    n_scenes = len(ids)
    require(n_scenes == 300 and len(set(ids)) == n_scenes, 'unexpected V_select manifest')
    arms = ('056', '094A', '094B')
    association = paths['056'] / 'association'
    old = read_json(association / 'summary.json')
    selection = old['selected']['selection']
    require(old['selected']['update'] == 1000 and selection == old['final_selection'], '056 final endpoint mismatch')
    thresholds = {'056': selection['threshold']}
    archives, old_tam = {}, {}
    for arm in arms[1:]:
        directory = paths[arm] / 'association-scores/update-004000'
        index = read_json(directory / 'index.json')
        saved = read_json(paths[arm] / 'summary.json')
        binding = index['binding']
        require(index['schema'] == 1 and binding['split'] == 'V_select' and binding['update'] == 4000
                and not binding['test_opened'] and [row['scene_id'] for row in index['records']] == ids,
                f'{arm} frozen index mismatch')
        thresholds[arm] = binding['threshold']
        require(thresholds[arm] == saved['selection']['threshold'] == saved['evaluation']['threshold'],
                f'{arm} threshold mismatch')
        archives[arm] = directory, index['records']
        old_tam[arm] = saved['evaluation']['association']
    ws = read_json(paths['114'] / 'summary.json')['full_V']
    require(ws['scene_ids'] == ids, '114 summary V order mismatch')

    scopes = {name: [] for name in RECALL_PCT}
    with np.load(association / 'final-scores.npz', allow_pickle=False) as archive:
        require(archive['scene_ids'].tolist() == ids, '056 score V order mismatch')
        for i, sid in enumerate(ids):
            pairs, truth, source, nodes, score = [archive[f'{sid}_{k}'] for k in ('pairs', 'labels', 'sources', 'node_ids', 'scores')]
            require(np.array_equal(pairs, np.column_stack(np.triu_indices(len(nodes), 1)))
                    and np.all(np.isin(truth, (0, 1))), '056 full candidate schema mismatch')
            membership = np.asarray(read_json(paths['114'] / 'private' / f'scene-{sid}.json.gz')['memberships']['WS'])
            require(membership.shape == nodes.shape and membership.dtype.kind in 'iu'
                    and (membership >= 0).all(), '114 membership shape mismatch')
            left, right = pairs.T
            decoded = membership[left] == membership[right]
            same = source[left] == source[right]
            scores = {'056': score}
            for arm, (directory, records) in archives.items():
                row = records[i]
                require(row['file'] == f'scene_{i:06d}.npz' and row['node_ids'] == nodes.tolist()
                        and row['component_id'] == manifest['components'][sid], f'{arm} node order mismatch')
                with np.load(directory / row['file'], allow_pickle=False) as saved:
                    require(all(np.array_equal(saved[key], value) for key, value in
                                [('pairs', pairs), ('sources', source), ('labels', truth)]), f'{arm} candidate mismatch')
                    scores[arm] = saved['scores']
            require(all(v.shape == truth.shape and np.isfinite(v).all() for v in scores.values()), 'invalid scores')
            for name, mask in (('itsa', same), ('mstsa', ~same)):
                scopes[name].append((truth[mask].astype(bool), decoded[mask], {k: v[mask] for k, v in scores.items()}))

    results = {}
    for scope, blocks in scopes.items():
        lengths = [len(t) for t, _, _ in blocks]
        bounds = np.r_[0, np.cumsum(lengths)]
        slices = [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
        scene = np.repeat(np.arange(n_scenes), lengths)
        truth = np.concatenate([t for t, _, _ in blocks])
        decoded = np.concatenate([d for _, d, _ in blocks])
        results[scope] = {}
        for arm in arms:
            scores = np.concatenate([s[arm] for _, _, s in blocks])
            row = ranking(scores, truth, scene, slices, scope, thresholds[arm])
            expected = old['selected_result']['metrics'][scope] if arm == '056' else old_tam[arm][scope]
            check_counts(row['actual_threshold'], expected)
            if arm == '056':
                change = transitions(scene, truth, scores >= thresholds[arm], decoded, n_scenes)
                check_counts(row['actual_threshold'], ws['raw'][scope])
                check_counts(change['decoded'], ws['decoded']['WS'][scope])
                results[scope]['056_to_114_ws'] = change
            results[scope][arm] = row
    return dict(
        status='COMPLETED', sources=RUNS, scene_count=n_scenes, components=len({manifest['components'][s] for s in ids}),
        manifest=str(manifest_path), cpu_affinity=sorted(os.sched_getaffinity(0)),
        test_opened=False, optimizer_updates=0, model_forwards=0, raw_payload_read=False,
        validated_saved_confusions=True, thresholds_for_diagnostics_only=True,
        seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        results=results,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.runs, args.manifest), allow_nan=False, separators=(',', ':')))
