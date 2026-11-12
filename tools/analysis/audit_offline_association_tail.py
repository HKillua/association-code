"""Independent C92 saved-score recount and frozen-head calibration replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from association.models.evidence_association import EvidenceAssociation, features_from_motion
from association.training.evidence_association import normalized_features


SCIENCE = 'LEVEL1-OFFLINE-ASSOCIATION-TAIL-CONTRAST-20261112-attempt-1'
PREP = 'LEVEL1-OFFLINE-ASSOCIATION-TAIL-PREP-20261112-attempt-1'
DATA = 'LEVEL1-FUSION-PEER-FULL-DATA-20261030-151-attempt-1'
OLD = 'LEVEL1-ASSOCIATION-20261022-056-attempt-1'
WS = 'LEVEL1-STATIC-PARTITION-PILOT-20261029-114-attempt-1'
ARMS = ('056', '056_T', 'BCE', 'RANK')
SCOPES = ('itsa', 'mstsa')
STAGES = ('raw', 'decoded', 'removed', 'added', 'retained')
OLD_THRESHOLD = 1.7320103112387182
SEED = 20261112


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class Comparisons:
    def __init__(self):
        self.exact = 0
        self.numerical = 0

    def verify(self, actual, expected, path='root'):
        if isinstance(actual, dict):
            for key, value in actual.items():
                self.verify(value, expected[key], path+'.'+str(key))
        elif isinstance(actual, (float, np.floating)):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10, err_msg=path)
            self.numerical += 1
        else:
            assert actual == expected, (path, actual, expected)
            self.exact += 1


def divide(numerator, denominator):
    return float(numerator/denominator) if denominator else None


def confusion(truth, prediction):
    tn, fp, fn, tp = np.bincount(2*truth.astype(int)+prediction, minlength=4).tolist()
    precision, recall = divide(tp, tp+fp), divide(tp, tp+fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, pair_count=len(truth), positive_count=tp+fn,
                f1=divide(2*tp, 2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                precision=precision, recall=recall, p_ca=recall,
                p_fa=1.-precision if precision is not None else None)


def aggregate(rows):
    rows = [row for row in rows if row['pair_count']]
    total = {key: sum(row[key] for row in rows) for key in ('tp', 'fp', 'fn', 'tn', 'pair_count')}
    tp, fp, fn = (total[key] for key in ('tp', 'fp', 'fn'))
    positive = [r for r in rows if r['positive_count']]
    negative = [r for r in rows if not r['positive_count']]
    predicted = [r for r in rows if r['tp']+r['fp']]
    denominator = tp+fp
    recall, precision = divide(tp, tp+fn), divide(tp, denominator)
    defined_recall = [r['recall'] for r in rows if r['recall'] is not None]
    defined_pfa = [r['p_fa'] for r in rows if r['p_fa'] is not None]
    mean_recall = float(np.mean(defined_recall)) if defined_recall else None
    mean_pfa = float(np.mean(defined_pfa)) if defined_pfa else None
    paper = dict(unit='fraction', eligible_scene_count=len(rows),
        m_pca=mean_recall if len(defined_recall) == len(rows) else None,
        m_pfa=mean_pfa if len(defined_pfa) == len(rows) else None,
        mw_pca=divide(sum((r['tp']+r['fp'])*r['recall'] for r in predicted), denominator)
               if all(r['recall'] is not None for r in predicted) else None,
        mw_pfa=divide(fp, denominator),
        defined_scene_count=dict(m_pca=len(defined_recall), m_pfa=len(defined_pfa)),
        defined_only_mean=dict(m_pca=mean_recall, m_pfa=mean_pfa),
        predicted_positive_count=denominator, prediction_weighted_scene_count=len(predicted))
    return dict(total, scene_count=len(rows), macro_f1=float(np.mean([r['f1'] for r in rows])) if rows else None,
        positive_scene_macro_f1=float(np.mean([r['f1'] for r in positive])) if positive else None,
        positive_scene_count=len(positive), negative_scene_count=len(negative),
        negative_scene_false_acceptance_rate=divide(sum(r['fp'] > 0 for r in negative), len(negative)),
        micro_f1=divide(2*tp, 2*tp+fp+fn) if 2*tp+fp+fn else (1. if rows else None),
        micro_precision=precision, micro_recall=recall, p_ca=recall,
        p_fa=1.-precision if precision is not None else None, paper_metrics=paper)


def ranks(scores, truth):
    values, inverse = np.unique(scores, return_inverse=True)
    positives = np.bincount(inverse[truth], minlength=len(values))
    negatives = np.bincount(inverse[~truth], minlength=len(values))
    p, n = int(positives.sum()), int(negatives.sum())
    assert p and n
    lower_negatives = np.cumsum(negatives)-negatives
    auc = float(np.sum(positives*(lower_negatives+.5*negatives))/(p*n))
    tp, fp = np.cumsum(positives[::-1]), np.cumsum(negatives[::-1])
    predicted = tp+fp
    result = dict(pooled_ap=float(np.sum(positives[::-1]*tp/predicted)/p), pooled_auc=auc)
    for percent in (80, 90):
        i = int(np.flatnonzero(tp >= (p*percent+99)//100)[0])
        result[str(percent)] = dict(threshold=float(values[::-1][i]), tp=int(tp[i]), fp=int(fp[i]),
                                   positive=p, predicted=int(predicted[i]), pfa=float(1.-tp[i]/predicted[i]))
    return result


def dense_ws(n, pairs, scores, threshold, original=False):
    """Replay sum-gain merges by full dense maximum search instead of a heap."""
    weights = scores-OLD_THRESHOLD if original else (scores-threshold)+OLD_THRESHOLD-OLD_THRESHOLD
    gain = np.full((n, n), -np.inf)
    gain[pairs[:, 0], pairs[:, 1]] = weights
    gain[pairs[:, 1], pairs[:, 0]] = weights
    clusters = {i: [i] for i in range(n)}
    while True:
        flat = int(np.argmax(gain))
        a, b = divmod(flat, n)
        if gain[a, b] <= 0.:
            break
        assert a < b
        others = np.asarray([i for i in clusters if i not in (a, b)], int)
        updated = gain[a, others]+gain[b, others]
        gain[a, others] = updated
        gain[others, a] = updated
        clusters[a].extend(clusters.pop(b))
        gain[b, :] = -np.inf
        gain[:, b] = -np.inf
        gain[a, a] = -np.inf
    group = np.empty(n, int)
    for number, key in enumerate(sorted(clusters)):
        group[clusters[key]] = number
    return group


def coverage(original, group, identities, support):
    def state(labels):
        anchors = []
        supported = np.zeros_like(support)
        for gid in np.unique(labels):
            nodes = np.flatnonzero(labels == gid)
            anchors.append(identities[nodes[0]])
            for identity in np.unique(identities[nodes]):
                same = nodes[identities[nodes] == identity]
                supported[same] = support[same].any(axis=0)
        represented = set(anchors)
        return represented, len(anchors)-len(represented), supported
    old, old_duplicates, before = state(original)
    new, duplicates, after = state(group)
    return dict(represented=len(new), identities_lost=len(old-new), identities_gained=len(new-old),
                duplicates=duplicates, duplicate_increase=duplicates-old_duplicates,
                duplicate_increase_scenes=int(duplicates > old_duplicates),
                same_ship_supported=int(after.sum()), same_ship_lost=int((before & ~after).sum()),
                same_ship_recovered=int((~before & after).sum()))


def scene_data(row):
    with np.load(row['motion'], allow_pickle=False) as saved:
        data = {key: saved[key] for key in ('node_ids', 'sources', 'query_seconds', 'support')}
    with np.load(row['evaluation'], allow_pickle=False) as saved:
        for key in ('node_ids', 'query_seconds'):
            np.testing.assert_array_equal(data[key], saved[key])
        data['identities'] = saved['identities']
    return data


def aggregate_arms(rows):
    result = {}
    for arm in ARMS:
        counts = Counter()
        for row in rows:
            counts.update(row['arms'][arm]['coverage'])
        result[arm] = dict(coverage=dict(counts))
        for scope in SCOPES:
            result[arm][scope] = {stage: aggregate([r['arms'][arm][scope][stage] for r in rows])
                                 for stage in STAGES}
    return result


def gates(summary, base):
    def direction(arms):
        a, b = arms['RANK'], arms[base]
        return all(a['mstsa']['decoded'][k] < b['mstsa']['decoded'][k] for k in ('fn', 'fp')) and all(
            a['itsa']['decoded'][k] <= b['itsa']['decoded'][k] for k in ('fn', 'fp'))
    rank = summary['ranking']
    raw = all(rank['mstsa']['RANK'][p]['fp'] < rank['mstsa'][base][p]['fp'] and
              rank['itsa']['RANK'][p]['fp'] <= rank['itsa'][base][p]['fp'] for p in ('80', '90'))
    support = summary['full']['RANK']['coverage']
    retained = all(support[k] == 0 for k in ('identities_lost', 'same_ship_lost', 'duplicate_increase_scenes'))
    components = sum(direction(r) for r in summary['by_component'].values())
    decoded = direction(summary['full'])
    leave = direction(summary['without_largest'])
    return dict(raw=raw, decoded=decoded, retained=retained, joint_benefit_components=components,
                without_largest=leave, passed=raw and decoded and retained and components >= 2 and leave)


def audit_split(root, split, ids, index, components, thresholds, expected, checks, prep, old_archive):
    historical = read(root/f'{split}-scenes.json')
    assert [row['scene'] for row in historical] == ids
    rows, blocks = [], {scope: defaultdict(list) for scope in SCOPES}
    pair_total = 0
    for sid, prior in zip(ids, historical, strict=True):
        data = scene_data(index[sid])
        n = len(data['node_ids'])
        pairs = np.column_stack(np.triu_indices(n, 1))
        with np.load(root/'private'/split/f'{sid}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(pairs, saved['pairs'])
            np.testing.assert_array_equal(saved['score_056'], saved['score_056_T'])
            if split == 'held':
                original = np.load(prep/'private'/f'scores-{sid}.npy', allow_pickle=False)
            else:
                original = old_archive[f'{sid}_scores']
                np.testing.assert_array_equal(old_archive[f'{sid}_node_ids'], data['node_ids'])
                np.testing.assert_array_equal(old_archive[f'{sid}_pairs'], pairs)
            np.testing.assert_array_equal(saved['score_056'], original)
            truth = data['identities'][pairs[:, 0]] == data['identities'][pairs[:, 1]]
            same = data['sources'][pairs[:, 0]] == data['sources'][pairs[:, 1]]
            row = dict(scene=sid, component=components[sid], pairs=len(pairs), arms={})
            for scope, mask in zip(SCOPES, (same, ~same), strict=True):
                blocks[scope]['truth'].append(truth[mask])
            for arm in ARMS:
                scores, group = saved[f'score_{arm}'], saved[f'group_{arm}']
                assert scores.shape == (len(pairs),) and np.isfinite(scores).all() and group.shape == (n,)
                rebuilt = dense_ws(n, pairs, scores, thresholds[arm], original=arm == '056')
                np.testing.assert_array_equal(group, rebuilt, err_msg=f'{split}/{sid}/{arm} WS')
                if split == 'V_select' and arm == '056':
                    old_group = read(root.parent/WS/'private'/f'scene-{sid}.json.gz')['memberships']['WS']
                    np.testing.assert_array_equal(group, old_group)
                raw, decoded = scores >= thresholds[arm], group[pairs[:, 0]] == group[pairs[:, 1]]
                row['arms'][arm] = dict(coverage=coverage(saved['group_056'], group, data['identities'], data['support']))
                choices = (raw, decoded, raw & ~decoded, ~raw & decoded, raw & decoded)
                for scope, mask in zip(SCOPES, (same, ~same), strict=True):
                    row['arms'][arm][scope] = {stage: confusion(truth[mask], decision[mask])
                                              for stage, decision in zip(STAGES, choices, strict=True)}
                    blocks[scope][arm].append(scores[mask])
            checks.verify(row, prior, f'{split}/{sid}')
            pair_total += len(pairs)
            rows.append(row)
    result = dict(scenes=len(ids), full=aggregate_arms(rows), ranking={}, by_component={})
    for scope, values in blocks.items():
        truth = np.concatenate(values['truth'])
        result['ranking'][scope] = {arm: ranks(np.concatenate(values[arm]), truth) for arm in ARMS}
    for component in sorted({r['component'] for r in rows}):
        result['by_component'][component] = aggregate_arms([r for r in rows if r['component'] == component])
    largest = max(result['by_component'], key=lambda c: (sum(r['component'] == c for r in rows), c))
    result['largest_component'] = largest
    result['without_largest'] = aggregate_arms([r for r in rows if r['component'] != largest])
    result['gates'] = {base: gates(result, base) for base in ('056', '056_T', 'BCE')}
    checks.verify(result, expected, split)
    return dict(scenes=len(ids), pairs=pair_total, components=len(result['by_component']),
                replayed_ws_partitions=len(ids)*len(ARMS), gates=result['gates'])


def training_and_pools(root, prep, index, roles, checks):
    training = read(root/'training.json')
    available = Counter()
    for sid in roles['fit']:
        data = scene_data(index[sid])
        pairs = np.column_stack(np.triu_indices(len(data['node_ids']), 1))
        old = np.load(prep/'private'/f'scores-{sid}.npy', allow_pickle=False)
        assert old.shape == (len(pairs),)
        truth = data['identities'][pairs[:, 0]] == data['identities'][pairs[:, 1]]
        same = data['sources'][pairs[:, 0]] == data['sources'][pairs[:, 1]]
        for scope, mask in zip(SCOPES, (same, ~same), strict=True):
            available[scope, 'positive'] += int((truth & mask).sum())
            neg = old[~truth & mask]
            if not len(neg):
                continue
            cut = [np.partition(neg, len(neg)-(len(neg)+d-1)//d)[len(neg)-(len(neg)+d-1)//d]
                   for d in (1000, 100)]
            available[scope, 'top'] += int((neg >= cut[0]).sum())
            available[scope, 'rest'] += int(((neg >= cut[1]) & (neg < cut[0])).sum())
    keys = [(scope, kind) for scope in SCOPES for kind in ('positive', 'top', 'rest')]
    seen = {key: np.zeros(available[key], bool) for key in keys}
    rng = np.random.default_rng(SEED)
    for _ in range(1024):
        for key in keys:
            count = 128 if key[1] == 'positive' else 64
            seen[key][rng.integers(0, available[key], count)] = True
    population = {scope: {kind: dict(available=available[scope, kind], unique_drawn=int(seen[scope, kind].sum()),
                    presentations=1024*(128 if kind == 'positive' else 64))
                    for kind in ('positive', 'top', 'rest')} for scope in SCOPES}
    checks.verify(dict(seed=SEED, updates_per_head=1024, parameters_per_head=5697,
                       u0_equal=True, paired_exposure=True, population=population), training, 'training')
    assert [row['step'] for row in training['curve']] == [1, *range(128, 1025, 128)]
    assert all(np.isfinite(row[arm][field]) for row in training['curve'] for arm in ('BCE', 'RANK')
               for field in ('loss', 'gradient_norm'))
    state = torch.load(root/'private/heads.pt', map_location='cpu', weights_only=True)
    heads = {}
    for arm in ('BCE', 'RANK'):
        head = EvidenceAssociation().eval()
        head.load_state_dict(state[arm], strict=True)
        assert sum(p.numel() for p in head.parameters()) == 5697
        assert all(torch.isfinite(value).all() for value in state[arm].values())
        heads[arm] = head
    return heads, dict(fit_scenes=len(roles['fit']), exposure_reconstructed_from_seed=True,
                       presentations_per_head=524288, population=population,
                       limitation='optimizer updates and initial equality are logged metadata; final weights do not independently prove every step')


@torch.inference_mode()
def calibration_replay(root, prep, index, roles, heads):
    checkpoint = torch.load(root.parent/OLD/'association/final.pt', map_location='cpu', weights_only=False)
    normalizer = checkpoint['normalizer']
    stored = read(root/'calibration.json')
    values = {arm: [] for arm in ('056_T', 'BCE', 'RANK')}
    truth = []
    for sid in roles['calibration']:
        row = index[sid]
        with np.load(row['motion'], allow_pickle=False) as saved:
            motion = {key: saved[key] for key in ('states', 'support', 'nearest_seconds', 'native_count',
                'span_seconds', 'fit_rmse_m', 'ols_velocity_mps', 'endpoint_velocity_mps', 'sources')}
        motion['states'][..., 2] /= 1852/3600
        data = scene_data(row)
        pairs = np.column_stack(np.triu_indices(len(data['node_ids']), 1))
        cross = data['sources'][pairs[:, 0]] != data['sources'][pairs[:, 1]]
        old = np.load(prep/'private'/f'scores-{sid}.npy', allow_pickle=False)
        output = {arm: np.empty(len(pairs)) for arm in ('BCE', 'RANK')}
        for start in range(0, len(pairs), 4096):
            section = slice(start, start+4096)
            raw, _ = features_from_motion(motion, pairs[section])
            x = torch.from_numpy(normalized_features(raw.astype(np.float32), normalizer))
            old_tensor = torch.from_numpy(old[section])
            for arm in output:
                output[arm][section] = heads[arm](x, old_tensor).numpy()
        values['056_T'].append(old[cross])
        for arm in output:
            values[arm].append(output[arm][cross])
        truth.append((data['identities'][pairs[:, 0]] == data['identities'][pairs[:, 1]])[cross])
    truth = np.concatenate(truth)
    result = {}
    for arm, blocks in values.items():
        scores = np.concatenate(blocks)
        recomputed = ranks(scores, truth)['90']
        old_point = stored['points'][arm]
        gap = abs(recomputed['threshold']-old_point['threshold'])
        assert gap <= 1e-4, (arm, gap)
        at_fixed = confusion(truth, scores >= stored['thresholds'][arm])
        result[arm] = dict(cpu_recomputed_point=recomputed, original_cuda_point=old_point,
            threshold_abs_difference=gap, exact_threshold=gap == 0.,
            cpu_at_original_threshold={key: at_fixed[key] for key in ('tp', 'fp', 'fn', 'tn')},
            fixed_threshold_tp_difference=at_fixed['tp']-old_point['tp'],
            fixed_threshold_fp_difference=at_fixed['fp']-old_point['fp'])
    return dict(scenes=len(roles['calibration']), original_calibration_scores_saved=False,
                frozen_head_cpu_recomputed=True, optimizer_updates=0, threshold_tolerance=1e-4, arms=result)


def run(runs, output, manifest_path):
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2)
    resource.setrlimit(resource.RLIMIT_AS, (12*1024**3, 12*1024**3))
    state = dict(status='RUNNING', optimizer_updates=0, raw_archive_reads=0,
                 started_at=datetime.now(timezone.utc).isoformat())
    checks = Comparisons()

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-start,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(output/'run.json', state)

    try:
        root, prep = runs/SCIENCE, runs/PREP
        assert read(root/'run.json')['status'] == read(prep/'run.json')['status'] == 'COMPLETED'
        assert read(runs/(SCIENCE.replace('-attempt-1', '-launch-attempt-1'))/'status.json')['returncode'] == 0
        manifest = read(manifest_path)
        roles = read(prep/'frozen-split.json')
        assert {k: len(v) for k, v in roles.items()} == dict(fit=1748, calibration=175, held=177)
        assert set(sum(roles.values(), [])) == set(manifest['splits']['T_core'])
        components = manifest['components']
        sets = {role: {components[sid] for sid in ids} for role, ids in roles.items()}
        assert all(not sets[a] & sets[b] for a, b in (('fit', 'calibration'), ('fit', 'held'), ('calibration', 'held')))
        index = read(runs/DATA/'private/index.json')
        expected = read(root/'summary.json')
        thresholds = read(root/'calibration.json')['thresholds']
        assert expected['fixed_thresholds'] == thresholds and thresholds['056'] == OLD_THRESHOLD
        for split, ids in (('held', roles['held']), ('V_select', manifest['splits']['V_select'])):
            files = [root/'private'/split/f'{sid}.npz' for sid in ids]
            assert {p.name for p in (root/'private'/split).glob('*.npz')} == {p.name for p in files}
            assert max(p.stat().st_mtime_ns for p in files) < (root/f'{split}-scenes.json').stat().st_mtime_ns
        assert (root/'held-scenes.json').stat().st_mtime_ns <= min(
            p.stat().st_mtime_ns for p in (root/'private/V_select').glob('*.npz'))
        checkpoint(phase='held_v_saved_array_recount')
        split_result = {}
        with np.load(runs/OLD/'association/final-scores.npz', allow_pickle=False) as old_archive:
            for split, ids, source in (('held', roles['held'], 'T_core'), ('V_select', manifest['splits']['V_select'], 'V_select')):
                split_result[split] = audit_split(root, split, ids, index[source], components,
                    thresholds, expected[split], checks, prep, old_archive)
                checkpoint(phase='saved_array_recount', completed_split=split)
        checkpoint(phase='fit_pool_exposure')
        heads, training = training_and_pools(root, prep, index['T_core'], roles, checks)
        checkpoint(phase='cpu_calibration_replay')
        calibration = calibration_replay(root, prep, index['T_core'], roles, heads)
        checks.verify(read(root/'run.json')['optimizer_updates'], 2048, 'reported_total_updates')
        write(output/'audit.json', dict(status='COMPLETED', saved_array_results=split_result,
            exact_comparisons=checks.exact, numerical_comparisons=checks.numerical,
            complete_ws_dense_replay=True, source_split_components_disjoint=True,
            saved_predictions_precede_science_evaluation_tables=True, training=training, calibration=calibration,
            limits=['No retraining or independent reconstruction of optimizer history.',
                    'Calibration scores were absent from science artifacts and were recomputed from fixed heads on CPU.',
                    'Frozen 056 had prior exposure to all T and V; this audit checks reproduction.']))
        checkpoint(status='COMPLETED', phase='done', exit_code=0,
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    run(args.runs.resolve(), args.output.resolve(), args.manifest.resolve())
