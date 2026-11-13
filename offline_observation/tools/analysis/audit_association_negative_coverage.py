"""C95 saved-score, calibration and independent sampling audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import resource
import time

import numpy as np
import torch

from association.models.evidence_association import EvidenceAssociation
from tools.analysis import audit_offline_association_tail as audit


SCIENCE = 'LEVEL1-OFFLINE-NEGATIVE-COVERAGE-20261112-attempt-1'
HEADS = ('T_BCE', 'T_ANCHORED', 'M_BCE', 'M_ANCHORED')
ARMS = ('056', '056_T', *HEADS)
SCOPES = audit.SCOPES
SEED, STEPS, RANK_WEIGHT = 20261112, 1024, .5


def pair_truth(data, pairs):
    left, right = pairs.T
    identities, sources = data['identities'], data['sources']
    same = sources[left] == sources[right]
    return identities[left] == identities[right], (same, ~same)


def calibration(root, prep, index, ids, stored, checks):
    blocks = {scope: defaultdict(list) for scope in SCOPES}
    for sid in ids:
        data = audit.scene_data(index[sid])
        pairs = np.column_stack(np.triu_indices(len(data['node_ids']), 1))
        truth, masks = pair_truth(data, pairs)
        with np.load(root/'private/calibration'/f'{sid}.npz', allow_pickle=False) as z:
            for field, expected in (('pairs', pairs), ('nodes', data['node_ids']), ('sources', data['sources'])):
                np.testing.assert_array_equal(z[field], expected)
            old = np.load(prep/'private'/f'scores-{sid}.npy', allow_pickle=False)
            for name in ('056', '056_T'):
                np.testing.assert_array_equal(z[name], old)
            for scope, mask in zip(SCOPES, masks, strict=True):
                blocks[scope]['truth'].append(truth[mask])
                for arm in ARMS[1:]:
                    assert z[arm].shape == truth.shape and np.isfinite(z[arm]).all()
                    blocks[scope][arm].append(z[arm][mask])
    points, ties = {}, {}
    for scope, block in blocks.items():
        truth = np.concatenate(block['truth'])
        positives = int(truth.sum())
        points[scope], ties[scope] = {}, {}
        for arm in ARMS[1:]:
            scores = np.concatenate(block[arm])
            positive_scores = scores[truth]
            points[scope][arm], ties[scope][arm] = {}, {}
            for percent in (80, 90):
                wanted = (positives*percent+99)//100
                cutoff = float(np.partition(positive_scores, positives-wanted)[positives-wanted])
                ge, gt = scores >= cutoff, scores > cutoff
                tp, fp = int((ge & truth).sum()), int((ge & ~truth).sum())
                assert int((gt & truth).sum()) < wanted <= tp
                points[scope][arm][str(percent)] = dict(threshold=cutoff, tp=tp, fp=fp,
                    positive=positives, predicted=tp+fp, pfa=1.-tp/(tp+fp))
                ties[scope][arm][str(percent)] = dict(size=int((ge & ~gt).sum()),
                    true=int((ge & ~gt & truth).sum()))
    thresholds = {'056': audit.OLD_THRESHOLD,
                  **{name: points['mstsa'][name]['90']['threshold'] for name in ARMS[1:]}}
    checks.verify(points, stored['points'], 'calibration.points')
    assert thresholds == stored['thresholds']
    return dict(scenes=len(ids), saved_pair_arrays_only=True, thresholds=thresholds, points=points, ties=ties)


def pools_and_schedule(prep, index, ids, recorded):
    available, pieces = Counter(), {scope: [] for scope in SCOPES}
    for sid in ids:
        row = index[sid]
        data = audit.scene_data(row)
        pairs = np.column_stack(np.triu_indices(len(data['node_ids']), 1))
        truth, masks = pair_truth(data, pairs)
        scores = np.load(prep/'private'/f'scores-{sid}.npy', allow_pickle=False)
        assert scores.shape == truth.shape
        for scope, mask in zip(SCOPES, masks, strict=True):
            available[scope, 'positive'] += int((mask & truth).sum())
            negative_scores = scores[mask & ~truth]
            n = len(negative_scores)
            if not n:
                continue
            k0, k1 = (n-(n+d-1)//d for d in (1000, 100))
            tail0, tail1 = np.partition(negative_scores, (k0, k1))[[k0, k1]]
            top = int((negative_scores >= tail0).sum())
            rest = int(((negative_scores < tail0) & (negative_scores >= tail1)).sum())
            available[scope, 'top'] += top
            available[scope, 'rest'] += rest
            pieces[scope].append((row['component_id'], n-top-rest))

    keys = [(scope, kind) for scope in SCOPES for kind in ('positive', 'top', 'rest')]
    seen = {key: np.zeros((2, available[key]), bool) for key in keys}
    rng = np.random.default_rng(SEED)
    for _ in range(STEPS):
        for key in keys:
            full, mixed = (128, 128) if key[1] == 'positive' else (64, 32)
            sample = rng.integers(0, available[key], full)
            seen[key][0, sample] = True
            seen[key][1, sample[:mixed]] = True
    pool_counts = {scope: {kind: dict(available=available[scope, kind],
                                    tail_unique=int(seen[scope, kind][0].sum()),
                                    mixed_unique=int(seen[scope, kind][1].sum()))
                           for kind in ('positive', 'top', 'rest')} for scope in SCOPES}
    assert pool_counts == recorded['sampled_unique']
    rng = np.random.default_rng(SEED+1)
    background = {}
    for scope in SCOPES:
        ends = np.cumsum([n for _, n in pieces[scope]])
        positions = np.unique(rng.integers(0, int(ends[-1]), (STEPS, 64)))
        piece_indices, totals = np.unique(np.searchsorted(ends, positions, side='right'), return_counts=True)
        by_component = Counter()
        for piece, total in zip(piece_indices, totals, strict=True):
            by_component[pieces[scope][piece][0]] += int(total)
        background[scope] = dict(available=int(ends[-1]), unique=len(positions), presentations=STEPS*64,
                                 unique_by_component=dict(by_component))
    assert background == recorded['background']
    return dict(fit_scenes=len(ids), reconstructed_pool_counts=pool_counts, reconstructed_background=background,
                reconstructed_draws=True, background_partition='negative ranks strictly below complete top 1% boundary',
                common_per_relation_step=dict(positive=128, top=32, rest=32),
                rank_comparisons_per_anchored_arm=2*64*STEPS)


def training(root, runs, prep, index, roles, checks):
    recorded = audit.read(root/'training.json')
    checks.verify(dict(updates_per_arm=STEPS, presentations_per_arm=512*STEPS,
        seed=SEED, background_seed=SEED+1, rank_weight=RANK_WEIGHT, u0_equal=True), recorded, 'training')
    assert [entry['step'] for entry in recorded['curve']] == [1, *range(128, 1025, 128)]
    for row in recorded['curve']:
        for arm in HEADS:
            entry = row[arm]
            assert np.isfinite([entry[key] for key in ('bce', 'loss', 'gradient_norm')]).all()
            rank = entry['rank']
            assert (rank is not None) == arm.endswith('ANCHORED')
            np.testing.assert_allclose(entry['loss'], entry['bce'] + RANK_WEIGHT*(rank or 0.), atol=1e-12)
    result = pools_and_schedule(prep, index, roles['fit'], recorded)
    state = torch.load(root/'private/heads.pt', map_location='cpu', weights_only=True)
    old = torch.load(runs/audit.SCIENCE/'private/heads.pt', map_location='cpu', weights_only=True)['BCE']
    assert set(state) == set(HEADS)
    for weights in state.values():
        model = EvidenceAssociation()
        model.load_state_dict(weights, strict=True)
        assert sum(parameter.numel() for parameter in model.parameters()) == 5697
        assert all(torch.isfinite(value).all() for value in weights.values())
    delta = max(float((value-old[key]).abs().max()) for key, value in state['T_BCE'].items())
    checks.verify(delta, recorded['c92_bce_weight_max_abs_delta'], 'C92 control weight maximum delta')
    return dict(result, parameters_per_arm=5697, logged_updates_per_arm=STEPS, presentations_per_arm=STEPS*512,
                c92_bce_weight_max_abs_delta=delta,
                limitation='Final weights cannot independently prove all optimizer steps or initial state equality.')


def factorial(result):
    def contrast(scope, left, right):
        a, b = (result['ranking'][scope][arm] for arm in (left, right))
        x, y = (result['full'][arm][scope]['decoded'] for arm in (left, right))
        return dict(fp_at_recall={p: a[p]['fp']-b[p]['fp'] for p in ('80', '90')},
                    decoded={key: x[key]-y[key] for key in ('fn', 'fp')})
    terms = {'background_bce': ('M_BCE', 'T_BCE'), 'background_anchored': ('M_ANCHORED', 'T_ANCHORED'),
             'rank_tail': ('T_ANCHORED', 'T_BCE'), 'rank_mixed': ('M_ANCHORED', 'M_BCE')}
    effects = {name: {scope: contrast(scope, *arms) for scope in SCOPES} for name, arms in terms.items()}
    effects['interaction'] = {scope: {metric: {key: effects['rank_mixed'][scope][metric][key]
                   - effects['rank_tail'][scope][metric][key] for key in values}
                   for metric, values in effects['rank_tail'][scope].items()} for scope in SCOPES}
    return effects


def run(runs, output, manifest_path):
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2)
    resource.setrlimit(resource.RLIMIT_AS, (12*1024**3, 12*1024**3))
    state = dict(status='RUNNING', optimizer_updates=0, new_predictions=0, raw_archive_reads=0,
                 started_at=datetime.now(timezone.utc).isoformat())
    checks = audit.Comparisons()

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-start,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        audit.write(output/'run.json', state)

    try:
        root, prep = runs/SCIENCE, runs/audit.PREP
        scientific = audit.read(root/'run.json')
        assert scientific['status'] == 'COMPLETED' and scientific['optimizer_updates'] == 4096
        assert scientific['test_opened'] is False
        assert audit.read(runs/SCIENCE.replace('-attempt-', '-launch-attempt-')/'status.json')['returncode'] == 0
        manifest = audit.read(manifest_path)
        index = audit.read(runs/audit.DATA/'private/index.json')
        roles = audit.read(prep/'frozen-split.json')
        components = manifest['components']
        assert {k: len(v) for k, v in roles.items()} == dict(fit=1748, calibration=175, held=177)
        assert set(sum(roles.values(), [])) == set(manifest['splits']['T_core'])
        component_sets = [{components[sid] for sid in role} for role in roles.values()]
        assert sum(map(len, component_sets)) == len(set.union(*component_sets))
        splits = {'held': roles['held'], 'V_select': manifest['splits']['V_select']}
        calibration_json = root/'calibration.json'
        expected = audit.read(root/'summary.json')
        stored = audit.read(calibration_json)
        assert expected['fixed_thresholds'] == stored['thresholds']
        for split, ids in dict(calibration=roles['calibration'], **splits).items():
            directory = root/'private'/split
            expected_table = calibration_json if split == 'calibration' else root/f'{split}-scenes.json'
            files = list(directory.glob('*.npz'))
            assert {p.stem for p in files} == set(ids)
            assert max(p.stat().st_mtime_ns for p in files) < expected_table.stat().st_mtime_ns
        checkpoint(phase='calibration')
        cal = calibration(root, prep, index['T_core'], roles['calibration'], stored, checks)
        subsets = {}
        with np.load(runs/audit.OLD/'association/final-scores.npz', allow_pickle=False) as archive:
            assert archive['scene_ids'].tolist() == splits['V_select']
            for split, ids in splits.items():
                checkpoint(phase=split)
                subsets[split] = audit.audit_split(root, split, ids,
                    index['T_core' if split == 'held' else split], components, stored['thresholds'],
                    expected[split], checks, prep, archive, arms=ARMS, factorial=factorial)
        checkpoint(phase='fit_pool_exposure')
        fit = training(root, runs, prep, index['T_core'], roles, checks)
        audit.write(output/'audit.json', dict(status='COMPLETED', saved_array_results=subsets,
            exact_comparisons=checks.exact, numerical_comparisons=checks.numerical,
            source_split_components_disjoint=True, complete_ws_dense_replay=True,
            calibration=cal, training=fit, saved_predictions_precede_evaluation_tables=True))
        checkpoint(status='COMPLETED', phase='done', exit_code=0,
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    run(args.runs.resolve(), args.output.resolve(), args.manifest.resolve())
