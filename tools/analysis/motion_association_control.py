"""Compare cached native motion positions using one complete-pair distance rule.

Zero training; full frozen V only. Position targets/masks are never loaded.
Threshold selection is development evaluation, not held-out performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.data.archive import validate_manifest
from association.evaluation.association import compare_predictions, evaluate_scenes, select_threshold
from association.evaluation.native_motion import METHODS


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def rms_haversine_scores(positions, pairs):
    """Negative RMS metres over every Q64 time; accepts no identity or masks."""
    positions = np.asarray(positions, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    if positions.ndim != 3 or positions.shape[1:] != (64, 2) or not np.isfinite(positions).all():
        raise ValueError('positions must be finite [N,64,2]')
    if pairs.shape != (len(pairs), 2):
        raise ValueError('pairs must be [E,2]')
    first, second = positions[pairs[:, 0]], positions[pairs[:, 1]]
    lat1, lat2 = np.deg2rad(first[..., 0]), np.deg2rad(second[..., 0])
    dlon = np.deg2rad((first[..., 1]-second[..., 1]+180.) % 360.-180.)
    a = np.clip(np.sin((lat1-lat2)/2)**2
                + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2, 0, 1)
    distances = 2*6371000.*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return -np.sqrt(np.mean(distances**2, axis=1))


def auc(labels, scores):
    """Mann–Whitney AUC with average ranks for exact score ties."""
    labels, scores = np.asarray(labels, bool), np.asarray(scores, np.float64)
    p, n = int(labels.sum()), int((~labels).sum())
    if not p or not n:
        return None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends-counts+1+ends)/2
    return float((ranks[inverse][labels].sum()-p*(p+1)/2)/(p*n))


def load_records(predictions_dir, manifest):
    """Read an explicit allowlist from R2 caches; score before reading identities."""
    validate_manifest(manifest)
    directory = Path(predictions_dir)
    scene_ids = manifest['splits']['V_select']
    provenance = json.loads((directory.parent/'run.json').read_text())
    if (provenance['status'] != 'COMPLETED' or provenance['scene_ids'] != scene_ids
            or provenance['data'] != dict(length=64, max_observations=128, time_unit_seconds=20.)
            or provenance['inference_uses_reference'] or provenance['inference_uses_identity']
            or provenance['test_opened']):
        raise ValueError('cache provenance must be completed, native-only, complete V with frozen data settings')
    index = json.loads((directory/'index.json').read_text())
    if not scene_ids or [row['scene_id'] for row in index] != scene_ids:
        raise ValueError('prediction index must match complete V_select order exactly')
    records, payload = {method: [] for method in METHODS}, {'scene_ids': np.asarray(scene_ids)}
    for item in index:
        sid = item['scene_id']
        filename = Path(item['path'])
        if filename.name != str(filename):
            raise ValueError('cache index paths must name files in predictions-dir')
        with np.load(directory/filename, allow_pickle=False) as cache:
            if str(cache['scene_id'].item()) != sid:
                raise ValueError('cache scene_id mismatch')
            node_ids, sources = cache['node_ids'], cache['sources']
            n = len(node_ids)
            query, support = cache['query_seconds'], cache['region_inside_support']
            if (n != item['fragment_count'] or len(np.unique(node_ids)) != n
                    or sources.shape != (n,) or support.shape != (n, 64)
                    or support.dtype != bool or query.shape != (64,)
                    or not np.isfinite(query).all() or not np.all(np.diff(query) > 0)):
                raise ValueError('cache population/query/support metadata mismatch')
            pairs = np.column_stack(np.triu_indices(n, 1))
            joint = (support[pairs[:, 0]] & support[pairs[:, 1]]).sum(axis=1)
            scores = {}
            for method in METHODS:
                positions = cache[method]
                if positions.shape != (n, 64, 2):
                    raise ValueError('method population differs from cache metadata')
                scores[method] = rms_haversine_scores(positions, pairs)
            identities = cache['identities']  # Evaluation labels enter only after every score exists.
            if identities.shape != (n,):
                raise ValueError('identity population differs from cache metadata')
            labels = identities[pairs[:, 0]] == identities[pairs[:, 1]]
        common = dict(scene_id=sid, component_id=manifest['components'][sid], pairs=pairs,
                      labels=labels, node_ids=node_ids, sources=sources, joint_support_queries=joint)
        for method in METHODS:
            records[method].append(dict(common, scores=scores[method]))
            payload[f'{sid}_{method}_scores'] = scores[method]
        for key in ('pairs', 'labels', 'sources', 'node_ids', 'joint_support_queries'):
            payload[f'{sid}_{key}'] = common[key]
    payload['component_ids'] = np.asarray([manifest['components'][sid] for sid in scene_ids])
    return records, payload


def _subset(records, kind):
    result = []
    for record in records:
        pairs = record['pairs']
        same = record['sources'][pairs[:, 0]] == record['sources'][pairs[:, 1]]
        mask = dict(itsa=same, mstsa=~same,
                    joint_q_support=record['joint_support_queries'] > 0,
                    no_joint_q_support=record['joint_support_queries'] == 0)[kind]
        result.append({**record, **{key: record[key][mask] for key in
                       ('pairs', 'scores', 'labels', 'joint_support_queries')}})
    return result


def _auc_summary(records):
    rows = [dict(scene_id=r['scene_id'], roc_auc=auc(r['labels'], r['scores'])) for r in records]
    defined = [row['roc_auc'] for row in rows if row['roc_auc'] is not None]
    return dict(pooled_roc_auc=auc(np.concatenate([r['labels'] for r in records]),
                                  np.concatenate([r['scores'] for r in records])),
                defined_scene_macro_auc=float(np.mean(defined)) if defined else None,
                defined_scene_count=len(defined), undefined_scene_count=len(rows)-len(defined),
                per_scene=rows)


def run(predictions_dir, manifest_path, output_dir, command=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    run_state = dict(status='RUNNING', command=command, predictions_dir=str(Path(predictions_dir).resolve()),
                     manifest=str(Path(manifest_path).resolve()), device='cpu', optimizer_updates=0,
                     reference_read=False, position_targets_read=False, test_opened=False,
                     training_data_read=False, inference_uses_identity=False)
    _write(output/'run.json', run_state)
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        records, payload = load_records(predictions_dir, manifest)
        baseline = records['native_hold']
        selections = {method: select_threshold(rows) for method, rows in records.items()}
        hold_threshold = selections['native_hold']['threshold']
        methods = {}
        for method, rows in records.items():
            groups = {kind: _subset(rows, kind) for kind in
                      ('itsa', 'mstsa', 'joint_q_support', 'no_joint_q_support')}
            own_threshold = selections[method]['threshold']
            methods[method] = dict(selection=selections[method], auc=_auc_summary(rows),
                subgroup_auc={kind: _auc_summary(group) for kind, group in groups.items()},
                own_threshold=evaluate_scenes(rows, own_threshold),
                hold_threshold=evaluate_scenes(rows, hold_threshold),
                support_strata={kind: dict(
                    own_threshold=evaluate_scenes(groups[kind], own_threshold),
                    hold_threshold=evaluate_scenes(groups[kind], hold_threshold))
                    for kind in ('joint_q_support', 'no_joint_q_support')})
            if method != 'native_hold':
                methods[method]['changes_vs_hold'] = {
                    name: dict(overall=compare_predictions(baseline, rows, hold_threshold, threshold),
                        **{kind: compare_predictions(_subset(baseline, kind), group,
                                                      hold_threshold, threshold)
                           for kind, group in groups.items()})
                    for name, threshold in (('own_threshold', own_threshold), ('hold_threshold', hold_threshold))}
        population = dict(scenes=len(baseline), components=len({r['component_id'] for r in baseline}),
            nodes=sum(len(r['node_ids']) for r in baseline), pairs=sum(len(r['pairs']) for r in baseline),
            positives=sum(int(r['labels'].sum()) for r in baseline),
            no_joint_q_support_pairs=sum(int((r['joint_support_queries'] == 0).sum()) for r in baseline))
        report = dict(**run_state, population=population, scene_ids=manifest['splits']['V_select'],
            method='negative RMS haversine distance over all Q64, earth radius 6371000m, float64',
            candidate_filter='none; all unordered i<j pairs; no support or source gating',
            selection='each arm selects scene-macro F1 on same full V; also evaluate every arm at hold threshold',
            support_definition='joint_q_support means at least one Q64 time inside both native endpoint spans; '
                               'zero does not imply disjoint continuous native supports; descriptive only',
            cache_provenance='R2 native predictions do not use reference; its separate position evaluation did. '
                             'This reader never accesses target or target_valid arrays.',
            limitations=['development selection and reporting reuse V; not held-out performance',
                         'full-fragment offline ordinary controls, not a learned or online contribution',
                         'association edge comparison only; no grouping or fusion result'],
            methods=methods)
        np.savez_compressed(output/'scores.npz', **payload)
        report.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        _write(output/'summary.json', report)
        run_state.update(status='COMPLETED', population=population, elapsed_seconds=report['elapsed_seconds'])
        _write(output/'run.json', run_state)
        return report
    except BaseException as exc:
        run_state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}',
                         elapsed_seconds=time.monotonic()-started)
        _write(output/'run.json', run_state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions-dir', 'manifest', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    result = run(args.predictions_dir, args.manifest, args.output_dir, command)
    print(json.dumps(dict(status=result['status'], population=result['population'],
        macro_f1={name: row['own_threshold']['macro_f1'] for name, row in result['methods'].items()})))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
