"""CPU position-distance control on the unchanged full V_select population.

No learned model, reference, training data, gating or test access. This is not
the paper's exact ED implementation. Formula is fixed before looking at scores:
native cap128 -> scene Q64 interpolation/endpoint hold -> negative RMS geodesic
distance. The control selects its own development macro-F1 threshold once.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from association.data.archive import MTADArchive
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from association.evaluation.association import evaluate_scenes, select_threshold

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('existing_position_control',
                                            HERE/'completion-base-local-check.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ARCHIVE = Path('/Users/a1/Downloads/数据集level2.zip')
MANIFEST = '612afea:HandOver/results/EVIDENCE-20261018-012/prepared007/manifest.json'


def auc(labels, scores):
    """Mann-Whitney ROC AUC with exact average ranks for tied scores."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - counts + 1 + ends) / 2
    p, n = int(labels.sum()), int((~labels).sum())
    return float((ranks[inverse][labels].sum()-p*(p+1)/2)/(p*n)) if p and n else None


def run():
    start = time.monotonic()
    manifest = json.loads(subprocess.check_output(['git', 'show', MANIFEST], cwd=ROOT))
    ids = manifest['splits']['V_select']
    records, payload, per_scene_input = [], {}, []
    with MTADArchive(ARCHIVE) as archive:
        for sid in ids:
            native = _record(archive.load_scene(sid), manifest['components'][sid], 'V_select', 128)
            scene = prepare_timed_scene(native, length=64, max_observations=128, time_unit_seconds=20.)
            positions = base.numpy_base(scene)[..., :2]
            pairs = scene['pairs']
            first, second = positions[pairs[:, 0]], positions[pairs[:, 1]]
            lat1, lat2 = np.deg2rad(first[..., 0]), np.deg2rad(second[..., 0])
            dlon = np.deg2rad((first[..., 1]-second[..., 1]+180.) % 360.-180.)
            a = np.clip(np.sin((lat1-lat2)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2, 0, 1)
            distances = 2*6371000.*np.arctan2(np.sqrt(a), np.sqrt(1-a))
            scores = -np.sqrt(np.mean(distances**2, axis=1))
            record = dict(scene_id=sid, component_id=scene['component_id'], scores=scores,
                          labels=scene['labels'].astype(bool), pairs=pairs,
                          sources=np.asarray(scene['sources']), node_ids=np.asarray(scene['node_ids']))
            records.append(record)
            for key in ('scores', 'labels', 'pairs', 'sources', 'node_ids'):
                payload[f'{sid}_{key}'] = record[key]
            # Small inference input retained for independent numerical review.
            payload[f'{sid}_positions'] = positions
            payload[f'{sid}_grid'] = scene['grid']
            per_scene_input.append(dict(scene_id=sid, nodes=len(positions), pairs=len(pairs),
                                        positives=int(record['labels'].sum())))
        members = list(archive.members_read)
    population = dict(scenes=len(records), nodes=sum(len(r['sources']) for r in records),
                      pairs=sum(len(r['pairs']) for r in records),
                      positives=sum(int(r['labels'].sum()) for r in records))
    if population != dict(scenes=64, nodes=1635, pairs=29008, positives=3297):
        raise ValueError(f'population differs from preserved 020/021: {population}')
    selection = select_threshold(records)
    metrics = evaluate_scenes(records, selection['threshold'])
    labels = np.concatenate([r['labels'] for r in records])
    scores = np.concatenate([r['scores'] for r in records])
    payload['scene_ids'] = np.asarray(ids)
    np.savez_compressed(HERE/'association-distance-local-records.npz', **payload)
    return dict(status='COMPLETED', command=[sys.executable, str(Path(__file__).resolve())],
        method='native Q64 position interpolation + endpoint hold, negative RMS haversine metres',
        archive=str(ARCHIVE), manifest_git_object=MANIFEST, scene_ids=ids,
        population=population, per_scene_input=per_scene_input,
        configuration=dict(cap=128, queries=64, time_unit_seconds=20.,
                           longitude_policy='shortest_arc', earth_radius_metres=6371000.,
                           score_dtype='float64', query_denominator=64, candidate_filter='none'),
        selection=selection, metrics=metrics, pooled_roc_auc=auc(labels, scores),
        observation_members_read=members, reference_read=False, training_data_read=False,
        model_constructed=False, checkpoint_loaded=False, optimizer_updates=0,
        test_opened=False, device='cpu', elapsed_seconds=time.monotonic()-start,
        limitations=['development threshold selected and reported on V, not held-out performance',
            'only positions; not paper ED/DTW or a trained-method reproduction',
            'endpoint hold supplies all query times, including unsupported long gaps',
            'no raw 020/021 pair scores available here for paired correction analysis',
            'physical CPU control is float64; neural inference remains original FP32'])


if __name__ == '__main__':
    result = run()
    (HERE/'association-distance-local-check.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'population', 'elapsed_seconds', 'pooled_roc_auc')},
                     ensure_ascii=False))
    print(json.dumps({k: result['metrics'][k] for k in
        ('macro_f1', 'micro_f1', 'micro_precision', 'micro_recall', 'tp', 'fp', 'fn', 'tn')}, ensure_ascii=False))
