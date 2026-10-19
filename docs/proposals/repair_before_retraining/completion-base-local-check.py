"""Four fixed V scenes: visible-position control, no learned model or training.

Run from any directory with the repository Python. Reads only the first four
V_select scenes of the original manifest, never creates a replacement split.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from association.data.archive import MTADArchive
from association.data.native import _record
from association.data.reference_states import iter_reference_states
from association.data.targets import build_scene_fusion_targets
from association.data.timed import prepare_timed_scene
from association.evaluation.physical import evaluate_aligned_fusion
from association.evaluation.scene import INPUT_DISTANCE_REGIONS
from association.evaluation.tmtaff import _aggregate_regions, scene_input_support
from association.models.cooperative import CooperativeModel

ARCHIVE = Path('/Users/a1/Downloads/数据集level2.zip')
MANIFEST = '612afea:HandOver/results/EVIDENCE-20261018-012/prepared007/manifest.json'
SCENES = ['1671', '2056', '290', '113']
REGIONS = ('all', 'inside_support', 'before_support', 'after_support',
           'exact_observation', 'inside_nonobserved')
OUTPUT = Path(__file__).with_suffix('.json')


def numpy_base(scene):
    """Independent physical-second float64 interpolation / endpoint hold."""
    prediction = np.full((len(scene['node_ids']), 64, 4), np.nan)
    for node, observed in enumerate(scene['native_observed']):
        times = scene['native_times'][node, observed]
        positions = scene['native_values'][node, observed, :2]
        for j, query in enumerate(scene['grid']):
            right = int(np.searchsorted(times, query))
            if right == 0 or right == len(times):
                value = positions[0 if right == 0 else -1].copy()
            else:
                left = right - 1
                alpha = (query-times[left])/(times[right]-times[left])
                value = positions[left].copy()
                value[0] += alpha*(positions[right, 0]-positions[left, 0])
                value[1] += alpha*((positions[right, 1]-positions[left, 1]+180.) % 360.-180.)
            value[1] = (value[1]+180.) % 360.-180.
            prediction[node, j, :2] = value
    return prediction


def position_only(report):
    return {kind: {region: metrics['position'] for region, metrics in values.items()}
            for kind, values in report.items()}


def run():
    started = time.monotonic()
    torch.set_num_threads(1)
    manifest = json.loads(subprocess.check_output(['git', 'show', MANIFEST], cwd=ROOT))
    if manifest['splits']['V_select'][:4] != SCENES:
        raise ValueError('fixed original V scene order changed')
    alignment = json.loads((ROOT/'configs/tmtaff-time-reference-alignment.json').read_text())
    scenes, references_read = [], []
    with MTADArchive(ARCHIVE) as archive:
        for sid in SCENES:
            native = _record(archive.load_scene(sid), manifest['components'][sid], 'V_select', 128)
            scene = prepare_timed_scene(native, length=64, max_observations=128, time_unit_seconds=20.)
            scene['target_identity_ids'] = list(dict.fromkeys(
                '-'.join(row['mmsi'].split('-')[:2]) for row in archive.read_labels(sid)))
            scenes.append(scene)
        observation_members = archive.members_read.copy()
    # Iteration reads just these four reference CSVs; other ZIP entries are indexed only.
    for scene, reference in zip(scenes, iter_reference_states(ARCHIVE, manifest,
            split='V_select', scene_ids=SCENES)):
        policy = alignment[scene['scene_id']]
        if policy['max_gap_seconds'] != 600.:
            raise ValueError('existing reference gap differs from approved 600 seconds')
        scene['fusion_targets'] = build_scene_fusion_targets(scene, reference,
            target_identity_ids=scene['target_identity_ids'], longitude_policy='shortest_arc', **policy)
        references_read.append(dict(scene_id=scene['scene_id'], member=reference['member']))
    all_reports = {key: [] for key in ('position_base_fp32', 'independent_float64')}
    per_scene, numerical_differences = [], []
    for scene in scenes:
        n, q = len(scene['node_ids']), 64
        encoded = dict(memory=torch.empty((n, 0, 0)),
            native_positions=torch.as_tensor(scene['native_values'][..., :2], dtype=torch.float32),
            times=torch.as_tensor((scene['native_times']-scene['time_origin_seconds'])/20., dtype=torch.float32),
            queries=torch.as_tensor(np.broadcast_to((scene['grid']-scene['time_origin_seconds'])/20.,
                                                   (n, q)).copy(), dtype=torch.float32),
            observed=torch.as_tensor(scene['native_observed']))
        base32 = np.full((n, q, 4), np.nan)
        with torch.inference_mode():
            base32[..., :2] = CooperativeModel.position_base(
                SimpleNamespace(config=SimpleNamespace(length=q)), encoded, list(range(n))).numpy()
        base64 = numpy_base(scene)
        prediction_valid = np.zeros((n, q, 4), bool)
        prediction_valid[..., :2] = True
        truth = np.stack([scene['fusion_targets'][identity]['states'] for identity in scene['identities']])
        valid = np.stack([scene['fusion_targets'][identity]['valid'] for identity in scene['identities']])
        support = scene_input_support(scene)
        nearest, inside = support['nearest_seconds'], support['support']
        before, after = np.zeros((n, q), bool), np.zeros((n, q), bool)
        for node, observed in enumerate(scene['native_observed']):
            times = scene['native_times'][node, observed]
            before[node], after[node] = scene['grid'] < times.min(), scene['grid'] > times.max()
        exact = nearest == 0
        masks = dict(zip(REGIONS, (np.ones((n, q), bool), inside, before, after, exact, inside & ~exact)))
        distance_masks = (nearest <= 300, (nearest > 300) & (nearest <= 600),
            (nearest > 600) & (nearest <= 1800), (nearest > 1800) & np.isfinite(nearest), ~np.isfinite(nearest))
        methods = {}
        for name, prediction in (('position_base_fp32', base32), ('independent_float64', base64)):
            def metric(mask):
                return evaluate_aligned_fusion(prediction, truth, valid,
                    prediction_valid=prediction_valid, region_mask=mask)
            report = dict(regions={key: metric(mask) for key, mask in masks.items()},
                input_distance_metrics={key: metric(mask) for key, mask in zip(INPUT_DISTANCE_REGIONS, distance_masks)})
            all_reports[name].append(report)
            methods[name] = position_only(report)
        difference = evaluate_aligned_fusion(base32, base64, prediction_valid,
            prediction_valid=prediction_valid, region_mask=masks['all'])['position']
        lat1, lat2 = np.deg2rad(base32[..., 0]), np.deg2rad(base64[..., 0])
        dlon = np.deg2rad((base32[..., 1]-base64[..., 1]+180.) % 360.-180.)
        hav = np.clip(np.sin((lat1-lat2)/2.)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.)**2, 0, 1)
        difference['max_error_metres'] = float((2*6371000.*np.arctan2(np.sqrt(hav), np.sqrt(1-hav))).max())
        numerical_differences.append(difference)
        per_scene.append(dict(scene_id=scene['scene_id'], fragment_count=n,
            unique_input_identity_count=len(set(scene['identities'])), query_count=q,
            reference_unavailable_fragment_count=int((~valid.any(axis=(1, 2))).sum()),
            original_target_identity_count=len(scene['target_identity_ids']),
            reference_alignment={**alignment[scene['scene_id']], 'longitude_policy': 'shortest_arc'},
            region_query_counts={key: int(mask.sum()) for key, mask in masks.items()},
            methods=methods, float32_vs_float64=difference))
    count = sum(row['intersection_count'] for row in numerical_differences)
    return dict(status='COMPLETED', archive=str(ARCHIVE), manifest_git_object=MANIFEST,
        scene_ids=SCENES, fragment_count=sum(len(scene['node_ids']) for scene in scenes),
        data=dict(cap=128, queries=64, time_unit_seconds=20., reference_gap_seconds=600., longitude_policy='shortest_arc'),
        methods={name: position_only(dict(regions=_aggregate_regions(rows, 'regions', REGIONS),
            input_distance_metrics=_aggregate_regions(rows, 'input_distance_metrics', INPUT_DISTANCE_REGIONS)))
            for name, rows in all_reports.items()},
        float32_vs_float64=dict(query_count=count,
            mae_metres=sum(row['mae']*row['intersection_count'] for row in numerical_differences)/count,
            max_error_metres=max(row['max_error_metres'] for row in numerical_differences)),
        per_scene=per_scene, observation_members_read=observation_members, reference_members_read=references_read,
        model_constructed=False, checkpoint_loaded=False, normalizer_fitted=False, optimizer_updates=0,
        test_opened=False, device='cpu', cpu_threads=1, elapsed_seconds=time.monotonic()-started,
        limitations=['four fixed development scenes; not full V or learned 021 quality',
            'no checkpoint normalizer FP32 round trip; raw retained physical positions cast to FP32',
            'no speed or heading control', 'point-pooled metrics over every fragment; repeated identities retained',
            '9001/reference reuse limits interpretation of errors within visible support'])


if __name__ == '__main__':
    try:
        result = run()
    except Exception as error:
        OUTPUT.write_text(json.dumps(dict(status='FAILED', error=f'{type(error).__name__}: {error}',
                                         test_opened=False), indent=2)+'\n')
        raise
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], scenes=result['scene_ids'], fragments=result['fragment_count'],
        elapsed_seconds=result['elapsed_seconds'], position=result['methods']['position_base_fp32']['regions']['all'],
        numerical_difference=result['float32_vs_float64']), ensure_ascii=False))
