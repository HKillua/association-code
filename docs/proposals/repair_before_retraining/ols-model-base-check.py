"""Audit production FP32 OLS-base arithmetic against the frozen native control.

No network training or reference CSV reads. T_core selected observations fit
only the original position moments; existing V cache targets score outputs
after prediction. This is a numerical audit, not learned-model performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from association.data.archive import MTADArchive, validate_manifest
from association.data.native import uniform_observation_indices
from association.evaluation.native_motion import predict_native_positions, position_metrics
from association.models.cooperative import CooperativeModel


def distances(a, b):
    # Independent unit-vector cross/dot geometry, unlike production haversine.
    a, b = np.deg2rad(a), np.deg2rad(b)
    def vector(x):
        return np.stack((np.cos(x[..., 0])*np.cos(x[..., 1]),
                         np.cos(x[..., 0])*np.sin(x[..., 1]), np.sin(x[..., 0])), -1)
    a, b = vector(a), vector(b)
    return 6371000.*np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a*b).sum(-1))


def run(archive_path, manifest_path, predictions_dir, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    torch.set_num_threads(1)
    manifest = json.loads(Path(manifest_path).read_text())
    validate_manifest(manifest)
    mean, m2, count = np.zeros(2), np.zeros(2), 0
    # Same per-scene online population moments as fit_timed_normalizer, bounded
    # memory. No labels, validation statistics, targets or reference CSVs fit it.
    with MTADArchive(archive_path) as archive:
        for sid in manifest['splits']['T_core']:
            native = archive.load_scene(sid)
            arrays = [np.asarray(t.values)[uniform_observation_indices(len(t.times), 128), :2]
                      for t in native.tracks]
            if not arrays:
                continue
            points = np.concatenate(arrays)
            local_mean = points.mean(0)
            local_m2 = ((points-local_mean)**2).sum(0)
            delta, combined = local_mean-mean, count+len(points)
            m2 += local_m2+delta**2*(count*len(points)/combined)
            mean += delta*len(points)/combined
            count = combined
        observation_members = list(archive.members_read)
    std = np.sqrt(np.maximum(m2/count, 0))
    std[std < 1e-6] = 1.
    mean32, std32 = torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
    directory = Path(predictions_dir)
    index = json.loads((directory/'index.json').read_text())
    if [x['scene_id'] for x in index] != manifest['splits']['V_select']:
        raise ValueError('must audit complete frozen V')
    rows, all_predictions, all_targets, all_valid = [], [], [], []
    diffs = dict(model_vs_raw=[], model_vs_quantized=[], input_quantization=[])
    for item in index:
        with np.load(directory/item['path'], allow_pickle=False) as cache:
            xy, times, visible = cache['native_values'], cache['native_times'], cache['native_observed']
            query = cache['query_seconds']
            origin = query[0]
            packed = torch.tensor(((xy-mean)/std).astype(np.float32))
            restored = torch.where(torch.tensor(visible)[..., None], packed*std32+mean32, 0.)
            tau = torch.tensor(np.where(visible, (times-origin)/20., 0.), dtype=torch.float32)
            q = torch.tensor((query-origin)/20., dtype=torch.float32)
            encoded = dict(native_positions=restored, times=tau,
                observed=torch.tensor(visible), queries=q.expand(len(xy), -1),
                memory=torch.empty((len(xy), 0, 0)))
            with torch.inference_mode():
                prediction = CooperativeModel.position_base(SimpleNamespace(
                    config=SimpleNamespace(length=64, position_base_policy='native_ols')),
                    encoded, list(range(len(xy)))).numpy().astype(np.float64)
            # Same quantized input; this isolates arithmetic from normalization.
            quantized = predict_native_positions(restored.numpy(), tau.numpy(), visible,
                q.numpy())['predictions']['native_ols']
            raw = cache['native_ols']
            for name, first, second in (('model_vs_raw', prediction, raw),
                    ('model_vs_quantized', prediction, quantized),
                    ('input_quantization', quantized, raw)):
                difference = distances(first, second)
                diffs[name].append(difference.ravel())
            target, valid = cache['target'], cache['target_valid']
            region = np.ones(prediction.shape[:2], bool)
            rows.append(dict(scene_id=item['scene_id'], fragments=len(xy),
                model_position=position_metrics(prediction, target, valid, region),
                raw_position=position_metrics(raw, target, valid, region),
                max_model_vs_raw_metres=float(diffs['model_vs_raw'][-1].max())))
            all_predictions.append(prediction)
            all_targets.append(target)
            all_valid.append(valid)
    prediction, target, valid = map(np.concatenate, (all_predictions, all_targets, all_valid))
    differences = {}
    for name, values in diffs.items():
        values = np.concatenate(values)
        differences[name] = dict(mean_metres=float(values.mean()), max_metres=float(values.max()),
                                p95_metres=float(np.percentile(values, 95)))
    report = dict(status='COMPLETED', command=sys.argv, optimizer_updates=0, test_opened=False,
        reference_csv_read=False, train_scenes=len(manifest['splits']['T_core']),
        training_position_moments=dict(mean=mean.tolist(), std=std.tolist(), point_count=count),
        observation_members=observation_members, scenes=len(rows), fragments=len(prediction),
        queries=int(np.prod(prediction.shape[:2])), differences=differences,
        model_position=position_metrics(prediction, target, valid, np.ones(prediction.shape[:2], bool)),
        per_scene=rows, elapsed_seconds=time.monotonic()-started,
        limits='FP32 base only; no learned network/weights, speed or heading inference; '
               'mean/std are fitted on original T_core selected observations; V reused development data')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in
        ('status', 'training_position_moments', 'differences', 'model_position', 'elapsed_seconds')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('archive', 'manifest', 'predictions-dir', 'output'):
        parser.add_argument('--'+option, required=True)
    args = parser.parse_args()
    run(args.archive, args.manifest, args.predictions_dir, args.output)
