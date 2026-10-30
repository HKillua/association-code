"""Read-only107 TFM trace audit; no forward, reference, V arrays or new fit.

Writes only a new aggregate evidence file, with no normalizer coordinate means,
identities, pairing plan or weights. Training windows have changing batches.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch


def audit(tfm, output):
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    read = lambda p: json.loads(p.read_text())
    run, summary, history = (read(tfm/name) for name in ('run.json', 'summary.json', 'history.json'))
    payload = torch.load(tfm/'checkpoint.pt', map_location='cpu', weights_only=True)
    if run['status'] != 'COMPLETED' or len(history) != 1000 or [r['update'] for r in history] != list(range(1, 1001)):
        raise ValueError('original completed TFM1000 required')
    if history != payload['history'] or summary['normalizer'] != payload['normalizer']:
        raise ValueError('original checkpoint/journal binding differs')
    norm = payload['normalizer']
    fusion, objective = payload['states'][1:3]
    bound = all(torch.equal(fusion['position_'+key], torch.tensor(norm[key], dtype=torch.float32)[:2])
                for key in ('mean', 'std'))
    if not bound:
        raise ValueError('position affine differs from normalizer')
    sigma = objective['log_variance'].double().numpy()
    coefficients = .5*np.exp(-sigma)
    def stats(x):
        x = np.asarray(x, np.float64)
        return dict(mean=float(x.mean()), median=float(np.median(x)), p95=float(np.quantile(x, .95)), max=float(x.max()))
    windows = []
    for a, b in ((0, 100), (400, 500), (900, 1000)):
        rows = history[a:b]
        windows.append(dict(first_update=a+1, last_update=b, loss=stats([r['loss'] for r in rows]),
            gradient_norm=stats([r['gradient_norm'] for r in rows]),
            components={name:stats([r['fusion_component_losses'][name] for r in rows]) for name in ('lat', 'lon', 'vel', 'cou')},
            task_valid_counts=np.sum([r['fusion_valid_counts'] for r in rows], axis=0).tolist()))
    presented = [(item['scene_id'], tuple(pair)) for r in history for item in r['actual_samples'] for pair in item['pairs']]
    result = dict(status='COMPLETED', exit_code=0, source_tfm_run=str(tfm),
        generated_at=datetime.now(timezone.utc).isoformat(), scope=__doc__, updates=1000,
        pair_presentations=len(presented), unique_scene_pair_count=len(set(presented)),
        unique_sampled_training_scenes=len({scene for scene, _ in presented}),
        checkpoint_history_exact=True, normalizer_exact=True, position_affine_exact=True,
        speed_unit_contract=int(fusion['speed_unit_contract']),
        physical_units=norm.get('physical_units'),
        one_normalized_lat_unit_km=float(norm['std'][0]*np.pi/180.*6371.),
        one_normalized_lon_unit_equator_km=float(norm['std'][1]*np.pi/180.*6371.),
        endpoint_log_variance=sigma.tolist(), endpoint_loss_coefficients=coefficients.tolist(), windows=windows,
        optimizer_updates_performed=0, model_forwards=0, raw_reference_read=False, test_opened=False,
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        limitations=['Changing T batches: window trends are not a fixed training probe or causal test',
            'Global affine sensitivity is a structural scale, not proof of a bug or the cause of107/113 failure',
            'No T physical predictions: cannot separate T fit failure, domain shift, rollout or wrong grouping from this trace',
            'Learned coefficients weight different native-unit losses; not comparable sensor trust'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'updates', 'unique_sampled_training_scenes', 'elapsed_seconds')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tfm-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.tfm_run.resolve(), args.output.resolve())
