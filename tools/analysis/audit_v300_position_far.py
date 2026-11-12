"""Independently recount C94 using manual first-anchor correspondence and errors."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import numpy as np


ARMS = ('B', 'FAR', 'T128_B', 'T128_FAR')
ATTRIBUTES = {'longitude': (1, 10000.), 'latitude': (0, 10000.),
              'speed': (2, 3600./1852.), 'heading': (3, 1.)}


def stats(values):
    absolute = np.abs(values)
    return dict(count=len(values), mae=float(np.mean(absolute)), rmse=float(np.sqrt(np.mean(values**2))),
                p95=float(np.percentile(absolute, 95)), p99=float(np.percentile(absolute, 99)),
                bias=float(np.mean(values)))


def audit(runs, output):
    output.parent.mkdir(exist_ok=False)
    started = time.monotonic()
    root = runs/'LEVEL1-V300-POSITION-FAR-20261112-attempt-1'
    summary = json.loads((root/'summary.json').read_text())
    record = json.loads((root/'run.json').read_text())
    assert record['status'] == 'COMPLETED' and record['exit_code'] == 0
    full = runs/'LEVEL1-FUSION-PEER-FULL-DATA-20261030-151-attempt-1'
    manifest = json.loads(Path(json.loads((full/'run.json').read_text())['manifest']).read_text())
    ids = record['scene_ids']
    assert ids == manifest['splits']['V_select']
    vectors, papers, counts, exact, numerical = [], [], [], 0, 0
    for sid in ids:
        with np.load(root/'private'/f'{sid}.npz', allow_pickle=False) as saved, np.load(
                full/'private/evaluation/V_select'/f'scene-{sid}.npz', allow_pickle=False) as truth:
            for original, frozen in (('node_ids', 'nodes'), ('query_seconds', 'query')):
                np.testing.assert_array_equal(truth[original], saved[frozen])
            targets = truth['target_ids'].tolist()
            lookup = {key: i for i, key in enumerate(targets)}
            first, groups = {}, []
            for gid in np.unique(saved['membership']):
                members = np.flatnonzero(saved['membership'] == gid)
                groups.append(members)
                target = lookup[truth['identities'][members[0]]]
                if target not in first:
                    first[target] = int(gid)
            matched = np.asarray(sorted(first))
            selected = np.asarray([first[i] for i in matched])
            y, valid = truth['target_states'][matched], truth['target_valid'][matched]
            scene, paper = {}, {}
            for arm in ARMS:
                pred = saved[arm][selected]
                scene[arm] = {}
                for name, (column, _) in ATTRIBUTES.items():
                    error = pred[..., column]-y[..., column]
                    if name in ('longitude', 'latitude'):
                        error = error*10000.
                    if name == 'heading':
                        error = (error+180.) % 360.-180.
                    scene[arm][name] = error[valid[..., column]]
                mask = valid[..., :2].all(axis=-1)
                a, b = np.deg2rad(pred[..., :2][mask]), np.deg2rad(y[..., :2][mask])
                d = b-a
                h = np.clip(np.sin(d[:, 0]/2.)**2+np.cos(a[:, 0])*np.cos(b[:, 0])*np.sin(d[:, 1]/2.)**2,
                            0., 1.)
                scene[arm]['position'] = 2.*6371000.*np.arcsin(np.sqrt(h))
                paper[arm] = {}
                for name, (column, scale) in ATTRIBUTES.items():
                    errors = [(y[i, valid[i, :, column], column]-pred[i, valid[i, :, column], column])*scale
                              for i in range(len(matched)) if valid[i, :, column].any()]
                    paper[arm][name] = dict(
                        ME=float(np.mean([e.mean() for e in errors])),
                        MAE=float(np.mean([np.abs(e).mean() for e in errors])),
                        residual_SD=float(np.mean([e.std(ddof=1) for e in errors if len(e) >= 2])))
            for field in ('longitude', 'latitude', 'position', 'heading'):
                np.testing.assert_array_equal(scene['B'][field], scene['FAR'][field])
                np.testing.assert_array_equal(scene['T128_B'][field], scene['T128_FAR'][field])
                exact += 2
            for a, b in (('B', 'T128_B'), ('FAR', 'T128_FAR')):
                np.testing.assert_array_equal(scene[a]['speed'], scene[b]['speed'])
                exact += 1
            vectors.append(scene)
            papers.append(paper)
            counts.append(dict(target_identity_count=len(targets), predicted_group_count=len(groups),
                duplicate_prediction_group_count=len(groups)-len(first), anchor_matched_target_count=len(first),
                reference_unavailable_target_count=int((~truth['target_valid'].any(axis=(1, 2))).sum())))

    scopes = {'all_scenes': (list(range(len(ids))), summary['all_scenes']),
              'without_largest': ([i for i, sid in enumerate(ids)
                                  if manifest['components'][sid] != summary['without_largest']['component']],
                                  summary['without_largest']['result'])}
    for component, value in summary['by_component'].items():
        scopes[component] = ([i for i, sid in enumerate(ids) if manifest['components'][sid] == component], value)
    for _, (indices, expected) in scopes.items():
        actual_counts = {key: sum(counts[i][key] for i in indices) for key in counts[0]}
        assert actual_counts == expected['population']
        exact += len(actual_counts)
        for arm in ARMS:
            for name in vectors[0][arm]:
                actual = stats(np.concatenate([vectors[i][arm][name] for i in indices]))
                for key, value in actual.items():
                    np.testing.assert_allclose(value, expected['point'][arm][name][key], atol=1e-7, rtol=1e-11)
                    numerical += 1
            for name in ATTRIBUTES:
                for metric in ('ME', 'MAE', 'residual_SD'):
                    value = np.mean([papers[i][arm][name][metric] for i in indices])
                    np.testing.assert_allclose(value, expected['paper'][arm]['attributes'][name]
                                               ['scene_equal_target_mean'][metric], atol=1e-7, rtol=1e-10)
                    numerical += 1
    result = dict(status='COMPLETED', exit_code=0, scenes=len(ids), scopes=len(scopes),
                  targets=sum(row['target_identity_count'] for row in counts),
                  exact_comparisons=exact, numerical_comparisons=numerical,
                  mapping='independent first group per anchor identity',
                  geometry='independent spherical arcsin haversine, R=6371000m',
                  metrics='point counts/MAE/RMSE/P95/P99/bias and target-then-scene ME/MAE/sample SD',
                  optimizer_updates=0, new_predictions=0, raw_archive_reads=0,
                  elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  finished_at=datetime.now(timezone.utc).isoformat())
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.runs.resolve(), args.output.resolve())
