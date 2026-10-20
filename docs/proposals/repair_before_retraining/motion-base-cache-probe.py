"""Exploratory V-only motion probe of cached position bases. Never train or read a ZIP."""
from pathlib import Path
import argparse
import io
import json
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'HandOver/results/COOPERATIVE-NOBCE-20261020-041/completion-predictions.tar.gz'
R = 6371000.

def circular(x):
    return (x + 180.) % 360. - 180.

def distance(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    la, lb = np.deg2rad(a[..., 0]), np.deg2rad(b[..., 0])
    dl = np.deg2rad(circular(a[..., 1] - b[..., 1]))
    h = np.sin((la-lb)/2)**2 + np.cos(la)*np.cos(lb)*np.sin(dl/2)**2
    return R * 2 * np.arctan2(np.sqrt(np.maximum(h, 0)), np.sqrt(np.maximum(1-h, 0)))

def predict(base, times, inside, before, after, nearest):
    # All arguments are input-derived; targets and target-valid mask are deliberately absent.
    result = base.astype(np.float64).copy()
    fitted = np.zeros(len(base), dtype=bool)
    for i in range(len(base)):
        idx = np.flatnonzero(inside[i])
        if len(idx) < 2:
            continue
        t = times[idx]
        dt = t - t.mean()
        xy = base[i, idx].astype(np.float64).copy()
        xy[:, 1] = np.rad2deg(np.unwrap(np.deg2rad(xy[:, 1])))
        velocity = (dt[:, None] * (xy - xy.mean(0))).sum(0) / (dt @ dt)
        result[i, before[i]] -= nearest[i, before[i], None] * velocity
        result[i, after[i]] += nearest[i, after[i], None] * velocity
        result[i, :, 0] = np.clip(result[i, :, 0], -90., 90.)
        result[i, :, 1] = circular(result[i, :, 1])
        fitted[i] = True
    return result, fitted

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    with tarfile.open(args.predictions) as stream:
        for member in stream.getmembers():
            if member.name.endswith('.npz'):
                with np.load(io.BytesIO(stream.extractfile(member).read()), allow_pickle=False) as loaded:
                    rows.append({key: loaded[key].copy() for key in loaded.files})
    errors = [[], [], []]
    valid, inside, sources, fitted_all, scene = [], [], [], [], []
    for row in rows:
        base = row['position_base'][..., :2].astype(np.float64)
        result, fitted = predict(base, row['query_seconds'], row['region_inside_support'],
                                 row['region_before_support'], row['region_after_support'],
                                 row['nearest_seconds'])
        # No inside-support change, and every under-supported fragment must fall back exactly.
        assert np.array_equal(result[row['region_inside_support']], base[row['region_inside_support']])
        assert np.array_equal(result[~fitted], base[~fitted])
        mask = row['target_valid'][..., :2].all(-1)
        es = [distance(x, row['target'][..., :2]) for x in (base, result, row['prediction'][..., :2])]
        for collection, error in zip(errors, es):
            collection.append(error)
        valid.append(mask)
        inside.append(row['region_inside_support'])
        sources.append(row['sources'])
        fitted_all.append(fitted)
        scene.append(dict(id=str(row['scene_id']), base=float(es[0][mask].mean()),
                          motion=float(es[1][mask].mean()), **{'041': float(es[2][mask].mean())},
                          fit_tracks=int(fitted.sum()), tracks=len(base)))
    base_err, motion_err, pred_err = map(np.concatenate, errors)
    valid, inside, sources, fitted = map(np.concatenate, (valid, inside, sources, fitted_all))
    report = dict(scope='Exploratory control on exposed V, cached Q interpolation, not native robust fit or independent validation. No inference or archive scan. No target or target mask is passed to predict().',
                  method='OLS latitude and unwrapped longitude against cached inside-support query times, >=2 Q points; extend existing native endpoint base using slope * signed nearest_seconds; <2 Q points keep hold. No tuned parameters. Full-fragment information is offline.',
                  source=str(args.predictions), population=dict(scenes=len(rows), queries=int(valid.size), points=int(valid.sum()), tracks=len(sources), fit_tracks=int(fitted.sum()), fallback_tracks=int((~fitted).sum())),
                  scene_better_worse=[sum(x['motion'] < x['base'] for x in scene), sum(x['motion'] > x['base'] for x in scene)], metrics={}, per_scene=scene)
    for name, mask in [('all', valid), ('inside', valid & inside), ('outside', valid & ~inside),
                       ('9001', valid & (sources == '9001')[:, None]), ('9002', valid & (sources == '9002')[:, None])]:
        report['metrics'][name] = dict(n=int(mask.sum()), base_mae=float(base_err[mask].mean()),
                                      motion_mae=float(motion_err[mask].mean()), **{'041_mae': float(pred_err[mask].mean())},
                                      base_rmse=float(np.sqrt((base_err[mask]**2).mean())),
                                      motion_rmse=float(np.sqrt((motion_err[mask]**2).mean())))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'per_scene'}, indent=2))

if __name__ == '__main__':
    main()
