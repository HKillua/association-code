"""C128: frozen two-vessel geometry checks on Autoferry's public detections."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np


SCENES = (2, 3, 4, 5, 6, 13, 16, 17, 22)
GATES = {sid: 20.0 if sid <= 6 else 10.0 for sid in SCENES}
REF_AGE_S, SCAN_GAP_S, CLOSE_M = 0.2, 0.5, 50.0
SENSORS = {1: 'lidar', 2: 'radar'}


def read_file(path):
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def positions(row):
    raw = np.asarray(row['measurement'], float)
    if raw.shape == (2,):
        raw = raw[:, None]
    if raw.size == 0:
        return np.empty((0, 2), float)
    if raw.ndim != 2 or raw.shape[0] != 2:
        raise ValueError('active sensor measurement must be 2xM')
    result = raw.T + np.asarray(row['ownshipPosition'], float)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite active sensor detection')
    return result


def nearest(times, time):
    right = np.searchsorted(times, time)
    candidates = sorted(set((max(0, right-1), min(len(times)-1, right))))
    return min(candidates, key=lambda i: abs(times[i]-time))


def assign(first, candidates):
    """Minimize the non-squared distance sum with distinct detection indices."""
    if len(candidates) < 2:
        return None
    dist = np.linalg.norm(first[:, None]-candidates[None], axis=-1)
    cost = dist[0, :, None]+dist[1, None, :]
    np.fill_diagonal(cost, np.inf)
    return np.unravel_index(np.argmin(cost), cost.shape)


def freeze(detections):
    scans = {sid: [] for sid in SENSORS}
    for i, row in enumerate(detections):
        sid = row['sensorID']
        if sid in scans:
            scans[sid].append((i, float(row['time']), positions(row)))
    for s in scans.values():
        s.sort(key=lambda row: (row[1], row[0]))
    lidars = [row for row in scans[1] if len(row[2]) == 2]
    radars = [row for row in scans[2] if len(row[2]) >= 2]
    lt, rt = ([row[1] for row in items] for items in (lidars, radars))
    frozen = []
    for li, (lidx, ltime, lidar) in enumerate(lidars):
        if not rt:
            break
        ri = nearest(rt, ltime)
        ridx, rtime, radar = radars[ri]
        if abs(ltime-rtime) <= SCAN_GAP_S and nearest(lt, rtime) == li:
            chosen = list(assign(lidar, radar))
            frozen.append((lidx, ridx, lidar, radar[chosen]))
    return scans, frozen


def reference(points, time):
    if len(points) != 2 or {p['targetID'] for p in points} != {1, 2}:
        return None
    ordered = sorted(points, key=lambda p: p['targetID'])
    if max(abs(p['time']-time) for p in ordered) > REF_AGE_S:
        return None
    target = np.array([p['position'][:2] for p in ordered])
    return target if np.isfinite(target).all() else None


def coverage(scans, gt, gate):
    totals = {}
    for code, rows in scans.items():
        count = Counter(events=len(rows))
        for index, time, points in rows:
            ref = reference(gt[index], time)
            if ref is None:
                continue
            close = bool(np.linalg.norm(ref[1]-ref[0]) <= CLOSE_M)
            count['ref_valid'] += 1
            count['close_ref_valid'] += close
            selected = assign(ref, points)
            if selected is not None and (np.linalg.norm(ref-points[list(selected)], axis=1) <= gate).all():
                count['both_detected'] += 1
                count['close_both_detected'] += close
        totals[SENSORS[code]] = dict(count)
    return totals


def error(pred, ref):
    normal = np.linalg.norm(pred-ref, axis=1)
    margin = np.linalg.norm(pred[::-1]-ref, axis=1).sum() - normal.sum()
    delta, truth = pred[1]-pred[0], ref[1]-ref[0]
    gap = np.linalg.norm(delta-truth)
    length = np.linalg.norm(delta)*np.linalg.norm(truth)
    angle = float(np.rad2deg(np.arccos(np.clip(delta @ truth/length, -1., 1.)))) if length else None
    return dict(distance_m=normal.tolist(), margin_m=float(margin), vector_m=float(gap), angle_deg=angle)


def evaluate(pairs, detections, gt, gate):
    counts = Counter(paired=len(pairs))
    cases = []
    for li, ri, lidar, radar in pairs:
        lt, rt = (detections[i]['time'] for i in (li, ri))
        lref, rref = (reference(gt[i], t) for i, t in ((li, lt), (ri, rt)))
        if lref is None or rref is None:
            continue
        close = bool(np.linalg.norm(lref[1]-lref[0]) <= CLOSE_M)
        counts['both_ref_valid'] += 1
        counts['close_both_ref_valid'] += close
        order = list(assign(lref, lidar))
        lidar, radar = lidar[order], radar[order]
        if (np.linalg.norm(lidar-lref, axis=1) > gate).any() or error(lidar, lref)['margin_m'] <= 1e-9:
            continue
        counts['lidar_identity_resolved'] += 1
        counts['close_lidar_identity_resolved'] += close
        r_direct = (np.linalg.norm(radar-rref, axis=1) <= gate).all()
        r_swap = (np.linalg.norm(radar[::-1]-rref, axis=1) <= gate).all()
        r_margin = error(radar, rref)['margin_m']
        if r_direct and r_margin > 0:
            radar_status = 'correct'
        elif r_swap and r_margin < 0:
            radar_status = 'swapped'
        elif r_direct or r_swap:
            radar_status = 'ambiguous'
        else:
            radar_status = 'outside'
        cases.append(dict(close=close, lidar_index=li, radar_index=ri, scan_gap_s=abs(lt-rt),
            radar_status=radar_status, lidar=error(lidar, lref), radar=error(radar, lref),
            mean=error((lidar+radar)/2., lref)))
    return dict(counts), cases


def describe(values):
    return dict(n=len(values), mae=float(np.mean(values)) if values else None,
                p99=float(np.quantile(values, .99)) if values else None)


def summary(rows):
    out = dict(pairs=len(rows), radar_at_own_time=dict(Counter(r['radar_status'] for r in rows)))
    for arm in ('lidar', 'radar', 'mean'):
        values = [r[arm] for r in rows]
        out[arm] = dict(position=describe([e for v in values for e in v['distance_m']]),
                        relative=describe([v['vector_m'] for v in values]),
                        angle=describe([v['angle_deg'] for v in values if v['angle_deg'] is not None]),
                        correct=sum(v['margin_m'] > 1e-9 for v in values),
                        swapped=sum(v['margin_m'] < -1e-9 for v in values))
    return out


def run(args):
    cases, frozen, paths = {}, {}, {}
    for scene in SCENES:
        path = args.input/f'scenario{scene}'
        detections, sha = read_file(path/f'scenario{scene}_detections.json')
        scans, pairs = freeze(detections)
        frozen[scene] = (detections, scans, pairs)
        paths[scene] = dict(detections=sha)
    write(args.output/'predictions-frozen.json',
          dict(scenes=list(SCENES), raw_detection_sha256=paths,
               sensor_pair_indices={s: [[a, b] for a, b, _, _ in f[2]] for s, f in frozen.items()}))
    stats = {}
    for scene, (detections, scans, pairs) in frozen.items():
        ref, sha = read_file(args.input/f'scenario{scene}'/f'scenario{scene}_groundTruth.json')
        paths[scene]['reference'] = sha
        if len(ref) != len(detections):
            raise ValueError(f'publication row alignment differs in scene {scene}')
        count, rows = evaluate(pairs, detections, ref, GATES[scene])
        cases[scene] = rows
        stats[scene] = dict(coverage=coverage(scans, ref, GATES[scene]), pairing=count,
                            all=summary(rows), close=summary([r for r in rows if r['close']]))
    flat = [r for scene in SCENES for r in cases[scene]]
    close = [r for r in flat if r['close']]
    qualified = sum(any(r['close'] and r['radar_status'] == 'correct' for r in rows) for rows in cases.values())
    write(args.output/'summary.json', dict(input_sha256=paths, scenes=stats, all=summary(flat),
        close=summary(close), close_both_correct_scenes=qualified, qualification_passed=qualified >= 2,
        paired_scans=sum(s['pairing']['paired'] for s in stats.values())))
    write(args.output/'scored-pairs.json', cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args)
    except Exception as error:
        write(args.output/'run.json', dict(status='FAILED', error=repr(error)))
        raise
    write(args.output/'run.json', dict(status='COMPLETED', seconds=time.monotonic()-start,
                                      train=0, model_forwards=0, mtad_opened=False, test_opened=False))


if __name__ == '__main__':
    main()
