"""C117: compare the native Level1 reference fields on a frozen V roster."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import time

import numpy as np

from association.data.archive import open_mtad_zip
from association.data.physical_units import KNOT_TO_MPS
from association.data.reference_states import iter_reference_states
from association.evaluation.physical import EARTH_RADIUS_METRES
from tools.analysis.held_fusion_population import PARENTS, SPLIT
from tools.analysis.raw_reference_speed_origin import raw_track
from tools.analysis.same_source_swap_pilot import MAX_AS, peak_rss_bytes, read, write

ORDINARY = ('25', '38', '152', '203', '490', '524', '571', '600', '626', '670', '684', '765')
TAIL = ('1315', '2036', '2396', '3035', '3692', '4145', '4393', '4771')
BINS = ('up_to_600_seconds', 'over_600_seconds')
HYPOTHESES = ('left_mps', 'right_mps', 'mean_mps', 'left_kn', 'right_kn', 'mean_kn', 'vector_mean_mps')


def wrap_degrees(angle):
    return (angle + 180.) % 360. - 180.


def geometry(start, stop):
    lat_a, lon_a = np.deg2rad(start.T)
    lat_b, lon_b = np.deg2rad(stop.T)
    delta = lon_b - lon_a
    delta_lat = lat_b - lat_a
    sin_lat = np.sin(delta_lat / 2.)
    sin_lon = np.sin(delta / 2.)
    haversine = np.clip(sin_lat**2 + np.cos(lat_a) * np.cos(lat_b) * sin_lon**2, 0., 1.)
    length = 2. * EARTH_RADIUS_METRES * np.arcsin(np.sqrt(haversine))
    east = np.cos(lat_b) * np.sin(delta)
    north = np.cos(lat_a) * np.sin(lat_b) - np.sin(lat_a) * np.cos(lat_b) * np.cos(delta)
    arrival_east = np.cos(lat_a) * np.sin(delta)
    arrival_north = np.cos(lat_a) * np.sin(lat_b) * np.cos(delta) - np.sin(lat_a) * np.cos(lat_b)
    dot = np.sin(lat_a) * np.sin(lat_b) + np.cos(lat_a) * np.cos(lat_b) * np.cos(delta)
    independent = EARTH_RADIUS_METRES * np.arctan2(np.hypot(east, north), dot)
    np.testing.assert_allclose(length, independent, atol=1e-5, rtol=1e-11, equal_nan=True)
    return length, np.rad2deg(np.arctan2(east, north)), np.rad2deg(np.arctan2(arrival_east, arrival_north))


def audit_track(rows, residuals):
    times, values, raw_counts, _ = raw_track(rows)
    dt = np.diff(times)
    distance, initial, arrival = geometry(values[:-1, :2], values[1:, :2])
    rate = distance / dt
    left, right = values[:-1, 2], values[1:, 2]
    mean = (left + right) / 2.
    left_angle = wrap_degrees(values[:-1, 3] - initial)
    right_angle = wrap_degrees(values[1:, 3] - arrival)
    left_radians, right_radians = np.deg2rad(left_angle), np.deg2rad(right_angle)
    along = (left * np.cos(left_radians) + right * np.cos(right_radians)) / 2.
    across = (left * np.sin(left_radians) + right * np.sin(right_radians)) / 2.
    vector_error = np.hypot(rate - along, across)
    available = np.isfinite(rate) & np.isfinite(left) & np.isfinite(right)
    error = [abs(rate - candidate) for candidate in (left, right, mean, left * KNOT_TO_MPS,
                                                      right * KNOT_TO_MPS, mean * KNOT_TO_MPS)]
    error.append(vector_error)
    both_mps = rate - np.maximum(left, right)
    both_kn = rate - np.maximum(left, right) * KNOT_TO_MPS
    audit = {}
    for name, population in zip(BINS, (dt <= 600., dt > 600.), strict=True):
        comparable = population & available
        moving = comparable & (distance >= 1.)
        vector_valid = moving & np.isfinite(vector_error)
        left_valid, right_valid = moving & np.isfinite(left_angle), moving & np.isfinite(right_angle)
        both_course = left_valid & right_valid
        counts = dict(
            intervals=int(population.sum()), comparable=int(comparable.sum()), moving=int(moving.sum()),
            vector_valid=int(vector_valid.sum()), both_course=int(both_course.sum()),
            above_gt10=int((comparable & (both_mps > 10.)).sum()),
            above_gt10_if_kn=int((comparable & (both_kn > 10.)).sum()),
            below_gt10=int((comparable & (np.minimum(left, right) - rate > 10.)).sum()),
            left_joint=int((both_course & (error[0] <= .01 * rate) & (abs(left_angle) <= 1.)).sum()),
            right_joint=int((both_course & (error[1] <= .01 * rate) & (abs(right_angle) <= 1.)).sum()),
            left_joint_if_kn=int((both_course & (error[3] <= .01 * rate) & (abs(left_angle) <= 1.)).sum()),
            right_joint_if_kn=int((both_course & (error[4] <= .01 * rate) & (abs(right_angle) <= 1.)).sum()),
        )
        for candidate, array in zip(HYPOTHESES, error, strict=True):
            cohort = vector_valid if candidate == 'vector_mean_mps' else moving
            residuals[name][candidate].append(array[cohort])
        audit[name] = counts

    hot = available & (dt <= 600.) & (both_mps > 10.)
    pair = hot[:-1] & hot[1:]
    angle = abs(wrap_degrees(initial[1:] - arrival[:-1]))
    double_distance = geometry(values[:-2, :2], values[2:, :2])[0]
    path = distance[:-1] + distance[1:]
    ratio = np.full_like(path, np.nan)
    np.divide(double_distance, path, out=ratio, where=pair)
    reversal = pair & (angle >= 120.) & (ratio <= .5)
    co_direction = pair & (angle <= 60.)
    audit['adjacent'] = dict(pairs=int(pair.sum()), reversal=int(reversal.sum()),
                             same_direction=int(co_direction.sum()),
                             other=int((pair & ~reversal & ~co_direction).sum()))
    audit['raw'] = {key: int(value) for key, value in raw_counts.items()}
    audit['geometry_formula_checks'] = int(np.isfinite(distance).sum() + np.isfinite(double_distance).sum())
    examples = {}
    for label, mask in [('most_excess', hot), ('reversal', reversal), ('same_direction', co_direction)]:
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        index = int(indices[np.argmax(both_mps[indices])])
        length = 1 if label == 'most_excess' else 2
        slc = slice(index, index + length)
        examples[label] = dict(
            ordinal=index + 1, dt=dt[slc].tolist(), displacement_m=distance[slc].tolist(),
            geometry_mps=rate[slc].tolist(), raw_fields=values[index:index + length + 1, 2].tolist(),
            relative_turn_degrees=None if length == 1 else float(angle[index]),
            net_over_path=None if length == 1 else float(ratio[index]),
            signed_gap_mps=both_mps[slc].tolist(),
        )
    return audit, examples


def describe(segments):
    values = np.concatenate(segments)
    if not len(values):
        return dict(n=0, mae=None, p50=None, p95=None)
    return dict(n=len(values), mae=float(values.mean()), p50=float(np.median(values)),
                p95=float(np.quantile(values, .95)))


def scene_diagnostics(scene, cohort_residuals):
    groups = defaultdict(list)
    invalid_identity = 0
    for row in scene['rows']:
        parts = row['identity_id'].split('-')
        if len(parts) == 2 and all(parts):
            groups[row['identity_id']].append(row)
        else:
            invalid_identity += 1
    by_track = []
    summary = defaultdict(Counter)
    examples = {}
    residuals = {name: defaultdict(list) for name in BINS}
    for index, identity in enumerate(sorted(groups), start=1):
        row, samples = audit_track(groups[identity], residuals)
        for name in (*BINS, 'adjacent', 'raw'):
            summary[name].update(row[name])
        by_track.append(dict(track=index, **row))
        for name, sample in samples.items():
            current = examples.get(name)
            if current is None or max(sample['signed_gap_mps']) > max(current['signed_gap_mps']):
                examples[name] = dict(track=index, **sample)
    scene_residuals = {}
    for name in BINS:
        scene_residuals[name] = {}
        for candidate in HYPOTHESES:
            segments = residuals[name][candidate] or [np.empty(0)]
            scene_residuals[name][candidate] = describe(segments)
            cohort_residuals[name][candidate].extend(segments)
    for name in BINS:
        assert summary[name]['intervals'] >= summary[name]['comparable'] >= summary[name]['moving']
        assert summary[name]['both_course'] <= summary[name]['vector_valid']
    return dict(scene=scene['scene_id'], rows=len(scene['rows']), invalid_identity_rows=invalid_identity,
                reader_issues=dict(Counter(row['code'] for row in scene['issues'])), tracks=len(groups),
                counts={key: dict(values) for key, values in summary.items()}, residuals=scene_residuals,
                examples=examples, by_track=by_track)


def run(runs, output):
    start = time.monotonic()
    output.mkdir(exist_ok=False)
    record = dict(idea='IDEA-C117-REFERENCE-INTERVAL-FINGERPRINT-20261112', plan='1.0', status='RUNNING',
                  start=datetime.now(timezone.utc).isoformat(), affinity=sorted(os.sched_getaffinity(0)),
                  cuda=os.environ.get('CUDA_VISIBLE_DEVICES'), test_opened=False, raw_measurement_members_read=0,
                  raw_label_members_read=0, updates=0, predictions=0, raw_reference_members_read=0)
    write(output/'run.json', record)
    try:
        assert record['affinity'] == [27] and record['cuda'] == ''
        parent = read(runs/PARENTS['full_data']/'run.json')
        assert parent['status'] == 'COMPLETED' and parent['exit_code'] == 0
        assert parent['reference_speed_unit'] == 'mps' and parent['measurement_speed_unit'] == 'kn'
        manifest = read(Path(parent['manifest']))
        archive = parent['archive']
        v, components = manifest['splits'][SPLIT], manifest['components']
        seen = {components[scene] for scene in TAIL}
        selected = []
        for scene in v:
            if components[scene] not in seen:
                selected.append(scene)
                seen.add(components[scene])
            if len(selected) == len(ORDINARY):
                break
        assert tuple(selected) == ORDINARY and set(TAIL).issubset(v)
        assert not set(ORDINARY + TAIL).intersection(manifest['splits']['T_test'])
        with open_mtad_zip(archive) as source:
            member_suffixes = Counter(Path(entry.filename).suffix.lower() for entry in source.infolist()
                                      if not entry.is_dir())
        write(output/'scope.json', dict(ordinary=ORDINARY, tail_context=TAIL,
                                       ordinary_components=len({components[s] for s in ORDINARY}),
                                       tail_components=len({components[s] for s in TAIL}),
                                       level1_archive_suffix_counts=dict(member_suffixes)))
        cohorts = {'ordinary': ORDINARY, 'tail_context': TAIL}
        for cohort, scenes in cohorts.items():
            evidence = []
            residuals = {name: defaultdict(list) for name in BINS}
            for raw in iter_reference_states(archive, manifest, split=SPLIT, scene_ids=scenes):
                record['raw_reference_members_read'] += 1
                evidence.append(scene_diagnostics(raw, residuals))
            assert tuple(row['scene'] for row in evidence) == scenes
            totals = defaultdict(Counter)
            for row in evidence:
                for key, values in row['counts'].items():
                    totals[key].update(values)
            cohort_summary = dict(
                scenes=len(evidence), tracks=sum(row['tracks'] for row in evidence),
                rows=sum(row['rows'] for row in evidence),
                invalid_identity_rows=sum(row['invalid_identity_rows'] for row in evidence),
                counts={name: dict(value) for name, value in totals.items()},
                residuals={name: {hypothesis: describe(arrays) for hypothesis, arrays in by_hypothesis.items()}
                           for name, by_hypothesis in residuals.items()},
                geometry_formula_checks=sum(track['geometry_formula_checks'] for row in evidence for track in row['by_track']),
            )
            write(output/f'{cohort}-by-scene.json.gz', evidence)
            write(output/f'{cohort}.json', cohort_summary)
        assert record['raw_reference_members_read'] == 20
        record.update(status='COMPLETED', exit_code=0)
    except BaseException as error:
        record.update(status='FAILED', exit_code=1, error_type=type(error).__name__)
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss_bytes(),
                      finished_at=datetime.now(timezone.utc).isoformat())
        write(output/'run.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
    run(args.runs, args.output)
