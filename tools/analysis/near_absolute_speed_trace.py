"""C97: one-scene read-only lineage of the frozen C94 worst near speed error."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from association.data.archive import MTADArchive
from association.data.physical_units import KNOT_TO_MPS
from tools.analysis.held_fusion_population import load_scene, read_evaluation, validate_measurement_archive
from tools.analysis.offline_endpoint_lineage import endpoint_arrays, speed_between
from tools.analysis.same_source_swap_pilot import group_members, read

C83 = 'LEVEL1-HELD-SPEED-READOUT-20261112-attempt-1'
C94 = 'LEVEL1-V300-POSITION-FAR-20261112-attempt-1'
SID, GROUP, QUERY = '2665', 2, 62


def interval_rows(scene, node, side, query):
    """Release only relative times, speeds and displacement, never coordinates."""
    _, times, _, offsets = scene['native']
    lo, hi = map(int, offsets[node:node+2])
    first, stop = (lo, min(hi, lo+5)) if side == 0 else (max(lo, hi-5), hi)
    indices = np.arange(first, stop)
    values = scene['values']
    rows = []
    for index in indices:
        row = dict(seconds_from_query=float(times[index]-query),
                   raw_speed_kn=float(values[index, 2]),
                   field_mps=float(values[index, 2]*KNOT_TO_MPS))
        if index > first:
            elapsed = float(times[index]-times[index-1])
            speed = float(speed_between(values[index-1, :2], values[index, :2], np.asarray(elapsed)))
            row.update(previous_interval_seconds=elapsed, previous_displacement_metres=speed*elapsed,
                       previous_secant_mps=speed)
        if index+1 < stop:
            elapsed = float(times[index+1]-times[index])
            speed = float(speed_between(values[index, :2], values[index+1, :2], np.asarray(elapsed)))
            row.update(next_interval_seconds=elapsed, next_displacement_metres=speed*elapsed,
                       next_secant_mps=speed)
        rows.append(row)
    return rows


def audit(runs, lineage_helper):
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location('c97_frozen_lineage', lineage_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    result = dict(diagnostic_id='DIAGNOSTIC-C97-NEAR-ABSOLUTE-SPEED-20261112-R1', plan_version='1.1',
                  status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
                  lineage_helper=str(lineage_helper),
                  source_runs=[str(runs/C83), str(runs/C94)], scene=SID, group=GROUP, query_index=QUERY,
                  affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  optimizer_updates=0, new_prediction_arms=0, test_opened=False)
    if result['affinity'] != [27] or result['cuda_visible_devices'] != '':
        raise ValueError('fixed CPU27 and empty CUDA required')
    for name in (C83, C94):
        record = read(runs/name/'run.json')
        assert record['status'] == 'COMPLETED' and record['exit_code'] == 0 and SID in record['scene_ids']
    old = read(runs/C83/'run.json')
    expected = read(runs/C94/'summary.json')['near']['production_worst']
    assert (expected['scene'], expected['group'], expected['query_index']) == (SID, GROUP, QUERY)
    points = helper.near_points(runs, runs/C83/'private', [SID])
    chosen = [point for point in points if (point['group'], point['query_index']) == (GROUP, QUERY)]
    assert len(chosen) == 1
    point = chosen[0]
    for key, value in (('reference', expected['reference']), ('base', expected['speed']['B']),
                       ('candidate', expected['near_control_speed']), ('error_change', expected['near_control_error_change'])):
        np.testing.assert_equal(point[key], value)
    with MTADArchive(old['archive_paths']) as archive:
        validate_measurement_archive(runs, archive)
        scene = load_scene(runs, SID, archive)
        with np.load(runs/C83/'private'/f'{SID}.npz', allow_pickle=False) as saved, np.load(
                runs/C94/'private'/f'{SID}.npz', allow_pickle=False) as production:
            artifact = {key: saved[key] for key in saved.files}
            for key in ('nodes', 'query', 'membership', 'support', 'nearest', 'B', 'FAR', 'far'):
                np.testing.assert_array_equal(artifact[key], production[key])
            assert artifact['near'][GROUP, QUERY] and not artifact['far'][GROUP, QUERY]
            for arm in ('B', 'FAR', 'T128_B', 'T128_FAR'):
                np.testing.assert_equal(production[arm][GROUP, QUERY, 2], expected['speed'][arm])
            ev = read_evaluation(runs, SID, artifact['nodes'], artifact['query'])
            detail = helper.describe_scene(scene, artifact, ev, chosen)[0]
        assert [(r['scene_id'], r['kind']) for r in archive.members_read] == [(SID, 'measurements')]
        result.update(raw_measurement_members_read=1, raw_label_members_read=0, raw_reference_members_read=0)
    query, times, _, offsets = scene['native']
    groups = group_members(scene['membership'])
    members = groups[GROUP]
    target = point['target']
    # Verify that this group is the original first claimant, with no preceding duplicate claimant.
    assert all(ev['identities'][group[0]] != ev['identities'][members[0]] for group in groups[:GROUP])
    endpoints = endpoint_arrays(times, scene['values'], offsets)
    side = 0 if detail['side'] == 'before_group' else 1
    detail['native_endpoint_neighborhoods'] = {}
    for label in ('first_member', 'selected_member'):
        node = int(members[detail[label]['member_slot']])
        detail['native_endpoint_neighborhoods'][label] = interval_rows(scene, node, side, query[QUERY])
    member_rows = []
    for slot, node in enumerate(members):
        member_rows.append(dict(member_slot=slot,
            source=f"S{np.searchsorted(np.unique(scene['sources']), scene['sources'][node])+1}",
            same_identity_as_anchor=bool(ev['identities'][node] == ev['identities'][members[0]]),
            native_events=int(offsets[node+1]-offsets[node]),
            start_seconds_from_query=float(endpoints['time'][node, 0]-query[QUERY]),
            end_seconds_from_query=float(endpoints['time'][node, 1]-query[QUERY]),
            first_field_mps=float(endpoints['field'][node, 0]), last_field_mps=float(endpoints['field'][node, 1]),
            first_secant_mps=float(endpoints['secant'][node, 0]), last_secant_mps=float(endpoints['secant'][node, 1])))
    detail['all_group_member_endpoints'] = member_rows
    detail['target_reference_q60_q63'] = [dict(query_index=q, seconds_from_Q62=float(query[q]-query[QUERY]),
        valid=bool(ev['target_valid'][target, q, 2]),
        speed_mps=float(ev['target_states'][target, q, 2]) if ev['target_valid'][target, q, 2] else None)
        for q in range(60, 64)]
    detail['reference_adjacent_query_secants'] = []
    for left in (60, 61, 62):
        if ev['target_valid'][target, left:left+2, :2].all():
            elapsed = float(query[left+1]-query[left])
            speed = float(speed_between(ev['target_states'][target, left, :2],
                                       ev['target_states'][target, left+1, :2], np.asarray(elapsed)))
            detail['reference_adjacent_query_secants'].append(dict(left_query=left, right_query=left+1,
                seconds=elapsed, position_secant_mps=speed, displacement_metres=speed*elapsed))
    result.update(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        frozen_C94_production=expected, lineage=detail,
        checks=dict(C83_C94_full_scene_group_query_support_B_FAR_exact=True, C94_four_arm_Q62_speed_exact=True,
                    C94_published_worst_exact=True, original_first_claimant_exact=True,
                    native_measurement_cache_unit_rebuild_exact=True, endpoint_lineage_exact=True),
        unit_policy=dict(measurement_speed='kn', state_and_cached_reference_speed='m/s',
                         knot_to_mps=KNOT_TO_MPS, seconds='original scene elapsed seconds'),
        scope='one prespecified existing V failure point; no fresh model or prediction arms; reference and identity diagnostic only; coordinates and identities withheld')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--lineage-helper', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    print(json.dumps(audit(args.runs.resolve(), args.lineage_helper.resolve()), allow_nan=False, ensure_ascii=False, indent=2))
