"""C103: blind same-supplier speed windows, then evaluate new V252 near endpoints."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import os
from pathlib import Path
import resource
import time

import numpy as np

from association.data.archive import MTADArchive
from association.data.physical_units import KNOT_TO_MPS
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.paper_fusion_metrics import COUNTS, aggregate_paper_fusion
from tools.analysis.held_fusion_population import (
    HELD_IDS, PARENTS, SPLIT, load_full_population, load_scene, read_evaluation,
    validate_measurement_archive,
)
from tools.analysis.motion_fallback_diagnostic import INVARIANTS, freeze_scene
from tools.analysis.motion_readout_evaluation import POPULATION_KEYS
from tools.analysis.motion_readout_pilot import readouts
from tools.analysis.near_endpoint_readout import build_readout
from tools.analysis.near_speed_interval_diagnostic import WINDOWS, alignment, comparison, independent_speed, stats
from tools.analysis.offline_endpoint_lineage import endpoint_arrays, freeze_endpoints, speed_between
from tools.analysis.same_source_swap_pilot import MAX_AS, group_members, peak_rss_bytes, read, write
from tools.analysis.v300_far_paper_bridge import C83

C89 = 'LEVEL1-V300-FAR-PAPER-BRIDGE-20261112-attempt-1'
TURN_DEGREES = 30.
MOVING_MPS = .5
ACCEL_MPS2 = .05
GAP_MULTIPLE = 2.5
READOUTS = ('field', 'net_1', 'net_3', 'net_5', 'path_1', 'path_3', 'path_5', 'guarded_net_5')
METHODS = ('B', 'NEAR', 'GUARDED')


def circular_spread(heading):
    if len(heading) < 2:
        return None
    delta = (heading[:, None] - heading[None, :] + 180.) % 360. - 180.
    return float(np.max(np.abs(delta)))


def headings_from_positions(positions):
    latitude = np.radians(positions[:, 0])
    delta_lon = np.radians((np.diff(positions[:, 1]) + 180.) % 360. - 180.)
    east = np.sin(delta_lon) * np.cos(latitude[1:])
    north = (np.cos(latitude[:-1]) * np.sin(latitude[1:])
             - np.sin(latitude[:-1]) * np.cos(latitude[1:]) * np.cos(delta_lon))
    return np.degrees(np.arctan2(east, north))


def classes(turn, accel):
    turn_class = 'unknown' if turn is None else ('turn' if turn >= TURN_DEGREES else 'straight')
    if accel is None:
        accel_class = 'unknown'
    elif accel >= ACCEL_MPS2:
        accel_class = 'accelerate'
    elif accel <= -ACCEL_MPS2:
        accel_class = 'decelerate'
    else:
        accel_class = 'steady'
    return turn_class, accel_class


def endpoint_details(t, v, side, field):
    _, (_, secants, fields, *_) = alignment(v, t)
    dt = np.diff(t)
    repeated = bool(fields[-1] == fields[-2])
    median = float(np.median(dt))
    speeds = {name: None for name in READOUTS}
    speeds['field'] = float(field)
    windows = {}
    motion = dict(turn_deg=None, position_turn_deg=None, accel_mps2=None)
    for k in WINDOWS:
        if len(dt) < k:
            continue
        a, b = (0, k) if side == 0 else (len(t) - 1 - k, len(t) - 1)
        elapsed = float(t[b] - t[a])
        net = float(speed_between(v[a, :2], v[b, :2], np.asarray(elapsed)))
        np.testing.assert_allclose(net, independent_speed(v[a, :2], v[b, :2], elapsed), rtol=0., atol=1e-8)
        path = float(np.dot(secants[a:b], dt[a:b]) / elapsed)
        speeds[f'net_{k}'], speeds[f'path_{k}'] = net, path
        windows[str(k)] = dict(dt_seconds=elapsed, max_dt_over_fragment_median=float(dt[a:b].max()/median),
                               net_over_path=(net/path if path else None))
        if k == 5:
            moving = fields[a:b] >= MOVING_MPS
            position_moving = secants[a:b] >= MOVING_MPS
            motion = dict(turn_deg=circular_spread(v[a:b, 3][moving]),
                          position_turn_deg=circular_spread(headings_from_positions(v[a:b+1, :2])[position_moving]),
                          accel_mps2=float((fields[b-1]-fields[a])/(t[b-1]-t[a])))
    turn, accel = classes(motion['turn_deg'], motion['accel_mps2'])
    geometric_turn, _ = classes(motion['position_turn_deg'], None)
    boundary = 'first' if side == 0 else ('last_repeated' if repeated else 'last_other')
    gap = 'unknown' if '5' not in windows else ('gap' if windows['5']['max_dt_over_fragment_median'] > GAP_MULTIPLE else 'regular')
    eligible = gap == 'regular' and turn == 'straight' and accel == 'steady' and boundary != 'last_other'
    speeds['guarded_net_5'] = speeds['net_5'] if eligible else speeds['field']
    return dict(speeds=speeds, windows=windows, motion=motion, turn=turn, geometric_turn=geometric_turn,
                accel=accel, boundary=boundary, gap=gap, guard_active=eligible, native_intervals=len(dt),
                fragment_dt_median_seconds=median,
                last_field_to_backward_secant_mps=abs(float(fields[-1]-secants[-1])))


def freeze(runs, sid, scene):
    nodes, sources, support, nearest, membership, native, baseline = (
        scene[key] for key in ('nodes', 'sources', 'support', 'nearest', 'membership', 'native', 'baseline'))
    query, times, _, offsets = native
    raw_values = scene['values']
    outputs, masks = readouts(native, membership, baseline, support)
    c74 = dict(outputs, **masks, membership=membership)
    legacy = scene['native_motion'][..., 2:]
    tags = freeze_scene((nodes, sources, support, nearest, membership, native, baseline), c74, legacy)
    endpoints = endpoint_arrays(times, raw_values, offsets)
    lineage = freeze_endpoints(tags, query, endpoints, sources, baseline, legacy)
    candidate = build_readout(c74, tags, lineage)
    with np.load(runs/C89/'private'/f'{sid}.npz', allow_pickle=False) as historical:
        for key, actual in dict(B=candidate['B'], nodes=nodes, query=query, membership=membership,
                                support=support, nearest=nearest).items():
            np.testing.assert_array_equal(actual, historical[key], err_msg=f'{sid} C89 {key}')
    rows = []
    details = {}
    guarded = candidate['NEAR'].copy()
    for g, q in zip(*np.nonzero(candidate['near_outside']), strict=True):
        switch = bool(candidate['switch'][g, q])
        row = dict(group=int(g), query_index=int(q), switch=switch, roles={},
                   endpoint_ids={}, same_source=bool(lineage['same_source'][g, q]))
        for role, prefix in (('B', 'first'), ('NEAREST', 'nearest')):
            node = int(lineage[f'{prefix}_member'][g, q])
            side = int(lineage[f'{prefix}_side'][g, q])
            field = float(lineage[f'{prefix}_field'][g, q])
            key = (node, side)
            if key not in details:
                a, b = map(int, offsets[node:node+2])
                ep = endpoint_details(times[a:b], raw_values[a:b], side, field)
                details[key] = dict(ep, endpoint_id=f'{sid}-E{len(details)+1:03d}')
            ep = details[key]
            np.testing.assert_equal(ep['speeds']['field'], field)
            row['endpoint_ids'][role] = ep['endpoint_id']
            row['roles'][role] = dict(node=node, side=side, age_seconds=float(lineage[f'{prefix}_age'][g, q]))
            expected = candidate['B' if role == 'B' else 'NEAR'][g, q, 2]
            np.testing.assert_equal(field, expected)
            if role == 'NEAREST' and switch:
                guarded[g, q, 2] = ep['speeds']['guarded_net_5']
        rows.append(row)
    assert sum(row['switch'] for row in rows) == int(candidate['switch'].sum())
    np.testing.assert_array_equal(guarded[..., [0, 1, 3]], candidate['B'][..., [0, 1, 3]])
    np.testing.assert_array_equal(guarded[~candidate['switch']], candidate['B'][~candidate['switch']])
    keys = ('nodes', 'query', 'membership', 'support', 'nearest')
    arrays = dict(B=candidate['B'], NEAR=candidate['NEAR'], GUARDED=guarded,
                  **{key: scene[key] for key in keys})
    return arrays, rows, details


def reference_motion(states, valid, query, target, q):
    if q == 0 or q == len(query)-1:
        return None, None
    s = states[target, q-1:q+2]
    mask = valid[target, q-1:q+2]
    turn = circular_spread(s[:, 3][(s[:, 2] >= MOVING_MPS) & mask[:, 2] & mask[:, 3]])
    accel = float((s[-1, 2]-s[0, 2])/(query[q+1]-query[q-1])) if mask[:, 2].all() else None
    return turn, accel


def evaluate(runs, sid, component, roster, historical, frozen_path, rows):
    with np.load(frozen_path, allow_pickle=False) as pack:
        artifact = {key: pack[key] for key in pack.files}
    members = artifact['membership']
    groups = [group.tolist() for group in group_members(members)]
    ev = read_evaluation(runs, sid, artifact['nodes'], artifact['query'], scene_ids=roster)
    reports, artifacts = evaluate_fusion_methods(
        {name: emit_group_outputs(artifact[name], groups, node_count=len(members)) for name in METHODS},
        prediction_speed_units=dict.fromkeys(METHODS, 'mps'), node_identities=ev['identities'],
        target_ids=ev['target_ids'], target_states=ev['target_states'], target_valid=ev['target_valid'],
        support=artifact['support'], nearest_seconds=artifact['nearest'], target_speed_unit='mps', split=SPLIT)
    papers = {name: reports[name]['paper_fusion_metrics'] for name in METHODS}
    base = artifacts['B']
    np.testing.assert_equal(papers['B'], historical['own_paper']['B'])
    population = {key: int(reports['B'][key]) for key in POPULATION_KEYS}
    for name in METHODS[1:]:
        for key in INVARIANTS:
            np.testing.assert_array_equal(base[key], artifacts[name][key])
        np.testing.assert_array_equal(base['aligned_prediction'][..., [0, 1, 3]],
                                      artifacts[name]['aligned_prediction'][..., [0, 1, 3]])
        assert all(reports[name][key] == value for key, value in population.items())
        for feature, old in papers['B']['attributes'].items():
            new = papers[name]['attributes'][feature]
            assert all(old[key] == new[key] for key in COUNTS)
            if feature != 'speed':
                assert old == new
    nt = len(ev['target_ids'])
    chosen = base['aligned_group_indices'][:nt]
    targets = {int(g): t for t, g in enumerate(chosen) if g >= 0}
    valid_speed = base['aligned_truth_valid'][:nt, :, 2] & base['aligned_prediction_valid'][:nt, :, 2]
    errors = {name: np.abs((artifacts[name]['aligned_prediction'][:nt, :, 2]
                            - ev['target_states'][..., 2])[valid_speed]) for name in METHODS}
    points = []
    excluded = Counter()
    for row in rows:
        target = targets.get(row['group'])
        q = row['query_index']
        reason = ('no_first_claim' if target is None else
                  'invalid_speed_reference' if not valid_speed[target, q] else None)
        if reason is not None:
            excluded[f'{reason}_all'] += 1
            excluded[f'{reason}_{"switch" if row["switch"] else "no_switch"}'] += 1
            continue
        donor = {role: ev['identities'][data['node']] == ev['target_ids'][target]
                 for role, data in row['roles'].items()}
        assert donor['B']
        turn, accel = reference_motion(ev['target_states'], ev['target_valid'], artifact['query'], target, q)
        rt, ra = classes(turn, accel)
        points.append(dict(scene=sid, component=component, group=row['group'], query_index=q,
                           switch=row['switch'], reference=float(ev['target_states'][target, q, 2]),
                           endpoint_ids=row['endpoint_ids'], same_source=row['same_source'],
                           reference_turn=rt, reference_accel=ra, reference_turn_deg=turn,
                           reference_accel_mps2=accel, donor_same_identity={k: bool(v) for k, v in donor.items()},
                           age_seconds={k: v['age_seconds'] for k, v in row['roles'].items()}))
    public = dict(scene=sid, component=component, population=population, paper=papers,
                  blind_near=len(rows), blind_switch=sum(row['switch'] for row in rows),
                  valid_near=len(points), valid_switch=sum(row['switch'] for row in points),
                  exclusions=dict(excluded), full_speed={k: stats(v) for k, v in errors.items()})
    return public, points, errors, ev['identities']


def measure_lineage(scene, details, identities):
    """Evaluation-only known same-identity external neighbor and forward alignment."""
    _, times, _, offsets = scene['native']
    values, sources = scene['values'], scene['sources']

    @lru_cache(maxsize=None)
    def fragment(node):
        a, b = map(int, offsets[node:node+2])
        summary, _ = alignment(values[a:b], times[a:b])
        return a, b, summary

    result = []
    for (node, side), item in details.items():
        a, b, summary = fragment(node)
        t = times[a] if side == 0 else times[b-1]
        others = np.flatnonzero((identities == identities[node]) & (sources == sources[node]))
        best = None
        for j in others:
            if j == node:
                continue
            x, y = offsets[j:j+2]
            signed = (t-times[x:y]) if side == 0 else (times[x:y]-t)
            eligible = signed[signed > 0.]
            if len(eligible):
                gap = float(eligible.min())
                best = gap if best is None else min(best, gap)
        result.append(dict(item, fragment_id=f'{item["endpoint_id"].split("-E")[0]}-F{node+1:04d}',
                           external_dt_seconds=best,
                           external_gap_multiple=best/item['fragment_dt_median_seconds'] if best is not None else None,
                           alignment=summary))
    return result


def summarize(points, endpoints):
    end = {row['endpoint_id']: row for row in endpoints}
    results = {}
    for domain, sample in (('near_switch', [r for r in points if r['switch']]),
                           ('near_no_switch', [r for r in points if not r['switch']]), ('near_all', points)):
        domain_result = {}
        for role in ('B', 'NEAREST'):
            enriched = []
            for point in sample:
                endpoint = end[point['endpoint_ids'][role]]
                row = dict(point, endpoint=endpoint, diagnostic_speeds=endpoint['speeds'])
                enriched.append(row)
            strata = {'all': enriched}
            for key, labels in dict(turn=('turn', 'straight', 'unknown'), accel=('accelerate', 'decelerate', 'steady', 'unknown'),
                                    geometric_turn=('turn', 'straight', 'unknown'), boundary=('first', 'last_repeated', 'last_other'),
                                    gap=('gap', 'regular', 'unknown')).items():
                for label in labels:
                    strata[f'{key}_{label}'] = [r for r in enriched if r['endpoint'][key] == label]
            for key, labels in dict(reference_turn=('turn', 'straight', 'unknown'),
                                    reference_accel=('accelerate', 'decelerate', 'steady', 'unknown')).items():
                for label in labels:
                    strata[f'{key}_{label}'] = [r for r in enriched if r[key] == label]
            for same in (False, True):
                strata[f'same_identity_{same}'] = [r for r in enriched if r['donor_same_identity'][role] == same]
            domain_result[role] = {}
            for name, rows in strata.items():
                common = [r for r in rows if r['diagnostic_speeds']['net_5'] is not None]
                fallback = [dict(r, diagnostic_speeds={k: v if v is not None else r['diagnostic_speeds']['field']
                                                      for k, v in r['diagnostic_speeds'].items()}) for r in rows]
                domain_result[role][name] = dict(
                    common_5={method: comparison(common, method, 'field') for method in READOUTS},
                    with_field_fallback={method: comparison(fallback, method, 'field') for method in READOUTS},
                    actual_available={method: sum(r['diagnostic_speeds'][method] is not None for r in rows) for method in READOUTS},
                    guard_activated=sum(r['endpoint']['guard_active'] for r in rows))
        results[domain] = domain_result
    return results


def run(args):
    began = time.monotonic()
    args.output.mkdir(exist_ok=False)
    private = args.output/'private'
    private.mkdir()
    record = dict(status='RUNNING', plan_version='1.0', idea_id='IDEA-C103-NEAR-SPEED-MANEUVER-HELD-20261112',
                  started_at=datetime.now(timezone.utc).isoformat(), affinity=sorted(os.sched_getaffinity(0)),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), optimizer_updates=0,
                  test_opened=False, stages=[])

    def checkpoint(**fields):
        record.update(fields, elapsed_seconds=time.monotonic()-began, peak_rss_bytes=peak_rss_bytes())
        write(args.output/'run.json', record)

    checkpoint()
    try:
        assert record['affinity'] == [27] and record['cuda_visible_devices'] == ''
        full_ids, components, archives = load_full_population(args.runs)
        c83, c89 = (read(args.runs/name/'run.json') for name in (C83, C89))
        assert all(r['status'] == 'COMPLETED' and r['exit_code'] == 0 for r in (c83, c89))
        assert tuple(c83['scene_ids']) == HELD_IDS and c89['scene_ids'] == full_ids
        ids = [sid for sid in full_ids if sid not in set(HELD_IDS)]
        roster = {sid: components[sid] for sid in ids}
        assert len(roster) == 252 and len(set(roster.values())) == 15
        assert not set(roster.values()) & {components[sid] for sid in HELD_IDS}
        frozen_roster = read(args.roster)
        assert frozen_roster['scenes'] == ids and frozen_roster['scene_components'] == roster
        write(args.output/'roster.json', frozen_roster)
        scenes, points, endpoints, full_errors = [], [], [], []
        with MTADArchive(archives) as archive:
            validate_measurement_archive(args.runs, archive)
            frozen_details = {}
            for n, sid in enumerate(ids, 1):
                scene = load_scene(args.runs, sid, archive, scene_ids=roster)
                arrays, rows, details = freeze(args.runs, sid, scene)
                np.savez_compressed(private/f'{sid}.npz', **arrays)
                # Record only allowed native arrays to support a single read of each archive member.
                if details:
                    _, times, _, offsets = scene['native']
                    np.savez_compressed(private/f'{sid}-native.npz', times=times, offsets=offsets,
                                        values=scene['values'], sources=scene['sources'])
                frozen_details[sid] = (rows, details)
                if n % 25 == 0:
                    checkpoint(frozen_scenes=n)
            # The complete blind choice record is serialized before opening any 151 evaluation file.
            write(private/'blind-endpoints.json.gz', {sid: dict(rows=rows,
                endpoints=[dict(ep, node=node, side=side) for (node, side), ep in details.items()])
                for sid, (rows, details) in frozen_details.items()})
            reads = [(row['scene_id'], row['kind']) for row in archive.members_read]
            assert reads == [(sid, 'measurements') for sid in ids]
        record['stages'].append('all_v252_blind_outputs_and_endpoint_rules_saved_before_evaluation')
        checkpoint(frozen_scenes=len(ids), measurement_members=len(ids), label_members=0, reference_members=0)
        historical = {row['scene']: row for row in read(args.runs/C89/'scenes.json.gz')}
        for n, sid in enumerate(ids, 1):
            rows, details = frozen_details[sid]
            public, selected, errors, identities = evaluate(args.runs, sid, roster[sid], roster,
                historical[sid], private/f'{sid}.npz', rows)
            scenes.append(public)
            points.extend(selected)
            full_errors.append(errors)
            if details:
                with np.load(private/f'{sid}-native.npz', allow_pickle=False) as raw:
                    scene = dict(native=(None, raw['times'], None, raw['offsets']),
                                 values=raw['values'], sources=raw['sources'])
                endpoints.extend(measure_lineage(scene, details, identities))
            if n % 25 == 0:
                checkpoint(evaluated_scenes=n)
        components_count = Counter(roster.values())
        largest = [c for c, _ in components_count.most_common(2)]
        selected_endpoints = {endpoint_id for point in points for endpoint_id in point['endpoint_ids'].values()}
        unique = {ep['fragment_id']: ep['alignment'] for ep in endpoints if ep['endpoint_id'] in selected_endpoints}
        core = summarize(points, endpoints)
        by_component = {}
        for component in components_count:
            sub = [row for row in points if row['component'] == component and row['switch']]
            known = {ep['endpoint_id']: ep for ep in endpoints}
            enriched = [dict(row, diagnostic_speeds=known[row['endpoint_ids']['NEAREST']]['speeds']) for row in sub]
            by_component[component] = dict(scenes=components_count[component], valid_switch=len(sub),
                nearest_net5= comparison(enriched, 'net_5', 'field'), guarded=comparison(enriched, 'guarded_net_5', 'field'))
        exclusions = Counter()
        for row in scenes:
            exclusions.update(row['exclusions'])
        result = dict(scenes=len(ids), components=len(components_count), component_sizes=dict(components_count),
                      blind_near=sum(row['blind_near'] for row in scenes), blind_switch=sum(row['blind_switch'] for row in scenes),
                      valid_near=len(points), valid_switch=sum(row['switch'] for row in points),
                      blind_distinct_endpoints=len(endpoints), valid_distinct_endpoints=len(selected_endpoints),
                      exclusions=dict(exclusions), full_speed={name: stats(np.concatenate([e[name] for e in full_errors]))
                                                               for name in METHODS},
                      paper={name: aggregate_paper_fusion([row['paper'][name] for row in scenes]) for name in METHODS},
                      near_comparisons=core, by_component=by_component,
                      exclude_largest_two=dict(components=largest, results=summarize(
                          [row for row in points if row['component'] not in largest], endpoints)),
                      fragment_alignment=dict(fragments=len(unique), intervals=sum(s['intervals'] for s in unique.values()),
                          interior_points=sum(s['interior_points'] for s in unique.values()),
                          closer_following=sum(s['interior_closer_following'] for s in unique.values()),
                          closer_preceding=sum(s['interior_closer_preceding'] for s in unique.values()),
                          tied=sum(s['interior_tied'] for s in unique.values()),
                          max_distance_gap=max((s['max_distance_implementation_gap_mps'] for s in unique.values()), default=None)))
        write(args.output/'points.json.gz', points)
        write(args.output/'endpoints.json.gz', endpoints)
        write(args.output/'scenes.json.gz', scenes)
        write(args.output/'summary.json.gz', result)
        record['stages'].append('new_holdout_references_and_fixed_metrics_evaluated')
        checkpoint(status='COMPLETED', exit_code=0, evaluated_scenes=len(ids),
                   valid_near=len(points), valid_switch=result['valid_switch'],
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        checkpoint(status='FAILED', exit_code=1, error=repr(error), finished_at=datetime.now(timezone.utc).isoformat())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roster', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
    run(args)
