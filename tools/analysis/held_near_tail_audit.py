"""Read-only audit of adverse near-endpoint switches in the frozen C83 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from association.data.archive import MTADArchive
from association.data.physical_units import KNOT_TO_MPS
from tools.analysis.held_fusion_population import (
    load_population, load_scene, read_evaluation, validate_measurement_archive,
)
from tools.analysis.motion_fallback_diagnostic import support_tags
from tools.analysis.offline_endpoint_lineage import endpoint_arrays, freeze_endpoints
from tools.analysis.same_source_swap_pilot import group_members, read


def speed_at(ev, target, query):
    if ev['target_valid'][target, query, 2]:
        return float(ev['target_states'][target, query, 2])
    return None


def near_points(runs, private, ids):
    points = []
    for sid in ids:
        with np.load(private / f'{sid}.npz', allow_pickle=False) as frozen:
            ev = read_evaluation(runs, sid, frozen['nodes'], frozen['query'])
            target_index = {key: index for index, key in enumerate(ev['target_ids'])}
            claimed = set()
            for gid, members in enumerate(group_members(frozen['membership'])):
                target = target_index[ev['identities'][members[0]]]
                if target in claimed:
                    continue
                claimed.add(target)
                mask = frozen['near'][gid] & ev['target_valid'][target, :, 2]
                for q in np.flatnonzero(mask):
                    reference = speed_at(ev, target, q)
                    base = float(frozen['B'][gid, q, 2])
                    candidate = float(frozen['NEAR'][gid, q, 2])
                    points.append(dict(scene=sid, group=gid, query_index=int(q), target=target,
                                       reference=reference, base=base, candidate=candidate,
                                       error_change=abs(candidate-reference)-abs(base-reference)))
    return points


def endpoint_info(node, side, query, members, sources, ev, scene, endpoints):
    _, times, _, offsets = scene['native']
    left, right = offsets[node:node+2]
    indices = range(left, min(right, left+3)) if side == 0 else range(max(left, right-3), right)
    event = endpoints['index'][node, side]
    identity = ev['identities'][node]
    raw_speed = float(scene['values'][event, 2])
    secant = float(endpoints['secant'][node, side])
    source_number = np.searchsorted(np.unique(sources), sources[node])+1
    return dict(member_slot=int(np.flatnonzero(members == node)[0]),
                source=f'S{source_number}', same_identity_as_anchor=bool(identity == ev['identities'][members[0]]),
                endpoint_side='first' if side == 0 else 'last', endpoint_time_scene_seconds=float(times[event]),
                age_seconds=float(abs(query-times[event])), raw_speed_kn=raw_speed,
                held_speed_mps=float(endpoints['field'][node, side]), adjacent_secant_mps=secant,
                adjacent_secant_interval_seconds=float(endpoints['elapsed'][node, side]),
                raw_to_adjacent_secant_ratio=(raw_speed*KNOT_TO_MPS/secant if secant > 0. else None),
                first_or_last_three_native_samples=[
                    dict(seconds_from_query=float(times[i]-query), raw_speed_kn=float(scene['values'][i, 2]))
                    for i in indices])


def describe_scene(scene, artifact, ev, adverse):
    members = artifact['membership']
    query, times, _, offsets = scene['native']
    groups = group_members(members)
    np.testing.assert_array_equal(artifact['nodes'], scene['nodes'])
    np.testing.assert_array_equal(query, artifact['query'])
    np.testing.assert_array_equal(members, scene['membership'])
    np.testing.assert_array_equal(artifact['support'], scene['support'])
    sources = scene['sources']
    tags = support_tags(query, times, offsets, groups, scene['support'], sources)
    endpoints = endpoint_arrays(times, scene['values'], offsets)
    lineage = freeze_endpoints(dict(tags, membership=members), query, endpoints, sources,
                               scene['baseline'], scene['native_motion'][..., 2:])
    target_index = {key: index for index, key in enumerate(ev['target_ids'])}
    rows = []
    for point in adverse:
        g, q = point['group'], point['query_index']
        target = point['target']
        first = int(lineage['first_member'][g, q])
        selected = int(lineage['nearest_member'][g, q])
        old_side = int(lineage['first_side'][g, q])
        new_side = int(lineage['nearest_side'][g, q])
        assert first == groups[g][0] and first != selected and old_side == new_side
        assert not tags['visible'][g, q]
        assert lineage['first_age'][g, q] > lineage['nearest_age'][g, q]
        assert bool(artifact['near_before'][g, q]) == (new_side == 0)
        np.testing.assert_equal(artifact['nearest_event_seconds'][g, q], lineage['nearest_age'][g, q])
        np.testing.assert_equal(artifact['B'][g, q, 2], lineage['first_field'][g, q])
        np.testing.assert_equal(artifact['NEAR'][g, q, 2], lineage['nearest_field'][g, q])
        same_source = bool(sources[first] == sources[selected])
        assert same_source == bool(artifact['near_same_source'][g, q])
        donor_target = target_index[ev['identities'][selected]]
        qstep = float((query[-1]-query[0])/(len(query)-1))
        old = endpoint_info(first, old_side, query[q], groups[g], sources, ev, scene, endpoints)
        donor = endpoint_info(selected, new_side, query[q], groups[g], sources, ev, scene, endpoints)
        others = groups[g][(groups[g] != first) & (ev['identities'][groups[g]] == ev['identities'][first])]
        ages = np.abs(endpoints['time'][others, new_side]-query[q])
        reference = point['reference']
        rows.append(dict(
            {key: value for key, value in point.items() if key != 'target'},
            query_time_scene_seconds=float(query[q]), q_step_seconds=qstep,
            original_group_member_count=len(groups[g]),
            distinct_truth_identities_in_group=len(set(ev['identities'][groups[g]].tolist())),
            side='before_group' if old_side == 0 else 'after_group',
            same_source=same_source, first_member=old, selected_member=donor,
            same_identity_alternative_members=len(others),
            same_identity_alternatives_within_two_q_steps=int((ages <= 2*qstep).sum()),
            nearest_same_identity_alternative_age_seconds=float(ages.min()) if len(ages) else None,
            target_reference_neighbor_speeds=[
                dict(query_index=i, mps=speed_at(ev, target, i))
                for i in range(max(0, q-1), min(len(query), q+2))],
            donor_own_reference_speed_at_query_mps=speed_at(ev, donor_target, q),
            first_field_signed_error_mps=point['base']-reference,
            selected_field_signed_error_mps=point['candidate']-reference,
            first_secant_absolute_error_mps=abs(old['adjacent_secant_mps']-reference),
            selected_secant_absolute_error_mps=abs(donor['adjacent_secant_mps']-reference)))
    return rows


def audit(runs, science):
    ids, _, archive_paths = load_population(runs)
    run = read(science / 'run.json')
    if (science.parent != runs or science.name != 'LEVEL1-HELD-SPEED-READOUT-20261112-attempt-1'
            or run['status'] != 'COMPLETED' or run['exit_code'] != 0 or run['scene_ids'] != ids
            or run['frozen_prediction_scenes'] != len(ids) or run['completed_scenes'] != len(ids)
            or run['stages'] != ['all_held_factorial_outputs_and_input_support_frozen_before_reference',
                                'fixed_reference_and_four_attribute_evaluation_complete']):
        raise ValueError('completed original frozen C83 run required')
    private = science / 'private'
    points = near_points(runs, private, ids)
    adverse = [point for point in points if point['error_change'] > 0.]
    improved = [point for point in points if point['error_change'] < 0.]
    observed = read(science / 'summary.json')['all_components']['speed']['near']['NEAR_vs_B']
    assert len(points) == observed['count'] and len(adverse) == observed['worsened_points']
    assert len(improved) == observed['improved_points']
    np.testing.assert_equal(max(point['error_change'] for point in adverse), observed['max_point_harm_mps'])
    np.testing.assert_allclose(sum(point['error_change'] for point in points), observed['absolute_error_sum_delta'])
    scenes = [sid for sid in ids if any(point['scene'] == sid for point in adverse)]
    rows = []
    with MTADArchive(archive_paths) as archive:
        validate_measurement_archive(runs, archive)
        for sid in scenes:
            scene = load_scene(runs, sid, archive)
            with np.load(private / f'{sid}.npz', allow_pickle=False) as artifact:
                ev = read_evaluation(runs, sid, artifact['nodes'], artifact['query'])
                rows.extend(describe_scene(scene, artifact, ev, [p for p in adverse if p['scene'] == sid]))
        assert [(r['scene_id'], r['kind']) for r in archive.members_read] == [(sid, 'measurements') for sid in scenes]
    rows.sort(key=lambda p: (-p['error_change'], p['scene'], p['group'], p['query_index']))
    return dict(science=str(science), all_held_scenes=len(ids), valid_near_points=len(points),
                improved_near_points=len(improved), worsened_near_points=len(adverse),
                adverse_scenes=scenes, maximum_harm_mps=observed['max_point_harm_mps'],
                count_within_one_nanometre_per_second_of_maximum=sum(
                    abs(point['error_change']-observed['max_point_harm_mps']) <= 1e-9 for point in adverse),
                total_harm_mps=float(sum(point['error_change'] for point in adverse)),
                total_gain_mps=float(-sum(point['error_change'] for point in improved)),
                raw_measurement_members_read=len(scenes), raw_label_members_read=0,
                raw_reference_members_read=0, cached_reference_split='V_select', new_prediction_arms=0,
                units=dict(speed='m/s except explicit raw knots', time='original scene elapsed seconds',
                           error_change='absolute candidate error minus absolute baseline error'),
                adverse_points=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--science', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.runs.resolve(), args.science.resolve()), allow_nan=False, ensure_ascii=False, indent=2))
