"""C150: compare two course readouts using the frozen C149 fit population."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from association.data.archive import MTADArchive
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.physical import _coverage
from tools.analysis import heading_interval_readout
from tools.analysis.heading_fallback_fit import (
    BRANCHES, RSS_LIMIT, SPLIT, TAIL_DEG, bind_inputs, file_hash, native_inputs,
)
from tools.analysis.held_fusion_population import validate_measurement_archive
from tools.analysis.motion_readout_evaluation import COVERAGE_KEYS, POPULATION_KEYS, _stats, _wrap
from tools.analysis.offline_association_tail import read_motion, require, summary_state
from tools.analysis.same_source_swap_pilot import canonical_membership, group_members, peak_rss_bytes, read, write


IDEA = 'IDEA-OFFLINE-HEADING-NATIVE-INTERVAL-C150-20261117'
C149 = 'LEVEL1-OFFLINE-HEADING-FALLBACK-FIT-20261116'
C149_COMMIT = '0c5d7444531e76d4814b6f4614db6cd8e11342fd'
C149_SHA256 = {
    'run': '84850ba38789698696ca9a15e2e04915d839851a6b8e317a07a7df8a558cd52a',
    'freeze': '00a45fa14922bcfd99dbbb44ea502ee2eb7a53ef42334903fe6bb8c24b0561ef',
    'summary': '142bba1c08f0f3bfb0a513b5843793d1c4d110d5d00be904c15dc526aa5fbc56',
    'request': 'cec22e60ed2e6627a0c7220a6d0bcc5eaaa01fe4630cc369e37c5bb6a8574e1f',
    'launcher': 'b9b18440ea2f3388ae0f112ad9802dd24d01a717924df58266bbf6b0cd6f3993',
}
ARMS = ('B_ORIGINAL', 'B_NEAREST_INTERVAL')
ATTRIBUTES = ('latitude', 'longitude', 'speed', 'heading', 'position')
ARTIFACT_INVARIANTS = ('aligned_prediction_valid', 'aligned_truth_valid', 'aligned_group_indices',
                       'aligned_target_ids', 'group_offsets', 'group_node_indices',
                       'group_anchor_target_indices', 'aligned_observed')
SCENE_HASH_KEYS = ('prediction_sha256', 'motion_sha256', 'ws_sha256')
EPSILON = 1e-9
LOW = BRANCHES.index('enabled_low_speed')
REASONS = ('non_low_branch', 'no_candidate', 'degenerate_azimuth', 'enabled')
TIME_SIDES = ('unselected', 'straddles', 'before', 'after')
PROVENANCE = ('unselected', 'all_same_ship', 'all_other_ship', 'mixed_identity')
SOURCE_NAMES = ('9001', '9002')
CHOSEN_NUMBERS = ('duration_seconds', 'max_age_seconds', 'midpoint_age_seconds', 'distance_metres')


def require_hash(path, expected, label):
    actual = file_hash(path)
    require(actual == expected, f'{label} digest differs: {path}')
    return actual


def bind_c149(runs, manifest):
    ids, components, index, partition, paths, digest, metadata = bind_inputs(runs, manifest)
    parent = runs / f'{C149}-attempt-1'
    launch = runs / f'{C149}-launch-attempt-1'
    inputs = dict(run=parent/'run.json', freeze=parent/'freeze.json',
                  request=launch/'request.json', launcher=launch/'status.json')
    for label, path in inputs.items():
        require_hash(path, C149_SHA256[label], f'frozen C149 {label}')
    status, freeze, request, launcher = (read(inputs[label]) for label in inputs)
    require(status['status'] == launcher['status'] == 'COMPLETED'
            and status['exit_code'] == launcher['returncode'] == 0
            and status['completed_scenes'] == len(ids) and status['phase'] == 'done'
            and status['optimizer_updates'] == status['model_forwards'] == status['raw_reference_or_label_reads'] == 0
            and not status['other_split_data_opened'] and launcher['backend'] == 'systemd'
            and not launcher['timed_out'] and launcher['failure'] is None,
            'C149 scientific or launcher completion differs')
    require(freeze['fit_scene_count'] == freeze['archive_measurement_reads'] == len(ids)
            and freeze['fit_component_count'] == len(set(components.values())) == 76
            and freeze['scene_order_sha256'] == digest and freeze['frozen_before_truth']
            and freeze['reference_reads_before_freeze'] == 0
            and freeze['canonical_archive_to_151_state_and_motion_parity']
            and freeze['angular_tail_gt_degrees'] == TAIL_DEG
            and list(freeze['scene_hashes']) == ids,
            'C149 freeze does not bind the C92 fit roster')
    require(all(set(freeze['scene_hashes'][sid]) == set(SCENE_HASH_KEYS) for sid in ids),
            'C149 scene hash schema differs')

    cwd = Path(request['cwd']).resolve()
    argv = request['argv']
    require('tools.analysis.heading_fallback_fit' in argv and Path(argv[argv.index('--runs')+1]).resolve() == runs
            and Path(argv[argv.index('--output')+1]).resolve() == parent.resolve(), 'C149 launcher command differs')
    commit = subprocess.check_output(['git', '-C', str(cwd), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == C149_COMMIT and not subprocess.check_output(
        ['git', '-C', str(cwd), 'status', '--porcelain', '--untracked-files=no'], text=True).strip(),
        'C149 prediction checkout differs from the recorded commit')
    original_manifest = Path(argv[argv.index('--manifest')+1]).resolve()
    original_metadata = {Path(key).resolve(): digest for key, digest in freeze['metadata_sha256'].items()}
    expected_paths = {original_manifest, *(path.resolve() for path in metadata[1:])}
    require(set(original_metadata) == expected_paths, 'C149 metadata paths differ')
    for path, expected in original_metadata.items():
        require_hash(path, expected, 'C149 source metadata')
    require_hash(manifest, original_metadata[original_manifest], 'C150 manifest')
    sizes = Counter(components.values())
    largest = min(sizes, key=lambda key: (-sizes[key], key))
    require(sizes[largest] == 966, 'original C149 largest fit component changed')
    return ids, components, index, partition, paths, parent, freeze, largest


def verify_arms(saved, baseline):
    for key in baseline.files:
        np.testing.assert_array_equal(saved[key], baseline[key], err_msg=f'original C149 field changed: {key}')
    old, new = saved['states'], saved['new_states']
    enabled, branch, reason = saved['enabled'], saved['branch'], saved['fallback_reason']
    chosen = saved['selected_interval_index']
    require(old.shape == new.shape == (*branch.shape, 4) and branch.ndim == 2 and branch.shape[1] == 64
            and enabled.shape == reason.shape == chosen.shape == branch.shape and enabled.dtype == bool
            and reason.dtype.kind in 'ui' and chosen.dtype.kind == 'i'
            and np.isfinite(old).all() and np.isfinite(new).all(), 'C150 frozen shape, mask or values differ')
    require(np.isin(branch, np.arange(len(BRANCHES))).all() and np.isin(reason, np.arange(len(REASONS))).all()
            and np.array_equal(reason > 0, branch == LOW) and np.array_equal(enabled, reason == 3)
            and np.array_equal(chosen >= 0, reason >= 2) and np.all(chosen[reason < 2] == -1)
            and np.all((new[..., 3][enabled] >= 0.) & (new[..., 3][enabled] < 360.)),
            'C150 enable, selection or fallback violates the original low-speed branch')
    np.testing.assert_array_equal(new[..., :3], old[..., :3])
    np.testing.assert_array_equal(new[..., 3][~enabled], old[..., 3][~enabled])


def freeze_one(archive, sid, row, baseline_path, ws_path, expected, output):
    motion_path = Path(row['motion'])
    checked = {key: require_hash(path, expected[key], f'C149 {sid} {key}')
               for key, path in zip(SCENE_HASH_KEYS, (baseline_path, motion_path, ws_path), strict=True)}
    motion = read_motion(row)
    ws = canonical_membership(np.asarray(read(ws_path)['ws']), len(motion['node_ids']))
    with np.load(motion_path, allow_pickle=False) as raw, np.load(baseline_path, allow_pickle=False) as baseline:
        tracks, native = native_inputs(archive, sid, motion, raw['states'])
        np.testing.assert_array_equal(ws, baseline['membership'], err_msg='C139 WS membership or order differs')
        arrays = heading_interval_readout.freeze_scene(motion, tracks, native, baseline)
        verify_arms(arrays, baseline)
        np.testing.assert_array_equal(arrays['sources'], motion['sources'])
    np.savez_compressed(output, **arrays)
    return dict(source_c149=checked, prediction_sha256=file_hash(output))


def direction_stats(old, new):
    old_abs, new_abs = np.abs(old), np.abs(new)
    delta = new_abs - old_abs
    old_tail, new_tail = old_abs > TAIL_DEG, new_abs > TAIL_DEG
    improved, harmed = delta < 0., delta > 0.
    non_tail = ~old_tail
    non_tail_delta = delta[non_tail]
    old_stats, new_stats = _stats(old), _stats(new)
    old_stats.update(absolute_error_sum=float(old_abs.sum()), tail_gt150=int(old_tail.sum()))
    new_stats.update(absolute_error_sum=float(new_abs.sum()), tail_gt150=int(new_tail.sum()))
    return dict(count=len(old), baseline=old_stats, candidate=new_stats,
                delta_mae=(new_stats['mae']-old_stats['mae'] if len(old) else None),
                delta_p99=(new_stats['p99']-old_stats['p99'] if len(old) else None),
                rescued_tail=int((old_tail & ~new_tail).sum()), new_tail=int((~old_tail & new_tail).sum()),
                improved_points=int(improved.sum()), worsened_points=int(harmed.sum()),
                unchanged_points=int((delta == 0.).sum()),
                maximum_harm_degrees=float(np.maximum(delta, 0.).max(initial=0.)),
                old_non_tail=dict(count=int(non_tail.sum()), harmed=int((non_tail_delta > 0.).sum()),
                    harmed_gt30=int((non_tail_delta > 30.).sum()), harmed_gt90=int((non_tail_delta > 90.).sum()),
                    added_absolute_error_sum=float(non_tail_delta[non_tail_delta > 0.].sum()),
                    reduced_absolute_error_sum=float(-non_tail_delta[non_tail_delta < 0.].sum()),
                    delta_absolute_error_sum=float(non_tail_delta.sum()),
                    mean_delta_absolute_error=(float(non_tail_delta.mean()) if len(non_tail_delta) else None),
                    maximum_harm_degrees=float(np.maximum(non_tail_delta, 0.).max(initial=0.))))


def heading_statistics(vectors, *, detailed):
    old, new = vectors['old_error'], vectors['new_error']
    enabled, branch, pure = (vectors[key] for key in ('enabled', 'branch', 'pure'))

    def region(mask):
        return direction_stats(old[mask], new[mask])

    result = dict(all=direction_stats(old, new), enabled=region(enabled), old_low_speed=region(branch == LOW),
                  branches={name: region(branch == i) for i, name in enumerate(BRANCHES)},
                  pure=region(pure), mixed=region(~pure),
                  enabled_pure=region(enabled & pure), enabled_mixed=region(enabled & ~pure))
    if detailed:
        result.update(
            enabled_time_side={name: region(enabled & (vectors['time_side'] == i))
                               for i, name in enumerate(TIME_SIDES) if i},
            enabled_provider_identity={name: region(enabled & (vectors['provider_identity'] == i))
                                       for i, name in enumerate(PROVENANCE) if i},
            enabled_provider_member={name: region(enabled & (vectors['provider_first'] == flag))
                                     for name, flag in (('anchor', True), ('other_member', False))},
            enabled_provider_source={name: region(enabled & (vectors['provider_source'] == i))
                                     for i, name in enumerate(SOURCE_NAMES)})
    return result


def chosen_metadata(saved, groups, identities):
    chosen = saved['selected_interval_index']
    used = chosen >= 0
    candidate_groups, offsets, providers = (saved[key] for key in
        ('candidate_groups', 'candidate_provider_offsets', 'candidate_provider_nodes'))
    require(candidate_groups.ndim == providers.ndim == offsets.ndim == 1 and len(offsets) == len(candidate_groups)+1
            and offsets[0] == 0 and offsets[-1] == len(providers) and np.all(np.diff(offsets) > 0)
            and np.all((candidate_groups >= 0) & (candidate_groups < len(groups)))
            and np.all((providers >= 0) & (providers < len(identities)))
            and np.all(chosen[used] < len(candidate_groups)), 'C150 selected catalogue or provider binding differs')
    anchors = np.asarray([members[0] for members in groups], int)
    sources = saved['sources']
    require(np.isin(sources, SOURCE_NAMES).all(), 'unexpected MTAD source in C150')
    source_codes = np.where(sources == SOURCE_NAMES[0], 0, 1).astype(np.int8)
    chosen_nodes = saved['selected_provider_node'][used]
    gids, _ = np.nonzero(used)
    require(np.array_equal(candidate_groups[chosen[used]], gids)
            and np.all((chosen_nodes >= 0) & (chosen_nodes < len(identities))), 'C150 selected provider roster differs')
    np.testing.assert_array_equal(saved['membership'][providers], np.repeat(candidate_groups, np.diff(offsets)))
    np.testing.assert_array_equal(saved['membership'][chosen_nodes], gids)
    identity_codes = np.zeros(chosen.shape, np.int8)
    source_members = np.zeros(chosen.shape, np.int8)
    n_sources = np.zeros(chosen.shape, np.int8)
    provider_first = np.zeros(chosen.shape, bool)
    provider_src = np.full(chosen.shape, -1, np.int8)
    if used.any():
        sizes = np.diff(offsets)
        anchor_for_provider = anchors[np.repeat(candidate_groups, sizes)]
        same = identities[providers] == identities[anchor_for_provider]
        same_counts = np.add.reduceat(same.astype(np.int64), offsets[:-1])
        category = np.where(same_counts == sizes, 1, np.where(same_counts == 0, 2, 3))
        masks = np.bitwise_or.reduceat(1 << source_codes[providers], offsets[:-1])
        identity_codes[used] = category[chosen[used]]
        source_members[used] = masks[chosen[used]]
        n_sources[used] = np.where(masks[chosen[used]] == 3, 2, 1)
        provider_first[used] = chosen_nodes == anchors[gids]
        provider_src[used] = source_codes[chosen_nodes]
    times = saved['selected_interval'][..., :2][used]
    qs = np.broadcast_to(saved['query'], chosen.shape)[used]
    timing = np.where(times[:, 1] < qs, 2, np.where(times[:, 0] > qs, 3, 1))
    side = saved['selected_time_side']
    require(np.all(side[~used] == 0), 'C150 unselected interval has a time class')
    np.testing.assert_array_equal(side[used], timing, err_msg='C150 inclusive interval timing differs')
    selected = {key: saved[f'selected_{key}'][used] for key in CHOSEN_NUMBERS}
    require(all(np.isfinite(v).all() for v in selected.values()) and (selected['duration_seconds'] > 0.).all(),
            'C150 selected interval metadata is invalid')
    copies = np.diff(offsets)[chosen[used]]
    return (dict(time_side=side, provider_identity=identity_codes, provider_first=provider_first, provider_source=provider_src),
            dict(time_side=timing, provider_identity=identity_codes[used], provider_first=provider_first[used],
                 provider_source=provider_src[used], provider_source_mask=source_members[used],
                 distinct_sources=n_sources[used], provider_copies=copies, enabled=saved['enabled'][used], **selected,
                 speed_mps=selected['distance_metres']/selected['duration_seconds']))


def measure_scene(saved, evaluation):
    with np.load(evaluation, allow_pickle=False) as ev:
        np.testing.assert_array_equal(saved['node_ids'], ev['node_ids'])
        np.testing.assert_array_equal(saved['query'], ev['query_seconds'])
        identities, targets, truth, valid = (ev[key] for key in ('identities', 'target_ids', 'target_states', 'target_valid'))
    groups = group_members(saved['membership'])
    outputs = {name: emit_group_outputs(saved[key], [members.tolist() for members in groups], node_count=len(identities))
               for name, key in zip(ARMS, ('states', 'new_states'), strict=True)}
    reports, artifacts = evaluate_fusion_methods(outputs, prediction_speed_units={name: 'mps' for name in ARMS},
        node_identities=identities, target_ids=targets, target_states=truth, target_valid=valid,
        support=saved['support'], nearest_seconds=saved['nearest'], target_speed_unit='mps', split=SPLIT)
    base, new = (artifacts[name] for name in ARMS)
    for key in ARTIFACT_INVARIANTS:
        np.testing.assert_array_equal(new[key], base[key], err_msg=f'C150 common evaluator field changed: {key}')
    np.testing.assert_array_equal(new['aligned_prediction'][..., :3], base['aligned_prediction'][..., :3])
    for key in POPULATION_KEYS + ('unmatched_target_ids',):
        require(reports[ARMS[0]][key] == reports[ARMS[1]][key], f'C150 target/group population changed: {key}')
    all_coverage = {}
    for name in ARMS:
        artifact, report = artifacts[name], reports[name]
        masks = report['full_target_metrics']['all'].copy()
        for i, attr in enumerate(('latitude', 'longitude')):
            _, masks[attr] = _coverage(artifact['aligned_truth_valid'][..., i], artifact['aligned_prediction_valid'][..., i])
        all_coverage[name] = {attr: {key: int(masks[attr][key]) for key in COVERAGE_KEYS} for attr in ATTRIBUTES}
    require(all_coverage[ARMS[0]] == all_coverage[ARMS[1]], 'C150 four-attribute/position coverage changed')
    for region in ('observed', 'missing'):
        for attr in ('heading', 'speed', 'position'):
            first, second = (reports[name]['full_target_metrics'][region][attr] for name in ARMS)
            require(all(first[key] == second[key] for key in COVERAGE_KEYS), 'C150 regional coverage changed')

    n = len(targets)
    truth_valid, predicted_valid = (base[key][:n, :, 3] for key in ('aligned_truth_valid', 'aligned_prediction_valid'))
    t, q = np.nonzero(truth_valid & predicted_valid)
    assigned = base['aligned_group_indices'][:n]
    g = assigned[t]
    require((g >= 0).all() and len(g) == all_coverage[ARMS[0]]['heading']['intersection_count'],
            'C150 heading common point count differs')
    actual = base['aligned_truth'][t, q, 3]
    pure = np.asarray([np.all(identities[members] == identities[members[0]]) for members in groups], bool)
    group_metadata, selected = chosen_metadata(saved, groups, identities)
    vectors = dict(old_error=_wrap(base['aligned_prediction'][t, q, 3]-actual),
                   new_error=_wrap(new['aligned_prediction'][t, q, 3]-actual), pure=pure[g],
                   branch=saved['branch'][g, q], enabled=saved['enabled'][g, q],
                   **{key: value[g, q] for key, value in group_metadata.items()})
    np.testing.assert_array_equal(vectors['old_error'][~vectors['enabled']], vectors['new_error'][~vectors['enabled']])

    claimed = assigned >= 0
    branch = saved['branch']
    enabled, changed = saved['enabled'], saved['states'][..., 3] != saved['new_states'][..., 3]
    ref = truth_valid[claimed]
    selected_branch = branch[assigned[claimed]]
    counts = {}
    for i, name in enumerate(BRANCHES):
        mask, claimed_mask = branch == i, selected_branch == i
        counts[name] = dict(group_query_count=int(mask.sum()), common_points=int((vectors['branch'] == i).sum()),
                            enabled_group_queries=int((mask & enabled).sum()),
                            enabled_common_points=int(((vectors['branch'] == i) & vectors['enabled']).sum()),
                            numeric_changes=int((mask & changed).sum()),
                            claimed_queries_with_heading_reference=int((claimed_mask & ref).sum()),
                            claimed_queries_without_heading_reference=int((claimed_mask & ~ref).sum()),
                            non_primary_group_queries=int(mask.sum()-claimed_mask.sum()),
                            **{f'fallback_{key}_count': int((mask & (saved['fallback_reason'] == j)).sum())
                               for j, key in enumerate(REASONS)})
    population = {key: int(reports[ARMS[0]][key]) for key in POPULATION_KEYS}
    population.update(unclaimed_target_heading_reference_points=int(truth_valid[~claimed].sum()),
                      group_query_count=branch.size, selected_group_queries=len(selected['enabled']),
                      enabled_group_queries=int(enabled.sum()), numeric_changed_group_queries=int(changed.sum()),
                      enabled_common_points=int(vectors['enabled'].sum()),
                      numeric_changed_common_points=int(changed[g, q].sum()))
    public = dict(population=population, coverage=all_coverage[ARMS[0]], branch_counts=counts,
                  stats=heading_statistics(vectors, detailed=False))
    return public, vectors, selected


def distribution(values):
    return dict(count=len(values), min=float(values.min()) if len(values) else None,
                p50=float(np.percentile(values, 50.)) if len(values) else None,
                p95=float(np.percentile(values, 95.)) if len(values) else None,
                max=float(values.max()) if len(values) else None)


def selected_statistics(selected, *, detailed):
    def region(mask):
        return dict(count=int(mask.sum()), actual_enabled=int((mask & selected['enabled']).sum()),
            repeated_endpoint_provenance=int((mask & (selected['provider_copies'] > 1)).sum()),
            both_source_provenance=int((mask & (selected['distinct_sources'] == 2)).sum()),
            chosen_anchor_count=int((mask & selected['provider_first']).sum()),
            chosen_other_member_count=int((mask & ~selected['provider_first']).sum()),
            chosen_source_counts={name: int((mask & (selected['provider_source'] == i)).sum())
                                  for i, name in enumerate(SOURCE_NAMES)},
            donor_identity_provenance={name: int((mask & (selected['provider_identity'] == i)).sum())
                                       for i, name in enumerate(PROVENANCE) if i},
            **({key: distribution(selected[key][mask]) for key in (*CHOSEN_NUMBERS, 'speed_mps', 'provider_copies')}
               if detailed else {}))

    return {name: region(selected['time_side'] == i) for i, name in enumerate(TIME_SIDES) if i}


def aggregate(rows, points, selected, *, detailed):
    vectors = {key: np.concatenate([row[key] for row in points]) for key in points[0]}
    donors = {key: np.concatenate([row[key] for row in selected]) for key in selected[0]}

    def counts(items):
        result = Counter()
        for item in items:
            result.update(item)
        return dict(result)

    return dict(scene_count=len(rows), component_count=len({row['component'] for row in rows}),
                population=counts(row['population'] for row in rows),
                coverage={attr: counts(row['coverage'][attr] for row in rows) for attr in ATTRIBUTES},
                branch_counts={name: counts(row['branch_counts'][name] for row in rows) for name in BRANCHES},
                stats=heading_statistics(vectors, detailed=detailed),
                selected_intervals=selected_statistics(donors, detailed=detailed))


def verify_c149_controls(current, previous):
    control_pairs = [(name, current['stats'][name]['baseline'], previous[name]) for name in ('all', 'pure', 'mixed')]
    control_pairs.extend((name, current['stats']['branches'][name]['baseline'], previous['branches'][name])
                         for name in BRANCHES)
    for role, actual, old in control_pairs:
        for key in ('count', 'tail_gt150'):
            require(actual[key] == old[key], f'C149 baseline {role} {key} differs')
        for key in ('mae', 'rmse', 'p95', 'p99', 'bias', 'absolute_error_sum'):
            require(actual[key] == old[key] or (actual[key] is not None and old[key] is not None
                    and np.isclose(actual[key], old[key], atol=1e-9, rtol=2e-13)),
                    f'C149 baseline {role} {key} differs')
    for key in POPULATION_KEYS:
        require(current['population'][key] == previous['population'].get(key, 0), f'C149 {key} differs')
    for attr in ATTRIBUTES:
        for key in COVERAGE_KEYS:
            require(current['coverage'][attr][key] == previous['coverages'][attr].get(key, 0),
                    f'C149 {attr} {key} differs')
    for name in BRANCHES:
        for key in ('group_query_count', 'claimed_queries_without_heading_reference'):
            require(current['branch_counts'][name][key] == previous['branch_counts'][name].get(key, 0),
                    f'C149 {name} {key} differs')


def promotion(full, without, components, largest):
    gates = {}
    for scope, pool in (('full', full), ('without_largest', without)):
        metrics = pool['stats']
        main, non_tail = {}, {}
        for key in ('all', 'enabled'):
            row = metrics[key]
            main[key] = dict(mae=bool(row['count'] and row['delta_mae'] < -EPSILON),
                             p99=bool(row['count'] and row['delta_p99'] < -EPSILON),
                             tail_gt150=(row['candidate']['tail_gt150'] < row['baseline']['tail_gt150']))
        for key in ('pure', 'mixed'):
            delta = metrics[key]['old_non_tail']['mean_delta_absolute_error']
            non_tail[key] = None if delta is None else delta <= EPSILON
        gates[scope] = dict(main=main, original_non_tail=non_tail)
    changed_components, supporting = dict(improved=0, tied=0, worse=0, no_common_points=0, zero_enabled=0), []
    for component, record in components.items():
        all_region = record['stats']['all']
        activated = record['stats']['enabled']['count']
        delta = all_region['delta_mae']
        label = ('no_common_points' if delta is None else 'improved' if delta < -EPSILON else
                 'worse' if delta > EPSILON else 'tied')
        changed_components[label] += 1
        changed_components['zero_enabled'] += int(not activated)
        if component != largest and activated and label == 'improved' and (
                all_region['candidate']['tail_gt150'] <= all_region['baseline']['tail_gt150']):
            supporting.append(component)
    allowed = len(supporting) >= 2 and all(
        all(all(metric.values()) for metric in row['main'].values()) and
        all(flag is not False for flag in row['original_non_tail'].values()) for row in gates.values())
    return dict(passes=allowed, population_gates=gates, qualifying_components_without_largest=supporting,
                qualifying_components_gate=len(supporting) >= 2, component_mae_direction=changed_components)


def run(args, checkpoint):
    ids, components, index, partition, paths, parent, previous, largest = bind_c149(args.runs, args.manifest)
    private = args.output/'private'
    private.mkdir(mode=0o700)
    frozen, hashes = {}, previous['scene_hashes']
    with MTADArchive(paths) as archive:
        validate_measurement_archive(args.runs, archive)
        for number, sid in enumerate(ids, 1):
            frozen[sid] = freeze_one(archive, sid, index[sid], parent/'private'/f'{sid}.npz',
                                    partition/'private/blind'/f'{sid}.json.gz', hashes[sid], private/f'{sid}.npz')
            if number % 25 == 0:
                checkpoint('freeze', predictions=number)
        require([(item['scene_id'], item['kind']) for item in archive.members_read]
                == [(sid, 'measurements') for sid in ids], 'C150 may read only fit measurements')
    require(len(frozen) == len(ids) and len(list(private.glob('*.npz'))) == len(ids), 'C150 prediction freeze incomplete')
    freeze_path = args.output/'freeze.json'
    write(freeze_path, dict(fit_scene_count=len(ids), fit_component_count=len(set(components.values())),
          scene_order_sha256=previous['scene_order_sha256'], source_C149_code=C149_COMMIT,
          source_C149_metadata_sha256=C149_SHA256, all_predictions_frozen_before_evaluation=True,
          reference_reads_before_freeze=0, archive_measurement_reads=len(ids), scene_hashes=frozen))
    frozen = read(freeze_path)['scene_hashes']
    previous_summary = parent/'summary.json'
    require_hash(previous_summary, C149_SHA256['summary'], 'C149 original evaluation summary')
    original = read(previous_summary)
    require(original['largest_scene_component'] == largest, 'C149 removed component differs')
    metrics_dir = private/'paired'
    metrics_dir.mkdir()
    rows, vectors, selections, eval_hashes = [], [], [], {}
    groups = defaultdict(list)
    checkpoint('evaluate', predictions=len(ids), scored=0)
    for number, sid in enumerate(ids, 1):
        path = private/f'{sid}.npz'
        require_hash(path, frozen[sid]['prediction_sha256'], f'C150 {sid} frozen prediction')
        evaluation = index[sid]['evaluation']
        eval_hashes[sid] = file_hash(evaluation)
        with np.load(path, allow_pickle=False) as saved:
            row, points, selected = measure_scene(saved, evaluation)
        row.update(scene_id=sid, component=components[sid], selected_intervals=selected_statistics(selected, detailed=False))
        np.savez_compressed(metrics_dir/f'{sid}.npz', **points)
        groups[components[sid]].append(len(rows))
        rows.append(row)
        vectors.append(points)
        selections.append(selected)
        if number % 25 == 0:
            checkpoint('evaluate', scored=number)

    def pooled(indices, *, detailed):
        return aggregate([rows[i] for i in indices], [vectors[i] for i in indices], [selections[i] for i in indices],
                         detailed=detailed)

    full = pooled(range(len(ids)), detailed=True)
    without = pooled([i for i, row in enumerate(rows) if row['component'] != largest], detailed=True)
    verify_c149_controls(full, original['full'])
    verify_c149_controls(without, original['without_largest'])
    by_component = {key: pooled(indices, detailed=False) for key, indices in groups.items()}
    write(args.output/'scenes.json.gz', rows)
    write(args.output/'summary.json', dict(attempt='C150 fit paired readout only', full=full, without_largest=without,
          largest_scene_component=largest, components=by_component, evaluation_sha256=eval_hashes,
          invariants=dict(frozen_positions_speed_support_exact=True, common_masks_and_claimants_exact=True,
                          coverage_and_duplicates_exact=True, original_C149_aggregates_verified=True),
          decision=promotion(full, without, by_component, largest)))
    checkpoint('done', status='COMPLETED', exit_code=0, scored=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.runs, args.manifest, args.output = (path.resolve() for path in (args.runs, args.manifest, args.output))
    args.output.mkdir(exist_ok=False)
    start = time.monotonic()
    state = dict(status='RUNNING', argv=sys.argv, idea_id=IDEA, plan_version='1.0', split='C92 fit',
                 optimizer_updates=0, model_forwards=0, raw_reference_or_label_reads=0,
                 other_split_data_opened=False, predictions=0, scored=0,
                 started_at=datetime.now(timezone.utc).isoformat())

    def checkpoint(phase, **fields):
        state.update(fields, phase=phase)
        if state['status'] != 'RUNNING':
            state['finished_at'] = datetime.now(timezone.utc).isoformat()
        summary_state(args.output, start, **state)
        if state['status'] == 'RUNNING' and peak_rss_bytes() > RSS_LIMIT:
            raise MemoryError('C150 8 GiB resident checkpoint exceeded')

    try:
        checkpoint('bind', code=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        run(args, checkpoint)
    except BaseException as exc:
        checkpoint('failed', status='FAILED', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
