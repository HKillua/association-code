"""Prepare explicit development clocks and population counts, one scene at a time.

The offset rule is the minimum labelled-reference time. Explicit policies check
either historical source-9001 exact states or Level1 temporal support; neither
optimizes offsets against model errors.
No model, normalization, test payload or persistent array cache is involved.
"""
from collections import Counter, defaultdict
from contextlib import closing
import math
import time

import numpy as np

from association.data.archive import DataError, MTADArchive, validate_manifest
from association.data.legacy_dataset import validate_archive
from association.data.native import _record, uniform_observation_indices
from association.data.reference_states import iter_reference_states
from association.data.targets import build_scene_fusion_targets
from association.data.timed import prepare_timed_scene


DATA = dict(length=64, max_observations=128, time_unit_seconds=20.)
MAX_GAP_SECONDS = 600.
EXACT_CLOCK_POLICY = 'min_reference_time_exact_state_v1'
TEMPORAL_CLOCK_POLICY = 'min_reference_time_temporal_support_v1'
CLOCK_POLICIES = (EXACT_CLOCK_POLICY, TEMPORAL_CLOCK_POLICY)
TIME_TOLERANCE_SECONDS = 1e-6
INTERVAL_LIMITS = np.array([10., 20., 30., 60., 120., 300., 600.])
INTERVAL_NAMES = ('le_10s', 'gt_10_le_20s', 'gt_20_le_30s', 'gt_30_le_60s',
                  'gt_60_le_120s', 'gt_120_le_300s', 'gt_300_le_600s', 'gt_600s')


def observation_profile(scene):
    """Count actual and capped sampling intervals without references or fitting.

    Interval histograms weight each interval; query counts weight each fragment
    query. Do not confuse either with an equal-scene or equal-target statistic.
    This runs even when reference clock verification fails.
    """
    sources = {}
    if not scene.tracks:
        return sources
    query = np.linspace(min(t.times[0] for t in scene.tracks),
                        max(t.times[-1] for t in scene.tracks), DATA['length'])
    for track in scene.tracks:
        raw = np.asarray(track.times, dtype=np.float64)
        selected = raw[uniform_observation_indices(len(raw), DATA['max_observations'])]
        row = sources.setdefault(track.source, Counter())
        row.update(fragments=1, canonical_points=len(raw), selected_points=len(selected),
                   capped_fragments=int(len(raw) > len(selected)),
                   selected_bidirectional_transitions=2*(len(selected)-1),
                   fragment_queries=len(query),
                   supported_queries=int(((query >= raw[0]) & (query <= raw[-1])).sum()))
        for name, times in (('canonical', raw), ('selected', selected)):
            gaps = np.diff(times)
            counts = np.bincount(np.searchsorted(INTERVAL_LIMITS, gaps, side='left'),
                                 minlength=len(INTERVAL_NAMES))
            row.update({name+'_interval_'+key: int(count) for key, count in zip(INTERVAL_NAMES, counts)})
            row[name+'_interval_count'] += len(gaps)
            row[name+'_interval_seconds_sum'] += float(gaps.sum())
            row[name+'_interval_seconds_max'] = max(row[name+'_interval_seconds_max'], float(gaps.max()))
            right = np.searchsorted(times, query).clip(0, len(times)-1)
            left = (np.searchsorted(times, query)-1).clip(0, len(times)-1)
            nearest = np.minimum(np.abs(query-times[left]), np.abs(query-times[right]))
            row[name+'_exact_queries'] += int((nearest == 0).sum())
            row[name+'_nearest_gt_300s_queries'] += int((nearest > 300).sum())
            row[name+'_nearest_gt_1800s_queries'] += int((nearest > 1800).sum())
    return {source: dict(counts) for source, counts in sources.items()}


def _combine_profiles(rows):
    result = {}
    for row in rows:
        for source, stats in row['sampling'].items():
            merged = result.setdefault(source, {})
            for key, value in stats.items():
                merged[key] = (max(merged.get(key, 0), value) if key.endswith('_max')
                               else merged.get(key, 0)+value)
    for stats in result.values():
        stats['point_retention_fraction'] = stats['selected_points']/stats['canonical_points']
        stats['query_support_fraction'] = stats['supported_queries']/stats['fragment_queries']
        for name in ('canonical', 'selected'):
            stats[name+'_mean_interval_seconds'] = (stats[name+'_interval_seconds_sum'] /
                                                    stats[name+'_interval_count'])
    return result


class _ClockArchive(MTADArchive):
    """Keep the current scene's raw measurements for an independent clock check.

    load_scene still performs all existing canonicalization. Capturing its raw
    rows avoids reading/parsing the same measurement CSV a second time.
    """
    def _rows(self, scene_id, kind):
        rows = super()._rows(scene_id, kind)
        if kind == 'measurements':
            self.raw_measurements = rows
        return rows


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _temporal_clock_evidence(scene, labels, times_by_identity, offset):
    """Check a fixed offset using times only; never edit or select model inputs."""
    labels_by_node = defaultdict(list)
    for row in labels:
        labels_by_node[row['source'], row['batch']].append(row)
    times_by_identity = {identity: np.asarray(sorted(times), dtype=np.float64)
                         for identity, times in times_by_identity.items()}
    matches, by_source, violations = Counter(), defaultdict(Counter), []
    lower, upper = -math.inf, math.inf
    tolerance = TIME_TOLERANCE_SECONDS
    for track in scene.tracks:
        row = by_source[track.source]
        row['valid_fragments'] += 1
        entries = labels_by_node.get((track.source, track.batch), [])
        reasons = []
        start, end = float(track.times[0]), float(track.times[-1])
        if (not entries or {item['mmsi'].rsplit('-', 1)[0] for item in entries} != {track.identity_id}
                or abs(min(item['t_s'] for item in entries) - start) > tolerance
                or abs(max(item['t_e'] for item in entries) - end) > tolerance):
            row['label_boundary_mismatch_fragments'] += 1
            reasons.append('label_measurement_bounds_disagree')
        else:
            row['label_boundary_match_fragments'] += 1
        reference_times = times_by_identity.get(track.identity_id)
        if reference_times is None or not len(reference_times):
            row['missing_reference_time_fragments'] += 1
            reasons.append('no_matching_identity_reference_times')
        else:
            lower = max(lower, float(reference_times[0]) - start)
            upper = min(upper, float(reference_times[-1]) - end)
            shifted = np.array([start + offset, end + offset])
            if (shifted[0] < reference_times[0] - tolerance
                    or shifted[1] > reference_times[-1] + tolerance):
                row['outside_reference_bounds_fragments'] += 1
                reasons.append('fixed_offset_outside_reference_bounds')
            else:
                row['inside_reference_bounds_fragments'] += 1
            right = np.searchsorted(reference_times, shifted).clip(0, len(reference_times)-1)
            left = (np.searchsorted(reference_times, shifted)-1).clip(0, len(reference_times)-1)
            exact = np.minimum(abs(reference_times[left]-shifted), abs(reference_times[right]-shifted)) <= tolerance
            row['reference_timestamp_endpoint_matches'] += int(exact.sum())
            row['fragments_with_reference_timestamp_endpoint'] += int(exact.any())
        if reasons:
            violations.append(dict(node_id=track.node_id, reasons=reasons))
    for row in by_source.values():
        matches.update(row)
    supported = (bool(scene.tracks) and not violations
                 and matches['reference_timestamp_endpoint_matches'] > 0)
    return dict(status='CLOCK_HYPOTHESIS_SUPPORTED' if supported else 'CLOCK_UNRESOLVED',
        time_offset_seconds=offset, clock_policy=TEMPORAL_CLOCK_POLICY, matches=dict(matches),
        by_source={source: dict(counts) for source, counts in by_source.items()},
        feasible_offset_range_seconds=[lower, upper] if math.isfinite(lower) and math.isfinite(upper) else None,
        violations=violations, time_tolerance_seconds=tolerance,
        interpretation='Consistency with the fixed min-reference-time rule, not a unique clock estimate; '
            'reference coordinates and model errors are not used.')


def derive_scene_alignment(scene, labels, measurements, reference, *, clock_policy=EXACT_CLOCK_POLICY):
    """Return the fixed clock hypothesis and explicit policy evidence.

    Exact-state policy is retained for historical reproduction. Temporal policy
    allows noisy coordinates and asynchronous times, checks every valid fragment,
    and requires at least one reference-time endpoint match within the scene.
    Labels are used only to validate the supervision clock, never as predictions.
    """
    if clock_policy not in CLOCK_POLICIES:
        raise DataError('Unknown clock policy: ' + str(clock_policy))
    if reference['split'] not in ('T_core', 'V_select') or reference['scene_id'] != scene.scene_id:
        raise DataError('Clock preparation requires a matching train/validation scene; T_test is sealed')
    identities = {row['mmsi'].rsplit('-', 1)[0] for row in labels}
    times, by_state, by_time, times_by_identity = [], defaultdict(set), defaultdict(list), defaultdict(set)
    for row in reference['rows']:
        if row['identity_id'] not in identities or not _finite(row['time']):
            continue
        times.append(row['time'])
        times_by_identity[row['identity_id']].add(row['time'])
        if clock_policy == EXACT_CLOCK_POLICY and all(_finite(value) for value in row['state']):
            state = tuple(row['state'])
            by_state[row['identity_id'], state].add(row['time'])
            by_time[row['identity_id'], row['time']].append(state)
    if not times:
        return dict(status='NO_LABELLED_REFERENCE_TIMES', time_offset_seconds=None,
                    clock_policy=clock_policy, matches={})
    offset = float(min(times))
    if clock_policy == TEMPORAL_CLOCK_POLICY:
        return _temporal_clock_evidence(scene, labels, times_by_identity, offset)
    valid9001 = {(track.source, track.batch): track.identity_id for track in scene.tracks if track.source == '9001'}
    matches, deltas = Counter(), Counter()
    for row in measurements:
        identity = valid9001.get((row['source'], row['batch']))
        if identity is None:
            continue
        timestamp = float(row['time'])
        values = tuple(float(row[key]) for key in ('lat', 'lon', 'vel', 'cou'))
        matches['raw_points_on_valid_source9001_nodes'] += 1
        state_times = by_state.get((identity, values), set())
        if len(state_times) == 1:
            implied = next(iter(state_times)) - timestamp
            deltas[implied] += 1
            matches['unique_state_match_points'] += 1
            matches['unique_state_match_agrees_fixed_offset' if implied == offset
                    else 'unique_state_match_disagrees_fixed_offset'] += 1
        else:
            matches['multiple_time_state_match_points' if state_times else 'no_exact_state_match_points'] += 1
        candidates = by_time.get((identity, timestamp + offset), [])
        if len(candidates) == 1:
            matches['unique_reference_time_match_points'] += 1
            matches['fixed_offset_raw_four_exact_points' if candidates[0] == values
                    else 'fixed_offset_raw_four_mismatch_points'] += 1
        else:
            matches['ambiguous_reference_time_points' if candidates else 'no_reference_time_points'] += 1
    supported = matches['unique_state_match_points'] > 0 and not matches['unique_state_match_disagrees_fixed_offset']
    return dict(status='CLOCK_HYPOTHESIS_SUPPORTED' if supported else 'CLOCK_UNRESOLVED',
        time_offset_seconds=offset, clock_policy=clock_policy, matches=dict(matches),
        implied_offsets_from_unique_exact_states={str(key): value for key, value in sorted(deltas.items())})


def _population(scene, native, prepared, targets, labels, reference):
    counts = np.zeros(4, dtype=np.int64)
    for target in targets.values():
        counts += target['valid'].sum(0)
    return dict(valid_fragments=len(scene.tracks), quarantined_fragments=scene.audit['quarantined_nodes'],
        raw_measurement_rows=scene.audit['raw_measurement_rows'], label_rows=len(labels),
        canonical_points=native['audit']['native_inputs']['canonical_points'],
        selected_native_points=int(prepared['native_observed'].sum()),
        candidate_pairs=len(prepared['pairs']), positive_pairs=int(prepared['labels'].sum()),
        negative_pairs=len(prepared['pairs'])-int(prepared['labels'].sum()),
        target_identities=len(targets), target_queries=len(targets)*DATA['length'],
        target_without_input_count=len(set(targets)-set(prepared['identities'])),
        target_without_reference_count=sum(item['report']['missing_identity'] for item in targets.values()),
        reference_rows=len(reference['rows']), reference_issue_count=len(reference['issues']),
        reference_valid_points_per_feature=counts.tolist(),
        fragments_without_query_support=prepared['timed_input_audit']['fragments_without_query_support'],
        fragment_sources=dict(Counter(track.source for track in scene.tracks)))


def prepare_scene_data(archive_path, manifest, existing_alignment=None, *, progress=None,
                       clock_policy=EXACT_CLOCK_POLICY):
    """Read exactly T_core/V_select; emit clocks and aggregate population metadata.

    Failures stay in the report; unsupported clocks never enter the output map.
    Existing clocks must agree exactly, including the fixed 600-second gap.
    One current scene's raw/canonical/target arrays are discarded before moving
    on. No training statistics or model inputs are written to disk.
    """
    validate_manifest(manifest)
    if clock_policy not in CLOCK_POLICIES:
        raise DataError('Unknown clock policy: ' + str(clock_policy))
    existing_alignment = {} if existing_alignment is None else existing_alignment
    requested = [(split, sid) for split in ('T_core', 'V_select') for sid in manifest['splits'][split]]
    allowed = {sid for _, sid in requested}
    role_by_scene = {sid: split for split, sid in requested}
    if not isinstance(existing_alignment, dict) or not set(existing_alignment).issubset(allowed):
        raise DataError('Every historical alignment scene must remain in requested train/validation')
    started = time.monotonic()
    alignment, rows, failures, clocks, accesses = {}, [], [], Counter(), Counter()
    existing_checked, offsets, scene_clocks = [], [], {}
    with _ClockArchive(archive_path) as archive:
        validate_archive(archive, manifest)
        for split in ('T_core', 'V_select'):
            ids = manifest['splits'][split]
            if not ids:
                continue
            with closing(iter_reference_states(archive_path, manifest, split=split, scene_ids=ids)) as references:
                for sid in ids:
                    # This loop supplies only manifest-listed train/V IDs to both
                    # readers. Reference reader independently enforces the role.
                    scene = archive.load_scene(sid)
                    sampling = observation_profile(scene)
                    labels = archive.read_labels(sid)
                    reference = next(references)
                    if reference['scene_id'] != sid:
                        raise DataError('Reference scene order changed')
                    accesses[split + ':reference'] += 1
                    clock = derive_scene_alignment(scene, labels, archive.raw_measurements, reference,
                                                   clock_policy=clock_policy)
                    scene_clocks[sid] = clock
                    clocks.update(clock['matches'])
                    policy = dict(time_offset_seconds=clock['time_offset_seconds'], max_gap_seconds=MAX_GAP_SECONDS)
                    mismatch = sid in existing_alignment and existing_alignment[sid] != policy
                    if clock['status'] != 'CLOCK_HYPOTHESIS_SUPPORTED' or mismatch:
                        failures.append(dict(scene_id=sid, split=split, clock=clock,
                            historical_alignment=existing_alignment.get(sid), historical_mismatch=mismatch))
                        rows.append(dict(scene_id=sid, split=split, status='FAILED', sampling=sampling))
                    else:
                        native = _record(scene, manifest['components'][sid], split, DATA['max_observations'],
                                         split_unit=manifest.get('split_unit'))
                        query = None
                        if not scene.tracks:
                            if not labels or max(row['t_e'] for row in labels) <= min(row['t_s'] for row in labels):
                                raise DataError(f'Scene {sid}: no valid empty-scene query bounds')
                            query = np.linspace(min(row['t_s'] for row in labels), max(row['t_e'] for row in labels), DATA['length'])
                        prepared = prepare_timed_scene(native, **DATA, query_times=query)
                        target_ids = list(dict.fromkeys(row['mmsi'].rsplit('-', 1)[0] for row in labels))
                        targets = build_scene_fusion_targets(prepared, reference, target_identity_ids=target_ids,
                            longitude_policy='shortest_arc', **policy)
                        population = _population(scene, native, prepared, targets, labels, reference)
                        rows.append(dict(scene_id=sid, split=split, status='COMPLETED',
                                         sampling=sampling, **population))
                        alignment[sid] = policy
                        offsets.append(policy['time_offset_seconds'])
                        if sid in existing_alignment:
                            existing_checked.append(sid)
                        del native, prepared, targets
                    del scene, labels, reference
                    archive.raw_measurements = None
                    archive._label_cache.clear()
                    if progress and (len(rows) == 8 or len(rows) % 64 == 0 or len(rows) == len(requested)):
                        elapsed = time.monotonic()-started
                        progress(dict(scenes_processed=len(rows), requested_scenes=len(requested),
                            supported_scenes=len(alignment), failed_scenes=len(failures), elapsed_seconds=elapsed,
                            projected_total_seconds=elapsed/len(rows)*len(requested)))
        for row in archive.members_read:
            # No source row/state is retained in this metadata report.
            split = role_by_scene[row['scene_id']]
            accesses[split + ':' + row['kind']] += 1
    population = {}
    for split in ('T_core', 'V_select'):
        selected = [row for row in rows if row['split'] == split and row['status'] == 'COMPLETED']
        totals, sources, valid = Counter(), Counter(), np.zeros(4, dtype=np.int64)
        for row in selected:
            totals.update({key: value for key, value in row.items() if isinstance(value, int)})
            sources.update(row['fragment_sources'])
            valid += row['reference_valid_points_per_feature']
        sizes = [row['valid_fragments'] for row in selected]
        population[split] = dict(requested_scenes=len(manifest['splits'][split]), prepared_scenes=len(selected),
            actual_shared_a_components=len({manifest['components'][sid] for sid in manifest['splits'][split]}),
            totals=dict(totals), fragment_sources=dict(sources), reference_valid_points_per_feature=valid.tolist(),
            empty_input_scene_ids=[row['scene_id'] for row in selected if not row['valid_fragments']],
            single_scene_maxima={key: max((row[key] for row in selected), default=0) for key in (
                'valid_fragments', 'canonical_points', 'selected_native_points', 'candidate_pairs',
                'positive_pairs', 'negative_pairs', 'target_identities', 'raw_measurement_rows', 'reference_rows')},
            fragment_count_quantiles=dict(zip(('min', 'median', 'p95', 'max'),
                np.quantile(sizes, [0, .5, .95, 1]).tolist())) if sizes else None)
        population[split]['sampling_by_source'] = _combine_profiles(
            [row for row in rows if row['split'] == split])
        population[split]['tcm_ordered_batches_per_epoch_at_batch64'] = (
            math.ceil(sum(sizes)/64) if len(selected) == len(manifest['splits'][split]) else None)
        population[split]['sampling_scope'] = 'all requested scenes, including unresolved reference clocks'
    summary = dict(status='COMPLETED' if not failures and len(alignment) == len(requested) else 'FAILED',
        dataset_level=manifest.get('dataset_level'),
        split_unit=manifest.get('split_unit', 'shared_a_component'), requested_scenes=len(requested),
        prepared_scenes=len(alignment), population=population, failures=failures,
        historical_alignment_scenes=len(existing_alignment), historical_alignment_verified=len(existing_checked),
        historical_alignment_verified_scene_ids=existing_checked,
        nonzero_offset_scenes=sum(offset != 0 for offset in offsets),
        max_offset_seconds=max(offsets, default=None), clock_matches=dict(clocks), payload_reads=dict(accesses),
        clock_policy=clock_policy, scene_clocks=scene_clocks,
        test_payload_reads=0, test_opened=False, data_config={**DATA, 'reference_longitude_policy': 'shortest_arc'},
        max_gap_seconds=MAX_GAP_SECONDS, elapsed_seconds=time.monotonic()-started,
        offset_policy='min finite raw reference time among all label-named A-B identities; raw_reference_time - offset',
        clock_check=('all unique exact raw four-state matches on valid source9001 nodes must imply that fixed offset; at least one required'
            if clock_policy == EXACT_CLOCK_POLICY else
            'all valid fragments must match label boundaries and fit same-identity reference time bounds after the fixed offset; '
            'at least one fragment endpoint must match a reference timestamp within 1e-6 seconds'),
        limits='Clock evidence supports the specified offset rule, not a unique clock estimate or independent sensor denoising. '
            'Shared-A components may cross scene splits. No prediction-error search, input correction or label-based prediction.',
        arrays_saved=False, normalizer_fitted=False, optimizer_updates=0)
    return alignment, summary
