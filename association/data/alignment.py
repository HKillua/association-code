"""Prepare explicit development clocks and population counts, one scene at a time.

The offset rule is the previously verified minimum labelled-reference time.
Raw source-9001 exact state matches independently check it, never optimize it.
No model, normalization, test payload or persistent array cache is involved.
"""
from collections import Counter, defaultdict
from contextlib import closing
import math
import time

import numpy as np

from association.data.archive import DataError, MTADArchive, validate_manifest
from association.data.legacy_dataset import validate_archive
from association.data.native import _record
from association.data.reference_states import iter_reference_states
from association.data.targets import build_scene_fusion_targets
from association.data.timed import prepare_timed_scene


DATA = dict(length=64, max_observations=128, time_unit_seconds=20.)
MAX_GAP_SECONDS = 600.


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


def derive_scene_alignment(scene, labels, measurements, reference):
    """Return the fixed clock hypothesis and independent raw-match evidence.

    Missing/ambiguous exact states do not invent offsets. A scene is supported
    only with at least one unique state match and no contradictory unique match.
    No coordinates, predictions or losses are minimized to choose the offset.
    """
    if reference['split'] not in ('T_core', 'V_select') or reference['scene_id'] != scene.scene_id:
        raise DataError('Clock preparation requires a matching train/validation scene; T_test is sealed')
    identities = {row['mmsi'].rsplit('-', 1)[0] for row in labels}
    times, by_state, by_time = [], defaultdict(set), defaultdict(list)
    for row in reference['rows']:
        if row['identity_id'] not in identities or not _finite(row['time']):
            continue
        times.append(row['time'])
        if all(_finite(value) for value in row['state']):
            state = tuple(row['state'])
            by_state[row['identity_id'], state].add(row['time'])
            by_time[row['identity_id'], row['time']].append(state)
    if not times:
        return dict(status='NO_LABELLED_REFERENCE_TIMES', time_offset_seconds=None, matches={})
    offset = float(min(times))
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
        time_offset_seconds=offset, matches=dict(matches),
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


def prepare_scene_data(archive_path, manifest, existing_alignment, *, progress=None):
    """Read exactly T_core/V_select; emit clocks and aggregate population metadata.

    Failures stay in the report; unsupported clocks never enter the output map.
    Existing clocks must agree exactly, including the fixed 600-second gap.
    One current scene's raw/canonical/target arrays are discarded before moving
    on. No training statistics or model inputs are written to disk.
    """
    validate_manifest(manifest)
    requested = [(split, sid) for split in ('T_core', 'V_select') for sid in manifest['splits'][split]]
    allowed = {sid for _, sid in requested}
    role_by_scene = {sid: split for split, sid in requested}
    if not isinstance(existing_alignment, dict) or not set(existing_alignment).issubset(allowed):
        raise DataError('Every historical alignment scene must remain in requested train/validation')
    started = time.monotonic()
    alignment, rows, failures, clocks, accesses = {}, [], [], Counter(), Counter()
    existing_checked, offsets = [], []
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
                    labels = archive.read_labels(sid)
                    reference = next(references)
                    if reference['scene_id'] != sid:
                        raise DataError('Reference scene order changed')
                    accesses[split + ':reference'] += 1
                    clock = derive_scene_alignment(scene, labels, archive.raw_measurements, reference)
                    clocks.update(clock['matches'])
                    policy = dict(time_offset_seconds=clock['time_offset_seconds'], max_gap_seconds=MAX_GAP_SECONDS)
                    mismatch = sid in existing_alignment and existing_alignment[sid] != policy
                    if clock['status'] != 'CLOCK_HYPOTHESIS_SUPPORTED' or mismatch:
                        failures.append(dict(scene_id=sid, split=split, clock=clock,
                            historical_alignment=existing_alignment.get(sid), historical_mismatch=mismatch))
                        rows.append(dict(scene_id=sid, split=split, status='FAILED'))
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
                        rows.append(dict(scene_id=sid, split=split, status='COMPLETED', **population))
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
    summary = dict(status='COMPLETED' if not failures and len(alignment) == len(requested) else 'FAILED',
        split_unit=manifest.get('split_unit', 'shared_a_component'), requested_scenes=len(requested),
        prepared_scenes=len(alignment), population=population, failures=failures,
        historical_alignment_scenes=len(existing_alignment), historical_alignment_verified=len(existing_checked),
        historical_alignment_verified_scene_ids=existing_checked,
        nonzero_offset_scenes=sum(offset != 0 for offset in offsets),
        max_offset_seconds=max(offsets, default=None), clock_matches=dict(clocks), payload_reads=dict(accesses),
        test_payload_reads=0, test_opened=False, data_config={**DATA, 'reference_longitude_policy': 'shortest_arc'},
        max_gap_seconds=MAX_GAP_SECONDS, elapsed_seconds=time.monotonic()-started,
        offset_policy='min finite raw reference time among all label-named A-B identities; raw_reference_time - offset',
        clock_check='all unique exact raw four-state matches on valid source9001 nodes must imply that fixed offset; at least one required',
        limits='Conditional exact-state matches support the clock, not independent sensor denoising. Shared-A components may cross scene splits. No prediction-error search.',
        arrays_saved=False, normalizer_fitted=False, optimizer_updates=0)
    return alignment, summary
