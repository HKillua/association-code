"""108: fixed T32 input-dependency diagnostic, with no model/reference access.

Implements IDEA-PAIR-REP-DEPENDENCY-20261028 plan 1.1. These four explicit
representation controls are not an author implementation or a learned result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SEED = 20261028
SCENE_COUNT = 32
ANCHORS_PER_SOURCE = 8
WINDOW_POINTS = 16
TOLERANCE = 1e-8
MAX_MEMBER_BYTES = 64 * 1024**2
MAX_ADDRESS_SPACE_BYTES = 8 * 1024**3
FIELDS = ('lat', 'lon', 'vel', 'cou')
VERSIONS = ('literal', 'stats', 'pointwise', 'rawB')
NATIVE_EPS_FACTOR = 64
PUBLIC_OUTPUTS = ('run.json', 'summary.json', 'scenes.json', 'read-members.json',
                  'member-inventory.json', 'progress.jsonl')
COMPARISON_CONTRACT = dict(
    normalized_absolute_tolerance=TOLERANCE,
    normalized_components='u in all four versions; entire pointwise normalized difference',
    native_component_rule='abs(x-y) <= 64*float64_eps*max(1,abs(x),abs(y)) componentwise',
    native_epsilon_factor=NATIVE_EPS_FACTOR, float64_epsilon=sys.float_info.epsilon,
    native_components=dict(stats='m and r, separately by attribute', rawB='aligned B_q, by attribute'),
    native_units=dict(lat='degrees', lon='degrees', vel='native MTAD velocity, no conversion',
                      cou='scalar degrees in printed-formula control'),
    meaning='Floating-point comparison only; not physical effect size, usefulness or learned identity')
COUNT_KEYS = ('available_anchor_nodes', 'unselected_anchor_nodes', 'selected_anchors',
    'anchors_shorter_than_window', 'anchors_with_window', 'cross_source_candidate_pairs',
    'outside_support_pairs', 'eligible_pairs', 'positive_eligible_pairs', 'negative_eligible_pairs',
    'literal_pair_error_gt_tolerance', 'anchors_without_eligible_candidates', 'anchors_only_positive',
    'anchors_only_negative', 'anchors_mixed', 'mixed_anchors_literal_cross_label_collision')
VERSION_COUNT_KEYS = ('anchors_with_eligible_candidates', 'anchors_all_candidates_within_tolerance',
    'mixed_anchors', 'mixed_anchors_any_exact_content_change', 'mixed_anchors_any_change_gt_tolerance')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                   indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def address_space_limit():
    if not sys.platform.startswith('linux'):
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    selected = min([MAX_ADDRESS_SPACE_BYTES] +
                   [x for x in (soft, hard) if x != resource.RLIM_INFINITY])
    resource.setrlimit(resource.RLIMIT_AS, (selected, hard))
    return selected


def _finite(array):
    import numpy as np
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite diagnostic computation')
    return array


def normalize(values):
    """Float64 column minmax; exactly zero range maps to zero, never drops a column."""
    import numpy as np
    values = _finite(np.asarray(values, dtype=np.float64))
    if values.shape != (WINDOW_POINTS, len(FIELDS)):
        raise ValueError('Expected one complete 16-by-4 window')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        minimum = values.min(axis=0)
        span = values.max(axis=0) - minimum
        unit = np.zeros_like(values)
        np.divide(values - minimum, span, out=unit, where=span != 0)
    return _finite(unit), _finite(minimum), _finite(span)


def pair_representations(anchor, candidate):
    """Only B-dependent portions are returned; A/q/validity are identical in every pair."""
    import numpy as np
    a = _finite(np.asarray(anchor, dtype=np.float64))
    b = _finite(np.asarray(candidate, dtype=np.float64))
    if a.shape != (WINDOW_POINTS, len(FIELDS)) or b.shape != a.shape:
        raise ValueError('Expected aligned complete 16-by-4 pair')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        difference = _finite(a - b.mean(axis=0))
        unit, minimum, span = normalize(difference)
        pointwise, _, _ = normalize(_finite(a - b))
        reconstructed = _finite(unit * span + minimum)
        absolute = _finite(np.max(np.abs(reconstructed - difference), axis=0))
        # A relative-error denominator of max(1, |d|) avoids inventing infinities
        # at d=0. Absolute errors remain available in the original attribute units.
        relative = _finite(np.max(np.abs(reconstructed - difference) /
                                  np.maximum(1., np.abs(difference)), axis=0))
        mean_error = _finite(np.abs((a - reconstructed).mean(axis=0) - b.mean(axis=0)))
        scale = np.maximum(1., np.max(np.abs(difference), axis=0))
        near_constant = (span > 0) & (span <= 64 * np.finfo(np.float64).eps * scale)
    return dict(payloads=dict(literal=unit.ravel(),
        stats=np.concatenate((unit.ravel(), minimum, span)),
        pointwise=pointwise.ravel(), rawB=np.concatenate((unit.ravel(), b.ravel()))),
        zero_range=span == 0, near_constant_range=near_constant,
        reconstruction_absolute=absolute, reconstruction_relative=relative,
        candidate_mean_reconstruction_absolute=mean_error,
        literal_error=float(np.max(np.abs(unit - normalize(a)[0]))))


def interpolate_candidate(track, query):
    """No extrapolation; unwrap only heading before interpolation, then map to [0,360)."""
    import numpy as np
    if query[0] < track.times[0] or query[-1] > track.times[-1]:
        return None
    result = np.column_stack([np.interp(query, track.times, track.values[:, i])
                              for i in range(3)] +
        [np.rad2deg(np.interp(query, track.times,
                             np.unwrap(np.deg2rad(track.values[:, 3])))) % 360.])
    return _finite(result)


def _distribution(values):
    import numpy as np
    if not values:
        return dict(count=0, min=None, p50=None, p95=None, p99=None, max=None)
    a = _finite(np.asarray(values, dtype=np.float64))
    return dict(count=len(a), min=float(a.min()),
        p50=float(np.quantile(a, .5)), p95=float(np.quantile(a, .95)),
        p99=float(np.quantile(a, .99)), max=float(a.max()))


def _envelope_update(envelopes, label, payloads):
    import numpy as np
    for version, payload in payloads.items():
        old = envelopes[version].get(label)
        envelopes[version][label] = ((payload.copy(), payload.copy()) if old is None else
                                    (np.minimum(old[0], payload), np.maximum(old[1], payload)))


def _component_comparison(version, left, right):
    """Keep normalized and native components on their separately declared scales."""
    import numpy as np
    normalized_size = WINDOW_POINTS * len(FIELDS)
    with np.errstate(over='raise', invalid='raise'):
        delta = _finite(np.abs(left - right))
        normalized_max = float(delta[:normalized_size].max())
        exact = bool(np.any(delta != 0))
        exceeds = bool(np.any(delta[:normalized_size] > TOLERANCE))
        native_max = None
        if version in ('stats', 'rawB'):
            a = left[normalized_size:].reshape(-1, len(FIELDS))
            b = right[normalized_size:].reshape(-1, len(FIELDS))
            native_delta = delta[normalized_size:].reshape(a.shape)
            tolerance = _finite(NATIVE_EPS_FACTOR * np.finfo(np.float64).eps *
                               np.maximum(1., np.maximum(np.abs(a), np.abs(b))))
            exceeds = exceeds or bool(np.any(native_delta > tolerance))
            native_max = {field: float(native_delta[:, i].max()) for i, field in enumerate(FIELDS)}
    return dict(normalized_max_difference=normalized_max,
        native_max_difference_by_attribute=native_max,
        any_exact_change=exact, any_change_gt_tolerance=exceeds)


def _new_maxima():
    return {version: dict(all_candidates_normalized_max_difference=None,
        all_candidates_native_max_difference_by_attribute=None,
        cross_label_normalized_max_difference=None,
        cross_label_native_max_difference_by_attribute=None) for version in VERSIONS}


def _merge_maxima(target, values):
    for version in VERSIONS:
        for key in target[version]:
            incoming = values[version][key]
            if incoming is None:
                continue
            old = target[version][key]
            if isinstance(incoming, dict):
                target[version][key] = {field: max(incoming[field], old[field] if old else 0.)
                                        for field in FIELDS}
            else:
                target[version][key] = incoming if old is None else max(old, incoming)


def _version_variation(envelopes):
    import numpy as np
    result = {}
    for version in VERSIONS:
        groups = envelopes[version]
        minimum = np.minimum.reduce([item[0] for item in groups.values()])
        maximum = np.maximum.reduce([item[1] for item in groups.values()])
        overall = _component_comparison(version, maximum, minimum)
        cross_normalized, cross_native, cross_exact, cross_exceeds = None, None, None, None
        if set(groups) == {False, True}:
            pmin, pmax = groups[True]
            nmin, nmax = groups[False]
            comparisons = [_component_comparison(version, pmax, nmin),
                           _component_comparison(version, nmax, pmin)]
            cross_normalized = max(x['normalized_max_difference'] for x in comparisons)
            cross_exact = any(x['any_exact_change'] for x in comparisons)
            cross_exceeds = any(x['any_change_gt_tolerance'] for x in comparisons)
            if version in ('stats', 'rawB'):
                cross_native = {field: max(x['native_max_difference_by_attribute'][field]
                                          for x in comparisons) for field in FIELDS}
        result[version] = dict(
            all_candidates_normalized_max_difference=overall['normalized_max_difference'],
            all_candidates_native_max_difference_by_attribute=overall['native_max_difference_by_attribute'],
            all_candidates_within_tolerance=not overall['any_change_gt_tolerance'],
            cross_label_normalized_max_difference=cross_normalized,
            cross_label_native_max_difference_by_attribute=cross_native,
            cross_label_any_exact_content_change=cross_exact,
            cross_label_any_change_gt_tolerance=cross_exceeds)
    return result


def _has_literal_collision(positive, negative):
    import numpy as np
    if not positive or not negative:
        return False
    # Chunked comparisons avoid a candidates-by-candidates-by-features tensor.
    if len(positive) > len(negative):
        positive, negative = negative, positive
    for left in positive:
        for offset in range(0, len(negative), 64):
            right = np.asarray(negative[offset:offset + 64])
            if np.any(np.max(np.abs(right - left), axis=1) <= TOLERANCE):
                return True
    return False


def _new_samples():
    return dict(literal_error=[], reconstruction_absolute=[[] for _ in FIELDS],
                reconstruction_relative=[[] for _ in FIELDS],
                candidate_mean_reconstruction_absolute=[[] for _ in FIELDS])


def analyze_scene(scene, rng, global_samples):
    """Reader labels define legal population; within it identity only stratifies outputs."""
    import numpy as np
    counts = Counter(dict.fromkeys(COUNT_KEYS, 0))
    zero_range, near_constant = Counter(), Counter()
    versions = {version: Counter(dict.fromkeys(VERSION_COUNT_KEYS, 0)) for version in VERSIONS}
    maxima = _new_maxima()
    sources = sorted({track.source for track in scene.tracks})
    directions = []
    for source_index, source in enumerate(sources):
        source_tracks = [track for track in scene.tracks if track.source == source]
        candidates = [track for track in scene.tracks if track.source != source]
        chosen = rng.sample(source_tracks, min(ANCHORS_PER_SOURCE, len(source_tracks)))
        direction_counts = Counter(available_anchor_nodes=len(source_tracks),
                                   unselected_anchor_nodes=len(source_tracks) - len(chosen))
        direction_versions = {version: Counter(dict.fromkeys(VERSION_COUNT_KEYS, 0)) for version in VERSIONS}
        direction_maxima = _new_maxima()
        for anchor in chosen:
            anchor_counts = Counter(selected_anchors=1)
            if len(anchor.times) < WINDOW_POINTS:
                anchor_counts['anchors_shorter_than_window'] += 1
                direction_counts.update(anchor_counts)
                continue
            anchor_counts['anchors_with_window'] += 1
            start = rng.randrange(len(anchor.times) - WINDOW_POINTS + 1)
            query = anchor.times[start:start + WINDOW_POINTS]
            a = anchor.values[start:start + WINDOW_POINTS]
            normalize(a)  # Validate the shared A branch before consulting candidate labels.
            envelopes = {version: {} for version in VERSIONS}
            literal_by_label = {False: [], True: []}
            positive = negative = 0
            for candidate in candidates:
                anchor_counts['cross_source_candidate_pairs'] += 1
                b = interpolate_candidate(candidate, query)
                if b is None:
                    anchor_counts['outside_support_pairs'] += 1
                    continue
                item = pair_representations(a, b)
                # Within the reader's label-cleaned legal population, identity
                # does not select windows, interpolation, features or candidates.
                same = candidate.identity_id == anchor.identity_id
                positive += int(same)
                negative += int(not same)
                anchor_counts['eligible_pairs'] += 1
                anchor_counts['positive_eligible_pairs' if same else 'negative_eligible_pairs'] += 1
                anchor_counts['literal_pair_error_gt_tolerance'] += int(item['literal_error'] > TOLERANCE)
                global_samples['literal_error'].append(item['literal_error'])
                for i, field in enumerate(FIELDS):
                    zero_range[field] += int(item['zero_range'][i])
                    near_constant[field] += int(item['near_constant_range'][i])
                    global_samples['reconstruction_absolute'][i].append(float(item['reconstruction_absolute'][i]))
                    global_samples['reconstruction_relative'][i].append(float(item['reconstruction_relative'][i]))
                    global_samples['candidate_mean_reconstruction_absolute'][i].append(
                        float(item['candidate_mean_reconstruction_absolute'][i]))
                _envelope_update(envelopes, same, item['payloads'])
                literal_by_label[same].append(item['payloads']['literal'])
            if not positive and not negative:
                anchor_counts['anchors_without_eligible_candidates'] += 1
            else:
                category = 'mixed' if positive and negative else ('only_positive' if positive else 'only_negative')
                anchor_counts['anchors_' + category] += 1
                variation = _version_variation(envelopes)
                _merge_maxima(direction_maxima, variation)
                for version, value in variation.items():
                    direction_versions[version]['anchors_with_eligible_candidates'] += 1
                    direction_versions[version]['anchors_all_candidates_within_tolerance'] += int(
                        value['all_candidates_within_tolerance'])
                    if category == 'mixed':
                        direction_versions[version]['mixed_anchors'] += 1
                        direction_versions[version]['mixed_anchors_any_exact_content_change'] += int(
                            value['cross_label_any_exact_content_change'])
                        direction_versions[version]['mixed_anchors_any_change_gt_tolerance'] += int(
                            value['cross_label_any_change_gt_tolerance'])
                if category == 'mixed':
                    anchor_counts['mixed_anchors_literal_cross_label_collision'] += int(
                        _has_literal_collision(literal_by_label[True], literal_by_label[False]))
            direction_counts.update(anchor_counts)
        for key in COUNT_KEYS:
            direction_counts.setdefault(key, 0)
        counts.update(direction_counts)
        for version in VERSIONS:
            versions[version].update(direction_versions[version])
        _merge_maxima(maxima, direction_maxima)
        directions.append(dict(source_direction_index=source_index, counts=dict(direction_counts),
            versions={k: dict(v) for k, v in direction_versions.items()},
            content_difference_maxima=direction_maxima))
    if (counts['selected_anchors'] != counts['anchors_shorter_than_window'] + counts['anchors_with_window'] or
            counts['anchors_with_window'] != sum(counts[key] for key in (
                'anchors_without_eligible_candidates', 'anchors_only_positive', 'anchors_only_negative', 'anchors_mixed')) or
            counts['cross_source_candidate_pairs'] != counts['eligible_pairs'] + counts['outside_support_pairs'] or
            counts['eligible_pairs'] != counts['positive_eligible_pairs'] + counts['negative_eligible_pairs']):
        raise ValueError('Diagnostic denominator accounting mismatch')
    audit_keys = ('raw_measurement_rows', 'label_rows', 'candidate_nodes', 'valid_nodes',
                  'quarantined_nodes', 'exact_duplicate_rows_removed',
                  'measurement_rows_outside_label_interval_diagnostic_only',
                  'positive_edges_lost_to_quarantine')
    return dict(scene_id=scene.scene_id, physical_source_count=len(sources), counts=dict(counts),
        versions={k: dict(v) for k, v in versions.items()}, source_directions=directions,
        content_difference_maxima=maxima,
        zero_range_pair_columns={field: zero_range[field] for field in FIELDS},
        near_constant_pair_columns={field: near_constant[field] for field in FIELDS},
        loader_audit={**{key: int(scene.audit.get(key, 0)) for key in audit_keys},
                      'quarantine_reasons': dict(scene.audit.get('quarantine_reasons', {}))})


def summarize(scenes, samples):
    counts, audit, reasons = Counter(dict.fromkeys(COUNT_KEYS, 0)), Counter(), Counter()
    versions = {version: Counter(dict.fromkeys(VERSION_COUNT_KEYS, 0)) for version in VERSIONS}
    maxima = _new_maxima()
    zeros, near = Counter(), Counter()
    for scene in scenes:
        counts.update(scene['counts'])
        zeros.update(scene['zero_range_pair_columns'])
        near.update(scene['near_constant_pair_columns'])
        reasons.update(scene['loader_audit']['quarantine_reasons'])
        audit.update({k: v for k, v in scene['loader_audit'].items() if k != 'quarantine_reasons'})
        for version in VERSIONS:
            versions[version].update(scene['versions'][version])
        _merge_maxima(maxima, scene['content_difference_maxima'])
    return dict(completed_scene_count=len(scenes), counts=dict(counts),
        versions={k: dict(v) for k, v in versions.items()},
        content_difference_maxima=maxima, comparison_contract=COMPARISON_CONTRACT,
        literal_input_error=_distribution(samples['literal_error']),
        stats_reconstruction={field: dict(
            absolute=_distribution(samples['reconstruction_absolute'][i]),
            relative=_distribution(samples['reconstruction_relative'][i]),
            candidate_mean_absolute=_distribution(samples['candidate_mean_reconstruction_absolute'][i]))
            for i, field in enumerate(FIELDS)},
        zero_range_pair_columns=dict(zeros), near_constant_pair_columns=dict(near),
        loader_audit={**dict(audit), 'quarantine_reasons': dict(reasons)},
        scene_count_ranges={key: dict(min=min((s['counts'].get(key, 0) for s in scenes), default=0),
            max=max((s['counts'].get(key, 0) for s in scenes), default=0)) for key in counts},
        interpretation_limits=[
            'Explicit representation controls, not author implementation, classification accuracy or method contribution.',
            'A/q/validity/source-difference relation are fixed; no B IDs or identity labels enter a representation.',
            'Content variation is not learned identity; any-change does not prove all cross-label collisions removed.',
            'Normalized components use 1e-8; native m/r and rawB use the per-component float64 scale rule in comparison_contract.',
            'Native maxima are separated by attribute/unit and are not physical effect-size thresholds.',
            'Exact-change counts are also reported; tiny floating changes are not useful information guarantees.',
            'Fixed random T32 exploration, not full-T rate or IID identity-independent significance.',
            'Reference, V, test, models, weights, normalization fitting and training are absent.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, nargs='+', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.cpu_threads <= 4:
        parser.error('cpu-threads must be 1..4')
    if args.output.exists():
        raise FileExistsError('Output exists; do not overwrite or retry in place')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = str(args.cpu_threads)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    args.output.mkdir(parents=True)
    private = args.output / 'private'
    private.mkdir(mode=0o700)
    os.chmod(private, 0o700)
    started = time.monotonic()
    run = dict(status='RUNNING', idea_id='IDEA-PAIR-REP-DEPENDENCY-20261028', plan_version='1.1',
        start_utc=datetime.now(timezone.utc).isoformat(), cpu_threads=args.cpu_threads,
        configuration=dict(seed=SEED, scene_count=SCENE_COUNT, anchors_per_source=ANCHORS_PER_SOURCE,
            window_points=WINDOW_POINTS, tolerance=TOLERANCE, max_member_bytes=MAX_MEMBER_BYTES,
            max_address_space_bytes=MAX_ADDRESS_SPACE_BYTES,
            near_constant_rule='0 < range <= 64*float64_eps*max(1,max_abs_difference)'),
        comparison_contract=COMPARISON_CONTRACT, public_output_allowlist=list(PUBLIC_OUTPUTS),
        argv=list(sys.argv if argv is None else argv), completed_scenes=0,
        read_member_count=0, parsed_member_count=0, payload_access_attempt_count=0,
        reference_members_read=0, validation_members_read=0, test_members_read=0,
        model_constructed=False, model_forwards=0, optimizer_updates=0,
        checkpoint_files_read=0, normalizer_fitted=False)
    write_json(args.output / 'run.json', run)
    reads, scenes = [], []
    prior_term = signal.getsignal(signal.SIGTERM)
    interrupted = [False]
    def terminated(signum, frame):
        interrupted[0] = True
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError('SIGTERM')
    signal.signal(signal.SIGTERM, terminated)
    code = 1
    try:
        run['address_space_limit_bytes'] = address_space_limit()
        run['address_space_limit_scope'] = 'Linux process virtual address space; not RSS; lower inherited limit retained'
        write_json(args.output / 'run.json', run)
        from association.data.archive import MTADArchive, validate_manifest
        from association.data.legacy_dataset import validate_archive
        manifest = json.loads(args.manifest.read_text())
        validate_manifest(manifest)
        if (manifest.get('dataset_level') != 'Level1' or
                manifest.get('split_unit') != 'scene' or
                {key: len(value) for key, value in manifest['splits'].items()} !=
                {'T_core': 2100, 'T_head': 0, 'V_select': 300, 'T_test': 600}):
            raise ValueError('Fixed Level1 scene3000 manifest required')
        rng = random.Random(SEED)
        selected = sorted(rng.sample(sorted(manifest['splits']['T_core'], key=int), SCENE_COUNT), key=int)
        run['selected_T_scene_ids'] = selected
        with MTADArchive(args.archive) as archive, (args.output / 'progress.jsonl').open('a') as progress:
            validate_archive(archive, manifest)
            inventory = []
            for sid in selected:
                for kind in ('labels', 'measurements'):
                    info, name = archive._members[kind][sid]
                    inventory.append(dict(scene_id=sid, kind=kind, member=name,
                        compressed_bytes=info.compress_size, declared_uncompressed_bytes=info.file_size,
                        within_size_limit=info.file_size <= MAX_MEMBER_BYTES))
            write_json(args.output / 'member-inventory.json', inventory)
            if not all(row['within_size_limit'] for row in inventory):
                raise ValueError('Selected member exceeds declared 64MiB ceiling; no payload read')
            allowed = {(row['scene_id'], row['kind']): row for row in inventory}
            original_rows = archive._rows
            def bounded_rows(sid, kind):
                key = (str(sid), kind)
                if key not in allowed:
                    raise ValueError('Payload outside fixed T32 label/measurement scope')
                entry = {**allowed[key], 'status': 'READING', 'payload_read_completed': False}
                reads.append(entry)
                write_json(args.output / 'read-members.json', reads)
                before = len(archive.members_read)
                try:
                    rows = original_rows(sid, kind)
                except BaseException:
                    entry['payload_read_completed'] = len(archive.members_read) > before
                    entry['status'] = 'FAILED'
                    write_json(args.output / 'read-members.json', reads)
                    raise
                entry['status'] = 'READ'
                entry['payload_read_completed'] = True
                write_json(args.output / 'read-members.json', reads)
                return rows
            archive._rows = bounded_rows
            samples = _new_samples()
            for sid in selected:
                run['current_scene_id'] = sid
                write_json(args.output / 'run.json', run)
                scene = archive.load_scene(sid)
                row = analyze_scene(scene, rng, samples)
                scenes.append(row)
                del scene
                archive._label_cache.pop(sid, None)
                run['completed_scenes'] = len(scenes)
                run['read_member_count'] = sum(entry['payload_read_completed'] for entry in reads)
                run['parsed_member_count'] = sum(entry['status'] == 'READ' for entry in reads)
                run['payload_access_attempt_count'] = len(reads)
                event = dict(scene_id=sid, completed_scenes=len(scenes), status='COMPLETED',
                             elapsed_seconds=time.monotonic() - started)
                progress.write(json.dumps(event) + '\n')
                progress.flush()
                write_json(args.output / 'scenes.json', scenes)
                write_json(args.output / 'run.json', run)
                print(f'completed scenes {len(scenes)}/{SCENE_COUNT}', flush=True)
            expected = {(sid, kind) for sid in selected for kind in ('labels', 'measurements')}
            actual = [(row['scene_id'], row['kind']) for row in reads if row['status'] == 'READ']
            if len(actual) != 2 * SCENE_COUNT or set(actual) != expected:
                raise ValueError('Read population differs from fixed T32 contract')
            write_json(args.output / 'summary.json', summarize(scenes, samples))
        run['status'] = 'COMPLETED'
        code = 0
        print('COMPLETED; representation diagnostic only, no model or reference', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        run.update(status='FAILED', error_type=type(error).__name__)
        (private / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('FAILED ' + type(error).__name__ + '; private trace retained', file=sys.stderr, flush=True)
        code = 143 if interrupted[0] else (130 if isinstance(error, KeyboardInterrupt) else 1)
    finally:
        run.update(exit_code=code, end_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            read_member_count=sum(row['payload_read_completed'] for row in reads),
            parsed_member_count=sum(row['status'] == 'READ' for row in reads),
            payload_access_attempt_count=len(reads),
            peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_rss_unit='KiB' if sys.platform.startswith('linux') else 'bytes')
        write_json(args.output / 'read-members.json', reads)
        write_json(args.output / 'scenes.json', scenes)
        write_json(args.output / 'run.json', run)
        signal.signal(signal.SIGTERM, prior_term)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
