"""103: fixed-population diagnosis of three previously selected V scenes.

Read predictions, never weights. Replay deterministic input/reference preparation
only for 3035/2819/1043. Neither oracle error nor these postselected scenes may
choose a new method, threshold, assignment or calibration. No torch dependency.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import gzip
import itertools
import json
from pathlib import Path
import resource
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.data.archive import MTADArchive, validate_manifest
from association.data.legacy_dataset import validate_archive
from association.data.motion_evidence import DATA, scene_motion_evidence
from association.data.native import _record
from association.data.reference_states import iter_reference_states
from association.data.targets import align_reference_states
from association.data.timed import prepare_timed_scene
from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.physical import _check_valid_states
from association.evaluation.support_gate_objectives import (
    _artifact, _check_artifact, _read, _same, _structure,
)

SCENES = ('3035', '2819', '1043')
METHODS = ('learned_sequence', 'anchor', 'mean_supported')
REGIONS = ('anchor_inside_support', 'anchor_outside_helper_supported',
           'predicted_group_no_support', 'no_predicted_group')
R = 6371000.


def read(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write(path, value):
    path = Path(path)
    value = clean(value)
    if path.suffix == '.gz':
        with gzip.open(path, 'xt', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    else:
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)


def stats(values):
    values = np.asarray(values, float).ravel()
    x = values[np.isfinite(values)]
    return dict(count=len(x), unavailable_count=len(values)-len(x),
                mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                p05=float(np.quantile(x, .05)) if len(x) else None,
                p95=float(np.quantile(x, .95)) if len(x) else None,
                p99=float(np.quantile(x, .99)) if len(x) else None,
                max=float(x.max()) if len(x) else None)


def radial(a, b):
    a, b = np.asarray(a), np.asarray(b)
    # Reuse the official spherical distance unchanged, with unused attributes.
    return attribute_errors(np.pad(a[..., :2], [(0, 0)]*(a.ndim-1)+[(0, 2)]),
                            np.pad(b[..., :2], [(0, 0)]*(b.ndim-1)+[(0, 2)]))['position']


def local_delta(a, b, latitude_origin):
    """[north,east] m, shared scene cos(latitude); shortest longitude difference.

    Same spherical/local-plane convention as motion_evidence, with one visible
    scene origin instead of per-fragment origins. This diagnostic approximation
    never replaces radial scoring. Report its norm/radial ratio for distortion.
    """
    delta = np.asarray(a)[..., :2]-np.asarray(b)[..., :2]
    delta = delta.copy()
    delta[..., 1] = (delta[..., 1]+180.) % 360.-180.
    return delta * (R*np.pi/180.*np.array([1., np.cos(np.deg2rad(latitude_origin))]))


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    out = np.full(denom.shape, np.nan)
    np.divide((a*b).sum(-1), denom, out=out, where=denom > 0)
    return np.clip(out, -1., 1.)


def brackets(reference, identity, query, policy):
    """Audit raw endpoint rows; never repair duplicates or replace the aligner."""
    groups = defaultdict(list)
    for row in reference['rows']:
        t = row['time']
        if row['identity_id'] == identity and t is not None and np.isfinite(t):
            groups[float(t)-policy['time_offset_seconds']].append(row)
    times = np.array(sorted(groups))
    result = []
    for q in query:
        right = int(np.searchsorted(times, q))
        if right < len(times) and times[right] == q:
            left = right
            kind = 'exact'
        elif 0 < right < len(times):
            left = right-1
            kind = 'interior' if times[right]-times[left] <= policy['max_gap_seconds'] else 'gap_exceeded'
        else:
            result.append(dict(kind='outside_support' if len(times) else 'no_reference_times',
                               left=None, right=None, gap_seconds=None))
            continue
        result.append(dict(kind=kind, gap_seconds=float(times[right]-times[left]),
            left=dict(aligned_time=times[left], rows=groups[times[left]]),
            right=dict(aligned_time=times[right], rows=groups[times[right]])))
    return result


def validate_request(manifest):
    validate_manifest(manifest)
    if not set(SCENES).issubset(manifest['splits']['V_select']):
        raise ValueError('103 only allows the three declared V_select scenes')


def prepare_scene(archive, manifest, sid):
    """Read exactly one scene, preserving original labels and canonicalization."""
    if sid not in SCENES or sid not in manifest['splits']['V_select']:
        raise ValueError('scene outside the three-scene contract')
    native = _record(archive.load_scene(sid), manifest['components'][sid], 'V_select',
                     DATA['max_observations'], split_unit=manifest.get('split_unit'))
    labels = archive.read_labels(sid)
    targets = list(dict.fromkeys('-'.join(row['mmsi'].split('-')[:2]) for row in labels))
    query = None
    if not native['tracks']:
        if not labels:
            raise ValueError('empty scene has no original label bounds')
        query = np.linspace(min(row['t_s'] for row in labels), max(row['t_e'] for row in labels), DATA['length'])
    prepared = prepare_timed_scene(native, **DATA, query_times=query)
    return prepared, targets


def read_cache(source, sid):
    fragment = _read(source/'final/predictions'/f'states-{sid}.npz')
    system = _read(source/'final/system-predictions'/f'states-{sid}.npz')
    arms = {m: _artifact(system, m) for m in METHODS}
    base = arms['anchor']
    # 090C has no 069 gate. Supply that checker's derived helper-only mask in
    # memory; its actual cached support regions and group mapping are still
    # independently checked. Never claim this derived field came from the NPZ.
    offsets, flat = fragment['learned_assoc_group_offsets'], fragment['learned_assoc_group_node_indices']
    group_members = [flat[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    eligible = np.array([~fragment['support'][g[0]] & fragment['support'][g[1:]].any(0)
                         for g in group_members])
    groups, selected, regions, population = _structure(dict(fragment, group_eligible=eligible), base)
    errors = {}
    for method, artifact in arms.items():
        _check_valid_states(artifact['aligned_prediction'], artifact['aligned_prediction_valid'], method)
        _check_valid_states(artifact['aligned_truth'], artifact['aligned_truth_valid'], 'reference')
        errors[method] = _check_artifact(fragment, artifact, base, method)
    for method in METHODS:
        _same(np.isfinite(errors[method]), np.isfinite(errors['anchor']), 'common method mask')
    return fragment, arms, groups, selected, regions, population, errors


def source_metadata(source, manifest):
    """Bind actual run/settings, without opening a checkpoint or other scenes."""
    run, final, summary = (read(source/p) for p in ('run.json', 'final/run.json', 'final/summary.json'))
    if any(doc.get('status') != 'COMPLETED' for doc in (run, final, summary)):
        raise ValueError('source must be completed')
    if set(summary['scene_ids']) != set(manifest['splits']['V_select']) or summary.get('test_opened') is not False:
        raise ValueError('source is not the declared frozen V population')
    config = run['configuration']
    expected = read(Path(__file__).resolve().parents[2]/'configs/frozen-position-adapter-full-finetune-level1.json')
    if (config != expected or run.get('optimizer_updates') != 4000 or run.get('test_opened') is not False):
        raise ValueError('source configuration or completion differs from 090C')
    for doc in (final, summary):
        checkpoint = doc.get('checkpoint', {})
        if (doc.get('configuration') != config or doc.get('initial') is not False
                or doc.get('groups_recomputed') is not False or doc.get('threshold_reselected') is not False
                or doc.get('test_opened') is not False or doc.get('optimizer_updates') != 0
                or doc.get('source_system') != run.get('source_system')
                or checkpoint.get('step') != 4000 or checkpoint.get('actual_updates') != 4000
                or Path(checkpoint.get('path', '')).resolve() != source/'fit/final.pt'):
            raise ValueError('final endpoint metadata differs from the completed 090C cache')
    return dict(source_run=str(source), final_scene_ids=summary['scene_ids'], config=config,
                checkpoint_metadata_only=final.get('checkpoint'), source_completed=True)


def analyze_scene(prepared, targets, reference, policy, cached, thresholds):
    fragment, arms, groups, selected, regions, population, errors = cached
    base, tracks = arms['anchor'], prepared['tracks']
    t, q = len(targets), len(prepared['grid'])
    for key, expected in (('node_ids', prepared['node_ids']), ('identities', prepared['identities']),
                          ('target_identity_ids', targets), ('query_seconds', prepared['grid'])):
        _same(fragment[key], expected, 'raw input/cache '+key)
    motion = scene_motion_evidence(prepared, motion_observation_policy='canonical_all')
    for key, other in (('native_states', 'states'), ('support', 'support'),
                       ('nearest_seconds', 'nearest_seconds'), ('exact_observation', 'exact_observation')):
        _same(fragment[key], motion[other], 'deterministic input replay '+key)
    aligned = [align_reference_states(reference, target, prepared['grid'],
               longitude_policy='shortest_arc', **policy) for target in targets]
    truth, valid = np.array([a['states'] for a in aligned]), np.array([a['valid'] for a in aligned])
    _same(base['aligned_truth'][:t], truth, 'reference states from raw rows')
    _same(base['aligned_truth_valid'][:t], valid, 'reference validity from raw rows')
    if not tracks:
        raise ValueError('selected diagnostic scene unexpectedly has no canonical observations')
    lat0 = float(sum(track['canonical_values'][:, 0].sum() for track in tracks)
                 / sum(len(track['canonical_times']) for track in tracks))
    chart = dict(latitude_origin_degrees=lat0, origin_rule='mean of all visible canonical latitudes',
                 axes=['north', 'east'], unit='m', radius_m=R,
                 approximation='fixed scene cos(latitude); shortest longitude difference; radial score unchanged')
    query = prepared['grid']
    point_rows, query_sources, pair_rows, native_pairs, node_rows, private_refs = [], [], [], [], [], []
    vectors = {m: local_delta(arms[m]['aligned_prediction'][:t], truth, lat0) for m in METHODS}
    for ti, target in enumerate(targets):
        ends = brackets(reference, target, query, policy)
        gid = int(selected[ti]); group = groups[gid] if gid >= 0 else []
        for qi, timestamp in enumerate(query):
            pv = base['aligned_prediction_valid'][ti, qi, :2].all()
            tv = valid[ti, qi, :2].all()
            row = dict(scene_id=prepared['scene_id'], target_index=ti, group_index=gid,
                query_index=qi, query_seconds=float(timestamp), prediction_valid=bool(pv), truth_valid=bool(tv),
                missing_reason=('no_position_reference' if not tv else 'no_predicted_group' if not pv else None),
                region=next(name for name in REGIONS if regions[name][ti, qi]),
                group_size=len(group), identity_consistent_members=sum(tracks[n]['identity_id'] == target for n in group),
                reference_kind=ends[qi]['kind'], reference_gap_seconds=ends[qi]['gap_seconds'],
                reference_endpoint_times=[ends[qi][side]['aligned_time'] if ends[qi][side] else None
                                          for side in ('left', 'right')],
                reference_endpoint_row_counts=[len(ends[qi][side]['rows']) if ends[qi][side] else 0
                                               for side in ('left', 'right')],
                methods={}, cosines={}, supported_member_count=int(motion['support'][group, qi].sum()))
            for m in METHODS:
                e = errors[m][ti, qi]
                row['methods'][m] = dict(error_m=e, error_NE_m=vectors[m][ti, qi] if pv and tv else [None, None],
                    plane_radial_ratio=np.linalg.norm(vectors[m][ti, qi])/e if np.isfinite(e) and e > 0 else None)
            for a, b in itertools.combinations(METHODS, 2):
                row['cosines'][a+'__'+b] = cosine(vectors[a][ti, qi], vectors[b][ti, qi]) if pv and tv else None
            row['prediction_separation_m'] = {a+'__'+b:
                radial(arms[a]['aligned_prediction'][ti, qi], arms[b]['aligned_prediction'][ti, qi])
                if pv else None for a, b in itertools.combinations(METHODS, 2)}
            source_errors = radial(motion['states'][group, qi], truth[ti, qi]) if group and tv else np.array([])
            row['oracle_source_min_error_m'] = float(source_errors.min()) if len(source_errors) else None
            row['oracle_source_max_error_m'] = float(source_errors.max()) if len(source_errors) else None
            row['input_identifiable_healthy_source'] = 'unknown; oracle closeness is not input evidence'
            point_rows.append(row)
        # Raw endpoint values/IDs stay in the server-only audit, not the return table.
        private_refs.append(dict(target_index=ti, target_identity=target, endpoints=ends,
                                 alignment_report=aligned[ti]['report']))
    for ni, track in enumerate(tracks):
        times, values = track['canonical_times'], track['canonical_values']
        ref = align_reference_states(reference, track['identity_id'], times, longitude_policy='shortest_arc', **policy)
        native_valid = ref['valid'][:, :2].all(-1)
        distance = np.where(native_valid, radial(values, ref['states']), np.nan)
        ne = local_delta(values, ref['states'], lat0)
        ti = targets.index(track['identity_id'])
        # Step geometry is observable; it is not a true speed/turn label.
        dt = np.diff(times)
        step_mps = radial(values[1:], values[:-1])/dt
        node_rows.append(dict(node_index=ni, target_index=ti, source=track['source'],
            canonical_event_count=len(times), support_start=times[0], support_stop=times[-1],
            native_to_own_reference_error_m=stats(distance), native_to_own_reference_N_m=stats(ne[native_valid, 0]),
            native_to_own_reference_E_m=stats(ne[native_valid, 1]),
            event_gap_seconds=stats(dt), observable_position_step_mps=stats(step_mps),
            native_fit_rmse_m=motion['fit_rmse_m'][ni],
            native_attribute_speed_unit='unverified per-scene native; not used for physical speed conclusions',
            reference_valid_events=int(native_valid.sum()),
            reference_alignment_counts={k: v for k, v in ref['report'].items() if k not in ('identity_id', 'issues')}))
        gid = int(base['node_group_indices'][ni])
        bound_ti = int(base['group_anchor_target_indices'][gid])
        for qi in range(q):
            nearest = int(np.argmin(np.abs(times-query[qi])))
            query_sources.append(dict(node_index=ni, target_index=ti, source=track['source'], query_index=qi,
                group_index=gid, bound_target_index=bound_ti,
                selected_unique_group=bool(selected[bound_ti] == gid),
                query_seconds=query[qi], support=motion['support'][ni, qi], exact_observation=motion['exact_observation'][ni, qi],
                nearest_seconds=motion['nearest_seconds'][ni, qi], nearest_event_time=times[nearest],
                native_fit_rmse_m=motion['fit_rmse_m'][ni],
                query_estimate_to_own_reference_error_m=radial(motion['states'][ni, qi], truth[ti, qi])
                    if valid[ti, qi, :2].all() else None,
                query_estimate_to_own_reference_NE_m=local_delta(motion['states'][ni, qi], truth[ti, qi], lat0)
                    if valid[ti, qi, :2].all() else [None, None],
                query_estimate_to_bound_reference_error_m=radial(motion['states'][ni, qi], truth[bound_ti, qi])
                    if valid[bound_ti, qi, :2].all() else None,
                query_estimate_to_bound_reference_NE_m=local_delta(motion['states'][ni, qi], truth[bound_ti, qi], lat0)
                    if valid[bound_ti, qi, :2].all() else [None, None],
                query_estimate_minus_nearest_event_NE_m=local_delta(motion['states'][ni, qi], values[nearest], lat0),
                nearest_event_to_own_reference_error_m=distance[nearest]))
    # Union of fixed-group pairs and same-label cross-source pairs. The latter
    # are explicitly oracle correspondences and are NEVER added to model groups.
    predicted_pairs = {tuple(pair): gi for gi, g in enumerate(groups) for pair in itertools.combinations(g, 2)}
    pairs = set(predicted_pairs)
    pairs.update((a, b) for a, b in itertools.combinations(range(len(tracks)), 2)
                 if tracks[a]['identity_id'] == tracks[b]['identity_id'])
    for a, b in sorted(pairs):
        # Canonical source orientation makes shared source-pair vectors comparable.
        if (tracks[a]['source'], a) > (tracks[b]['source'], b): a, b = b, a
        ta, tb = tracks[a], tracks[b]
        metadata = dict(node_a=a, node_b=b, source_a=ta['source'], source_b=tb['source'],
            target_a=targets.index(ta['identity_id']), target_b=targets.index(tb['identity_id']),
            oracle_same_identity=ta['identity_id'] == tb['identity_id'],
            fixed_group_index=predicted_pairs.get(tuple(sorted((a, b)))), direction='source_b minus source_a')
        for qi in range(q):
            pair_rows.append(dict(**metadata, query_index=qi, query_seconds=query[qi],
                kind='query_estimate_not_raw', both_supported=bool(motion['support'][a, qi] and motion['support'][b, qi]),
                both_exact=bool(motion['exact_observation'][a, qi] and motion['exact_observation'][b, qi]),
                delta_NE_m=local_delta(motion['states'][b, qi], motion['states'][a, qi], lat0),
                distance_m=radial(motion['states'][a, qi], motion['states'][b, qi])))
        common, ia, ib = np.intersect1d(ta['canonical_times'], tb['canonical_times'], return_indices=True)
        if len(common):
            delta = local_delta(tb['canonical_values'][ib], ta['canonical_values'][ia], lat0)
            dist = radial(tb['canonical_values'][ib], ta['canonical_values'][ia])
            for k, timestamp in enumerate(common):
                native_pairs.append(dict(**metadata, kind='simultaneous_canonical_observations',
                    time_seconds=timestamp, delta_NE_m=delta[k], distance_m=dist[k]))
        else:
            native_pairs.append(dict(**metadata, kind='no_exact_common_native_time',
                                     time_seconds=None, delta_NE_m=[None, None], distance_m=None))
    summary = dict(scene_id=prepared['scene_id'], component_id=prepared['component_id'], population=population,
        declared_query_points=t*q, raw_replay_verified=True, reference_replay_verified=True, chart=chart,
        input_audit_counts={k: v for k, v in prepared['audit'].items()
                            if k not in ('quarantined', 'valid_raw_mmsi', 'native_inputs')}, alignment_policy=policy,
        native_events=sum(len(tr['canonical_times']) for tr in tracks), strata={},
        method_pair_cosines={key: stats([r['cosines'][key] for r in point_rows]) for key in point_rows[0]['cosines']},
        relative_geometry_bias_candidate=relative_summary(pair_rows, native_pairs),
        unknowns=['Observable healthy-source identifiability is not established by oracle error.',
                  'Shared relative geometry, relative clock error and similar motion may be confounded.',
                  'Replay equality checks implementation consistency, not correctness of supplied reference/labels.',
                  'Three scenes are postselected; no population incidence or causal/algorithm benefit estimate.'])
    for region in ('all', *REGIONS):
        subset = [r for r in point_rows if region == 'all' or r['region'] == region]
        summary['strata'][region] = aggregate_points(subset, thresholds)
    public = dict(points=point_rows, query_sources=query_sources, query_pairs=pair_rows,
                  simultaneous_native_pairs=native_pairs, nodes=node_rows)
    private = dict(reference_endpoints=private_refs, input_audit=prepared['audit'],
        input_tracks=[dict(node_index=i, **tr) for i, tr in enumerate(tracks)],
        raw_reference=reference, input_motion=motion)
    return summary, public, private


def aggregate_points(rows, thresholds):
    out = dict(declared=len(rows), available=sum(r['truth_valid'] for r in rows),
        evaluated=sum(r['truth_valid'] and r['prediction_valid'] for r in rows), methods={}, overlap={})
    for m in METHODS:
        v = np.array([r['methods'][m]['error_m'] for r in rows], float)
        v = v[np.isfinite(v)]
        out['methods'][m] = dict(**stats(v), AE=float(v.sum()), SSE=float((v*v).sum()),
                                RMSE=float(np.sqrt(np.mean(v*v))) if len(v) else None)
    # Fixed full-V per-method thresholds from saved 101. Zero/unknown never
    # count as disagreement or a negative case. No threshold fitted here.
    for label, cut in thresholds.items():
        evaluated = [r for r in rows if r['truth_valid'] and r['prediction_valid']]
        counts = defaultdict(int)
        chosen = []
        for r in evaluated:
            pattern = ''.join('1' if r['methods'][m]['error_m'] >= cut[m] else '0' for m in METHODS)
            counts[pattern] += 1
            if pattern == '111': chosen.append(r)
        overlap = dict(pattern_order=list(METHODS), thresholds_m=cut, pattern_counts=dict(counts),
                       common_tail_count=len(chosen), common_tail_AE_SSE_share={})
        for m in METHODS:
            ae = sum(r['methods'][m]['error_m'] for r in chosen)
            sse = sum(r['methods'][m]['error_m']**2 for r in chosen)
            total = out['methods'][m]
            overlap['common_tail_AE_SSE_share'][m] = dict(
                AE_fraction=ae/total['AE'] if total['AE'] else None,
                SSE_fraction=sse/total['SSE'] if total['SSE'] else None)
        out['overlap'][label] = overlap
    return out


def relative_summary(query_pairs, native_pairs):
    result = []
    for kind, rows in (('query_estimate_not_raw', query_pairs), ('simultaneous_native', native_pairs)):
        buckets = defaultdict(list)
        for r in rows:
            if (r['oracle_same_identity'] and r['source_a'] != r['source_b']
                    and r['distance_m'] is not None and (kind != 'query_estimate_not_raw' or r['both_supported'])):
                buckets[(r['source_a'], r['source_b'])].append(r)
        for pair, bucket in buckets.items():
            by_target = defaultdict(list)
            for row in bucket: by_target[row['target_a']].append(row['delta_NE_m'])
            means = np.array([np.mean(v, axis=0) for v in by_target.values()])
            centre = means.mean(0)  # Target equal, not dense-target point equal.
            result.append(dict(kind=kind, sources=pair, oracle_correspondence=True,
                matched_target_count=len(by_target), pair_time_count=len(bucket), target_equal_mean_NE_m=centre,
                between_target_mean_distance_m=stats(np.linalg.norm(means-centre, axis=-1)),
                targets=[dict(target_index=k, pair_time_count=len(v), mean_NE_m=np.mean(v, axis=0),
                    within_target_distance_to_mean_m=stats(np.linalg.norm(np.asarray(v)-np.mean(v, axis=0), axis=-1)))
                    for k, v in by_target.items()],
                interpretation='descriptive consistency only; no fitted calibration or common absolute origin'))
    return result


def load_thresholds(path, source, manifest):
    if path is None:
        return {}, dict(status='unavailable; continuous distributions only')
    doc = read(path)
    if (doc.get('status') != 'COMPLETED' or set(doc['scene_ids']) != set(manifest['splits']['V_select'])
            or Path(doc['source_run']).name != source.name):
        raise ValueError('101 threshold source differs from declared final V run')
    cuts = {label: {m: float(doc['methods'][m]['all']['position_m'][label]) for m in METHODS}
            for label in ('abs_P95', 'abs_P99')}
    if not all(np.isfinite(v) and v >= 0 for cut in cuts.values() for v in cut.values()):
        raise ValueError('invalid saved full-V thresholds')
    return cuts, dict(status='saved_full_V_101', path=str(path), values=cuts)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id='IDEA-STATE-DIAG-20261027', plan_version='1.3-implementation',
        argv=sys.argv, scene_ids=list(SCENES), completed_scenes=[], optimizer_updates=0, model_forwards=0,
        checkpoint_files_read=0, test_opened=False, groups_recomputed=False, threshold_reselected=False,
        deterministic_input_replay=True, source_run=str(Path(args.source_run).resolve()))
    write(output/'run.json', state)
    try:
        manifest, policies = read(args.manifest), read(args.alignment)
        validate_request(manifest)
        source = Path(args.source_run).resolve()
        state['source_metadata'] = source_metadata(source, manifest)
        thresholds, provenance = load_thresholds(args.threshold_summary, source, manifest)
        state['thresholds'] = provenance
        rows, pooled = [], []
        with MTADArchive(args.archive) as archive:
            validate_archive(archive, manifest)
            for sid in SCENES:
                cached = read_cache(source, sid)
                prepared, targets = prepare_scene(archive, manifest, sid)
                with closing(iter_reference_states(args.archive, manifest, split='V_select', scene_ids=[sid])) as refs:
                    reference = next(refs)
                policy = policies[sid]
                summary, public, private = analyze_scene(prepared, targets, reference, policy, cached, thresholds)
                write(output/f'scene-{sid}.json.gz', summary)
                write(output/f'points-{sid}.json.gz', public)
                write(output/f'private-evidence-{sid}.json.gz', private)
                rows.append(summary); pooled.extend(public['points'])
                state['completed_scenes'].append(sid)
                state['elapsed_seconds'] = time.monotonic()-started
                write(output/'run.json', state)
                print(f'103 completed scene {sid}; {len(public["points"])} original target/query rows', flush=True)
                del cached, prepared, reference, public, private
            state['observation_members_read'] = list(archive.members_read)
        result = dict(status='COMPLETED', scene_ids=list(SCENES), scenes=rows,
            pooled_selected_scenes=aggregate_points(pooled, thresholds), thresholds=provenance,
            interpretation='Postselected descriptive diagnosis; no method selection or performance experiment.')
        write(output/'summary.json.gz', result)
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started,
                     peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     peak_rss_unit='bytes' if sys.platform == 'darwin' else 'KiB')
    except BaseException as error:
        state.update(status='FAILED', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write(output/'run.json', state)
        raise
    write(output/'run.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--archive', nargs='+', required=True, help='ordered original Level1 ZIP parts')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--alignment', required=True)
    parser.add_argument('--threshold-summary')
    parser.add_argument('--output', required=True)
    def stop(signum, frame):
        raise InterruptedError(f'signal {signum}; preserve partial output, do not retry')
    signal.signal(signal.SIGTERM, stop)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
