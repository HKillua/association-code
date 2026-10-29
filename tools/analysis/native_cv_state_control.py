"""127: native-event CV filtering/smoothing, T-only scale choice and fixed V32.

The covariances below are declared working smoothing scales, not calibrated
sensor noise. No input event, support, output population or reference is repaired.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
from itertools import combinations
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.data.archive import MTADArchive, validate_manifest
from association.data.legacy_dataset import validate_archive
from association.data.motion_evidence import DATA, scene_motion_evidence
from association.data.native import _record
from association.data.reference_states import iter_reference_states
from association.data.source_events import source_events
from association.data.targets import build_scene_fusion_targets
from association.data.timed import prepare_timed_scene
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.native_motion import _short_delta
from association.evaluation.query_contribution import recover_group_states
from tools.analysis import source_preference_control as prior

qc, old = prior.qc, prior.old
IDEA = 'IDEA-NATIVE-CV-STATE-CONTROL-20261029'
CONTROL_RUN = 'LEVEL1-SOURCE-PREFERENCE-20261029-121-attempt-1'
METHODS = ('WS_MEAN', 'WS_LOW_SOURCE_SUPPORTED', 'WS_CV_FORWARD_MEAN',
           'WS_CV_FORWARD_LOW', 'WS_CV_RTS_MEAN', 'WS_CV_RTS_LOW')
COMPARISONS = tuple(combinations(METHODS, 2))
RHOS = tuple(10.**k for k in range(-10, -1))
RADIUS = 6371000.
NATIVE_KEYS = ('node_ids', 'sources', 'query_seconds', 'event_offsets', 'event_times_seconds',
               'event_values', 'event_valid', 'event_node_ids', 'event_canonical_indices')


def cv_transition(dt, rho):
    """Positive-time integrated white-acceleration CV working model."""
    if not np.isfinite(dt) or dt < 0 or not np.isfinite(rho) or rho <= 0:
        raise ValueError('finite nonnegative dt and positive rho required')
    eye = np.eye(2)
    f = np.block([[eye, dt*eye], [np.zeros((2, 2)), eye]])
    q = rho*np.block([[dt**3/3*eye, dt**2/2*eye], [dt**2/2*eye, dt*eye]])
    return f, q


def _covariance(p):
    p = (p+p.T)/2
    if not np.isfinite(p).all():
        raise FloatingPointError('nonfinite CV covariance')
    # Symmetrization only removes asymmetric roundoff; no diagonal jitter or
    # eigenvalue truncation changes the specified working model.
    np.linalg.cholesky(p)
    return p


def cv_native_positions(times, latlon, query, rho):
    """Prediction-only per-fragment API; native events, real seconds, no labels.

    Only query times INSIDE observation support enter the KF/RTS timeline.
    Outside queries use endpoint states after filtering/smoothing, never Q(dt<0).
    First observation defines the prior and is not used a second time.
    """
    times, latlon, query = (np.asarray(v, np.float64) for v in (times, latlon, query))
    if (times.ndim != 1 or len(times) < 2 or latlon.shape != (len(times), 2)
            or query.ndim != 1 or not np.isfinite(times).all() or (np.diff(times) <= 0).any()
            or not np.isfinite(latlon).all() or not np.isfinite(query).all()
            or (np.abs(latlon[:, 0]) > 90).any() or (np.abs(latlon[:, 1]) > 180).any()
            or (np.diff(query) <= 0).any()):
        raise ValueError('legal native positions/increasing seconds and query required')
    cv_transition(0., rho)
    angular_scale = RADIUS*np.pi/180
    east_scale = angular_scale*np.cos(np.deg2rad(latlon[:, 0].mean()))
    if not np.isfinite(east_scale) or east_scale <= 1e-9:
        raise ValueError('native chart has no finite usable east scale')
    unwrapped = latlon[0, 1]+np.r_[0., np.cumsum(_short_delta(np.diff(latlon[:, 1])))]
    y = np.column_stack(((latlon[:, 0]-latlon[0, 0])*angular_scale,
                         (unwrapped-latlon[0, 1])*east_scale))
    inside = (query >= times[0]) & (query <= times[-1])
    timeline = np.unique(np.r_[times, query[inside]])
    event_indices = np.searchsorted(timeline, times)
    observation_index = np.full(len(timeline), -1, np.int64)
    observation_index[event_indices] = np.arange(len(times))
    x = np.r_[y[0], 0., 0.]
    p = np.diag([1., 1., 1/3600., 1/3600.])
    h, r, eye = np.eye(4)[:2], np.eye(2), np.eye(4)
    xf, pf, xp, pp = (np.empty((len(timeline), 4)), np.empty((len(timeline), 4, 4)),
                       np.empty((len(timeline), 4)), np.empty((len(timeline), 4, 4)))
    xf[0], pf[0], xp[0], pp[0] = x, p, x, p
    for index in range(1, len(timeline)):
        f, q = cv_transition(timeline[index]-timeline[index-1], rho)
        x, p = f@x, _covariance(f@p@f.T+q)
        xp[index], pp[index] = x, p
        observed = observation_index[index]
        if observed >= 0:
            k = np.linalg.solve(h@p@h.T+r, h@p).T
            x = x+k@(y[observed]-h@x)
            residual = eye-k@h
            p = _covariance(residual@p@residual.T+k@r@k.T)
        xf[index], pf[index] = x, p
    xs = xf.copy()
    for index in range(len(timeline)-2, -1, -1):
        f, _ = cv_transition(timeline[index+1]-timeline[index], rho)
        gain = np.linalg.solve(pp[index+1], f@pf[index]).T
        xs[index] = xf[index]+gain@(xs[index+1]-xp[index+1])
    outputs = {}
    for name, states in (('forward', xf), ('rts', xs)):
        position = np.empty((len(query), 2))
        position[inside] = states[np.searchsorted(timeline, query[inside]), :2]
        before, after = query < times[0], query > times[-1]
        position[before] = states[0, :2]+(query[before]-times[0])[:, None]*states[0, 2:]
        position[after] = states[-1, :2]+(query[after]-times[-1])[:, None]*states[-1, 2:]
        value = np.column_stack((latlon[0, 0]+position[:, 0]/angular_scale,
            (latlon[0, 1]+position[:, 1]/east_scale+180.) % 360.-180.))
        if not np.isfinite(value).all() or (np.abs(value[:, 0]) > 90).any():
            raise FloatingPointError('CV query position is nonfinite or outside legal latitude')
        outputs[name] = value
    return outputs, dict(native_event_count=len(times), kalman_observation_updates=len(times)-1,
        transition_steps=len(timeline)-1, rts_backward_steps=len(timeline)-1,
        inside_query_count=int(inside.sum()), outside_query_count=int((~inside).sum()),
        first_observation_prior_not_reused=True, negative_transition_steps=0)


def pack_native_events(scene, events):
    """Ragged immutable physical cache, deliberately without identity/reference."""
    offsets = np.r_[0, np.cumsum([len(event['times']) for event in events])].astype(np.int64)
    payload = dict(node_ids=np.asarray(scene['node_ids'], str), sources=np.asarray(scene['sources'], str),
        query_seconds=np.asarray(scene['grid'], np.float64), event_offsets=offsets,
        event_times_seconds=np.concatenate([e['times'] for e in events]) if events else np.empty(0),
        event_values=np.concatenate([e['values'] for e in events]) if events else np.empty((0, 4)),
        event_valid=np.concatenate([e['valid'] for e in events]) if events else np.empty((0, 4), bool),
        event_node_ids=np.asarray([key[0] for e in events for key in e['event_ids']], str),
        event_canonical_indices=np.asarray([key[1] for e in events for key in e['event_ids']], np.int64))
    if set(payload) != set(NATIVE_KEYS):
        raise ValueError('native cache schema differs')
    return payload


def native_records(cache):
    """Yield one node's physical events from the publicised cache schema."""
    if set(cache) != set(NATIVE_KEYS):
        raise ValueError('native cache must have physical-only schema')
    n = len(cache['node_ids'])
    offsets = np.asarray(cache['event_offsets'])
    size = len(cache['event_times_seconds'])
    if (offsets.shape != (n+1,) or offsets.dtype.kind not in 'iu' or offsets[0] != 0
            or offsets[-1] != size or (np.diff(offsets) < 2).any()
            or np.shape(cache['event_values']) != (size, 4)
            or np.shape(cache['event_valid']) != (size, 4) or cache['event_valid'].dtype != bool
            or np.shape(cache['sources']) != (n,)
            or np.shape(cache['event_node_ids']) != (size,)
            or np.shape(cache['event_canonical_indices']) != (size,)):
        raise ValueError('complete native node/event population required')
    for node in range(n):
        use = slice(int(offsets[node]), int(offsets[node+1]))
        if (not cache['event_valid'][use, :2].all()
                or not np.all(cache['event_node_ids'][use] == cache['node_ids'][node])
                or not np.array_equal(cache['event_canonical_indices'][use], np.arange(use.stop-use.start))):
            raise ValueError('native canonical position/lineage invalid; no event deletion allowed')
        yield dict(node_id=str(cache['node_ids'][node]), source=str(cache['sources'][node]),
            times=cache['event_times_seconds'][use], values=cache['event_values'][use],
            valid=cache['event_valid'][use], canonical_indices=cache['event_canonical_indices'][use])


def predict_scene(cache, rho, check=None):
    outputs = {key: np.empty((len(cache['node_ids']), len(cache['query_seconds']), 2))
               for key in ('forward', 'rts')}
    counts = dict(native_event_count=0, kalman_observation_updates=0, transition_steps=0,
                  rts_backward_steps=0, inside_query_count=0, outside_query_count=0)
    for node, record in enumerate(native_records(cache)):
        prediction, audit = cv_native_positions(record['times'], record['values'][:, :2],
                                               cache['query_seconds'], rho)
        for key in outputs:
            outputs[key][node] = prediction[key]
        for key in counts:
            counts[key] += audit[key]
        if check is not None and node % 16 == 0:
            check()
    return outputs, counts


def _load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def export_native_cache(args, manifest, ids, output, check):
    """Read exactly declared T32/V32 observation and reference members once."""
    alignment = old.frozen.read(args.alignment)
    archive_paths = [str(Path(p).resolve()) for p in args.archive]
    rows = []
    with MTADArchive(archive_paths) as archive:
        validate_archive(archive, manifest)
        for split in ('T_core', 'V_select'):
            for subdir in ('native-events', 'native-evaluation', 'native-motion'):
                (output/'private'/subdir/split).mkdir(parents=True)
            with closing(iter_reference_states(archive_paths, manifest, split=split, scene_ids=ids[split])) as refs:
                for sid in ids[split]:
                    check()
                    native = _record(archive.load_scene(sid), manifest['components'][sid], split,
                        DATA['max_observations'], split_unit=manifest.get('split_unit'))
                    check(archive_read=True)
                    labels = archive.read_labels(sid)
                    target_ids = list(dict.fromkeys('-'.join(r['mmsi'].split('-')[:2]) for r in labels))
                    query = None
                    if not native['tracks']:
                        start, stop = min(r['t_s'] for r in labels), max(r['t_e'] for r in labels)
                        if stop <= start:
                            raise ValueError('empty scene cannot invent positive query range')
                        query = np.linspace(start, stop, 64)
                    scene = prepare_timed_scene(native, **DATA, query_times=query)
                    motion = scene_motion_evidence(scene, motion_observation_policy='canonical_all')
                    cache = pack_native_events(scene, source_events(scene))
                    reference = next(refs)
                    check(reference_read=True)
                    if reference['scene_id'] != sid or alignment[sid]['max_gap_seconds'] != 600.:
                        raise ValueError('frozen scene/alignment policy differs')
                    targets = build_scene_fusion_targets(scene, reference, target_identity_ids=target_ids,
                        longitude_policy='shortest_arc', **alignment[sid])
                    evaluation = dict(node_ids=np.asarray(scene['node_ids'], str), identities=np.asarray(scene['identities'], str),
                        target_ids=np.asarray(target_ids, str), query_seconds=cache['query_seconds'],
                        target_states=np.asarray([targets[t]['states'] for t in target_ids], np.float64).reshape(-1, 64, 4),
                        target_valid=np.asarray([targets[t]['valid'] for t in target_ids], bool).reshape(-1, 64, 4))
                    for subdir, payload in (('native-events', cache), ('native-evaluation', evaluation),
                            ('native-motion', dict(motion, node_ids=np.asarray(scene['node_ids'], str),
                                                   pairs=np.asarray(scene['pairs'], np.int64)))):
                        np.savez_compressed(output/'private'/subdir/split/f'scene-{sid}.npz', **payload)
                    rows.append(dict(scene_id=sid, split=split, component_id=str(manifest['components'][sid]),
                        node_count=len(scene['node_ids']), native_event_count=len(cache['event_times_seconds']),
                        target_count=len(target_ids), valid_reference_position_points=int(evaluation['target_valid'][..., :2].all(-1).sum()),
                        raw_reference_issue_count=len(reference['issues'])))
                    archive._label_cache.pop(sid, None)
                    check(exported_native_scenes=len(rows), last_exported_scene=sid)
    payload = dict(status='CACHE_COMPLETED', idea_id=IDEA, schema_version='native-events-ragged-v1',
        physical_keys=list(NATIVE_KEYS), identity_reference_stored_separately=True,
        scene_ids=ids, rows=rows, test_opened=False, native_motion_policy='canonical_all',
        archive_paths=archive_paths, alignment_path=str(Path(args.alignment).resolve()))
    old.frozen.write(output/'native-cache.json', payload)
    return payload


def fit_scale(output, ids, check):
    rows, updates = [], {key: 0 for key in ('native_event_count', 'kalman_observation_updates',
        'transition_steps', 'rts_backward_steps', 'inside_query_count', 'outside_query_count')}
    fits = 0
    for sid in ids:
        cache = _load_npz(output/'private/native-events/T_core'/f'scene-{sid}.npz')
        ev = _load_npz(output/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
        registry = {key: i for i, key in enumerate(ev['target_ids'])}
        truth = ev['target_states'][[registry[key] for key in ev['identities']]]
        valid = ev['target_valid'][[registry[key] for key in ev['identities']], :, :2].all(-1)
        maes = np.full((len(RHOS), len(cache['node_ids'])), np.nan)
        for index, rho in enumerate(RHOS):
            pred, counts = predict_scene(cache, rho, check)
            fits += len(cache['node_ids'])
            padded = np.concatenate((pred['rts'], np.zeros((*pred['rts'].shape[:2], 2))), axis=-1)
            error = old.attribute_errors(padded, truth)['position']
            for node in range(len(truth)):
                if valid[node].any():
                    maes[index, node] = error[node, valid[node]].mean()
            for key in updates:
                updates[key] += counts[key]
            check(phase='T_scale_fit', last_training_scene=sid, last_training_rho=rho,
                  analytic_filter_smoother_fits=fits)
        eligible = valid.any(1)
        criteria = maes[:, eligible].mean(1) if eligible.any() else None
        if criteria is not None and not np.isfinite(criteria).all():
            raise FloatingPointError('nonfinite T-side scale objective')
        np.savez_compressed(output/'private'/f't-scale-{sid}.npz', node_mae_by_rho=maes,
                            valid_reference_query_counts=valid.sum(1))
        rows.append(dict(scene_id=sid, fragment_count=len(truth), eligible_reference_fragments=int(eligible.sum()),
            missing_reference_fragments=int((~eligible).sum()), objective_by_rho=None if criteria is None else criteria.tolist()))
    defined = [r['objective_by_rho'] for r in rows if r['objective_by_rho'] is not None]
    if not defined:
        raise ValueError('no T-side reference-supported fragment; cannot select rho')
    scores = np.mean(defined, axis=0)
    selected = int(np.argmin(scores))
    result = dict(rhos=list(RHOS), objective_by_rho=scores.tolist(), selected_rho=RHOS[selected],
        selection='RTS scene-equal then eligible-fragment-equal then valid-query MAE; exact ties lower rho',
        forward_uses_same_rho=True, scene_count=len(rows), eligible_scene_count=len(defined),
        missing_reference_scene_count=len(rows)-len(defined), per_scene=rows, analytic_work=updates,
        analytic_filter_smoother_fits=fits)
    old.frozen.write(output/'scale-selection.json', result)
    return result


def low_weights(support, sources, groups):
    weights = np.zeros(support.shape, np.float64)
    order = sorted(set(sources.tolist()))
    if not 1 <= len(order) <= 2:
        raise ValueError('original one/two-source population required')
    for group in groups:
        members = np.asarray(group)
        low = members[sources[members] == order[0]]
        high = members[sources[members] == order[1]] if len(order) == 2 else np.empty(0, np.int64)
        lc, hc = support[low].sum(0), support[high].sum(0)
        weights[low] = support[low]/np.maximum(lc, 1)
        weights[high] = support[high]*(lc == 0)/np.maximum(hc, 1)
    return weights


def analyze_scene(fragment, groups, retained, predictions, endpoint_seconds):
    """All predictions already frozen; evaluation alone receives identities/truth."""
    targets = fragment['target_identity_ids']
    saved = retained['WS_MEAN']
    old._same(saved['target_ids'], targets, '121 full target registry')
    truth, valid = saved['aligned_truth'][:len(targets)], saved['aligned_truth_valid'][:len(targets)]
    regions = old.fixed_regions(fragment, targets)
    registry = {key: i for i, key in enumerate(targets)}
    exact = np.zeros((len(targets), 64), bool)
    distance = np.full((len(targets), 64), np.inf)
    for node, identity in enumerate(fragment['identities']):
        old._same(fragment['target'][node], truth[registry[identity]], '056/121 reference')
        old._same(fragment['target_valid'][node], valid[registry[identity]], '056/121 validity')
        exact[registry[identity]] |= fragment['exact_observation'][node]
        distance[registry[identity]] = np.minimum(distance[registry[identity]], endpoint_seconds[node])
    regions.update(exact_native=exact, endpoint_le_60s=distance <= 60,
        endpoint_gt60_le600s=(distance > 60) & (distance <= 600),
        endpoint_gt600s=(distance > 600) & np.isfinite(distance))
    artifacts, methods, arrays = {}, {}, {}
    for name in METHODS:
        outputs = emit_group_outputs(predictions[name], groups, node_count=len(fragment['native_states']))
        report = old.evaluate_full_scene(outputs, node_identities=fragment['identities'], target_identities=targets,
            target_states=truth, target_valid=valid, node_observed=fragment['support'], split='V_select',
            grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
        artifact = report.pop('artifact')
        if name in retained:
            if set(artifact) != set(retained[name]):
                raise ValueError('121 complete control artifact schema differs')
            for key in artifact:
                old._same(artifact[key], retained[name][key], '121 exact '+name+'/'+key)
        else:
            for key in saved:
                if key not in ('aligned_prediction', 'group_states'):
                    old._same(artifact[key], saved[key], 'fixed full population '+name+'/'+key)
            old._same(artifact['group_states'][..., 2:], saved['group_states'][..., 2:], 'unchanged speed/heading')
        methods[name], arrays[name] = qc.describe(report, artifact, fragment, regions)
        artifacts[name] = artifact
    pairs, paired = {}, {}
    for baseline, candidate in COMPARISONS:
        key = candidate+'_vs_'+baseline
        pairs[key], paired[key], _ = old.compare_pair(artifacts[baseline], artifacts[candidate], regions)
        pairs[key].update(baseline_method=baseline, candidate_method=candidate,
                         side_labels=dict(A0=baseline, candidate=candidate))
    return dict(methods=methods, pairwise=pairs, fixed_R_population_invariants_verified=True,
                original121_two_controls_exact=True), dict(methods=arrays, pairs=paired), artifacts


def run(args):
    source, partition, control, output = (Path(getattr(args, key)).resolve()
        for key in ('source_system', 'partition_run', 'source_preference_run', 'output'))
    for protected in (source.parent, partition, control):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new output may not overlap immutable inputs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0', source_system=str(source),
        partition_run=str(partition), source_preference_run=str(control), started_at=datetime.now(timezone.utc).isoformat(),
        methods=list(METHODS), rhos=list(RHOS), cpu_threads=args.cpu_threads, completed_scenes=0,
        models_constructed=0, model_forwards=0, optimizer_updates=0, gpu_used=False,
        neural_models_constructed=0, analytic_filter_smoother_fits=0,
        scores_recomputed=False, groups_recomputed=False, threshold_reselected=False,
        normalizer_refitted=False, archive_read=False, reference_read=False, test_opened=False,
        source_bias_correction_applied=False, covariances_are_working_scales=True,
        imported_prediction_scene_arms=0, newly_computed_prediction_scene_arms=0)
    def check(**extra):
        state.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes(),
            advisory_exceeded=time.monotonic()-started > 7200, **extra)
        old.frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > old.frozen.RAM_BUDGET:
            raise MemoryError('4GiB resident-memory budget exceeded')
    result = None
    try:
        check()
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([old.frozen.MAX_AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
            state['address_space_limit_bytes'] = limit
            if hasattr(os, 'sched_getaffinity'):
                cpus = sorted(os.sched_getaffinity(0))[:args.cpu_threads]
                os.sched_setaffinity(0, cpus)
                state['cpu_affinity'] = cpus
        manifest = old.frozen.read(args.manifest)
        validate_manifest(manifest)
        meta = old.metadata(source, partition, manifest)
        cr, cs = (old.frozen.read(control/name) for name in ('run.json', 'summary.json'))
        if (control.name != CONTROL_RUN or cr.get('status') != cs.get('status') or cr.get('status') != 'COMPLETED'
                or cr.get('exit_code') != 0 or cr.get('idea_id') != prior.IDEA or cs.get('idea_id') != prior.IDEA
                or cr.get('plan_version') != '1.0' or cs.get('plan_version') != '1.0'
                or cr.get('methods') != list(prior.METHODS) or cr.get('completed_scenes') != len(meta['ids'])
                or cr.get('scene_ids') != meta['ids']
                or cs.get('scene_ids') != meta['ids'] or Path(cr['source_system']).resolve() != source
                or Path(cr['partition_run']).resolve() != partition):
            raise ValueError('completed original121 same V32/source/partition required')
        ids = dict(T_core=manifest['splits']['T_core'][:32], V_select=meta['ids'])
        if len(ids['T_core']) != 32 or len(ids['V_select']) != 32:
            raise ValueError('fixed T32 and original V32 required')
        check(training_scene_ids=ids['T_core'], scene_ids=ids['V_select'], total_scenes=32,
              phase='native_export', archive_paths=[str(p) for p in args.archive])
        export_native_cache(args, manifest, ids, output, check)
        check(phase='T_scale_fit', native_cache_status='CACHE_COMPLETED')
        selection = fit_scale(output, ids['T_core'], check)
        rho = selection['selected_rho']
        check(phase='V_evaluation', selected_rho=rho)
        (output/'scenes').mkdir()
        rows, arrays, work = [], [], []
        for sid in ids['V_select']:
            fragment, _ = old._load_arrays(source, sid)
            cache = _load_npz(output/'private/native-events/V_select'/f'scene-{sid}.npz')
            motion = _load_npz(output/'private/native-motion/V_select'/f'scene-{sid}.npz')
            for key in ('node_ids', 'sources', 'query_seconds'):
                old._same(cache[key], fragment[key], 'raw/056 '+key)
            for key, cached in (('native_states', 'states'), ('support', 'support'),
                    ('nearest_seconds', 'nearest_seconds'), ('exact_observation', 'exact_observation')):
                old._same(motion[cached], fragment[key], 'raw/056 '+key)
            membership = np.asarray(old.frozen.read(partition/'private'/f'scene-{sid}.json.gz')['memberships']['WS'])
            groups = old.groups_from_membership(membership, len(fragment['native_states']))
            saved = _load_npz(control/'private'/f'states-{sid}.npz')
            fields = {key[len('WS_MEAN_'):] for key in saved if key.startswith('WS_MEAN_')}
            retained = {name: {key: saved[name+'_'+key] for key in fields} for name in METHODS[:2]}
            weights = low_weights(fragment['support'], fragment['sources'], groups)
            old._same(weights, saved['source_preference_query_weights'], '121 legal LOW contribution')
            mean = old.ordinary_fusion(fragment['native_states'], fragment['nearest_seconds'],
                                      fragment['support'], groups)['predictions']['mean_supported']
            first = np.asarray([group[0] for group in groups], np.int64)
            predictions = dict(WS_MEAN=mean[first])
            low = recover_group_states(fragment['native_states'], weights, groups)['group_states']
            low[..., 2:] = predictions['WS_MEAN'][..., 2:]
            predictions['WS_LOW_SOURCE_SUPPORTED'] = low
            positions, audit = predict_scene(cache, rho, check)
            endpoint_seconds = np.empty(fragment['support'].shape)
            for node, record in enumerate(native_records(cache)):
                endpoint_seconds[node] = np.minimum(np.abs(cache['query_seconds']-record['times'][0]),
                                                   np.abs(cache['query_seconds']-record['times'][-1]))
            for label, key in (('FORWARD', 'forward'), ('RTS', 'rts')):
                state_input = fragment['native_states'].copy()
                state_input[..., :2] = positions[key]
                new_mean = old.ordinary_fusion(state_input, fragment['nearest_seconds'], fragment['support'],
                                              groups)['predictions']['mean_supported'][first]
                new_low = recover_group_states(state_input, weights, groups)['group_states']
                new_mean[..., 2:] = predictions['WS_MEAN'][..., 2:]
                new_low[..., 2:] = predictions['WS_MEAN'][..., 2:]
                predictions['WS_CV_'+label+'_MEAN'] = new_mean
                predictions['WS_CV_'+label+'_LOW'] = new_low
            # All six predictions are frozen before evaluating or inspecting
            # the separately exported V reference/identity alignment values.
            ev = _load_npz(output/'private/native-evaluation/V_select'/f'scene-{sid}.npz')
            old._same(ev['node_ids'], fragment['node_ids'], 'raw/056 evaluation node order')
            old._same(ev['identities'], fragment['identities'], 'raw/056 evaluation identities')
            old._same(ev['target_ids'], fragment['target_identity_ids'], 'raw/056 target registry')
            registry = {key: i for i, key in enumerate(ev['target_ids'])}
            for node, identity in enumerate(ev['identities']):
                old._same(ev['target_states'][registry[identity]], fragment['target'][node], 'raw/056 target')
                old._same(ev['target_valid'][registry[identity]], fragment['target_valid'][node], 'raw/056 validity')
            row, error_arrays, artifacts = analyze_scene(fragment, groups, retained, predictions, endpoint_seconds)
            row.update(scene_id=sid, component_id=str(meta['components'][sid]), node_count=len(cache['node_ids']),
                       native_cv_audit=audit, selected_rho=rho)
            old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
            np.savez_compressed(output/'private'/f'states-{sid}.npz',
                **{name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()})
            rows.append(row)
            arrays.append(error_arrays)
            work.append(audit)
            check(completed_scenes=len(rows), last_scene_id=sid,
                analytic_filter_smoother_fits=selection['analytic_filter_smoother_fits']+sum(r['node_count'] for r in rows),
                imported_prediction_scene_arms=2*len(rows), newly_computed_prediction_scene_arms=4*len(rows))
        components = sorted({row['component_id'] for row in rows})
        largest = max(components, key=lambda cid: sum(row['component_id'] == cid for row in rows))
        def subset(indices):
            return qc.aggregate([rows[i] for i in indices], [arrays[i] for i in indices], method_names=METHODS) if indices else None
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0',
            **qc.aggregate(rows, arrays, method_names=METHODS), selected_rho=rho, scale_selection=selection,
            training_scene_ids=ids['T_core'],
            components={cid: subset([i for i, row in enumerate(rows) if row['component_id'] == cid]) for cid in components},
            without_largest_component=subset([i for i, row in enumerate(rows) if row['component_id'] != largest]),
            original121_two_controls_exact=True, fixed_R_population_invariants_verified=True,
            imported_scene_method_count=64, new_scene_method_count=128,
            analytic_filter_smoother_fits=selection['analytic_filter_smoother_fits']+sum(r['node_count'] for r in rows),
            V_analytic_work={key: sum(item[key] for item in work) for key in work[0]},
            no_expansion=True, covariances_are_working_scales=True,
            limitations=['T-only supervised working rho is not a measured noise covariance',
                'offline full-window chart/RTS and frozen complete-scene WS, not strict online fusion',
                'repeated development V32 and shared identities, not independent testing',
                'no shared bias/clock correction or learned weights; no causal mechanism or peer victory'])
        old.frozen.write(output/'scene-summary.json.gz', rows)
        check(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        old.frozen.write(output/'summary.json', result)
        check(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=old.frozen.peak_rss_bytes(), finished_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'run.json', state)
        if result is not None and (output/'summary.json').exists():
            result.update(status=state['status'], error=state['error'])
            old.frozen.write(output/'summary.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', nargs='+', type=Path, required=True)
    for key in ('manifest', 'alignment', 'source-system', 'partition-run', 'source-preference-run', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        return 1
    print(f'COMPLETED: {args.output}/summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
