"""131: extend the four frozen125 value heads to the remaining268 V scenes.

The original32 predictions/public rows are imported, never reconstructed.
Only native quality evidence, soft75 and four frozen policy actions are new.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[thread_setting] = '4'

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.data.archive import MTADArchive
from association.data.legacy_dataset import validate_archive
from association.data.motion_evidence import DATA, scene_motion_evidence
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from tools.analysis import conditional_source_value as pilot

old, qc = pilot.old, pilot.qc
IDEA = 'IDEA-CONDITIONAL-SOURCE-VALUE-FULL-20261029'
METHODS = pilot.METHODS
RAM, AS = 6*1024**3, 12*1024**3
ACTION_INPUTS = ('states', 'support', 'nearest_seconds', 'sources', 'native_count',
                 'fit_rmse_m', 'ols_velocity_mps')


def load_frozen_heads(path):
    """Restore the exact125 array schema; fixed names determine head mode.

    125 did not serialize a separate mode string. No estimator is fitted here.
    Its T-only mean/std and all coefficients are used bit-for-bit.
    """
    expected = {h+'_'+k for h in pilot.HEADS for k in ('mean', 'std', 'beta', 'permutation')}
    models = {}
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != expected:
            raise ValueError('exact125 full-T-head array schema required')
        for head in pilot.HEADS:
            model = {k: np.asarray(saved[head+'_'+k]).copy() for k in ('mean', 'std', 'beta', 'permutation')}
            size = 1 if head == 'TMEAN' else 16 if head == 'LINEAR' else 136
            if (model['mean'].shape != (15,) or model['std'].shape != (15,)
                    or model['beta'].shape != (size,)
                    or any(model[k].dtype != np.float64 or not np.isfinite(model[k]).all()
                           for k in ('mean', 'std', 'beta'))
                    or (model['std'] < 1e-6).any() or model['permutation'].ndim != 1
                    or model['permutation'].dtype.kind not in 'iu'):
                raise ValueError('invalid125 frozen head shape/dtype/finite parameter')
            n = len(model['permutation'])
            expected_order = np.random.default_rng(20).permutation(n) if head == 'SHUFFLED' else np.arange(n)
            old._same(model['permutation'], expected_order, '125 fixed training permutation '+head)
            model.update(head=head, training_label_count=n, no_labels_fallback=not n)
            if not n:
                old._same(model['beta'], np.zeros(size), '125 no-label hard fallback')
            models[head] = model
    for head in pilot.HEADS[1:]:
        for key in ('mean', 'std'):
            old._same(models[head][key], models['TMEAN'][key], '125 common frozen '+key)
        if models[head]['training_label_count'] != models['TMEAN']['training_label_count']:
            raise ValueError('125 four heads must use the same fullT label population')
    return models


def metadata(source, previous, control, prior, manifest):
    meta = old.metadata(source, previous, manifest)
    control_run, control_summary = (old.frozen.read(control/p) for p in ('run.json', 'summary.json'))
    prior_run, prior_summary = (old.frozen.read(prior/p) for p in ('run.json', 'summary.json'))
    train_ids = manifest['splits']['T_core'][:32]
    for run, summary, idea, ids, completed in (
            (control_run, control_summary, pilot.mix.FULL_IDEA, meta['all_ids'], 'completed_scenes'),
            (prior_run, prior_summary, pilot.IDEA, meta['ids'], 'completed_V_scenes')):
        if (run.get('status') != summary.get('status') or run.get('status') != 'COMPLETED'
                or run.get('exit_code') != 0 or run.get('idea_id') != summary.get('idea_id')
                or run.get('idea_id') != idea or run.get('plan_version') != '1.0'
                or summary.get('plan_version') != '1.0' or run.get('scene_ids') != ids
                or summary.get('scene_ids') != ids or run.get(completed) != len(ids)
                or Path(run['source_system']).resolve() != source
                or Path(run['partition_run']).resolve() != previous
                or not summary.get('fixed_R_population_invariants_verified')):
            raise ValueError('completed original123/125 same-source/population required')
    if (control_run.get('original32_ids') != meta['ids']
            or control_run.get('methods') != list(pilot.mix.p.METHODS)
            or prior_run.get('methods') != list(METHODS)
            or prior_run.get('feature_names') != list(pilot.FEATURE_NAMES)
            or prior_run.get('threshold') != pilot.THRESHOLD
            or prior_run.get('training_scene_ids') != train_ids
            or prior_run.get('completed_T_scenes') != len(train_ids)
            or Path(prior_run['source_control_run']).resolve() != control
            or not prior_summary.get('full_T_models_frozen_before_V')
            or not prior_run.get('upstream_teacher_trained_on_all_T')
            or not prior_run.get('upstream_threshold_historically_V_developed')
            or prior_summary.get('optimizer_updates') != 0):
        raise ValueError('exact frozen125 head recipe and original123 lineage required')
    if len(meta['ids']) != 32 or len(meta['all_ids']) != 300:
        raise ValueError('fixed original32/full300 population required')
    meta.update(pilot_ids=meta['ids'], ids=meta['all_ids'], training_ids=train_ids,
        training_components={str(manifest['components'][sid]) for sid in train_ids})
    return meta


def rebuild_native_motion(archive, manifest, sid, source):
    """Original native factory; no reference archive reader or Q64 refitting.

    Raw loader metadata contains identities/pair labels. They are not passed to
    scene_motion_evidence or the allowlisted value-head prediction API.
    """
    if sid not in manifest['splits']['V_select']:
        raise ValueError('131 permits V_select only; T/test are not opened')
    native = _record(archive.load_scene(sid), manifest['components'][sid], 'V_select',
                     DATA['max_observations'], split_unit=manifest.get('split_unit'))
    query = None
    if not native['tracks']:
        labels = archive.read_labels(sid)
        if not labels:
            raise ValueError('empty native scene has no declared query bounds')
        query = np.linspace(min(r['t_s'] for r in labels), max(r['t_e'] for r in labels), 64)
        if query[-1] <= query[0]:
            raise ValueError('empty scene query bounds invalid')
    scene = prepare_timed_scene(native, **DATA, query_times=query)
    motion = scene_motion_evidence(scene, motion_observation_policy='canonical_all')
    nodes, pairs = np.asarray(scene['node_ids'], str), np.asarray(scene['pairs'], np.int64)
    with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as original:
        old._same(nodes, original['node_ids'], '056 canonical node order')
        for key, cached in (('states', 'native_states'), ('sources', 'sources'),
                ('query_seconds', 'query_seconds'), ('support', 'support'),
                ('nearest_seconds', 'nearest_seconds'), ('exact_observation', 'exact_observation')):
            old._same(motion[key], original[cached], '056 exact native '+key)
    old._same(pairs, np.column_stack(np.triu_indices(len(nodes), 1)), 'all original unordered pairs')
    return {k: motion[k] for k in ACTION_INPUTS}, nodes, pairs, dict(
        node_count=len(nodes), canonical_event_count=int(np.asarray(motion['native_count']).sum()),
        canonical_all=True, physical_binding_exact=True, native_quality_reconstructed_from_events=True)


def regions_from_both(fragment, artifact, both):
    targets = artifact['target_ids']; regions = old.fixed_regions(fragment, targets)
    selected = artifact['aligned_group_indices'][:len(targets)]
    matched = selected >= 0
    region = np.zeros((len(targets), 64), bool)
    region[matched] = both[selected[matched]]
    regions['predicted_group_both_sources'] = region
    return regions


def import_pilot_scene(path, sid, fragment):
    """Import public data and prediction artifacts; extract errors for pooling.

    No candidate construction, model inference or emitted output is performed.
    Public methods/pairs remain the exact original125 JSON values.
    """
    row = old.frozen.read(path/'scenes'/f'scene-{sid}.json.gz')
    artifacts = pilot.full.load_artifacts(path/'private'/f'states-{sid}.npz', METHODS)
    if (row['scene_id'] != sid or set(row['methods']) != set(METHODS)
            or len(row['pairwise']) != 55 or not row.get('fixed_R_population_invariants_verified')):
        raise ValueError('complete original125 scene/public schema required')
    with np.load(path/'private'/f'V-features-{sid}.npz', allow_pickle=False) as saved:
        both = saved['both']
    regions = regions_from_both(fragment, artifacts['WS_MEAN'], both)
    arrays = dict(methods={}, pairs={})
    for name, artifact in artifacts.items():
        _, _, errors, own = old.target_arrays(artifact)
        arrays['methods'][name] = {region: errors[own & mask] for region, mask in regions.items()}
        for region, values in arrays['methods'][name].items():
            pilot.mix.p._close(row['methods'][name]['fixed_target_strata'][region], old.stats(values),
                               '125 imported method '+name+'/'+region)
    for base, candidate in combinations(METHODS, 2):
        key = candidate+'_vs_'+base
        comparison, arrays['pairs'][key], _ = old.compare_pair(artifacts[base], artifacts[candidate], regions)
        comparison.update(baseline_method=base, candidate_method=candidate,
                          side_labels=dict(A0=base, candidate=candidate))
        pilot.mix.p._close(comparison, row['pairwise'][key], '125 imported pair '+key)
    return row, arrays, artifacts


def new_scene_predictions(motion, nodes, pairs, scores, membership, source_path, models):
    """No identity/ref fields accepted or forwarded to legal policy inputs."""
    if set(motion) != set(ACTION_INPUTS):
        raise ValueError('strict physical prediction allowlist required')
    groups = old.groups_from_membership(membership, len(nodes))
    with np.load(source_path, allow_pickle=False) as saved:
        predictions = {name: saved[name+'_group_states'].copy() for name in pilot.BASES}
        calibration = {key: saved[key] for key in pilot.mix.p.CALIBRATION_KEYS}
        for name in pilot.BASES:
            old._same(saved[name+'_node_group_indices'], membership, '123 imported membership '+name)
    x, both, hard, soft = pilot.make_actions_features(**motion, groups=groups, calibration=calibration,
        pairs=pairs, scores=scores, mean_states=predictions['WS_MEAN'],
        imported_hard=predictions['WS_FIXED_LOW_SOURCE_SUPPORTED'])
    old._same(hard, predictions['WS_FIXED_LOW_SOURCE_SUPPORTED'], '123 hard exact')
    added, predicted_gains, gates = pilot.apply_heads(models, x, both, hard, soft)
    predictions.update({pilot.SOFT: soft}, **added)
    return groups, predictions, x, both, predicted_gains, gates, calibration


def compare_imported_controls(row, original):
    # 123 also reported low-only/high-only/no-support strata;125's eleven-arm
    # schema contains the common fixed regions plus both. Preserve125's schema.
    for name in pilot.BASES:
        current, previous = row['methods'][name], original['methods'][name]
        for key in ('population', 'unmatched_target_count', 'position', 'paper_position'):
            pilot.mix.p._close(current[key], previous[key], '123 imported '+name+'/'+key)
        for region in current['fixed_target_strata']:
            pilot.mix.p._close(current['fixed_target_strata'][region], previous['fixed_target_strata'][region],
                               '123 imported '+name+'/'+region)


def aggregate(rows, arrays):
    result = qc.aggregate(rows, arrays, method_names=METHODS)
    result.update(scene_prediction_arms_imported=sum(len(r['imported_prediction_methods']) for r in rows),
                  scene_prediction_arms_computed=sum(len(r['newly_computed_prediction_methods']) for r in rows))
    return result


def run(args):
    paths = {k: Path(getattr(args, k)).resolve() for k in
             ('source_system', 'partition_run', 'source_control_run', 'pilot_run', 'output')}
    output, source = paths['output'], paths['source_system']
    archives = [str(Path(p).resolve()) for p in args.archive]
    if len(archives) != 3 or len(set(archives)) != 3:
        raise ValueError('three distinct original Level1 archive volumes required')
    for protected in (source.parent, paths['partition_run'], paths['source_control_run'], paths['pilot_run']):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('new output must not overlap frozen input runs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('CPU threads1..4')
    output.mkdir(parents=True, exist_ok=False); (output/'private').mkdir(); (output/'scenes').mkdir()
    started = time.monotonic(); before = resource.getrusage(resource.RUSAGE_SELF)
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(),
        **{key: str(value) for key, value in paths.items() if key != 'output'},
        archive_paths=archives, methods=list(METHODS), feature_names=list(pilot.FEATURE_NAMES),
        cpu_threads=args.cpu_threads, models_constructed=0, teacher_forward_calls=0, model_forwards=0,
        optimizer_updates=0, analytic_head_fits=0, ridge_system_solves=0,
        scores_recomputed=False, groups_recomputed=False, threshold_reselected=False,
        normalizer_refitted=False, centers_refitted=False, head_reselected=False,
        completed_scenes=0, imported_prediction_scene_arms=0, newly_computed_prediction_scene_arms=0,
        completed_native_scenes=0, canonical_event_count=0, gpu_used=False, test_opened=False,
        archive_read=False, csv_read=False, reference_archive_read=False, cached_reference_read=False,
        raw_loader_reads_identity_metadata=True, prediction_api_accepts_identity_reference=False,
        restored_head_count=0, declared_primary_head='QUADRATIC', analytic_head_application_count=0,
        analytic_query_prediction_count=0, neural_model_forwards=0)
    def check(**extra):
        state.update(extra, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes(),
            user_cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime-before.ru_utime,
            system_cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime-before.ru_stime,
            advisory_exceeded=time.monotonic()-started > 5400)
        old.frozen.write(output/'run.json', state)
        if state['advisory_exceeded'] and not (output/'time_advisory.json').exists():
            old.frozen.write(output/'time_advisory.json', dict(elapsed_seconds=state['elapsed_seconds'],
                action='continue only fixed remaining268; no refit, retry or automatic successor'))
        if state['peak_rss_bytes'] > RAM:
            raise MemoryError('6GiB resident budget exceeded')
    result = None
    try:
        check(phase='binding')
        import torch
        torch.set_num_threads(args.cpu_threads)
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard)); state['address_space_limit_bytes'] = limit
            if hasattr(os, 'sched_getaffinity'):
                cpus = sorted(os.sched_getaffinity(0))[:args.cpu_threads]
                os.sched_setaffinity(0, cpus); state['cpu_affinity'] = cpus
        manifest = old.frozen.read(args.manifest)
        meta = metadata(source, paths['partition_run'], paths['source_control_run'], paths['pilot_run'], manifest)
        models = load_frozen_heads(paths['pilot_run']/'private'/'full-T-heads.npz')
        np.savez_compressed(output/'private'/'full-T-heads.npz',
            **{h+'_'+k: models[h][k] for h in pilot.HEADS for k in ('mean', 'std', 'beta', 'permutation')})
        old._same(np.asarray(pilot.HEADS), np.asarray(list(models)), 'four original head modes')
        v_components = {str(meta['components'][sid]) for sid in meta['pilot_ids']}
        check(scene_ids=meta['ids'], original32_ids=meta['pilot_ids'], training_scene_ids=meta['training_ids'],
            training_component_ids=sorted(meta['training_components']), restored_head_count=4,
            restored_head_coefficient_counts={h: len(models[h]['beta']) for h in models},
            restored_T_label_count=models['TMEAN']['training_label_count'], total_scenes=300)
        rows, arrays = [], []
        with MTADArchive(archives) as archive, np.load(meta['association']/'final-scores.npz', allow_pickle=False) as scores:
            validate_archive(archive, manifest)
            old._same(scores['scene_ids'], meta['ids'], '056 allV score scene order')
            for sid in meta['ids']:
                check(phase='V_import' if sid in meta['pilot_ids'] else 'V_native_frozen_policy', last_scene_id=sid)
                cid = str(meta['components'][sid]); original = sid in meta['pilot_ids']
                if original:
                    fragment, _ = old._load_arrays(source, sid)
                    state['cached_reference_read'] = True
                    row, error_arrays, artifacts = import_pilot_scene(paths['pilot_run'], sid, fragment)
                    if row['component_id'] != cid or row['node_count'] != len(fragment['node_ids']):
                        raise ValueError('125 original scene/component/node binding differs')
                else:
                    motion, nodes, pairs, native_audit = rebuild_native_motion(archive, manifest, sid, source)
                    state.update(archive_read=True, csv_read=True)
                    old._same(nodes, scores[sid+'_node_ids'], '056 score node order')
                    old._same(pairs, scores[sid+'_pairs'], '056 score pairs')
                    old._same(motion['sources'], scores[sid+'_sources'], '056 score sources')
                    membership = np.asarray(old.frozen.read(paths['partition_run']/'private'/f'scene-{sid}.json.gz')['memberships']['WS'])
                    source_path = paths['source_control_run']/'private'/f'states-{sid}.npz'
                    groups, predictions, x, both, gains, gates, calibration = new_scene_predictions(
                        motion, nodes, pairs, scores[sid+'_scores'], membership, source_path, models)
                    # The new decisions are persistent before explicit V evaluation reads.
                    np.savez_compressed(output/'private'/f'V-features-{sid}.npz', features=x, both=both,
                        node_ids=nodes, native_count=motion['native_count'], fit_rmse_m=motion['fit_rmse_m'],
                        ols_velocity_mps=motion['ols_velocity_mps'], **calibration,
                        **{h+'_predicted_gain': value for h, value in gains.items()},
                        **{h+'_gate': value for h, value in gates.items()})
                    retained = pilot.full.load_artifacts(source_path, pilot.BASES)
                    fragment, _ = old._load_arrays(source, sid)
                    state['cached_reference_read'] = True
                    row, error_arrays, artifacts = pilot.evaluate_v(fragment, groups, predictions, retained, both)
                    control_public = old.frozen.read(paths['source_control_run']/'scenes'/f'scene-{sid}.json.gz')
                    if control_public['scene_id'] != sid or control_public['component_id'] != cid:
                        raise ValueError('123 scene binding differs')
                    compare_imported_controls(row, control_public)
                    row.update(scene_id=sid, component_id=cid, node_count=len(nodes),
                        fixed_R_population_invariants_verified=True, native_input_audit=native_audit,
                        policy_action_counts={h: int(gate.sum()) for h, gate in gates.items()})
                    state['completed_native_scenes'] += 1
                    state['canonical_event_count'] += native_audit['canonical_event_count']
                    state['analytic_head_application_count'] += 4
                    state['analytic_query_prediction_count'] += 4*int(both.size)
                    archive._label_cache.pop(sid, None)
                row.update(stratum='original32' if original else 'remaining268',
                    component_seen_in_original32=cid in v_components,
                    component_seen_in_T32_head_training=cid in meta['training_components'],
                    imported_prediction_methods=list(METHODS if original else pilot.BASES),
                    newly_computed_prediction_methods=[] if original else [pilot.SOFT]+list(pilot.HEAD_METHODS),
                    original125_public_and_artifacts_imported_exact=original)
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                np.savez_compressed(output/'private'/f'states-{sid}.npz',
                    **{name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()})
                rows.append(row); arrays.append(error_arrays)
                state['imported_prediction_scene_arms'] += len(row['imported_prediction_methods'])
                state['newly_computed_prediction_scene_arms'] += len(row['newly_computed_prediction_methods'])
                check(completed_scenes=len(rows), last_scene_id=sid)
        components = sorted({r['component_id'] for r in rows})
        if (len(components) != 38 or state['completed_native_scenes'] != 268
                or state['imported_prediction_scene_arms'] != 1960
                or state['newly_computed_prediction_scene_arms'] != 1340):
            raise ValueError('fixed300/38components/1960imports/1340new arms differs')
        def subset(indices):
            check(phase='aggregating')
            return aggregate([rows[i] for i in indices], [arrays[i] for i in indices]) if indices else None
        def selected(predicate):
            return subset([i for i, row in enumerate(rows) if predicate(row)])
        largest = max(components, key=lambda c: sum(r['component_id'] == c for r in rows))
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **subset(list(range(len(rows)))),
            original32_ids=meta['pilot_ids'], training_scene_ids=meta['training_ids'],
            components={c: selected(lambda r: r['component_id'] == c) for c in components},
            original32=selected(lambda r: r['stratum'] == 'original32'),
            remaining268=selected(lambda r: r['stratum'] == 'remaining268'),
            remaining268_seen_components=selected(lambda r: r['stratum'] == 'remaining268' and r['component_seen_in_original32']),
            remaining268_unseen_components=selected(lambda r: r['stratum'] == 'remaining268' and not r['component_seen_in_original32']),
            T32_head_training_seen_components=selected(lambda r: r['component_seen_in_T32_head_training']),
            T32_head_training_unseen_components=selected(lambda r: not r['component_seen_in_T32_head_training']),
            remaining268_T32_head_training_seen_components=selected(lambda r: r['stratum'] == 'remaining268' and r['component_seen_in_T32_head_training']),
            remaining268_T32_head_training_unseen_components=selected(lambda r: r['stratum'] == 'remaining268' and not r['component_seen_in_T32_head_training']),
            without_largest_component=selected(lambda r: r['component_id'] != largest),
            largest_component_id=largest, declared_primary_head='QUADRATIC', restored_head_count=4,
            analytic_head_fits=0, teacher_forward_calls=0, models_constructed=0, optimizer_updates=0,
            all125_fullT_head_arrays_imported_exact=True, original125_public_and_artifacts_imported_exact=True,
            fixed_R_population_invariants_verified=True, no_expansion_beyond_V300=True, position_only=True,
            limitations=['V300 repeatedly developed and shares identities; remaining268 is broader development, not independent test',
                'teacher056 trained allT and threshold historicallyV developed; unseen describes only declared component membership',
                'original125 four heads/15 features/normalizers/soft75/decision threshold retained; no V-based selection',
                'raw loader sees identity metadata, but physical-only prediction API accepts no identity/reference',
                'residuals/counts are proxies, not covariance/effective independent sample size or physical source quality',
                'offline position-only selection of frozen candidates; no online, end-to-end or credible peer claim'])
        original_summary = old.frozen.read(paths['pilot_run']/'summary.json')
        for key in ('methods', 'pairwise'):
            pilot.mix.p._close(result['original32'][key], original_summary[key], '125 original32 aggregate '+key)
        old.frozen.write(output/'scene-summary.json.gz', rows)
        check(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'],
            user_cpu_seconds=state['user_cpu_seconds'], system_cpu_seconds=state['system_cpu_seconds'],
            canonical_event_count=state['canonical_event_count'], completed_native_scenes=state['completed_native_scenes'],
            analytic_head_application_count=state['analytic_head_application_count'],
            analytic_query_prediction_count=state['analytic_query_prediction_count'], neural_model_forwards=0)
        old.frozen.write(output/'summary.json', result)
        check(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=old.frozen.peak_rss_bytes(), finished_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'run.json', state)
        if result is not None and (output/'summary.json').exists():
            result.update(status=state['status'], error=state['error']); old.frozen.write(output/'summary.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'source-system', 'partition-run', 'source-control-run', 'pilot-run', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--archive', type=Path, nargs='+', required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal{signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc(); return 1
    print(f'COMPLETED: {args.output}/summary.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
