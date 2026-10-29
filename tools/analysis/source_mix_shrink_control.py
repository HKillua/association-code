"""124: separately test fixed source mixing and empirical center shrinkage.

Import immutable121 V32 predictions. Construct five new position controls with
fixed .75/.5 source mass and m/(m+4) center scale. Neither variance estimation
nor empirical Bayes identification is claimed. Run only after123 is recorded.
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

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.grouping import _partition
from association.evaluation.query_contribution import recover_group_states
from tools.analysis import source_preference_control as p

old, qc = p.old, p.qc
IDEA = 'IDEA-SOURCE-MIX-SHRINK-CONTROL-20261029'
CONTROL_RUN = 'LEVEL1-SOURCE-PREFERENCE-20261029-121-attempt-1'
FULL_IDEA = 'IDEA-SOURCE-PREFERENCE-FULL-CONTROL-20261029'
FULL_RUN = 'LEVEL1-SOURCE-PREFERENCE-FULL-20261029-123-attempt-1'
LEGACY_METHODS = p.METHODS
NEW_METHODS = ('WS_SOFT75_SOURCE_SUPPORTED', 'WS_FIXED_SOFT75_SOURCE_SUPPORTED',
               'WS_SHRINK_LOW_SOURCE_SUPPORTED', 'WS_SHRINK_SOFT75_SOURCE_SUPPORTED',
               'WS_FIXED_EQUAL_SOURCE_SUPPORTED')
METHODS = LEGACY_METHODS + NEW_METHODS
COMPARISONS = tuple(combinations(METHODS, 2))
SOFT_LOW_MASS, SHRINK_M0 = .75, 4.
CATEGORIES = ('both_sources', 'low_only', 'high_only', 'no_support')


def load_saved(payload):
    prefix = 'WS_MEAN_'
    keys = {key[len(prefix):] for key in payload if key.startswith(prefix)}
    if not {'group_states', 'aligned_truth', 'node_group_indices'} <= keys:
        raise ValueError('complete original121 saved artifact required')
    retained = {name: {key: np.asarray(payload[name+'_'+key]).copy() for key in keys}
                for name in LEGACY_METHODS}
    calibration = {key: np.asarray(payload[key]).copy() for key in p.CALIBRATION_KEYS}
    return retained, calibration


def construct_predictions(states, nearest, support, sources, groups, calibration, mean_states):
    """Visible prediction inputs only; no target identity/reference accepted."""
    states = np.asarray(states, np.float64)
    nearest, support, sources = np.asarray(nearest), np.asarray(support), np.asarray(sources)
    if (states.ndim != 3 or states.shape[1:] != (64, 4) or not np.isfinite(states).all()
            or support.shape != states.shape[:2] or support.dtype != bool
            or nearest.shape != support.shape or not np.isfinite(nearest).all() or (nearest < 0).any()
            or sources.shape != (len(states),)
            or any(not isinstance(source, str) or not source for source in sources)):
        raise ValueError('finite Q64 physical input, boolean support and visible source per node required')
    groups = _partition(groups, len(states))
    order = np.asarray(calibration['source_order']).tolist()
    folds = np.asarray(calibration['group_fold'])
    centers = np.asarray(calibration['fold_center_ne_m'], np.float64)
    eligible = np.asarray(calibration['group_fit_eligible'])
    deltas = np.asarray(calibration['group_mean_delta_degrees'], np.float64)
    if (not 1 <= len(order) <= 2 or order != sorted(set(sources.tolist()))
            or folds.dtype.kind not in 'iu' or folds.shape != (len(groups),)
            or not np.array_equal(folds, np.arange(len(groups)) % 3)
            or centers.shape != (3, 2) or not np.isfinite(centers).all()
            or eligible.dtype != bool or eligible.shape != (len(groups),)
            or deltas.shape != (len(groups), 2)
            or not np.array_equal(eligible, np.isfinite(deltas).all(1))):
        raise ValueError('original121 source order, group folds, eligibility and finite centers required')
    mean_states = np.asarray(mean_states, np.float64)
    if mean_states.shape != (len(groups), 64, 4) or not np.isfinite(mean_states).all():
        raise ValueError('original121 Mean group states required')
    # This is a count of eligible predicted groups, never an independent sample
    # count or estimated precision. No residual or center is fitted here.
    counts = np.array([np.sum((folds != fold) & eligible) for fold in range(3)], np.int64)
    if np.any(centers[counts < 2] != 0.):
        raise ValueError('original insufficient-group zero-center fallback differs')
    scale = counts / (counts + SHRINK_M0)
    masses = {'hard': 1., 'soft75': SOFT_LOW_MASS, 'equal': .5}
    weights = {key: np.zeros(states.shape[:2], np.float64) for key in masses}
    categories = {key: np.zeros((len(groups), 64), bool) for key in CATEGORIES}
    corrected, shrunk = states.copy(), states.copy()
    for gid, group in enumerate(groups):
        members = np.asarray(group, np.int64)
        low = members[sources[members] == order[0]]
        high = members[sources[members] == order[1]] if len(order) == 2 else np.empty(0, np.int64)
        low_count, high_count = support[low].sum(0), support[high].sum(0)
        has_low, has_high = low_count > 0, high_count > 0
        categories['both_sources'][gid] = has_low & has_high
        categories['low_only'][gid] = has_low & ~has_high
        categories['high_only'][gid] = ~has_low & has_high
        categories['no_support'][gid] = ~has_low & ~has_high
        for key, low_mass in masses.items():
            lm = np.where(has_high, low_mass, 1.)
            hm = np.where(has_low, 1.-low_mass, 1.)
            weights[key][low] = support[low] * lm / np.maximum(low_count, 1)
            weights[key][high] = support[high] * hm / np.maximum(high_count, 1)
        if len(high):
            center = centers[folds[gid]]
            corrected[high] = p.spherical_shift(states[high], -center)
            shrunk[high] = p.spherical_shift(states[high], -scale[folds[gid]]*center)
    recipes = ((states, 'soft75'), (corrected, 'soft75'), (shrunk, 'hard'),
               (shrunk, 'soft75'), (corrected, 'equal'))
    predictions = {}
    for name, (inputs, key) in zip(NEW_METHODS, recipes):
        value = recover_group_states(inputs, weights[key], groups)['group_states']
        value[..., 2:] = mean_states[..., 2:]
        predictions[name] = value
    outside_both = ~categories['both_sources']
    if not np.array_equal(predictions[NEW_METHODS[3]][outside_both], predictions[NEW_METHODS[2]][outside_both]):
        raise ValueError('soft mixture changed a query outside both-source support')
    audit = dict(supported_query_types={key: int(mask.sum()) for key, mask in categories.items()},
        total_group_queries=len(groups)*64, frozen_original118_centers_reused=True,
        centers_refitted=False, center_shrink_m0=SHRINK_M0, soft_low_source_mass=SOFT_LOW_MASS,
        equal_control_low_source_mass=.5, fold_training_group_counts=counts.tolist(),
        fold_center_scale=scale.tolist(), fold_center_norm_m=np.linalg.norm(centers, axis=1).tolist(),
        fold_shrunk_center_norm_m=np.linalg.norm(centers*scale[:, None], axis=1).tolist(),
        count_is_effective_independent_sample_size=False, variance_identified=False,
        empirical_bayes_prior_fitted=False, only_new_arms_constructed=list(NEW_METHODS))
    private = dict(**categories, fold_training_group_counts=counts, fold_center_scale=scale,
        **{key+'_query_weights': value for key, value in weights.items()})
    return predictions, audit, private


def analyze_scene(fragment, membership, retained, calibration, public):
    groups = old.groups_from_membership(membership, len(fragment['native_states']))
    for name in LEGACY_METHODS:
        old._same(retained[name]['node_group_indices'], membership, '121 '+name+' membership')
    predictions = {name: retained[name]['group_states'].copy() for name in LEGACY_METHODS}
    added, audit, private = construct_predictions(fragment['native_states'], fragment['nearest_seconds'],
        fragment['support'], fragment['sources'], groups, calibration, predictions['WS_MEAN'])
    predictions.update(added)
    outside_both = ~private['both_sources']
    old._same(added['WS_FIXED_SOFT75_SOURCE_SUPPORTED'][outside_both],
              predictions['WS_FIXED_LOW_SOURCE_SUPPORTED'][outside_both], 'fixed soft outside both support')
    old._same(added['WS_SOFT75_SOURCE_SUPPORTED'][outside_both],
              predictions['WS_LOW_SOURCE_SUPPORTED'][outside_both], 'uncorrected soft outside both support')
    # Predictions above are complete and frozen before evaluation labels bind.
    saved = retained['WS_MEAN']
    targets, identities = fragment['target_identity_ids'], fragment['identities']
    old._same(saved['target_ids'], targets, '121 complete target registry')
    truth, valid = saved['aligned_truth'][:len(targets)], saved['aligned_truth_valid'][:len(targets)]
    registry = {identity: row for row, identity in enumerate(targets)}
    if len(registry) != len(targets):
        raise ValueError('duplicate target registry')
    for node, identity in enumerate(identities):
        old._same(fragment['target'][node], truth[registry[identity]], '056/121 full target')
        old._same(fragment['target_valid'][node], valid[registry[identity]], '056/121 target valid')
    regions = old.fixed_regions(fragment, targets)
    aligned_groups = saved['aligned_group_indices'][:len(targets)]
    matched = aligned_groups >= 0
    for category in CATEGORIES:
        region = np.zeros((len(targets), 64), bool)
        region[matched] = private[category][aligned_groups[matched]]
        regions['predicted_group_'+category] = region
    artifacts, methods, arrays = {}, {}, {}
    for name in METHODS:
        output = emit_group_outputs(predictions[name], groups, node_count=len(fragment['native_states']))
        report = old.evaluate_full_scene(output, node_identities=identities, target_identities=targets,
            target_states=truth, target_valid=valid, node_observed=fragment['support'], split='V_select',
            grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
        artifact = report.pop('artifact')
        if name in LEGACY_METHODS:
            if set(artifact) != set(retained[name]):
                raise ValueError('original121 full artifact schema differs')
            for key in artifact:
                old._same(artifact[key], retained[name][key], '121 exact '+name+'/'+key)
        else:
            for key in artifacts['WS_MEAN']:
                if key not in ('aligned_prediction', 'group_states'):
                    old._same(artifact[key], artifacts['WS_MEAN'][key], 'fixed WS '+name+'/'+key)
            old._same(artifact['group_states'][..., 2:], saved['group_states'][..., 2:], name+' speed/heading')
        methods[name], arrays[name] = qc.describe(report, artifact, fragment, regions)
        if name in LEGACY_METHODS:
            p._close(methods[name]['position'], public['methods'][name]['position'], '121 position '+name)
            p._close(methods[name]['paper_position'], public['methods'][name]['paper_position'], '121 paper '+name)
        if (methods[name]['population'] != methods['WS_MEAN']['population']
                or methods[name]['unmatched_target_count'] != methods['WS_MEAN']['unmatched_target_count']
                or any(methods[name]['position']['all'][key] != methods['WS_MEAN']['position']['all'][key]
                       for key in old.COUNT_KEYS)):
            raise ValueError('fixed WS population/coverage changed')
        artifacts[name] = artifact
    pairs, pair_arrays = {}, {}
    for baseline, candidate in COMPARISONS:
        key = candidate+'_vs_'+baseline
        pairs[key], pair_arrays[key], _ = old.compare_pair(artifacts[baseline], artifacts[candidate], regions)
        pairs[key].update(baseline_method=baseline, candidate_method=candidate,
                         side_labels=dict(A0=baseline, candidate=candidate))
    row = dict(scene_id=public['scene_id'], component_id=public['component_id'],
        node_count=len(fragment['native_states']), original121_artifacts_exact=True,
        fixed_R_population_invariants_verified=True, methods=methods, pairwise=pairs,
        source_mix_shrink_audit=audit)
    audit['position_changed_group_queries_vs_fixedLOW'] = {
        name: int(np.any(predictions[name][..., :2] != predictions['WS_FIXED_LOW_SOURCE_SUPPORTED'][..., :2], axis=-1).sum())
        for name in NEW_METHODS}
    return row, dict(methods=arrays, pairs=pair_arrays), artifacts, private


def metadata(source, previous, control, full, manifest):
    meta = old.metadata(source, previous, manifest)
    for path, idea, ids in ((control, p.IDEA, meta['ids']), (full, FULL_IDEA, meta['all_ids'])):
        run, summary = (old.frozen.read(path/name) for name in ('run.json', 'summary.json'))
        if (run.get('status') != 'COMPLETED' or summary.get('status') != 'COMPLETED'
                or run.get('exit_code') != 0
                or run.get('idea_id') != idea or summary.get('idea_id') != idea
                or run.get('plan_version') != '1.0' or summary.get('plan_version') != '1.0'
                or Path(run.get('source_system', '')).resolve() != source
                or Path(run.get('partition_run', '')).resolve() != previous
                or run.get('scene_ids') != ids or summary.get('scene_ids') != ids
                or run.get('completed_scenes') != len(ids) or run.get('methods') != list(LEGACY_METHODS)
                or not summary.get('fixed_R_population_invariants_verified')):
            raise ValueError('completed original121 and full123 source/population/method binding required')
        if path == full and (run.get('original32_ids') != meta['ids']
                             or summary.get('original32_ids') != meta['ids']):
            raise ValueError('123 original32 population differs')
    if control.name != CONTROL_RUN or full.name != FULL_RUN:
        raise ValueError('original121 and completed123 run paths required')
    if Path(old.frozen.read(full/'run.json')['pilot_run']).resolve() != control:
        raise ValueError('123 original121 prediction lineage differs')
    control_summary = old.frozen.read(control/'summary.json')
    if not control_summary.get('original118_artifacts_exact') or not control_summary.get('no_expansion'):
        raise ValueError('original121 fixed32 artifact provenance required')
    return meta


def run(args):
    source, previous, control, full, output = (Path(getattr(args, key)).resolve() for key in
        ('source_system', 'partition_run', 'source_preference_run', 'source_full_run', 'output'))
    for protected in (source.parent, previous, control, full):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new output must not overlap immutable inputs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be 1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), source_system=str(source), partition_run=str(previous),
        source_preference_run=str(control), source_full_run=str(full), methods=list(METHODS),
        cpu_threads=args.cpu_threads, completed_scenes=0, new_prediction_arms=list(NEW_METHODS),
        imported_prediction_arms=list(LEGACY_METHODS), soft_low_source_mass=SOFT_LOW_MASS, center_shrink_m0=SHRINK_M0,
        models_constructed=0, model_forwards=0, optimizer_updates=0, gpu_used=False,
        scores_recomputed=False, groups_recomputed=False, threshold_reselected=False, normalizer_refitted=False,
        centers_refitted=False, huber_recomputed=False, archive_read=False, csv_read=False,
        cached_reference_read=False, test_opened=False, parameters_selected_from123=False)
    def checkpoint(**extra):
        state.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes(),
                     advisory_exceeded=time.monotonic()-started > 600, **extra)
        old.frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > old.frozen.RAM_BUDGET:
            raise MemoryError('4GiB resident-memory budget exceeded')
    result = None
    try:
        checkpoint()
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([old.frozen.MAX_AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
            state['address_space_limit_bytes'] = limit
            if hasattr(os, 'sched_getaffinity'):
                cpus = sorted(os.sched_getaffinity(0))[:args.cpu_threads]
                os.sched_setaffinity(0, cpus)
                state['cpu_affinity'] = cpus
        import torch
        torch.set_num_threads(args.cpu_threads)
        meta = metadata(source, previous, control, full, old.frozen.read(args.manifest))
        checkpoint(scene_ids=meta['ids'], total_scenes=len(meta['ids']))
        (output/'scenes').mkdir()
        (output/'private').mkdir()
        rows, arrays = [], []
        with np.load(meta['association']/'final-scores.npz', allow_pickle=False) as scores:
            old._same(scores['scene_ids'], meta['all_ids'], '056 score scene order')
            for sid in meta['ids']:
                record = old.frozen.load_record(scores, sid, str(meta['components'][sid]))
                fragment, _ = old._load_arrays(source, sid)
                state['cached_reference_read'] = True
                for key in ('node_ids', 'sources'):
                    old._same(fragment[key], record[key], '056 '+key)
                prior = old.frozen.read(previous/'private'/f'scene-{sid}.json.gz')
                membership = np.asarray(prior['memberships']['WS'])
                with np.load(control/'private'/f'states-{sid}.npz', allow_pickle=False) as payload:
                    retained, calibration = load_saved(payload)
                public = old.frozen.read(control/'scenes'/f'scene-{sid}.json.gz')
                if (public['scene_id'] != sid or public['component_id'] != str(meta['components'][sid])
                        or public['node_count'] != len(record['node_ids'])
                        or not public.get('fixed_R_population_invariants_verified')):
                    raise ValueError('121 scene/source binding differs')
                row, errors, artifacts, private = analyze_scene(fragment, membership, retained, calibration, public)
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                payload = {name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()}
                payload.update(calibration)
                payload.update({'source_mix_shrink_'+key: value for key, value in private.items()})
                np.savez_compressed(output/'private'/f'states-{sid}.npz', **payload)
                rows.append(row)
                arrays.append(errors)
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        components = sorted({row['component_id'] for row in rows})
        largest = max(components, key=lambda cid: sum(row['component_id'] == cid for row in rows))
        def subset(indices):
            return qc.aggregate([rows[i] for i in indices], [arrays[i] for i in indices], method_names=METHODS) if indices else None
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0',
            **qc.aggregate(rows, arrays, method_names=METHODS),
            components={cid: subset([i for i, row in enumerate(rows) if row['component_id'] == cid]) for cid in components},
            without_largest_component=subset([i for i, row in enumerate(rows) if row['component_id'] != largest]),
            original121_artifacts_exact=True, fixed_R_population_invariants_verified=True,
            no_expansion=True, position_only=True, cached_reference_read=True,
            imported_scene_method_count=len(rows)*len(LEGACY_METHODS),
            new_scene_method_count=len(rows)*len(NEW_METHODS),
            soft_low_source_mass=SOFT_LOW_MASS, center_shrink_m0=SHRINK_M0,
            centers_refitted=False, huber_recomputed=False, parameters_selected_from123=False,
            limitations=['repeated original V32 development with shared identities; not independent testing',
                'fixed source masses and group-count shrinkage are empirical controls, not identified precision or empirical Bayes',
                'relative center and support do not identify absolute physical bias or root cause',
                'full frozen WS population/coverage/duplicates unchanged; position-only intervention',
                'offline Q64 inputs; no online end-to-end learning or credible peer victory'])
        old.frozen.write(output/'scene-summary.json.gz', rows)
        checkpoint(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        old.frozen.write(output/'summary.json', result)
        checkpoint(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
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
    for key in ('source-system', 'partition-run', 'source-preference-run', 'source-full-run', 'manifest', 'output'):
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
