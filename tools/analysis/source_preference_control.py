"""121: source-priority attribution on immutable original118 V32 predictions.

Import all eight saved arms; construct only two new position controls using
visible support and frozen original118 source order/fold centers. No refit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import math
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
from tools.analysis import query_contribution_control as qc
from tools.analysis.shared_component_control import spherical_shift

old = qc.old
IDEA = 'IDEA-SOURCE-PREFERENCE-CONTROL-20261029'
ORIGINAL_IDEA = 'IDEA-SHARED-COMPONENT-CONTROL-20261029'
SHARED_RUN = 'LEVEL1-SHARED-COMPONENT-20261029-118-attempt-1'
LEGACY_METHODS = ('WS_MEAN', 'WS_HUBER', 'WS_FIXED_MEAN', 'WS_FIXED_HUBER',
                  'WS_SYMMETRIC_MEAN', 'WS_SYMMETRIC_HUBER', 'WS_ANCHOR', 'WS_NEAREST_SUPPORTED')
NEW_METHODS = ('WS_LOW_SOURCE_SUPPORTED', 'WS_FIXED_LOW_SOURCE_SUPPORTED')
METHODS = LEGACY_METHODS + NEW_METHODS
COMPARISONS = tuple(combinations(METHODS, 2))
CALIBRATION_KEYS = ('fold_center_ne_m', 'fold_chart_latitude', 'group_mean_delta_degrees',
                    'group_fit_eligible', 'group_fold', 'source_order')


def load_saved(payload):
    """Extract only the original artifact schema; never treat weights as fields."""
    prefix = 'WS_MEAN_'
    keys = {key[len(prefix):] for key in payload if key.startswith(prefix)}
    if 'group_states' not in keys or 'aligned_truth' not in keys:
        raise ValueError('original118 complete saved artifact required')
    retained = {name: {key: np.asarray(payload[name+'_'+key]).copy() for key in keys}
                for name in LEGACY_METHODS}
    calibration = {key: np.asarray(payload[key]).copy() for key in CALIBRATION_KEYS}
    return retained, calibration


def construct_predictions(states, nearest, support, sources, groups, calibration, mean_states):
    """Prediction-only inputs: neither identity nor reference nor target support."""
    states = np.asarray(states, np.float64)
    nearest, support, sources = np.asarray(nearest), np.asarray(support), np.asarray(sources)
    if (states.ndim != 3 or states.shape[1:] != (64, 4) or not np.isfinite(states).all()
            or support.shape != states.shape[:2] or support.dtype != bool
            or nearest.shape != support.shape or not np.isfinite(nearest).all() or (nearest < 0).any()
            or sources.shape != (len(states),)):
        raise ValueError('complete Q64 physical inputs and one source per node required')
    groups = _partition(groups, len(states))
    if any(not isinstance(source, str) or not source for source in sources):
        raise ValueError('nonempty visible source strings required')
    order = np.asarray(calibration['source_order']).tolist()
    if not 1 <= len(order) <= 2 or order != sorted(set(sources.tolist())):
        raise ValueError('original118 source order differs from physical input')
    folds = np.asarray(calibration['group_fold'])
    centers = np.asarray(calibration['fold_center_ne_m'], np.float64)
    if (folds.dtype.kind not in 'iu' or folds.shape != (len(groups),)
            or not np.array_equal(folds, np.arange(len(groups)) % 3)
            or centers.shape != (3, 2) or not np.isfinite(centers).all()):
        raise ValueError('original118 group folds or finite frozen centers differ')
    mean_states = np.asarray(mean_states, np.float64)
    if mean_states.shape != (len(groups), 64, 4) or not np.isfinite(mean_states).all():
        raise ValueError('original118 Mean group states required')
    weights = np.zeros(states.shape[:2], np.float64)
    categories = {name: np.zeros((len(groups), 64), bool)
                  for name in ('both_sources', 'low_only', 'high_only', 'no_support')}
    corrected = states.copy()
    for gid, group in enumerate(groups):
        members = np.asarray(group, np.int64)
        low = members[sources[members] == order[0]]
        high = members[sources[members] == order[1]] if len(order) == 2 else np.empty(0, np.int64)
        low_count = support[low].sum(0)
        high_count = support[high].sum(0)
        has_low, has_high = low_count > 0, high_count > 0
        weights[low] = support[low] / np.maximum(low_count, 1)
        weights[high] = support[high] * (~has_low) / np.maximum(high_count, 1)
        categories['both_sources'][gid] = has_low & has_high
        categories['low_only'][gid] = has_low & ~has_high
        categories['high_only'][gid] = ~has_low & has_high
        categories['no_support'][gid] = ~has_low & ~has_high
        if len(high):
            corrected[high] = spherical_shift(states[high], -centers[folds[gid]])
    predictions = {}
    for name, inputs in zip(NEW_METHODS, (states, corrected)):
        prediction = recover_group_states(inputs, weights, groups)['group_states']
        prediction[..., 2:] = mean_states[..., 2:]
        predictions[name] = prediction
    # Saved group masks describe only visible input. Evaluation may map them to
    # target rows after all predictions have been frozen.
    audit = dict(supported_query_types={name: int(mask.sum()) for name, mask in categories.items()},
                 total_group_queries=len(groups)*64, only_new_arms_constructed=list(NEW_METHODS),
                 frozen_original118_centers_reused=True, centers_refitted=False,
                 fixed_position_changed_group_queries=int(np.any(
                     predictions[NEW_METHODS[0]][..., :2] != predictions[NEW_METHODS[1]][..., :2], axis=-1).sum()))
    return predictions, audit, dict(query_weights=weights, **categories)


def _close(actual, expected, label):
    if isinstance(expected, dict):
        for key, value in expected.items():
            _close(actual[key], value, label+'/'+key)
    elif expected is None or actual is None:
        if actual is not expected:
            raise ValueError(label+' differs')
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(label+' differs')
    elif actual != expected:
        raise ValueError(label+' differs')


def analyze_scene(fragment, membership, retained, calibration, public):
    groups = old.groups_from_membership(membership, len(fragment['native_states']))
    for name in LEGACY_METHODS:
        old._same(retained[name]['node_group_indices'], membership, '118 '+name+' membership')
    predictions = {name: retained[name]['group_states'].copy() for name in LEGACY_METHODS}
    added, audit, private = construct_predictions(fragment['native_states'], fragment['nearest_seconds'],
        fragment['support'], fragment['sources'], groups, calibration, predictions['WS_MEAN'])
    predictions.update(added)
    # All predictions are now immutable; truth and identity bind evaluation only.
    saved = retained['WS_MEAN']
    targets, identities = fragment['target_identity_ids'], fragment['identities']
    old._same(saved['target_ids'], targets, '118 full target registry')
    truth, valid = saved['aligned_truth'][:len(targets)], saved['aligned_truth_valid'][:len(targets)]
    registry = {identity: row for row, identity in enumerate(targets)}
    if len(registry) != len(targets):
        raise ValueError('duplicate target registry')
    for node, identity in enumerate(identities):
        old._same(fragment['target'][node], truth[registry[identity]], '056/118 full target')
        old._same(fragment['target_valid'][node], valid[registry[identity]], '056/118 target valid')
    regions = old.fixed_regions(fragment, targets)
    aligned_groups = saved['aligned_group_indices'][:len(targets)]
    matched = aligned_groups >= 0
    for category in ('both_sources', 'low_only', 'high_only', 'no_support'):
        region = np.zeros((len(targets), 64), bool)
        region[matched] = private[category][aligned_groups[matched]]
        regions['predicted_group_'+category] = region
    artifacts, methods, arrays = {}, {}, {}
    for name in METHODS:
        outputs = emit_group_outputs(predictions[name], groups, node_count=len(fragment['native_states']))
        report = old.evaluate_full_scene(outputs, node_identities=identities, target_identities=targets,
            target_states=truth, target_valid=valid, node_observed=fragment['support'], split='V_select',
            grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
        artifact = report.pop('artifact')
        if name in LEGACY_METHODS:
            if set(artifact) != set(retained[name]):
                raise ValueError('original118 full artifact schema differs')
            for key in artifact:
                old._same(artifact[key], retained[name][key], '118 exact '+name+'/'+key)
        else:
            for key in ('aligned_prediction_valid', 'aligned_group_indices', 'node_group_indices',
                        'group_anchor_target_indices', 'group_valid', 'group_node_indices', 'group_offsets'):
                old._same(artifact[key], artifacts['WS_MEAN'][key], 'fixed WS '+name+'/'+key)
        methods[name], arrays[name] = qc.describe(report, artifact, fragment, regions)
        if name in LEGACY_METHODS:
            _close(methods[name]['position'], public['methods'][name]['position'], '118 position '+name)
            _close(methods[name]['paper_position'], public['methods'][name]['paper_position'], '118 paper '+name)
        if (methods[name]['population'] != methods['WS_MEAN']['population']
                or methods[name]['unmatched_target_count'] != methods['WS_MEAN']['unmatched_target_count']
                or any(methods[name]['position']['all'][k] != methods['WS_MEAN']['position']['all'][k]
                       for k in old.COUNT_KEYS)):
            raise ValueError('fixed WS population/coverage changed')
        artifacts[name] = artifact
    pairs, pair_arrays = {}, {}
    for base, candidate in COMPARISONS:
        key = candidate+'_vs_'+base
        pairs[key], pair_arrays[key], _ = old.compare_pair(artifacts[base], artifacts[candidate], regions)
        pairs[key].update(baseline_method=base, candidate_method=candidate,
                         side_labels=dict(A0=base, candidate=candidate))
    row = dict(scene_id=public['scene_id'], component_id=public['component_id'],
               node_count=len(fragment['native_states']), original118_artifacts_exact=True,
               fixed_R_population_invariants_verified=True, methods=methods, pairwise=pairs,
               source_preference_audit=audit)
    return row, dict(methods=arrays, pairs=pair_arrays), artifacts, private


def metadata(source, previous, shared, manifest):
    meta = old.metadata(source, previous, manifest)
    run, summary = (old.frozen.read(shared/name) for name in ('run.json', 'summary.json'))
    if (shared.name != SHARED_RUN or run.get('status') != 'COMPLETED' or summary.get('status') != 'COMPLETED'
            or run.get('idea_id') != ORIGINAL_IDEA or summary.get('idea_id') != ORIGINAL_IDEA
            or run.get('plan_version') != '1.0' or summary.get('plan_version') != '1.0'
            or Path(run.get('source_system', '')).resolve() != source
            or Path(run.get('partition_run', '')).resolve() != previous
            or run.get('scene_ids') != meta['ids'] or summary.get('scene_ids') != meta['ids']
            or run.get('completed_scenes') != len(meta['ids']) or run.get('methods') != list(LEGACY_METHODS)
            or not summary.get('fixed_R_population_invariants_verified') or not summary.get('no_expansion')):
        raise ValueError('completed original118 fixed V32/source/methods required')
    return meta


def run(args):
    source, previous, shared, output = (Path(getattr(args, key)).resolve()
        for key in ('source_system', 'partition_run', 'shared_run', 'output'))
    for protected in (source.parent, previous, shared):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new output must not overlap immutable inputs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be 1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), source_system=str(source), partition_run=str(previous),
        shared_run=str(shared), methods=list(METHODS), cpu_threads=args.cpu_threads, completed_scenes=0,
        new_prediction_arms=list(NEW_METHODS), imported_prediction_arms=list(LEGACY_METHODS),
        model_forwards=0, models_constructed=0, optimizer_updates=0, gpu_used=False,
        scores_recomputed=False, groups_recomputed=False, threshold_reselected=False, normalizer_refitted=False,
        centers_refitted=False, huber_recomputed=False, archive_read=False, csv_read=False,
        cached_reference_read=False, test_opened=False)
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
        meta = metadata(source, previous, shared, old.frozen.read(args.manifest))
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
                with np.load(shared/'private'/f'states-{sid}.npz', allow_pickle=False) as payload:
                    retained, calibration = load_saved(payload)
                public = old.frozen.read(shared/'scenes'/f'scene-{sid}.json.gz')
                if (public['scene_id'] != sid or public['component_id'] != str(meta['components'][sid])
                        or public['node_count'] != len(record['node_ids'])
                        or not public.get('fixed_R_population_invariants_verified')):
                    raise ValueError('118 scene/source binding differs')
                row, errors, artifacts, private = analyze_scene(fragment, membership, retained, calibration, public)
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                payload = {name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()}
                payload.update(calibration)
                payload.update({'source_preference_'+key: value for key, value in private.items()})
                np.savez_compressed(output/'private'/f'states-{sid}.npz', **payload)
                rows.append(row)
                arrays.append(errors)
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0',
            **qc.aggregate(rows, arrays, method_names=METHODS),
            components={cid: qc.aggregate([r for r in rows if r['component_id'] == cid],
                [a for r, a in zip(rows, arrays) if r['component_id'] == cid], method_names=METHODS)
                for cid in sorted({r['component_id'] for r in rows})},
            original118_artifacts_exact=True, fixed_R_population_invariants_verified=True,
            no_expansion=True, position_only=True, cached_reference_read=True,
            imported_scene_method_count=len(rows)*len(LEGACY_METHODS),
            new_scene_method_count=len(rows)*len(NEW_METHODS),
            centers_refitted=False, huber_recomputed=False,
            limitations=['repeated original V32 development population, not independent testing',
                'source priority is a declared empirical gauge, not known physical source reliability',
                'relative center and input support do not identify absolute physical bias',
                'fixed WS/first representative/full population; no coverage or duplicate contribution',
                'position-only controls with offline Q64 inputs, not learned online fusion or credible peer victory'])
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
    for key in ('source-system', 'partition-run', 'shared-run', 'manifest', 'output'):
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
