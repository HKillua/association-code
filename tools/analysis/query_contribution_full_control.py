"""117: reuse the original six V32 predictions; evaluate fixed controls on V300.

Only nearest_all is new on V32. Eight controls are calculated on remaining268.
No model, training, threshold tuning, grouping inference, ZIP or test access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from association.evaluation.group_prediction import emit_group_outputs, predict_group_outputs
from tools.analysis import query_contribution_control as q

old = q.old
IDEA = 'IDEA-QUERY-CONTRIBUTION-FULL-CONTROL-20261029'
PILOT_RUN = 'LEVEL1-QUERY-CONTRIBUTION-20261029-116-attempt-1'
RULES = dict(MEAN='mean_supported', ANCHOR='anchor', NEAREST_ALL='nearest_observation',
             NEAREST_SUPPORTED='nearest_supported')
METHODS = tuple(r+'_'+w for r in q.PARTITIONS for w in RULES)
COMPARISONS = tuple((r+'_MEAN', r+'_'+w) for r in q.PARTITIONS for w in tuple(RULES)[1:]) + tuple(
    ('A0_'+w, 'WS_'+w) for w in RULES) + tuple(
    (r+'_ANCHOR', r+'_'+w) for r in q.PARTITIONS for w in ('NEAREST_ALL', 'NEAREST_SUPPORTED')) + tuple(
    (r+'_NEAREST_ALL', r+'_NEAREST_SUPPORTED') for r in q.PARTITIONS)


def metadata(source, previous, pilot, manifest):
    meta = old.metadata(source, previous, manifest)
    run, summary = (old.frozen.read(pilot/k) for k in ('run.json', 'summary.json'))
    if (pilot.name != PILOT_RUN or run.get('status') != 'COMPLETED' or summary.get('status') != 'COMPLETED'
            or run.get('idea_id') != q.IDEA or summary.get('idea_id') != q.IDEA
            or run.get('plan_version') != '1.0' or summary.get('plan_version') != '1.0'
            or Path(run.get('source_system', '')).resolve() != source
            or Path(run.get('partition_run', '')).resolve() != previous
            or run.get('scene_ids') != meta['ids'] or summary.get('scene_ids') != meta['ids']
            or run.get('completed_scenes') != len(meta['ids']) or run.get('methods') != list(q.METHODS)
            or not summary.get('original115_exact_replays_verified')
            or not summary.get('fixed_R_population_invariants_verified') or not summary.get('no_expansion')):
        raise ValueError('completed original116 fixed pilot/source binding required')
    meta['pilot_ids'] = meta['ids']
    meta['ids'] = meta['all_ids']
    return meta


def artifact_from_pilot(saved):
    keys = {key[len('A0_MEAN_'):] for key in saved if key.startswith('A0_MEAN_')}
    return {name: {key: saved[name+'_'+key] for key in keys}
            for name in METHODS if not name.endswith('_NEAREST_ALL')}


def compare_saved_report(actual, expected, path='116 method report'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(path+' schema differs')
        for key in expected:
            compare_saved_report(actual[key], expected[key], path+'/'+key)
    elif isinstance(expected, float):
        if actual is None or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(path+' differs')
    elif actual != expected:
        raise ValueError(path+' differs')


def gate_input_masks(nearest, support, groups):
    masks = {k: np.zeros((len(groups), support.shape[1]), bool) for k in
             ('has_support_choice_changed', 'no_support_all_not_anchor', 'same_choice')}
    for gid, group in enumerate(groups):
        members = np.asarray(group)
        visible = support[members]
        any_support = visible.any(0)
        all_choice = members[np.argmin(nearest[members], axis=0)]
        supported_choice = members[np.argmin(np.where(visible, nearest[members], np.inf), axis=0)]
        masks['has_support_choice_changed'][gid] = any_support & (all_choice != supported_choice)
        masks['no_support_all_not_anchor'][gid] = ~any_support & (all_choice != members[0])
        masks['same_choice'][gid] = ~(masks['has_support_choice_changed'][gid] | masks['no_support_all_not_anchor'][gid])
    return masks


def analyze_scene(fragment, system, memberships, scene_id, component_id, *, pilot_artifacts=None, pilot_row=None):
    """Freeze new predictions and imported artifacts before any truth evaluation."""
    states, support, nearest, sources = (fragment[k] for k in ('native_states', 'support', 'nearest_seconds', 'sources'))
    query = np.asarray(fragment['query_seconds'])
    if (states.shape[1:] != (64, 4) or support.shape != states.shape[:2] or support.dtype != bool
            or query.shape != (64,) or not np.isfinite(query).all() or not (np.diff(query) > 0).all()
            or fragment['exact_observation'].shape != support.shape or fragment['exact_observation'].dtype != bool):
        raise ValueError('original complete finite ordered Q64 input required')
    groups = {r: old.groups_from_membership(memberships[r], len(states)) for r in q.PARTITIONS}
    predictions, audits = {}, {}
    input_masks = {r: gate_input_masks(nearest, support, groups[r]) for r in q.PARTITIONS}
    for r in q.PARTITIONS:
        for label, method in RULES.items():
            name = r+'_'+label
            if pilot_artifacts is not None and name in pilot_artifacts:
                old._same(pilot_artifacts[name]['node_group_indices'], memberships[r], '116 imported membership')
                continue
            result = predict_group_outputs(states, nearest, support, sources, groups[r], method=method)
            predictions[name] = result['group_outputs']
            audits[name] = result['audit']

    # Identities/reference only bind and evaluate the already frozen predictions.
    targets, identities = fragment['target_identity_ids'], fragment['identities']
    original = {k[len(old.BASE)+1:]: value for k, value in system.items() if k.startswith(old.BASE+'_')}
    old._same(original['target_ids'], targets, '056 complete target registry')
    truth, valid = original['aligned_truth'][:len(targets)], original['aligned_truth_valid'][:len(targets)]
    registry = {identity: row for row, identity in enumerate(targets)}
    if len(registry) != len(targets):
        raise ValueError('complete unique target registry required')
    for node, identity in enumerate(identities):
        if identity not in registry:
            raise ValueError('node missing in target registry')
        old._same(fragment['target'][node], truth[registry[identity]], '056 node target')
        old._same(fragment['target_valid'][node], valid[registry[identity]], '056 node target valid')
    regions = old.fixed_regions(fragment, targets)
    artifacts, methods, error_arrays = {}, {}, {}
    for name in METHODS:
        r = name.split('_', 1)[0]
        if pilot_artifacts is not None and name in pilot_artifacts:
            artifact = pilot_artifacts[name]
            if pilot_row is None or not pilot_row['original115_exact_replays_verified']:
                raise ValueError('original116 verified public row required')
            old._same(artifact['target_ids'], targets, '116 imported target registry')
            old._same(artifact['aligned_truth'][:len(targets)], truth, '116 imported truth')
            old._same(artifact['aligned_truth_valid'][:len(targets)], valid, '116 imported target valid')
            imported_outputs = emit_group_outputs(artifact['group_states'], groups[r], node_count=len(states))
            report = old.evaluate_full_scene(imported_outputs, node_identities=identities,
                target_identities=targets, target_states=truth, target_valid=valid, node_observed=support,
                split='V_select', grouping_kind='predicted', correspondence_policy='anchor_identity_first',
                return_artifact=True, node_input_support=dict(support=support, nearest_seconds=nearest))
            replay = report.pop('artifact')
            if set(replay) != set(artifact):
                raise ValueError('116 imported artifact schema differs')
            for key in replay:
                old._same(replay[key], artifact[key], '116 imported evaluation replay/'+name+'/'+key)
            methods[name], error_arrays[name] = q.describe(report, artifact, fragment, regions)
            compare_saved_report(methods[name], pilot_row['methods'][name])
        else:
            report = old.evaluate_full_scene(predictions[name], node_identities=identities,
                target_identities=targets, target_states=truth, target_valid=valid, node_observed=support,
                split='V_select', grouping_kind='predicted', correspondence_policy='anchor_identity_first',
                return_artifact=True, node_input_support=dict(support=support, nearest_seconds=nearest))
            artifact = report.pop('artifact')
            methods[name], error_arrays[name] = q.describe(report, artifact, fragment, regions)
        if name == 'A0_MEAN':
            if set(artifact) != set(original):
                raise ValueError('056 A0 Mean artifact schema differs')
            for key in artifact:
                old._same(artifact[key], original[key], '056 A0 Mean exact/'+key)
        elif not name.endswith('_MEAN'):
            base = artifacts[r+'_MEAN']
            for key in ('aligned_prediction_valid', 'aligned_group_indices', 'node_group_indices',
                        'group_anchor_target_indices', 'group_valid', 'group_node_indices', 'group_offsets'):
                old._same(artifact[key], base[key], 'fixed R output invariant/'+name+'/'+key)
            baseline = methods[r+'_MEAN']
            if (methods[name]['population'] != baseline['population']
                    or methods[name]['unmatched_target_count'] != baseline['unmatched_target_count']
                    or any(methods[name]['position']['all'][k] != baseline['position']['all'][k] for k in old.COUNT_KEYS)):
                raise ValueError('fixed R population/coverage differs')
        artifacts[name] = artifact
    pairs, pair_arrays = {}, {}
    for baseline, candidate in COMPARISONS:
        name = candidate+'_vs_'+baseline
        pair_regions = dict(regions)
        if baseline.endswith('_NEAREST_ALL') and candidate.endswith('_NEAREST_SUPPORTED'):
            r = baseline.split('_', 1)[0]
            selected_groups = artifacts[baseline]['aligned_group_indices'][:len(targets)]
            matched = selected_groups >= 0
            for condition, mask in input_masks[r].items():
                selected = np.zeros((len(targets), 64), bool)
                selected[matched] = mask[selected_groups[matched]]
                pair_regions['gate_'+condition] = selected
        pairs[name], pair_arrays[name], _ = old.compare_pair(artifacts[baseline], artifacts[candidate], pair_regions)
        pairs[name].update(baseline_method=baseline, candidate_method=candidate,
                           side_labels=dict(A0=baseline, candidate=candidate))
    row = dict(scene_id=scene_id, component_id=component_id, node_count=len(states),
        stratum='original32' if pilot_artifacts is not None else 'remaining268',
        imported_prediction_methods=list(pilot_artifacts) if pilot_artifacts is not None else [],
        newly_computed_prediction_methods=list(predictions), original056_A0_exact_verified=True,
        fixed_R_population_invariants_verified=True, methods=methods, pairwise=pairs, recovery_audits=audits,
        gate_input_group_query_counts={r: {k: int(v.sum()) for k,v in masks.items()} for r,masks in input_masks.items()})
    return row, dict(methods=error_arrays, pairs=pair_arrays), artifacts


def aggregate(rows, arrays):
    result = q.aggregate(rows, arrays, method_names=METHODS)
    result['scene_prediction_arms_imported'] = sum(len(r['imported_prediction_methods']) for r in rows)
    result['scene_prediction_arms_computed'] = sum(len(r['newly_computed_prediction_methods']) for r in rows)
    result['gate_input_group_query_counts'] = {r: {key: sum(row['gate_input_group_query_counts'][r][key] for row in rows)
        for key in rows[0]['gate_input_group_query_counts'][r]} for r in q.PARTITIONS}
    return result


def bind_partition_source(record, fragment, public114, memberships):
    if (public114['scene_id'] != record['scene_id'] or public114['component_id'] != record['component_id']
            or public114['node_count'] != len(record['node_ids'])
            or not public114.get('raw_invariance_verified') or not public114.get('original_A0_membership_verified')):
        raise ValueError('114 original scene/source public binding differs')
    for key in ('node_ids', 'sources'):
        old._same(fragment[key], record[key], '056 source binding/'+key)
    old._same(record['labels'], fragment['identities'][record['pairs'][:, 0]] ==
              fragment['identities'][record['pairs'][:, 1]], '056 labels/fragment identities')
    old._same(memberships['A0'], fragment['learned_assoc_membership'], '056 A0 membership')
    raw = old.frozen.evaluate_scenes([record], old.frozen.THRESHOLD)
    old.frozen.counts_match(raw, old.frozen.metric_rows(public114['raw']), '114 original raw')
    for r in q.PARTITIONS:
        co = memberships[r][record['pairs'][:, 0]] == memberships[r][record['pairs'][:, 1]]
        decoded = old.frozen.evaluate_scenes([dict(record, scores=co.astype(float))], .5)
        old.frozen.counts_match(decoded, old.frozen.metric_rows(public114['arms'][r]['decoded']), '114 original '+r)


def run(args):
    source, previous, pilot, output = (Path(getattr(args, k)).resolve()
                                     for k in ('source_system', 'partition_run', 'pilot_run', 'output'))
    for protected in (source.parent, previous, pilot):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('new output must not overlap original inputs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be 1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), source_system=str(source), partition_run=str(previous),
        pilot_run=str(pilot), methods=list(METHODS), cpu_threads=args.cpu_threads, completed_scenes=0,
        model_forwards=0, models_constructed=0, optimizer_updates=0, gpu_used=False, scores_recomputed=False,
        groups_recomputed=False, threshold_reselected=False, normalizer_refitted=False, archive_read=False,
        csv_read=False, cached_reference_read=False, test_opened=False,
        imported_prediction_scene_arms=0, newly_computed_prediction_scene_arms=0)
    def checkpoint(**extra):
        state.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes(),
                     advisory_exceeded=time.monotonic()-started > 1200, **extra)
        old.frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > old.frozen.RAM_BUDGET:
            raise MemoryError('4GiB resident-memory budget exceeded')
    result = None
    try:
        checkpoint()
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([old.frozen.MAX_AS]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard)); state['address_space_limit_bytes'] = limit
            if hasattr(os, 'sched_getaffinity'):
                cpus = sorted(os.sched_getaffinity(0))[:args.cpu_threads]
                os.sched_setaffinity(0, cpus); state['cpu_affinity'] = cpus
        import torch
        torch.set_num_threads(args.cpu_threads)
        meta = metadata(source, previous, pilot, old.frozen.read(args.manifest))
        checkpoint(scene_ids=meta['ids'], original32_ids=meta['pilot_ids'], total_scenes=len(meta['ids']))
        (output/'scenes').mkdir(); (output/'private').mkdir()
        rows, arrays = [], []
        with np.load(meta['association']/'final-scores.npz', allow_pickle=False) as saved:
            old._same(saved['scene_ids'], meta['all_ids'], '056 score scene order')
            for sid in meta['ids']:
                record = old.frozen.load_record(saved, sid, str(meta['components'][sid]))
                fragment, system = old._load_arrays(source, sid)
                state['cached_reference_read'] = True
                original114 = old.frozen.read(previous/'private'/f'scene-{sid}.json.gz')
                memberships = {r: np.asarray(original114['memberships'][r]) for r in q.PARTITIONS}
                public114 = old.frozen.read(previous/'scenes'/f'scene-{sid}.json.gz')
                bind_partition_source(record, fragment, public114, memberships)
                artifacts, public = None, None
                if sid in meta['pilot_ids']:
                    with np.load(pilot/'private'/f'states-{sid}.npz', allow_pickle=False) as imported:
                        artifacts = artifact_from_pilot(imported)
                    public = old.frozen.read(pilot/'scenes'/f'scene-{sid}.json.gz')
                    if public['scene_id'] != sid or public['component_id'] != str(meta['components'][sid]):
                        raise ValueError('116 original public scene/component binding differs')
                row, errors, retained = analyze_scene(fragment, system, memberships, sid,
                    str(meta['components'][sid]), pilot_artifacts=artifacts, pilot_row=public)
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                # Imported arrays are immutable originals; retain only new predictions here.
                payload = {name+'_'+key: value for name, artifact in retained.items()
                           if name in row['newly_computed_prediction_methods'] for key, value in artifact.items()}
                np.savez_compressed(output/'private'/f'states-{sid}.npz', **payload)
                rows.append(row); arrays.append(errors)
                state['imported_prediction_scene_arms'] += len(row['imported_prediction_methods'])
                state['newly_computed_prediction_scene_arms'] += len(row['newly_computed_prediction_methods'])
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        components = sorted({r['component_id'] for r in rows})
        remaining = [i for i,r in enumerate(rows) if r['stratum'] == 'remaining268']
        original = [i for i,r in enumerate(rows) if r['stratum'] == 'original32']
        def subset(indices):
            return aggregate([rows[i] for i in indices], [arrays[i] for i in indices])
        largest = max(components, key=lambda cid: sum(r['component_id'] == cid for r in rows))
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **aggregate(rows, arrays),
            components={cid: subset([i for i,r in enumerate(rows) if r['component_id'] == cid]) for cid in components},
            original32=subset(original), remaining268=subset(remaining),
            without_largest_component=subset([i for i,r in enumerate(rows) if r['component_id'] != largest]),
            all_original056_A0_exact_verified=True, fixed_R_population_invariants_verified=True,
            no_expansion_beyond_V300=True, position_only=True, cached_reference_read=True,
            limitations=['repeated development V300 with shared identities; not independent test/peer reproduction',
                'Q64 interpolation/OLS and same cached reference; reference generation unknown',
                'fixed R and output correspondence; no learned reliability or output deduplication'])
        old.frozen.write(output/'scene-summary.json.gz', rows)
        checkpoint(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        old.frozen.write(output/'summary.json', result)
        checkpoint(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes())
        old.frozen.write(output/'run.json', state)
        if result is not None and (output/'summary.json').exists():
            result.update(status=state['status'], error=state['error']); old.frozen.write(output/'summary.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-system', 'partition-run', 'pilot-run', 'manifest', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc(); return 1
    print(f'COMPLETED: {args.output}/summary.json'); return 0


if __name__ == '__main__':
    raise SystemExit(main())
