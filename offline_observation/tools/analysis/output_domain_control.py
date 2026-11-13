"""Fixed original V32: reuse saved NEAR states, vary only declared position domain.

All groups/targets remain; no prediction, grouping, model or threshold is refit.
Original states are in run116, and their exact replay is bound to run117 reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from tools.analysis import query_contribution_full_control as prior

q, old = prior.q, prior.old
IDEA = 'IDEA-OA-OUTPUT-DOMAIN-20261029'
DOMAINS = ('FULL', 'UNION', 'SPAN')
METHODS = tuple(r+'_'+d for r in q.PARTITIONS for d in DOMAINS)
COMPARISONS = tuple((r+'_FULL', r+'_'+d) for r in q.PARTITIONS for d in DOMAINS[1:])


def position_domains(support, groups):
    """Input support only. No identity, reference, score, state or target mask."""
    support = np.asarray(support)
    if support.ndim != 2 or support.dtype != bool or support.shape[1] == 0:
        raise ValueError('support must be boolean [N,Q], Q>0')
    # emit_group_outputs checks the complete ordered node partition independently.
    emit_group_outputs(np.zeros((len(groups), support.shape[1], 4)), groups,
                       node_count=len(support))
    union = np.asarray([support[g].any(0) for g in groups], bool).reshape(len(groups), support.shape[1])
    interval = np.zeros_like(union)
    for index, row in enumerate(union):
        indices = np.flatnonzero(row)
        if len(indices):
            interval[index, indices[0]:indices[-1]+1] = True
    return dict(FULL=np.ones_like(union), UNION=union, SPAN=interval)


def construct_outputs(support, groups, states, validity):
    """Freeze all six arms before evaluation. Only position validity may change."""
    outputs, counts = {}, {}
    for r in q.PARTITIONS:
        masks = position_domains(support, groups[r])
        values, valid = np.asarray(states[r]), np.asarray(validity[r])
        if (values.shape != (len(groups[r]), support.shape[1], 4)
                or valid.shape != values.shape or valid.dtype != bool):
            raise ValueError('saved complete group states/validity required')
        for domain in DOMAINS:
            records = emit_group_outputs(values, groups[r], node_count=len(support))
            mask = valid.copy()
            mask[..., :2] &= masks[domain][..., None]
            for index, record in enumerate(records):
                record['prediction_valid'] = mask[index].copy()
            name = r+'_'+domain
            outputs[name] = records
            counts[name] = dict(group_query_count=int(masks[domain].size),
                                declared_domain_group_queries=int(masks[domain].sum()),
                                declared_position_group_queries=int(mask[..., :2].all(-1).sum()),
                                no_domain_group_count=int((~masks[domain].any(-1)).sum()))
    return outputs, counts


def artifacts_from_pilot(saved):
    keys = {key[len('A0_MEAN_'):] for key in saved if key.startswith('A0_MEAN_')}
    if not keys:
        raise ValueError('116 artifact schema required')
    return {r: {key: saved[r+'_NEAREST_SUPPORTED_'+key] for key in keys} for r in q.PARTITIONS}


def analyze_scene(fragment, memberships, retained, public117):
    support, nearest = fragment['support'], fragment['nearest_seconds']
    if support.ndim != 2 or support.shape[1] != 64 or nearest.shape != support.shape:
        raise ValueError('original complete Q64 input metadata required')
    query = fragment['query_seconds']
    if query.shape != (64,) or not np.isfinite(query).all() or not (np.diff(query) > 0).all():
        raise ValueError('finite ordered Q64 required')
    groups = {r: old.groups_from_membership(memberships[r], len(support)) for r in q.PARTITIONS}
    for r in q.PARTITIONS:
        old._same(retained[r]['node_group_indices'], memberships[r], '116 frozen membership')
    predictions, counts = construct_outputs(support, groups,
        {r: retained[r]['group_states'] for r in q.PARTITIONS},
        {r: retained[r]['group_valid'] for r in q.PARTITIONS})

    # Reference/identity may only bind and evaluate the already frozen outputs.
    targets, identities = fragment['target_identity_ids'], fragment['identities']
    original = retained['A0']
    old._same(original['target_ids'], targets, '116 complete target registry')
    truth, valid = original['aligned_truth'][:len(targets)], original['aligned_truth_valid'][:len(targets)]
    registry = {identity: index for index, identity in enumerate(targets)}
    if len(registry) != len(targets):
        raise ValueError('unique complete target registry required')
    for node, identity in enumerate(identities):
        if identity not in registry:
            raise ValueError('node absent from target registry')
        old._same(fragment['target'][node], truth[registry[identity]], '116 target reference')
        old._same(fragment['target_valid'][node], valid[registry[identity]], '116 target validity')
    regions = old.fixed_regions(fragment, targets)
    methods, errors, artifacts, non_position = {}, {}, {}, {}
    for name in METHODS:
        r, domain = name.split('_', 1)
        report = old.evaluate_full_scene(predictions[name], node_identities=identities,
            target_identities=targets, target_states=truth, target_valid=valid,
            node_observed=support, split='V_select', grouping_kind='predicted',
            correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=support, nearest_seconds=nearest))
        artifact = report.pop('artifact')
        methods[name], errors[name] = q.describe(report, artifact, fragment, regions)
        non_position[name] = {region: {key: report['full_target_metrics'][region][key]
                                      for key in ('speed', 'heading')}
                              for region in ('all', 'observed', 'missing')}
        if domain == 'FULL':
            if set(artifact) != set(retained[r]):
                raise ValueError('116 baseline artifact schema differs')
            for key in artifact:
                old._same(artifact[key], retained[r][key], '116 baseline artifact/'+r+'/'+key)
            prior.compare_saved_report(methods[name], public117['methods'][r+'_NEAREST_SUPPORTED'],
                                       '117 baseline report/'+r)
        else:
            base = artifacts[r+'_FULL']
            for key in artifact:
                if key not in ('group_valid', 'aligned_prediction_valid'):
                    old._same(artifact[key], base[key], 'frozen domain non-mask/'+name+'/'+key)
                else:
                    old._same(artifact[key][..., 2:], base[key][..., 2:], 'speed/heading mask')
                    if (artifact[key][..., :2] & ~base[key][..., :2]).any():
                        raise ValueError('domain must not add position validity')
            if methods[name]['population'] != methods[r+'_FULL']['population']:
                raise ValueError('fixed node/target/group population changed')
            prior.compare_saved_report(non_position[name], non_position[r+'_FULL'], 'non-position invariance')
        artifacts[name] = artifact
    pairs, pair_arrays = {}, {}
    for base, candidate in COMPARISONS:
        key = candidate+'_vs_'+base
        pairs[key], pair_arrays[key], masks = old.compare_pair(artifacts[base], artifacts[candidate], regions)
        pairs[key].update(baseline_method=base, candidate_method=candidate,
                          side_labels=dict(A0=base, candidate=candidate))
        if pairs[key]['coverage_transitions']['gained_target_points']:
            raise ValueError('nested domain gained an unavailable position')
        if np.any(pair_arrays[key]['all']['delta'] != 0):
            raise ValueError('same-state common-point error changed')
    row = dict(scene_id=public117['scene_id'], component_id=public117['component_id'],
        node_count=len(support), methods=methods, pairwise=pairs, domain_counts=counts,
        non_position_metrics=non_position, baseline117_exact_replay_verified=True,
        frozen_non_mask_arrays_verified=True, speed_heading_invariance_verified=True)
    return row, dict(methods=errors, pairs=pair_arrays), artifacts


def aggregate(rows, arrays):
    result = q.aggregate(rows, arrays, method_names=METHODS)
    result['domain_counts'] = {name: {key: sum(row['domain_counts'][name][key] for row in rows)
        for key in rows[0]['domain_counts'][name]} for name in METHODS}
    for key, pair in result['pairwise'].items():
        pair['rejected_baseline_position'] = old.stats(np.concatenate([a['pairs'][key]['_lost'] for a in arrays]))
        pair['newly_available_position'] = old.stats(np.concatenate([a['pairs'][key]['_gained'] for a in arrays]))
        pair['common_point_error_change_exact_zero'] = all(
            not np.any(a['pairs'][key]['all']['delta'] != 0) for a in arrays)
    return result


def metadata(source, partition, pilot, full, manifest):
    meta = prior.metadata(source, partition, pilot, manifest)
    run, summary = (old.frozen.read(full/name) for name in ('run.json', 'summary.json'))
    if (run.get('status') != 'COMPLETED' or summary.get('status') != 'COMPLETED'
            or run.get('idea_id') != prior.IDEA or summary.get('idea_id') != prior.IDEA
            or run.get('plan_version') != '1.0' or summary.get('plan_version') != '1.0'
            or Path(run.get('source_system', '')).resolve() != source
            or Path(run.get('partition_run', '')).resolve() != partition
            or Path(run.get('pilot_run', '')).resolve() != pilot
            or run.get('scene_ids') != meta['all_ids'] or summary.get('scene_ids') != meta['all_ids']
            or run.get('original32_ids') != meta['pilot_ids']
            or run.get('completed_scenes') != len(meta['all_ids'])
            or summary.get('original32', {}).get('scene_ids') != meta['pilot_ids']
            or not summary.get('all_original056_A0_exact_verified')
            or not summary.get('fixed_R_population_invariants_verified')):
        raise ValueError('completed original117 full/original32/source binding required')
    meta['ids'] = meta['pilot_ids']
    return meta


def run(args):
    source, partition, pilot, full, output = (Path(getattr(args, key)).resolve()
        for key in ('source_system', 'partition_run', 'pilot_run', 'full_run', 'output'))
    for protected in (source.parent, partition, pilot, full):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new output must not overlap original inputs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be 1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), source_system=str(source), partition_run=str(partition),
        pilot_run=str(pilot), full_run=str(full), methods=list(METHODS), completed_scenes=0,
        cpu_threads=args.cpu_threads, models_constructed=0, model_forwards=0, optimizer_updates=0,
        gpu_used=False, scores_recomputed=False, groups_recomputed=False, recovery_recomputed=False,
        threshold_reselected=False, normalizer_refitted=False, archive_read=False, csv_read=False,
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
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard)); state['address_space_limit_bytes'] = limit
            if hasattr(os, 'sched_getaffinity'):
                cpus = sorted(os.sched_getaffinity(0))[:args.cpu_threads]
                os.sched_setaffinity(0, cpus); state['cpu_affinity'] = cpus
        import torch
        torch.set_num_threads(args.cpu_threads)
        meta = metadata(source, partition, pilot, full, old.frozen.read(args.manifest))
        checkpoint(scene_ids=meta['ids'], total_scenes=len(meta['ids']))
        (output/'scenes').mkdir(); (output/'private').mkdir()
        rows, arrays = [], []
        with np.load(meta['association']/'final-scores.npz', allow_pickle=False) as saved:
            old._same(saved['scene_ids'], meta['all_ids'], '056 score scene order')
            for sid in meta['ids']:
                record = old.frozen.load_record(saved, sid, str(meta['components'][sid]))
                fragment, _ = old._load_arrays(source, sid)
                state['cached_reference_read'] = True
                original114 = old.frozen.read(partition/'private'/f'scene-{sid}.json.gz')
                memberships = {r: np.asarray(original114['memberships'][r]) for r in q.PARTITIONS}
                public114 = old.frozen.read(partition/'scenes'/f'scene-{sid}.json.gz')
                prior.bind_partition_source(record, fragment, public114, memberships)
                with np.load(pilot/'private'/f'states-{sid}.npz', allow_pickle=False) as imported:
                    retained = artifacts_from_pilot(imported)
                public117 = old.frozen.read(full/'scenes'/f'scene-{sid}.json.gz')
                if (public117['scene_id'] != sid or public117['component_id'] != str(meta['components'][sid])
                        or public117['node_count'] != len(record['node_ids'])
                        or not public117.get('original056_A0_exact_verified')
                        or not public117.get('fixed_R_population_invariants_verified')
                        or public117.get('stratum') != 'original32'):
                    raise ValueError('117 original32 scene binding differs')
                row, errors, artifacts = analyze_scene(fragment, memberships, retained, public117)
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                np.savez_compressed(output/'private'/f'states-{sid}.npz', **{
                    name+'_'+key: value for name, artifact in artifacts.items() for key,value in artifact.items()})
                rows.append(row); arrays.append(errors)
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        def subset(indices):
            return aggregate([rows[i] for i in indices], [arrays[i] for i in indices])
        components = sorted({row['component_id'] for row in rows})
        largest = max(components, key=lambda cid: sum(row['component_id'] == cid for row in rows))
        others = [i for i,row in enumerate(rows) if row['component_id'] != largest]
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **aggregate(rows, arrays),
            components={cid: subset([i for i,row in enumerate(rows) if row['component_id'] == cid]) for cid in components},
            without_largest_component=subset(others) if others else None,
            all_original117_baselines_exact_replayed=True, fixed_node_target_groups_verified=True,
            position_only_masks_changed=True, speed_heading_invariance_verified=True, no_expansion=True,
            limitations=['original V32 repeatedly used for development with shared identities',
                'domain masks describe declared output availability, not physical existence or death',
                'dropping hard points can reduce conditional MAE; coverage and rejected baseline risk must be read together',
                'groups and duplicate group counts are fixed; no de-duplication is claimed',
                'saved Q64 interpolation/OLS sees future observations; this is offline, not online performance'])
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
    for key in ('source-system', 'partition-run', 'pilot-run', 'full-run', 'manifest', 'output'):
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
    print(f'COMPLETED: {args.output}/summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
