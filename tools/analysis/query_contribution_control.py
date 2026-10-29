"""116: fixed original V32, two saved identity partitions x four query rules.

No training, score inference or reference-dependent prediction. Coordinates,
identity, per-node weights and full evaluation artifacts stay private on server.
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

from association.evaluation.query_contribution import query_contribution
from association.evaluation.paper_fusion_metrics import COUNTS, STATISTICS
from tools.analysis import static_partition_position_control as old

IDEA = 'IDEA-QUERY-CONTRIBUTION-CONTROL-20261029'
CONTROL_RUN = 'LEVEL1-STATIC-PARTITION-POSITION-20261029-115-attempt-1'
PARTITIONS = ('A0', 'WS')
RULES = ('MEAN', 'ANCHOR', 'SOURCE_BALANCED', 'NEAREST_SUPPORTED')
METHODS = tuple(r+'_'+w for r in PARTITIONS for w in RULES)
COMPARISONS = tuple((r+'_MEAN', r+'_'+w) for r in PARTITIONS for w in RULES[1:]) + tuple(
    ('A0_'+w, 'WS_'+w) for w in RULES) + tuple(
    (r+'_ANCHOR', r+'_'+w) for r in PARTITIONS for w in RULES[2:])


def construct_predictions(states, nearest, support, sources, memberships):
    """Explicit prediction inputs only: never accepts a fragment/reference dict."""
    groups, outputs, audits, weights = {}, {}, {}, {}
    for r in PARTITIONS:
        groups[r] = old.groups_from_membership(memberships[r], len(states))
        first = np.asarray([g[0] for g in groups[r]], np.int64)
        ordinary = old.ordinary_fusion(states, nearest, support, groups[r])
        outputs[r+'_MEAN'] = ordinary['predictions']['mean_supported'][first]
        outputs[r+'_ANCHOR'] = np.asarray(states)[first].copy()
        for label, method in (('SOURCE_BALANCED', 'source_balanced_supported'),
                              ('NEAREST_SUPPORTED', 'nearest_supported')):
            result = query_contribution(states, nearest, support, sources, groups[r], method=method)
            outputs[r+'_'+label] = result['group_states']
            weights[r+'_'+label] = result['weights']
            audits[r+'_'+label] = result['audit']
            differences = []
            nearest_changes = 0
            for gid, members in enumerate(groups[r]):
                visible = support[members]
                mass = visible/np.maximum(visible.sum(0), 1)
                differences.append(np.abs(result['weights'][members]-mass).sum(0))
                if label == 'NEAREST_SUPPORTED':
                    chosen = np.asarray(members)[np.argmax(result['weights'][members], axis=0)]
                    original_nearest = ordinary['selected_nodes'][members[0]]
                    nearest_changes += int(((chosen != original_nearest) & visible.any(0)).sum())
            audits[r+'_'+label]['fragment_mean_weight_l1_change'] = np.asarray(differences).tolist()
            audits[r+'_'+label]['weight_changed_group_queries_tolerance_1e_12'] = int((np.asarray(differences) > 1e-12).sum())
            if label == 'NEAREST_SUPPORTED':
                audits[r+'_'+label]['choice_changed_from_all_member_nearest_supported_queries'] = nearest_changes
    return groups, outputs, audits, weights


def describe(report, artifact, fragment, regions):
    _, valid, errors, own = old.target_arrays(artifact)
    result = dict(population={k: report[k] for k in old.POPULATION_KEYS},
                  unmatched_target_count=len(report['unmatched_target_ids']),
                  position={}, fixed_target_strata={},
                  paper_position={k: report['paper_fusion_metrics']['attributes'][k]
                                  for k in ('latitude', 'longitude')})
    for region in ('all', 'observed', 'missing'):
        mask = np.ones_like(own) if region == 'all' else artifact['aligned_observed'][:len(own)]
        if region == 'missing':
            mask = ~mask
        values = errors[own & mask]
        result['position'][region] = dict(report['full_target_metrics'][region]['position'],
            p95=old.stats(values)['p95'], p99=old.stats(values)['p99'])
    arrays = {}
    for region, mask in regions.items():
        values = errors[own & mask]
        available = int((valid[..., :2].all(-1) & mask).sum())
        result['fixed_target_strata'][region] = dict(old.stats(values), available_target_points=available,
            missing_prediction_points=available-len(values))
        arrays[region] = values
    return result, arrays


def analyze_scene(fragment, memberships, retained115, public115):
    """Fix every prediction before extracting reference/identities for evaluation."""
    states, nearest, support = (fragment[k] for k in ('native_states', 'nearest_seconds', 'support'))
    if states.shape[1:] != (64, 4) or support.dtype != bool:
        raise ValueError('original complete Q64 physical input required')
    groups, predictions, audits, weights = construct_predictions(
        states, nearest, support, fragment['sources'], memberships)
    for r in PARTITIONS:
        old._same(retained115[r]['node_group_indices'], memberships[r], '115 '+r+' membership')
        old._same(predictions[r+'_MEAN'], retained115[r]['group_states'], '115 exact '+r+' Mean states')

    # Everything below may inspect identity/reference; it cannot alter predictions.
    targets = fragment['target_identity_ids']
    identities = fragment['identities']
    anchor = retained115['A0']
    old._same(anchor['target_ids'], targets, '115 complete target registry')
    truth, valid = anchor['aligned_truth'][:len(targets)], anchor['aligned_truth_valid'][:len(targets)]
    registry = {identity: row for row, identity in enumerate(targets)}
    if len(registry) != len(targets):
        raise ValueError('duplicate target registry')
    for node, identity in enumerate(identities):
        if identity not in registry:
            raise ValueError('node absent from full target registry')
        old._same(fragment['target'][node], truth[registry[identity]], '056/115 full target')
        old._same(fragment['target_valid'][node], valid[registry[identity]], '056/115 full target valid')
    regions = old.fixed_regions(fragment, targets)
    artifacts, methods, arrays = {}, {}, {}
    for name in METHODS:
        r, _ = name.split('_', 1)
        outputs = [dict(node_indices=g, states=value) for g, value in zip(groups[r], predictions[name])]
        report = old.evaluate_full_scene(outputs, node_identities=identities, target_identities=targets,
            target_states=truth, target_valid=valid, node_observed=support, split='V_select',
            grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True,
            node_input_support=dict(support=support, nearest_seconds=nearest))
        artifact = report.pop('artifact')
        if name.endswith('_MEAN'):
            if set(artifact) != set(retained115[r]):
                raise ValueError('115 Mean full artifact schema differs')
            for key in artifact:
                old._same(artifact[key], retained115[r][key], '115 exact '+name+'/'+key)
        else:
            baseline = artifacts[r+'_MEAN']
            for key in ('aligned_prediction_valid', 'aligned_group_indices', 'node_group_indices',
                        'group_anchor_target_indices', 'group_valid', 'group_node_indices', 'group_offsets'):
                old._same(artifact[key], baseline[key], 'fixed R output invariant '+name+'/'+key)
        methods[name], arrays[name] = describe(report, artifact, fragment, regions)
        if name.endswith('_MEAN'):
            for region, expected in public115['methods'][r]['position'].items():
                for key, value in expected.items():
                    actual = methods[name]['position'][region][key]
                    equal = (actual is value if value is None or actual is None else
                             math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-9)
                             if isinstance(value, float) else actual == value)
                    if not equal:
                        raise ValueError('115 position summary mismatch '+name+'/'+region+'/'+key)
        else:
            baseline = methods[r+'_MEAN']
            if (methods[name]['population'] != baseline['population']
                    or methods[name]['unmatched_target_count'] != baseline['unmatched_target_count']
                    or any(methods[name]['position']['all'][k] != baseline['position']['all'][k]
                           for k in old.COUNT_KEYS)):
                raise ValueError('fixed R population/coverage changed under W')
        artifacts[name] = artifact
    pairs, pair_arrays = {}, {}
    for base, candidate in COMPARISONS:
        key = candidate+'_vs_'+base
        pairs[key], pair_arrays[key], _ = old.compare_pair(artifacts[base], artifacts[candidate], regions)
        pairs[key].update(baseline_method=base, candidate_method=candidate,
                         side_labels=dict(A0=base, candidate=candidate))
    row = dict(scene_id=public115['scene_id'], component_id=public115['component_id'],
        node_count=len(states), original115_exact_replays_verified=True,
        fixed_R_population_invariants_verified=True, methods=methods, pairwise=pairs,
        contribution_audits=audits)
    return row, dict(methods=arrays, pairs=pair_arrays), artifacts, weights


def paper_aggregate(rows):
    out = {}
    for attribute in ('latitude', 'longitude'):
        values = [v['paper_position'][attribute] for v in rows]
        result = {k: sum(v[k] for v in values) for k in COUNTS}
        result.update(unit=values[0]['unit'], scene_equal_target_mean={}, defined_scenes={}, defined_targets={})
        for statistic in STATISTICS:
            selected = [v['target_mean'][statistic] for v in values if v['target_mean'][statistic] is not None]
            result['scene_equal_target_mean'][statistic] = float(np.mean(selected)) if selected else None
            result['defined_scenes'][statistic] = len(selected)
            result['defined_targets'][statistic] = sum(v['defined_targets'][statistic] for v in values)
        out[attribute] = result
    return out


def aggregate(rows, arrays):
    out = dict(scene_count=len(rows), scene_ids=[r['scene_id'] for r in rows], methods={}, pairwise={})
    for name in METHODS:
        values = [r['methods'][name] for r in rows]
        out['methods'][name] = dict(population={k: sum(v['population'][k] for v in values) for k in old.POPULATION_KEYS},
            unmatched_target_count=sum(v['unmatched_target_count'] for v in values),
            position=old.pooled_position([v['position']['all'] for v in values], [a['methods'][name]['all'] for a in arrays]),
            paper_position=paper_aggregate(values),
            fixed_target_strata={region: dict(old.stats(np.concatenate([a['methods'][name][region] for a in arrays])),
                available_target_points=sum(v['fixed_target_strata'][region]['available_target_points'] for v in values),
                missing_prediction_points=sum(v['fixed_target_strata'][region]['missing_prediction_points'] for v in values))
                for region in arrays[0]['methods'][name]})
        for region in out['methods'][name]['fixed_target_strata'].values():
            region['coverage'] = region['point_count']/region['available_target_points'] if region['available_target_points'] else None
    for key in rows[0]['pairwise']:
        values = [r['pairwise'][key] for r in rows]
        out['pairwise'][key] = dict(
            baseline_method=values[0]['baseline_method'], candidate_method=values[0]['candidate_method'],
            side_labels=values[0]['side_labels'],
            coverage_transitions={k: sum(v['coverage_transitions'][k] for v in values) for k in values[0]['coverage_transitions']},
            common_valid_position={side: old.pooled_position([v['common_valid_position'][side] for v in values],
                [a['pairs'][key]['all'][side] for a in arrays]) for side in ('A0', 'candidate')},
            strata={region: dict(A0=old.stats(np.concatenate([a['pairs'][key][region]['A0'] for a in arrays])),
                candidate=old.stats(np.concatenate([a['pairs'][key][region]['candidate'] for a in arrays])),
                candidate_minus_A0=old.delta_stats(np.concatenate([a['pairs'][key][region]['delta'] for a in arrays])))
                for region in arrays[0]['pairs'][key] if not region.startswith('_')},
            scene_mae_changes={r['scene_id']: r['pairwise'][key]['strata']['all']['candidate_minus_A0']['mean_error_change'] for r in rows})
    return out


def metadata(source, previous, control, manifest):
    meta = old.metadata(source, previous, manifest)
    run, summary = (old.frozen.read(control/name) for name in ('run.json', 'summary.json'))
    if (control.name != CONTROL_RUN or run.get('status') != 'COMPLETED' or summary.get('status') != 'COMPLETED'
            or run.get('idea_id') != old.IDEA or run.get('plan_version') != '1.0'
            or summary.get('idea_id') != old.IDEA or summary.get('plan_version') != '1.0'
            or Path(run.get('source_system', '')).resolve() != source
            or Path(run.get('partition_run', '')).resolve() != previous
            or run.get('scene_ids') != meta['ids'] or run.get('completed_scenes') != len(meta['ids'])
            or run.get('methods') != list(old.METHODS) or summary.get('scene_ids') != meta['ids']
            or not summary.get('all_A0_exact_replays_verified') or not summary.get('no_expansion')):
        raise ValueError('completed original115 fixed V32 required')
    return meta


def run(args):
    source, previous, control, output = (Path(getattr(args, k)).resolve()
                                       for k in ('source_system', 'partition_run', 'control_run', 'output'))
    for protected in (source.parent, previous, control):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new output must not overlap immutable inputs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be 1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), source_system=str(source), partition_run=str(previous),
        control_run=str(control), methods=list(METHODS), cpu_threads=args.cpu_threads, completed_scenes=0,
        model_forwards=0, models_constructed=0, optimizer_updates=0, gpu_used=False,
        scores_recomputed=False, groups_recomputed=False, threshold_reselected=False, normalizer_refitted=False,
        archive_read=False, csv_read=False, cached_reference_read=False, test_opened=False)
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
                os.sched_setaffinity(0, cpus); state['cpu_affinity'] = cpus
        import torch
        torch.set_num_threads(args.cpu_threads)
        meta = metadata(source, previous, control, old.frozen.read(args.manifest))
        checkpoint(scene_ids=meta['ids'], total_scenes=len(meta['ids']))
        (output/'scenes').mkdir(); (output/'private').mkdir()
        rows, arrays = [], []
        with np.load(meta['association']/'final-scores.npz', allow_pickle=False) as scores:
            old._same(scores['scene_ids'], meta['all_ids'], '056 score scene order')
            for sid in meta['ids']:
                record = old.frozen.load_record(scores, sid, str(meta['components'][sid]))
                fragment, _ = old._load_arrays(source, sid)
                state['cached_reference_read'] = True
                for key in ('node_ids', 'sources'):
                    old._same(fragment[key], record[key], 'original056 '+key)
                private114 = old.frozen.read(previous/'private'/f'scene-{sid}.json.gz')
                memberships = {r: np.asarray(private114['memberships'][r]) for r in PARTITIONS}
                old._same(memberships['A0'], fragment['learned_assoc_membership'], '056 A0 membership')
                with np.load(control/'private'/f'states-{sid}.npz', allow_pickle=False) as saved:
                    # A0 has no candidate pair masks; WS also stores three comparison
                    # masks with the same prefix. Reuse only the bound A0 artifact schema.
                    artifact_keys = {k[3:] for k in saved.files if k.startswith('A0_')}
                    retained = {r: {k: saved[r+'_'+k] for k in artifact_keys}
                                for r in PARTITIONS}
                public = old.frozen.read(control/'scenes'/f'scene-{sid}.json.gz')
                if (public['scene_id'] != sid or public['component_id'] != str(meta['components'][sid])
                        or public['node_count'] != len(record['node_ids']) or not public['A0_exact_replay_verified']):
                    raise ValueError('115 scene/source binding differs')
                row, errors, artifacts, weights = analyze_scene(fragment, memberships, retained, public)
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                payload = {name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()}
                payload.update({name+'_query_weights': value for name, value in weights.items()})
                np.savez_compressed(output/'private'/f'states-{sid}.npz', **payload)
                rows.append(row); arrays.append(errors)
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **aggregate(rows, arrays),
            components={cid: aggregate([r for r in rows if r['component_id'] == cid],
                [a for r, a in zip(rows, arrays) if r['component_id'] == cid]) for cid in sorted({r['component_id'] for r in rows})},
            original115_exact_replays_verified=True, fixed_R_population_invariants_verified=True,
            no_expansion=True, position_only=True, cached_reference_read=True,
            limitations=['repeated development V32, not independent generalization or credible peer comparison',
                'sources are equivalence classes, not independent errors or raw-observation deduplication',
                'nearest time and equal source mass are transparent controls, not learned/calibrated reliability',
                'fixed partition and first-anchor correspondence; W cannot remove duplicate outputs',
                'cached Q64 interpolation/OLS, not original native observations; reference generation unknown'])
        old.frozen.write(output/'scene-summary.json.gz', rows)
        checkpoint(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        old.frozen.write(output/'summary.json', result)
        checkpoint(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes(),
            finished_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'run.json', state)
        if result is not None and (output/'summary.json').exists():
            result.update(status=state['status'], error=state['error'])
            old.frozen.write(output/'summary.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-system', 'partition-run', 'control-run', 'manifest', 'output'):
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
