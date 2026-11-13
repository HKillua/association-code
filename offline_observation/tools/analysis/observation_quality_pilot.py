"""136: fixed offline observation-quality contribution pilot, T32/V32 once.

Uses completed physical caches, performs no reference fitting, and preserves
the full output population. Existing 133 full artifacts are imported only on V.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.evaluation import observation_quality as quality
from association.evaluation import source_risk as risk
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source

old = common.old
IDEA = 'IDEA-OBSERVATION-QUALITY-20261030'
EQUAL = 'WS_SOURCE_EQUAL'
POLICIES = {
    EQUAL: ('ols', 'equal', None),
    'WS_SOURCE_NEAREST': ('ols', 'nearest', None),
    'WS_OBS_OLS_F1': ('ols', 'inverse', 1.),
    'WS_OBS_OLS_F100': ('ols', 'inverse', 100.),
    'WS_OBS_LOO_F1': ('loo', 'inverse', 1.),
    'WS_OBS_LOO_F100': ('loo', 'inverse', 100.),
}
T_METHODS = tuple(POLICIES)
V_METHODS = source.METHODS + tuple(name for name in POLICIES if name != EQUAL)
PRIMARY = tuple(name+'_vs_'+EQUAL for name in POLICIES if name.startswith('WS_OBS_'))
RAM, ADDRESS_SPACE = 6*1024**3, 12*1024**3


def predict(motion, groups, mean, pack, native):
    """Physical-only entry point: outputs are frozen before any reference read."""
    node = quality.native_quality(native_cache=native, motion=motion)
    proxies = {
        'ols': quality.source_proxy(node_proxy_m2=node['ols_proxy_m2'],
            node_known=np.ones(len(motion['states']), bool), support=motion['support'], pack=pack),
        'loo': quality.source_proxy(node_proxy_m2=node['loo_proxy_m2'],
            node_known=node['loo_known'], support=motion['support'], pack=pack),
    }
    predictions, stored, audit = {}, {}, {}
    stored.update({'native_'+k: np.asarray(v) for k, v in node.items()})
    for label, proxy in proxies.items():
        stored[label+'__source_proxy_m2'] = proxy['quality_m2']
        stored[label+'__source_known'] = proxy['known']
    for method, (label, kind, floor) in POLICIES.items():
        weights = quality.observation_weights(proxy=proxies[label], pack=pack,
            nearest_seconds=motion['nearest_seconds'], support=motion['support'], kind=kind, floor_m2=floor)
        recovered = quality.recover_observation_policy(states=motion['states'], support=motion['support'],
            groups=groups, pack=pack, source_weights=weights['source_weights'], mean_states=mean)
        predictions[method] = recovered['group_states']
        stored[method+'__group_states'] = recovered['group_states']
        for key, value in weights.items():
            stored[method+'__'+key] = value
        audit[method] = dict(source.contribution_summary(pack, weights['source_weights']),
            uninformative_fallback_group_queries=int(weights['uninformative_fallback'].sum()),
            floor_hit_source_queries=int(weights['floor_hit'].sum()),
            available_source_queries=int(pack['available'].sum()),
            recovery=recovered['recovery_audit'])
    return predictions, stored, audit, proxies, node


def proxy_diagnostics(pack, groups, ev, proxies):
    """After-prediction descriptive diagnostics only; never a policy input."""
    labels = risk.source_labels(pack, groups, ev['identities'], ev['target_ids'], ev['target_states'], ev['target_valid'])
    out = {}
    for name, proxy in proxies.items():
        use = labels['labeled'] & proxy['known']
        x, y = np.log1p(proxy['quality_m2'][use]), np.log1p(labels['errors_m'][use])
        sx, sy = np.std(x) if len(x) else 0., np.std(y) if len(y) else 0.
        correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and sx > 0 and sy > 0 else None
        out[name] = dict(known_reference_source_queries=int(use.sum()),
            unknown_reference_source_queries=int((labels['labeled'] & ~proxy['known']).sum()),
            log_proxy_log_error_pearson=correlation,
            interpretation='descriptive source-query association; not calibration, independent samples or causation')
    return out, labels


def run(args):
    keys = ('native_run', 'value_run', 'partition_run', 'source_risk_run', 'output')
    paths = {key: Path(getattr(args, key)).resolve() for key in keys}
    output = paths['output']
    for key, path in paths.items():
        if key != 'output' and (output == path or output in path.parents or path in output.parents):
            raise ValueError('new output overlaps frozen input')
    if not 1 <= args.cpu_threads <= 4 or getattr(args, 'time_policy', 'advisory') != 'advisory':
        raise ValueError('CPU1..4 and advisory time policy required')
    output.mkdir(parents=True, exist_ok=False)
    (output/'private').mkdir()
    for split in ('T_core', 'V_select'):
        (output/'scenes'/split).mkdir(parents=True)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        **{key: str(value) for key, value in paths.items() if key != 'output'},
        manifest=str(Path(args.manifest).resolve()), T_methods=list(T_METHODS), methods=list(V_METHODS),
        fixed_floors_m2=[1., 100.], completed_training_scene_count=0, completed_scene_count=0,
        native_event_count=0, native_fragment_count=0, teacher_forward_calls=0,
        neural_models_constructed=0, optimizer_updates=0, target_reference_fit_count=0,
        analytic_head_fits=0, archive_read=False, csv_read=False, test_opened=False,
        predictions_frozen_before_reference=True, source_id_feature=False, cpu_threads=args.cpu_threads,
        gpu_used=False, time_policy='advisory', costs={})

    def check(**extra):
        state.update(extra, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes())
        state['advisory_exceeded'] = state['elapsed_seconds'] > 1800
        old.frozen.write(output/'run.json', state)
        if state['advisory_exceeded'] and not (output/'time_advisory.json').exists():
            old.frozen.write(output/'time_advisory.json', dict(elapsed_seconds=state['elapsed_seconds'],
                action='continue only original T32/V32 fixed policies; no new fitting or retry'))
        if state['peak_rss_bytes'] > RAM:
            raise MemoryError('6GiB resident budget exceeded')

    try:
        check(phase='binding')
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([ADDRESS_SPACE]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        manifest = old.frozen.read(args.manifest)
        tids, vids = common.metadata(paths['native_run'], paths['value_run'], paths['partition_run'],
                                      paths['source_risk_run'], manifest)
        check(training_scene_ids=tids, scene_ids=vids)
        summaries = {}
        for split, ids, methods in (('T_core', tids, T_METHODS), ('V_select', vids, V_METHODS)):
            rows, arrays = [], []
            stage = time.monotonic()
            for sid in ids:
                check(phase=split+'_physical_prediction', last_scene_id=sid)
                members = common.membership(paths['value_run'], paths['partition_run'], split, sid)
                motion, groups, mean, pack, saved = common.physical(paths['native_run'], split, sid, members)
                native = common.load_npz(paths['native_run']/'private/native-events'/split/f'scene-{sid}.npz')
                predictions, policies, wa, proxies, node = predict(motion, groups, mean, pack, native)
                saved.update(policies)
                saved['support'] = motion['support']
                np.savez_compressed(output/'private'/f'policy-{split}-{sid}.npz', **saved)
                retained = {}
                if split == 'V_select':
                    retained = common.load_artifacts(paths['source_risk_run']/'private'/f'states-{sid}.npz', source.METHODS)
                    old._same(retained['WS_MEAN']['node_group_indices'], members, '133/114 WS membership')
                    old._same(mean, retained['WS_MEAN']['group_states'], 'native exact Mean')
                    old._same(predictions[EQUAL], retained[EQUAL]['group_states'], '133 exact source-equal')
                    predictions.update({key: value['group_states'] for key, value in retained.items()})
                # New predictions are on disk before identity/reference evaluation.
                ev = common.evaluation(paths['native_run'], split, sid, motion)
                row, local_arrays, artifacts = common.evaluate_predictions(motion, groups, pack, ev, predictions,
                    retained, method_names=methods, split=split)
                diagnostic, labels = proxy_diagnostics(pack, groups, ev, proxies)
                np.savez_compressed(output/'private'/f'diagnostic-{split}-{sid}.npz',
                    errors_m=labels['errors_m'], labeled=labels['labeled'], sample_weight=labels['sample_weight'])
                common.save_artifacts(output/'private'/f'states-{split}-{sid}.npz', artifacts)
                row.update(scene_id=sid, component_id=str(manifest['components'][sid]), split=split,
                    node_count=len(motion['states']), source_support_scope=source.support_counts(pack),
                    source_contributions=wa, proxy_diagnostics=diagnostic, reference_diagnostic_scope=labels['audit'],
                    native_event_count=node['native_event_count'],
                    native_loo_unknown_fragments=int((~node['loo_known']).sum()),
                    native_loo_residual_count=int(node['loo_residual_count'].sum()),
                    fixed_population_invariants_verified=True, original133_exact_replays_verified=split == 'V_select')
                old.frozen.write(output/'scenes'/split/f'scene-{sid}.json.gz', row)
                rows.append(row)
                arrays.append(local_arrays)
                state['native_event_count'] += node['native_event_count']
                state['native_fragment_count'] += node['native_fragment_count']
                check(**{('completed_training_scene_count' if split == 'T_core' else 'completed_scene_count'): len(rows)})
            if len(rows) != 32 or any(len(row['methods']) != len(methods)
                or len(row['pairwise']) != len(methods)*(len(methods)-1)//2 for row in rows):
                raise ValueError('fixed scene/method/pair endpoint incomplete')
            state['costs'][split+'_seconds'] = time.monotonic()-stage
            summaries[split] = common.summarize(rows, arrays, method_names=methods, primary_pairs=PRIMARY)
            summaries[split]['contribution_counts'] = {
                name: {key: sum(row['source_contributions'][name][key] for row in rows)
                       for key in ('uninformative_fallback_group_queries', 'floor_hit_source_queries',
                                   'available_source_queries')} for name in POLICIES}
            old.frozen.write(output/('training-scene-summary.json.gz' if split == 'T_core' else 'scene-summary.json.gz'), rows)
        check(phase='finalizing')
        summary = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **summaries['V_select'],
            training_summary=summaries['T_core'], primary_comparisons=list(PRIMARY),
            costs=state['costs'], elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'],
            native_event_count=state['native_event_count'], native_fragment_count=state['native_fragment_count'],
            target_reference_fit_count=0, limitations=[
                'Offline full-fragment OLS and double-sided interpolation; no online claim',
                'Predictability proxies are not covariance; constant bias, correlated noise and maneuvers remain',
                'Two fixed engineering floors are compared without selecting on V',
                'Actual K2, previously developed V32 and shared components limit transfer claims',
                'Only position changes; original speed/heading and full population preserved',
                'Upstream teacher/grouping is supervised; only new contribution is reference-free'])
        old.frozen.write(output/'summary.json', summary)
        check(status='COMPLETED', exit_code=0, phase='completed', finished_at=datetime.now(timezone.utc).isoformat())
        return summary
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started,
            finished_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-run', 'value-run', 'partition-run', 'source-risk-run', 'manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'signal{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        return 1
    print(f'COMPLETED: {args.output}/summary.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
