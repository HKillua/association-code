"""145 fixed T32/V32: endpoint/velocity factors and source-balanced fallback."""
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

from association.evaluation import unsupported_recovery as method
from tools.analysis import observation_common as common
from tools.analysis import observation_scale_pilot as original
from tools.analysis import source_risk_pilot as source
from tools.analysis import state_recovery_tail_probe as probe

old = common.old
IDEA = 'IDEA-UNSUPPORTED-RECOVERY-20261030'
INPUTS = ('native_run', 'value_run', 'partition_run', 'source_risk_run', 'quality_run', 'state_run', 'contribution_run')
T_OLD = ('WS_SOURCE_EQUAL', 'WS_SOURCE_NEAREST', 'WS_SHARED_Q_SOFT', method.QHARD)+original.NEW
V_OLD = source.METHODS+('WS_SOURCE_NEAREST',)+original.NEW
PAIRS = ((method.OBS, method.NEW[0]), (method.OBS, method.NEW[2]), (method.OBS, method.NEW[3]),
         (method.QHARD, method.NEW[4]), (method.NEW[4], method.NEW[5]))
PRIMARY = tuple(b+'_vs_'+a for a, b in PAIRS)


def harm_stats(value):
    d = np.asarray(value, float)
    return dict(count=len(d), improved=int((d < 0).sum()), harmed=int((d > 0).sum()), tied=int((d == 0).sum()),
        harm_sum_m=float(d[d > 0].sum()), benefit_sum_m=float(-d[d < 0].sum()),
        harm_gt1000_count=int((d > 1000).sum()), harm_gt1000_sum_m=float(d[d > 1000].sum()),
        maximum_harm_m=float(max(0., d.max())) if len(d) else None)


def additional_harm(rows, arrays):
    components = sorted({r['component_id'] for r in rows})
    largest = max(components, key=lambda c: sum(r['component_id'] == c for r in rows))
    def block(indices):
        return {p: harm_stats(np.concatenate([arrays[i]['pairs'][p]['all']['delta'] for i in indices])) for p in PRIMARY}
    return dict(all_components=block(range(len(rows))),
        by_component={c: block([i for i, r in enumerate(rows) if r['component_id'] == c]) for c in components},
        without_largest_component=block([i for i, r in enumerate(rows) if r['component_id'] != largest]),
        scenes=[dict(scene_id=r['scene_id'], **block([i])) for i, r in enumerate(rows)], largest_component_id=largest)


def run(args):
    paths = {key: Path(getattr(args, key)).resolve() for key in INPUTS}
    output = Path(args.output).resolve()
    if any(output == p or output in p.parents or p in output.parents for p in paths.values()):
        raise ValueError('output must not overlap immutable input')
    output.mkdir(parents=True, exist_ok=False)
    (output/'private').mkdir()
    start = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0', started_at=datetime.now(timezone.utc).isoformat(),
        command=sys.argv, **{k: str(p) for k, p in paths.items()}, manifest=str(args.manifest),
        completed_training_scene_count=0, completed_scene_count=0, analytic_replay_calls=0,
        native_events_processed=0, transition_steps=0, rts_steps=0, neural_updates=0, new_reference_fits=0,
        test_opened=False, gpu_used=False, cpu_threads=args.cpu_threads, costs={}, time_policy='advisory')
    def check(**extra):
        state.update(extra, elapsed_seconds=time.monotonic()-start, peak_rss_bytes=old.frozen.peak_rss_bytes())
        state['advisory_exceeded'] = state['elapsed_seconds'] > 1800
        old.frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > 6*1024**3:
            raise MemoryError('6GiB RSS exceeded')
    try:
        if sys.platform.startswith('linux'):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            limit = min([12*1024**3]+[v for v in (soft, hard) if v != resource.RLIM_INFINITY])
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        manifest = old.frozen.read(args.manifest)
        tids, vids = common.metadata(*(paths[k] for k in INPUTS[:4]), manifest)
        for key in ('quality_run', 'state_run', 'contribution_run'):
            record = old.frozen.read(paths[key]/'run.json')
            if record['status'] != 'COMPLETED' or record['exit_code'] != 0 or record['scene_ids'] != vids or record['training_scene_ids'] != tids:
                raise ValueError('original population/terminal '+key)
        check(training_scene_ids=tids, scene_ids=vids)
        summaries = {}
        for split, ids, retained_names in (('T_core', tids, T_OLD), ('V_select', vids, V_OLD)):
            rows, arrays, stage = [], [], time.monotonic()
            (output/'scenes'/split).mkdir(parents=True)
            names = retained_names+method.NEW
            for sid in ids:
                check(phase=split+'_prediction', last_scene_id=sid)
                members = common.membership(paths['value_run'], paths['partition_run'], split, sid)
                motion, groups, mean, pack, _ = common.physical(paths['native_run'], split, sid, members)
                quality = common.load_artifacts(paths['quality_run']/'private'/f'states-{split}-{sid}.npz', ('WS_SOURCE_NEAREST',), template='WS_SOURCE_EQUAL')
                risk_file = paths['contribution_run']/'private'/f'states-{split}-{sid}.npz' if split == 'T_core' else paths['source_risk_run']/'private'/f'states-{sid}.npz'
                risk_names = ('WS_SOURCE_EQUAL', 'WS_SHARED_Q_SOFT', method.QHARD) if split == 'T_core' else source.METHODS
                retained = dict(common.load_artifacts(risk_file, risk_names, template='WS_SOURCE_EQUAL'), **quality)
                retained.update(common.load_artifacts(paths['state_run']/'private'/f'states-{split}-{sid}.npz', original.NEW, template='WS_SOURCE_EQUAL'))
                stored = common.load_npz(paths['state_run']/'private'/f'native-state-{split}-{sid}.npz')
                for key in ('node_ids', 'query_seconds'):
                    old._same(motion[key], stored[key], '137 motion '+key)
                native = common.load_npz(paths['native_run']/'private/native-events'/split/f'scene-{sid}.npz')
                old._same(motion['node_ids'], native['node_ids'], 'native node binding')
                old._same(motion['query_seconds'], native['query_seconds'], 'native query binding')
                zero = np.asarray([~motion['support'][g].any(0) for g in groups])
                needed = sorted({int(node) for gid, group in enumerate(groups) if zero[gid].any() for node in group})
                node_positions, traces = {}, {}
                rho_index = int(stored['obs_rho_index'])
                rho = float(stored['unique_rhos'][rho_index])
                for node in needed:
                    a, b = native['event_offsets'][node:node+2]
                    times = native['event_times_seconds'][a:b]
                    trace = probe.trace_cv(times, native['event_values'][a:b, :2], motion['query_seconds'], rho)
                    probe.same(trace['RTS']['positions'], stored[f'rho_{rho_index}__rts_positions'][node], '137 Q64 replay', numeric=True)
                    node_positions[node] = method.outside_positions(times, motion['query_seconds'], trace, motion['ols_velocity_mps'][node])
                    traces[str(node)] = trace['RTS']['endpoints']
                    state['analytic_replay_calls'] += 1
                    state['native_events_processed'] += trace['native_event_count']
                    state['transition_steps'] += trace['transition_steps']
                    state['rts_steps'] += trace['rts_backward_steps']
                prediction, zero = method.assemble(motion, groups, retained[method.OBS]['group_states'], retained[method.QHARD]['group_states'], node_positions)
                # Compare full OLS fallback against its original saved physical states before evaluation.
                probe.same(prediction[method.NEW[2]][zero, :2], mean[zero, :2], 'native endpoint/OLS velocity original fallback', numeric=True)
                np.savez_compressed(output/'private'/f'predictions-{split}-{sid}.npz', zero=zero,
                    **{name: values for name, values in prediction.items()}, **{'endpoint_'+key: value for key, value in traces.items()})
                prediction.update({name: artifact['group_states'] for name, artifact in retained.items()})
                ev = common.evaluation(paths['native_run'], split, sid, motion)
                row, local, artifacts = common.evaluate_predictions(motion, groups, pack, ev, prediction, retained, method_names=names, split=split)
                row.update(scene_id=sid, component_id=str(manifest['components'][sid]), split=split,
                    zero_group_queries=int(zero.sum()), replayed_nodes=len(needed), fixed_support_in_outputs_exact=True)
                common.save_artifacts(output/'private'/f'states-{split}-{sid}.npz', artifacts)
                old.frozen.write(output/'scenes'/split/f'scene-{sid}.json.gz', row)
                rows.append(row); arrays.append(local)
                check(**{('completed_training_scene_count' if split == 'T_core' else 'completed_scene_count'): len(rows)})
            summaries[split] = common.summarize(rows, arrays, method_names=names, primary_pairs=PRIMARY)
            summaries[split]['additional_actual_harm'] = additional_harm(rows, arrays)
            old.frozen.write(output/('T-summary.json' if split == 'T_core' else 'summary.json'), summaries[split])
            old.frozen.write(output/('T-scene-summary.json.gz' if split == 'T_core' else 'scene-summary.json.gz'), rows)
            state['costs'][split+'_seconds'] = time.monotonic()-stage
        check(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat(), phase='completed')
    except BaseException as error:
        check(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}', finished_at=datetime.now(timezone.utc).isoformat())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in INPUTS+('manifest', 'output'):
        parser.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args()
    def stop(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        run(args)
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
