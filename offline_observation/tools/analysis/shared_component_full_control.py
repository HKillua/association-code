"""120: import all original118 arms and extend the frozen recipe to remaining268.

Repeated development V300, never independent test or a physical-bias claim.
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

from tools.analysis import shared_component_control as p
from tools.analysis import query_contribution_full_control as full

q, old = p.qc, p.old
IDEA = 'IDEA-SHARED-COMPONENT-FULL-CONTROL-20261029'
PILOT_RUN = 'LEVEL1-SHARED-COMPONENT-20261029-118-attempt-1'
CONTROL_RUN = 'LEVEL1-QUERY-CONTRIBUTION-FULL-20261029-117-attempt-1'


def load_artifacts(path, names):
    with np.load(path, allow_pickle=False) as saved:
        keys = {key[len('WS_MEAN_'):] for key in saved.files if key.startswith('WS_MEAN_')}
        if not keys:
            raise ValueError('complete saved WS_MEAN artifact required')
        return {name: {key: saved[name+'_'+key] for key in keys} for name in names}


def metadata(source, previous, pilot, control, manifest):
    meta = old.metadata(source, previous, manifest)
    for path, idea, ids, methods in ((pilot, p.IDEA, meta['ids'], p.METHODS),
                                   (control, full.IDEA, meta['all_ids'], full.METHODS)):
        run, summary = (old.frozen.read(path/k) for k in ('run.json', 'summary.json'))
        if (run.get('status') != 'COMPLETED' or summary.get('status') != 'COMPLETED'
                or run.get('idea_id') != idea or summary.get('idea_id') != idea
                or run.get('plan_version') != '1.0' or summary.get('plan_version') != '1.0'
                or Path(run.get('source_system', '')).resolve() != source
                or Path(run.get('partition_run', '')).resolve() != previous
                or run.get('scene_ids') != ids or summary.get('scene_ids') != ids
                or run.get('completed_scenes') != len(ids) or run.get('methods') != list(methods)
                or not summary.get('fixed_R_population_invariants_verified')):
            raise ValueError('completed original118/117 source, population and recipe binding required')
    if pilot.name != PILOT_RUN or control.name != CONTROL_RUN:
        raise ValueError('original118 and full117 runs required')
    summary = old.frozen.read(pilot/'summary.json')
    if not summary.get('original115_exact_replays_verified') or not summary.get('no_expansion'):
        raise ValueError('original118 fixed32 exact control binding required')
    meta['pilot_ids'], meta['ids'] = meta['ids'], meta['all_ids']
    return meta


def analyze_scene(fragment, memberships, controls, public, *, imported=None, pilot_public=None):
    """Use the original constructor only for new scenes, then bind frozen controls."""
    if imported is not None:
        if pilot_public is None or not pilot_public.get('original115_exact_replay_verified'):
            raise ValueError('original118 public provenance required')
        public = dict(public, shared_component_audit=pilot_public['shared_component_audit'])
    row, arrays, artifacts, private = p.analyze_scene(fragment, memberships,
        {'WS': controls['WS_MEAN']}, public, imported_artifacts=imported, replay_source='117')
    for name in ('WS_MEAN', 'WS_ANCHOR', 'WS_NEAREST_SUPPORTED'):
        if set(controls[name]) != set(artifacts[name]):
            raise ValueError('117 ordinary control artifact schema differs')
        for key in artifacts[name]:
            old._same(artifacts[name][key], controls[name][key], '117 exact ordinary control/'+name+'/'+key)
        full.compare_saved_report(row['methods'][name], public['methods'][name], '117 ordinary report/'+name)
    if imported is not None:
        for name in p.METHODS:
            full.compare_saved_report(row['methods'][name], pilot_public['methods'][name], '118 imported report/'+name)
        row['original118_all_artifacts_exact_verified'] = True
    row.update(stratum='original32' if imported is not None else 'remaining268',
        imported_prediction_methods=list(p.METHODS) if imported is not None else [],
        newly_computed_prediction_methods=[] if imported is not None else list(p.METHODS),
        original117_ordinary_controls_exact_verified=True)
    return row, arrays, artifacts, private


def aggregate(rows, arrays):
    out = q.aggregate(rows, arrays, method_names=p.METHODS)
    out.update(scene_prediction_arms_imported=sum(len(r['imported_prediction_methods']) for r in rows),
               scene_prediction_arms_computed=sum(len(r['newly_computed_prediction_methods']) for r in rows))
    return out


def run(args):
    source, previous, pilot, control, output = (Path(getattr(args, key)).resolve()
        for key in ('source_system', 'partition_run', 'pilot_run', 'control_run', 'output'))
    for protected in (source.parent, previous, pilot, control):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new output must not overlap immutable inputs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be 1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), source_system=str(source), partition_run=str(previous),
        pilot_run=str(pilot), control_run=str(control), methods=list(p.METHODS), completed_scenes=0,
        cpu_threads=args.cpu_threads, model_forwards=0, models_constructed=0, optimizer_updates=0,
        gpu_used=False, scores_recomputed=False, groups_recomputed=False, threshold_reselected=False,
        normalizer_refitted=False, archive_read=False, csv_read=False, cached_reference_read=False,
        test_opened=False, correction_folds=3, minimum_training_groups=2,
        huber_delta_m=p.HUBER_DELTA, huber_steps=p.HUBER_STEPS,
        imported_prediction_scene_arms=0, newly_computed_prediction_scene_arms=0)
    def checkpoint(**extra):
        state.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes(),
                     advisory_exceeded=time.monotonic()-started > 1800, **extra)
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
        meta = metadata(source, previous, pilot, control, old.frozen.read(args.manifest))
        original116 = Path(old.frozen.read(control/'run.json')['pilot_run'])
        pilot_components = {str(meta['components'][sid]) for sid in meta['pilot_ids']}
        checkpoint(scene_ids=meta['ids'], original32_ids=meta['pilot_ids'], total_scenes=len(meta['ids']))
        (output/'scenes').mkdir()
        (output/'private').mkdir()
        rows, arrays = [], []
        with np.load(meta['association']/'final-scores.npz', allow_pickle=False) as scores:
            old._same(scores['scene_ids'], meta['all_ids'], '056 score scene order')
            for sid in meta['ids']:
                cid = str(meta['components'][sid])
                record = old.frozen.load_record(scores, sid, cid)
                fragment, _ = old._load_arrays(source, sid)
                state['cached_reference_read'] = True
                private114 = old.frozen.read(previous/'private'/f'scene-{sid}.json.gz')
                memberships = {r: np.asarray(private114['memberships'][r]) for r in q.PARTITIONS}
                public114 = old.frozen.read(previous/'scenes'/f'scene-{sid}.json.gz')
                full.bind_partition_source(record, fragment, public114, memberships)
                controls = load_artifacts(control/'private'/f'states-{sid}.npz',
                    ('WS_MEAN', 'WS_ANCHOR', 'WS_NEAREST_SUPPORTED')) if sid not in meta['pilot_ids'] else None
                # 117 imported original32 ordinary arrays by reference, keeping
                # only its new NEAREST_ALL arrays. Read their immutable116 source.
                if controls is None:
                    controls = load_artifacts(original116/'private'/f'states-{sid}.npz',
                        ('WS_MEAN', 'WS_ANCHOR', 'WS_NEAREST_SUPPORTED'))
                public = old.frozen.read(control/'scenes'/f'scene-{sid}.json.gz')
                if public['scene_id'] != sid or public['component_id'] != cid:
                    raise ValueError('117 scene/component binding differs')
                imported, pilot_public = None, None
                if sid in meta['pilot_ids']:
                    imported = load_artifacts(pilot/'private'/f'states-{sid}.npz', p.METHODS)
                    pilot_public = old.frozen.read(pilot/'scenes'/f'scene-{sid}.json.gz')
                    if pilot_public['scene_id'] != sid or pilot_public['component_id'] != cid:
                        raise ValueError('118 scene/component binding differs')
                row, error_arrays, artifacts, private = analyze_scene(fragment, memberships, controls, public,
                    imported=imported, pilot_public=pilot_public)
                row['component_seen_in_original32'] = cid in pilot_components
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                if imported is None:
                    payload = {name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()}
                    payload.update(private)
                    np.savez_compressed(output/'private'/f'states-{sid}.npz', **payload)
                rows.append(row)
                arrays.append(error_arrays)
                state['imported_prediction_scene_arms'] += len(row['imported_prediction_methods'])
                state['newly_computed_prediction_scene_arms'] += len(row['newly_computed_prediction_methods'])
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        components = sorted({r['component_id'] for r in rows})
        def subset(indices):
            return aggregate([rows[i] for i in indices], [arrays[i] for i in indices]) if indices else None
        largest = max(components, key=lambda cid: sum(r['component_id'] == cid for r in rows))
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **aggregate(rows, arrays),
            components={cid: subset([i for i,r in enumerate(rows) if r['component_id'] == cid]) for cid in components},
            original32=subset([i for i,r in enumerate(rows) if r['stratum'] == 'original32']),
            remaining268=subset([i for i,r in enumerate(rows) if r['stratum'] == 'remaining268']),
            remaining268_seen_components=subset([i for i,r in enumerate(rows)
                if r['stratum'] == 'remaining268' and r['component_seen_in_original32']]),
            remaining268_unseen_components=subset([i for i,r in enumerate(rows)
                if r['stratum'] == 'remaining268' and not r['component_seen_in_original32']]),
            without_largest_component=subset([i for i,r in enumerate(rows) if r['component_id'] != largest]),
            original118_all_artifacts_exact_verified=True, original117_ordinary_controls_exact_verified=True,
            fixed_R_population_invariants_verified=True, no_expansion_beyond_V300=True,
            limitations=['repeated development V300 and shared identities; not independent test/peer reproduction',
                'gauge was fixed before results; no identified absolute bias or unbiased source',
                'position only; full output population/duplicate groups unchanged',
                'cached Q64 and grouping are offline; no online end-to-end claim'])
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
    for key in ('source-system', 'partition-run', 'pilot-run', 'control-run', 'manifest', 'output'):
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
