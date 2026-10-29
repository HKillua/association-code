"""123: import121 V32 and120 eight controls; compute only268x2 priority arms."""
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

from tools.analysis import source_preference_control as p
from tools.analysis import shared_component_full_control as shared

q, old = p.qc, p.old
IDEA = 'IDEA-SOURCE-PREFERENCE-FULL-CONTROL-20261029'
PILOT_RUN = 'LEVEL1-SOURCE-PREFERENCE-20261029-121-attempt-1'
SHARED_RUN = 'LEVEL1-SHARED-COMPONENT-FULL-20261029-120-attempt-1'


def metadata(source, previous, pilot, control, manifest):
    meta = old.metadata(source, previous, manifest)
    for path, idea, ids, methods in ((pilot, p.IDEA, meta['ids'], p.METHODS),
                                   (control, shared.IDEA, meta['all_ids'], p.LEGACY_METHODS)):
        run, summary = (old.frozen.read(path/name) for name in ('run.json', 'summary.json'))
        if (run.get('status') != summary.get('status') or run.get('status') != 'COMPLETED'
                or run.get('exit_code') != 0 or run.get('idea_id') != idea or summary.get('idea_id') != idea
                or run.get('plan_version') != '1.0' or summary.get('plan_version') != '1.0'
                or Path(run.get('source_system', '')).resolve() != source
                or Path(run.get('partition_run', '')).resolve() != previous
                or run.get('scene_ids') != ids or summary.get('scene_ids') != ids
                or run.get('completed_scenes') != len(ids) or run.get('methods') != list(methods)
                or not summary.get('fixed_R_population_invariants_verified')):
            raise ValueError('completed120/121 source, population and method binding required')
    if pilot.name != PILOT_RUN or control.name != SHARED_RUN:
        raise ValueError('immutable original120/121 runs required')
    pilot_run, control_run = (old.frozen.read(path/'run.json') for path in (pilot, control))
    if (Path(pilot_run['shared_run']).resolve() != Path(control_run['pilot_run']).resolve()
            or control_run.get('original32_ids') != meta['ids']):
        raise ValueError('120/121 original118 calibration lineage differs')
    meta['pilot_ids'], meta['ids'] = meta['ids'], meta['all_ids']
    return meta


def aggregate(rows, arrays):
    result = q.aggregate(rows, arrays, method_names=p.METHODS)
    result.update(scene_prediction_arms_imported=sum(len(r['imported_prediction_methods']) for r in rows),
                  scene_prediction_arms_computed=sum(len(r['newly_computed_prediction_methods']) for r in rows))
    return result


def run(args):
    source, previous, pilot, control, output = (Path(getattr(args, key)).resolve()
        for key in ('source_system', 'partition_run', 'pilot_run', 'control_run', 'output'))
    for protected in (source.parent, previous, pilot, control):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new output must not overlap immutable inputs')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), source_system=str(source), partition_run=str(previous),
        pilot_run=str(pilot), control_run=str(control), methods=list(p.METHODS), completed_scenes=0,
        cpu_threads=args.cpu_threads, model_forwards=0, models_constructed=0, optimizer_updates=0,
        gpu_used=False, scores_recomputed=False, groups_recomputed=False, threshold_reselected=False,
        normalizer_refitted=False, centers_refitted=False, huber_recomputed=False,
        archive_read=False, csv_read=False, cached_reference_read=False, test_opened=False,
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
                for key in ('node_ids', 'sources'):
                    old._same(fragment[key], record[key], '056 '+key)
                membership = np.asarray(old.frozen.read(previous/'private'/f'scene-{sid}.json.gz')['memberships']['WS'])
                original = sid in meta['pilot_ids']
                path = pilot if original else control
                public = old.frozen.read(path/'scenes'/f'scene-{sid}.json.gz')
                if (public['scene_id'] != sid or public['component_id'] != cid
                        or public['node_count'] != len(record['node_ids'])
                        or not public.get('fixed_R_population_invariants_verified')):
                    raise ValueError('120/121 scene binding differs')
                with np.load(path/'private'/f'states-{sid}.npz', allow_pickle=False) as payload:
                    retained, calibration = p.load_saved(payload)
                    imported = ({name: {key: np.asarray(payload[name+'_'+key]).copy() for key in retained['WS_MEAN']}
                                 for name in p.METHODS} if original else None)
                    private = ({key[len('source_preference_'):]: np.asarray(payload[key]).copy()
                                for key in payload.files if key.startswith('source_preference_')} if original else None)
                row, error_arrays, artifacts, private = p.analyze_scene(fragment, membership, retained, calibration,
                    public, imported_artifacts=imported, imported_private=private)
                row.update(stratum='original32' if original else 'remaining268',
                    component_seen_in_original32=cid in pilot_components,
                    imported_prediction_methods=list(p.METHODS if original else p.LEGACY_METHODS),
                    newly_computed_prediction_methods=[] if original else list(p.NEW_METHODS),
                    original121_artifacts_exact_verified=original)
                old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                if not original:
                    payload = {name+'_'+key: value for name, artifact in artifacts.items() for key, value in artifact.items()}
                    payload.update(calibration)
                    payload.update({'source_preference_'+key: value for key, value in private.items()})
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
            original32_ids=meta['pilot_ids'],
            components={cid: subset([i for i,r in enumerate(rows) if r['component_id'] == cid]) for cid in components},
            original32=subset([i for i,r in enumerate(rows) if r['stratum'] == 'original32']),
            remaining268=subset([i for i,r in enumerate(rows) if r['stratum'] == 'remaining268']),
            remaining268_seen_components=subset([i for i,r in enumerate(rows)
                if r['stratum'] == 'remaining268' and r['component_seen_in_original32']]),
            remaining268_unseen_components=subset([i for i,r in enumerate(rows)
                if r['stratum'] == 'remaining268' and not r['component_seen_in_original32']]),
            without_largest_component=subset([i for i,r in enumerate(rows) if r['component_id'] != largest]),
            original121_all_artifacts_exact_verified=True, original120_eight_artifacts_exact_verified=True,
            fixed_R_population_invariants_verified=True, no_expansion_beyond_V300=True,
            centers_refitted=False, huber_recomputed=False,
            limitations=['repeated development V300 with shared identities; not independent test',
                'source order is empirical; absolute source reliability and center uncertainty not identified',
                'frozen WS/full output/first representative, no coverage/duplicate improvement',
                'offline position-only analytic control, not learned fusion or credible peer victory'])
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
