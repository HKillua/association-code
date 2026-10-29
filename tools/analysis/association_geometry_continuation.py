"""130: new bounded-memory successor to failed128, importing six complete scenes.

The original128 attempt remains failed/stale on disk. This is an independent
published task, not a restart; candidate/formula/tolerance/population stay fixed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import filecmp
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tools.analysis import association_geometry_signal as p

old = p.old
IDEA = 'IDEA-ASSOCIATION-GEOMETRY-CONTINUATION-20261029'
FAILED_RUN = 'LEVEL1-ASSOCIATION-GEOMETRY-SIGNAL-20261029-128-attempt-1'
REUSE_COUNT, PAIR_BLOCK = 6, 4096


def chunked_pair_geometry(states, pairs):
    """Only bound transient pair batches; call the unchanged128 distance."""
    states, pairs = np.asarray(states, np.float64), np.asarray(pairs)
    result = np.empty((len(pairs), states.shape[1]), np.float64)
    for left in range(0, len(pairs), PAIR_BLOCK):
        block = pairs[left:left+PAIR_BLOCK]
        result[left:left+len(block)] = p.distance(states[block[:, 0], :, :2], states[block[:, 1], :, :2])
    return result


def validate_failed_prefix(source, partition, failed, meta, failure_path):
    """Explicit oomd terminal evidence supersedes stale RUNNING metadata only."""
    run = old.frozen.read(failed/'run.json')
    failure = old.frozen.read(failure_path)
    launch = old.frozen.read(Path(str(failed)+'-launch')/'launch.json')
    if (failed.name != FAILED_RUN or run.get('idea_id') != p.IDEA or run.get('plan_version') != '1.0'
            or run.get('scene_ids') != meta['ids'] or run.get('completed_scenes') != REUSE_COUNT
            or run.get('last_scene_id') != meta['ids'][REUSE_COUNT-1]
            or Path(run.get('source_system', '')).resolve() != source
            or Path(run.get('partition_run', '')).resolve() != partition
            or any(run.get(key) != 0 for key in ('models_constructed', 'model_forwards', 'optimizer_updates'))
            or run.get('test_opened') is not False or run.get('reference_array_fields_loaded') is not False
            or run.get('partition_changed') is not False or run.get('physical_predictions_changed') is not False
            or failure.get('effective_status') != 'FAILED' or failure.get('cause') != 'systemd_oomd'
            or failure.get('completed_scenes') != REUSE_COUNT or failure.get('unit') != launch.get('unit')
            or failure.get('systemd', {}).get('ExecMainStatus') != '9'
            or failure.get('systemd', {}).get('Result') != 'signal'
            or failure.get('systemd', {}).get('ActiveState') not in ('failed', 'inactive')
            or failure.get('preserved_run_status') != run.get('status')
            or failure.get('launcher_status_is_stale') is not True):
        raise ValueError('explicit failed128 oomd terminal/source/complete-six-prefix binding required')
    for sid in meta['ids'][:REUSE_COUNT]:
        for relative in ('private/features-'+sid+'.npz', 'private/evaluation-'+sid+'.npz', 'scenes/scene-'+sid+'.json.gz'):
            if not (failed/relative).is_file():
                raise ValueError('complete128 prefix feature/evaluation/public artifacts required')
    return run, failure


def copy_exact(source, destination):
    shutil.copyfile(source, destination)
    if not filecmp.cmp(source, destination, shallow=False):
        raise ValueError('imported128 artifact bytes differ after copy')


def compare_evidence(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('imported128 evaluation schema differs from frozen features/labels')
    for key in expected:
        old._same(actual[key], expected[key], '128 exact imported evaluation/'+key)


def import_scene(failed, output, sid, component, physical, pairs, scores, membership, identities, labels):
    """No feature constructor/candidate solver/distance call on imported scenes."""
    with np.load(output/'private'/f'features-{sid}.npz', allow_pickle=False) as saved:
        features = {key: saved[key].copy() for key in saved.files}
    with np.load(output/'private'/f'evaluation-{sid}.npz', allow_pickle=False) as saved:
        evidence = {key: saved[key].copy() for key in saved.files}
    row = old.frozen.read(output/'scenes'/f'scene-{sid}.json.gz')
    n = len(physical['node_ids'])
    if (row.get('scene_id') != sid or row.get('component_id') != component or row.get('node_count') != n
            or row.get('feature_file_written_before_identity_read') is not True
            or row.get('reference_array_fields_loaded') is not False
            or not np.array_equal(features['feature_names'], p.FEATURE_NAMES)
            or features['features'].shape != (len(features['candidate_offsets'])-1, len(p.FEATURE_NAMES))
            or features['pair_distance_m'].shape != (len(pairs), 64)
            or features['pair_joint_support'].shape != (len(pairs), 64)
            or not np.isfinite(features['pair_distance_m']).all()
            or row['feature_audit']['candidate_count'] != len(features['features'])
            or row['feature_audit']['original_group_count'] != len(old.groups_from_membership(membership, n))):
        raise ValueError('imported128 complete feature/public physical population differs')
    old._same(features['pairs'], pairs, '128 exact source pairs')
    old._same(features['pair_scores'], scores, '128 exact source scores')
    old._same(features['threshold'], p.old.frozen.THRESHOLD, '128 exact threshold')
    for left in range(0, len(pairs), PAIR_BLOCK):
        block = pairs[left:left+PAIR_BLOCK]
        old._same(features['pair_joint_support'][left:left+len(block)],
            physical['support'][block[:, 0]] & physical['support'][block[:, 1]], '128 exact visible joint support')
    known_pair = (identities[pairs[:, 0]] != '') & (identities[pairs[:, 1]] != '')
    old._same((identities[pairs[:, 0]] == identities[pairs[:, 1]])[known_pair], labels[known_pair],
              '128 source identity/pair label binding')
    replay = p.evaluate_features(features, identities)
    replay['original'] = features['candidate_is_original']
    compare_evidence(evidence, replay)
    if p.summarize(evidence) != row['candidate_population']:
        raise ValueError('128 exact public evaluation differs from imported arrays')
    return row, evidence


def run(args):
    source, partition, failed, output = (Path(getattr(args, key)).resolve()
        for key in ('source_system', 'partition_run', 'failed_run', 'output'))
    failure_path = Path(args.failure_evidence).resolve() if args.failure_evidence is not None else failed/'terminal-synthesis.json'
    for protected in (source.parent, partition, failed, Path(str(failed)+'-launch')):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('new130 output must not overlap frozen inputs or failed128')
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 4:
        raise ValueError('cpu_threads must be1..4')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), source_system=str(source), partition_run=str(partition),
        failed_run=str(failed), failure_evidence=str(failure_path), completed_scenes=0,
        imported_feature_scenes=0, newly_constructed_feature_scenes=0, pair_geometry_block_size=PAIR_BLOCK,
        original128_restarted=False, original128_files_changed=False, imported_features_recomputed=False,
        original128_round_mean_fallback_preserved=True, original128_formulas_tolerances_preserved=True,
        models_constructed=0, model_forwards=0, optimizer_updates=0, gpu_used=False,
        partition_changed=False, physical_predictions_changed=False, scores_recomputed=False,
        reference_array_fields_loaded=False, identity_array_fields_loaded=False,
        label_arrays_loaded_only_after_feature_freeze=True, test_opened=False,
        metadata_binding_uses_existing_evaluation_summaries=True,
        two_member_tolerance=dict(atol_metres=p.TWO_ATOL_M, rtol=p.TWO_RTOL),
        euclidean_tolerance=dict(atol_m2=p.EUCLIDEAN_ATOL_M2, rtol=p.EUCLIDEAN_RTOL))
    def checkpoint(**extra):
        state.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=old.frozen.peak_rss_bytes(),
            advisory_exceeded=time.monotonic()-started > 900, **extra)
        old.frozen.write(output/'run.json', state)
        if state['peak_rss_bytes'] > old.frozen.RAM_BUDGET:
            raise MemoryError('4GiB resident-memory budget exceeded')
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
        meta = old.metadata(source, partition, old.frozen.read(args.manifest))
        run128, failure = validate_failed_prefix(source, partition, failed, meta, failure_path)
        prefix = meta['ids'][:REUSE_COUNT]
        checkpoint(scene_ids=meta['ids'], total_scenes=len(meta['ids']), original128_imported_scene_ids=prefix,
            original128_effective_status=failure['effective_status'], original128_cause=failure['cause'],
            original128_last_reported_peak_rss_bytes=run128.get('peak_rss_bytes'),
            original128_true_peak_rss_bytes=None)
        (output/'scenes').mkdir(); (output/'private').mkdir()
        rows, all_evidence, provenance = [], [], []
        with np.load(meta['association']/'final-scores.npz', allow_pickle=False) as scores:
            old._same(scores['scene_ids'], meta['all_ids'], '056 full score scene order')
            for sid in meta['ids']:
                with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as saved:
                    physical = {key: saved[key].copy() for key in p.PHYSICAL_KEYS}
                for key in ('node_ids', 'sources'):
                    old._same(physical[key], scores[f'{sid}_{key}'], '056 '+key)
                pairs, values = scores[f'{sid}_pairs'], scores[f'{sid}_scores']
                membership = np.asarray(old.frozen.read(partition/'private'/f'scene-{sid}.json.gz')['memberships']['WS'])
                component = str(meta['components'][sid])
                if sid in prefix:
                    # The original feature files already predate original128's
                    # identity read; copy them now before evaluating the labels.
                    for relative in ('private/features-'+sid+'.npz', 'private/evaluation-'+sid+'.npz', 'scenes/scene-'+sid+'.json.gz'):
                        copy_exact(failed/relative, output/relative)
                    with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as saved:
                        identities = saved['identities'].copy()
                    state['identity_array_fields_loaded'] = True
                    row, evidence = import_scene(failed, output, sid, component, physical, pairs, values,
                        membership, identities, np.asarray(scores[f'{sid}_labels']))
                    state['imported_feature_scenes'] += 1
                    mode = 'imported_complete128_without_feature_recomputation'
                else:
                    groups = old.groups_from_membership(membership, len(physical['node_ids']))
                    features, audit = p.construct_features(physical['native_states'], physical['support'], physical['sources'],
                        groups, pairs, values, old.frozen.THRESHOLD, pair_geometry_builder=chunked_pair_geometry)
                    np.savez_compressed(output/'private'/f'features-{sid}.npz', **features)
                    with np.load(source/'predictions'/f'states-{sid}.npz', allow_pickle=False) as saved:
                        identities = saved['identities'].copy()
                    state['identity_array_fields_loaded'] = True
                    labels = np.asarray(scores[f'{sid}_labels'])
                    known = (identities[pairs[:, 0]] != '') & (identities[pairs[:, 1]] != '')
                    old._same((identities[pairs[:, 0]] == identities[pairs[:, 1]])[known], labels[known],
                              '056 post-freeze identity/pair label binding')
                    evidence = p.evaluate_features(features, identities)
                    evidence['original'] = features['candidate_is_original']
                    np.savez_compressed(output/'private'/f'evaluation-{sid}.npz', **evidence)
                    row = dict(scene_id=sid, component_id=component, node_count=len(identities),
                        feature_audit=audit, candidate_population=p.summarize(evidence),
                        feature_file_written_before_identity_read=True, reference_array_fields_loaded=False)
                    old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz', row)
                    # Evidence retains only small feature-column views. Release
                    # this scene's full pair tensors before constructing next.
                    del features
                    state['newly_constructed_feature_scenes'] += 1
                    mode = 'new130_same_formulas_pair4096'
                provenance.append(dict(scene_id=sid, mode=mode))
                rows.append(row); all_evidence.append(evidence)
                checkpoint(completed_scenes=len(rows), last_scene_id=sid)
        components = sorted({row['component_id'] for row in rows})
        def subset(indices):
            return p.aggregate([rows[i] for i in indices], [all_evidence[i] for i in indices]) if indices else None
        largest = max(components, key=lambda cid: sum(row['component_id'] == cid for row in rows))
        result = dict(status='COMPLETED', idea_id=IDEA, plan_version='1.0', **p.aggregate(rows, all_evidence),
            components={cid: subset([i for i, row in enumerate(rows) if row['component_id'] == cid]) for cid in components},
            without_largest_component=subset([i for i, row in enumerate(rows) if row['component_id'] != largest]),
            imported_complete128=subset(list(range(REUSE_COUNT))),
            newly_computed130=subset(list(range(REUSE_COUNT, len(rows)))), scene_provenance=provenance,
            imported_feature_scenes=state['imported_feature_scenes'],
            newly_constructed_feature_scenes=state['newly_constructed_feature_scenes'],
            pair_geometry_block_size=PAIR_BLOCK, original128_restarted=False, imported_features_recomputed=False,
            no_expansion=True, model_forwards=0, optimizer_updates=0,
            limitations=['original128 failed by systemd-oomd; preserved partial status is stale, not completion',
                'six imported scenes are reused evidence, not an independent second replication',
                'bounded pair transient batches do not guarantee immunity to host memory pressure',
                'candidate AUROC/coverage is not new partition F1 or physical fusion accuracy',
                'unchanged128 L1 circular-mean fallback/formulas/tolerances; no matched learned heads'])
        old.frozen.write(output/'scene-summary.json.gz', rows)
        old.frozen.write(output/'scene-provenance.json', provenance)
        checkpoint(phase='finalizing')
        result.update(elapsed_seconds=state['elapsed_seconds'], peak_rss_bytes=state['peak_rss_bytes'])
        old.frozen.write(output/'summary.json', result)
        checkpoint(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
        return result
    except BaseException as error:
        checkpoint(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            exit_code=1, error=f'{type(error).__name__}: {error}', finished_at=datetime.now(timezone.utc).isoformat())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-system', 'partition-run', 'failed-run', 'manifest', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--failure-evidence', type=Path)
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
