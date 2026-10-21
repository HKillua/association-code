"""Frozen Level1 V_select: same-input OLS/learned grouping x ordinary/learned fusion.

Read-only evaluation; never fits a model or reads T_test. This is a project
protocol, not a claim of exact TMTAFF author-protocol reproduction.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.data.dataset import load_timed_dataset
from association.evaluation.association import evaluate_scenes
from association.evaluation.evidence_fusion_combo import read_scores, predict_partition
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.physical import evaluate_aligned_fusion
from association.evaluation.fusion_controls import ordinary_fusion, metrics
from association.evaluation.grouping import group_predicted_tracks
from association.evaluation.tmtaff import aggregate_fusion_reports
from association.training.evidence_fusion import (
    bind_ols_working_point, load_checkpoint, prepare_evidence_scenes,
)
from association.utils.runtime import read_json, write_json

DATA = dict(length=64, max_observations=128, time_unit_seconds=20.,
            reference_longitude_policy='shortest_arc')
POLICY = 'canonical_all'


def exact_score_population(scenes, scores):
    """Reject a partial or reordered score dump before making any prediction."""
    if set(scores) != {str(scene['scene_id']) for scene in scenes}:
        raise ValueError('association scores must cover every V scene, with no extras')
    for scene in scenes:
        row = scores[str(scene['scene_id'])]
        for name in ('node_ids', 'sources', 'pairs'):
            if not np.array_equal(row[name], np.asarray(scene[name])):
                raise ValueError(f'{scene["scene_id"]}: association {name} changed')
        if not np.isfinite(row['scores']).all():
            raise ValueError('association scores contain nonfinite values')


def _validate_assoc(summary, audit, manifest, scores, scenes):
    if (summary.get('status') != 'COMPLETED'
            or summary.get('scope') != 'native-ols-evidence-association-v1'
            or summary.get('motion_observation_policy') != POLICY
            or audit.get('motion_observation_policy') != POLICY
            or audit.get('split_unit') != manifest.get('split_unit')
            or audit.get('training_scene_ids') != manifest['splits']['T_core']
            or audit.get('validation_scene_ids') != manifest['splits']['V_select']):
        raise ValueError('association selected run is not the current complete Level1 population')
    exact_score_population(scenes, scores)
    selected_step = summary['selected']['update']
    if (not isinstance(selected_step, int) or isinstance(selected_step, bool)
            or selected_step < 0 or selected_step > summary.get('optimizer_updates', -1)):
        raise ValueError('association selected update is inconsistent with its completed run')
    threshold = summary['selected']['selection']['threshold']
    return threshold


def verify_fusion_snapshot(fusion, checkpoint, run, manifest, grouping):
    """A completed job's actual final weights must match its final run record.

    Refuse e.g. a step-0 snapshot accompanied by a COMPLETED step-1000 run.
    A checkpoint passed here is explicitly *final*, not a selected intermediate.
    """
    cfg = checkpoint.get('configuration')
    steps = cfg.get('training', {}).get('updates') if isinstance(cfg, dict) else None
    if (not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
            or checkpoint.get('step') != steps
            or run.get('optimizer_updates') != steps
            or run.get('status') != 'COMPLETED'
            or run.get('configuration') != cfg
            or run.get('final_evaluation') != f'evaluation-step-{steps}.json'
            or run.get('training_scene_ids') != manifest['splits']['T_core']
            or run.get('validation_scene_ids') != manifest['splits']['V_select']
            or fusion.motion_observation_policy != POLICY
            or fusion.ols_grouping != grouping
            or cfg.get('motion_observation_policy') != POLICY
            or cfg.get('grouping') != grouping):
        raise ValueError('fusion weights/step/config/COMPLETED run do not describe the same final Level1 job')
    return steps


def evaluate_unique_population(scene, partitions, by_group):
    """Keep all original target identities, including those with zero input fragments.

    Truth/identity is supplied only to evaluate_full_scene AFTER all partitions
    and physical predictions have been frozen. It never repairs the predicted
    group or determines the representative.
    """
    registry = [str(x) for x in scene['target_identity_ids']]
    truth_data = scene['fusion_targets']
    if len(registry) != len(set(registry)) or set(registry) != set(truth_data):
        raise ValueError('complete original-label identity registry required')
    identities = np.asarray(scene['identities'], str)
    if not set(identities).issubset(registry):
        raise ValueError('valid fragments contain unknown target identities')
    q = len(scene['motion']['query_seconds'])
    target = np.stack([np.asarray(truth_data[i]['states'], np.float64) for i in registry])
    valid = np.stack([np.asarray(truth_data[i]['valid'], bool) for i in registry])
    if target.shape != (len(registry), q, 4) or valid.shape != target.shape:
        raise ValueError('full target rows/masks must be aligned [T,Q,4]')
    # Fragment-level targets are redundant, but verify that they agree with
    # original identity targets before scoring or writing any result.
    for node, identity in enumerate(identities):
        reference = truth_data[str(identity)]
        if (not np.array_equal(scene['target'][node], reference['states'], equal_nan=True)
                or not np.array_equal(scene['valid'][node], reference['valid'])):
            raise ValueError('fragment and complete-identity target arrays disagree')
    support = np.asarray(scene['motion']['support'])
    if support.shape != (len(identities), q) or support.dtype != bool:
        raise ValueError('full native support mask required')
    reports, artifacts = {}, {}
    for group_name, groups in partitions.items():
        for arm, predictions in by_group[group_name].items():
            name = group_name+'_'+arm
            values = np.asarray(predictions)
            if values.shape != (len(identities), q, 4):
                raise ValueError('complete per-fragment predictions required')
            # The first node in every predicted group is chosen without reference.
            generated = [dict(node_indices=members, states=values[members[0]]) for members in groups]
            item = evaluate_full_scene(generated, node_identities=identities,
                target_identities=registry, target_states=target, target_valid=valid,
                node_observed=support, split='V_select', grouping_kind='predicted',
                correspondence_policy='anchor_identity_first', return_artifact=True,
                node_input_support=dict(support=support,
                    nearest_seconds=scene['motion']['nearest_seconds']))
            artifacts[name] = item.pop('artifact')
            reports[name] = item
    return reports, artifacts


def common_target_metrics(artifacts):
    """Paired errors on the intersection of valid predictions for ALL arms.

    Preserve target rows and missing-prediction counts; aggregate via the same
    evaluate_aligned_fusion core as the project's existing 048 comparison.
    Duplicate/unmatched group rows have no assigned target and are not used to
    create extra truth points in the common intersection.
    """
    if not artifacts:
        raise ValueError('nonempty method artifacts required')
    names = sorted(artifacts)
    base = artifacts[names[0]]
    count = len(base['target_ids'])
    truth = base['aligned_truth'][:count]
    target_valid = base['aligned_truth_valid'][:count]
    observed = base['aligned_observed'][:count]
    common = np.ones_like(target_valid, bool)
    for name in names:
        artifact = artifacts[name]
        if (not np.array_equal(artifact['target_ids'], base['target_ids'])
                or not np.array_equal(artifact['aligned_truth'][:count], truth, equal_nan=True)
                or not np.array_equal(artifact['aligned_truth_valid'][:count], target_valid)
                or not np.array_equal(artifact['aligned_observed'][:count], observed)):
            raise ValueError('common-target comparison uses different truth, masks or target rows')
        common &= artifact['aligned_prediction_valid'][:count]
    regions = dict(all=np.ones_like(observed, bool), observed=observed, missing=~observed)
    reports = {}
    for name in names:
        reports[name] = dict(full_target_metrics={region: evaluate_aligned_fusion(
            artifacts[name]['aligned_prediction'][:count], truth, target_valid,
            prediction_valid=common, region_mask=mask) for region, mask in regions.items()})
    return reports, common


def _fragment_regions(motion):
    return dict(all=np.ones_like(motion['support'], bool),
                inside_support=motion['support'],
                outside_support=~motion['support'],
                far_gt_1800s=motion['nearest_seconds'] > 1800.)


def _pool_fragment(per_scene, methods):
    """Weighted MAE/MSE with explicit intersection counts, never mean RMSEs."""
    pooled = {}
    for name in methods:
        pooled[name] = {}
        for region in ('all', 'inside_support', 'outside_support', 'far_gt_1800s'):
            pooled[name][region] = {}
            for attr in ('position', 'speed', 'heading'):
                records = [scene['fragment'][name][region][attr] for scene in per_scene]
                n = sum(record['intersection_count'] for record in records)
                total = sum(record['available_target_count'] for record in records)
                mse = (sum(record['mse'] * record['intersection_count'] for record in records
                           if record['intersection_count']) / n) if n else None
                pooled[name][region][attr] = dict(
                    intersection_count=n, available_target_count=total,
                    coverage=n/total if total else None,
                    mae=(sum(record['mae'] * record['intersection_count'] for record in records
                             if record['intersection_count']) / n) if n else None,
                    rmse=float(np.sqrt(mse)) if mse is not None else None,
                    scene_macro_mae=(float(np.mean([record['mae'] for record in records
                                           if record['mae'] is not None]))
                                     if any(record['mae'] is not None for record in records) else None),
                    unit=records[0]['unit'])
    return pooled


def _groups(scene, score_row, selected_threshold):
    partitions = {'ols': scene['groups']['predicted_anchor_direct']}
    if score_row is not None:
        partitions['learned_assoc'] = group_predicted_tracks(
            len(scene['node_ids']), score_row['pairs'], score_row['scores'],
            threshold=selected_threshold, grouping_policy='anchor_direct', node_order='input')
    return partitions


def run(args):
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise ValueError('requested GPU unavailable; no silent CPU fallback')
    manifest = read_json(args.manifest)
    if manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene':
        raise ValueError('this entry point accepts the declared Level1 scene-level protocol only')
    wp = read_json(args.ols_working_point)
    # Bind the same pure-OLS workpoint as Level1 fusion training.
    grouping = bind_ols_working_point(
        dict(data=DATA, motion_observation_policy=POLICY), manifest, wp)['grouping']
    if (args.association_scores is None) != (args.association_summary is None):
        raise ValueError('association scores and summary must be provided together')
    if args.association_scores and not args.association_data_audit:
        raise ValueError('selected association needs its data_audit for split/policy checks')
    if bool(args.fusion_checkpoint) != bool(args.fusion_run):
        raise ValueError('fusion checkpoint needs run.json to verify its full scene population')
    if args.cpu_threads < 1:
        raise ValueError('cpu threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', dataset_level='Level1', split='V_select',
                 optimizer_updates=0, test_opened=False, threshold_reselected=False,
                 normalizer_refitted=False, motion_observation_policy=POLICY,
                 stage='validation_preparation', completed_scenes=0,
                 total_scenes=len(manifest['splits']['V_select']),
                 inputs={key: getattr(args, key) for key in (
                     'archive', 'manifest', 'alignment', 'ols_working_point',
                     'association_scores', 'association_summary', 'association_data_audit',
                     'fusion_checkpoint', 'fusion_run')})
    write_json(output/'run.json', state)
    try:
        # Reuse the registered loader without opening training or test payloads.
        loaded = load_timed_dataset(args.archive, manifest, train_splits=['T_core'],
                                    data_config=DATA, alignment_by_scene=read_json(args.alignment),
                                    motion_observation_policy=POLICY, validation_only=True)
        write_json(output/'preparation.json', loaded['report'])
        originals = loaded['validation']
        ids = [str(s['scene_id']) for s in originals]
        if ids != manifest['splits']['V_select']:
            raise ValueError('validation scene order differs from Level1 manifest')
        prepared = prepare_evidence_scenes(originals, motion_observation_policy=POLICY,
                                           grouping=grouping)
        scenes = [{**original, **item} for original, item in zip(originals, prepared)]
        scores = read_scores(args.association_scores) if args.association_scores else None
        selected_threshold = None
        summary = None
        if scores is not None:
            summary = read_json(args.association_summary)
            selected_threshold = _validate_assoc(summary, read_json(args.association_data_audit),
                                                manifest, scores, scenes)
        fusion = None
        loaded_fusion_step = None
        if args.fusion_checkpoint:
            fusion, checkpoint = load_checkpoint(args.fusion_checkpoint, device=args.device)
            run = read_json(args.fusion_run)
            loaded_fusion_step = verify_fusion_snapshot(fusion, checkpoint, run, manifest, grouping)
            fusion.eval()
        per_scene, system_rows, common_rows = [], [], []
        prediction_dir = output/'predictions'
        prediction_dir.mkdir()
        system_prediction_dir = output/'system-predictions'
        system_prediction_dir.mkdir()
        expected_methods = None
        score_records = []
        state.update(stage='system_evaluation', preparation_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        for scene in scenes:
            sid = str(scene['scene_id'])
            motion, target, valid = scene['motion'], scene['target'], scene['valid']
            row = scores[sid] if scores is not None else None
            partitions = _groups(scene, row, selected_threshold)
            by_group, flat, routing_by_group, memberships = {}, {}, {}, {}
            for group_name, groups in partitions.items():
                controls = ordinary_fusion(motion['states'], motion['nearest_seconds'],
                                           motion['support'], groups)
                memberships[group_name] = controls['membership']
                by_group[group_name] = {'mean_supported': controls['predictions']['mean_supported']}
                if fusion is not None:
                    learned, routing = predict_partition(fusion, motion, groups, args.device)
                    routing_by_group[group_name] = routing
                    if not np.array_equal(learned['mean_supported'], by_group[group_name]['mean_supported']):
                        raise ValueError(f'{sid}: learned run changed the ordinary control')
                    by_group[group_name]['learned'] = learned['learned']
                flat.update({group_name+'_'+k: v for k, v in by_group[group_name].items()})
            methods = sorted(flat)
            if expected_methods is None:
                expected_methods = methods
            elif methods != expected_methods:
                raise ValueError('method set differs across V scenes')
            regions = _fragment_regions(motion)
            fragment = {name: {region: metrics(values, target, valid, mask)
                               for region, mask in regions.items()} for name, values in flat.items()}
            unique, artifacts = evaluate_unique_population(scene, partitions, by_group)
            common, common_mask = common_target_metrics(artifacts)
            system_rows.append(unique)
            common_rows.append(common)
            per_scene.append(dict(scene_id=sid, component_id=scene['component_id'],
                                  fragment=fragment,
                                  group_count={key: len(groups) for key, groups in partitions.items()},
                                  unique_target=unique,
                                  common_valid_target_points={name: row['full_target_metrics'] for name, row in common.items()}))
            payload = dict(node_ids=np.asarray(scene['node_ids']),
                identities=np.asarray(scene['identities']), sources=np.asarray(scene['sources']),
                target_identity_ids=np.asarray(scene['target_identity_ids']),
                query_seconds=motion['query_seconds'], target=target, target_valid=valid,
                native_states=motion['states'], nearest_seconds=motion['nearest_seconds'],
                support=motion['support'], exact_observation=motion['exact_observation'],
                **flat)
            for group_name, groups in partitions.items():
                payload[group_name+'_membership'] = memberships[group_name]
                payload[group_name+'_group_offsets'] = np.r_[0, np.cumsum([len(g) for g in groups])]
                payload[group_name+'_group_node_indices'] = np.asarray(
                    [i for group in groups for i in group], dtype=np.int64)
                for key, value in routing_by_group.get(group_name, {}).items():
                    payload[group_name+'_routing_'+key] = value
            for region, mask in regions.items():
                payload['region_'+region] = mask
            np.savez_compressed(prediction_dir/f'states-{sid}.npz', **payload)
            # Exact aligned arrays used for all system metrics, not reconstructed
            # from a lossy summary. Different method group counts are preserved.
            system_payload = {'common_prediction_valid_on_target_rows': common_mask}
            for name, artifact in artifacts.items():
                system_payload.update({name+'_'+key: value for key, value in artifact.items()})
            np.savez_compressed(system_prediction_dir/f'states-{sid}.npz', **system_payload)
            if row is not None:
                score_records.append(dict(scene_id=sid, component_id=scene['component_id'],
                    scores=row['scores'], pairs=row['pairs'], labels=scene['labels'], sources=scene['sources']))
            state.update(completed_scenes=len(per_scene), last_scene_id=sid,
                         elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        association = None
        if scores is not None:
            # Labels are read for validation only after all predicted partitions exist.
            association = evaluate_scenes(score_records, selected_threshold)
            expected = summary['selected_result']['metrics']
            if (any(association[key] != expected[key] for key in ('tp', 'fp', 'fn', 'tn'))
                    or not np.isclose(association['macro_f1'], expected['macro_f1'], atol=1e-12, rtol=0)):
                raise ValueError('selected association score dump does not reproduce summary')
        reports = dict(status='COMPLETED', dataset_level='Level1', split='V_select',
            scene_ids=ids, scene_count=len(scenes), fragment_count=sum(len(s['node_ids']) for s in scenes),
            method_names=expected_methods, ols_working_point=wp,
            association=association,
            association_selected_step=(summary['selected']['update'] if summary is not None else None),
            association_evidence='frozen selected-scores; no association model weights loaded',
            fragment=_pool_fragment(per_scene, expected_methods),
            unique_target={name: aggregate_fusion_reports([row[name] for row in system_rows])
                           for name in expected_methods},
            common_valid_target_points={name: aggregate_fusion_reports([row[name] for row in common_rows])
                                        for name in expected_methods},
            fusion_checkpoint_step=loaded_fusion_step,
            inputs=state['inputs'],
            fusion_snapshot_semantics=('final completed run; step matches final checkpoint and run'
                                       if fusion is not None else 'no learned fusion weights loaded'),
            per_scene=per_scene,
            limitations=['V is repeatedly exposed development data, not sealed T_test',
                         'one-group-one-output follows first-input-anchor correspondence; no truth-based selection',
                         'only V_select payloads read; no refit or training performed',
                         'all original-label targets retained, including identities with no input fragments',
                         'ordinary mean_supported falls back to anchor OLS if all members are unsupported'],
            test_opened=False, optimizer_updates=0)
        write_json(output/'summary.json', reports)
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started,
                     stage='finished', method_names=expected_methods, fusion_checkpoint_step=loaded_fusion_step)
        write_json(output/'run.json', state)
        return reports
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                     error=f'{type(error).__name__}: {error}',
                     elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', nargs='+', required=True)
    for key in ('manifest', 'alignment', 'ols-working-point', 'output'):
        parser.add_argument('--'+key, required=True)
    for key in ('association-scores', 'association-summary', 'association-data-audit',
                'fusion-checkpoint', 'fusion-run'):
        parser.add_argument('--'+key)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda:0'))
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    result = run(args)
    print('COMPLETED: '+str(Path(args.output).resolve()), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
