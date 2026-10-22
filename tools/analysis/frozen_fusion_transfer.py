"""062: frozen057 fusion on original056 groups, with an observable support control.

CPU inference only. Native evidence is rebuilt once from V measurements because
056 did not cache all27 fusion features. Association scores/groups are never
recomputed. Reference is accepted only by the evaluation path after prediction.
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
from association.data.motion_evidence import scene_motion_evidence
from association.evaluation.evidence_fusion_combo import predict_partition, REPLAY_TOLERANCE
from association.evaluation.frozen_fusion_transfer import summarize_scene, aggregate
from association.evaluation.support_control import support_switch_prediction
from association.evaluation.system_attribution import _same, _metrics, assert_metrics, pool_metrics
from association.utils.runtime import write_json
from tools.analysis.compare_association_duration import ARMS, _load_arrays, _replay
from tools.analysis.fusion_transfer_sources import bind_sources, replay_fusion_source
from tools.analysis.system_attribution import read, write_gzip
from tools.experiments.level1_system_evaluate import DATA, POLICY, evaluate_unique_population

LEARNED = ('helper_quality', 'direct_only', 'mixed_action')
# From061's mutually exclusive visible-support strata, before062 execution.
# Accumulation may round; cached ordinary arrays and population remain exact.
EXPECTED_SUPPORT_POSITION = dict(intersection_count=442263, mae=729.9808435828807,
                                 rmse=3332.536401612317)
METRIC_TOLERANCE = dict(rtol=1e-10, atol=1e-9)


def groups_from_cache(fragment, prefix):
    offsets = fragment[prefix+'_group_offsets']
    members = fragment[prefix+'_group_node_indices']
    return [members[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]


def frozen_motion(scene, fragment, source_artifact):
    """Rebuild only missing visible features, then check all cached evidence."""
    if scene['split'] != 'V_select':
        raise ValueError('062 may prepare V_select only')
    for key in ('node_ids', 'sources', 'identities', 'target_identity_ids'):
        _same(scene[key], fragment[key], '062 native input/'+key)
    motion = scene_motion_evidence(scene, motion_observation_policy=POLICY)
    for key, old in (('states', 'native_states'), ('query_seconds', 'query_seconds'),
                     ('support', 'support'), ('nearest_seconds', 'nearest_seconds'),
                     ('exact_observation', 'exact_observation')):
        _same(motion[key], fragment[old], '062 native evidence/'+key)
    truth, valid = source_artifact['aligned_truth'], source_artifact['aligned_truth_valid']
    targets = fragment['target_identity_ids']
    if set(scene['fusion_targets']) != set(targets):
        raise ValueError('062 complete reference registry changed')
    for i, identity in enumerate(targets):
        reference = scene['fusion_targets'][str(identity)]
        _same(reference['states'], truth[i], '062 reference/'+str(identity))
        _same(reference['valid'], valid[i], '062 reference mask/'+str(identity))
    return dict(scene, motion=motion, target=fragment['target'], valid=fragment['target_valid'])


def verify_support_control(metrics):
    for key, expected in EXPECTED_SUPPORT_POSITION.items():
        actual = metrics[key]
        equal = (actual == expected if key == 'intersection_count' else
                 np.isclose(actual, expected, **METRIC_TOLERANCE))
        if not equal:
            raise ValueError('062 support control differs from frozen061 analytical '+key)


def _check_output(output, sources):
    for source in sources:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('062 output must be separate from immutable056/057 runs')


def run(args):
    source = Path(args.source_system).resolve()
    fusion_source = Path(args.source_fusion).resolve()
    output = Path(args.output).resolve()
    _check_output(output, (source.parent, fusion_source))
    if type(args.cpu_threads) is not int or args.cpu_threads < 1:
        raise ValueError('positive CPU thread count required')
    torch.set_num_threads(args.cpu_threads)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', scope='062-frozen057-on056-fusion-transfer',
        dataset_level='Level1', split='V_select', source_system=str(source),
        source_fusion=str(fusion_source), optimizer_updates=0, gpu_used=False,
        device='cpu', cpu_threads=args.cpu_threads, test_opened=False,
        scores_recomputed=False, groups_recomputed=False, threshold_reselected=False,
        normalizer_refitted=False, native_evidence_reconstructed=True,
        stage='source_binding', completed_scenes=0)
    write_json(output/'run.json', state)
    try:
        manifest = read(args.manifest)
        metadata, models, fusion_metadata = bind_sources(source, fusion_source, manifest, device='cpu')
        if set(models) != set(LEARNED):
            raise ValueError('062 requires all three frozen057 objectives')
        ids = metadata['scene_ids']
        state.update(total_scenes=len(ids), stage='validation_preparation')
        write_json(output/'run.json', state)
        loaded = load_timed_dataset(args.archive, manifest, train_splits=['T_core'],
            data_config=DATA, alignment_by_scene=read(args.alignment),
            motion_observation_policy=POLICY, validation_only=True)
        preparation = loaded['report']
        if (preparation.get('test_opened') is not False
                or preparation.get('training_scene_ids') != []
                or preparation.get('validation_scene_ids') != ids
                or [str(scene['scene_id']) for scene in loaded['validation']] != ids):
            raise ValueError('062 must read exactly V_select and no training/test payloads')
        write_json(output/'preparation.json', preparation)
        for directory in ('scenes', 'predictions', 'system-predictions'):
            (output/directory).mkdir()
        rows, point_arrays = [], []
        fragment_count = 0
        state.update(stage='frozen_inference_and_cache_replay')
        for original in loaded['validation']:
            sid = str(original['scene_id'])
            fragment, saved = _load_arrays(source, sid)
            replay = _replay(fragment, saved, metadata['expected'][sid])
            base = replay[1]['learned_assoc_anchor']
            scene = frozen_motion(original, fragment, base)
            if str(original['component_id']) != str(metadata['component_ids'][sid]):
                raise ValueError('062 shared-identity component registration changed')
            groups = groups_from_cache(fragment, 'learned_assoc')
            ols_groups = groups_from_cache(fragment, 'ols')
            predictions = {arm: fragment['learned_assoc_'+arm] for arm in ARMS}
            predictions['support_switch'], switch = support_switch_prediction(
                predictions['anchor'], predictions['mean_supported'], fragment['support'], groups)
            routes, replays = {}, {}
            for arm in LEARNED:
                model = models[arm]
                model.eval()
                old_fragment, old_system = _load_arrays(Path(fusion_metadata[arm]['system']), sid)
                ordinary_inference, ordinary_route = predict_partition(model, scene['motion'], ols_groups, 'cpu')
                replays[arm] = replay_fusion_source(fragment, saved, old_fragment, old_system,
                    ordinary_inference, ordinary_route, expected=fusion_metadata[arm]['expected'][sid])
                inferred, routes[arm] = predict_partition(model, scene['motion'], groups, 'cpu')
                _same(inferred['mean_supported'], predictions['mean_supported'], '062 learned-group Mean')
                predictions[arm] = inferred['learned']
            reports, artifacts = evaluate_unique_population(scene, dict(learned_assoc=groups),
                                                             dict(learned_assoc=predictions))
            artifacts = {name.removeprefix('learned_assoc_'): value for name, value in artifacts.items()}
            # Every ordinary array, including duplicate/unretained group rows,
            # must match056. No learned inference tolerance applies here.
            for arm in ARMS:
                old = replay[1]['learned_assoc_'+arm]
                if set(artifacts[arm]) != set(old):
                    raise ValueError('062 ordinary system artifact fields changed')
                for key in old:
                    _same(artifacts[arm][key], old[key], '062 ordinary '+arm+'/'+key)
                assert_metrics(reports['learned_assoc_'+arm]['full_target_metrics'],
                    metadata['expected'][sid]['learned_assoc_'+arm]['full_target_metrics'], '062 '+arm)
            analysis_fragment = dict(fragment, direct_only_prediction=predictions['direct_only'])
            row, arrays = summarize_scene(analysis_fragment, artifacts, routes, sid, original['component_id'])
            fragment_regions = dict(all=np.ones(fragment['support'].shape, bool),
                inside_support=fragment['support'], outside_support=~fragment['support'],
                far_gt_1800s=fragment['nearest_seconds'] > 1800.)
            row['fragment_metrics'] = {arm: _metrics(value, fragment['target'], fragment['target_valid'],
                np.isfinite(value), fragment_regions) for arm, value in predictions.items()}
            row['source057_replay'] = replays
            write_gzip(output/'scenes'/f'scene-{sid}.json.gz', row)
            payload = dict(fragment, **{'output_'+key: value for key, value in predictions.items()},
                           support_switch_position_mask=switch)
            for arm, routing in routes.items():
                payload.update({arm+'_routing_'+key: value for key, value in routing.items()})
            np.savez_compressed(output/'predictions'/f'states-{sid}.npz', **payload)
            np.savez_compressed(output/'system-predictions'/f'states-{sid}.npz',
                **{arm+'_'+key: value for arm, artifact in artifacts.items() for key, value in artifact.items()})
            rows.append(row)
            point_arrays.append(arrays)
            fragment_count += len(fragment['node_ids'])
            state.update(completed_scenes=len(rows), last_scene_id=sid, elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        if fragment_count != metadata['fragment_count']:
            raise ValueError('062 fragment population changed')
        result = aggregate(rows, point_arrays)
        result['fragment_metrics'] = {arm: pool_metrics([row['fragment_metrics'][arm] for row in rows])
                                      for arm in predictions}
        for arm in ARMS:
            assert_metrics(result['methods'][arm]['full_target_metrics'],
                           metadata['pooled']['learned_assoc_'+arm], '062 pooled '+arm)
        verify_support_control(result['methods']['support_switch']['full_target_metrics']['all']['position'])
        result.update(status='COMPLETED', scope=state['scope'], dataset_level='Level1', split='V_select',
            scene_ids=ids, fragment_count=fragment_count, source_system=str(source), source_fusion=str(fusion_source),
            fusion_sources=fusion_metadata, association_endpoint='final', association_endpoint_step=1000,
            association_threshold=metadata['association_threshold'], ols_working_point=metadata['ols_working_point'],
            ordinary_source_replay_exact=True, fusion_source_replay_passed=True,
            fusion_replay_tolerance=REPLAY_TOLERANCE, support_control_analytical_verified=True,
            support_control_analytical_expected=EXPECTED_SUPPORT_POSITION,
            metric_tolerance=METRIC_TOLERANCE, optimizer_updates=0, gpu_used=False, test_opened=False,
            scores_recomputed=False, groups_recomputed=False, threshold_reselected=False,
            normalizer_refitted=False, native_evidence_reconstructed=True,
            definitions=dict(support_control='all predicted groups before reference alignment; position Mean only when own support absent and helper support present; speed/heading Mean',
                routing='inference terminal weights; separately actual first-anchor output groups and unique-target retained groups',
                quantiles='linear quantiles over eligible unique-target points; absolute errors and conditional positive error changes have separate denominators',
                numerical_change='1e-9 in attribute units is numerical bookkeeping, not engineering significance',
                source057='same frozen models replayed on original OLS groups before learned-group transfer'),
            limitations=['development V was reused to design the support control; no independent generalization claim',
                'full-fragment/offline evidence can use measurements after the query time',
                'shared-identity components are descriptive dependence groups, not independent ship validation',
                'no Oracle choice enters inference; this is a frozen transfer test, not new training',
                'negative or attribute-specific results do not justify retuning, retries or extra updates'])
        write_gzip(output/'scene-summary.json.gz', rows)
        write_json(output/'summary.json', result)
        state.update(status='COMPLETED', stage='finished', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-system', 'source-fusion', 'manifest', 'alignment', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--archive', nargs='+', required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    run(args)
    print(f'COMPLETED: {args.output}/summary.json')


if __name__ == '__main__':
    main()
