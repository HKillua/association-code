"""071 temporal position baselines on every frozen056 output representative.

Truth is used only after all groups, including duplicate/unreferenced outputs,
have predictions. The unrestricted two-dimensional residual is never treated
as a support gate; support regions are descriptive evaluation strata only.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import torch

from association.evaluation.frozen_fusion_transfer import _changes, _population
from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.learned_group_oracle import _helper_evidence, _visible_regions, _quantiles
from association.evaluation.support_control import support_switch_prediction
from association.evaluation.system_attribution import ATTRIBUTES, _same, assert_metrics, pool_metrics
from association.evaluation.paper_fusion_metrics import aggregate_paper_fusion
from association.models.sequence_position import sequence_inputs, tangent_position
from association.models.support_gate import mix_gate_states
from association.utils.runtime import read_json, write_json, write_json_gzip
from tools.analysis.compare_association_duration import ARMS, _load_arrays, _replay
from tools.analysis.frozen_fusion_transfer import frozen_motion, groups_from_cache
from tools.experiments.level1_system_evaluate import evaluate_unique_population

METHODS = (*ARMS, 'support_switch', 'fixed_prior095', 'learned_sequence')
REGIONS = ('all', 'anchor_inside_support', 'anchor_outside_helper_supported',
           'predicted_group_no_support', 'no_predicted_group')
WEIGHTINGS = ('point_equal', 'scene_target_equal', 'scene_equal_point_mean', 'scene_equal_target_mean')


def _tensor(value, shape, name, *, physical=False):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all()
            or (physical and value.dtype != torch.float64)):
        raise ValueError('071 expected finite CPU tensor with fixed shape/dtype: '+name)
    return value.detach().numpy()


def _prior(anchor, mean, switch):
    """Exactly the float32 .95 sigmoid initialization used by the fixed control."""
    result = mean.copy()
    q = anchor.shape[1]
    fraction = torch.sigmoid(torch.full((q,), math.log(19.), dtype=torch.float32))
    for node in range(len(anchor)):
        result[node] = mix_gate_states(torch.from_numpy(anchor[node]), torch.from_numpy(mean[node]),
                                      fraction, torch.from_numpy(switch[node])).numpy()
    return result


def _predict(model, motion, groups, predictions, check, *, initial=False,
             input_builder=sequence_inputs, observer=None):
    q = motion['states'].shape[1]
    output = predictions['fixed_prior095'].copy()
    residuals = []
    with torch.inference_mode():
        for group in groups:
            if check:
                check()
            args = input_builder(motion, group, device='cpu')
            result = model(*args)
            states = _tensor(result['states'], (q, 4), 'states', physical=True)
            base = _tensor(result['base_states'], (q, 4), 'baseline_states', physical=True)
            residual = _tensor(result['residual_m'], (q, 2), 'tangent_residual_m')
            _same(base, predictions['fixed_prior095'][group[0]], '071 independently constructed fixed .95 base')
            mean = _tensor(result['mean_states'], (q, 4), 'mean_states', physical=True)
            _same(mean, predictions['mean_supported'][group[0]], '071 complete ordinary Mean candidate')
            _same(states[:, 2:], predictions['mean_supported'][group[0], :, 2:], '071 fixed Mean speed/heading')
            _same(states, tangent_position(result['base_states'], result['residual_m']).numpy(),
                  '071 saved tangent residual/physical state relation')
            if initial:
                _same(residual, np.zeros_like(residual), '071 zero-head initialization')
                _same(states, base, '071 initial output exactly fixed .95')
            output[group[0]] = states
            residuals.append(residual)
            if observer is not None:
                observer(group, args, result)
    return output, np.asarray(residuals)


def _metric(values):
    values = np.asarray(values, np.float64).reshape(-1, 2)
    if not len(values):
        return dict(unit_count=0, mae=None, mse=None, rmse=None)
    mean = values.mean(0)
    return dict(unit_count=len(values), mae=float(mean[0]), mse=float(mean[1]), rmse=float(np.sqrt(mean[1])))


def _weighted(rows, arrays, methods=METHODS):
    """Four explicitly distinct populations; RMSE is sqrt(weighted MSE)."""
    result = {}
    for region in REGIONS:
        result[region] = {}
        for method in methods:
            points, targets, scenes, scene_targets = [], [], [], []
            for data in arrays:
                error = data[method+'_position_error']
                mask = data['region_'+region] & np.isfinite(error)
                values = error[mask]
                observed = np.stack((values, values*values), -1)
                points.append(observed)
                target_values = [np.array([error[i, use].mean(), np.square(error[i, use]).mean()])
                                 for i, use in enumerate(mask) if use.any()]
                targets.extend(target_values)
                if len(values):
                    scenes.append(observed.mean(0))
                    scene_targets.append(np.mean(target_values, axis=0))
            units = (np.concatenate(points), targets, scenes, scene_targets)
            result[region][method] = {key: _metric(value) for key, value in zip(WEIGHTINGS, units)}
            values = units[0][:, 0]
            result[region][method]['point_equal'].update(_quantiles(values))
    return result


def _scene(fragment, artifacts, reports, residual, sid, component):
    base = artifacts['anchor']
    for method, artifact in artifacts.items():
        if set(artifact) != set(base):
            raise ValueError('071 artifact schema changed')
        for key in base:
            if key not in ('aligned_prediction', 'group_states'):
                _same(artifact[key], base[key], '071 unchanged population/'+method+'/'+key)
    t = len(base['target_ids'])
    truth = base['aligned_truth'][:t]
    common = base['aligned_truth_valid'][:t] & base['aligned_prediction_valid'][:t]
    groups = groups_from_cache(fragment, 'learned_assoc')
    anchors = np.asarray([group[0] for group in groups], np.int64)
    selected = base['aligned_group_indices'][:t]
    _, helpers, _ = _helper_evidence(fragment, groups)
    visible = _visible_regions(fragment, anchors, selected, helpers)[0]
    regions = dict(all=np.ones(common.shape[:2], bool), **{key: visible[key] for key in REGIONS[1:]})
    arrays = {'region_'+key: value for key, value in regions.items()}
    methods = {}
    for method in METHODS:
        errors = attribute_errors(artifacts[method]['aligned_prediction'][:t], truth)
        for attr, columns in ATTRIBUTES.items():
            mask = common[..., list(columns)].all(-1)
            if not np.isfinite(errors[attr][mask]).all():
                raise ValueError('071 nonfinite valid '+attr+' error')
            arrays[method+'_'+attr+'_error'] = np.where(mask, errors[attr], np.nan)
        methods[method] = dict(population=_population(artifacts[method]),
            full_target_metrics=reports['learned_assoc_'+method]['full_target_metrics'],
            paper_fusion_metrics=reports['learned_assoc_'+method]['paper_fusion_metrics'])
    chosen = np.zeros(len(groups), bool)
    chosen[selected[selected >= 0]] = True
    routing = {}
    for name, use in (('all_output_groups', np.ones(len(groups), bool)), ('unique_selected_groups', chosen)):
        norm = np.linalg.norm(residual[use], axis=-1)
        routing[name] = dict(query_count=int(norm.size), nonzero_residual_query_count=int((norm > 0).sum()),
                             residual_norm_m=_quantiles(norm.ravel()))
    return dict(scene_id=sid, component_id=component, fragment_count=len(fragment['node_ids']),
                methods=methods, routing=routing), arrays


def _pool(rows, arrays):
    result = dict(scene_count=len(rows), scene_ids=[row['scene_id'] for row in rows], methods={},
                  position_weightings=_weighted(rows, arrays))
    for method in METHODS:
        source = [row['methods'][method] for row in rows]
        result['methods'][method] = dict(
            population={key: sum(item['population'][key] for item in source) for key in source[0]['population']},
            full_target_metrics=pool_metrics([item['full_target_metrics'] for item in source]),
            paper_fusion_metrics=aggregate_paper_fusion([item['paper_fusion_metrics'] for item in source]),
            attribute_quantiles={attr: {region: _quantiles(np.concatenate([
                data[method+'_'+attr+'_error'][data['region_'+region] & np.isfinite(data[method+'_'+attr+'_error'])]
                for data in arrays])) for region in REGIONS} for attr in ATTRIBUTES})
    result['pairwise'] = {}
    for control in METHODS[:-1]:
        result['pairwise']['learned_sequence__minus__'+control] = {}
        for region in REGIONS:
            differences = []
            for data in arrays:
                a, b = data['learned_sequence_position_error'], data[control+'_position_error']
                _same(np.isfinite(a), np.isfinite(b), '071 common target mask')
                mask = data['region_'+region] & np.isfinite(a)
                differences.append((a-b)[mask])
            result['pairwise']['learned_sequence__minus__'+control][region] = _changes(np.concatenate(differences))
    return result


def evaluate_sequence_position(model, validation_raw, source056, metadata, output, check=None, *, initial=False,
                               predictor=None, evaluation_scope='071-sequence-position-evaluation',
                               residual_key='group_tangent_residual_m'):
    source056, output = Path(source056).resolve(), Path(output).resolve()
    protected = source056.parent
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError('071 output must be separate from immutable056')
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('071 evaluation requires a CPU model')
    ids = [str(scene['scene_id']) for scene in validation_raw]
    if not ids or ids != metadata['scene_ids'] or len(set(ids)) != len(ids):
        raise ValueError('071 requires complete V population/order')
    if set(metadata['component_ids']) != set(ids):
        raise ValueError('071 requires exact component registry')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', scope=evaluation_scope, device='cpu',
        source_system=str(source056), scene_ids=ids, total_scenes=len(ids), completed_scenes=0,
        initial=initial, test_opened=False, groups_recomputed=False, threshold_reselected=False,
        model_forwards=0, optimizer_updates=0,
        inference_scope='first representative of every original output group, including duplicates')
    write_json(output/'run.json', state)
    try:
        for directory in ('scenes', 'predictions', 'system-predictions'):
            (output/directory).mkdir()
        model.eval()
        rows, arrays = [], []
        for original, sid in zip(validation_raw, ids):
            if check:
                check()
            fragment, saved = _load_arrays(source056, sid)
            replay = _replay(fragment, saved, metadata['expected'][sid])
            scene = frozen_motion(original, fragment, replay[1]['learned_assoc_anchor'])
            component = str(metadata['component_ids'][sid])
            if str(original['component_id']) != component:
                raise ValueError('071 scene component changed')
            groups = groups_from_cache(fragment, 'learned_assoc')
            predictions = {arm: fragment['learned_assoc_'+arm] for arm in ARMS}
            predictions['support_switch'], switch = support_switch_prediction(predictions['anchor'],
                predictions['mean_supported'], fragment['support'], groups)
            predictions['fixed_prior095'] = _prior(predictions['anchor'], predictions['mean_supported'], switch)
            predicted = (_predict if predictor is None else predictor)(model, scene['motion'], groups,
                predictions, check, initial=initial)
            predictions['learned_sequence'], residual = predicted[:2]
            extra_payload = {} if len(predicted) == 2 else predicted[2]
            reports, artifact = evaluate_unique_population(scene, {'learned_assoc': groups}, {'learned_assoc': predictions})
            artifact = {key.removeprefix('learned_assoc_'): value for key, value in artifact.items()}
            for arm in ARMS:
                old = replay[1]['learned_assoc_'+arm]
                if set(artifact[arm]) != set(old):
                    raise ValueError('071 ordinary artifact schema changed')
                for key in old:
                    _same(artifact[arm][key], old[key], '071 ordinary replay/'+arm+'/'+key)
                assert_metrics(reports['learned_assoc_'+arm]['full_target_metrics'],
                    metadata['expected'][sid]['learned_assoc_'+arm]['full_target_metrics'], '071 ordinary '+arm)
            row, data = _scene(fragment, artifact, reports, residual, sid, component)
            row['position_weightings'] = _weighted([row], [data])
            write_json_gzip(output/'scenes'/f'scene-{sid}.json.gz', row)
            payload = dict(
                group_anchor_node_indices=np.asarray([group[0] for group in groups], np.int64),
                **{key: fragment[key] for key in ('node_ids', 'identities', 'target_identity_ids',
                    'native_states', 'support', 'nearest_seconds', 'exact_observation', 'query_seconds',
                    'learned_assoc_group_offsets', 'learned_assoc_group_node_indices')},
                **{method+'_group_states': artifact[method]['group_states'] for method in METHODS}, **data)
            if not residual_key.startswith('group_') or residual_key in payload:
                raise ValueError('sequence residual cache key must be explicit and distinct from physical inputs')
            payload[residual_key] = residual
            if set(extra_payload) & set(payload):
                raise ValueError('sequence diagnostic payload cannot replace physical inputs or outputs')
            payload.update(extra_payload)
            np.savez_compressed(output/'predictions'/f'states-{sid}.npz', **payload)
            np.savez_compressed(output/'system-predictions'/f'states-{sid}.npz',
                **{method+'_'+key: value for method, values in artifact.items() for key, value in values.items()})
            rows.append(row); arrays.append(data)
            state.update(completed_scenes=len(rows), model_forwards=state['model_forwards']+len(groups),
                         elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        if sum(row['fragment_count'] for row in rows) != metadata['fragment_count']:
            raise ValueError('071 fragment population changed')
        result = _pool(rows, arrays)
        for arm in ARMS:
            assert_metrics(result['methods'][arm]['full_target_metrics'], metadata['pooled']['learned_assoc_'+arm], '071 pooled '+arm)
        components = {}
        for component in sorted(set(metadata['component_ids'].values())):
            chosen = [i for i, row in enumerate(rows) if row['component_id'] == component]
            components[component] = _pool([rows[i] for i in chosen], [arrays[i] for i in chosen])
        largest = max(components, key=lambda key: components[key]['position_weightings']['all']['anchor']['point_equal']['unit_count'])
        rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        result.update(status='COMPLETED', method_names=list(METHODS), source_system=str(source056), device='cpu',
            dataset_level='Level1', split='V_select', initial=initial, test_opened=False, optimizer_updates=0,
            ordinary_source_replay_exact=True, fixed_population_verified=True, component_count=len(components),
            fragment_count=metadata['fragment_count'], components=components,
            largest_component=dict(component_id=largest, selection='most valid fixed-population Anchor position points',
                                   metrics=components[largest]),
            remaining_components=_pool([rows[i] for i in rest], [arrays[i] for i in rest]) if rest else None,
            model_forwards=state['model_forwards'], groups_recomputed=False, threshold_reselected=False,
            inference_scope=state['inference_scope'],
            interpretation='Generic position-only temporal residual baseline; not an original published method reproduction',
            limitations=['Repeated development V and shared identities; single seed is not independent generalization.',
                'Speed/heading are frozen Mean outputs, not learned multivariate fusion.',
                'Training synthetic groups differ from frozen predicted deployment groups.'])
        write_json_gzip(output/'scene-summary.json.gz', rows)
        write_json(output/'summary.json', result)
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise


def compare_endpoints(initial_dir, final_dir):
    """Use complete fixed caches; do not rerun inference or select a V checkpoint."""
    initial_dir, final_dir = Path(initial_dir), Path(final_dir)
    before, after = [read_json(path/'summary.json') for path in (initial_dir, final_dir)]
    if (before.get('status') != 'COMPLETED' or after.get('status') != 'COMPLETED'
            or before['scene_ids'] != after['scene_ids'] or before['source_system'] != after['source_system']
            or before['method_names'] != list(METHODS) or after['method_names'] != list(METHODS)
            or before.get('initial') is not True or after.get('initial') is not False
            or before['device'] != 'cpu' or after['device'] != 'cpu'):
        raise ValueError('071 comparison requires completed paired CPU endpoints')
    components = {sid: key for key, row in before['components'].items() for sid in row['scene_ids']}
    if components != {sid: key for key, row in after['components'].items() for sid in row['scene_ids']}:
        raise ValueError('071 endpoint components differ')
    for method in METHODS:
        if before['methods'][method]['population'] != after['methods'][method]['population']:
            raise ValueError('071 endpoint population differs')
    rows, arrays = [], []
    for sid in before['scene_ids']:
        with np.load(initial_dir/'system-predictions'/f'states-{sid}.npz', allow_pickle=False) as old, \
                np.load(final_dir/'system-predictions'/f'states-{sid}.npz', allow_pickle=False) as new:
            if set(old.files) != set(new.files):
                raise ValueError('071 endpoint system schemas differ')
            for key in old.files:
                if key not in ('learned_sequence_aligned_prediction', 'learned_sequence_group_states'):
                    _same(old[key], new[key], '071 fixed endpoint system/'+key)
                else:
                    _same(old[key][..., 2:], new[key][..., 2:], '071 unchanged endpoint speed/heading')
        with np.load(initial_dir/'predictions'/f'states-{sid}.npz', allow_pickle=False) as a, \
                np.load(final_dir/'predictions'/f'states-{sid}.npz', allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError('071 endpoint cache schemas differ')
            for key in a.files:
                if not key.startswith('learned_sequence_') and key != 'group_tangent_residual_m':
                    _same(a[key], b[key], '071 fixed endpoint input/'+key)
            for attr in ATTRIBUTES:
                _same(np.isfinite(a['learned_sequence_'+attr+'_error']),
                      np.isfinite(b['learned_sequence_'+attr+'_error']), '071 common endpoint masks')
                if attr != 'position':
                    _same(a['learned_sequence_'+attr+'_error'], b['learned_sequence_'+attr+'_error'],
                          '071 unchanged endpoint non-position error')
            data = {region: a[region].copy() for region in a.files if region.startswith('region_')}
            data.update(initial_position_error=a['learned_sequence_position_error'].copy(),
                        final_position_error=b['learned_sequence_position_error'].copy())
            rows.append(dict(scene_id=sid, component_id=components[sid])); arrays.append(data)
    def pool(indices):
        chosen = [arrays[i] for i in indices]
        values = _weighted([rows[i] for i in indices], chosen, ('initial', 'final'))
        for region in REGIONS:
            delta = np.concatenate([(data['final_position_error']-data['initial_position_error'])[
                data['region_'+region] & np.isfinite(data['initial_position_error'])] for data in chosen])
            values[region]['final_minus_initial'] = _changes(delta)
        return values
    largest = before['largest_component']['component_id']
    if largest != after['largest_component']['component_id']:
        raise ValueError('071 largest component changed despite fixed population')
    rest = [i for i, row in enumerate(rows) if row['component_id'] != largest]
    return dict(status='COMPLETED', comparison='final_minus_step0', population_and_masks_exact=True,
        model_forwards=0, scene_count=len(rows), paired=pool(list(range(len(rows)))),
        per_scene=[dict(**row, paired=pool([i])) for i, row in enumerate(rows)],
        largest_component=dict(component_id=largest, selection=before['largest_component']['selection'],
                               paired=pool([i for i, row in enumerate(rows) if row['component_id'] == largest])),
        remaining_components=pool(rest) if rest else None,
        components={key: pool([i for i, row in enumerate(rows) if row['component_id'] == key]) for key in sorted(set(components.values()))})
