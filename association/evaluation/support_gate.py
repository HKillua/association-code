"""068 CPU reference evaluation on the immutable 056 final1000 partition.

All output groups, including duplicates and groups without valid reference,
are predicted before identity alignment. Only original first representatives
are forwarded; this evaluator makes no all-fragment inference claim.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from association.evaluation.frozen_fusion_transfer import _changes, _population
from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.learned_group_oracle import _helper_evidence, _visible_regions, _quantiles
from association.evaluation.support_control import support_switch_prediction
from association.evaluation.system_attribution import ATTRIBUTES, _metrics, _same, assert_metrics, pool_metrics
from association.models.support_gate import gate_inputs, mix_gate_states
from association.utils.runtime import read_json, write_json, write_json_gzip
from tools.analysis.compare_association_duration import ARMS, _load_arrays, _replay
from tools.analysis.frozen_fusion_transfer import frozen_motion, groups_from_cache
from tools.experiments.level1_system_evaluate import evaluate_unique_population

METHODS = (*ARMS, 'support_switch', 'learned_gate')
REGIONS = ('all', 'anchor_inside_support', 'anchor_outside_helper_supported',
           'predicted_group_no_support', 'no_predicted_group')
PAIRS = (('learned_gate', 'support_switch'), ('learned_gate', 'anchor'),
         ('learned_gate', 'attribute_hybrid'))
SENSITIVITY_SCENE, SENSITIVITY_TARGET = '3035', '374302000-1'


def _array(value, shape, name, *, boolean=False):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
        raise ValueError('068 model output must be CPU tensors: ' + name)
    result = value.detach().numpy()
    if result.shape != shape or (boolean and result.dtype != bool):
        raise ValueError('068 invalid output shape or dtype: ' + name)
    if name in ('states', 'mean_states') and result.dtype != np.float64:
        raise ValueError('068 physical outputs must remain float64: ' + name)
    if not np.isfinite(result).all():
        raise ValueError('068 nonfinite model output: ' + name)
    return result


def _predict(model, motion, groups, predictions, switch, check):
    """Infer once per original representative, before any target correspondence."""
    n, q = motion['states'].shape[:2]
    learned = predictions['attribute_hybrid'].copy()
    route = dict(gate=np.zeros((len(groups), q)), eligible=np.zeros((len(groups), q), bool),
                 logits=np.zeros((len(groups), q)))
    with torch.inference_mode():
        for index, members in enumerate(groups):
            if check:
                check()
            anchor = members[0]
            inputs = gate_inputs(motion, members, device='cpu')
            result = model(*inputs)
            states = _array(result['states'], (q, 4), 'states')
            mean = _array(result['mean_states'], (q, 4), 'mean_states')
            gate = _array(result['gate'], (q,), 'gate')
            eligible = _array(result['eligible'], (q,), 'eligible', boolean=True)
            logits = _array(result['logits'], (q,), 'logits')
            if (gate < 0).any() or (gate > 1).any():
                raise ValueError('068 gate must stay in [0,1]')
            _same(eligible, switch[anchor], '068 observable opportunity mask')
            _same(mean, predictions['mean_supported'][anchor], '068 model Mean candidate')
            _same(states[~eligible, :2], predictions['anchor'][anchor, ~eligible, :2],
                  '068 position outside opportunity must remain Anchor')
            _same(states[:, 2:], mean[:, 2:], '068 speed/heading must remain Mean')
            _same(result['gate'].numpy(), torch.sigmoid(result['logits']).numpy(),
                  '068 saved gate/logit relation')
            _same(states, mix_gate_states(inputs[0], inputs[1], result['gate'], inputs[3]).numpy(),
                  '068 saved gate/state relation')
            learned[anchor] = states
            for name, values in (('gate', gate), ('eligible', eligible), ('logits', logits)):
                route[name][index] = values
    if learned.shape != (n, q, 4):
        raise ValueError('068 complete carrier shape changed')
    return learned, route


def _routing(route, selected):
    unique = np.zeros(len(route['gate']), bool)
    unique[selected[selected >= 0]] = True
    result = {}
    for name, groups in (('all_output_groups', np.ones(len(unique), bool)),
                         ('unique_selected_groups', unique)):
        gate, eligible = route['gate'][groups], route['eligible'][groups]
        active = gate[eligible]
        result[name] = dict(output_query_count=int(gate.size), eligible_query_count=int(eligible.sum()),
            effective_gate_sum=float(active.sum()), gate_zero_count=int((active == 0).sum()),
            gate_one_count=int((active == 1).sum()), gate_between_count=int(((active > 0) & (active < 1)).sum()),
            gate_mean=float(active.mean()) if active.size else None)
    return result


def _summarize_scene(fragment, artifacts, reports, route, sid, component):
    base = artifacts['anchor']
    for method, artifact in artifacts.items():
        if set(artifact) != set(base):
            raise ValueError('068 artifact fields changed')
        for key in base:
            if key not in ('aligned_prediction', 'group_states'):
                _same(artifact[key], base[key], '068 fixed population/' + method + '/' + key)
    t = len(base['target_ids'])
    truth, valid = base['aligned_truth'][:t], base['aligned_truth_valid'][:t]
    common = valid & base['aligned_prediction_valid'][:t]
    groups = groups_from_cache(fragment, 'learned_assoc')
    anchors = np.asarray([g[0] for g in groups], np.int64)
    selected = base['aligned_group_indices'][:t]
    _, helpers, _ = _helper_evidence(fragment, groups)
    visible = _visible_regions(fragment, anchors, selected, helpers)[0]
    regions = dict(all=np.ones(common.shape[:2], bool), **{k: visible[k] for k in REGIONS[1:]})
    arrays = {'region_' + k: v for k, v in regions.items()}
    arrays['sensitivity_target'] = np.broadcast_to(
        (base['target_ids'] == SENSITIVITY_TARGET)[:, None] & (sid == SENSITIVITY_SCENE), common.shape[:2])
    methods = {}
    for method in METHODS:
        artifact = artifacts[method]
        prediction = artifact['aligned_prediction'][:t]
        errors = attribute_errors(prediction, truth)
        quantiles = {}
        for attr, columns in ATTRIBUTES.items():
            mask = common[..., list(columns)].all(-1)
            if not np.isfinite(errors[attr][mask]).all():
                raise ValueError('068 nonfinite valid physical error')
            arrays[method + '_' + attr + '_error'] = np.where(mask, errors[attr], np.nan)
            quantiles[attr] = {region: _quantiles(errors[attr][mask & select])
                               for region, select in regions.items()}
        methods[method] = dict(population=_population(artifact),
            full_target_metrics=reports['learned_assoc_' + method]['full_target_metrics'],
            common_target_strata=_metrics(prediction, truth, valid, common, regions),
            attribute_quantiles=quantiles)
    pairs = {a + '__minus__' + b: {region: {attr: _changes((
        arrays[a + '_' + attr + '_error'] - arrays[b + '_' + attr + '_error'])[
            select & np.isfinite(arrays[a + '_' + attr + '_error'])])
        for attr in ATTRIBUTES} for region, select in regions.items()} for a, b in PAIRS}
    return dict(scene_id=sid, component_id=component, fragment_count=len(fragment['node_ids']),
        fixed_population_verified=True, ordinary_source_replay_exact=True,
        methods=methods, pairwise=pairs, routing=_routing(route, selected)), arrays


def _pool(rows, arrays):
    result = dict(scene_count=len(rows), scene_ids=[r['scene_id'] for r in rows], methods={}, pairwise={}, routing={})
    for method in METHODS:
        source = [row['methods'][method] for row in rows]
        result['methods'][method] = dict(
            population={key: sum(item['population'][key] for item in source) for key in source[0]['population']},
            full_target_metrics=pool_metrics([item['full_target_metrics'] for item in source]),
            common_target_strata=pool_metrics([item['common_target_strata'] for item in source]),
            attribute_quantiles={attr: {region: _quantiles(np.concatenate([
                item[method+'_'+attr+'_error'][item['region_'+region] &
                    np.isfinite(item[method+'_'+attr+'_error'])] for item in arrays]))
                for region in REGIONS} for attr in ATTRIBUTES})
    for a, b in PAIRS:
        pair = a + '__minus__' + b
        result['pairwise'][pair] = {}
        for region in REGIONS:
            result['pairwise'][pair][region] = {}
            for attr in ATTRIBUTES:
                deltas = np.concatenate([(item[a+'_'+attr+'_error']-item[b+'_'+attr+'_error'])[
                    item['region_'+region] & np.isfinite(item[a+'_'+attr+'_error'])] for item in arrays])
                values = [row['pairwise'][pair][region][attr]['mean_error_change'] for row in rows]
                values = [v for v in values if v is not None]
                result['pairwise'][pair][region][attr] = dict(_changes(deltas),
                    scene_mean_error_change=float(np.mean(values)) if values else None,
                    better_scene_count=sum(v < -1e-9 for v in values),
                    worse_scene_count=sum(v > 1e-9 for v in values),
                    tied_scene_count=sum(abs(v) <= 1e-9 for v in values))
    for scope in rows[0]['routing']:
        items = [r['routing'][scope] for r in rows]
        counts = {key: sum(item[key] for item in items) for key in items[0] if key != 'gate_mean'}
        result['routing'][scope] = dict(counts, gate_mean=(
            counts['effective_gate_sum']/counts['eligible_query_count'] if counts['eligible_query_count'] else None))
    return result


def _sensitivity(arrays):
    """Fixed named target, no posthoc selection/removal from primary metrics."""
    result = dict(scene_id=SENSITIVITY_SCENE, target_id=SENSITIVITY_TARGET,
        interpretation='Leave-one scene-target sensitivity only; full-population results remain primary.',
        comparisons={})
    for a, b in PAIRS:
        selected, full, square_selected, square_full = [], [], [], []
        for item in arrays:
            av, bv = item[a+'_position_error'], item[b+'_position_error']
            valid = np.isfinite(av)
            use = valid & item['sensitivity_target']
            full.append((av-bv)[valid]); selected.append((av-bv)[use])
            square_full.append((av*av-bv*bv)[valid]); square_selected.append((av*av-bv*bv)[use])
        delta, subset = np.concatenate(full), np.concatenate(selected)
        square, subset_square = np.concatenate(square_full), np.concatenate(square_selected)
        remaining = len(delta)-len(subset)
        result['comparisons'][a+'__minus__'+b] = dict(
            full_point_count=len(delta), omitted_point_count=len(subset), remaining_point_count=remaining,
            full_signed_error_sum_m=float(delta.sum()), omitted_signed_error_sum_m=float(subset.sum()),
            full_squared_error_difference_sum_m2=float(square.sum()),
            omitted_squared_error_difference_sum_m2=float(subset_square.sum()),
            leave_one_mean_error_difference_m=float((delta.sum()-subset.sum())/remaining) if remaining else None,
            leave_one_mse_difference_m2=float((square.sum()-subset_square.sum())/remaining) if remaining else None,
            named_target_present=bool(len(subset)))
    return result


def compare_endpoints(initial_dir, final_dir):
    """Paired step0/final physical changes, using already saved CPU outputs only."""
    initial_dir, final_dir = Path(initial_dir), Path(final_dir)
    before, after = (read_json(directory/'summary.json') for directory in (initial_dir, final_dir))
    if (before['status'] != 'COMPLETED' or after['status'] != 'COMPLETED'
            or before['scene_ids'] != after['scene_ids'] or before['source_system'] != after['source_system']
            or before['method_names'] != list(METHODS) or after['method_names'] != list(METHODS)
            or before['device'] != 'cpu' or after['device'] != 'cpu'):
        raise ValueError('068 endpoint comparison requires the same completed CPU evaluation population')
    for arm in METHODS:
        if before['methods'][arm]['population'] != after['methods'][arm]['population']:
            raise ValueError('068 endpoint population differs')
    components = {sid: component for component, row in before['components'].items() for sid in row['scene_ids']}
    if components != {sid: component for component, row in after['components'].items() for sid in row['scene_ids']}:
        raise ValueError('068 endpoint component registry differs')
    rows = []
    for sid in before['scene_ids']:
        with np.load(initial_dir/'predictions'/f'states-{sid}.npz', allow_pickle=False) as a, \
                np.load(final_dir/'predictions'/f'states-{sid}.npz', allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError('068 endpoint arrays differ')
            for key in a.files:
                if not key.startswith('learned_gate_') and key not in ('group_gate', 'group_logits'):
                    _same(a[key], b[key], '068 endpoint fixed input/' + key)
            arrays = {name+'_'+attr: cache['learned_gate_'+attr+'_error'].copy()
                      for name, cache in (('initial', a), ('final', b)) for attr in ATTRIBUTES}
            for attr in ATTRIBUTES:
                _same(np.isfinite(arrays['initial_'+attr]), np.isfinite(arrays['final_'+attr]),
                      '068 endpoint common point mask/' + attr)
            arrays.update({region: a['region_'+region].copy() for region in REGIONS})
            rows.append(dict(scene_id=sid, component_id=components[sid], arrays=arrays))
    def pool(items):
        result = {}
        for region in REGIONS:
            result[region] = {}
            for attr in ATTRIBUTES:
                pairs = [(r['arrays']['initial_'+attr][r['arrays'][region] & np.isfinite(r['arrays']['initial_'+attr])],
                          r['arrays']['final_'+attr][r['arrays'][region] & np.isfinite(r['arrays']['final_'+attr])])
                         for r in items]
                first, second = (np.concatenate([pair[index] for pair in pairs]) for index in (0, 1))
                scene_means = [float((b-a).mean()) for a, b in pairs if len(a)]
                result[region][attr] = dict(_changes(second-first),
                    initial_mae=float(first.mean()) if len(first) else None,
                    final_mae=float(second.mean()) if len(second) else None,
                    initial_rmse=float(np.sqrt(np.square(first).mean())) if len(first) else None,
                    final_rmse=float(np.sqrt(np.square(second).mean())) if len(second) else None,
                    signed_squared_error_difference_sum=float((second*second-first*first).sum()),
                    scene_mean_error_change=float(np.mean(scene_means)) if scene_means else None,
                    better_scene_count=sum(v < -1e-9 for v in scene_means),
                    worse_scene_count=sum(v > 1e-9 for v in scene_means),
                    tied_scene_count=sum(abs(v) <= 1e-9 for v in scene_means))
        return result
    return dict(status='COMPLETED', comparison='final_minus_step0', population_and_masks_exact=True,
        model_forwards=0, reference='CPU saved unique-target outputs; all original targets retained',
        scene_count=len(rows), paired=pool(rows),
        per_scene=[dict(scene_id=row['scene_id'], component_id=row['component_id'], paired=pool([row])) for row in rows],
        components={component: pool([r for r in rows if r['component_id'] == component])
                    for component in sorted(set(components.values()))})


def evaluate_support_gate(model, validation_raw, source056: Path, metadata: dict, output: Path, check=None):
    """One complete CPU V pass; caller binds original056 and controls step0/final calls."""
    source056, output = Path(source056).resolve(), Path(output).resolve()
    protected = source056.parent
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError('068 output must be separate from immutable056')
    if any(parameter.device.type != 'cpu' for parameter in model.parameters()):
        raise ValueError('068 evaluation requires an independent CPU model')
    ids = [str(scene['scene_id']) for scene in validation_raw]
    if not ids or ids != metadata['scene_ids'] or len(set(ids)) != len(ids):
        raise ValueError('068 evaluation must retain the complete V population and order')
    if set(metadata['component_ids']) != set(ids):
        raise ValueError('068 component registry must cover exactly the same V population')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', scope='068-frozen056-support-gate-evaluation', device='cpu',
        source_system=str(source056), scene_ids=ids, total_scenes=len(ids), completed_scenes=0,
        inference_scope='first representative of every original output group, including duplicates',
        test_opened=False, scores_recomputed=False, groups_recomputed=False,
        threshold_reselected=False, optimizer_updates=0, model_forwards=0)
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
                raise ValueError('068 scene component identity changed')
            groups = groups_from_cache(fragment, 'learned_assoc')
            predictions = {arm: fragment['learned_assoc_'+arm] for arm in ARMS}
            predictions['support_switch'], switch = support_switch_prediction(
                predictions['anchor'], predictions['mean_supported'], fragment['support'], groups)
            predictions['learned_gate'], routing = _predict(model, scene['motion'], groups, predictions, switch, check)
            reports, artifacts = evaluate_unique_population(scene, {'learned_assoc': groups},
                                                             {'learned_assoc': predictions})
            artifacts = {key.removeprefix('learned_assoc_'): value for key, value in artifacts.items()}
            for arm in ARMS:
                old = replay[1]['learned_assoc_'+arm]
                if set(artifacts[arm]) != set(old):
                    raise ValueError('068 ordinary artifact schema changed')
                for key in old:
                    _same(artifacts[arm][key], old[key], '068 ordinary '+arm+'/'+key)
                assert_metrics(reports['learned_assoc_'+arm]['full_target_metrics'],
                    metadata['expected'][sid]['learned_assoc_'+arm]['full_target_metrics'], '068 '+arm)
            row, point_arrays = _summarize_scene(fragment, artifacts, reports, routing, sid, component)
            write_json_gzip(output/'scenes'/f'scene-{sid}.json.gz', row)
            payload = dict(group_gate=routing['gate'], group_eligible=routing['eligible'],
                group_logits=routing['logits'], group_anchor_node_indices=np.asarray([g[0] for g in groups]),
                inference_scope=np.asarray(state['inference_scope']),
                **{key: fragment[key] for key in ('node_ids', 'identities', 'target_identity_ids',
                    'native_states', 'support', 'nearest_seconds', 'exact_observation', 'query_seconds',
                    'learned_assoc_group_offsets', 'learned_assoc_group_node_indices')},
                **{method+'_group_states': artifacts[method]['group_states'] for method in METHODS}, **point_arrays)
            np.savez_compressed(output/'predictions'/f'states-{sid}.npz', **payload)
            np.savez_compressed(output/'system-predictions'/f'states-{sid}.npz',
                **{method+'_'+key: value for method, artifact in artifacts.items() for key, value in artifact.items()})
            rows.append(row); arrays.append(point_arrays)
            state.update(completed_scenes=len(rows), model_forwards=state['model_forwards']+len(groups),
                         last_scene_id=sid, elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        if sum(row['fragment_count'] for row in rows) != metadata['fragment_count']:
            raise ValueError('068 fragment population changed')
        result = _pool(rows, arrays)
        for arm in ARMS:
            assert_metrics(result['methods'][arm]['full_target_metrics'],
                           metadata['pooled']['learned_assoc_'+arm], '068 pooled '+arm)
        result['components'] = {}
        for component in sorted(set(metadata['component_ids'].values())):
            indices = [i for i, row in enumerate(rows) if row['component_id'] == str(component)]
            result['components'][str(component)] = _pool([rows[i] for i in indices], [arrays[i] for i in indices])
        result.update(status='COMPLETED', dataset_level='Level1', split='V_select', source_system=str(source056),
            ordinary_source_replay_exact=True, fixed_population_verified=True, method_names=list(METHODS),
            component_count=len(result['components']), fragment_count=metadata['fragment_count'],
            model_forwards=state['model_forwards'], device='cpu', optimizer_updates=0,
            test_opened=False, groups_recomputed=False, scores_recomputed=False, threshold_reselected=False,
            reference_usage='Evaluation only after every original output group has a prediction.',
            limitations=['Repeated development V; components are descriptive, not independent trials.',
                         'Only output representatives are forwarded; no all-fragment learned metrics.',
                         'CPU is the new method reference; no old057 cross-device utility gate.'],
            leave_one_named_target_sensitivity=_sensitivity(arrays))
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
