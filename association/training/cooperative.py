"""Bounded local candidate training; native experiments require separate approval.

Stages use existing scene/pair sampling. References supply losses only. This
module never opens archives, changes a split, launches a server or retries a run.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from association.utils import runtime
from association.models.fusion import FusionLoss
from association.training.common import fit_normalizer, normalize_scene
from association.training.sampling import BalancedScenePairSampler


_LEGACY_ROUTES = ('plain', 'identity', 'utility', 'support', 'learned')
_V2_ROUTES = ('plain', 'scalar-u', 'attribute-u', 'learned-attribute')


def _is_v2(model_config):
    return getattr(model_config, 'model_variant', 'legacy') == 'support_residual_v2'


def resolve_device(name):
    """Fail explicitly if the requested accelerator is unavailable."""
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda', 'mps'):
        raise ValueError('device must be cpu, cuda[:index], or mps')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError('CUDA device index unavailable')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable; no CPU fallback')
    if device.type == 'mps' and device.index not in (None, 0):
        raise ValueError('MPS exposes only device 0')
    return device


def endpoint_view(values, observed, maximum, rng, *, fraction):
    """Remove a prefix/suffix or neither, then repack native rows only.

    Query time stays fixed. Query states and hidden native values are cleared;
    no interpolation/cache is reused. Two-point fragments remain untouched.
    """
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('endpoint fraction must be in [0,1)')
    if (values.ndim != 3 or values.shape[-1] != 5 or observed.shape != values.shape[:2]
            or observed.dtype != torch.bool or not 2 <= maximum < values.shape[1]):
        raise ValueError('expected packed native/query inputs and explicit native width')
    result, mask = torch.zeros_like(values), torch.zeros_like(observed)
    result[:, maximum:, 4] = values[:, maximum:, 4]
    removed = []
    for node in range(len(values)):
        indices = torch.nonzero(observed[node, :maximum], as_tuple=True)[0]
        count = len(indices)
        if count < 2:
            raise ValueError('every fragment must retain at least two native points')
        mode = int(rng.integers(0, 3))
        amount = min(int(count * fraction), count - 2) if mode else 0
        kept = indices[amount:] if mode == 1 else indices[:count-amount]
        result[node, :len(kept)] = values[node, kept]
        mask[node, :len(kept)] = True
        removed.append(count - len(kept))
    return result, mask, removed


def physical_errors(prediction, target, valid, scales):
    """Per-attribute absolute physical error, in fixed training scale units.

    Longitude and course use shortest angular differences. Invalid targets are
    excluded before arithmetic; returned zeros outside valid are not labels.
    """
    if (prediction.shape != target.shape or valid.shape != target.shape
            or prediction.ndim != 3 or prediction.shape[-1] != 4 or valid.dtype != torch.bool):
        raise ValueError('physical errors require matching [B,Q,4] arrays and bool mask')
    scales = torch.as_tensor(scales, dtype=prediction.dtype, device=prediction.device)
    if scales.shape != (4,) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('four explicit finite positive training error scales required')
    if (target.device != prediction.device or valid.device != prediction.device
            or not torch.isfinite(prediction[valid]).all() or not torch.isfinite(target[valid]).all()):
        raise ValueError('valid physical predictions/targets must be finite on one device')
    error = torch.zeros_like(prediction)
    for column in range(4):
        mask = valid[..., column]
        difference = prediction[..., column][mask] - target[..., column][mask]
        if column in (1, 3):
            difference = torch.remainder(difference + 180., 360.) - 180.
        error[..., column][mask] = difference.abs() / scales[column]
    return error


def utility_objective(prediction, self_error, cross_error, valid, *, epsilon,
                      normalization_counts=None):
    """Detached error-comparison residual, not direct small-denominator division."""
    if (prediction.shape != self_error.shape or prediction.shape != cross_error.shape
            or valid.shape != prediction.shape or prediction.ndim != 3
            or prediction.shape[-1] != 4 or valid.dtype != torch.bool):
        raise ValueError('utility loss requires matching [B,Q,4] arrays and bool mask')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('utility epsilon must be finite and positive')
    local = valid.sum((0, 1))
    counts = local if normalization_counts is None else torch.as_tensor(normalization_counts, device=valid.device)
    if counts.shape != (4,) or (counts < local).any() or (counts <= 0).any():
        raise ValueError('effective utility batch needs supervision for every attribute')
    total = prediction.sum() * 0
    for column in range(4):
        mask = valid[..., column]
        u = prediction[..., column][mask]
        e0, e1 = self_error[..., column][mask].detach(), cross_error[..., column][mask].detach()
        if not all(torch.isfinite(item).all() for item in (u, e0, e1)):
            raise ValueError('nonfinite utility labels/predictions')
        residual = (e0 - e1) - e0.clamp_min(epsilon) * u
        total = total + F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction='sum') / counts[column] / 4
    return total


@torch.no_grad()
def teacher_error_pair(teacher, values, observed, targets, valid, *, scales):
    """A fixed teacher compares i alone and i<-j against i's own reference.

    May be used for negative helpers; it does not assert a common true identity.
    Both directional rows retain their own valid mask and target. No V fitting.
    """
    if len(values) != 2 or targets.shape[0] != 2:
        raise ValueError('teacher_error_pair requires exactly two anchored fragments')
    teacher.eval().requires_grad_(False)
    encoded = teacher.encode(values, observed)
    single = torch.stack([teacher.decode_group(encoded, [i], routing='plain') for i in (0, 1)])
    cross = torch.stack([teacher.decode_group(encoded, [i, 1-i], routing='plain') for i in (0, 1)])
    return (physical_errors(single, targets, valid, scales),
            physical_errors(cross, targets, valid, scales))


def stage_parameters(model, stage, *, routing='plain'):
    """Set stage modes/trainability explicitly; return exact optimizer members."""
    model.eval().requires_grad_(False)
    if stage == 'teacher':
        names = ['input_projection', 'encoder', 'decoder', 'state_projection',
                 'role_embedding', 'identity_base']
        if _is_v2(model.config):
            names += ['support_projection', 'query_pool', 'local_identity', 'fusion_projection']
    elif stage == 'utility':
        names = ['utility_head'] if _is_v2(model.config) else ['query_pool', 'utility_head']
    elif stage == 'identity':
        names = ['identity_with_utility']
    elif stage == 'adapt':
        if routing not in (_V2_ROUTES if _is_v2(model.config) else _LEGACY_ROUTES):
            raise ValueError('unknown fusion route')
        names = ['decoder', 'state_projection', 'role_embedding']
        if _is_v2(model.config):
            names.append('fusion_projection')
        if routing in ('learned', 'learned-attribute'):
            names.append('ordinary_gate')
    else:
        raise ValueError('stage must be teacher, utility, identity or adapt')
    model.zero_grad(set_to_none=True)
    for name in names:
        getattr(model, name).requires_grad_(True).train()
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _validate_training_scenes(scenes):
    from association.data.timed import validate_timed_scene
    from association.evaluation.physical import _check_valid_states
    if not scenes or len({s['scene_id'] for s in scenes}) != len(scenes):
        raise ValueError('nonempty distinct training scenes required')
    for scene in scenes:
        if scene.get('split') != 'T_core':
            raise ValueError('candidate first fit uses fixed T_core only; T_test sealed')
        validate_timed_scene(scene)
        size = len(scene['native_values'])
        identities = np.asarray(scene['identities'])
        if (identities.shape != (size,) or any(not isinstance(x, (str, np.str_)) or not x for x in identities)
                or len(set(map(str, scene['node_ids']))) != size):
            raise ValueError('training nodes need distinct IDs and explicit identity strings')
        pairs, labels = np.asarray(scene['pairs']), np.asarray(scene['labels'])
        canonical = np.column_stack(np.triu_indices(size, 1))
        if pairs.dtype.kind not in 'iu' or not np.array_equal(pairs, canonical):
            raise ValueError('training requires all canonical candidate pairs in original order')
        if (labels.shape != (len(pairs),) or not np.isin(labels, (0, 1)).all()
                or not np.array_equal(labels, identities[pairs[:, 0]] == identities[pairs[:, 1]])):
            raise ValueError('training pair labels must equal the supplied identities')
        for identity in scene['identities']:
            if str(identity) not in scene.get('fusion_targets', {}):
                raise ValueError('each training node needs an explicit reference row, possibly masked')
        for target in scene['fusion_targets'].values():
            if not np.array_equal(target['query_times'], scene['grid']):
                raise ValueError('reference queries must equal physical output times')
            values, valid = np.asarray(target['states']), np.asarray(target['valid'])
            if (values.shape != (len(scene['grid']), 4) or valid.shape != values.shape
                    or valid.dtype != np.bool_ or not np.isfinite(values[valid]).all()):
                raise ValueError('reference target needs [Q,4] states and boolean valid mask with finite labels')
            _check_valid_states(values, valid, 'candidate reference target')


def _batch(scenes, packed, sampler, batch_size, device, rng, fraction, maximum):
    groups = sampler.sample(batch_size)
    items = []
    for scene_index, pair_rows in groups:
        scene = scenes[scene_index]
        for row in pair_rows:
            nodes = np.asarray(scene['pairs'])[row]
            values = torch.as_tensor(packed[scene_index][0][nodes], device=device)
            observed = torch.as_tensor(packed[scene_index][1][nodes], device=device)
            values, observed, removed = endpoint_view(values, observed, maximum, rng, fraction=fraction)
            targets = [scene['fusion_targets'][str(scene['identities'][node])] for node in nodes]
            target = torch.as_tensor(np.stack([x['states'] for x in targets]), dtype=values.dtype, device=device)
            valid = torch.as_tensor(np.stack([x['valid'] for x in targets]), dtype=torch.bool, device=device)
            label = float(scene['labels'][row])
            items.append(dict(values=values, observed=observed, target=target, valid=valid,
                              label=label, removed=removed, scene_id=str(scene['scene_id']), pair_row=int(row)))
    return items, sampler.batch_metadata(groups)


def train_batch(model, objective, optimizer, items, *, stage, routing='plain',
                teacher=None, error_scales=None, utility_epsilon=1e-3,
                gradient_clip=1., check=None):
    """One effective batch, streaming one sampled pair graph at a time.

    Pair streaming does not resample or discard supervision. Denominators are
    computed over the full batch before any backward/optimizer update.
    Detached loss components are diagnostic sums: physical includes its
    original variance regularizer; identity_bce and utility are separate.
    """
    if not items or not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError('nonempty batch and positive gradient clip required')
    parameters = stage_parameters(model, stage, routing=routing)
    objective.requires_grad_(stage in ('teacher', 'adapt')).train(stage in ('teacher', 'adapt'))
    objective.zero_grad(set_to_none=True)
    if stage in ('teacher', 'adapt'):
        parameters += list(objective.parameters())
    actual = [p for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(parameters) or {id(p) for p in actual} != {id(p) for p in parameters}:
        raise ValueError('optimizer parameters do not match this explicit stage')
    if stage == 'utility' and (teacher is None or teacher is model or error_scales is None):
        raise ValueError('utility requires a separate frozen teacher and training error scales')
    counts = torch.zeros(4, dtype=torch.long, device=items[0]['values'].device)
    for item in items:
        if item['label'] not in (0., 1.):
            raise ValueError('training pair labels must be binary')
        # valid has both directional targets already. Teacher adds cross only
        # for positives; adapt uses those two cross predictions and no singles.
        factor = (1 + int(item['label'] == 1)) if stage == 'teacher' else (
            1 if stage == 'utility' else int(item['label'] == 1))
        counts += item['valid'].sum((0, 1)) * factor
    if stage != 'identity' and (counts <= 0).any():
        raise ValueError('no supervision for an effective-batch attribute; no resampling')
    optimizer.zero_grad(set_to_none=True)
    total, regularized, used = 0., False, 0
    components = dict(physical=0., identity_bce=0., utility=0.)
    try:
        for item in items:
            if check:
                check()
            x, mask, target, valid = (item[k] for k in ('values', 'observed', 'target', 'valid'))
            pair = torch.tensor([[0, 1]], dtype=torch.long, device=x.device)
            if stage == 'adapt' and item['label'] != 1:
                continue  # Same preselected positive batch for every fusion arm.
            encoded = model.encode(x, mask)
            item_components = dict.fromkeys(components, None)
            if stage == 'utility':
                e0, e1 = teacher_error_pair(teacher, x, mask, target, valid, scales=error_scales)
                u = model.utility(encoded, pair)[0]
                loss = utility_objective(u, e0, e1, valid, epsilon=utility_epsilon, normalization_counts=counts)
                item_components['utility'] = loss
            elif stage == 'identity':
                score = model.score_pairs(encoded, pair, use_utility=True)
                loss = F.binary_cross_entropy_with_logits(score, score.new_tensor([item['label']])) / len(items)
                item_components['identity_bce'] = loss
            else:
                predicted, truths, masks = [], [], []
                if stage == 'teacher':
                    predicted.extend(model.decode_group(encoded, [i], routing='plain') for i in (0, 1))
                    truths.extend(target.unbind(0)); masks.extend(valid.unbind(0))
                if item['label'] == 1:
                    predicted.extend(model.decode_group(encoded, [i, 1-i], routing=routing) for i in (0, 1))
                    truths.extend(target.unbind(0)); masks.extend(valid.unbind(0))
                stats = objective(torch.stack(predicted), torch.stack(truths), torch.stack(masks),
                                  normalization_counts=counts, regularizer_weight=0. if regularized else 1.)
                regularized = True
                loss = stats['total']
                item_components['physical'] = loss
                if stage == 'teacher':
                    score = model.score_pairs(encoded, pair, use_utility=False)
                    identity_loss = F.binary_cross_entropy_with_logits(score, score.new_tensor([item['label']])) / len(items)
                    loss = loss + identity_loss
                    item_components['identity_bce'] = identity_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite candidate objective')
            loss.backward()
            # One host transfer replaces the existing scalar loss transfer; no
            # additional forward, RNG draw, backward or device synchronization.
            detached = [loss.detach()] + [value.detach() if value is not None else loss.new_zeros(())
                                         for value in item_components.values()]
            values = torch.stack(detached).cpu().tolist()
            total += values[0]
            for key, value in zip(components, values[1:]):
                components[key] += value
            used += 1
        if not used:
            raise ValueError('no effective training items')
        if check:
            check()
        norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
        optimizer.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    return dict(stage=stage, routing=routing, loss=total, loss_components=components,
                valid_counts=counts.cpu().tolist(),
                gradient_norm=float(norm.detach().cpu()), sampled_pairs=len(items), used_pairs=used,
                removed_observations=sum(sum(x['removed']) for x in items))


def training_configuration(config):
    required = {'seed', 'batch_size', 'teacher_updates', 'utility_updates', 'adapt_updates',
                'learning_rate', 'gradient_clip', 'endpoint_fraction', 'utility_epsilon',
                'error_scale_policy', 'routes', 'max_seconds', 'batch_tracks', 'pair_batch_size'}
    if not required.issubset(config) or set(config) - required - {'longitude_loss_mode'}:
        raise ValueError('candidate training config must explicitly contain ' + ', '.join(sorted(required)))
    cfg = dict(config)
    cfg.setdefault('longitude_loss_mode', 'linear')
    if cfg['longitude_loss_mode'] not in ('linear', 'shortest_arc'):
        raise ValueError('longitude_loss_mode must be linear or shortest_arc')
    for key in ('seed', 'batch_size', 'teacher_updates', 'utility_updates', 'adapt_updates', 'batch_tracks', 'pair_batch_size'):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'seed' else 1):
            raise ValueError(key + ' must be an explicit nonnegative seed or positive count')
    if cfg['batch_size'] < 2 or cfg['batch_size'] % 2:
        raise ValueError('batch_size must be even and >=2')
    for key in ('learning_rate', 'gradient_clip', 'utility_epsilon', 'max_seconds'):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + ' must be positive and finite')
    fraction = cfg['endpoint_fraction']
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('endpoint_fraction must be in [0,1)')
    if cfg['error_scale_policy'] != 'training_native_std':
        raise ValueError('first candidate explicitly uses fixed training_native_std error scales')
    routes = cfg['routes']
    if (not isinstance(routes, list) or not routes
            or any(not isinstance(route, str) for route in routes)
            or len(set(routes)) != len(routes)):
        raise ValueError('comparison routes must be a nonempty unique string list')
    if any(route in _V2_ROUTES[1:] for route in routes):
        if set(routes) != set(_V2_ROUTES):
            raise ValueError('v2 comparison requires plain, scalar-u, attribute-u and learned-attribute')
    elif any(route not in _LEGACY_ROUTES for route in routes) or not {'plain', 'utility'}.issubset(routes):
        raise ValueError('unique comparison routes must include plain and utility')
    return cfg


def fit_support_time_scaling(train_scenes, packed, max_observations, check=None):
    """Fit the declared MLP-only scale once from packed T_core FP32 times.

    Physical states, labels, reference values and masked native padding never
    enter the statistic. All query times are inputs, even outside native support.
    This function neither modifies packed arrays nor normalizes encoder time.
    """
    if (not train_scenes or len(train_scenes) != len(packed)
            or any(scene.get('split') != 'T_core' for scene in train_scenes)
            or len({scene['scene_id'] for scene in train_scenes}) != len(train_scenes)):
        raise ValueError('support time scale requires unique nonempty T_core scenes only')
    if isinstance(max_observations, bool) or not isinstance(max_observations, int) or max_observations < 2:
        raise ValueError('max_observations must be an integer >=2')
    scale, tracks, native_count, query_count = 1., 0, 0, 0
    for values, observed in packed:
        if check:
            check()
        values, observed = np.asarray(values), np.asarray(observed)
        if (values.dtype != np.float32 or values.ndim != 3 or values.shape[-1] != 5
                or not len(values) or values.shape[1] <= max_observations
                or observed.shape != values.shape[:2] or observed.dtype != np.bool_):
            raise ValueError('support time scale requires packed FP32 values and boolean observation masks')
        mask = observed[:, :max_observations]
        if (mask.sum(1) < 2).any() or observed[:, max_observations:].any():
            raise ValueError('support time scale requires >=2 native points and no observed query rows')
        native_times = values[:, :max_observations, 4][mask]
        query_times = values[:, max_observations:, 4]
        if not np.isfinite(native_times).all() or not np.isfinite(query_times).all():
            raise ValueError('support time scale requires finite valid native/query times')
        scale = max(scale, float(np.abs(native_times).max()), float(np.abs(query_times).max()))
        tracks += len(values)
        native_count += native_times.size
        query_count += query_times.size
    return dict(policy='train_absmax_v1', scale_tau=scale, fit_splits=['T_core'],
                training_scene_count=len(train_scenes), training_track_count=tracks,
                valid_native_time_count=native_count, query_time_count=query_count,
                scope='packed FP32 input times only; fixed across all stages, arms and evaluation')


def _snapshot(model, objective, optimizer, normalizer, config, *, stage, route, updates):
    def cpu(state):
        if isinstance(state, torch.Tensor):
            return state.detach().cpu().clone()
        if isinstance(state, dict):
            return {k: cpu(v) for k, v in state.items()}
        if isinstance(state, (list, tuple)):
            return type(state)(cpu(v) for v in state)
        return state
    result = dict(format='cooperative-physical-v2' if _is_v2(model.config) else 'cooperative-physical-v1',
                model_config=asdict(model.config),
                model_state=cpu(model.state_dict()), objective_state=cpu(objective.state_dict()),
                support_time_scale_tau=float(model.support_time_scale),
                optimizer_state=cpu(optimizer.state_dict()), normalizer=normalizer,
                training=config, stage=stage, route=route, completed_updates=updates,
                selection_policy='predetermined stage endpoint; no fusion selection on unchanged F1',
                resume_supported=False)
    if _is_v2(model.config):
        result.update(model_variant=model.config.model_variant,
                      representation_version=normalizer['cooperative_representation']['version'])
    return result


def load_cooperative_checkpoint(path, *, device):
    """Restore an inference snapshot; this is not an optimizer-resume API."""
    from association.models.cooperative import CooperativeConfig, CooperativeModel
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') not in ('cooperative-physical-v1', 'cooperative-physical-v2'):
        raise ValueError('unsupported cooperative checkpoint')
    config = CooperativeConfig(**payload['model_config'])
    if _is_v2(config):
        if (payload['format'] != 'cooperative-physical-v2'
                or payload.get('model_variant') != config.model_variant
                or payload.get('representation_version') != config.model_variant):
            raise ValueError('v2 checkpoint format, model and representation versions must agree')
    elif payload['format'] != 'cooperative-physical-v1':
        raise ValueError('legacy model requires a v1 cooperative checkpoint')
    model = CooperativeModel(config, payload['normalizer'],
                             support_time_scale=payload.get('support_time_scale_tau'))
    model.load_state_dict(payload['model_state'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite model checkpoint')
    return model.to(resolve_device(device)).eval(), payload


def _load_shared(directory, model_config, config, contract, validation_scenes):
    """Load one completed shared endpoint by its actual protocol/population."""
    from association.evaluation.cooperative import _fixed_records
    directory = Path(directory)
    source = runtime.read_json(directory / 'run.json')
    if source.get('status') != 'COMPLETED' or source.get('phase') != 'shared':
        raise ValueError('arm requires a COMPLETED shared run')
    model, payload = load_cooperative_checkpoint(directory / 'shared.pt', device='cpu')
    if payload.get('shared_format') != 'cooperative-shared-v1':
        raise ValueError('unsupported shared endpoint format')
    for key in ('training', 'model_config', 'scene_contract', 'support_time_scaling',
                'identity_selection', 'identity_selection_phase'):
        if payload.get(key) != source.get(key):
            raise ValueError('shared checkpoint/run disagree on ' + key)
    protocol = lambda cfg: {key: value for key, value in cfg.items() if key != 'max_seconds'}
    if protocol(payload['training']) != protocol(config) or payload['model_config'] != asdict(model_config):
        raise ValueError('shared training/model protocol differs; only job budget may change')
    if payload['scene_contract'] != contract:
        raise ValueError('shared scene contract differs from current train/validation population')
    if (payload.get('stage') != 'utility' or payload.get('completed_updates') != config['utility_updates']
            or source.get('completed_updates') != config['teacher_updates'] + config['utility_updates']
            or source.get('completed_routes') != []):
        raise ValueError('shared endpoint does not contain the complete teacher/utility stages')
    for stage, count in (('teacher', config['teacher_updates']), ('utility', config['utility_updates'])):
        phase = source.get('phases', {}).get(stage, {})
        if phase.get('status') != 'COMPLETED' or phase.get('completed_updates') != count:
            raise ValueError('shared stage is incomplete: ' + stage)
    if (payload['identity_selection_phase'] != 'teacher_endpoint_before_utility'
            or source.get('threshold') != payload['identity_selection'].get('threshold')
            or payload['support_time_scaling'].get('scale_tau') != payload['support_time_scale_tau']):
        raise ValueError('shared identity selection or time scale is inconsistent')
    records = runtime.load_records(directory / 'identity-records',
        dict(split='V_select', scene_contract=contract[1]))
    records = _fixed_records(validation_scenes, records)
    return model, payload, records


def train_cooperative(train_scenes, validation_scenes, model_config, training_config, output_dir,
                      *, device, initial_elapsed_seconds=0., check=None, progress=None,
                      save_predictions=False, phase='full', shared_from=None, route=None):
    """Teacher -> frozen utility -> matched fixed-identity fusion arms.

    The first experiment intentionally excludes changing identity. Full-model
    identity modules/step are available, but not silently added to this fit.
    ``full`` retains the original single-job path. V2 ``shared`` writes the
    teacher/utility endpoint and fixed identity records; ``arm`` loads that
    completed endpoint and performs one fresh, identically seeded adaptation.
    The wall budget covers this job's stages, evaluation and initial data
    preparation. No resume/retry or implicit training schedule is provided.
    Phase wall times include sampling and checkpoint/result writes. Historical
    history[*].seconds retains its narrower train_batch plus device-sync scope.
    An optional observer receives the same small records as progress.json.
    """
    started = time.monotonic()
    from association.models.cooperative import CooperativeModel
    from association.evaluation.cooperative import evaluate_cooperative, score_cooperative
    from association.evaluation.association import select_threshold
    cfg = training_configuration(training_config)
    if phase not in ('full', 'shared', 'arm'):
        raise ValueError('phase must be full, shared or arm')
    if phase != 'full' and not _is_v2(model_config):
        raise ValueError('shared/arm execution is supported only by cooperative v2')
    if phase == 'arm':
        if shared_from is None or route not in cfg['routes']:
            raise ValueError('arm requires a shared source and one configured route')
    elif shared_from is not None or route is not None:
        raise ValueError('shared_from and route are only valid for arm execution')
    if (_is_v2(model_config) and set(cfg['routes']) != set(_V2_ROUTES)
            or not _is_v2(model_config) and any(route not in _LEGACY_ROUTES for route in cfg['routes'])):
        raise ValueError('comparison routes do not match the cooperative model variant')
    if not isinstance(save_predictions, bool):
        raise ValueError('save_predictions must be boolean')
    device = resolve_device(device)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    if not math.isfinite(initial_elapsed_seconds) or initial_elapsed_seconds < 0:
        raise ValueError('initial elapsed time must be finite and nonnegative')
    _validate_training_scenes(train_scenes)
    if not validation_scenes or any(s.get('split') != 'V_select' for s in validation_scenes):
        raise ValueError('nonempty V_select required; T_test sealed')
    if ({s['scene_id'] for s in train_scenes} & {s['scene_id'] for s in validation_scenes}
            or {s['component_id'] for s in train_scenes} & {s['component_id'] for s in validation_scenes}):
        raise ValueError('training and validation scene/components must be disjoint')
    # Reuse the reference's full native/data contract before any optimizer or
    # archive-derived training. In particular V errors cannot surface only after
    # spending all teacher updates. This records content, never a SHA gate.
    from association.training.tmtaff import _scenes
    contract = _scenes(train_scenes, validation_scenes, model_config.length)
    if any(s['max_observations'] != model_config.max_observations for s in [*train_scenes, *validation_scenes]):
        raise ValueError('model native width must match the declared input cap')
    history, completed = [], {}
    def bounded():
        if check:
            check()
        if initial_elapsed_seconds + time.monotonic() - started >= cfg['max_seconds']:
            raise TimeoutError('cumulative candidate budget exhausted')
    output.mkdir(parents=True, exist_ok=False)
    run = dict(status='RUNNING', device=str(device), training=cfg, model_config=asdict(model_config),
               model_variant=getattr(model_config, 'model_variant', 'legacy'),
               scope='fixed identity candidate development comparison', test_opened=False,
               save_predictions=save_predictions,
               scene_contract=contract, completed_updates=0, completed_routes=[],
               phases={'external_preparation': dict(status='COMPLETED', seconds=initial_elapsed_seconds)},
               timing_scope=dict(phase_seconds='wall time including sampling and checkpoint/result writes; '
                                             'terminal run.json write excluded',
                   external_preparation='caller-supplied elapsed time before train_cooperative',
                   local_preparation='input contract validation, normalization, packing and model/loss allocation',
                   history_seconds='train_batch plus device synchronization only; excludes sampling and artifact writes'))
    if phase != 'full':
        run['phase'] = phase
    if phase == 'arm':
        run.update(route=route, shared_source=dict(path=str(Path(shared_from).resolve()),
                   format='cooperative-shared-v1'))
    active, active_started = None, started

    def report_progress(stage, *, update=0, **details):
        now = time.monotonic()
        item = dict(phase=active['phase'], route=active['route'], stage=stage, update=update,
                    completed_updates=run['completed_updates'],
                    phase_completed_updates=active['completed_updates'],
                    total_updates=active['total_updates'], elapsed_seconds=initial_elapsed_seconds+now-started,
                    phase_elapsed_seconds=now-active_started, **details)
        if 'identity_selection' in run:
            item.update(threshold=run['threshold'], identity_selection=run['identity_selection'])
        run['last_progress'] = item
        if progress is not None:
            progress(dict(item))
        runtime.write_json(output / 'progress.json', dict(item, status='RUNNING'))

    def begin_phase(phase, *, route=None, total_updates=0, tick=None):
        nonlocal active, active_started
        active_started = time.monotonic() if tick is None else tick
        active = dict(phase=phase, route=route, status='RUNNING', completed_updates=0,
                      total_updates=total_updates, checkpoint_updates=0)
        run['phases'][phase + (':' + route if route else '')] = active
        report_progress('starting')

    def finish_phase():
        report_progress('completed', update=active['completed_updates'])
        active.update(status='COMPLETED', seconds=time.monotonic()-active_started)
        run['elapsed_seconds'] = initial_elapsed_seconds+time.monotonic()-started
        runtime.write_json(output / 'run.json', run)
        # Include the phase's completed-run write without a second write loop.
        active['seconds'] = time.monotonic()-active_started

    def evaluation_progress(item):
        detail = dict(item)
        stage = detail.pop('stage')
        if 'elapsed_seconds' in detail:
            detail['evaluation_elapsed_seconds'] = detail.pop('elapsed_seconds')
        report_progress(stage, **detail)

    try:
        begin_phase('local_preparation', tick=started)
        runtime.write_json(output / 'run.json', run)
        bounded()
        torch.manual_seed(cfg['seed'])
        if phase == 'arm':
            model, shared, records = _load_shared(shared_from, model_config, cfg, contract, validation_scenes)
            normalizer = shared['normalizer']
            from association.data.cooperative import normalize_cooperative_scene
            packed = []
            for scene in train_scenes:
                bounded()
                packed.append(normalize_cooperative_scene(scene, normalizer, model_config))
            scaling = copy.deepcopy(shared['support_time_scaling'])
            run.update(identity_selection=copy.deepcopy(shared['identity_selection']),
                       identity_selection_phase=shared['identity_selection_phase'],
                       threshold=shared['identity_selection']['threshold'],
                       cooperative_representation=copy.deepcopy(normalizer['cooperative_representation']))
            threshold = run['threshold']
            model = model.to(device)
        elif _is_v2(model_config):
            from association.data.cooperative import fit_cooperative_normalizer, normalize_cooperative_scene
            normalizer = fit_cooperative_normalizer(train_scenes, model_config, check=bounded)
            packed = []
            for scene in train_scenes:
                bounded()
                packed.append(normalize_cooperative_scene(scene, normalizer, model_config))
            run['cooperative_representation'] = copy.deepcopy(normalizer['cooperative_representation'])
        else:
            normalizer = fit_normalizer(train_scenes)
            packed = [normalize_scene(scene, normalizer) for scene in train_scenes]
        if phase != 'arm':
            if model_config.support_time_policy == 'train_absmax_v1':
                scaling = fit_support_time_scaling(train_scenes, packed, model_config.max_observations, check=bounded)
            else:
                scaling = dict(policy='legacy', scale_tau=1., scope='historical unscaled hand-crafted tau features')
        run['support_time_scaling'] = scaling
        if phase != 'arm':
            model = CooperativeModel(model_config, normalizer, support_time_scale=scaling['scale_tau']).to(device)
        objective = FusionLoss(position_loss_mode='standard_smooth_l1',
                               longitude_loss_mode=cfg['longitude_loss_mode']).to(device)
        if phase == 'arm':
            objective.load_state_dict(shared['objective_state'], strict=True)
        teacher = None
        finish_phase()
        def fit_stage(target_model, target_objective, stage, steps, seed, route='plain'):
            parameters = stage_parameters(target_model, stage, routing=route)
            target_objective.requires_grad_(stage in ('teacher', 'adapt'))
            if stage in ('teacher', 'adapt'):
                parameters += list(target_objective.parameters())
            names = ['model.' + name for name, value in target_model.named_parameters() if value.requires_grad]
            names += ['objective.' + name for name, value in target_objective.named_parameters() if value.requires_grad]
            run.setdefault('training_parameters', {})[stage + (':' + route if stage == 'adapt' else '')] = dict(
                names=names, parameter_count=sum(parameter.numel() for parameter in parameters),
                seed=seed, updates=steps)
            optimizer = torch.optim.Adam(parameters, lr=cfg['learning_rate'])
            sampler = BalancedScenePairSampler(train_scenes, seed)
            view_rng = np.random.default_rng(seed + 100)
            torch.manual_seed(seed)
            for step in range(steps):
                report_progress('sampling', update=step+1)
                bounded()
                items, population = _batch(train_scenes, packed, sampler, cfg['batch_size'], device,
                    view_rng, cfg['endpoint_fraction'], model_config.max_observations)
                report_progress('training', update=step+1)
                tick = time.monotonic()
                report = train_batch(target_model, target_objective, optimizer, items, stage=stage,
                    routing=route, teacher=teacher, error_scales=normalizer['std'],
                    utility_epsilon=cfg['utility_epsilon'], gradient_clip=cfg['gradient_clip'], check=bounded)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elif device.type == 'mps':
                    torch.mps.synchronize()
                report.update(update=step+1, seconds=time.monotonic()-tick, sampling=population)
                history.append(report)
                active['completed_updates'] = step+1
                run['completed_updates'] += 1
                report_progress('checkpoint', update=step+1)
                path = output / (stage + ('-' + route if stage == 'adapt' else '') + '.pt')
                temporary = path.with_suffix('.tmp')
                torch.save(_snapshot(target_model, target_objective, optimizer, normalizer, cfg,
                                     stage=stage, route=route, updates=step+1), temporary)
                temporary.replace(path)
                active['checkpoint_updates'] = step+1
                runtime.write_json(output / 'history.json', history)
                report_progress('update_completed', update=step+1)
            return sampler.describe()
        def select_identity():
            # In v2, only u is trained after this fixed teacher endpoint. Legacy
            # keeps its historical post-utility order and RNG stream unchanged.
            begin_phase('identity_scoring_selection')
            records = score_cooperative(model, normalizer, validation_scenes, device=device,
                batch_tracks=cfg['batch_tracks'], pair_batch_size=cfg['pair_batch_size'],
                use_utility_identity=False, check=bounded, progress=evaluation_progress)
            report_progress('threshold_selection')
            selection = select_threshold(records)
            run['threshold'] = selection['threshold']
            run['identity_selection'] = selection
            run['identity_selection_phase'] = ('teacher_endpoint_before_utility' if _is_v2(model_config)
                                               else 'after_utility')
            report_progress('threshold_selected', threshold=selection['threshold'], selection=selection)
            finish_phase()
            return records, selection['threshold']

        if phase != 'arm':
            begin_phase('teacher', total_updates=cfg['teacher_updates'])
            fit_stage(model, objective, 'teacher', cfg['teacher_updates'], cfg['seed'])
            finish_phase()
            if _is_v2(model_config):
                records, threshold = select_identity()
            begin_phase('utility', total_updates=cfg['utility_updates'])
            teacher = copy.deepcopy(model).eval().requires_grad_(False)
            fit_stage(model, objective, 'utility', cfg['utility_updates'], cfg['seed']+1)
            finish_phase()
            if not _is_v2(model_config):
                records, threshold = select_identity()
        if phase == 'shared':
            begin_phase('shared_export')
            bounded()
            bundle = torch.load(output / 'utility.pt', map_location='cpu', weights_only=True)
            bundle.update(shared_format='cooperative-shared-v1', scene_contract=contract,
                          support_time_scaling=copy.deepcopy(scaling),
                          identity_selection=copy.deepcopy(run['identity_selection']),
                          identity_selection_phase=run['identity_selection_phase'])
            torch.save(bundle, output / 'shared.tmp')
            (output / 'shared.tmp').replace(output / 'shared.pt')
            runtime.save_records(output / 'identity-records', records,
                                 dict(split='V_select', scene_contract=contract[1]))
            run.update(shared_checkpoint='shared.pt', identity_records='identity-records',
                       shared_format='cooperative-shared-v1')
            finish_phase()
        routes = [] if phase == 'shared' else ([route] if phase == 'arm' else cfg['routes'])
        for route in routes:
            bounded()
            begin_phase('adapt', route=route, total_updates=cfg['adapt_updates'])
            arm, loss = copy.deepcopy(model), copy.deepcopy(objective)
            fit_stage(arm, loss, 'adapt', cfg['adapt_updates'], cfg['seed']+2, route)
            finish_phase()
            begin_phase('evaluation', route=route)
            prediction_writer = None
            if save_predictions:
                from association.evaluation.artifacts import ScenePredictionWriter
                prediction_writer = ScenePredictionWriter(output / (route+'-predictions'))
            report = evaluate_cooperative(arm, normalizer, validation_scenes, device=device,
                batch_tracks=cfg['batch_tracks'], pair_batch_size=cfg['pair_batch_size'],
                select_on_validation=False, threshold=threshold, routing=route,
                use_utility_identity=False, fixed_records=records, check=bounded, progress=evaluation_progress,
                prediction_writer=prediction_writer)
            # Numpy score arrays remain in explicit NPZ caches, not JSON magic.
            report_progress('writing_evaluation', completed_scenes=len(validation_scenes),
                            total_scenes=len(validation_scenes), scene_id=None)
            cache = {}
            for index, record in enumerate(report.pop('records')):
                cache[f'scores_{index}'] = np.asarray(record['scores'])
                cache[f'pairs_{index}'] = np.asarray(record['pairs'])
            np.savez_compressed(output / (route + '-scores.npz'), **cache)
            runtime.write_json(output / (route + '-evaluation.json'), report)
            completed[route] = report
            run['completed_routes'] = list(completed)
            finish_phase()
        bounded()
        run.update(status='COMPLETED', completed_routes=list(completed),
                   parameter_count=sum(p.numel() for p in model.parameters()), threshold=threshold,
                   error_scales=normalizer['std'], elapsed_seconds=initial_elapsed_seconds+time.monotonic()-started,
                   identity_policy='fixed s0 and one full-V threshold for every arm',
                   ordinary_gate_features=('same frozen teacher-trained query pool; fresh ordinary attribute gate, no gain outputs'
                       if _is_v2(model_config) else 'same frozen utility-trained query pool; fresh ordinary gate, no gain outputs'),
                   controls_missing_for_utility_attribution=(['identity-only'] if _is_v2(model_config)
                       else [x for x in ('identity', 'learned') if x not in cfg['routes']]),
                   selection_policy='predetermined final updates', resume_supported=False)
        runtime.write_json(output / 'progress.json', dict(run['last_progress'], status='COMPLETED',
                           completed_routes=list(completed), elapsed_seconds=run['elapsed_seconds']))
        runtime.write_json(output / 'run.json', run)
        return run
    except BaseException as error:
        terminal = 'INTERRUPTED' if isinstance(error, (TimeoutError, InterruptedError, KeyboardInterrupt)) else 'FAILED'
        if active is not None and active['status'] == 'RUNNING':
            active.update(status=terminal, seconds=time.monotonic()-active_started)
        run.update(status=terminal,
                   error=f'{type(error).__name__}: {error}', completed_routes=list(completed),
                   elapsed_seconds=initial_elapsed_seconds+time.monotonic()-started, resume_supported=False)
        # A failure while preserving evidence must not replace the original
        # training/timeout/interruption exception. Try both small terminal files.
        for name, payload in (('progress.json', dict(run.get('last_progress', {}), status=terminal,
                                  error=run['error'], completed_routes=list(completed),
                                  elapsed_seconds=run['elapsed_seconds'])), ('run.json', run)):
            try:
                runtime.write_json(output / name, payload)
            except BaseException as writing_error:
                message = f'{name}: {type(writing_error).__name__}: {writing_error}'
                run.setdefault('evidence_write_errors', []).append(message)
                if hasattr(error, 'add_note'):
                    error.add_note(message)
        raise
