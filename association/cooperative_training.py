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

from . import runtime
from .fusion import FusionLoss
from .reference import fit_normalizer, normalize_scene
from .sampling import BalancedScenePairSampler


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
    elif stage == 'utility':
        names = ['query_pool', 'utility_head']
    elif stage == 'identity':
        names = ['identity_with_utility']
    elif stage == 'adapt':
        if routing not in ('plain', 'identity', 'utility', 'support', 'learned'):
            raise ValueError('unknown fusion route')
        names = ['decoder', 'state_projection', 'role_embedding']
        if routing == 'learned':
            names.append('ordinary_gate')
    else:
        raise ValueError('stage must be teacher, utility, identity or adapt')
    model.zero_grad(set_to_none=True)
    for name in names:
        getattr(model, name).requires_grad_(True).train()
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _validate_training_scenes(scenes):
    from .timed_inputs import validate_timed_scene
    from .fusion_metrics import _check_valid_states
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
    try:
        for item in items:
            if check:
                check()
            x, mask, target, valid = (item[k] for k in ('values', 'observed', 'target', 'valid'))
            pair = torch.tensor([[0, 1]], dtype=torch.long, device=x.device)
            if stage == 'adapt' and item['label'] != 1:
                continue  # Same preselected positive batch for every fusion arm.
            encoded = model.encode(x, mask)
            if stage == 'utility':
                e0, e1 = teacher_error_pair(teacher, x, mask, target, valid, scales=error_scales)
                u = model.utility(encoded, pair)[0]
                loss = utility_objective(u, e0, e1, valid, epsilon=utility_epsilon, normalization_counts=counts)
            elif stage == 'identity':
                score = model.score_pairs(encoded, pair, use_utility=True)
                loss = F.binary_cross_entropy_with_logits(score, score.new_tensor([item['label']])) / len(items)
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
                if stage == 'teacher':
                    score = model.score_pairs(encoded, pair, use_utility=False)
                    loss = loss + F.binary_cross_entropy_with_logits(score, score.new_tensor([item['label']])) / len(items)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite candidate objective')
            loss.backward()
            total += float(loss.detach().cpu())
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
    return dict(stage=stage, routing=routing, loss=total, valid_counts=counts.cpu().tolist(),
                gradient_norm=float(norm.detach().cpu()), sampled_pairs=len(items), used_pairs=used,
                removed_observations=sum(sum(x['removed']) for x in items))


def training_configuration(config):
    required = {'seed', 'batch_size', 'teacher_updates', 'utility_updates', 'adapt_updates',
                'learning_rate', 'gradient_clip', 'endpoint_fraction', 'utility_epsilon',
                'error_scale_policy', 'routes', 'max_seconds', 'batch_tracks', 'pair_batch_size'}
    if set(config) != required:
        raise ValueError('candidate training config must explicitly contain ' + ', '.join(sorted(required)))
    cfg = dict(config)
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
    if (not isinstance(cfg['routes'], list) or not cfg['routes'] or len(set(cfg['routes'])) != len(cfg['routes'])
            or any(x not in ('plain', 'identity', 'utility', 'support', 'learned') for x in cfg['routes'])
            or not {'plain', 'utility'}.issubset(cfg['routes'])):
        raise ValueError('unique comparison routes must include plain and utility')
    return cfg


def _snapshot(model, objective, optimizer, normalizer, config, *, stage, route, updates):
    def cpu(state):
        if isinstance(state, torch.Tensor):
            return state.detach().cpu().clone()
        if isinstance(state, dict):
            return {k: cpu(v) for k, v in state.items()}
        if isinstance(state, (list, tuple)):
            return type(state)(cpu(v) for v in state)
        return state
    return dict(format='cooperative-physical-v1', model_config=asdict(model.config),
                model_state=cpu(model.state_dict()), objective_state=cpu(objective.state_dict()),
                optimizer_state=cpu(optimizer.state_dict()), normalizer=normalizer,
                training=config, stage=stage, route=route, completed_updates=updates,
                selection_policy='predetermined stage endpoint; no fusion selection on unchanged F1',
                resume_supported=False)


def load_cooperative_checkpoint(path, *, device):
    """Restore an inference snapshot; this is not an optimizer-resume API."""
    from .cooperative import CooperativeConfig, CooperativeModel
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != 'cooperative-physical-v1':
        raise ValueError('unsupported cooperative checkpoint')
    model = CooperativeModel(CooperativeConfig(**payload['model_config']), payload['normalizer'])
    model.load_state_dict(payload['model_state'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite model checkpoint')
    return model.to(resolve_device(device)).eval(), payload


def train_cooperative(train_scenes, validation_scenes, model_config, training_config, output_dir,
                      *, device, initial_elapsed_seconds=0., check=None):
    """Teacher -> frozen utility -> matched fixed-identity fusion arms.

    The first experiment intentionally excludes changing identity. Full-model
    identity modules/step are available, but not silently added to this fit.
    The configured wall budget includes all routes, evaluation and initial data
    preparation. No resume/retry or implicit training schedule is provided.
    """
    started = time.monotonic()
    from .cooperative import CooperativeModel
    from .cooperative_evaluation import evaluate_cooperative, score_cooperative
    from .metrics import select_threshold
    cfg = training_configuration(training_config)
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
    from .complete_training import _scenes
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
               scope='fixed identity candidate development comparison', test_opened=False,
               scene_contract=contract)
    runtime.write_json(output / 'run.json', run)
    try:
        bounded()
        torch.manual_seed(cfg['seed'])
        normalizer = fit_normalizer(train_scenes)
        packed = [normalize_scene(scene, normalizer) for scene in train_scenes]
        model = CooperativeModel(model_config, normalizer).to(device)
        objective = FusionLoss(position_loss_mode='standard_smooth_l1').to(device)
        teacher = None
        def fit_stage(target_model, target_objective, stage, steps, seed, route='plain'):
            parameters = stage_parameters(target_model, stage, routing=route)
            target_objective.requires_grad_(stage in ('teacher', 'adapt'))
            if stage in ('teacher', 'adapt'):
                parameters += list(target_objective.parameters())
            optimizer = torch.optim.Adam(parameters, lr=cfg['learning_rate'])
            sampler = BalancedScenePairSampler(train_scenes, seed)
            view_rng = np.random.default_rng(seed + 100)
            torch.manual_seed(seed)
            for step in range(steps):
                bounded()
                items, population = _batch(train_scenes, packed, sampler, cfg['batch_size'], device,
                    view_rng, cfg['endpoint_fraction'], model_config.max_observations)
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
                path = output / (stage + ('-' + route if stage == 'adapt' else '') + '.pt')
                temporary = path.with_suffix('.tmp')
                torch.save(_snapshot(target_model, target_objective, optimizer, normalizer, cfg,
                                     stage=stage, route=route, updates=step+1), temporary)
                temporary.replace(path)
                runtime.write_json(output / 'history.json', history)
            return sampler.describe()
        fit_stage(model, objective, 'teacher', cfg['teacher_updates'], cfg['seed'])
        teacher = copy.deepcopy(model).eval().requires_grad_(False)
        fit_stage(model, objective, 'utility', cfg['utility_updates'], cfg['seed']+1)
        # Base identity never reads the newly learned utility branch.
        records = score_cooperative(model, normalizer, validation_scenes, device=device,
            batch_tracks=cfg['batch_tracks'], pair_batch_size=cfg['pair_batch_size'],
            use_utility_identity=False, check=bounded)
        threshold = select_threshold(records)['threshold']
        for route in cfg['routes']:
            bounded()
            arm, loss = copy.deepcopy(model), copy.deepcopy(objective)
            fit_stage(arm, loss, 'adapt', cfg['adapt_updates'], cfg['seed']+2, route)
            report = evaluate_cooperative(arm, normalizer, validation_scenes, device=device,
                batch_tracks=cfg['batch_tracks'], pair_batch_size=cfg['pair_batch_size'],
                select_on_validation=False, threshold=threshold, routing=route,
                use_utility_identity=False, fixed_records=records, check=bounded)
            # Numpy score arrays remain in explicit NPZ caches, not JSON magic.
            cache = {}
            for index, record in enumerate(report.pop('records')):
                cache[f'scores_{index}'] = np.asarray(record['scores'])
                cache[f'pairs_{index}'] = np.asarray(record['pairs'])
            np.savez_compressed(output / (route + '-scores.npz'), **cache)
            runtime.write_json(output / (route + '-evaluation.json'), report)
            completed[route] = report
        bounded()
        run.update(status='COMPLETED', completed_routes=list(completed),
                   parameter_count=sum(p.numel() for p in model.parameters()), threshold=threshold,
                   error_scales=normalizer['std'], elapsed_seconds=initial_elapsed_seconds+time.monotonic()-started,
                   identity_policy='fixed s0 and one full-V threshold for every arm',
                   ordinary_gate_features='same frozen utility-trained query pool; fresh ordinary gate, no gain outputs',
                   controls_missing_for_utility_attribution=[x for x in ('identity', 'learned') if x not in cfg['routes']],
                   selection_policy='predetermined final updates', resume_supported=False)
        runtime.write_json(output / 'run.json', run)
        return run
    except BaseException as error:
        run.update(status='INTERRUPTED' if isinstance(error, (TimeoutError, InterruptedError, KeyboardInterrupt)) else 'FAILED',
                   error=f'{type(error).__name__}: {error}', completed_routes=list(completed),
                   elapsed_seconds=initial_elapsed_seconds+time.monotonic()-started, resume_supported=False)
        runtime.write_json(output / 'run.json', run)
        raise
