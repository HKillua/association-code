"""098: source-final warm start and a fixed, exactly replayed additional budget.

All parameters train with a fresh Adam. The original Transformer and the new
query-set branch share the same position objective, samples and physical prior.
The older step-zero helper experiments keep their independent contracts.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from association.models.helper_sequence import helper_sequence_inputs
from association.models.query_relation import (
    MODES, ROUTING_SEMANTICS, SCOPE, QueryRelationPosition, initialize_from_reference,
)
from association.models.sequence_position import sequence_objectives
from association.training import sequence_position as sequence
from association.training.evidence_fusion import (
    TRAIN_MODES, prepare_evidence_scenes, training_members,
)
from association.utils.runtime import write_json

OPTIMIZER_INITIALIZATION = 'fresh Adam; source optimizer never resumed'
HEARTBEAT_INTERVAL = 100


def source_configuration(value):
    original = copy.deepcopy(value)
    original.update(scope=sequence.SCOPE, routing_semantics=sequence.ROUTING_SEMANTICS)
    original['model'] = dict(backbone='transformer', hidden_size=64, num_layers=2, initial_gate=.95)
    return sequence.configuration(original)


def configuration(value):
    model = value.get('model') if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get('scope') != SCOPE
            or value.get('routing_semantics') != ROUTING_SEMANTICS
            or value.get('position_loss') != 'linear' or not isinstance(model, dict)
            or set(model) != {'mode', 'hidden_size', 'num_layers', 'initial_gate'}
            or model['mode'] not in MODES or type(model['hidden_size']) is not int
            or model['hidden_size'] != 64 or type(model['num_layers']) is not int
            or model['num_layers'] != 2 or isinstance(model['initial_gate'], bool)
            or model['initial_gate'] != .95):
        raise ValueError('098 requires declared generic/relation mode, fixed architecture and linear objective')
    source_configuration(value)
    return value


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _common_exact(model, source):
    current = model.state_dict()
    return set(source).issubset(current) and all(
        current[key].dtype == value.dtype
        and torch.equal(current[key].detach().cpu(), value.detach().cpu())
        for key, value in source.items())


def _load_exact_state(model, state, label):
    expected = model.state_dict()
    if (not isinstance(state, dict) or set(state) != set(expected)
            or any(not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                   or value.dtype != expected[key].dtype or not torch.isfinite(value).all()
                   for key, value in state.items())):
        raise ValueError(f'098 {label} requires exact finite tensor keys, shapes and dtypes')
    model.load_state_dict(state, strict=True)


def _validate_rng(saved):
    if any(key not in saved for key in ('numpy_rng', 'torch_rng', 'cuda_rng')):
        raise ValueError('098 checkpoint requires complete sampler and Torch RNG states')
    try:
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(saved['numpy_rng'])
        generator = torch.Generator(device='cpu')
        generator.set_state(saved['torch_rng'])
    except (ValueError, TypeError, RuntimeError, KeyError) as error:
        raise ValueError('098 checkpoint RNG state is invalid') from error
    cuda = saved['cuda_rng']
    if cuda is not None and (not isinstance(cuda, list) or any(
            not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
            or state.ndim != 1 or state.numel() == 0 for state in cuda)):
        raise ValueError('098 checkpoint CUDA RNG states are invalid')


def _save(path, model, optimizer, step, config, rng, source):
    torch.save(dict(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS,
        loss_semantics=sequence.LOSS_SEMANTICS, motion_observation_policy='canonical_all',
        configuration=config, step=step, actual_updates=step, model=model.state_dict(),
        optimizer=optimizer.state_dict(), numpy_rng=copy.deepcopy(rng.bit_generator.state),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        source_teacher=source, branch_seed=config['training']['seed']+98000,
        optimizer_initialization=OPTIMIZER_INITIALIZATION), path)


def _validate_optimizer(saved, config, model):
    optimizer = saved.get('optimizer', {})
    groups, entries = optimizer.get('param_groups', []), optimizer.get('state', {})
    parameters = list(model.parameters())
    if (len(groups) != 1 or groups[0].get('lr') != config['training']['learning_rate']
            or groups[0].get('weight_decay') != 0 or groups[0].get('betas') != (0.9, 0.999)
            or groups[0].get('eps') != 1e-8 or groups[0].get('amsgrad') is not False
            or len(groups[0].get('params', [])) != len(parameters)
            or len(set(groups[0]['params'])) != len(parameters)):
        raise ValueError('098 optimizer must include all model parameters in one fresh-Adam group')
    ids = groups[0]['params']
    if ((saved['step'] == 0 and entries) or (saved['step'] > 0
            and (set(entries) != set(ids) or any(
                float(entry.get('step', -1)) != saved['step'] for entry in entries.values())))):
        raise ValueError('098 optimizer per-parameter successful update count mismatch')
    for key, parameter in zip(ids, parameters):
        if key not in entries:
            continue
        entry = entries[key]
        for moment in ('exp_avg', 'exp_avg_sq'):
            tensor = entry.get(moment)
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape
                    or tensor.dtype != parameter.dtype or not torch.isfinite(tensor).all()):
                raise ValueError('098 optimizer moment shape/dtype/finite contract mismatch')
        if any(not torch.isfinite(value).all() for value in entry.values()
               if isinstance(value, torch.Tensor)):
            raise ValueError('098 optimizer state must remain finite')


def load_checkpoint(path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get('configuration'), dict):
        raise ValueError('098 checkpoint configuration required')
    config = configuration(saved['configuration'])
    if (saved.get('scope') != SCOPE or saved.get('routing_semantics') != ROUTING_SEMANTICS
            or saved.get('loss_semantics') != sequence.LOSS_SEMANTICS
            or saved.get('motion_observation_policy') != 'canonical_all'
            or type(saved.get('step')) is not int
            or not 0 <= saved['step'] <= config['training']['updates']
            or type(saved.get('actual_updates')) is not int
            or saved['actual_updates'] != saved['step']
            or saved.get('optimizer_initialization') != OPTIMIZER_INITIALIZATION
            or saved.get('branch_seed') != config['training']['seed']+98000):
        raise ValueError('098 checkpoint scope/update/optimizer contract mismatch')
    _validate_rng(saved)
    source = saved.get('source_teacher')
    if (not isinstance(source, dict) or source.get('configuration') != source_configuration(config)
            or type(source.get('step')) is not int
            or source['step'] != config['training']['updates']
            or not isinstance(source.get('checkpoint'), str) or not source['checkpoint']
            or not isinstance(source.get('model'), dict)):
        raise ValueError('098 checkpoint requires the complete declared source-final model')
    reference = sequence.SequencePosition(**source['configuration']['model'])
    _load_exact_state(reference, source['model'], 'source-final')
    model = QueryRelationPosition(**config['model']).to(device)
    _load_exact_state(model, saved.get('model'), 'model')
    if any(not parameter.requires_grad or not torch.isfinite(parameter).all()
           for parameter in model.parameters()):
        raise ValueError('098 requires finite, fully trainable model parameters')
    if saved['step'] == 0 and (not _common_exact(model, source['model'])
            or any(torch.count_nonzero(parameter) for parameter in model.branch_projection.parameters())):
        raise ValueError('098 step-zero common source-final or zero branch projection mismatch')
    if saved['step'] == 0:
        expected = QueryRelationPosition(**config['model'])
        initialize_from_reference(expected, source['model'], branch_seed=saved['branch_seed'])
        if not _common_exact(model, expected.state_dict()):
            raise ValueError('098 step-zero new branch differs from its declared initialization seed')
    _validate_optimizer(saved, config, model)
    return model, saved


def _heartbeat(output, state, *, phase):
    """Persist bounded live progress; large cumulative scene maps stay in RAM."""
    exposure = state['exposure']
    write_json(output/'heartbeat.json', dict(status=state['status'], phase=phase,
        optimizer_updates=state['optimizer_updates'],
        sampled_anchors=exposure['sampled_anchors'],
        valid_position_points=exposure['valid_position_points'],
        elapsed_seconds=state.get('elapsed_seconds', 0.),
        training_seconds=state.get('training_seconds', 0.),
        advisory_reached=state['advisory_reached']))


def train_query_relation(train_raw, config, output, *, paired_reference_fit,
                         device='cpu', check=None):
    config = copy.deepcopy(configuration(config))
    train_raw = list(train_raw)
    scene_ids = [str(scene['scene_id']) for scene in train_raw]
    if (not train_raw or len(set(scene_ids)) != len(scene_ids)
            or any(scene.get('split') != 'T_core' for scene in train_raw)):
        raise ValueError('098 requires nonempty distinct T_core scenes only')
    old_config = source_configuration(config)
    reference = sequence._reference_pair(paired_reference_fit, old_config, scene_ids)
    old_model, old = sequence.load_checkpoint(Path(paired_reference_fit)/'final.pt')
    if old['configuration'] != old_config or old['step'] != config['training']['updates']:
        raise ValueError('098 source-final configuration and completed budget must match')
    _load_exact_state(old_model, old['model'], 'source-final')
    _validate_optimizer(old, old_config, old_model)
    _validate_rng(old)
    _validate_rng(reference['initial'])
    source = dict(configuration=old_config, step=old['step'],
        checkpoint=str((Path(paired_reference_fit)/'final.pt').resolve()), model=_cpu_state(old_model))
    source_final_sampler = copy.deepcopy(old['numpy_rng'])
    del old_model, old
    training = config['training']
    torch.manual_seed(training['seed'])
    rng = np.random.default_rng(training['seed'])
    if rng.bit_generator.state != reference['initial']['numpy_rng']:
        raise ValueError('098 original initial sampler RNG must match')
    model = QueryRelationPosition(**config['model'])
    initialize_from_reference(model, source['model'], branch_seed=training['seed']+98000)
    if (not _common_exact(model, source['model'])
            or any(torch.count_nonzero(p) for p in model.branch_projection.parameters())):
        raise ValueError('098 must begin at source final with zero added projection')
    model.to(device)
    parameters = list(model.parameters())
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError('098 trains all inherited and new parameters')
    optimizer = torch.optim.Adam(parameters, lr=training['learning_rate'])
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    audit = dict(successful_updates=0, sampled_anchors=0, valid_position_points=0,
        nonzero_gradient_updates=0, scenes={}, anchors={},
        modes={mode: dict(updates=0, anchors=0, valid_position_points=0) for mode in TRAIN_MODES})
    branch_prefixes = sorted({name.split('.')[0] for name, _ in model.named_parameters()
                              if name not in source['model']})
    state = dict(status='RUNNING', scope=SCOPE, configuration=config,
        routing_semantics=ROUTING_SEMANTICS, loss_semantics=sequence.LOSS_SEMANTICS,
        motion_observation_policy='canonical_all', optimizer_updates=0,
        training_scene_ids=scene_ids, test_opened=False, final_checkpoint=None,
        device=str(device), exposure=audit,
        parameter_count=sum(p.numel() for p in parameters),
        trainable_parameter_count=sum(p.numel() for p in parameters),
        source_teacher_checkpoint=source['checkpoint'], source_teacher_updates=source['step'],
        total_stage_updates=source['step']+training['updates'], teacher_frozen=False,
        branch_seed=training['seed']+98000,
        initialization='source-final common tensors; zero added projection; fresh Adam; original sampler replay',
        prediction_scope='position only; all queries retained; speed/course remain ordinary Mean',
        selection_policy='fixed final step; no validation or intermediate selection in training',
        helper_support_exposure={population: {name: 0 for name in ('zero', 'one', 'multiple')}
            for population in ('all_queries', 'valid_position_queries')},
        helper_group_exposure=dict(sampled_anchors=0, any_multi_queries=0, any_valid_multi_queries=0),
        helper_gradients={name+'_nonzero_updates': 0 for name in branch_prefixes},
        gradient_clipping=dict(clipped_updates=0, successful_update_fraction=0.),
        paired_control=dict(status='PENDING', fit=str(Path(paired_reference_fit).resolve()),
            source_final_parameters_exact=True, added_projection_zero=True,
            initial_sampler_rng_exact=True, actual_successful_batches_matched=0,
            final_sampler_rng_exact=False, sampling_exposure_exact=False, teacher_unchanged=None),
        advisory_reached=False, heartbeat_interval_updates=HEARTBEAT_INTERVAL)
    write_json(output/'run.json', state)
    _heartbeat(output, state, phase='preparation')
    rows = []
    try:
        _save(output/'step-0.pt', model, optimizer, 0, config, rng, source)
        train = prepare_evidence_scenes(train_raw, check=check, motion_observation_policy='canonical_all')
        if any(not len(scene['identities']) for scene in train):
            raise ValueError('098 cannot silently drop empty training scenes')
        state['preparation_seconds'] = time.monotonic()-started
        state['elapsed_seconds'] = state['preparation_seconds']
        fit_started = time.monotonic()
        _heartbeat(output, state, phase='training')
        with (output/'training.jsonl').open('x', buffering=1) as history:
            for step in range(1, training['updates']+1):
                if check:
                    check()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                mode = TRAIN_MODES[(step-1) % len(TRAIN_MODES)]
                batch, diagnostics, linear, logarithmic, helper_batch, group_batch = [], [], [], [], [], []
                for _ in range(training['batch_size']):
                    scene = train[int(rng.integers(len(train)))]
                    anchor = int(rng.integers(len(scene['identities'])))
                    members = training_members(scene, anchor, mode)
                    inputs = helper_sequence_inputs(scene['motion'], members, device=device)
                    result = model(*inputs)
                    target = torch.as_tensor(scene['target'][anchor], dtype=torch.float64, device=device)
                    valid = torch.as_tensor(scene['valid'][anchor], dtype=torch.bool, device=device)
                    losses, diag = sequence_objectives(result, target, valid)
                    if not torch.isfinite(losses['linear']):
                        raise FloatingPointError('098 nonfinite position objective')
                    (losses['linear']/training['batch_size']).backward()
                    diagnostics.append(diag)
                    linear.append(float(losses['linear'].detach()))
                    logarithmic.append(float(losses['log1p'].detach()))
                    support = scene['motion']['support']
                    valid_positions = valid[:, :2].all(-1).cpu().numpy()
                    counts = support[members[1:]].sum(0)
                    if not np.array_equal(result['helper_support_count'].detach().cpu().numpy(), counts):
                        raise ValueError('098 model helper support differs from visible support')
                    eligible = ~support[anchor] & (counts > 0)
                    batch.append(dict(scene_id=str(scene['scene_id']), anchor_index=anchor,
                        node_id=str(scene['node_ids'][anchor]), members=[int(x) for x in members],
                        valid_position_points=diag['valid_position_points'],
                        eligible_query_points=int(eligible.sum()),
                        eligible_valid_position_points=int((eligible & valid_positions).sum())))
                    helper_batch.append({population: {name: int((selection & mask).sum())
                        for name, selection in (('zero', counts == 0), ('one', counts == 1), ('multiple', counts >= 2))}
                        for population, mask in (('all_queries', np.ones_like(counts, bool)),
                                                 ('valid_position_queries', valid_positions))})
                    group_batch.append(dict(sampled_anchors=1,
                        any_multi_queries=int(np.any(counts >= 2)),
                        any_valid_multi_queries=int(np.any((counts >= 2) & valid_positions))))
                old_row = reference['rows'][step-1]
                if old_row['mode'] != mode or old_row['batch'] != batch:
                    raise ValueError(f'098 original actual batch differs before update {step}')
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError('098 every trainable parameter requires a finite gradient')
                norm = torch.nn.utils.clip_grad_norm_(parameters, training['gradient_clip'], error_if_nonfinite=True)
                branch_grad = {prefix+'_nonzero_updates': any(
                    bool(torch.count_nonzero(p.grad)) for name, p in model.named_parameters()
                    if name.startswith(prefix+'.')) for prefix in branch_prefixes}
                optimizer.step()
                # Only an optimizer operation that returned commits successful exposure.
                state['optimizer_updates'] = step
                count = sum(diag['valid_position_points'] for diag in diagnostics)
                audit['successful_updates'] += 1
                audit['sampled_anchors'] += len(batch)
                audit['valid_position_points'] += count
                audit['nonzero_gradient_updates'] += int(norm > 0)
                audit['modes'][mode]['updates'] += 1
                audit['modes'][mode]['anchors'] += len(batch)
                audit['modes'][mode]['valid_position_points'] += count
                for sample in batch:
                    for key, name in (('scenes', sample['scene_id']), ('anchors', sample['node_id'])):
                        audit[key][name] = audit[key].get(name, 0)+1
                for helper_counts, group_counts in zip(helper_batch, group_batch):
                    for population, categories in helper_counts.items():
                        for category, number in categories.items():
                            state['helper_support_exposure'][population][category] += number
                    for key, number in group_counts.items():
                        state['helper_group_exposure'][key] += number
                for key, nonzero in branch_grad.items():
                    state['helper_gradients'][key] += int(nonzero)
                state['paired_control']['actual_successful_batches_matched'] = step
                clipped = bool(norm > training['gradient_clip'])
                state['gradient_clipping']['clipped_updates'] += int(clipped)
                state['gradient_clipping']['successful_update_fraction'] = state['gradient_clipping']['clipped_updates']/step
                elapsed = time.monotonic()-started
                if elapsed >= training['advisory_seconds'] and not state['advisory_reached']:
                    print('ADVISORY: 098 fixed fit continues; no retry or extension.', flush=True)
                state.update(elapsed_seconds=elapsed, training_seconds=time.monotonic()-fit_started,
                    advisory_reached=elapsed >= training['advisory_seconds'],
                    peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0)
                row = dict(step=step, mode=mode, batch=batch, loss=float(np.mean(linear)),
                    objectives=dict(anchor_mean_linear=float(np.mean(linear)),
                        anchor_mean_log1p=float(np.mean(logarithmic)), valid_position_points=count,
                        point_mean_linear=sum(diag['linear_sum'] for diag in diagnostics)/count if count else None,
                        point_mean_log1p=sum(diag['log1p_sum'] for diag in diagnostics)/count if count else None),
                    gradient_norm=float(norm), gradient_clipped=clipped,
                    gradient_clip_scale=min(1., training['gradient_clip']/(float(norm)+1e-6)),
                    helper_support_batch=helper_batch, helper_group_batch=group_batch,
                    helper_gradients=branch_grad)
                rows.append(row)
                history.write(json.dumps(row, allow_nan=False)+'\n')
                if not all(torch.isfinite(p).all() for p in parameters):
                    raise FloatingPointError('098 nonfinite parameter after optimizer update')
                if step % HEARTBEAT_INTERVAL == 0:
                    _heartbeat(output, state, phase='training')
        if sequence._sampling_exposure(audit) != sequence._sampling_exposure(reference['exposure']):
            raise ValueError('098 actual successful exposure differs from original')
        if rng.bit_generator.state != source_final_sampler:
            raise ValueError('098 final sampler RNG differs from original replay')
        unchanged = _common_exact(model, source['model'])
        if unchanged:
            raise ValueError('098 full-model training made no inherited parameter update')
        state['paired_control'].update(status='COMPLETED', sampling_exposure_exact=True,
            final_sampler_rng_exact=True, teacher_unchanged=False)
        _save(output/'final.pt', model, optimizer, training['updates'], config, rng, source)
        write_json(output/'training-windows.json', sequence._windows(rows))
        state.update(status='COMPLETED', final_checkpoint='final.pt',
            elapsed_seconds=time.monotonic()-started, trained_scene_count=len(audit['scenes']))
        write_json(output/'run.json', state)
        _heartbeat(output, state, phase='completed')
        return state
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        state['paired_control']['status'] = state['status']
        write_json(output/'run.json', state)
        write_json(output/'training-windows.json', sequence._windows(rows))
        _heartbeat(output, state, phase=state['status'].lower())
        try:
            _save(output/'interrupted.pt', model, optimizer, state['optimizer_updates'], config, rng, source)
        except BaseException as save_error:
            state['recovery_checkpoint_error'] = f'{type(save_error).__name__}: {save_error}'
            write_json(output/'run.json', state)
        raise
