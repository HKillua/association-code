"""090 one additional fixed budget; exact original T batches, fresh Adam."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from association.models.frozen_position_adapter import (SCOPE, ROUTING, MODES,
    FrozenPositionAdapter, initialize_from_teacher, trainable_parameters)
from association.models.sequence_position import sequence_inputs, sequence_objectives
from association.training import sequence_position as sequence
from association.training.evidence_fusion import TRAIN_MODES, prepare_evidence_scenes, training_members
from association.utils.runtime import write_json


def source_configuration(value):
    original = copy.deepcopy(value)
    original.update(scope=sequence.SCOPE, routing_semantics=sequence.ROUTING_SEMANTICS)
    original['model'] = dict(backbone='transformer', hidden_size=64, num_layers=2, initial_gate=.95)
    return sequence.configuration(original)


def configuration(value):
    if (not isinstance(value, dict) or value.get('scope') != SCOPE
            or value.get('routing_semantics') != ROUTING or value.get('position_loss') != 'linear'
            or not isinstance(value.get('model'), dict)
            or set(value['model']) != {'mode', 'adapter_hidden_size'}
            or value['model']['mode'] not in MODES or type(value['model']['adapter_hidden_size']) is not int
            or value['model']['adapter_hidden_size'] != 32):
        raise ValueError('090 requires explicit mode, routing and linear objective')
    source_configuration(value)
    return value


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _exact(left, right):
    return set(left) == set(right) and all(left[k].dtype == right[k].dtype
        and torch.equal(left[k].detach().cpu(), right[k].detach().cpu()) for k in left)


def _save(path, model, optimizer, step, config, rng, source):
    torch.save(dict(scope=SCOPE, routing_semantics=ROUTING, loss_semantics=sequence.LOSS_SEMANTICS,
        configuration=config, step=step, actual_updates=step, model=model.state_dict(),
        optimizer=optimizer.state_dict(), numpy_rng=copy.deepcopy(rng.bit_generator.state),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        source_teacher=source, optimizer_initialization='fresh Adam; source optimizer never resumed'), path)


def load_checkpoint(path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = configuration(saved['configuration'])
    if (saved.get('scope') != SCOPE or saved.get('routing_semantics') != ROUTING
            or saved.get('loss_semantics') != sequence.LOSS_SEMANTICS
            or type(saved.get('step')) is not int or not 0 <= saved['step'] <= config['training']['updates']
            or type(saved.get('actual_updates')) is not int or saved['actual_updates'] != saved['step']
            or saved.get('optimizer_initialization') != 'fresh Adam; source optimizer never resumed'
            or any(k not in saved for k in ('numpy_rng', 'torch_rng', 'cuda_rng', 'source_teacher'))):
        raise ValueError('090 checkpoint scope/step/RNG/optimizer contract mismatch')
    source = saved['source_teacher']
    if (source.get('configuration') != source_configuration(config)
            or source.get('step') != config['training']['updates'] or not source.get('checkpoint')):
        raise ValueError('090 checkpoint requires declared complete strong source-final')
    model = FrozenPositionAdapter(**config['model']).to(device)
    model.load_state_dict(saved['model'], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('090 checkpoint parameters must be finite')
    # Validate source architecture and finite tensors even for the changing C arm.
    reference = sequence.SequencePosition(**source['configuration']['model'])
    reference.load_state_dict(source['model'], strict=True)
    if not all(torch.isfinite(p).all() for p in reference.parameters()):
        raise ValueError('090 source tensors must be finite')
    if (saved['step'] == 0 or model.mode != 'full_finetune') and not _exact(model.teacher.state_dict(), source['model']):
        raise ValueError('090 frozen/source-final teacher tensor mismatch')
    if saved['step'] == 0 and model.adapter is not None and (
            torch.count_nonzero(model.adapter[2].weight) or torch.count_nonzero(model.adapter[2].bias)):
        raise ValueError('090 adapter initial output must be identically zero')
    optimizer = saved['optimizer']; groups = optimizer.get('param_groups', []); entries = optimizer.get('state', {})
    if (len(groups) != 1 or groups[0].get('lr') != config['training']['learning_rate']
            or groups[0].get('weight_decay') != 0 or groups[0].get('betas') != (0.9, 0.999)
            or len(groups[0].get('params', [])) != len(trainable_parameters(model))):
        raise ValueError('090 optimizer trainable parameter contract mismatch')
    if ((saved['step'] == 0 and entries) or (saved['step'] > 0 and (
            set(entries) != set(groups[0]['params']) or any(float(e.get('step', -1)) != saved['step'] for e in entries.values())))):
        raise ValueError('090 optimizer successful update count mismatch')
    if any(not torch.isfinite(v).all() for entry in entries.values() for v in entry.values() if isinstance(v, torch.Tensor)):
        raise ValueError('090 optimizer state must remain finite')
    return model, saved


def train_frozen_position_adapter(train_raw, config, output, *, paired_reference_fit, device='cpu', check=None):
    config = copy.deepcopy(configuration(config)); train_raw = list(train_raw)
    scene_ids = [str(s['scene_id']) for s in train_raw]
    if not train_raw or len(set(scene_ids)) != len(scene_ids) or any(s.get('split') != 'T_core' for s in train_raw):
        raise ValueError('090 requires complete distinct T_core scenes')
    old_config = source_configuration(config)
    reference = sequence._reference_pair(paired_reference_fit, old_config, scene_ids)
    old_model, old = sequence.load_checkpoint(Path(paired_reference_fit)/'final.pt')
    if old['configuration'] != old_config or old['step'] != config['training']['updates']:
        raise ValueError('090 source-final configuration/budget mismatch')
    source = dict(configuration=old_config, step=old['step'],
        checkpoint=str((Path(paired_reference_fit)/'final.pt').resolve()), model=_cpu_state(old_model))
    del old_model, old
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); training = config['training']
    torch.manual_seed(training['seed']); rng = np.random.default_rng(training['seed'])
    if rng.bit_generator.state != reference['initial']['numpy_rng']:
        raise ValueError('090 original initial sampler RNG must match')
    model = FrozenPositionAdapter(**config['model'])
    initialize_from_teacher(model, source['model'], branch_seed=training['seed']+90000)
    model.to(device); parameters = trainable_parameters(model)
    optimizer = torch.optim.Adam(parameters, lr=training['learning_rate'])
    cuda = torch.device(device).type == 'cuda'
    if cuda: torch.cuda.reset_peak_memory_stats(device)
    audit = dict(successful_updates=0, sampled_anchors=0, valid_position_points=0,
        nonzero_gradient_updates=0, scenes={}, anchors={},
        modes={mode: dict(updates=0, anchors=0, valid_position_points=0) for mode in TRAIN_MODES})
    state = dict(status='RUNNING', scope=SCOPE, configuration=config, optimizer_updates=0,
        training_scene_ids=scene_ids, test_opened=False, final_checkpoint=None, exposure=audit,
        parameter_count=sum(p.numel() for p in model.parameters()), trainable_parameter_count=sum(p.numel() for p in parameters),
        teacher_parameter_count=sum(p.numel() for p in model.teacher.parameters()),
        adapter_parameter_count=sum(p.numel() for p in model.adapter.parameters()) if model.adapter is not None else 0,
        source_teacher_checkpoint=source['checkpoint'], source_teacher_updates=source['step'],
        initialization='all source-final tensors; zero new output; fresh Adam, original sampler replay',
        total_stage_updates=source['step']+training['updates'], teacher_frozen=model.mode != 'full_finetune',
        branch_seed=training['seed']+90000, prediction_scope=config['model']['mode'],
        loss_semantics=sequence.LOSS_SEMANTICS, routing_semantics=ROUTING,
        gradient_clipping=dict(clipped_updates=0, successful_update_fraction=0.),
        route_exposure=dict(all_queries=0, anchor_supported_queries=0, valid_queries=0, valid_anchor_supported_queries=0,
            increment_allowed_queries=0, valid_increment_allowed_queries=0),
        paired_control=dict(status='PENDING', fit=str(Path(paired_reference_fit).resolve()),
            source_final_parameters_exact=True, initial_sampler_rng_exact=True, actual_successful_batches_matched=0,
            sampling_exposure_exact=False, teacher_unchanged=None), advisory_reached=False)
    write_json(output/'run.json', state); rows = []
    try:
        _save(output/'step-0.pt', model, optimizer, 0, config, rng, source)
        train = prepare_evidence_scenes(train_raw, check=check, motion_observation_policy='canonical_all')
        if any(not len(s['identities']) for s in train):
            raise ValueError('090 cannot silently drop empty training scenes')
        state['preparation_seconds'] = time.monotonic()-started; fit_started = time.monotonic()
        for step in range(1, training['updates']+1):
            if check: check()
            model.train(); optimizer.zero_grad(set_to_none=True)
            mode = TRAIN_MODES[(step-1) % len(TRAIN_MODES)]
            batch, diagnostics, linear, logarithmic, routes = [], [], [], [], []
            for _ in range(training['batch_size']):
                scene = train[int(rng.integers(len(train)))]; anchor = int(rng.integers(len(scene['identities'])))
                members = training_members(scene, anchor, mode)
                inputs = sequence_inputs(scene['motion'], members, device=device)
                support = scene['motion']['support']; anchor_support = torch.as_tensor(support[anchor], dtype=torch.bool, device=device)
                result = model(*inputs, anchor_support)
                target = torch.as_tensor(scene['target'][anchor], dtype=torch.float64, device=device)
                valid = torch.as_tensor(scene['valid'][anchor], dtype=torch.bool, device=device)
                losses, diag = sequence_objectives(result, target, valid)
                if not torch.isfinite(losses['linear']): raise FloatingPointError('090 nonfinite objective')
                (losses['linear']/training['batch_size']).backward()
                linear.append(float(losses['linear'].detach())); logarithmic.append(float(losses['log1p'].detach()))
                diagnostics.append(diag); valid_positions = valid[:, :2].all(-1).cpu().numpy()
                eligible = ~support[anchor] & support[members[1:]].any(0)
                batch.append(dict(scene_id=str(scene['scene_id']), anchor_index=anchor, node_id=str(scene['node_ids'][anchor]),
                    members=[int(x) for x in members], valid_position_points=diag['valid_position_points'],
                    eligible_query_points=int(eligible.sum()), eligible_valid_position_points=int((eligible & valid_positions).sum())))
                allowed = (result['update_allowed'].detach().cpu().numpy() if model.adapter is not None
                           else np.ones(len(anchor_support), dtype=bool))
                expected_allowed = ~support[anchor] if model.mode == 'anchor_unsupported' else np.ones(len(anchor_support), dtype=bool)
                if not np.array_equal(allowed, expected_allowed):
                    raise ValueError('090 actual training route differs from declared visible scope')
                routes.append(dict(all_queries=len(anchor_support), anchor_supported_queries=int(support[anchor].sum()),
                    valid_queries=int(valid_positions.sum()), valid_anchor_supported_queries=int((valid_positions & support[anchor]).sum()),
                    increment_allowed_queries=int(allowed.sum()), valid_increment_allowed_queries=int((valid_positions & allowed).sum())))
            old_row = reference['rows'][step-1]
            if old_row['mode'] != mode or old_row['batch'] != batch:
                raise ValueError(f'090 original actual batch differs before update {step}')
            if model.mode != 'full_finetune' and any(p.grad is not None for p in model.teacher.parameters()):
                raise ValueError('090 frozen teacher acquired gradient')
            norm = torch.nn.utils.clip_grad_norm_(parameters, training['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); state['optimizer_updates'] = step
            count = sum(d['valid_position_points'] for d in diagnostics)
            audit['successful_updates'] += 1; audit['sampled_anchors'] += len(batch); audit['valid_position_points'] += count
            audit['nonzero_gradient_updates'] += int(norm > 0)
            audit['modes'][mode]['updates'] += 1; audit['modes'][mode]['anchors'] += len(batch)
            audit['modes'][mode]['valid_position_points'] += count
            for sample in batch:
                for key, name in (('scenes', sample['scene_id']), ('anchors', sample['node_id'])):
                    audit[key][name] = audit[key].get(name, 0)+1
            for route in routes:
                for key, value in route.items(): state['route_exposure'][key] += value
            state['paired_control']['actual_successful_batches_matched'] = step
            clipped = bool(norm > training['gradient_clip'])
            state['gradient_clipping']['clipped_updates'] += int(clipped)
            state['gradient_clipping']['successful_update_fraction'] = state['gradient_clipping']['clipped_updates']/step
            if not all(torch.isfinite(p).all() for p in model.parameters()): raise FloatingPointError('090 nonfinite parameter')
            elapsed = time.monotonic()-started
            if elapsed >= training['advisory_seconds'] and not state['advisory_reached']:
                print('ADVISORY: 090 fixed fit continues; no retry or extension.', flush=True)
            state.update(elapsed_seconds=elapsed, training_seconds=time.monotonic()-fit_started,
                advisory_reached=elapsed >= training['advisory_seconds'],
                peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0)
            row = dict(step=step, mode=mode, batch=batch, loss=float(np.mean(linear)),
                objectives=dict(anchor_mean_linear=float(np.mean(linear)), anchor_mean_log1p=float(np.mean(logarithmic)),
                    valid_position_points=count, point_mean_linear=sum(d['linear_sum'] for d in diagnostics)/count if count else None,
                    point_mean_log1p=sum(d['log1p_sum'] for d in diagnostics)/count if count else None),
                gradient_norm=float(norm), gradient_clipped=clipped,
                gradient_clip_scale=min(1., training['gradient_clip']/(float(norm)+1e-6)), route_batch=routes)
            rows.append(row)
            with (output/'training.jsonl').open('a') as handle: handle.write(json.dumps(row, allow_nan=False)+'\n')
            write_json(output/'run.json', state)
        if sequence._sampling_exposure(audit) != sequence._sampling_exposure(reference['exposure']):
            raise ValueError('090 actual successful exposure differs from original')
        unchanged = _exact(model.teacher.state_dict(), source['model'])
        if model.mode != 'full_finetune' and not unchanged: raise ValueError('090 teacher changed during adapter training')
        if model.mode == 'full_finetune' and unchanged: raise ValueError('090C full-model control made no parameter update')
        state['paired_control'].update(status='COMPLETED', sampling_exposure_exact=True, teacher_unchanged=unchanged)
        _save(output/'final.pt', model, optimizer, training['updates'], config, rng, source)
        write_json(output/'training-windows.json', sequence._windows(rows))
        state.update(status='COMPLETED', final_checkpoint='final.pt', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return state
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', error=f'{type(error).__name__}: {error}')
        state['paired_control']['status'] = state['status']; write_json(output/'run.json', state)
        write_json(output/'training-windows.json', sequence._windows(rows))
        try:
            _save(output/'interrupted.pt', model, optimizer, state['optimizer_updates'], config, rng, source)
        except BaseException as save_error:
            state['recovery_checkpoint_error'] = f'{type(save_error).__name__}: {save_error}'
            write_json(output/'run.json', state)
        raise
