"""Bounded T-only generic sequence position baselines; no V model selection."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from association.models.sequence_position import (BACKBONES, ROUTING_SEMANTICS, SCOPE,
    SequencePosition, sequence_inputs, sequence_objectives)
from association.training.evidence_fusion import TRAIN_MODES, prepare_evidence_scenes, training_members
from association.utils.runtime import write_json

LOSS_SEMANTICS = 'mean_valid_position_linear_haversine_m_div_1000_all_queries_no_auxiliary'


def configuration(value):
    expected = {'scope', 'routing_semantics', 'position_loss', 'data', 'motion_observation_policy',
                'train_splits', 'model', 'training'}
    if (not isinstance(value, dict) or set(value) != expected or value['scope'] != SCOPE
            or value['routing_semantics'] != ROUTING_SEMANTICS or value['position_loss'] != 'linear'
            or value['motion_observation_policy'] != 'canonical_all' or value['train_splits'] != ['T_core']
            or value['data'] != dict(length=64, max_observations=128, time_unit_seconds=20.,
                                     reference_longitude_policy='shortest_arc')):
        raise ValueError('explicit sequence position scope, linear loss and Level1 data contract required')
    model = value['model']
    if (not isinstance(model, dict) or set(model) != {'backbone', 'hidden_size', 'num_layers', 'initial_gate'}
            or model['backbone'] not in BACKBONES or type(model['hidden_size']) is not int
            or model['hidden_size'] != 64 or type(model['num_layers']) is not int or model['num_layers'] != 2
            or isinstance(model['initial_gate'], bool) or model['initial_gate'] != .95):
        raise ValueError('sequence baseline architecture is fixed apart from the declared backbone')
    training = value['training']
    if not isinstance(training, dict) or set(training) != {
            'seed', 'updates', 'batch_size', 'learning_rate', 'gradient_clip', 'advisory_seconds'}:
        raise ValueError('unexpected sequence position training configuration')
    for key in ('seed', 'updates', 'batch_size'):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    for key in ('learning_rate', 'gradient_clip', 'advisory_seconds'):
        number = training[key]
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or number <= 0):
            raise ValueError(key + ' must be finite and positive')
    return value


def _save(path, model, optimizer, step, config, rng):
    torch.save(dict(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS,
        loss_semantics=LOSS_SEMANTICS, configuration=config, step=step, actual_updates=step,
        motion_observation_policy='canonical_all', model=model.state_dict(),
        optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None), path)


def load_checkpoint(path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (not isinstance(saved, dict) or saved.get('scope') != SCOPE
            or saved.get('routing_semantics') != ROUTING_SEMANTICS
            or saved.get('loss_semantics') != LOSS_SEMANTICS
            or saved.get('motion_observation_policy') != 'canonical_all'):
        raise ValueError('sequence checkpoint scope, routing or loss contract mismatch')
    config = configuration(saved['configuration'])
    if (type(saved.get('step')) is not int or not 0 <= saved['step'] <= config['training']['updates']
            or type(saved.get('actual_updates')) is not int or saved['actual_updates'] != saved['step']
            or any(key not in saved for key in ('optimizer', 'numpy_rng', 'torch_rng', 'cuda_rng'))):
        raise ValueError('sequence checkpoint update or optimizer/RNG contract mismatch')
    model = SequencePosition(**config['model']).to(device)
    model.load_state_dict(saved['model'], strict=True)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('sequence checkpoint parameters must be finite')
    optimizer = saved['optimizer']
    groups = optimizer.get('param_groups', [])
    if (len(groups) != 1 or groups[0].get('lr') != config['training']['learning_rate']
            or len(groups[0].get('params', [])) != len(list(model.parameters()))):
        raise ValueError('sequence optimizer/model/configuration mismatch')
    entries = optimizer.get('state', {})
    if saved['step'] == 0:
        if entries:
            raise ValueError('initial sequence checkpoint cannot contain optimizer updates')
    elif (set(entries) != set(groups[0]['params'])
          or any(float(entry.get('step', -1)) != saved['step'] for entry in entries.values())):
        raise ValueError('sequence optimizer updates disagree with checkpoint step')
    return model, saved


def _windows(rows):
    """Fixed 100-update summaries; different batches, not fixed-probe convergence."""
    output = []
    for start in range(0, len(rows), 100):
        current = rows[start:start + 100]
        norms = np.asarray([r['gradient_norm'] for r in current])
        output.append(dict(first_step=current[0]['step'], last_step=current[-1]['step'],
            successful_updates=len(current), linear_anchor_mean=float(np.mean([r['loss'] for r in current])),
            log1p_anchor_mean=float(np.mean([r['objectives']['anchor_mean_log1p'] for r in current])),
            gradient_norm_median=float(np.median(norms)), gradient_norm_max=float(norms.max()),
            clipped_updates=sum(r['gradient_clipped'] for r in current),
            valid_position_points=sum(r['objectives']['valid_position_points'] for r in current)))
    return dict(window_updates=100, selection=False,
        interpretation='Different random training batches per window; no fixed-probe convergence claim.', windows=output)


def train_sequence_position(train_raw, config, output, *, device='cpu', check=None):
    """One fit, uniform T scene/anchor and three predeclared corruption modes.

    NumPy sampling is independent of each backbone's initialization RNG. Every
    actual sample is recorded only following a successful optimizer operation.
    Only T_core enters this API; endpoint V inference is owned by the caller.
    """
    config = copy.deepcopy(configuration(config))
    train_raw = list(train_raw)
    scene_ids = [str(scene['scene_id']) for scene in train_raw]
    if (not train_raw or any(scene.get('split') != 'T_core' for scene in train_raw)
            or len(set(scene_ids)) != len(scene_ids)):
        raise ValueError('sequence fit requires distinct nonempty T_core scenes only')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    training = config['training']
    torch.manual_seed(training['seed'])
    rng = np.random.default_rng(training['seed'])
    model = SequencePosition(**config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training['learning_rate'])
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    audit = dict(successful_updates=0, sampled_anchors=0, valid_position_points=0,
                 nonzero_gradient_updates=0, scenes={}, anchors={},
                 modes={mode: dict(updates=0, anchors=0, valid_position_points=0) for mode in TRAIN_MODES})
    state = dict(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS, loss_semantics=LOSS_SEMANTICS,
        status='RUNNING', configuration=config, optimizer_updates=0, test_opened=False,
        motion_observation_policy='canonical_all', device=str(device), training_scene_ids=scene_ids,
        selection_policy='fixed final step; no validation or intermediate selection in training',
        sampling='uniform T_core scene then uniform anchor; fixed three modes by update',
        parameter_count=sum(p.numel() for p in model.parameters()), exposure=audit,
        gradient_clipping=dict(clipped_updates=0, successful_update_fraction=0.),
        advisory_reached=False, model_initial_output='fixed095 physical control; zero tangent residual',
        feature_scaling='fixed visible physical/log scales; no data-fitted or validation statistics',
        prediction_scope='position only; all queries may change; speed/course remain ordinary Mean',
        scientific_status='generic task-adapted equal-exposure pilot, not PTFM/TMTAFF or converged benchmark')
    write_json(output / 'run.json', state)
    rows = []
    try:
        _save(output / 'step-0.pt', model, optimizer, 0, config, rng)
        train = prepare_evidence_scenes(train_raw, check=check, motion_observation_policy='canonical_all')
        if any(not len(scene['identities']) for scene in train):
            raise ValueError('uniform scene sampling cannot silently discard empty scenes')
        state['preparation_seconds'] = time.monotonic() - started
        fit_started = time.monotonic()
        for step in range(1, training['updates'] + 1):
            if check:
                check()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            mode = TRAIN_MODES[(step - 1) % len(TRAIN_MODES)]
            batch, samples, linear, logarithmic = [], [], [], []
            for _ in range(training['batch_size']):
                scene = train[int(rng.integers(len(train)))]
                anchor = int(rng.integers(len(scene['identities'])))
                members = training_members(scene, anchor, mode)
                result = model(*sequence_inputs(scene['motion'], members, device=device))
                target = torch.as_tensor(scene['target'][anchor], dtype=torch.float64, device=device)
                valid = torch.as_tensor(scene['valid'][anchor], dtype=torch.bool, device=device)
                losses, diagnostics = sequence_objectives(result, target, valid)
                if not torch.isfinite(losses['linear']):
                    raise FloatingPointError('nonfinite sequence position loss')
                (losses['linear'] / training['batch_size']).backward()
                linear.append(float(losses['linear'].detach()))
                logarithmic.append(float(losses['log1p'].detach()))
                support = scene['motion']['support']
                eligible = ~support[anchor] & support[members[1:]].any(0)
                samples.append(diagnostics)
                batch.append(dict(scene_id=str(scene['scene_id']), anchor_index=anchor,
                    node_id=str(scene['node_ids'][anchor]), members=[int(x) for x in members],
                    valid_position_points=diagnostics['valid_position_points'],
                    eligible_query_points=int(eligible.sum()),
                    eligible_valid_position_points=int((eligible & valid[:, :2].all(-1).cpu().numpy()).sum())))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'],
                                                 error_if_nonfinite=True)
            optimizer.step()
            # Commit actual optimizer exposure before later serialization checks.
            state['optimizer_updates'] = step
            count = sum(s['valid_position_points'] for s in samples)
            audit['successful_updates'] += 1
            audit['sampled_anchors'] += len(batch)
            audit['valid_position_points'] += count
            audit['nonzero_gradient_updates'] += int(norm > 0)
            audit['modes'][mode]['updates'] += 1
            audit['modes'][mode]['anchors'] += len(batch)
            audit['modes'][mode]['valid_position_points'] += count
            for sample in batch:
                sid, node = sample['scene_id'], sample['node_id']
                audit['scenes'][sid] = audit['scenes'].get(sid, 0) + 1
                audit['anchors'][node] = audit['anchors'].get(node, 0) + 1
            state['gradient_clipping']['clipped_updates'] += int(norm > training['gradient_clip'])
            state['gradient_clipping']['successful_update_fraction'] = state['gradient_clipping']['clipped_updates'] / step
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('nonfinite sequence position parameters after update')
            elapsed = time.monotonic() - started
            if elapsed >= training['advisory_seconds'] and not state['advisory_reached']:
                print('ADVISORY: fixed sequence fit continues; no automatic extension or retry.', flush=True)
            state.update(elapsed_seconds=elapsed, training_seconds=time.monotonic() - fit_started,
                         advisory_reached=elapsed >= training['advisory_seconds'],
                         peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0)
            objectives = dict(anchor_mean_linear=float(np.mean(linear)),
                anchor_mean_log1p=float(np.mean(logarithmic)), valid_position_points=count,
                point_mean_linear=sum(r['linear_sum'] for r in samples) / count if count else None,
                point_mean_log1p=sum(r['log1p_sum'] for r in samples) / count if count else None)
            row = dict(step=step, mode=mode, batch=batch, loss=objectives['anchor_mean_linear'],
                objectives=objectives, gradient_norm=float(norm), gradient_clipped=bool(norm > training['gradient_clip']),
                gradient_clip_scale=min(1., training['gradient_clip'] / (float(norm) + 1e-6)))
            rows.append(row)
            with (output / 'training.jsonl').open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            write_json(output / 'run.json', state)
        _save(output / 'final.pt', model, optimizer, training['updates'], config, rng)
        write_json(output / 'training-windows.json', _windows(rows))
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic() - started,
                     final_checkpoint='final.pt', trained_scene_count=len(audit['scenes']))
        write_json(output / 'run.json', state)
        return state
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started)
        write_json(output / 'run.json', state)
        write_json(output / 'training-windows.json', _windows(rows))
        try:
            _save(output / 'interrupted.pt', model, optimizer, state['optimizer_updates'], config, rng)
        except BaseException as save_error:
            state['recovery_checkpoint_error'] = f'{type(save_error).__name__}: {save_error}'
            write_json(output / 'run.json', state)
        raise
