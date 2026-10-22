"""Fixed-budget 068 position-gate training with successful-update exposure audit."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from association.models.support_gate import (INITIAL_GATE, ROUTING_SEMANTICS, SCOPE,
    SupportGate, gate_inputs, support_gate_loss)
from association.training.evidence_fusion import prepare_evidence_scenes, training_members, TRAIN_MODES
from association.utils.runtime import write_json

LOSS_SEMANTICS = 'mean_valid_position_log1p_haversine_m_div_1000_all_queries_no_auxiliary'


def configuration(value):
    expected = {'scope', 'routing_semantics', 'data', 'train_splits', 'model', 'training',
                'motion_observation_policy'}
    if (not isinstance(value, dict) or set(value) != expected or value['scope'] != SCOPE
            or value['routing_semantics'] != ROUTING_SEMANTICS
            or value['motion_observation_policy'] != 'canonical_all'
            or value['train_splits'] != ['T_core']
            or value['data'] != dict(length=64, max_observations=128, time_unit_seconds=20.,
                                     reference_longitude_policy='shortest_arc')
            or value['model'] != dict(initial_gate=INITIAL_GATE)):
        raise ValueError('explicit frozen support-gate data, model and routing_semantics required')
    training = value['training']
    if not isinstance(training, dict) or set(training) != {
            'seed', 'updates', 'batch_size', 'learning_rate', 'gradient_clip', 'advisory_seconds'}:
        raise ValueError('unexpected support-gate training configuration')
    for key in ('seed', 'updates', 'batch_size'):
        if isinstance(training[key], bool) or not isinstance(training[key], int) or training[key] < 1:
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
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (not isinstance(state, dict) or state.get('scope') != SCOPE
            or state.get('routing_semantics') != ROUTING_SEMANTICS
            or state.get('loss_semantics') != LOSS_SEMANTICS
            or state.get('motion_observation_policy') != 'canonical_all'):
        raise ValueError('checkpoint support-gate routing/loss semantics disagree')
    config = configuration(state['configuration'])
    if (isinstance(state.get('step'), bool) or not isinstance(state.get('step'), int)
            or not 0 <= state['step'] <= config['training']['updates']):
        raise ValueError('checkpoint step is outside the declared training budget')
    if (isinstance(state.get('actual_updates'), bool)
            or not isinstance(state.get('actual_updates'), int) or state['actual_updates'] != state['step']):
        raise ValueError('checkpoint actual_updates and step disagree')
    if any(key not in state for key in ('optimizer', 'numpy_rng', 'torch_rng', 'cuda_rng')):
        raise ValueError('checkpoint omits optimizer or RNG state')
    model = SupportGate(**config['model']).to(device)
    model.motion_observation_policy = 'canonical_all'
    model.load_state_dict(state['model'], strict=True)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('checkpoint model parameters must be finite')
    optimizer = state['optimizer']
    groups = optimizer.get('param_groups', [])
    if (len(groups) != 1 or groups[0].get('lr') != config['training']['learning_rate']
            or len(groups[0].get('params', [])) != len(list(model.parameters()))):
        raise ValueError('checkpoint optimizer and configured model/learning rate disagree')
    entries = optimizer.get('state', {})
    if state['step'] == 0:
        if entries:
            raise ValueError('initial checkpoint cannot contain updated optimizer state')
    elif (set(entries) != set(groups[0]['params'])
            or any(float(entry.get('step', -1)) != state['step'] for entry in entries.values())):
        raise ValueError('checkpoint Adam update counts disagree with actual_updates')
    return model, state


def train_support_gate(train_raw, config, output, *, device='cpu', check=None):
    """Uniform scene/anchor batches; evaluation belongs to the CPU entry point.

    T-only samples are staged per batch and committed to the exposure audit only
    after optimizer.step succeeds. Zero-opportunity batches still consume the
    predeclared optimizer update, and are separately reported (Adam may retain
    momentum). This API never accepts V_select or T_test scenes.
    """
    config = copy.deepcopy(configuration(config))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    training = config['training']
    torch.manual_seed(training['seed'])
    rng = np.random.default_rng(training['seed'])
    model = SupportGate(**config['model']).to(device)
    model.motion_observation_policy = 'canonical_all'
    optimizer = torch.optim.Adam(model.parameters(), lr=training['learning_rate'])
    audit = dict(successful_updates=0, nonzero_gradient_updates=0, zero_eligible_updates=0,
        sampled_anchors=0, eligible_query_points=0, eligible_valid_position_points=0,
        valid_position_points=0, scenes={}, anchors={}, modes={mode: dict(
            updates=0, anchors=0, eligible_query_points=0, eligible_valid_position_points=0,
            valid_position_points=0, nonzero_gradient_updates=0, zero_eligible_updates=0)
            for mode in TRAIN_MODES})
    state = dict(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS, loss_semantics=LOSS_SEMANTICS,
        status='RUNNING', configuration=config, optimizer_updates=0, test_opened=False,
        motion_observation_policy='canonical_all', device=str(device),
        selection_policy='fixed final step; no validation or intermediate selection in training',
        sampling='uniform T_core scene then uniform anchor; fixed three modes by update',
        training_scene_ids=[str(scene['scene_id']) for scene in train_raw], exposure=audit,
        advisory_reached=False, initial_gate=INITIAL_GATE)
    write_json(output/'run.json', state)
    try:
        if (not len(train_raw) or any(scene['split'] != 'T_core' for scene in train_raw)
                or len(set(state['training_scene_ids'])) != len(train_raw)):
            raise ValueError('training requires distinct nonempty T_core scenes only')
        _save(output/'step-0.pt', model, optimizer, 0, config, rng)
        train = prepare_evidence_scenes(train_raw, check=check, motion_observation_policy='canonical_all')
        if any(not len(scene['identities']) for scene in train):
            raise ValueError('uniform scene training cannot silently drop an empty scene')
        for step in range(1, training['updates'] + 1):
            if check:
                check()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            mode = TRAIN_MODES[(step - 1) % len(TRAIN_MODES)]
            batch, loss_sum = [], 0.
            for _ in range(training['batch_size']):
                scene = train[int(rng.integers(len(train)))]
                anchor = int(rng.integers(len(scene['identities'])))
                members = training_members(scene, anchor, mode)
                result = model(*gate_inputs(scene['motion'], members, device=device))
                valid = torch.as_tensor(scene['valid'][anchor], dtype=torch.bool, device=device)
                target = torch.as_tensor(scene['target'][anchor], dtype=torch.float64, device=device)
                loss = support_gate_loss(result, target, valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite support-gate position loss')
                (loss / training['batch_size']).backward()
                loss_sum += float(loss.detach()) / training['batch_size']
                valid_position = valid[:, :2].all(-1)
                batch.append(dict(scene_id=str(scene['scene_id']), anchor_index=anchor,
                    node_id=str(scene['node_ids'][anchor]), members=[int(value) for value in members],
                    eligible_query_points=int(result['eligible'].sum()),
                    eligible_valid_position_points=int((result['eligible'] & valid_position).sum()),
                    valid_position_points=int(valid_position.sum())))
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'],
                                                      error_if_nonfinite=True)
            optimizer.step()
            # Commit exposure immediately after a successful optimizer operation.
            state['optimizer_updates'] = step
            eligible = sum(row['eligible_query_points'] for row in batch)
            eligible_valid = sum(row['eligible_valid_position_points'] for row in batch)
            valid_count = sum(row['valid_position_points'] for row in batch)
            nonzero = bool(gradient > 0)
            zero_eligible = eligible == 0
            audit['successful_updates'] += 1
            audit['nonzero_gradient_updates'] += int(nonzero)
            audit['zero_eligible_updates'] += int(zero_eligible)
            audit['sampled_anchors'] += len(batch)
            audit['eligible_query_points'] += eligible
            audit['eligible_valid_position_points'] += eligible_valid
            audit['valid_position_points'] += valid_count
            counters = audit['modes'][mode]
            for key, value in (('updates', 1), ('anchors', len(batch)),
                    ('eligible_query_points', eligible), ('eligible_valid_position_points', eligible_valid),
                    ('valid_position_points', valid_count), ('nonzero_gradient_updates', int(nonzero)),
                    ('zero_eligible_updates', int(zero_eligible))):
                counters[key] += value
            for row in batch:
                name = row['scene_id']
                audit['scenes'][name] = audit['scenes'].get(name, 0) + 1
                anchor_key = row['node_id']
                audit['anchors'][anchor_key] = audit['anchors'].get(anchor_key, 0) + 1
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError('nonfinite support-gate parameters after optimizer update')
            elapsed = time.monotonic() - started
            if elapsed >= training['advisory_seconds'] and not state['advisory_reached']:
                print('ADVISORY: fixed 068 updates continue; no automatic extension or retry.', flush=True)
            state.update(elapsed_seconds=elapsed,
                         advisory_reached=elapsed >= training['advisory_seconds'])
            with (output/'training.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(step=step, mode=mode, loss=loss_sum,
                    gradient_norm=float(gradient), batch=batch), allow_nan=False) + '\n')
            write_json(output/'run.json', state)
        _save(output/'final.pt', model, optimizer, training['updates'], config, rng)
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic() - started,
                     final_checkpoint='final.pt', trained_scene_count=len(audit['scenes']))
        write_json(output/'run.json', state)
        return state
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        try:
            _save(output/'interrupted.pt', model, optimizer, state['optimizer_updates'], config, rng)
        except BaseException as save_error:
            state['recovery_checkpoint_error'] = f'{type(save_error).__name__}: {save_error}'
            write_json(output/'run.json', state)
        raise
