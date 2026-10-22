"""Fixed-budget position-gate training with paired successful-update auditing."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from association.models.support_gate import (INITIAL_GATE, ROUTING_SEMANTICS, SCOPE,
    SupportGate, gate_inputs, support_gate_objectives)
from association.training.evidence_fusion import prepare_evidence_scenes, training_members, TRAIN_MODES
from association.utils.runtime import write_json

LOSS_SEMANTICS = 'mean_valid_position_log1p_haversine_m_div_1000_all_queries_no_auxiliary'
LINEAR_LOSS_SEMANTICS = 'mean_valid_position_linear_haversine_m_div_1000_all_queries_no_auxiliary'


def loss_semantics(config):
    return LINEAR_LOSS_SEMANTICS if config.get('position_loss', 'log1p') == 'linear' else LOSS_SEMANTICS


def configuration(value):
    expected = {'scope', 'routing_semantics', 'data', 'train_splits', 'model', 'training',
                'motion_observation_policy'}
    if (not isinstance(value, dict) or set(value) not in (expected, expected | {'position_loss'})
            or value.get('position_loss', 'log1p') not in ('log1p', 'linear') or value['scope'] != SCOPE
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
        loss_semantics=loss_semantics(config), configuration=config, step=step, actual_updates=step,
        motion_observation_policy='canonical_all', model=model.state_dict(),
        optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None), path)


def load_checkpoint(path, *, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (not isinstance(state, dict) or state.get('scope') != SCOPE
            or state.get('routing_semantics') != ROUTING_SEMANTICS
            or state.get('motion_observation_policy') != 'canonical_all'):
        raise ValueError('checkpoint support-gate routing/loss semantics disagree')
    config = configuration(state['configuration'])
    if state.get('loss_semantics') != loss_semantics(config):
        raise ValueError('checkpoint support-gate configuration/loss semantics disagree')
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


def _paired_reference(path, config, model, scene_ids):
    """Bind a completed log-loss fit without consuming the training RNG."""
    path = Path(path)
    source = json.loads((path/'run.json').read_text())
    expected = copy.deepcopy(config)
    expected.pop('position_loss', None)
    reference_config = copy.deepcopy(configuration(source['configuration']))
    reference_loss = reference_config.pop('position_loss', 'log1p')
    updates = config['training']['updates']
    if (reference_loss != 'log1p' or reference_config != expected
            or source.get('status') != 'COMPLETED' or source.get('optimizer_updates') != updates
            or source.get('loss_semantics') != LOSS_SEMANTICS
            or source.get('test_opened') is not False
            or source.get('exposure', {}).get('successful_updates') != updates
            or source.get('training_scene_ids') != scene_ids):
        raise ValueError('paired reference requires completed matching log-loss fit and ordered T_core scenes')
    # load_checkpoint instantiates a model; preserve CPU RNG around that validation.
    with torch.random.fork_rng(devices=[]):
        _, initial = load_checkpoint(path/'step-0.pt')
        _, final = load_checkpoint(path/'final.pt')
    if (initial['step'] != 0 or final['step'] != updates
            or initial['configuration'] != source['configuration']
            or final['configuration'] != source['configuration']):
        raise ValueError('paired reference checkpoint endpoint/configuration mismatch')
    own = model.state_dict()
    original = initial['model']
    if (set(own) != set(original) or any(own[key].shape != original[key].shape
            or own[key].dtype != original[key].dtype
            or not torch.equal(own[key].detach().cpu(), original[key]) for key in own)):
        raise ValueError('paired reference initial model parameters are not exactly equal')
    rows = [json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()]
    if len(rows) != updates:
        raise ValueError('paired reference training log must contain every successful update')
    keys = {'scene_id', 'anchor_index', 'node_id', 'members', 'eligible_query_points',
            'eligible_valid_position_points', 'valid_position_points'}
    for step, row in enumerate(rows, 1):
        if (row.get('step') != step or row.get('mode') != TRAIN_MODES[(step - 1) % len(TRAIN_MODES)]
                or len(row.get('batch', [])) != config['training']['batch_size']
                or any(set(sample) != keys for sample in row['batch'])):
            raise ValueError('paired reference training sequence/mode/batch contract mismatch')
    return rows


def _batch_diagnostics(samples, losses):
    """Detached observations from the exact forwards used by this update."""
    valid = sum(row['valid_position_points'] for row in samples)
    eligible = sum(row['eligible_valid_position_points'] for row in samples)
    result = {f'anchor_mean_{name}': sum(row[name] for row in losses) / len(losses)
              for name in ('linear', 'log1p')}
    for region, count in (('all_valid', valid), ('eligible_valid', eligible)):
        for objective in ('linear', 'log1p'):
            total = sum(row[f'{region}_{objective}_sum'] for row in samples)
            result[f'{region}_{objective}'] = total / count if count else None
    result.update(valid_position_points=valid, eligible_valid_position_points=eligible,
        eligible_valid_gate_mean=sum(row['eligible_valid_gate_sum'] for row in samples) / eligible if eligible else None,
        eligible_valid_gate_zero_count=sum(row['eligible_valid_gate_zero_count'] for row in samples),
        eligible_valid_gate_one_count=sum(row['eligible_valid_gate_one_count'] for row in samples),
        eligible_valid_gate_min=min(row['eligible_valid_gate_min'] for row in samples
                                   if row['eligible_valid_gate_min'] is not None) if eligible else None,
        eligible_valid_gate_max=max(row['eligible_valid_gate_max'] for row in samples
                                   if row['eligible_valid_gate_max'] is not None) if eligible else None)
    if any(not math.isfinite(value) for value in result.values() if value is not None):
        raise FloatingPointError('nonfinite support-gate detached diagnostics')
    return result


def train_support_gate(train_raw, config, output, *, device='cpu', check=None, paired_reference_fit=None):
    """Uniform scene/anchor batches; evaluation belongs to the CPU entry point.

    T-only samples are staged per batch and committed to the exposure audit only
    after optimizer.step succeeds. Zero-opportunity batches still consume the
    predeclared optimizer update, and are separately reported (Adam may retain
    momentum). This API never accepts V_select or T_test scenes.
    """
    config = copy.deepcopy(configuration(config))
    output = Path(output)
    if paired_reference_fit is not None:
        reference_path, output_path = Path(paired_reference_fit).resolve(), output.resolve()
        if (reference_path == output_path or reference_path in output_path.parents
                or output_path in reference_path.parents):
            raise ValueError('paired reference and output must be separate directories')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    training = config['training']
    detailed_audit = config.get('position_loss', 'log1p') == 'linear' or paired_reference_fit is not None
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
    state = dict(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS, loss_semantics=loss_semantics(config),
        status='RUNNING', configuration=config, optimizer_updates=0, test_opened=False,
        motion_observation_policy='canonical_all', device=str(device),
        selection_policy='fixed final step; no validation or intermediate selection in training',
        sampling='uniform T_core scene then uniform anchor; fixed three modes by update',
        training_scene_ids=[str(scene['scene_id']) for scene in train_raw], exposure=audit,
        advisory_reached=False, initial_gate=INITIAL_GATE)
    if detailed_audit:
        state['gradient_clipping'] = dict(clipped_updates=0, successful_update_fraction=0.)
    write_json(output/'run.json', state)
    try:
        if (not len(train_raw) or any(scene['split'] != 'T_core' for scene in train_raw)
                or len(set(state['training_scene_ids'])) != len(train_raw)):
            raise ValueError('training requires distinct nonempty T_core scenes only')
        reference_rows = None
        if paired_reference_fit is not None:
            reference_rows = _paired_reference(paired_reference_fit, config, model, state['training_scene_ids'])
            state['pairing'] = dict(reference_fit=str(Path(paired_reference_fit).resolve()),
                initial_parameters_exact=True, ordered_training_scene_ids_exact=True,
                source_successful_updates=len(reference_rows), matched_updates=0, matched_anchors=0,
                matching='each successful update, mode and ordered batch scene/anchor/node/members/counts')
            write_json(output/'run.json', state)
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
            batch, loss_sum, objective_samples, objective_losses = [], 0., [], []
            for batch_index in range(training['batch_size']):
                scene = train[int(rng.integers(len(train)))]
                anchor = int(rng.integers(len(scene['identities'])))
                members = training_members(scene, anchor, mode)
                result = model(*gate_inputs(scene['motion'], members, device=device))
                valid = torch.as_tensor(scene['valid'][anchor], dtype=torch.bool, device=device)
                target = torch.as_tensor(scene['target'][anchor], dtype=torch.float64, device=device)
                objectives, diagnostics = support_gate_objectives(result, target, valid)
                loss = objectives[config.get('position_loss', 'log1p')]
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite support-gate position loss')
                valid_position = valid[:, :2].all(-1)
                sample = dict(scene_id=str(scene['scene_id']), anchor_index=anchor,
                    node_id=str(scene['node_ids'][anchor]), members=[int(value) for value in members],
                    eligible_query_points=int(result['eligible'].sum()),
                    eligible_valid_position_points=int((result['eligible'] & valid_position).sum()),
                    valid_position_points=int(valid_position.sum()))
                if reference_rows is not None and sample != reference_rows[step - 1]['batch'][batch_index]:
                    raise ValueError(f'paired sampling mismatch before optimizer update {step}, slot {batch_index}')
                batch.append(sample)
                objective_samples.append(diagnostics)
                objective_losses.append({name: float(value.detach()) for name, value in objectives.items()})
                (loss / training['batch_size']).backward()
                loss_sum += float(loss.detach()) / training['batch_size']
            objectives = _batch_diagnostics(objective_samples, objective_losses)
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'],
                                                      error_if_nonfinite=True)
            optimizer.step()
            # Commit exposure immediately after a successful optimizer operation.
            state['optimizer_updates'] = step
            if detailed_audit:
                state['gradient_clipping']['clipped_updates'] += int(gradient > training['gradient_clip'])
                state['gradient_clipping']['successful_update_fraction'] = state['gradient_clipping']['clipped_updates'] / step
            if reference_rows is not None:
                state['pairing']['matched_updates'] = step
                state['pairing']['matched_anchors'] += len(batch)
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
                print('ADVISORY: fixed support-gate updates continue; no automatic extension or retry.', flush=True)
            state.update(elapsed_seconds=elapsed,
                         advisory_reached=elapsed >= training['advisory_seconds'])
            with (output/'training.jsonl').open('a') as handle:
                row = dict(step=step, mode=mode, loss=loss_sum, gradient_norm=float(gradient), batch=batch)
                if detailed_audit:
                    row.update(gradient_clipped=bool(gradient > training['gradient_clip']),
                        gradient_clip_scale=min(1., training['gradient_clip'] / (float(gradient) + 1e-6)),
                        cumulative_clip_fraction=state['gradient_clipping']['successful_update_fraction'],
                        objectives=objectives)
                handle.write(json.dumps(row, allow_nan=False) + '\n')
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
