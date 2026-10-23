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
    RELATIVE_HARM_SCOPE, RELATIVE_HARM_LOSS,
    SequencePosition, sequence_inputs, sequence_objectives)
from association.models.helper_sequence import (SCOPE as HELPER_SCOPE,
    ROUTING_SEMANTICS as HELPER_ROUTING, AGGREGATIONS, HelperSequencePosition,
    helper_sequence_inputs, initialize_from_reference)
from association.training.evidence_fusion import TRAIN_MODES, prepare_evidence_scenes, training_members
from association.utils.runtime import write_json

LOSS_SEMANTICS = 'mean_valid_position_linear_haversine_m_div_1000_all_queries_no_auxiliary'
RELATIVE_HARM_SEMANTICS = 'mean_valid_position_linear_plus_relu_error_minus_detached_fixed095_error_haversine_km_all_queries_lambda1'


def loss_semantics(config):
    return RELATIVE_HARM_SEMANTICS if config['position_loss'] == RELATIVE_HARM_LOSS else LOSS_SEMANTICS


def routing_semantics(config):
    return HELPER_ROUTING if config['scope'] == HELPER_SCOPE else ROUTING_SEMANTICS


def _make_model(config):
    return (HelperSequencePosition if config['scope'] == HELPER_SCOPE else SequencePosition)(**config['model'])


def configuration(value):
    expected = {'scope', 'routing_semantics', 'position_loss', 'data', 'motion_observation_policy',
                'train_splits', 'model', 'training'}
    if (not isinstance(value, dict) or set(value) != expected
            or (value['scope'], value['position_loss']) not in ((SCOPE, 'linear'), (RELATIVE_HARM_SCOPE, RELATIVE_HARM_LOSS), (HELPER_SCOPE, 'linear'))
            or value['routing_semantics'] != routing_semantics(value)
            or value['motion_observation_policy'] != 'canonical_all' or value['train_splits'] != ['T_core']
            or value['data'] != dict(length=64, max_observations=128, time_unit_seconds=20.,
                                     reference_longitude_policy='shortest_arc')):
        raise ValueError('explicit sequence position scope/loss and Level1 data contract required')
    model = value['model']
    helper = value['scope'] == HELPER_SCOPE
    fields = {'hidden_size', 'num_layers', 'initial_gate', 'aggregation'} if helper else {'backbone', 'hidden_size', 'num_layers', 'initial_gate'}
    if (not isinstance(model, dict) or set(model) != fields
            or (model['aggregation'] not in AGGREGATIONS if helper else model['backbone'] not in BACKBONES)
            or type(model['hidden_size']) is not int
            or model['hidden_size'] != 64 or type(model['num_layers']) is not int or model['num_layers'] != 2
            or isinstance(model['initial_gate'], bool) or model['initial_gate'] != .95):
        raise ValueError('sequence baseline architecture is fixed apart from the declared backbone')
    if value['position_loss'] == RELATIVE_HARM_LOSS and model['backbone'] != 'transformer':
        raise ValueError('relative-harm intervention declares only the Transformer backbone')
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
    torch.save(dict(scope=config['scope'], routing_semantics=routing_semantics(config),
        loss_semantics=loss_semantics(config), configuration=config, step=step, actual_updates=step,
        motion_observation_policy='canonical_all', model=model.state_dict(),
        optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None), path)


def load_checkpoint(path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get('configuration'), dict):
        raise ValueError('sequence checkpoint configuration required')
    config = configuration(saved['configuration'])
    if (saved.get('scope') != config['scope']
            or saved.get('routing_semantics') != routing_semantics(config)
            or saved.get('loss_semantics') != loss_semantics(config)
            or saved.get('motion_observation_policy') != 'canonical_all'):
        raise ValueError('sequence checkpoint scope, routing or loss contract mismatch')
    if (type(saved.get('step')) is not int or not 0 <= saved['step'] <= config['training']['updates']
            or type(saved.get('actual_updates')) is not int or saved['actual_updates'] != saved['step']
            or any(key not in saved for key in ('optimizer', 'numpy_rng', 'torch_rng', 'cuda_rng'))):
        raise ValueError('sequence checkpoint update or optimizer/RNG contract mismatch')
    model = _make_model(config).to(device)
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


def _reference_pair(path, config, scene_ids):
    """Read the frozen same-seed linear control, never resume its optimizer."""
    path = Path(path).resolve()
    record = json.loads((path/'run.json').read_text())
    expected = copy.deepcopy(config)
    expected.update(scope=SCOPE, position_loss='linear', routing_semantics=ROUTING_SEMANTICS)
    if config['scope'] == HELPER_SCOPE:
        expected['model'].pop('aggregation')
        expected['model']['backbone'] = 'transformer'
    if (record.get('status') != 'COMPLETED' or record.get('configuration') != expected
            or record.get('optimizer_updates') != config['training']['updates']
            or record.get('training_scene_ids') != scene_ids):
        raise ValueError('relative-harm control must complete the same seed, data and fixed budget with linear loss')
    initial = None
    for filename, step in (('step-0.pt', 0), ('final.pt', config['training']['updates'])):
        model, saved = load_checkpoint(path/filename)
        del model
        if saved['configuration'] != expected or saved['step'] != step:
            raise ValueError('relative-harm control checkpoint/configuration/update mismatch')
        if step == 0:
            initial = saved
    rows = [json.loads(line) for line in (path/'training.jsonl').read_text().splitlines()]
    if len(rows) != config['training']['updates']:
        raise ValueError('relative-harm control must retain every actual successful batch')
    for step, row in enumerate(rows, 1):
        if (row.get('step') != step or row.get('mode') != TRAIN_MODES[(step-1) % len(TRAIN_MODES)]
                or len(row.get('batch', [])) != config['training']['batch_size']):
            raise ValueError('relative-harm control actual successful batch order/count differs')
    return dict(path=str(path), initial=initial, rows=rows, exposure=record['exposure'])


def _sampling_exposure(audit):
    return {key: value for key, value in audit.items() if key != 'nonzero_gradient_updates'}


def _windows(rows):
    """Fixed 100-update summaries; different batches, not fixed-probe convergence."""
    output = []
    for start in range(0, len(rows), 100):
        current = rows[start:start + 100]
        norms = np.asarray([r['gradient_norm'] for r in current])
        output.append(dict(first_step=current[0]['step'], last_step=current[-1]['step'],
            successful_updates=len(current), linear_anchor_mean=float(np.mean([r['objectives']['anchor_mean_linear'] for r in current])),
            log1p_anchor_mean=float(np.mean([r['objectives']['anchor_mean_log1p'] for r in current])),
            gradient_norm_median=float(np.median(norms)), gradient_norm_max=float(norms.max()),
            clipped_updates=sum(r['gradient_clipped'] for r in current),
            valid_position_points=sum(r['objectives']['valid_position_points'] for r in current)))
        if 'anchor_mean_relative_harm' in current[0]['objectives']:
            output[-1].update(relative_harm_anchor_mean=float(np.mean([r['objectives']['anchor_mean_relative_harm'] for r in current])),
                objective_anchor_mean=float(np.mean([r['loss'] for r in current])))
    return dict(window_updates=100, selection=False,
        interpretation='Different random training batches per window; no fixed-probe convergence claim.', windows=output)


def train_sequence_position(train_raw, config, output, *, device='cpu', check=None, paired_reference_fit=None):
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
    harm_objective = config['position_loss'] == RELATIVE_HARM_LOSS
    helper = config['scope'] == HELPER_SCOPE
    if (harm_objective or helper) != (paired_reference_fit is not None):
        raise ValueError('relative-harm/helper training requires a frozen same-seed linear control')
    reference = _reference_pair(paired_reference_fit, config, scene_ids) if paired_reference_fit is not None else None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    training = config['training']
    torch.manual_seed(training['seed'])
    rng = np.random.default_rng(training['seed'])
    model = _make_model(config)
    if helper:
        initialize_from_reference(model, reference['initial']['model'], branch_seed=training['seed'] + 78000)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training['learning_rate'])
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    audit = dict(successful_updates=0, sampled_anchors=0, valid_position_points=0,
                 nonzero_gradient_updates=0, scenes={}, anchors={},
                 modes={mode: dict(updates=0, anchors=0, valid_position_points=0) for mode in TRAIN_MODES})
    state = dict(scope=config['scope'], routing_semantics=routing_semantics(config), loss_semantics=loss_semantics(config),
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
    if helper:
        state.update(feature_width=92, helper_aggregation=config['model']['aggregation'],
            helper_support_exposure={population: {name: 0 for name in ('zero', 'one', 'multiple')}
                                     for population in ('all_queries', 'valid_position_queries')},
            helper_gradients=dict(phi_nonzero_updates=0, scorer_nonzero_updates=0),
            scientific_status='single-seed helper representation screen; weights are latent, not physical reliability')
    if reference is not None:
        state['paired_control'] = dict(fit=reference['path'], status='PENDING', initial_parameters_exact=False,
            initial_sampler_rng_exact=False, actual_successful_batches_matched=0,
            sampling_exposure_exact=False, initialization='fresh; control optimizer never resumed')
    write_json(output / 'run.json', state)
    rows = []
    try:
        _save(output / 'step-0.pt', model, optimizer, 0, config, rng)
        if reference is not None:
            old = reference['initial']
            current = model.state_dict()
            if helper:
                common_exact = all(torch.equal(
                    (current[key][:, :76] if key == 'input_projection.0.weight' else current[key]).detach().cpu(), value)
                    for key, value in old['model'].items())
            else:
                common_exact = set(current) == set(old['model']) and all(
                    torch.equal(value.detach().cpu(), old['model'][key]) for key, value in current.items())
            if (not common_exact or rng.bit_generator.state != old['numpy_rng']
                    or (not helper and not torch.equal(torch.get_rng_state(), old['torch_rng']))):
                raise ValueError('relative-harm fresh initial parameters or sampler RNG differ from linear control')
            state['paired_control'].update(initial_parameters_exact=True, initial_sampler_rng_exact=True)
            if helper:
                state['paired_control'].update(initial_parameters_scope='all original common tensors; first76 projection columns',
                    added_projection_zero=True, branch_seed=training['seed'] + 78000,
                    initialization='original step0 common tensors; independent new branch; fresh Adam')
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
            batch, samples, linear, logarithmic, optimized = [], [], [], [], []
            helper_batch = []
            for _ in range(training['batch_size']):
                scene = train[int(rng.integers(len(train)))]
                anchor = int(rng.integers(len(scene['identities'])))
                members = training_members(scene, anchor, mode)
                result = model(*(helper_sequence_inputs if helper else sequence_inputs)(scene['motion'], members, device=device))
                target = torch.as_tensor(scene['target'][anchor], dtype=torch.float64, device=device)
                valid = torch.as_tensor(scene['valid'][anchor], dtype=torch.bool, device=device)
                losses, diagnostics = sequence_objectives(result, target, valid, position_loss=config['position_loss'])
                objective = losses[config['position_loss']]
                if not torch.isfinite(objective):
                    raise FloatingPointError('nonfinite sequence position loss')
                (objective / training['batch_size']).backward()
                linear.append(float(losses['linear'].detach()))
                logarithmic.append(float(losses['log1p'].detach()))
                optimized.append(float(objective.detach()))
                support = scene['motion']['support']
                eligible = ~support[anchor] & support[members[1:]].any(0)
                samples.append(diagnostics)
                if helper:
                    counts = support[members[1:]].sum(0)
                    valid_positions = valid[:, :2].all(-1).cpu().numpy()
                    helper_batch.append({population: {name: int((selection & mask).sum())
                        for name, selection in (('zero', counts == 0), ('one', counts == 1), ('multiple', counts >= 2))}
                        for population, mask in (('all_queries', np.ones_like(counts, bool)),
                                                  ('valid_position_queries', valid_positions))})
                batch.append(dict(scene_id=str(scene['scene_id']), anchor_index=anchor,
                    node_id=str(scene['node_ids'][anchor]), members=[int(x) for x in members],
                    valid_position_points=diagnostics['valid_position_points'],
                    eligible_query_points=int(eligible.sum()),
                    eligible_valid_position_points=int((eligible & valid[:, :2].all(-1).cpu().numpy()).sum())))
            if reference is not None:
                old = reference['rows'][step-1]
                if old['mode'] != mode or old['batch'] != batch:
                    raise ValueError(f'relative-harm actual sampling differs from linear control before update {step}')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'],
                                                 error_if_nonfinite=True)
            branch_grad = {}
            if helper:
                branch_grad = {label: any(p.grad is not None and bool(torch.count_nonzero(p.grad))
                    for name, p in model.named_parameters() if name.startswith(prefix))
                    for label, prefix in (('phi_nonzero_updates', 'phi.'), ('scorer_nonzero_updates', 'scorer.'))}
            optimizer.step()
            # Commit actual optimizer exposure before later serialization checks.
            state['optimizer_updates'] = step
            count = sum(s['valid_position_points'] for s in samples)
            audit['successful_updates'] += 1
            audit['sampled_anchors'] += len(batch)
            audit['valid_position_points'] += count
            audit['nonzero_gradient_updates'] += int(norm > 0)
            if helper:
                for sample_counts in helper_batch:
                    for population, counts in sample_counts.items():
                        for category, value in counts.items():
                            state['helper_support_exposure'][population][category] += value
                for label, nonzero in branch_grad.items():
                    state['helper_gradients'][label] += int(nonzero)
            audit['modes'][mode]['updates'] += 1
            audit['modes'][mode]['anchors'] += len(batch)
            audit['modes'][mode]['valid_position_points'] += count
            if reference is not None:
                state['paired_control']['actual_successful_batches_matched'] = step
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
            if harm_objective:
                objectives.update(anchor_mean_objective=float(np.mean(optimized)),
                    anchor_mean_relative_harm=float(np.mean([s['relative_harm_sum']/s['valid_position_points']
                        if s['valid_position_points'] else 0. for s in samples])),
                    point_mean_relative_harm=sum(s['relative_harm_sum'] for s in samples)/count if count else None,
                    point_mean_relative_gain=sum(s['relative_gain_sum'] for s in samples)/count if count else None,
                    point_mean_baseline=sum(s['baseline_sum'] for s in samples)/count if count else None,
                    point_mean_objective=sum(s['objective_sum'] for s in samples)/count if count else None)
            row = dict(step=step, mode=mode, batch=batch, loss=float(np.mean(optimized)),
                objectives=objectives, gradient_norm=float(norm), gradient_clipped=bool(norm > training['gradient_clip']),
                gradient_clip_scale=min(1., training['gradient_clip'] / (float(norm) + 1e-6)))
            if helper:
                row.update(helper_support_batch=helper_batch, helper_gradients=branch_grad)
            rows.append(row)
            with (output / 'training.jsonl').open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            write_json(output / 'run.json', state)
        if reference is not None:
            if _sampling_exposure(audit) != _sampling_exposure(reference['exposure']):
                raise ValueError('relative-harm successful sampling exposure differs from linear control')
            state['paired_control'].update(status='COMPLETED', sampling_exposure_exact=True)
        _save(output / 'final.pt', model, optimizer, training['updates'], config, rng)
        write_json(output / 'training-windows.json', _windows(rows))
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic() - started,
                     final_checkpoint='final.pt', trained_scene_count=len(audit['scenes']))
        write_json(output / 'run.json', state)
        return state
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic() - started)
        if reference is not None:
            state['paired_control']['status'] = state['status']
        write_json(output / 'run.json', state)
        write_json(output / 'training-windows.json', _windows(rows))
        try:
            _save(output / 'interrupted.pt', model, optimizer, state['optimizer_updates'], config, rng)
        except BaseException as save_error:
            state['recovery_checkpoint_error'] = f'{type(save_error).__name__}: {save_error}'
            write_json(output / 'run.json', state)
        raise
