"""099 scratch DeepAF training on two exact cycles of the source batches.

Behavior contract
-----------------
Inputs are distinct T_core scenes, a declared DeepAF P/V configuration, and a
completed same-seed 072C/073C fit. No source model or optimizer is inherited.
Each cycle regenerates the source's successful batches from its initial NumPy
sampler state. Only that sampler resets: model weights and Adam continue across
the cycle boundary. Mode is indexed by source_step, not by the global step
(4000 is not divisible by the three corruption modes).

The model owns the P/V desired-increment MSE definition and never receives
training targets as inference inputs. This module owns sampling, optimization,
successful-update accounting, checkpoint binding, and failure preservation.
There is no validation inference, early stopping, retry, resume, or deployment.

Acceptance evidence: the real CPU source4 -> peer8 smoke must reproduce both
four-batch cycles, retain steps0/4/8, prove continued Adam moments/step counters,
update both networks, and reject corrupted checkpoints or replay records.
Output creation is exclusive; a rejected duplicate cannot rewrite old results.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from association.models import deepaf_peer as peer
from association.models.evidence_fusion import physical_errors
from association.training import sequence_position as sequence
from association.training.evidence_fusion import (
    TRAIN_MODES, prepare_evidence_scenes, training_members,
)
from association.utils.runtime import read_json, write_json

SCOPE = 'deepaf-pv-position-peer-v1'
ROUTING_SEMANTICS = 'fixed056-pv-closed-loop-tbptt1-v1'
LOSS_SEMANTICS = 'pv-desired-increment-mse-v1'
OPTIMIZER_INITIALIZATION = 'scratch Adam; continuous across replay cycles; no source state inherited'
HEARTBEAT_INTERVAL = 100
REPLAY_QUERY_BATCH_SIZE = 64


def configuration(value):
    """Validate the reusable fit contract; the runner freezes production files."""
    if (not isinstance(value, dict) or set(value) != {
            'scope', 'routing_semantics', 'loss_semantics', 'data',
            'motion_observation_policy', 'train_splits', 'model', 'training'}
            or value.get('scope') != SCOPE
            or value.get('routing_semantics') != ROUTING_SEMANTICS
            or value.get('loss_semantics') != LOSS_SEMANTICS
            or value.get('motion_observation_policy') != 'canonical_all'
            or value.get('train_splits') != ['T_core']
            or value.get('data') != dict(length=64, max_observations=128,
                time_unit_seconds=20., reference_longitude_policy='shortest_arc')
            or value.get('model') != peer.MODEL_CONFIG):
        raise ValueError('099 requires the declared DeepAF P/V and fixed Level1 data contract')
    training = value['training']
    if not isinstance(training, dict) or set(training) != {
            'seed', 'updates', 'source_updates', 'replay_cycles', 'batch_size',
            'learning_rate', 'gradient_clip', 'advisory_seconds'}:
        raise ValueError('099 unexpected training configuration')
    for key in ('seed', 'updates', 'source_updates', 'replay_cycles', 'batch_size'):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError('099 '+key+' must be a positive integer')
    if training['replay_cycles'] != 2 or training['updates'] != 2*training['source_updates']:
        raise ValueError('099 requires exactly two complete source replay cycles')
    for key in ('learning_rate', 'gradient_clip', 'advisory_seconds'):
        number = training[key]
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or number <= 0):
            raise ValueError('099 '+key+' must be finite and positive')
    if training['learning_rate'] not in (.001, .0003):
        raise ValueError('099 allows only the two predeclared learning rates')
    return value


def source_configuration(value):
    """The source keeps its own original learning rate/budget, not the peer's."""
    config = configuration(value)
    training = config['training']
    original = dict(scope=sequence.SCOPE, routing_semantics=sequence.ROUTING_SEMANTICS,
        position_loss='linear', data=copy.deepcopy(config['data']),
        motion_observation_policy='canonical_all', train_splits=['T_core'],
        model=dict(backbone='transformer', hidden_size=64, num_layers=2, initial_gate=.95),
        training=dict(seed=training['seed'], updates=training['source_updates'],
            batch_size=training['batch_size'], learning_rate=.001,
            gradient_clip=1., advisory_seconds=14400.))
    return sequence.configuration(original)


def _state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _exact(left, right):
    return (isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right)
        and all(isinstance(left[key], torch.Tensor) and isinstance(right[key], torch.Tensor)
                and left[key].dtype == right[key].dtype and torch.equal(left[key], right[key])
                for key in left))


def _load_state(model, value):
    expected = model.state_dict()
    if (not isinstance(value, dict) or set(value) != set(expected)
            or any(not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape
                   or tensor.dtype != expected[key].dtype or not torch.isfinite(tensor).all()
                   for key, tensor in value.items())):
        raise ValueError('099 checkpoint model keys/shapes/dtypes/finite values differ')
    model.load_state_dict(value, strict=True)


def _validate_rng(saved):
    try:
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(saved['numpy_rng'])
        generator = torch.Generator(device='cpu')
        generator.set_state(saved['torch_rng'])
        cuda = saved['cuda_rng']
    except (ValueError, TypeError, RuntimeError, KeyError) as error:
        raise ValueError('099 checkpoint RNG state is invalid') from error
    if cuda is not None and (not isinstance(cuda, list) or any(
            not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
            or state.ndim != 1 or state.numel() == 0 for state in cuda)):
        raise ValueError('099 checkpoint CUDA RNG state is invalid')


def _validate_optimizer(saved, model):
    config = saved['configuration']
    optimizer = saved.get('optimizer', {})
    groups, entries = optimizer.get('param_groups', []), optimizer.get('state', {})
    parameters = list(model.parameters())
    if (len(groups) != 1 or groups[0].get('lr') != config['training']['learning_rate']
            or groups[0].get('weight_decay') != 0 or groups[0].get('betas') != (0.9, 0.999)
            or groups[0].get('eps') != 1e-8 or groups[0].get('amsgrad') is not False
            or len(groups[0].get('params', [])) != len(parameters)
            or len(set(groups[0]['params'])) != len(parameters)):
        raise ValueError('099 checkpoint requires all parameters in one continuous Adam group')
    ids, step = groups[0]['params'], saved['step']
    if ((step == 0 and entries) or (step > 0 and (set(entries) != set(ids)
            or any(float(entry.get('step', -1)) != step for entry in entries.values())))):
        raise ValueError('099 optimizer successful update counts disagree with the checkpoint')
    for key, parameter in zip(ids, parameters):
        if key not in entries:
            continue
        for moment in ('exp_avg', 'exp_avg_sq'):
            value = entries[key].get(moment)
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                raise ValueError('099 optimizer moment shape/dtype/finite contract differs')


def _save(path, model, optimizer, step, config, rng, source, *, recovery=False):
    per_cycle = config['training']['source_updates']
    torch.save(dict(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS,
        loss_semantics=LOSS_SEMANTICS, motion_observation_policy='canonical_all',
        configuration=config, step=step, actual_updates=step,
        checkpoint_kind='recovery' if recovery else 'endpoint',
        completed_replay_cycles=step//per_cycle,
        source_step=0 if step == 0 else (step-1) % per_cycle+1,
        model=model.state_dict(), optimizer=optimizer.state_dict(),
        numpy_rng=copy.deepcopy(rng.bit_generator.state), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        source_sampling=source, source_weights_inherited=False,
        optimizer_initialization=OPTIMIZER_INITIALIZATION), path)


def load_checkpoint(path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get('configuration'), dict):
        raise ValueError('099 checkpoint configuration required')
    config = configuration(saved['configuration'])
    steps = config['training']['source_updates']
    step = saved.get('step')
    kind = saved.get('checkpoint_kind')
    if (saved.get('scope') != SCOPE or saved.get('routing_semantics') != ROUTING_SEMANTICS
            or saved.get('loss_semantics') != LOSS_SEMANTICS
            or saved.get('motion_observation_policy') != 'canonical_all'
            or type(step) is not int or not 0 <= step <= config['training']['updates']
            or type(saved.get('actual_updates')) is not int or saved['actual_updates'] != step
            or kind not in ('endpoint', 'recovery')
            or (kind == 'endpoint' and step not in (0, steps, 2*steps))
            or saved.get('completed_replay_cycles') != step//steps
            or saved.get('source_step') != (0 if step == 0 else (step-1) % steps+1)
            or saved.get('source_weights_inherited') is not False
            or saved.get('optimizer_initialization') != OPTIMIZER_INITIALIZATION):
        raise ValueError('099 checkpoint scope/update/replay/initialization contract differs')
    _validate_rng(saved)
    source = saved.get('source_sampling', {})
    if (source.get('configuration') != source_configuration(config)
            or source.get('source_updates') != steps or not isinstance(source.get('fit'), str)
            or not source['fit'] or source.get('weights_inherited') is not False
            or not isinstance(source.get('initial_numpy_rng'), dict)
            or not isinstance(source.get('final_numpy_rng'), dict)):
        raise ValueError('099 checkpoint source sampling binding differs')
    for state in (source['initial_numpy_rng'], source['final_numpy_rng']):
        try:
            np.random.default_rng().bit_generator.state = copy.deepcopy(state)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError('099 source sampler RNG invalid') from error
    if kind == 'endpoint' and ((step == 0 and saved['numpy_rng'] != source['initial_numpy_rng'])
            or (step > 0 and step % steps == 0 and saved['numpy_rng'] != source['final_numpy_rng'])):
        raise ValueError('099 endpoint sampler RNG disagrees with the source replay')
    # Model reconstruction must not consume the caller's training RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config['training']['seed'])
        model = peer.DeepAFPeer(**config['model'])
        fresh = _state(model)
        initial_torch_rng = torch.get_rng_state()
    _load_state(model, saved.get('model'))
    if step == 0 and not _exact(_state(model), fresh):
        raise ValueError('099 step-zero model differs from the declared scratch initialization seed')
    if step == 0 and kind == 'endpoint' and not torch.equal(saved['torch_rng'], initial_torch_rng):
        raise ValueError('099 initial Torch RNG differs from the declared scratch initialization')
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('099 all P/V network parameters must remain trainable')
    _validate_optimizer(saved, model)
    return model.to(device), saved


def _empty_exposure():
    return dict(successful_updates=0, sampled_anchors=0, valid_position_points=0,
        scenes={}, anchors={}, modes={mode: dict(updates=0, anchors=0,
            valid_position_points=0) for mode in TRAIN_MODES})


def _commit_exposure(audit, batch, mode):
    count = sum(sample['valid_position_points'] for sample in batch)
    audit['successful_updates'] += 1
    audit['sampled_anchors'] += len(batch)
    audit['valid_position_points'] += count
    audit['modes'][mode]['updates'] += 1
    audit['modes'][mode]['anchors'] += len(batch)
    audit['modes'][mode]['valid_position_points'] += count
    for sample in batch:
        for key, name in (('scenes', sample['scene_id']), ('anchors', sample['node_id'])):
            audit[key][name] = audit[key].get(name, 0)+1


def _scaled(value, factor):
    if isinstance(value, dict):
        return {key: _scaled(item, factor) for key, item in value.items()}
    if type(value) is not int:
        raise ValueError('099 sampling exposure counters must be integers')
    return value*factor


def verify_replay(fit_path, reference_fit=None):
    """Rebuild successful exposure with source-step, not global-mode, semantics."""
    fit_path = Path(fit_path).resolve()
    record = read_json(fit_path/'run.json')
    config = configuration(record['configuration'])
    training = config['training']
    if (record.get('status') != 'COMPLETED'
            or record.get('optimizer_updates') != training['updates']
            or record.get('test_opened') is not False
            or record.get('source_weights_inherited') is not False):
        raise ValueError('099 replay requires a completed scratch fit with actual updates')
    path = Path(record['source_sampling']['fit']).resolve()
    if reference_fit is not None and path != Path(reference_fit).resolve():
        raise ValueError('099 replay source path differs from the paired source')
    reference = sequence._reference_pair(path, source_configuration(config), record['training_scene_ids'])
    source_exposure = _empty_exposure()
    for row in reference['rows']:
        _commit_exposure(source_exposure, row['batch'], row['mode'])
    if source_exposure != sequence._sampling_exposure(reference['exposure']):
        raise ValueError('099 source successful batches disagree with recorded exposure')
    rows = [json.loads(line) for line in (fit_path/'training.jsonl').read_text().splitlines()]
    if len(rows) != training['updates']:
        raise ValueError('099 missing successful optimizer records')
    exposure = _empty_exposure()
    for step, row in enumerate(rows, 1):
        source_step = (step-1) % training['source_updates']+1
        cycle = (step-1)//training['source_updates']
        old = reference['rows'][source_step-1]
        if (row.get('step') != step or row.get('replay_cycle') != cycle
                or row.get('source_step') != source_step
                or row.get('mode') != old['mode'] or row.get('batch') != old['batch']
                or not math.isfinite(row.get('loss', math.nan))
                or not math.isfinite(row.get('gradient_norm', math.nan))):
            raise ValueError('099 actual batch/mode/cycle/source-step replay differs')
        _commit_exposure(exposure, row['batch'], row['mode'])
    if exposure != _scaled(source_exposure, 2) or exposure != record.get('exposure'):
        raise ValueError('099 actual two-cycle sampling exposure differs')
    return dict(status='PAIR_VERIFIED', actual_successful_batches_exact=True,
        ordered_training_scene_ids_exact=True, sampling_exposure_exact=True,
        source_updates=training['source_updates'], replay_cycles=2,
        matched_updates=len(rows), matched_anchors=exposure['sampled_anchors'],
        seed=training['seed'], source_fit=str(path), source_weights_inherited=False,
        optimizer=OPTIMIZER_INITIALIZATION, exposure=exposure,
        equal_compute_claimed=False, convergence_claimed=False)


def _heartbeat(output, state, *, phase):
    write_json(output/'heartbeat.json', dict(status=state['status'], phase=phase,
        optimizer_updates=state['optimizer_updates'],
        completed_replay_cycles=state.get('completed_replay_cycles', 0),
        sampled_anchors=state['exposure']['sampled_anchors'],
        valid_position_points=state['exposure']['valid_position_points'],
        elapsed_seconds=state.get('elapsed_seconds', 0.),
        training_seconds=state.get('training_seconds', 0.),
        advisory_reached=state['advisory_reached']))


def _windows(rows):
    windows = []
    for start in range(0, len(rows), 100):
        current = rows[start:start+100]
        windows.append(dict(first_step=current[0]['step'], last_step=current[-1]['step'],
            successful_updates=len(current), objective_mean=float(np.mean([row['loss'] for row in current])),
            position_mse_mean=float(np.mean([row['objectives']['position_mse'] for row in current])),
            velocity_mse_mean=float(np.mean([row['objectives']['velocity_mse'] for row in current])),
            gradient_norm_median=float(np.median([row['gradient_norm'] for row in current])),
            gradient_norm_max=float(max(row['gradient_norm'] for row in current)),
            clipped_updates=sum(row['gradient_clipped'] for row in current)))
    return dict(window_updates=100, selection=False, fixed_training_probe=False,
        interpretation='Different T batches per window; the second cycle repeats source batches.',
        windows=windows)


def _fixed_training_probe(model, train, source_row, config, *, step, device, check=None):
    """Source-step1 batch at fixed endpoints; no sampling, gradients, or update."""
    by_scene = {str(scene['scene_id']): scene for scene in train}
    inputs, targets, masks = [], [], []
    for item in source_row['batch']:
        scene = by_scene[item['scene_id']]
        anchor = item['anchor_index']
        members = training_members(scene, anchor, source_row['mode'])
        if members != item['members'] or str(scene['node_ids'][anchor]) != item['node_id']:
            raise ValueError('099 fixed T probe differs from source-step1 members')
        inputs.append(peer.deepaf_inputs(scene['motion'], members, device=device))
        targets.append(scene['target'][anchor])
        masks.append(scene['valid'][anchor])
    before_mode = model.training
    started = time.monotonic()
    model.eval()
    accumulated = dict(total=0., p_mse=0., v_mse=0.)
    try:
        with torch.no_grad():
            rollout = model.forward_batch(peer.deepaf_batch_inputs(inputs), cache=True)
            target = torch.as_tensor(np.asarray(targets), dtype=torch.float64, device=device)
            valid = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=device)
            supervision = peer.deepaf_supervision(rollout, target, valid,
                velocity_scale_mps=config['model']['velocity_scale_mps'])
            total_queries = len(inputs)*rollout['states'].shape[1]
            for start in range(0, total_queries, REPLAY_QUERY_BATCH_SIZE):
                if check:
                    check()
                stop = min(start+REPLAY_QUERY_BATCH_SIZE, total_queries)
                prediction = model.replay_batch(rollout, start, stop)
                losses, _ = peer.deepaf_objectives(prediction, supervision, start=start, stop=stop)
                for key in accumulated:
                    value = float(losses[key])
                    if not math.isfinite(value):
                        raise FloatingPointError('099 nonfinite fixed training probe objective')
                    accumulated[key] += value
            use = valid[..., :2].all(-1)
            safe = torch.where(valid, target, torch.zeros_like(target))
            errors = physical_errors(rollout['states'], safe)[..., 0]
            if not torch.isfinite(errors[use]).all():
                raise FloatingPointError('099 nonfinite fixed training probe position errors')
            per_anchor = [float(values[mask].mean()) for values, mask in zip(errors, use) if mask.any()]
        return dict(step=step, source_step=1, source_mode=source_row['mode'],
            batch=copy.deepcopy(source_row['batch']), objective=accumulated['total'],
            position_mse=accumulated['p_mse'], velocity_mse=accumulated['v_mse'],
            position_mae_m=float(errors[use].mean()) if use.any() else None,
            anchor_equal_position_mae_m=float(np.mean(per_anchor)) if per_anchor else None,
            valid_position_points=int(use.sum()), sampled=False, optimizer_updates=0,
            closed_loop_group_forwards=len(inputs), query_replay_forwards=math.ceil(total_queries/REPLAY_QUERY_BATCH_SIZE),
            elapsed_seconds=time.monotonic()-started,
            interpretation='Fixed T batch diagnostic only; no validation selection or convergence claim.')
    finally:
        model.train(before_mode)


def train_deepaf_peer(train_raw, config, output, *, paired_reference_fit,
                      device='cpu', check=None):
    """One scratch fit; cached closed-loop rollout, P/V MSE, continuous Adam."""
    config = copy.deepcopy(configuration(config))
    training = config['training']
    train_raw = list(train_raw)
    scene_ids = [str(scene['scene_id']) for scene in train_raw]
    if (not train_raw or len(set(scene_ids)) != len(scene_ids)
            or any(scene.get('split') != 'T_core' for scene in train_raw)):
        raise ValueError('099 requires nonempty distinct T_core scenes only')
    output, source_path = Path(output).resolve(), Path(paired_reference_fit).resolve()
    if output == source_path or output in source_path.parents or source_path in output.parents:
        raise ValueError('099 output must be separate from the immutable source fit')
    if output.exists():
        raise FileExistsError('099 output already exists; no reuse/retry')
    old_config = source_configuration(config)
    reference = sequence._reference_pair(source_path, old_config, scene_ids)
    old_model, old_final = sequence.load_checkpoint(source_path/'final.pt')
    if old_final['configuration'] != old_config or old_final['step'] != training['source_updates']:
        raise ValueError('099 source checkpoint does not bind the complete original budget')
    source = dict(fit=str(source_path), configuration=old_config,
        source_updates=training['source_updates'], weights_inherited=False,
        initial_numpy_rng=copy.deepcopy(reference['initial']['numpy_rng']),
        final_numpy_rng=copy.deepcopy(old_final['numpy_rng']))
    del old_model, old_final
    source_exposure = _empty_exposure()
    for row in reference['rows']:
        _commit_exposure(source_exposure, row['batch'], row['mode'])
    if source_exposure != sequence._sampling_exposure(reference['exposure']):
        raise ValueError('099 source successful records do not reproduce source exposure')
    torch.manual_seed(training['seed'])
    rng = np.random.default_rng(training['seed'])
    if rng.bit_generator.state != source['initial_numpy_rng']:
        raise ValueError('099 source initial sampler does not match the paired seed')
    model = peer.DeepAFPeer(**config['model']).to(device)
    parameters = list(model.parameters())
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError('099 both P/V networks must be fully trainable')
    initial_state = _state(model)
    optimizer = torch.optim.Adam(parameters, lr=training['learning_rate'])
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    prefixes = sorted({name.split('.')[0] for name, _ in model.named_parameters()})
    state = dict(status='RUNNING', scope=SCOPE, routing_semantics=ROUTING_SEMANTICS,
        loss_semantics=LOSS_SEMANTICS, configuration=config,
        motion_observation_policy='canonical_all', training_scene_ids=scene_ids,
        device=str(device), optimizer_updates=0, completed_replay_cycles=0,
        exposure=_empty_exposure(), source_sampling=source, source_weights_inherited=False,
        optimizer_initialization=OPTIMIZER_INITIALIZATION,
        parameter_count=sum(parameter.numel() for parameter in parameters),
        trainable_parameter_count=sum(parameter.numel() for parameter in parameters),
        source_updates=training['source_updates'], total_training_updates=training['updates'],
        replay_cycles=2, sampler_reset_count=0, actual_successful_batches_matched=0,
        initial_sampler_rng_exact=True, final_sampler_rng_exact=False,
        sampling_exposure_exact=False, initial_parameters='declared-seed scratch Xavier initialization',
        training_graph='closed-loop detached rollout followed by exact cached-input TBPTT1 replay',
        query_replay_batch_size=REPLAY_QUERY_BATCH_SIZE,
        selection_policy='fixed training budget; no validation or intermediate selection in trainer',
        inference_targets_used=False, validation_opened=False, test_opened=False,
        nonzero_gradient_updates=0,
        network_gradients={prefix+'_nonzero_updates': 0 for prefix in prefixes},
        gradient_clipping=dict(clipped_updates=0, successful_update_fraction=0.),
        advisory_reached=False, heartbeat_interval_updates=HEARTBEAT_INTERVAL,
        completed_checkpoints=[], final_checkpoint=None,
        training_diagnostics=dict(fixed_source_step=1, endpoint_steps=[0, training['source_updates'], training['updates']],
            completed_steps=[], diagnostic_group_forwards=0, diagnostic_query_replay_forwards=0,
            optimizer_updates=0, sampling_exposure_included=False),
        scientific_status='Published-method task adaptation; equal sample exposure is not a convergence claim.')
    write_json(output/'run.json', state)
    _heartbeat(output, state, phase='preparation')
    rows = []
    probes = []

    def probe(train, step):
        numpy_before = copy.deepcopy(rng.bit_generator.state)
        torch_before = torch.get_rng_state()
        result = _fixed_training_probe(model, train, reference['rows'][0], config,
            step=step, device=device, check=check)
        if rng.bit_generator.state != numpy_before or not torch.equal(torch.get_rng_state(), torch_before):
            raise ValueError('099 fixed T probe must not consume training sampler/Torch RNG')
        probes.append(result)
        diagnostic = state['training_diagnostics']
        diagnostic['completed_steps'].append(step)
        diagnostic['diagnostic_group_forwards'] += result['closed_loop_group_forwards']
        diagnostic['diagnostic_query_replay_forwards'] += result['query_replay_forwards']
        write_json(output/'training-probe.json', dict(status='COMPLETED' if step == training['updates'] else 'RUNNING',
            scope='fixed-source-step1-T-only-diagnostic', endpoints=probes,
            selection=False, test_opened=False, optimizer_updates=0))

    try:
        _save(output/'step-0.pt', model, optimizer, 0, config, rng, source)
        state['completed_checkpoints'].append('step-0.pt')
        train = prepare_evidence_scenes(train_raw, check=check, motion_observation_policy='canonical_all')
        if any(not len(scene['identities']) for scene in train):
            raise ValueError('099 cannot silently drop empty T scenes')
        probe(train, 0)
        state['preparation_seconds'] = time.monotonic()-started
        state['elapsed_seconds'] = state['preparation_seconds']
        fit_started = time.monotonic()
        _heartbeat(output, state, phase='training')
        with (output/'training.jsonl').open('x', buffering=1) as history:
            for step in range(1, training['updates']+1):
                if check:
                    check()
                cycle = (step-1)//training['source_updates']
                source_step = (step-1) % training['source_updates']+1
                if source_step == 1 and cycle:
                    # Adam/model/Torch RNG deliberately continue unchanged.
                    rng.bit_generator.state = copy.deepcopy(source['initial_numpy_rng'])
                    state['sampler_reset_count'] += 1
                old_row = reference['rows'][source_step-1]
                mode = TRAIN_MODES[(source_step-1) % len(TRAIN_MODES)]
                batch, inputs, targets, masks = [], [], [], []
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for _ in range(training['batch_size']):
                    scene = train[int(rng.integers(len(train)))]
                    anchor = int(rng.integers(len(scene['identities'])))
                    members = training_members(scene, anchor, mode)
                    valid_positions = scene['valid'][anchor, :, :2].all(-1)
                    support = scene['motion']['support']
                    eligible = ~support[anchor] & support[members[1:]].any(0)
                    batch.append(dict(scene_id=str(scene['scene_id']), anchor_index=anchor,
                        node_id=str(scene['node_ids'][anchor]), members=[int(node) for node in members],
                        valid_position_points=int(valid_positions.sum()),
                        eligible_query_points=int(eligible.sum()),
                        eligible_valid_position_points=int((eligible & valid_positions).sum())))
                    inputs.append(peer.deepaf_inputs(scene['motion'], members, device=device))
                    targets.append(scene['target'][anchor])
                    masks.append(scene['valid'][anchor])
                if mode != old_row['mode'] or batch != old_row['batch']:
                    raise ValueError(f'099 source batch differs before successful update {step}')
                # The source batch is checked before optimizer use. Supervision
                # is built only after the target-free closed-loop model rollout.
                arguments = peer.deepaf_batch_inputs(inputs)
                with torch.no_grad():
                    rollout = model.forward_batch(arguments, cache=True)
                target = torch.as_tensor(np.asarray(targets), dtype=torch.float64, device=device)
                valid = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=device)
                supervision = peer.deepaf_supervision(rollout, target, valid,
                    velocity_scale_mps=config['model']['velocity_scale_mps'])
                query_count = training['batch_size']*rollout['states'].shape[1]
                accumulated = dict(total=0., p_mse=0., v_mse=0.)
                raw_sse = dict(p_sse=0., v_sse=0.)
                expected_counts = [sample['valid_position_points'] for sample in batch]
                velocity_counts = None
                for start in range(0, query_count, REPLAY_QUERY_BATCH_SIZE):
                    if check:
                        check()
                    stop = min(start+REPLAY_QUERY_BATCH_SIZE, query_count)
                    predicted = model.replay_batch(rollout, start, stop)
                    losses, diagnostics = peer.deepaf_objectives(predicted, supervision, start=start, stop=stop)
                    if (set(losses) != set(accumulated)
                            or any(not isinstance(value, torch.Tensor) or value.ndim != 0
                                   or not torch.isfinite(value) for value in losses.values())):
                        raise FloatingPointError('099 P/V objective must contain finite scalar losses')
                    if diagnostics['valid_position_points'] != expected_counts:
                        raise ValueError('099 P/V objective changed valid position target counts')
                    current_counts = diagnostics['valid_velocity_points']
                    if velocity_counts is not None and current_counts != velocity_counts:
                        raise ValueError('099 P/V supervision counts changed between query chunks')
                    velocity_counts = current_counts
                    # Each chunk already has the full anchor/batch denominator.
                    losses['total'].backward()
                    for key in accumulated:
                        accumulated[key] += float(losses[key].detach())
                    for key in raw_sse:
                        raw_sse[key] += float(diagnostics[key])
                if (any(parameter.grad is None for parameter in parameters)
                        or not torch.stack([torch.isfinite(parameter.grad).all()
                                            for parameter in parameters]).all()):
                    raise FloatingPointError('099 every trainable P/V parameter requires a finite gradient')
                norm = torch.nn.utils.clip_grad_norm_(parameters, training['gradient_clip'], error_if_nonfinite=True)
                gradients = {prefix+'_nonzero_updates': any(bool(torch.count_nonzero(parameter.grad))
                    for name, parameter in model.named_parameters() if name.startswith(prefix+'.'))
                    for prefix in prefixes}
                optimizer.step()
                # Only a returned optimizer operation commits successful exposure.
                state['optimizer_updates'] = step
                state['actual_successful_batches_matched'] = step
                state['completed_replay_cycles'] = step//training['source_updates']
                _commit_exposure(state['exposure'], batch, mode)
                state['nonzero_gradient_updates'] += int(norm > 0)
                for key, nonzero in gradients.items():
                    state['network_gradients'][key] += int(nonzero)
                clipped = bool(norm > training['gradient_clip'])
                state['gradient_clipping']['clipped_updates'] += int(clipped)
                state['gradient_clipping']['successful_update_fraction'] = state['gradient_clipping']['clipped_updates']/step
                elapsed = time.monotonic()-started
                if elapsed >= training['advisory_seconds'] and not state['advisory_reached']:
                    print('ADVISORY: 099 fixed fit continues; no extension or retry.', flush=True)
                state.update(elapsed_seconds=elapsed, training_seconds=time.monotonic()-fit_started,
                    advisory_reached=elapsed >= training['advisory_seconds'],
                    peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0)
                row = dict(step=step, replay_cycle=cycle, source_step=source_step, mode=mode, batch=batch,
                    loss=accumulated['total'], objectives=dict(position_mse=accumulated['p_mse'],
                        velocity_mse=accumulated['v_mse'], **raw_sse,
                        valid_position_points=sum(expected_counts),
                        valid_velocity_points=sum(velocity_counts)),
                    gradient_norm=float(norm), gradient_clipped=clipped,
                    gradient_clip_scale=min(1., training['gradient_clip']/(float(norm)+1e-6)),
                    network_gradients=gradients, query_replay_chunks=math.ceil(query_count/REPLAY_QUERY_BATCH_SIZE))
                rows.append(row)
                history.write(json.dumps(row, allow_nan=False)+'\n')
                if not torch.stack([torch.isfinite(parameter).all() for parameter in parameters]).all():
                    raise FloatingPointError('099 nonfinite parameter after optimizer update')
                if source_step == training['source_updates']:
                    if rng.bit_generator.state != source['final_numpy_rng']:
                        raise ValueError('099 completed cycle sampler RNG differs from source final')
                    if state['exposure'] != _scaled(source_exposure, cycle+1):
                        raise ValueError('099 completed cycle sampling exposure differs from source')
                    filename = 'final.pt' if step == training['updates'] else f'step-{step}.pt'
                    _save(output/filename, model, optimizer, step, config, rng, source)
                    state['completed_checkpoints'].append(filename)
                    probe(train, step)
                    write_json(output/'run.json', state)
                if step % HEARTBEAT_INTERVAL == 0:
                    _heartbeat(output, state, phase='training')
        if _exact(_state(model), initial_state):
            raise ValueError('099 scratch training changed no model parameters')
        if state['sampler_reset_count'] != 1:
            raise ValueError('099 sampling RNG must reset exactly once between two cycles')
        state.update(status='COMPLETED', final_checkpoint='final.pt',
            final_sampler_rng_exact=True, sampling_exposure_exact=True,
            final_parameters_changed=True, elapsed_seconds=time.monotonic()-started,
            trained_scene_count=len(state['exposure']['scenes']))
        write_json(output/'training-windows.json', _windows(rows))
        write_json(output/'run.json', state)
        state['replay_verification'] = verify_replay(output, source_path)
        write_json(output/'run.json', state)
        _heartbeat(output, state, phase='completed')
        return state
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        write_json(output/'training-windows.json', _windows(rows))
        _heartbeat(output, state, phase=state['status'].lower())
        try:
            _save(output/'interrupted.pt', model, optimizer, state['optimizer_updates'], config, rng, source,
                recovery=True)
        except BaseException as save_error:
            state['recovery_checkpoint_error'] = f'{type(save_error).__name__}: {save_error}'
            write_json(output/'run.json', state)
        raise
