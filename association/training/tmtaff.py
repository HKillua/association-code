"""Four-stage training on declared registered or native-time development data.

No archive reader, resampling, split construction, CLI, or test-set evaluation.
Selection is association macro-F1 only; it is not full-task optimality. A bounded
INTERRUPTED run can resume into a new directory without resetting elapsed cost.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from association.utils import runtime
from association.evaluation.tmtaff import evaluate_complete
from association.models.fusion import FusionLoss, PairFusion, PairFusionConfig
from association.evaluation.physical import _check_valid_states
from association.training.fusion import fusion_training_parameters, fusion_training_step
from association.evaluation.association import evaluate_scenes, select_threshold
from association.training.calibration import MARGIN_POLICY, calibrate_training_margin
from association.models.tmtaff import AssociationModel, ModelConfig
from association.training.common import fit_normalizer, normalize_scene, score_reference, _optimizer, _update
from association.training.sampling import BalancedScenePairSampler
from association.training.batches import FrozenCompletionCache, _pair_batch

STAGES = ('TCM', 'TAM_FROZEN', 'TFM_FROZEN', 'JOINT')


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _record_cuda_peaks(run, device):
    """Optional allocator telemetry must not replace a training exception."""
    if device.type != 'cuda' or not run['cuda_peak_reset_completed']:
        return
    for key, reader in (
            ('cuda_process_peak_allocated_bytes', torch.cuda.max_memory_allocated),
            ('cuda_process_peak_reserved_bytes', torch.cuda.max_memory_reserved)):
        try:
            run[key] = int(reader(device))
        except Exception as error:
            # Keep any earlier readable value; one failed counter must not
            # prevent reading the other. No retry or synchronization here.
            run.setdefault('cuda_memory_measurement_errors', {})[key] = f'{type(error).__name__}: {error}'


def _configuration(config):
    config = copy.deepcopy(config)
    config.setdefault('margin_policy', 'fixed')
    config.setdefault('longitude_loss_mode', 'linear')
    required = {'updates', 'learning_rates', 'seed', 'branch', 'batch_tracks',
                'pair_batch_size', 'pair_microbatch', 'track_microbatch',
                'completion_weight', 'gradient_clip', 'max_hours',
                'validate_every_updates', 'completion_cache_mib',
                'position_loss_mode', 'longitude_loss_mode', 'lr_schedule', 'singleton_policy',
                'validation_batches', 'margin_policy'}
    if set(config) != required:
        raise ValueError(f'config requires exactly {sorted(required)}')
    for field in ('updates', 'learning_rates'):
        if set(config[field]) != set(STAGES):
            raise ValueError(f'{field} must explicitly contain all four stages')
    for name, value in [('seed', config['seed']), ('completion_cache_mib', config['completion_cache_mib'])]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    for name, value in list(config['updates'].items()) + [(k, config[k]) for k in
            ('batch_tracks', 'pair_batch_size', 'pair_microbatch', 'track_microbatch', 'validate_every_updates')]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config['pair_batch_size'] % 2:
        raise ValueError('pair_batch_size must be even')
    for name, value in list(config['learning_rates'].items()) + [(k, config[k]) for k in
            ('completion_weight', 'gradient_clip', 'max_hours')]:
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value < 0 or (name != 'completion_weight' and value == 0)):
            raise ValueError(f'{name} must be finite and positive (completion_weight may be zero)')
    if config['branch'] not in ('ce', 'contrastive'):
        raise ValueError('branch must be ce or contrastive')
    if config['margin_policy'] not in ('fixed', MARGIN_POLICY):
        raise ValueError('unsupported margin_policy')
    if config['margin_policy'] == MARGIN_POLICY and config['branch'] != 'contrastive':
        raise ValueError('training margin calibration requires the contrastive branch')
    if config['position_loss_mode'] not in ('paper_printed', 'standard_smooth_l1'):
        raise ValueError('position_loss_mode must be explicit')
    if config['longitude_loss_mode'] not in ('linear', 'shortest_arc'):
        raise ValueError('longitude_loss_mode must be linear or shortest_arc')
    if config['lr_schedule'] not in ('constant', 'quarter_cosine'):
        raise ValueError('lr_schedule must be constant or quarter_cosine')
    if config['singleton_policy'] not in ('observed', 'missing'):
        raise ValueError('singleton_policy must be observed or missing')
    if (set(config['validation_batches']) != {'tracks', 'pairs'}
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in config['validation_batches'].values())):
        raise ValueError('validation_batches requires positive tracks and pairs sizes')
    return config


def _scenes(train, validation, length):
    """Validate declared physical time semantics before constructing a run."""
    if not train or not validation:
        raise ValueError('nonempty training and V_select scenes are required')
    adapters = {scene.get('input_adapter', 'registered_v1') for scene in [*train, *validation]}
    if len(adapters) != 1 or not adapters.issubset({'registered_v1', 'native_time_v1'}):
        raise ValueError('all train/validation scenes must use one explicit supported input adapter')
    timed = adapters == {'native_time_v1'}
    if timed:
        from association.data.timed import validate_timed_scene
        policies = {(s['max_observations'], s['time_unit_seconds']) for s in [*train, *validation]}
        if len(policies) != 1:
            raise ValueError('native observation limit and time unit must agree across train/validation')
    contracts, seen, components = [], set(), []
    for population, allowed in ((train, {'T_core', 'T_head'}), (validation, {'V_select'})):
        population_contract, component_ids = [], set()
        for scene in population:
            if timed:
                validate_timed_scene(scene)
            sid, split = scene['scene_id'], scene['split']
            component = scene['component_id']
            if (not isinstance(sid, str) or not sid or not isinstance(component, str)
                    or not component or split not in allowed or sid in seen):
                raise ValueError('invalid split/component or duplicate/overlapping scene')
            seen.add(sid)
            component_ids.add(component)
            values, observed = np.asarray(scene['values']), np.asarray(scene['observed'])
            grid, actual = np.asarray(scene['grid']), np.asarray(scene['observation_times'])
            if (values.ndim != 3 or values.shape[1:] != (length, 4)
                    or observed.shape != values.shape[:2] or observed.dtype != np.bool_
                    or (not timed and not observed.any(1).all()) or not np.isfinite(values[observed]).all()):
                raise ValueError('invalid observations or an empty effective fragment')
            _check_valid_states(values, np.repeat(observed[..., None], 4, axis=-1), 'observations')
            if (grid.shape != (length,) or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any()
                    or actual.shape != observed.shape or not np.isfinite(actual[observed]).all()
                    or not np.array_equal(actual[observed], np.broadcast_to(grid, observed.shape)[observed])):
                raise ValueError('actual observation times must equal the physical grid at every observed slot')
            for mask in observed:
                slots = np.flatnonzero(mask)
                if timed and not len(slots):
                    continue  # A short native fragment may have zero common-grid observations.
                if slots[-1] - slots[0] + 1 != len(slots):
                    raise ValueError('internal support gaps are not endpoint completion')
            count = len(values)
            for key in ('node_ids', 'identities', 'sources'):
                if np.asarray(scene[key]).shape != (count,):
                    raise ValueError(f'{key} must match the fragment count')
            if len(set(map(str, scene['node_ids']))) != count:
                raise ValueError('node_ids must be unique within a scene')
            if any(not isinstance(x, (str, np.str_)) or not x for x in scene['identities']):
                raise ValueError('node identities must be nonempty strings')
            pairs, labels = np.asarray(scene['pairs']), np.asarray(scene['labels'])
            if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu'
                    or labels.shape != (len(pairs),) or not np.isin(labels, (0, 1)).all()
                    or ((pairs < 0) | (pairs >= count)).any() or (pairs[:, 0] >= pairs[:, 1]).any()
                    or len(np.unique(pairs, axis=0)) != len(pairs)
                    or len(pairs) != count * (count - 1) // 2
                    or not np.array_equal(np.lexsort((pairs[:, 1], pairs[:, 0])), np.arange(len(pairs)))):
                raise ValueError('invalid unordered candidate pairs/labels')
            identities = np.asarray(scene['identities'])
            if not np.array_equal(labels.astype(bool), identities[pairs[:, 0]] == identities[pairs[:, 1]]):
                raise ValueError('pair labels disagree with identities')
            intervals = np.asarray(scene['intervals'])
            if intervals.shape != (count, 2) or not np.isfinite(intervals).all() or (intervals[:, 1] < intervals[:, 0]).any():
                raise ValueError('invalid fragment intervals')
            targets = scene['fusion_targets']
            if not set(map(str, identities)).issubset(targets):
                raise ValueError('fusion_targets must include every identity')
            for identity, item in targets.items():
                if not isinstance(identity, str) or not identity:
                    raise ValueError('target identities must be nonempty strings')
                query = np.asarray(item.get('query_times'))
                if query.shape != grid.shape or not np.array_equal(query, grid):
                    raise ValueError('fusion target query_times must equal the physical scene grid')
                target, valid = np.asarray(item['states']), np.asarray(item['valid'])
                if (target.shape != (length, 4) or valid.shape != target.shape or valid.dtype != np.bool_
                        or not np.isfinite(target[valid]).all() or (target[:, 2][valid[:, 2]] < 0).any()):
                    raise ValueError('invalid physical fusion target or feature mask')
                _check_valid_states(target, valid, 'fusion target')
            row = dict(scene_id=sid, split=split, component_id=component,
                grid=grid.tolist(), node_ids=list(map(str, scene['node_ids'])),
                identities=list(map(str, identities)), sources=list(map(str, scene['sources'])),
                pair_count=len(pairs), positive_count=int(labels.sum()), intervals=intervals.tolist(),
                observed_counts=observed.sum(1).tolist(), target_ids=list(targets),
                input_adapter=scene.get('input_adapter', 'registered_v1'))
            if timed:
                row.update(max_observations=scene['max_observations'],
                           time_unit_seconds=scene['time_unit_seconds'],
                           time_origin_seconds=scene['time_origin_seconds'],
                           native_observed_counts=np.asarray(scene['native_observed']).sum(1).tolist())
            if 'reference_alignment' in scene:
                row['reference_alignment'] = copy.deepcopy(scene['reference_alignment'])
            population_contract.append(row)
        contracts.append(population_contract)
        components.append(component_ids)
    if components[0] & components[1]:
        raise ValueError('training and validation components overlap')
    if not any(len(s['pairs']) for s in validation):
        raise ValueError('V_select requires candidate pairs')
    return contracts


def _states(modules):
    return [{key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for module in modules]


def _restore(modules, states):
    for module, state in zip(modules, states):
        module.load_state_dict(state)


def load_complete_snapshot(path, *, device='cpu', selected=True):
    """Restore an explicit complete snapshot with its original input semantics.

    The selected development snapshot is the default; ``selected=False`` loads
    the last checkpoint weights for inspection, not an optimizer resume. The
    trainer alone restores optimizer/RNG state and cumulative elapsed budget.
    """
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 1:
        raise ValueError('unsupported complete checkpoint format')
    adapter = payload.get('input_adapter', 'registered_v1')
    snapshot = payload['selected'] if selected else payload
    if snapshot is None:
        raise ValueError('checkpoint has no selected development snapshot yet')
    if (payload['config'].get('margin_policy', 'fixed') == MARGIN_POLICY
            and selected and 'effective_margin' not in snapshot):
        raise ValueError('calibrated selected snapshot is missing its effective margin')
    model_config = ModelConfig(**payload['model_config'])
    model_config = replace(model_config, margin=snapshot.get('effective_margin', model_config.margin))
    if adapter == 'native_time_v1':
        from association.models.timed import TimedAssociationModel
        model = TimedAssociationModel(model_config, payload['config']['branch'],
                                     max_observations=payload['max_observations'])
    elif adapter == 'registered_v1':
        model = AssociationModel(model_config, payload['config']['branch'])
    else:
        raise ValueError('unsupported checkpoint input adapter')
    fusion = PairFusion(PairFusionConfig(**payload['fusion_config']), payload['normalizer'])
    objective = FusionLoss(position_loss_mode=payload['config']['position_loss_mode'],
                           longitude_loss_mode=payload['config'].get('longitude_loss_mode', 'linear'))
    _restore((model, fusion, objective), snapshot['states'])
    for module in (model, fusion, objective):
        module.to(device).eval()
    return dict(model=model, fusion=fusion, objective=objective,
                normalizer=payload['normalizer'], input_adapter=adapter,
                selection=({key: value for key, value in snapshot.items() if key != 'states'}
                           if selected else None))


def _compatible_device(previous, requested):
    """A compatible GPU index may change; CPU/GPU fallback is never implicit."""
    return torch.device(previous).type == torch.device(requested).type


def _rng_state(track_rng, tfm_rng, sampler, device):
    numpy = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [numpy[0], numpy[1].tolist(), *numpy[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            'tracks': track_rng.bit_generator.state, 'tfm': tfm_rng.bit_generator.state,
            'pairs': sampler._rng.bit_generator.state}


def _restore_rng(state, track_rng, tfm_rng, sampler, device):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)
    track_rng.bit_generator.state = state['tracks']
    tfm_rng.bit_generator.state = state['tfm']
    sampler._rng.bit_generator.state = state['pairs']


def _stage_parameters(modules, stage):
    model, fusion, objective = modules
    if stage in ('TFM_FROZEN', 'JOINT'):
        return fusion_training_parameters(*modules, stage=stage)
    model.requires_grad_(True)
    fusion.requires_grad_(False)
    objective.requires_grad_(False)
    model.set_completion_trainable(stage == 'TCM', include_embedding=True)
    if stage == 'TCM':
        model.input_embedding.requires_grad_(False)
        parameters = list(model.completion_parameters())
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]
    model.train()
    for module in modules:
        module.zero_grad(set_to_none=True)
    return parameters


def _targets(scenes, groups, device):
    states, masks = [], []
    for scene_index, indices in groups:
        scene = scenes[scene_index]
        for index in indices:
            if scene['labels'][index] == 1:
                left = scene['pairs'][index][0]
                item = scene['fusion_targets'][str(scene['identities'][left])]
                states.append(item['states'])
                masks.append(item['valid'])
    return (torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device))


def train_complete(train_scenes, validation_scenes, model_config, fusion_config, config,
                   output, *, device='cpu', resume_from=None, max_updates_this_call=None,
                   evaluation_fn=None, initial_elapsed_seconds=0.0, save_predictions=False):
    """Train supplied physical scenes; native ZIP preparation is a separate caller.

    ``fusion_targets`` maps identity strings to states[L,4]/valid[L,4] and
    query_times[L], explicitly equal to the scene's physical grid.
    A fresh output directory is always required. Only explicitly INTERRUPTED
    runs resume; FAILED attempts need separate authorization, not an automatic
    retry of their last checkpoint. Optional evaluation_fn is called at the end
    with the same interface as evaluate_complete. The default full evaluation
    uses the selected complete snapshot and is included in the cumulative cost.
    Resume requires the same frozen inputs. Compact semantic metadata is checked;
    no full dataset snapshot/hash is stored to detect arbitrary content changes.
    Explicit save_predictions writes selected-state physical predictions during
    the default final evaluation, with its I/O included in the same budget.
    """
    started = time.monotonic()
    config = _configuration(config)
    if not isinstance(save_predictions, bool):
        raise ValueError('save_predictions must be boolean')
    if save_predictions and evaluation_fn is not None:
        raise ValueError('save_predictions requires the default complete evaluator')
    if (isinstance(initial_elapsed_seconds, bool) or not isinstance(initial_elapsed_seconds, (int, float))
            or not math.isfinite(initial_elapsed_seconds) or initial_elapsed_seconds < 0):
        raise ValueError('initial_elapsed_seconds must be finite and nonnegative')
    if max_updates_this_call is not None and (isinstance(max_updates_this_call, bool)
            or not isinstance(max_updates_this_call, int) or max_updates_this_call < 1):
        raise ValueError('max_updates_this_call must be a positive integer')
    if (model_config.length, model_config.d_model) != (fusion_config.length, fusion_config.d_model):
        raise ValueError('association and fusion length/dimension must agree')
    contract = _scenes(train_scenes, validation_scenes, model_config.length)
    calibrating = config['margin_policy'] == MARGIN_POLICY
    if calibrating and (model_config.contrastive_positive_power != 1
                        or any(scene['split'] != 'T_core' for scene in train_scenes)):
        raise ValueError('training margin calibration requires p=1 and only T_core scenes')
    normalizer = fit_normalizer(train_scenes)
    sampler = BalancedScenePairSampler(train_scenes, config['seed'])
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('an available explicit cpu/cuda device is required')
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    elif device.type == 'cpu':
        device = torch.device('cpu')
    output = Path(output)
    saved, spent_before = None, 0.0
    if resume_from is not None:
        previous = Path(resume_from)
        previous_run = runtime.read_json(previous / 'run.json')
        if previous_run.get('status') != 'INTERRUPTED':
            raise ValueError('only an INTERRUPTED run can resume; failed runs are not automatic retries')
        saved = torch.load(previous / 'checkpoint.pt', map_location='cpu', weights_only=True)
        if (_configuration(saved['config']) != config or saved['model_config'] != asdict(model_config)
                or asdict(PairFusionConfig(**saved['fusion_config'])) != asdict(fusion_config)
                or saved['data_contract'] != contract
                or saved['normalizer'] != normalizer or not _compatible_device(saved['device'], device)):
            raise ValueError('resume configuration, registered scene contract, normalizer or device differs')
        if calibrating:
            calibration = saved.get('margin_calibration')
            done = calibration is not None and calibration.get('status') == 'COMPLETED'
            if done:
                if (calibration.get('policy') != MARGIN_POLICY
                        or calibration.get('effective_margin') != saved.get('effective_margin')):
                    raise ValueError('resume margin calibration disagrees with effective margin')
            elif calibration is not None:
                raise ValueError('resume checkpoint contains incomplete margin calibration')
            elif (saved['cursor']['stage'] > 1
                  or (saved['cursor']['stage'] == 1 and saved['cursor']['update'] > 0)):
                raise ValueError('cannot resume trained TAM without its completed margin calibration')
        spent_before = max(float(saved['elapsed_seconds']), float(previous_run['elapsed_seconds']))
    if spent_before + initial_elapsed_seconds >= config['max_hours'] * 3600:
        raise TimeoutError('cumulative training budget is exhausted')
    output.mkdir(parents=True, exist_ok=False)
    history = [] if saved is None else copy.deepcopy(saved['history'])
    coverage = {stage: {'scenes': [], 'tracks': [], 'pairs': [], 'track_presentations': 0,
                       'pair_presentations': 0, 'positive_pair_presentations': 0, 'scene_ids': [],
                       'unique_scene_count': 0, 'unique_track_count': 0, 'unique_pair_count': 0,
                       'positive_scene_ids': [], 'negative_scene_ids': []} for stage in STAGES} \
               if saved is None else copy.deepcopy(saved['coverage'])
    cursor = {'stage': 0, 'update': 0, 'completed_updates': 0} if saved is None else dict(saved['cursor'])
    selected = None if saved is None else saved['selected']
    calibration = None if saved is None else copy.deepcopy(saved.get('margin_calibration'))
    effective_margin = model_config.margin if saved is None else saved.get('effective_margin', model_config.margin)
    adapter = train_scenes[0].get('input_adapter', 'registered_v1')
    run = {'status': 'RUNNING', 'config': config, 'input_adapter': adapter,
           'requested_model_config': asdict(model_config), 'effective_margin': effective_margin,
           'margin_calibration': calibration,
           'scope': (('TMTAFF time/output-scale-adapted approximate reference'
                      if fusion_config.output_parameterization == 'train_position_affine'
                      else 'TMTAFF time-adapted approximate reference') if adapter == 'native_time_v1'
                     else 'registered-data training kernel'),
           'fusion_output_parameterization': fusion_config.output_parameterization,
           'selection_scope': 'association V_select macro-F1 only; not full-task optimality',
           'device': str(device),
           'resume_device_changed': saved is not None and saved['device'] != str(device),
           'resume_reproducibility': 'bitwise replay requires the same frozen inputs, hardware and software; compatible GPU changes are permitted',
           'sampling': {'association': sampler.describe(),
                        'tcm': 'uniform over training fragments with at least two observed points',
                        'tfm': 'uniform positive-eligible scene, then uniform positive pair; replacement iff necessary'},
           'test_opened': False, 'resume_from': str(resume_from) if resume_from is not None else None,
           'validation_timings': [],
           'validation_timing_scope': 'current invocation; full scoring, threshold selection and selected snapshot CPU copy/metrics; excludes checkpoint I/O',
           'final_evaluation_seconds': None, 'final_evaluation_status': 'NOT_STARTED',
           'final_evaluation_scope': 'selected state restore, full association/fusion evaluation and threshold consistency check; excludes summary I/O',
           'timing_note': 'successful CUDA boundaries synchronize; failed timings stop at exception without waiting for outstanding CUDA work',
           'cuda_process_peak_allocated_bytes': None, 'cuda_process_peak_reserved_bytes': None,
           'cuda_peak_reset_completed': False if device.type == 'cuda' else None,
           'cuda_memory_scope': 'current invocation, selected device, process PyTorch allocator peak since pre-model reset; includes existing live allocations/cache; not model-only, whole-device usage or cumulative across resume'}

    def elapsed():
        return spent_before + initial_elapsed_seconds + time.monotonic() - started

    def check():
        if elapsed() >= config['max_hours'] * 3600:
            raise TimeoutError('cumulative training budget exceeded')

    def write_run(status, **extra):
        _record_cuda_peaks(run, device)
        run.update(status=status, elapsed_seconds=elapsed(), cursor=dict(cursor), **extra)
        runtime.write_json(output / 'run.json', run)

    try:
        write_run('RUNNING')
        check()
        runtime.seed_everything(config['seed'])
        random.seed(config['seed'])
        if device.type == 'cuda':
            _synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            run['cuda_peak_reset_completed'] = True
        if adapter == 'native_time_v1':
            from association.models.timed import TimedAssociationModel
            model = TimedAssociationModel(replace(model_config, margin=effective_margin), config['branch'],
                                         max_observations=train_scenes[0]['max_observations'])
        else:
            model = AssociationModel(replace(model_config, margin=effective_margin), config['branch'])
        modules = (model.to(device),
                   PairFusion(fusion_config, normalizer).to(device),
                   FusionLoss(position_loss_mode=config['position_loss_mode'],
                              longitude_loss_mode=config['longitude_loss_mode']).to(device))
        model, fusion, objective = modules
        normalized = [normalize_scene(scene, normalizer) for scene in train_scenes]
        track_rng, tfm_rng = np.random.default_rng(config['seed']), np.random.default_rng(config['seed'] + 1)
        transitions = [(s, t) for s, (_, masks) in enumerate(normalized)
                       for t, mask in enumerate(masks) if mask.sum() >= 2]
        optimizer, cache, parameters = None, None, None
        if saved is not None:
            if selected is not None:
                # A selected snapshot may not be loaded until final evaluation.
                # Validate it before resuming any optimizer updates as well.
                fusion.validate_output_statistics(selected['states'][1])
            _restore(modules, saved['states'])
            if cursor['stage'] < len(STAGES):
                phase = STAGES[cursor['stage']]
                parameters = _stage_parameters(modules, phase)
                optimizer = _optimizer(parameters, {'learning_rate': config['learning_rates'][phase]})
                optimizer.load_state_dict(saved['optimizer'])
            _restore_rng(saved['rng'], track_rng, tfm_rng, sampler, device)

        def checkpoint():
            check()
            payload = {'format_version': 1, 'config': config, 'model_config': asdict(model_config),
                       'effective_margin': effective_margin, 'margin_calibration': calibration,
                       'input_adapter': adapter,
                       'max_observations': train_scenes[0].get('max_observations'),
                       'fusion_config': asdict(fusion_config), 'normalizer': normalizer,
                       'data_contract': contract, 'device': str(device), 'states': _states(modules),
                       'optimizer': optimizer.state_dict() if optimizer is not None else None,
                       'cursor': dict(cursor), 'history': history, 'coverage': coverage,
                       'selected': selected, 'rng': _rng_state(track_rng, tfm_rng, sampler, device),
                       'elapsed_seconds': elapsed()}
            temporary = output / 'checkpoint.pt.tmp'
            torch.save(payload, temporary)
            temporary.replace(output / 'checkpoint.pt')
            runtime.write_json(output / 'history.json', history)
            runtime.write_json(output / 'coverage.json', coverage)
            write_run('RUNNING')

        calls = 0
        while cursor['stage'] < len(STAGES):
            phase = STAGES[cursor['stage']]
            total_updates = config['updates'][phase]
            if cursor['update'] == total_updates:
                cursor.update(stage=cursor['stage'] + 1, update=0)
                optimizer, cache = None, None
                continue
            if phase == 'TCM' and not transitions:
                history.append({'stage': phase, 'status': 'SKIPPED', 'reason': 'no observed temporal transitions',
                                'completed_updates': 0})
                cursor['update'] = total_updates
                continue
            if optimizer is None:
                if phase == 'TFM_FROZEN':
                    model.load_state_dict(selected['states'][0])
                parameters = _stage_parameters(modules, phase)
                optimizer = _optimizer(parameters, {'learning_rate': config['learning_rates'][phase]})
            if phase == 'TAM_FROZEN' and cache is None:
                cache = FrozenCompletionCache(model, normalized, device,
                    config['completion_cache_mib'] * 1024 * 1024, config['batch_tracks'], check)
            if phase == 'TAM_FROZEN' and calibrating and calibration is None:
                if cursor['update'] != 0:
                    raise ValueError('margin calibration must precede the first TAM update')
                calibration = {}
                run['margin_calibration'] = calibration
                effective_margin = calibrate_training_margin(model, train_scenes, normalized, cache,
                    device=device, batch_tracks=config['batch_tracks'], check=check, report=calibration)
                model.config = replace(model_config, margin=effective_margin)
                run['effective_margin'] = effective_margin
                # Durable boundary: a later interruption resumes the same m,
                # before any main pair-sampler draw or TAM parameter update.
                checkpoint()
            check()
            update = cursor['update'] + 1
            plateau = max(1, math.ceil(total_updates / 4))
            factor = (1.0 if config['lr_schedule'] == 'constant' or update <= plateau else
                      0.5 * (1 + math.cos(math.pi * (update - plateau) / (total_updates - plateau))))
            lr = config['learning_rates'][phase] * factor
            for group in optimizer.param_groups:
                group['lr'] = lr
            step_started = time.monotonic()
            groups = []
            if phase == 'TCM':
                model.train()
                indices = track_rng.choice(len(transitions), config['batch_tracks'],
                                           replace=len(transitions) < config['batch_tracks'])
                keys = [transitions[int(i)] for i in indices]
                x = torch.as_tensor(np.stack([normalized[s][0][t] for s, t in keys]), device=device)
                mask = torch.as_tensor(np.stack([normalized[s][1][t] for s, t in keys]), device=device)
                loss, norm = _update(model.completion_loss(x, mask), optimizer, parameters, config['gradient_clip'])
                row = {'loss': loss, 'gradient_norm': norm, 'track_presentations': len(keys)}
            else:
                if phase == 'TFM_FROZEN':
                    si = int(tfm_rng.choice(sampler._eligible[1]))
                    pool = sampler._pools[1][si]
                    groups = [(si, tfm_rng.choice(pool, config['pair_batch_size'],
                                                 replace=len(pool) < config['pair_batch_size']))]
                else:
                    groups = sampler.sample(config['pair_batch_size'])
                keys, x, mask, pairs, labels, metadata = _pair_batch(train_scenes, normalized, groups, device)
                if phase == 'TAM_FROZEN':
                    model.train()
                    complete = cache.get(keys)
                    encoded = torch.cat([model.tam_encoder(complete[i:i + config['batch_tracks']]).flatten(1)
                                         for i in range(0, len(keys), config['batch_tracks'])])
                    loss = model.association_loss(model.score_pairs(encoded, pairs), labels)
                    loss, norm = _update(loss, optimizer, parameters, config['gradient_clip'])
                    row = {'loss': loss, 'association_loss': loss, 'gradient_norm': norm}
                else:
                    target, valid = _targets(train_scenes, groups, device)
                    row = fusion_training_step(*modules, optimizer, stage=phase, values=x, observed=mask,
                        pairs=pairs, labels=labels, target=target, valid=valid,
                        pair_microbatch=config['pair_microbatch'], track_microbatch=config['track_microbatch'],
                        completion_weight=config['completion_weight'] if phase == 'JOINT' else 0,
                        gradient_clip=config['gradient_clip'], check=check)
                row['sampling'] = sampler.batch_metadata(groups)
                row['pair_count'], row['positive_pairs'] = metadata['pair_count'], metadata['positive_pairs']
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            row.update(stage=phase, update=update, learning_rate=lr, seconds=time.monotonic() - step_started)
            cursor.update(update=update, completed_updates=cursor['completed_updates'] + 1)
            item = coverage[phase]
            item['scenes'] = sorted(set(item['scenes']) | {s for s, _ in keys})
            item['tracks'] = sorted(set(map(tuple, item['tracks'])) | set(keys))
            item['track_presentations'] += len(keys)
            item['pairs'] = sorted(set(map(tuple, item['pairs'])) |
                                   {(s, int(i)) for s, indices in groups for i in indices})
            item['pair_presentations'] += row.get('pair_count', 0)
            item['positive_pair_presentations'] += row.get('positive_pairs', 0)
            item['scene_ids'] = [train_scenes[s]['scene_id'] for s in item['scenes']]
            item['unique_scene_count'], item['unique_track_count'], item['unique_pair_count'] = (
                len(item['scenes']), len(item['tracks']), len(item['pairs']))
            for label, name in ((1, 'positive_scene_ids'), (0, 'negative_scene_ids')):
                item[name] = sorted({train_scenes[s]['scene_id'] for s, i in item['pairs']
                                     if train_scenes[s]['labels'][i] == label})
            if phase in ('TAM_FROZEN', 'JOINT') and (update % config['validate_every_updates'] == 0 or update == total_updates):
                _synchronize(device)
                validation_started = time.monotonic()
                timing = {'stage': phase, 'update': update, 'status': 'FAILED',
                          'seconds': None, 'selected_snapshot_copied': False}
                run['validation_timings'].append(timing)
                try:
                    records = score_reference(model, normalizer, validation_scenes, device,
                        config['validation_batches']['tracks'], config['validation_batches']['pairs'], check=check)
                    selection = select_threshold(records)
                    row['selection'] = selection
                    if selected is None or selection['macro_f1'] > selected['macro_f1'] + 1e-12:
                        selected = {'macro_f1': selection['macro_f1'], 'threshold': selection['threshold'],
                                    'effective_margin': effective_margin,
                                    'margin_policy': config['margin_policy'],
                                    'stage': phase, 'update': update, 'states': _states(modules),
                                    'validation': evaluate_scenes(records, selection['threshold'])}
                        timing['selected_snapshot_copied'] = True
                    _synchronize(device)
                    timing['status'] = 'COMPLETED'
                finally:
                    row['validation_seconds'] = timing['seconds'] = time.monotonic() - validation_started
            if phase == 'TFM_FROZEN' and update == total_updates:
                selected = {**selected, 'states': _states(modules), 'fusion_stage': phase, 'fusion_update': update}
            history.append(row)
            calls += 1
            interrupted = max_updates_this_call is not None and calls >= max_updates_this_call
            if update % config['validate_every_updates'] == 0 or update == total_updates or interrupted:
                checkpoint()
            if interrupted:
                if not (phase == 'JOINT' and update == total_updates):
                    write_run('INTERRUPTED')
                    return run
        _synchronize(device)
        evaluation_started = time.monotonic()
        run['final_evaluation_status'] = 'FAILED'
        try:
            _restore(modules, selected['states'])
            model.eval()
            fusion.eval()
            evaluator = evaluate_complete if evaluation_fn is None else evaluation_fn
            prediction_options = {}
            if save_predictions:
                from association.evaluation.artifacts import ScenePredictionWriter
                prediction_options['prediction_writer'] = ScenePredictionWriter(output / 'predictions')
                run['prediction_artifacts'] = 'predictions/index.json'
            def evaluation_progress(item):
                run['final_evaluation_progress'] = item
                runtime.write_json(output / 'evaluation-progress.json', item)
            evaluation = evaluator(model, fusion, normalizer, validation_scenes, device=device,
                batch_tracks=config['validation_batches']['tracks'],
                pair_batch_size=config['validation_batches']['pairs'],
                select_on_validation=True, threshold=None,
                singleton_policy=config['singleton_policy'], check=check,
                **({'progress': evaluation_progress} if evaluation_fn is None else {}),
                **prediction_options)
            final_threshold, saved_threshold = evaluation['threshold'], selected['threshold']
            if ((final_threshold is None) != (saved_threshold is None)
                    or (final_threshold is not None and not math.isclose(
                        final_threshold, saved_threshold, rel_tol=0, abs_tol=1e-7))):
                raise ValueError('selected snapshot final threshold differs from its saved validation threshold')
            _synchronize(device)
            run['final_evaluation_status'] = 'COMPLETED'
        finally:
            run['final_evaluation_seconds'] = time.monotonic() - evaluation_started
        check()
        runtime.write_json(output / 'summary.json', {
            'selection': {k: v for k, v in selected.items() if k != 'states'},
            'selection_scope': run['selection_scope'], 'evaluation': evaluation,
            'evaluation_seconds': run['final_evaluation_seconds'],
            'normalizer': normalizer, 'coverage': coverage, 'test_opened': False,
            **({'prediction_artifacts': 'predictions/index.json'} if save_predictions else {})})
        check()
        write_run('COMPLETED')
        return run
    except BaseException as error:
        # Never replace the last successful checkpoint with partially updated
        # parameters/RNG. FAILED runs cannot silently resume as a fresh budget.
        write_run('FAILED', failure_type=type(error).__name__, failure=str(error))
        raise
