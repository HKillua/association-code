"""074D native-input, fixed-final TSA adaptation; never reads physical references.

This deliberately keeps the existing two-head InfoNCE recipe. Its finite budget,
uniform128 point cap and offline all-pair scores are not the author's online TSA.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from association.data.archive import MTADArchive
from association.data.native import SELECTION_POLICY, iter_native_split_from_archive
from association.utils.runtime import write_json
from baselines.tsactfer import TSAConfig, TSActfer, symmetric_info_nce
from baselines.tsactfer_native import _view
from baselines.tsactfer_data import fit_normalizer, pack_scene, build_positive_pools, sample_batch
from tools.experiments.tsactfer_fit import learning_rate, score_scenes

SCOPE = 'tsactfer-native-level1-final-v1'
FULL_SCOPE = 'tsactfer-native-level1-full-view-final-v1'
META = ('scene_id', 'component_id', 'split', 'node_ids', 'identities', 'sources',
        'intervals', 'pairs', 'labels')
PACKED = (*META, 'positions', 'time_ids', 'observed', 'times', 'packing_audit')


def configuration(config):
    if (set(config) != {'scope', 'seed', 'model', 'data', 'training'}
            or config['scope'] not in (SCOPE, FULL_SCOPE) or type(config['seed']) is not int or config['seed'] < 0):
        raise ValueError('explicit native TSA scope and seed required')
    model = TSAConfig(**config['model'])
    data, training = config['data'], config['training']
    expected_data = {'max_points', 'time_bin_seconds', 'min_crop_fraction'}
    if config['scope'] == FULL_SCOPE:
        expected_data.add('training_view')
        if data.get('training_view') != 'full_native':
            raise ValueError('082 requires explicit full_native training view')
    if set(data) != expected_data:
        raise ValueError('unexpected native TSA data configuration')
    if type(data['max_points']) is not int or not 2 <= data['max_points'] <= model.length:
        raise ValueError('native cap must fit the declared encoder slots')
    if set(training) != {'updates', 'warmup_updates', 'pair_batch_size', 'learning_rate',
                        'temperature', 'gradient_clip', 'inference_batch_tracks', 'advisory_seconds'}:
        raise ValueError('unexpected fixed-final TSA training configuration')
    for key in ('updates', 'warmup_updates', 'pair_batch_size', 'inference_batch_tracks'):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError('positive integer training.' + key + ' required')
    if (training['updates'] < 2 or training['updates'] % 2 or training['pair_batch_size'] < 2
            or training['warmup_updates'] >= training['updates']):
        raise ValueError('even calls, both tasks, and a shorter warmup required')
    for values, names in ((training, ('learning_rate', 'temperature', 'gradient_clip', 'advisory_seconds')),
                          (data, ('time_bin_seconds', 'min_crop_fraction'))):
        for key in names:
            if (isinstance(values[key], bool) or not isinstance(values[key], (int, float))
                    or not math.isfinite(values[key]) or values[key] <= 0):
                raise ValueError('positive finite ' + key + ' required')
    if data['min_crop_fraction'] > 1:
        raise ValueError('crop fraction cannot exceed one')
    return deepcopy(config)


def compact_native(native, max_points):
    """Retain only selected float64 observations, metadata and point-count audit.

    The private view performs the same canonical uniform selection as the
    existing native adapter. Full canonical arrays/tracks are not retained.
    """
    view, audit = _view(native, max_points)
    result = {key: view[key] for key in (*META, 'values', 'observed', 'observation_times', 'grid')}
    result['native_audit'] = dict(scene_id=str(native['scene_id']), fragments=len(audit),
        canonical_points=sum(row['canonical_points'] for row in audit),
        sampled_points=sum(row['sampled_points'] for row in audit),
        omitted_points=sum(row['omitted_points'] for row in audit),
        capped_fragments=sum(row['omitted_points'] > 0 for row in audit),
        selection_policy=SELECTION_POLICY)
    return result


def pack_compact(view, normalizer, config):
    """No clipping or scene-clock rescaling on a fixed timestamp vocabulary."""
    if view['split'] not in ('T_core', 'V_select'):
        raise ValueError('074D packs only T_core/V_select; T_test is sealed')
    if (view['split'] == 'V_select'
            and str(view['scene_id']) in normalizer['training_scene_ids']):
        raise ValueError('V was included in training normalizer')
    actual = view['observation_times'][view['observed']]
    bins = np.floor((actual-view['grid'][0]) / config['data']['time_bin_seconds'])
    if ((bins < 0) | (bins >= config['model']['time_bins']) | ~np.isfinite(bins)).any():
        raise ValueError('native timestamps exceed frozen vocabulary; do not clip or change clock')
    packed = pack_scene(view, normalizer, length=config['model']['length'],
        time_bin_seconds=config['data']['time_bin_seconds'], time_bins=config['model']['time_bins'])
    packed['packing_audit'].update(view['native_audit'], source_layout='canonical uniform capped actual times')
    return {key: packed[key] for key in PACKED}


def prepare_native_dataset(archives, manifest, config, *, verify_validation=None, check=None):
    """One streamed native read; no reference member and no whole-canonical list."""
    config = configuration(config)
    views, audit = {}, {}
    with MTADArchive(archives) as archive:
        for split in ('T_core', 'V_select'):
            views[split] = []
            for scene in iter_native_split_from_archive(archive, manifest, split,
                    max_points=config['data']['max_points'], check=check):
                if split == 'V_select' and verify_validation:
                    verify_validation(scene)
                views[split].append(compact_native(scene, config['data']['max_points']))
                # No packed/view record contains tracks, canonical_values or scene.audit.
                del scene
            if [str(s['scene_id']) for s in views[split]] != manifest['splits'][split]:
                raise ValueError('native full split/order differs from frozen manifest')
            audit[split] = [s['native_audit'] for s in views[split]]
        members = list(archive.members_read)
    normalizer = fit_normalizer(views['T_core'])
    normalizer.update(training_splits=['T_core'], native_max_points=config['data']['max_points'],
        native_selection_policy=SELECTION_POLICY,
        policy='T_core selected canonical lon/lat moments; no V or physical reference')
    train = [pack_compact(s, normalizer, config) for s in views.pop('T_core')]
    validation = [pack_compact(s, normalizer, config) for s in views.pop('V_select')]
    report = dict(training_scene_ids=normalizer['training_scene_ids'],
        validation_scene_ids=[str(s['scene_id']) for s in validation],
        normalizer=normalizer, packing=audit, archive_members_read=members,
        reference_read=False, test_opened=False, canonical_arrays_retained=False)
    return train, validation, normalizer, report


def _save(path, model, optimizer, config, normalizer, rng, step):
    torch.save(dict(scope=config['scope'], configuration=config, normalizer=normalizer,
        step=step, actual_updates=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None), path)


def load_checkpoint(path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = configuration(saved['configuration'])
    steps = saved.get('step')
    if (saved.get('scope') != config['scope'] or type(steps) is not int
            or not 0 <= steps <= config['training']['updates'] or steps % 2
            or saved.get('actual_updates') != steps
            or any(key not in saved for key in ('normalizer', 'optimizer', 'numpy_rng', 'torch_rng', 'cuda_rng'))):
        raise ValueError('TSA checkpoint endpoint/configuration/RNG binding differs')
    model = TSActfer(TSAConfig(**config['model'])).to(device)
    model.load_state_dict(saved['model'], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('TSA checkpoint has nonfinite weights')
    optimizer = saved['optimizer']
    groups = optimizer.get('param_groups', [])
    expected_lr = learning_rate(steps, config['training']) if steps else config['training']['learning_rate']
    names = [name for name, _ in model.named_parameters()]
    if (len(groups) != 1 or len(groups[0]['params']) != len(names)
            or groups[0]['lr'] != expected_lr or groups[0]['eps'] != 1e-6
            or tuple(groups[0]['betas']) != (.9, .999) or groups[0]['weight_decay'] != 0):
        raise ValueError('TSA optimizer configuration differs')
    state = optimizer['state']
    if steps == 0 and state:
        raise ValueError('TSA step0 optimizer cannot have calls')
    if steps:
        if set(state) != set(groups[0]['params']):
            raise ValueError('TSA optimizer parameter inventory differs')
        for name, index in zip(names, groups[0]['params']):
            count = steps//2 if name.startswith('heads.') else steps
            entry = state[index]
            if (float(entry.get('step', -1)) != count
                    or not all(torch.isfinite(entry[key]).all() for key in ('exp_avg', 'exp_avg_sq'))):
                raise ValueError('TSA shared/task-head optimizer call counts or state differ')
    return model, saved


def train_native(train, normalizer, config, output, *, device='cpu', check=None, reference_fit=None):
    """One fixed fit; actual exposure commits only after a successful Adam call."""
    config = configuration(config)
    full_view = config['scope'] == FULL_SCOPE
    if full_view != (reference_fit is not None):
        raise ValueError('082 requires074D reference fit;074D cannot use082 reference binding')
    if full_view:
        verify_reference_fit(reference_fit, config, normalizer)
    output = Path(output)
    if full_view and (Path(reference_fit).resolve().parent in output.resolve().parents
                      or output.resolve() in Path(reference_fit).resolve().parents):
        raise ValueError('082 output cannot overlap immutable074D')
    output.mkdir(parents=True, exist_ok=False)
    started, settings = time.monotonic(), config['training']
    ids = [str(s['scene_id']) for s in train]
    if (not ids or len(set(ids)) != len(ids) or any(s['split'] != 'T_core' for s in train)
            or normalizer['training_scene_ids'] != ids or normalizer.get('training_splits') != ['T_core']):
        raise ValueError('TSA fit requires complete distinct ordered T_core with its own normalizer')
    pools, pool_audit = build_positive_pools(train)
    write_json(output/'pool-audit.json', pool_audit)
    for task in ('ITSA', 'MSTSA'):
        if len(pools[task]) < settings['pair_batch_size']:
            raise ValueError(task+' lacks declared distinct identities; no automatic batch reduction')
    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    model = TSActfer(TSAConfig(**config['model'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
                                 betas=(.9, .999), eps=1e-6, weight_decay=0.)
    exposure = dict(successful_optimizer_calls=0, nonzero_lr_updates=0, positive_pair_exposures=0,
        track_exposures=0, token_exposures=0, task_calls={'ITSA': 0, 'MSTSA': 0},
        scenes={}, identities={}, pairs={}, cropped_pairs=0, fallback_pairs=0)
    if full_view:
        exposure.update(planned_token_exposures=0, planned_cropped_pairs=0, planned_fallback_pairs=0,
                        token_exposures_by_task={'ITSA': 0, 'MSTSA': 0})
    run = dict(scope=config['scope'], status='RUNNING', configuration=config, normalizer=normalizer,
        optimizer_updates=0, training_scene_ids=ids, test_opened=False, reference_read=False,
        exposure=exposure, device=str(device), parameter_count=sum(p.numel() for p in model.parameters()),
        gradient_clipping=dict(clipped_updates=0), selection_policy='fixed final; V never supplied to fit')
    if full_view:
        # Already validated before torch.manual_seed; do not instantiate another
        # model here, which would advance the paired dropout RNG stream.
        initial = torch.load(Path(reference_fit)/'step-0.pt', map_location='cpu', weights_only=False)
        if initial['step'] != 0 or set(initial['model']) != set(model.state_dict()) or any(
                not torch.equal(value.cpu(), initial['model'][name]) for name, value in model.state_dict().items()):
            raise ValueError('082 initial tensors differ from074D step0')
        run['pairing'] = dict(reference_fit=str(Path(reference_fit).resolve()), reference_scope=SCOPE,
            initial_parameters_equal=True, normalizer_equal=True, configuration_except_view_equal=True,
            successful_paired_updates=0, completed=False, crop_log_semantics='planned but not applied')
    write_json(output/'run.json', run)
    _save(output/'step-0.pt', model, optimizer, config, normalizer, rng, 0)
    rows = []
    reference_log = (Path(reference_fit)/'training.jsonl').open() if full_view else None
    try:
        with (output/'training.jsonl').open('x') as log:
            for step in range(1, settings['updates']+1):
                if check:
                    check()
                tick = time.monotonic()
                task = 'ITSA' if step % 2 else 'MSTSA'
                batch = sample_batch(train, pools, task, settings['pair_batch_size'], rng,
                                     min_crop_fraction=config['data']['min_crop_fraction'])
                samples = batch_samples(train, batch)
                planned = deepcopy(batch['augmentation'])
                if full_view:
                    source_row = json.loads(reference_log.readline())
                    verify_planned_batch(source_row, step, task, samples, planned, settings)
                    batch = full_native_batch(train, batch)
                lr = learning_rate(step, settings)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                model.train()
                inputs = [torch.as_tensor(batch[key], device=device) for key in ('positions', 'time_ids', 'observed')]
                loss = symmetric_info_nce(model(*inputs, task=task.lower()), settings['temperature'])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'],
                                                     error_if_nonfinite=True)
                optimizer.step()
                run['optimizer_updates'] = step
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('nonfinite weights after TSA Adam call')
                for sample in samples:
                    for field, key in (('scenes', sample['scene_id']), ('identities', sample['identity']),
                            ('pairs', json.dumps([sample['scene_id'], sample['left_index'], sample['right_index']], separators=(',', ':')))):
                        exposure[field][key] = exposure[field].get(key, 0)+1
                exposure['successful_optimizer_calls'] += 1
                exposure['nonzero_lr_updates'] += int(lr > 0)
                exposure['task_calls'][task] += 1
                exposure['positive_pair_exposures'] += len(samples)
                exposure['track_exposures'] += len(batch['observed'])
                exposure['token_exposures'] += int(batch['observed'].sum())
                for key in ('cropped_pairs', 'fallback_pairs'):
                    exposure[key] += batch['augmentation'][key]
                run['gradient_clipping']['clipped_updates'] += int(float(norm) > settings['gradient_clip'])
                row = dict(step=step, task=task, loss=float(loss.detach()), learning_rate=lr,
                    gradient_norm=float(norm), gradient_clip_scale=min(1., settings['gradient_clip']/(float(norm)+1e-6)),
                    seconds=time.monotonic()-tick, token_count=int(batch['observed'].sum()),
                    augmentation=batch['augmentation'], batch=samples)
                if full_view:
                    row.update(input_view='full_native', planned_augmentation=planned)
                    exposure['planned_token_exposures'] += planned['output_points']
                    exposure['planned_cropped_pairs'] += planned['cropped_pairs']
                    exposure['planned_fallback_pairs'] += planned['fallback_pairs']
                    exposure['token_exposures_by_task'][task] += row['token_count']
                    run['pairing']['successful_paired_updates'] += 1
                log.write(json.dumps(row)+'\n'); log.flush()
                rows.append({key: row[key] for key in ('step', 'task', 'loss', 'learning_rate', 'gradient_norm', 'seconds')})
                if step % 100 == 0:
                    write_json(output/'run.json', run)
        if full_view:
            if reference_log.readline() or run['pairing']['successful_paired_updates'] != settings['updates']:
                raise ValueError('082 paired source update count differs')
            run['pairing']['completed'] = True
        _save(output/'final.pt', model, optimizer, config, normalizer, rng, settings['updates'])
        windows = [dict(first_step=part[0]['step'], last_step=part[-1]['step'],
            mean_loss=float(np.mean([r['loss'] for r in part])),
            gradient_norm_mean=float(np.mean([r['gradient_norm'] for r in part])),
            clipped_calls=sum(r['gradient_norm'] > settings['gradient_clip'] for r in part),
            seconds=sum(r['seconds'] for r in part)) for begin in range(0, len(rows), 100)
            if (part := rows[begin:begin+100])]
        write_json(output/'training-windows.json', windows)
        exposure.update(unique_scenes=len(exposure['scenes']), unique_identities=len(exposure['identities']),
                        unique_pairs=len(exposure['pairs']), repeated_pair_exposures=exposure['positive_pair_exposures']-len(exposure['pairs']))
        run.update(status='COMPLETED', final_checkpoint='final.pt', elapsed_seconds=time.monotonic()-started,
            peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0)
        write_json(output/'run.json', run)
        return run
    except BaseException as error:
        run.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                   error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', run)
        raise
    finally:
        if reference_log is not None:
            reference_log.close()


def final_model(fit_dir, config, normalizer):
    """Bind actual completed calls, both heads, fitted moments and final weights."""
    fit_dir = Path(fit_dir)
    run = json.loads((fit_dir/'run.json').read_text())
    model, saved = load_checkpoint(fit_dir/'final.pt', device='cpu')
    steps = config['training']['updates']
    if (run.get('scope') != config['scope'] or run.get('status') != 'COMPLETED' or run.get('configuration') != config
            or run.get('normalizer') != normalizer or saved['configuration'] != config
            or saved['normalizer'] != normalizer or saved['step'] != steps
            or run.get('optimizer_updates') != steps or run.get('final_checkpoint') != 'final.pt'
            or run.get('test_opened') is not False or run.get('reference_read') is not False
            or run['exposure']['successful_optimizer_calls'] != steps
            or run['exposure']['task_calls'] != {'ITSA': steps//2, 'MSTSA': steps//2}):
        raise ValueError('completed TSA final, configuration, normalization or actual calls differ')
    verify_exposure(fit_dir, run)
    model.eval()
    return model, saved


def verify_exposure(fit_dir, run):
    """Rebuild actual successful-call exposure from every logged sampled pair."""
    config, original = run['configuration'], run['exposure']
    steps, batch_size = config['training']['updates'], config['training']['pair_batch_size']
    full_view = config['scope'] == FULL_SCOPE
    counts = dict(successful_optimizer_calls=0, nonzero_lr_updates=0, positive_pair_exposures=0,
        track_exposures=0, token_exposures=0, task_calls={'ITSA': 0, 'MSTSA': 0},
        scenes={}, identities={}, pairs={}, cropped_pairs=0, fallback_pairs=0)
    if full_view:
        counts.update(planned_token_exposures=0, planned_cropped_pairs=0, planned_fallback_pairs=0,
                      token_exposures_by_task={'ITSA': 0, 'MSTSA': 0})
    with (Path(fit_dir)/'training.jsonl').open() as handle:
        for step, line in enumerate(handle, 1):
            row = json.loads(line)
            task = 'ITSA' if step % 2 else 'MSTSA'
            samples = row['batch']
            if (row['step'] != step or step > steps or row['task'] != task
                    or row['learning_rate'] != learning_rate(step, config['training'])
                    or len(samples) != batch_size or len({s['identity'] for s in samples}) != batch_size
                    or not np.isfinite(row['loss']) or not np.isfinite(row['gradient_norm'])):
                raise ValueError('TSA actual successful-call sample log differs')
            if full_view:
                planned, actual = row['planned_augmentation'], row['augmentation']
                if (row.get('input_view') != 'full_native' or row['token_count'] != planned['input_points']
                        or actual != dict(pair_count=batch_size, cropped_pairs=0, fallback_pairs=0,
                            input_points=row['token_count'], output_points=row['token_count'])):
                    raise ValueError('082 planned and actual full input exposure differ')
                counts['planned_token_exposures'] += planned['output_points']
                counts['planned_cropped_pairs'] += planned['cropped_pairs']
                counts['planned_fallback_pairs'] += planned['fallback_pairs']
                counts['token_exposures_by_task'][task] += row['token_count']
            for sample in samples:
                if (sample['scene_id'] not in run['training_scene_ids']
                        or sample['left_node'] == sample['right_node']
                        or sample['left_index'] == sample['right_index']):
                    raise ValueError('TSA actual exposure contains unknown scene or self pair')
                for field, key in (('scenes', sample['scene_id']), ('identities', sample['identity']),
                        ('pairs', json.dumps([sample['scene_id'], sample['left_index'], sample['right_index']], separators=(',', ':')))):
                    counts[field][key] = counts[field].get(key, 0)+1
            counts['successful_optimizer_calls'] += 1
            counts['nonzero_lr_updates'] += int(row['learning_rate'] > 0)
            counts['positive_pair_exposures'] += batch_size
            counts['track_exposures'] += 2*batch_size
            counts['token_exposures'] += row['token_count']
            counts['task_calls'][task] += 1
            for key in ('cropped_pairs', 'fallback_pairs'):
                counts[key] += row['augmentation'][key]
    counts.update(unique_scenes=len(counts['scenes']), unique_identities=len(counts['identities']),
                  unique_pairs=len(counts['pairs']), repeated_pair_exposures=counts['positive_pair_exposures']-len(counts['pairs']))
    if counts['successful_optimizer_calls'] != steps or counts != original:
        raise ValueError('TSA actual exposure does not reproduce completed fit audit')
    if full_view:
        verify_completed_pairing(fit_dir, run)
    return counts


def crop_reference_configuration(config):
    result = configuration(config)
    if result['scope'] != FULL_SCOPE:
        raise ValueError('082 full-view profile required')
    result['scope'] = SCOPE
    result['data'].pop('training_view')
    return result


def batch_samples(train, batch):
    return [dict(scene_id=str(train[int(si)]['scene_id']), identity=str(identity),
        left_node=str(train[int(si)]['node_ids'][int(left)]),
        right_node=str(train[int(si)]['node_ids'][int(right)]),
        left_index=int(left), right_index=int(right), crop=window)
        for identity, (si, left, right), window in zip(batch['identities'], batch['pairs'], batch['crop_windows'])]


def full_native_batch(train, planned):
    """Restore full already-packed inputs after the original sampler consumes RNG.

    Positive refs/identities, planned crop metadata and their order are unchanged.
    No new draws, selection, interpolation, normalization or reference data occur.
    """
    batch = dict(planned)
    refs = planned['pairs']
    for key in ('positions', 'time_ids', 'times', 'observed'):
        batch[key] = np.stack([train[int(si)][key][int(node)]
            for side in (1, 2) for ref in refs
            for si, node in [(ref[0], ref[side])]])
    count = int(batch['observed'].sum())
    if count != planned['augmentation']['input_points']:
        raise ValueError('082 full input tokens differ from planned pre-crop population')
    batch['augmentation'] = dict(pair_count=len(refs), cropped_pairs=0, fallback_pairs=0,
                                input_points=count, output_points=count)
    return batch


def verify_planned_batch(source, step, task, samples, planned, settings):
    if (source['step'] != step or source['task'] != task or source['batch'] != samples
            or source['augmentation'] != planned or source['token_count'] != planned['output_points']
            or source['learning_rate'] != learning_rate(step, settings)):
        raise ValueError('082 actual batch/positive/crop plan differs from074D')


def verify_reference_fit(reference_fit, config, normalizer):
    path = Path(reference_fit)
    run = json.loads((path/'run.json').read_text())
    expected = crop_reference_configuration(config)
    if (run.get('scope') != SCOPE or run.get('status') != 'COMPLETED'
            or run.get('configuration') != expected or run.get('normalizer') != normalizer
            or run.get('training_scene_ids') != normalizer['training_scene_ids']
            or run.get('optimizer_updates') != config['training']['updates']
            or run.get('final_checkpoint') != 'final.pt' or run.get('test_opened') is not False
            or run.get('reference_read') is not False):
        raise ValueError('082 requires completed074D with identical configuration and T normalizer')
    verify_exposure(path, run)
    _, initial = load_checkpoint(path/'step-0.pt')
    if initial['step'] != 0 or initial['configuration'] != expected or initial['normalizer'] != normalizer:
        raise ValueError('082 reference step0 configuration/normalizer differ')
    return run


def verify_completed_pairing(fit_dir, run):
    pairing = run.get('pairing', {})
    steps = run['configuration']['training']['updates']
    if (pairing.get('completed') is not True or pairing.get('successful_paired_updates') != steps
            or pairing.get('initial_parameters_equal') is not True or pairing.get('normalizer_equal') is not True
            or pairing.get('configuration_except_view_equal') is not True or pairing.get('reference_scope') != SCOPE):
        raise ValueError('082 completed pairing is not bound')
    reference = Path(pairing['reference_fit'])
    verify_reference_fit(reference, run['configuration'], run['normalizer'])
    with (reference/'training.jsonl').open() as source, (Path(fit_dir)/'training.jsonl').open() as actual:
        for step, line in enumerate(actual, 1):
            row = json.loads(line)
            verify_planned_batch(json.loads(source.readline()), step, row['task'], row['batch'],
                row['planned_augmentation'], run['configuration']['training'])
        if source.readline() or step != steps:
            raise ValueError('082 completed pairing count differs')
    _, original = load_checkpoint(reference/'step-0.pt')
    _, initial = load_checkpoint(Path(fit_dir)/'step-0.pt')
    if any(not torch.equal(initial['model'][name], value) for name, value in original['model'].items()):
        raise ValueError('082 completed initial parameter pairing differs')
    return pairing
