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
META = ('scene_id', 'component_id', 'split', 'node_ids', 'identities', 'sources',
        'intervals', 'pairs', 'labels')
PACKED = (*META, 'positions', 'time_ids', 'observed', 'times', 'packing_audit')


def configuration(config):
    if (set(config) != {'scope', 'seed', 'model', 'data', 'training'}
            or config['scope'] != SCOPE or type(config['seed']) is not int or config['seed'] < 0):
        raise ValueError('explicit native TSA scope and seed required')
    model = TSAConfig(**config['model'])
    data, training = config['data'], config['training']
    if set(data) != {'max_points', 'time_bin_seconds', 'min_crop_fraction'}:
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
    torch.save(dict(scope=SCOPE, configuration=config, normalizer=normalizer,
        step=step, actual_updates=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None), path)


def load_checkpoint(path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = configuration(saved['configuration'])
    steps = saved.get('step')
    if (saved.get('scope') != SCOPE or type(steps) is not int
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


def train_native(train, normalizer, config, output, *, device='cpu', check=None):
    """One fixed fit; actual exposure commits only after a successful Adam call."""
    config = configuration(config)
    output = Path(output)
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
    run = dict(scope=SCOPE, status='RUNNING', configuration=config, normalizer=normalizer,
        optimizer_updates=0, training_scene_ids=ids, test_opened=False, reference_read=False,
        exposure=exposure, device=str(device), parameter_count=sum(p.numel() for p in model.parameters()),
        gradient_clipping=dict(clipped_updates=0), selection_policy='fixed final; V never supplied to fit')
    write_json(output/'run.json', run)
    _save(output/'step-0.pt', model, optimizer, config, normalizer, rng, 0)
    rows = []
    try:
        with (output/'training.jsonl').open('x') as log:
            for step in range(1, settings['updates']+1):
                if check:
                    check()
                tick = time.monotonic()
                task = 'ITSA' if step % 2 else 'MSTSA'
                batch = sample_batch(train, pools, task, settings['pair_batch_size'], rng,
                                     min_crop_fraction=config['data']['min_crop_fraction'])
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
                samples = []
                for identity, (si, left, right), window in zip(batch['identities'], batch['pairs'], batch['crop_windows']):
                    scene = train[int(si)]
                    sample = dict(scene_id=str(scene['scene_id']), identity=str(identity),
                        left_node=str(scene['node_ids'][int(left)]), right_node=str(scene['node_ids'][int(right)]),
                        left_index=int(left), right_index=int(right), crop=window)
                    samples.append(sample)
                    for field, key in (('scenes', sample['scene_id']), ('identities', str(identity)),
                                       ('pairs', json.dumps([sample['scene_id'], int(left), int(right)], separators=(',', ':')))):
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
                log.write(json.dumps(row)+'\n'); log.flush()
                rows.append({key: row[key] for key in ('step', 'task', 'loss', 'learning_rate', 'gradient_norm', 'seconds')})
                if step % 100 == 0:
                    write_json(output/'run.json', run)
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


def final_model(fit_dir, config, normalizer):
    """Bind actual completed calls, both heads, fitted moments and final weights."""
    fit_dir = Path(fit_dir)
    run = json.loads((fit_dir/'run.json').read_text())
    model, saved = load_checkpoint(fit_dir/'final.pt', device='cpu')
    steps = config['training']['updates']
    if (run.get('status') != 'COMPLETED' or run.get('configuration') != config
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
    counts = dict(successful_optimizer_calls=0, nonzero_lr_updates=0, positive_pair_exposures=0,
        track_exposures=0, token_exposures=0, task_calls={'ITSA': 0, 'MSTSA': 0},
        scenes={}, identities={}, pairs={}, cropped_pairs=0, fallback_pairs=0)
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
    return counts
