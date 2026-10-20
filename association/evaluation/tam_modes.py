"""Same-input, zero-update inspection of a frozen contrastive TAM snapshot.

This is a bounded development diagnostic, not an association benchmark. The
original completion, normalizer, margin and first-power contrastive loss are
reused. There is no reference, threshold selection, calibration or optimizer.
"""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np
import torch
from torch import nn


MODES = ('eval', 'train', 'train_all_dropout_zero')


def source_sample_plan(saved, manifest):
    """Choose two mixed-label scenes per split in original manifest order."""
    from association.data.archive import validate_manifest
    validate_manifest(manifest)
    if saved.get('format_version') != 1 or saved.get('input_adapter') != 'native_time_v1':
        raise ValueError('native_time_v1 complete checkpoint required')
    selected = saved.get('selected')
    if not isinstance(selected, dict) or selected.get('stage') != 'TAM_FROZEN':
        raise ValueError('selected snapshot must be from TAM_FROZEN')
    if (saved['config']['branch'] != 'contrastive'
            or saved['model_config'].get('contrastive_positive_power', 1) != 1):
        raise ValueError('diagnostic requires the original first-power contrastive branch')
    normalizer = saved['normalizer']
    data = {key: normalizer['adapter_config'][key] for key in
            ('length', 'max_observations', 'time_unit_seconds')}
    if (normalizer.get('input_adapter') != 'native_time_v1'
            or normalizer.get('training_splits') != ['T_core']
            or data['length'] != saved['model_config']['length']
            or data['max_observations'] != saved['max_observations']):
        raise ValueError('checkpoint normalizer/model/native adapter semantics disagree')
    contract = saved['data_contract']
    if not isinstance(contract, list) or len(contract) != 2:
        raise ValueError('checkpoint must retain training and V data contracts')
    result = []
    for split, population in zip(('T_core', 'V_select'), contract):
        if [row['scene_id'] for row in population] != manifest['splits'][split]:
            raise ValueError('manifest population/order differs from checkpoint')
        if split == 'T_core' and normalizer['training_scene_ids'] != manifest['splits'][split]:
            raise ValueError('normalizer training population differs from checkpoint')
        selected_rows, examined = [], []
        for row in population:
            sid = row['scene_id']
            if (row['split'] != split or row['component_id'] != manifest['components'][sid]
                    or row.get('input_adapter') != 'native_time_v1'
                    or row.get('max_observations') != data['max_observations']
                    or row.get('time_unit_seconds') != data['time_unit_seconds']):
                raise ValueError('checkpoint scene role/component/adapter disagrees with manifest')
            eligible = 0 < row['positive_count'] < row['pair_count']
            examined.append(dict(scene_id=sid, eligible=eligible,
                                 reason='mixed positive/negative candidates' if eligible else 'missing one pair class'))
            if eligible:
                selected_rows.append(row)
            if len(selected_rows) == 2:
                break
        if not selected_rows:
            raise ValueError(f'{split} has no scene with both pair classes')
        result.append(dict(split=split, requested_scenes=2, selected_scenes=len(selected_rows),
            shortfall=2-len(selected_rows), examined=examined, rows=selected_rows))
    return data, result


def prepare_fixed_samples(archive_path, manifest, saved, *, check=None):
    """Read only chosen measurements/labels, using the original native adapter."""
    from association.data.archive import MTADArchive
    from association.data.legacy_dataset import validate_archive
    from association.data.native import _record
    from association.data.timed import prepare_timed_scene
    from association.training.common import normalize_scene
    data, plan = source_sample_plan(saved, manifest)
    samples = []
    with MTADArchive(archive_path) as archive:
        validate_archive(archive, manifest)
        for population in plan:
            for row in population['rows']:
                if check:
                    check()
                native = _record(archive.load_scene(row['scene_id']), row['component_id'],
                                 population['split'], data['max_observations'])
                scene = prepare_timed_scene(native, **data)
                actual = dict(scene_id=scene['scene_id'], split=scene['split'],
                    component_id=scene['component_id'], grid=scene['grid'].tolist(),
                    node_ids=list(map(str, scene['node_ids'])), identities=list(map(str, scene['identities'])),
                    sources=list(map(str, scene['sources'])), pair_count=len(scene['pairs']),
                    positive_count=int(scene['labels'].sum()), intervals=scene['intervals'].tolist(),
                    observed_counts=scene['observed'].sum(1).tolist(),
                    native_observed_counts=scene['native_observed'].sum(1).tolist())
                if any(row[key] != value for key, value in actual.items()):
                    raise ValueError(f'prepared observation contract differs for scene {scene["scene_id"]}')
                # Fixed lexicographic candidate indices, never score-selected.
                indices = np.concatenate([np.flatnonzero(scene['labels'] == label)[:4]
                                          for label in (1, 0)])
                original_pairs = scene['pairs'][indices]
                nodes, inverse = np.unique(original_pairs, return_inverse=True)
                values, observed = normalize_scene(scene, saved['normalizer'])
                metadata = dict(split=scene['split'], scene_id=scene['scene_id'],
                    component_id=scene['component_id'], candidate_pair_indices=indices.tolist(),
                    original_pairs=original_pairs.tolist(), node_indices=nodes.tolist(),
                    node_ids=[scene['node_ids'][i] for i in nodes],
                    node_identities=[str(scene['identities'][i]) for i in nodes],
                    node_sources=[str(scene['sources'][i]) for i in nodes],
                    original_scene_node_count=len(scene['node_ids']),
                    positive_pairs=int(scene['labels'][indices].sum()),
                    negative_pairs=int((scene['labels'][indices] == 0).sum()),
                    original_full_scene_grid=scene['grid'].tolist(), reference_read=False)
                samples.append(dict(metadata=metadata, values=values[nodes], observed=observed[nodes],
                    pairs=inverse.reshape(-1, 2), labels=scene['labels'][indices]))
        members = list(archive.members_read)
    report = dict(data=data, rule='first two mixed-label scenes per split in manifest order; '
        'first four positive and four negative candidates in original lexicographic order; '
        'keep all if a class has fewer than four; original full-scene query grid retained',
        populations=[{key: value for key, value in p.items() if key != 'rows'} for p in plan],
        members_read=members, reference_members_read=0, test_opened=False)
    return samples, report


@contextmanager
def _preserved_inspection(model, device, seed, audit):
    """Restore mutable model/runtime state even after an inspection exception."""
    modules = list(model.modules())
    training = [module.training for module in modules]
    dropouts = [(module, module.p) for module in modules if isinstance(module, nn.Dropout)]
    attention = [(module, module.dropout) for module in modules if isinstance(module, nn.MultiheadAttention)]
    parameters = list(model.parameters())
    requires_grad = [parameter.requires_grad for parameter in parameters]
    gradients = [(parameter.grad, None if parameter.grad is None else parameter.grad.detach().clone())
                 for parameter in parameters]
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    fastpath = torch.backends.mha.get_fastpath_enabled()
    cpu_rng = torch.get_rng_state().clone()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    gpu_rng = [torch.cuda.get_rng_state(index).clone() for index in devices]
    try:
        with torch.random.fork_rng(devices=devices):
            # Seed only the CPU and requested GPU, not unrelated CUDA devices.
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            if devices:
                generator = torch.Generator(device=device).manual_seed(seed)
                torch.cuda.set_rng_state(generator.get_state(), devices[0])
            torch.backends.mha.set_fastpath_enabled(False)
            model.eval()
            model.requires_grad_(False)
            with torch.inference_mode():
                yield dropouts, attention
    finally:
        with torch.no_grad():
            current = model.state_dict()
            unchanged = all(torch.equal(value, current[key]) for key, value in state.items())
            if not unchanged:
                model.load_state_dict(state)
            gradients_unchanged = True
            for parameter, enabled, (original, cloned) in zip(parameters, requires_grad, gradients):
                if original is None:
                    gradients_unchanged &= parameter.grad is None
                else:
                    gradients_unchanged &= parameter.grad is original and torch.equal(parameter.grad, cloned)
                    original.copy_(cloned)
                parameter.grad = original
                parameter.requires_grad_(enabled)
            for module, value in zip(modules, training):
                module.training = value
            for module, probability in dropouts:
                module.p = probability
            for module, probability in attention:
                module.dropout = probability
            torch.backends.mha.set_fastpath_enabled(fastpath)
        audit.update(parameter_and_buffer_values_unchanged=unchanged,
            gradients_unchanged=bool(gradients_unchanged), requires_grad_restored=all(
                p.requires_grad == enabled for p, enabled in zip(parameters, requires_grad)),
            module_modes_restored=all(m.training == value for m, value in zip(modules, training)),
            dropout_probabilities_restored=all(m.p == value for m, value in dropouts)
                and all(m.dropout == value for m, value in attention),
            fastpath_restored=torch.backends.mha.get_fastpath_enabled() == fastpath,
            cpu_rng_restored=torch.equal(cpu_rng, torch.get_rng_state()),
            cuda_rng_restored=all(torch.equal(value, torch.cuda.get_rng_state(index))
                                  for index, value in zip(devices, gpu_rng)))


def _distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return dict(count=0, mean=None, std=None, quantiles=None)
    return dict(count=int(len(values)), mean=float(values.mean()), std=float(values.std()),
        quantiles={str(q): float(np.quantile(values, q)) for q in (0, .05, .25, .5, .75, .95, 1)})


def _loss_report(distances, labels, margin):
    positive, negative = distances[:, labels == 1], distances[:, labels == 0]
    positive_terms = .5 * positive
    negative_terms = .5 * np.maximum(margin-negative, 0.)**2
    return dict(positive_distance=_distribution(positive), negative_distance=_distribution(negative),
        positive_original_cl_term=_distribution(positive_terms),
        negative_original_cl_term=_distribution(negative_terms),
        original_sample_mean_loss=float((positive_terms.sum()+negative_terms.sum())/distances.size),
        class_balanced_loss=float(.5*(positive_terms.mean()+negative_terms.mean())),
        class_balanced_loss_per_repeat=(.5*(positive_terms.mean(1)+negative_terms.mean(1))).tolist(),
        negative_fraction_inside_margin=float((negative < margin).mean()))


def inspect_tam_modes(model, samples, *, device='cpu', batch_tracks=64, repeats=8, seed=2035, check=None):
    """Complete each fixed input once, then compare all TAM modes with no updates.

    Main pair distances use one independently sampled embedding per unique node
    per repeat, matching the training encoder/flatten route. A second complete
    pass of the *same* cached tracks uses independent masks for same-track noise.
    Losses are averaged over per-repeat distances, never MC-mean embeddings.
    """
    if model.branch != 'contrastive' or model.config.contrastive_positive_power != 1:
        raise ValueError('original first-power contrastive TAM required')
    if repeats < 2 or batch_tracks < 1 or not samples:
        raise ValueError('nonempty samples, repeats>=2 and positive batch_tracks required')
    device = torch.device(device)
    state_audit = {}
    arrays, sample_reports, modes = {}, [], {}
    with _preserved_inspection(model, device, seed, state_audit) as (dropouts, attention):
        completed = []
        for i, sample in enumerate(samples):
            if check:
                check()
            values = torch.as_tensor(sample['values'], device=device)
            observed = torch.as_tensor(sample['observed'], device=device)
            chunks = []
            for start in range(0, len(values), batch_tracks):
                if check:
                    check()
                chunks.append(model.complete(values[start:start+batch_tracks], observed[start:start+batch_tracks]))
            cached = torch.cat(chunks).detach()
            if not torch.isfinite(cached).all():
                raise FloatingPointError('nonfinite frozen completion')
            completed.append(cached)
            sample_reports.append(dict(**sample['metadata'], sample_index=i,
                local_pairs=np.asarray(sample['pairs']).tolist(), labels=np.asarray(sample['labels']).tolist(),
                completion_cached_once_per_node=True))
        for i, sample in enumerate(samples):
            arrays[f'sample_{i}_labels'] = np.asarray(sample['labels'])
            arrays[f'sample_{i}_pairs'] = np.asarray(sample['pairs'])
        tam_modules = list(model.tam_encoder.modules())
        tam_ids = {id(module) for module in tam_modules}
        for mode in MODES:
            model.eval()
            model.tam_encoder.train(mode != 'eval')
            for module, probability in dropouts:
                module.p = 0. if mode == 'train_all_dropout_zero' and id(module) in tam_ids else probability
            for module, probability in attention:
                module.dropout = 0. if mode == 'train_all_dropout_zero' and id(module) in tam_ids else probability
            if model.tcm_encoder.training or model.tcm_decoder.training:
                raise AssertionError('TCM must remain eval')
            per_sample = []
            for i, (sample, cached) in enumerate(zip(samples, completed)):
                pairs = torch.as_tensor(sample['pairs'], dtype=torch.long, device=device)
                labels = np.asarray(sample['labels'])
                distances, noise, variance, cl_losses = [], [], [], []
                for repeat in range(repeats):
                    if check:
                        check()
                    # Two independent forwards ensure same-track noise is not
                    # an embedding subtracted from itself or a reused mask.
                    encoded = [torch.cat([model.tam_encoder(cached[start:start+batch_tracks]).flatten(1)
                                for start in range(0, len(cached), batch_tracks)]) for _ in range(2)]
                    if not all(torch.isfinite(item).all() for item in encoded):
                        raise FloatingPointError('nonfinite TAM embedding')
                    scores = model.score_pairs(encoded[0], pairs)
                    distances.append((-scores).cpu().numpy().copy())
                    noise.append(torch.linalg.vector_norm(encoded[0]-encoded[1], dim=1).cpu().numpy().copy())
                    variance.append(float(encoded[0].double().var(dim=0, unbiased=False).mean().cpu()))
                    cl_losses.append(float(model.association_loss(scores,
                        torch.as_tensor(labels, device=device)).cpu()))
                distances, noise = np.stack(distances), np.stack(noise)
                arrays[f'{mode}_sample_{i}_distances'] = distances
                arrays[f'{mode}_sample_{i}_same_track_noise'] = noise
                arrays[f'{mode}_sample_{i}_cross_track_variance'] = np.asarray(variance)
                report = _loss_report(distances, labels, model.config.margin)
                report.update(sample_index=i, split=sample['metadata']['split'],
                    scene_id=sample['metadata']['scene_id'], same_track_noise=_distribution(noise),
                    cross_track_embedding_variance_per_repeat=variance,
                    model_original_loss_per_repeat=cl_losses,
                    raw_array_prefix=f'{mode}_sample_{i}',
                    repeat_pair_distance_std_mean=float(distances.astype(np.float64).std(axis=0).mean()))
                per_sample.append(report)
            by_split = {}
            for split in ('T_core', 'V_select'):
                members = [i for i, sample in enumerate(samples) if sample['metadata']['split'] == split]
                if members:
                    distances = np.concatenate([arrays[f'{mode}_sample_{i}_distances'] for i in members], axis=1)
                    labels = np.concatenate([np.asarray(samples[i]['labels']) for i in members])
                    noise = np.concatenate([arrays[f'{mode}_sample_{i}_same_track_noise'] for i in members], axis=1)
                    by_split[split] = _loss_report(distances, labels, model.config.margin)
                    by_split[split].update(scene_ids=[samples[i]['metadata']['scene_id'] for i in members],
                        same_track_noise=_distribution(noise),
                        scope='fixed small-sample diagnostic; not population performance or training-loss replay')
            modes[mode] = dict(tam_training=model.tam_encoder.training,
                tcm_encoder_training=model.tcm_encoder.training, tcm_decoder_training=model.tcm_decoder.training,
                tam_dropout_probabilities=[float(module.p) for module, _ in dropouts if id(module) in tam_ids],
                tam_attention_dropout_probabilities=[float(module.dropout) for module, _ in attention if id(module) in tam_ids],
                samples=per_sample, by_split=by_split)
        # Reproduce the source validation's default eval fastpath once on the
        # identical cached inputs. This reference is not a fourth MC arm.
        model.eval()
        for module, probability in dropouts:
            module.p = probability
        for module, probability in attention:
            module.dropout = probability
        torch.backends.mha.set_fastpath_enabled(True)
        original_path = []
        for i, (sample, cached) in enumerate(zip(samples, completed)):
            if check:
                check()
            encoded = torch.cat([model.tam_encoder(cached[start:start+batch_tracks]).flatten(1)
                for start in range(0, len(cached), batch_tracks)])
            pairs = torch.as_tensor(sample['pairs'], dtype=torch.long, device=device)
            distances = (-model.score_pairs(encoded, pairs)).cpu().numpy().copy()
            if not np.isfinite(distances).all():
                raise FloatingPointError('nonfinite default-fastpath eval distance')
            key = f'eval_default_fastpath_sample_{i}_distances'
            arrays[key] = distances
            controlled = arrays[f'eval_sample_{i}_distances'][0]
            original_path.append(dict(sample_index=i, split=sample['metadata']['split'],
                scene_id=sample['metadata']['scene_id'], raw_array=key,
                max_absolute_pair_distance_difference_from_controlled_eval=float(
                    np.max(np.abs(distances-controlled))),
                original_sample_mean_loss=float(model.association_loss(
                    -torch.as_tensor(distances, device=device),
                    torch.as_tensor(sample['labels'], device=device)).cpu())))
        comparisons = []
        for i, sample in enumerate(samples):
            eval_d = arrays[f'eval_sample_{i}_distances']
            zero_d = arrays[f'train_all_dropout_zero_sample_{i}_distances']
            comparisons.append(dict(sample_index=i, max_absolute_pair_distance_difference_eval_vs_train_zero=float(
                np.max(np.abs(eval_d-zero_d)))))
    if not all(state_audit.values()):
        raise RuntimeError(f'inspection changed model/runtime state: {state_audit}')
    report = dict(mode='same_input_zero_update_tam_inspection', optimizer_updates=0,
        repeats=repeats, seed=seed, effective_margin=float(model.config.margin),
        contrastive_positive_power=1, model_encoder_route='tam_encoder(cached_complete).flatten(1)',
        dropout_noise_route='independent second encoder pass on identical cached tracks',
        loss_aggregation='per-repeat distances; no MC-mean-embedding loss',
        fastpath_during_controlled_modes=False, autograd_during_inspection=False,
        inference_mode_during_inspection=True,
        default_fastpath_eval_reference=dict(repeats=1, mha_fastpath_enabled=True,
            inference_mode=True, requires_grad=False, samples=original_path,
            scope='same cached fixed inputs; numerical path check only, no threshold or population replay'),
        completion_mode='eval, once per unique node, reused across all modes/repeats',
        samples=sample_reports, modes=modes, comparisons=comparisons, state_restoration=state_audit,
        normalizer_refit=False, margin_recalibrated=False, threshold_reselected=False,
        reference_read=False, test_opened=False)
    return report, arrays
