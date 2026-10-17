"""Update-bounded, class-balanced TCM -> frozen TAM -> joint training.

The explicit probe purpose cannot produce an eligible reference. Completion
caches are run-local CPU tensors owned by a single frozen stage, never TAM
features or a persistent shortcut around checkpoint provenance.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import signal
import threading
import time

import numpy as np
import torch

from . import runtime
from .config import validate_manifest_config
from .dataset import load_split
from .metrics import evaluate_scenes, select_threshold
from .models import AssociationModel
from .reference import (_binding, _check_configuration, _coverage, _optimizer,
                        _save_checkpoint, _update, cache_binding, fit_normalizer,
                        normalize_scene, score_reference)
from .sampling import BalancedScenePairSampler


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def module_hashes(model):
    result = {}
    for name in ('input_embedding', 'tcm_encoder', 'tcm_decoder', 'tam_encoder', 'ce_classifier'):
        module = getattr(model, name)
        if module is None:
            continue
        digest = hashlib.sha256()
        for key, tensor in module.state_dict().items():
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


@contextmanager
def _termination_evidence():
    # The server still owns hard deadlines. Normal SIGTERM preserves the most
    # recent progress and a failed terminal record instead of claiming success.
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise InterruptedError('SIGTERM received; partial staged run is ineligible')
    signal.signal(signal.SIGTERM, terminated)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class FrozenCompletionCache:
    """Bounded phase-local LRU, bound to one frozen model and normalized input.

    The caller must discard this object before any completion-weight update.
    Public access also rejects a trainable/training-mode completion module.
    """
    def __init__(self, model, normalized, device, max_bytes, batch_tracks, check=None):
        self.model, self.normalized = model, normalized
        self.device = torch.device(device)
        self.max_bytes, self.batch_tracks = int(max_bytes), int(batch_tracks)
        if self.max_bytes < 0 or self.batch_tracks < 1:
            raise ValueError('cache capacity must be nonnegative and batch_tracks positive')
        self.check = check
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0
        self.peak_bytes = 0
        self.weight_versions = tuple((id(p), p._version) for p in
                                     model.completion_parameters(include_embedding=True))

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    bytes=self.bytes, peak_bytes=self.peak_bytes, max_bytes=self.max_bytes)

    def get(self, keys):
        if tuple((id(p), p._version) for p in self.model.completion_parameters(include_embedding=True)) != self.weight_versions:
            raise ValueError('completion weights changed; discard the stale stage cache')
        if len(set(keys)) != len(keys):
            raise ValueError('cache keys must be unique within a batch')
        if any(p.requires_grad for p in self.model.completion_parameters(include_embedding=True)):
            raise ValueError('completion cache requires frozen embedding and TCM')
        if self.model.tcm_encoder.training or self.model.tcm_decoder.training:
            raise ValueError('completion cache requires TCM eval mode')
        result, missing = {}, []
        for key in keys:
            if key in self.entries:
                result[key] = self.entries[key]
                self.entries.move_to_end(key)
                self.hits += 1
            else:
                missing.append(key)
                self.misses += 1
        for start in range(0, len(missing), self.batch_tracks):
            if self.check:
                self.check()
            chunk = missing[start:start + self.batch_tracks]
            x = torch.as_tensor(np.stack([self.normalized[s][0][t] for s, t in chunk]), device=self.device)
            mask = torch.as_tensor(np.stack([self.normalized[s][1][t] for s, t in chunk]), device=self.device)
            # no_grad rather than inference_mode: TAM's backward saves inputs.
            with torch.no_grad():
                completed = self.model.complete(x, mask)
            if not torch.isfinite(completed).all():
                raise FloatingPointError('nonfinite frozen completion')
            for key, tensor in zip(chunk, completed):
                value = tensor.detach().cpu().clone()
                result[key] = value
                size = value.numel() * value.element_size()
                if size <= self.max_bytes:
                    while self.entries and self.bytes + size > self.max_bytes:
                        _, victim = self.entries.popitem(last=False)
                        self.bytes -= victim.numel() * victim.element_size()
                        self.evictions += 1
                    self.entries[key] = value
                    self.bytes += size
                    self.peak_bytes = max(self.peak_bytes, self.bytes)
        return torch.stack([result[key] for key in keys]).to(self.device)


def _pair_batch(scenes, normalized, groups, device):
    keys, pairs, labels, same = [], [], [], []
    for scene_index, indices in groups:
        scene = scenes[scene_index]
        selected = np.asarray(scene['pairs'])[indices]
        nodes, inverse = np.unique(selected, return_inverse=True)
        pairs.append(inverse.reshape(-1, 2) + len(keys))
        keys.extend((scene_index, int(node)) for node in nodes)
        labels.append(np.asarray(scene['labels'])[indices])
        sources = np.asarray(scene['sources'])
        same.append(sources[selected[:, 0]] == sources[selected[:, 1]])
    labels, same = np.concatenate(labels), np.concatenate(same)
    x = torch.as_tensor(np.stack([normalized[s][0][t] for s, t in keys]), device=device)
    observed = torch.as_tensor(np.stack([normalized[s][1][t] for s, t in keys]), device=device)
    metadata = dict(pair_count=len(labels), positive_pairs=int((labels == 1).sum()),
                    negative_pairs=int((labels == 0).sum()), unique_endpoints=len(keys),
                    same_source_positive_pairs=int(((labels == 1) & same).sum()),
                    cross_source_positive_pairs=int(((labels == 1) & ~same).sum()),
                    same_source_negative_pairs=int(((labels == 0) & same).sum()),
                    mean_observed_slots=float(observed.sum(1).float().mean().cpu()),
                    scene_ids=[scenes[s]['scene_id'] for s, _ in groups],
                    pair_indices=[indices.tolist() for _, indices in groups])
    return keys, x, observed, np.concatenate(pairs), torch.as_tensor(labels, device=device), metadata


def train_staged_reference(archive_path, manifest, config, role, branch, seed, device, output):
    """Run the opt-in v2 fit or a strictly training-only, ineligible probe."""
    model_config, ref = _check_configuration(config, role, branch)
    if ref.get('schedule') != 'staged_updates_v2':
        raise ValueError('staged trainer requires staged_updates_v2')
    probe = ref['purpose'] == 'probe'
    if probe and (role != 'A0' or branch != 'ce'):
        raise ValueError('staged probe requires A0 / ce')
    validate_manifest_config(manifest, config)
    if seed not in config['seeds']:
        raise ValueError('seed must be declared in config')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('staged training requires an available explicit cpu/cuda device')
    binding = _binding(manifest, config, role, branch, seed)
    run = runtime.Run(output, {**binding, 'kind': 'staged_training_probe' if probe else 'association_reference',
                              'eligible_reference': False, 'config': config,
                              'provenance': runtime.provenance(device)}, ref['max_hours'])
    history, stage_reports = [], []
    report = {**binding, 'status': 'RUNNING', 'purpose': ref['purpose'],
              'eligible_reference': False, 'phase': 'preparation', 'history': history,
              'stages': stage_reports, 'test_measurements_read': False,
              'validation_measurements_read': False, 'checkpoint_written': False,
              'completed_updates': 0, 'phase_seconds': {},
              'scope': 'sequential-stage feasibility only; no convergence or causal speedup claim' if probe
                       else 'independent association reference; convergence not automatically established'}
    def persist():
        report['elapsed_seconds'] = time.monotonic() - run.started
        if device.type == 'cuda':
            try:
                report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
            except Exception as error:
                report['memory_measurement_error'] = str(error)
        runtime.write_json(run.path / 'progress.json', report)
        runtime.write_json(run.path / 'history.json', history)
    with _termination_evidence():
        try:
            persist()
            runtime.seed_everything(seed)
            track_rng = np.random.default_rng(seed)
            start = time.monotonic()
            scenes = load_split(archive_path, manifest, 'T_core', model_config.length)
            if role == 'A_full':
                scenes += load_split(archive_path, manifest, 'T_head', model_config.length)
            run.check()
            normalizer = fit_normalizer(scenes)
            normalized = [normalize_scene(scene, normalizer) for scene in scenes]
            sampler = BalancedScenePairSampler(scenes, seed)
            transitions = [(s, t) for s, (_, masks) in enumerate(normalized)
                           for t, mask in enumerate(masks) if int(mask.sum()) >= 2]
            if not transitions:
                raise ValueError('TCM requires observed temporal transitions')
            report.update(training_scene_ids=[scene['scene_id'] for scene in scenes],
                          normalizer=normalizer, sampler=sampler.describe(),
                          tcm_eligible_tracks=len(transitions),
                          tcm_excluded_singleton_tracks=sum(len(x) for x, _ in normalized)-len(transitions))
            validation = None
            if not probe:
                validation = load_split(archive_path, manifest, 'V_select', model_config.length)
                report['validation_measurements_read'] = True
                if not any(len(scene['pairs']) for scene in validation):
                    raise ValueError('reference selection requires validation pairs')
            model = AssociationModel(model_config, branch).to(device)
            report['phase_seconds']['preparation'] = time.monotonic() - start
            checkpoint = run.path / 'checkpoint.pt'
            best = None
            association_updates = 0
            clip = float(ref.get('gradient_clip', 1.0))
            for phase, updates in (('TCM', ref['tcm_updates']), ('TAM_FROZEN', ref['tam_updates']),
                                   ('JOINT', ref['joint_updates'])):
                if not updates:
                    continue
                run.check()
                report['phase'] = phase
                # Joint starts at the best full-V frozen-TAM snapshot, not at
                # a possibly worse last training step. Probe has no selection.
                if phase == 'JOINT' and not probe:
                    payload = torch.load(checkpoint, map_location=device, weights_only=True)
                    model.load_state_dict(payload['state_dict'])
                if phase == 'TCM':
                    model.set_completion_trainable(True, include_embedding=True)
                    model.input_embedding.requires_grad_(False)
                    parameters = list(model.completion_parameters())
                else:
                    model.set_completion_trainable(phase == 'JOINT', include_embedding=True)
                    parameters = [p for p in model.parameters() if p.requires_grad]
                model.train()
                # Clear residual grads too: frozen parameters must not look as
                # though they received gradients in a subsequent stage.
                model.zero_grad(set_to_none=True)
                optimizer = _optimizer(parameters, ref)
                before = module_hashes(model)
                cache = FrozenCompletionCache(model, normalized, device,
                            ref['completion_cache_mib'] * 1024 * 1024, ref['batch_tracks'], run.check) \
                        if phase == 'TAM_FROZEN' else None
                stage_start = time.monotonic()
                stage_report = dict(phase=phase, requested_updates=updates, completed_updates=0,
                                    before_hashes=before, learning_rate=ref['learning_rate'])
                if phase == 'JOINT' and not probe:
                    stage_report['starts_from_selection'] = dict(payload['selection'])
                stage_reports.append(stage_report)
                persist()
                for update in range(1, updates + 1):
                    run.check()
                    model.train()
                    _sync(device)
                    started = time.perf_counter()
                    row = dict(phase=phase, update=update, learning_rate=ref['learning_rate'])
                    if phase == 'TCM':
                        indices = track_rng.choice(len(transitions), ref['batch_tracks'],
                                                   replace=len(transitions) < ref['batch_tracks'])
                        keys = [transitions[int(i)] for i in indices]
                        x = torch.as_tensor(np.stack([normalized[s][0][t] for s,t in keys]), device=device)
                        mask = torch.as_tensor(np.stack([normalized[s][1][t] for s,t in keys]), device=device)
                        loss = model.completion_loss(x, mask)
                        row.update(track_count=len(keys), track_keys=[list(key) for key in keys])
                    else:
                        groups = sampler.sample(ref['pair_batch_size'])
                        keys, x, mask, pairs, labels, metadata = _pair_batch(scenes, normalized, groups, device)
                        row.update(metadata)
                        row['sampling'] = sampler.batch_metadata(groups)
                        if cache is not None:
                            completed = cache.get(keys)
                            embeddings = torch.cat([model.tam_encoder(completed[k:k+ref['batch_tracks']]).flatten(1)
                                         for k in range(0, len(keys), ref['batch_tracks'])])
                            row['completion_cache'] = cache.stats()
                        else:
                            embeddings = torch.cat([model.encode_tracks(x[k:k+ref['batch_tracks']],
                                                                  mask[k:k+ref['batch_tracks']])
                                         for k in range(0, len(keys), ref['batch_tracks'])])
                        scores = model.score_pairs(embeddings, pairs)
                        association_loss = model.association_loss(scores, labels)
                        loss = association_loss
                        row['association_loss'] = float(association_loss.detach())
                        if phase == 'JOINT' and ref['completion_weight']:
                            # One concatenated node population retains the
                            # original effective-target-element weighting.
                            auxiliary = model.completion_loss(x, mask)
                            loss = loss + ref['completion_weight'] * auxiliary
                            row['completion_loss'] = float(auxiliary.detach())
                        association_updates += 1
                    value, norm = _update(loss, optimizer, parameters, clip)
                    _sync(device)
                    row.update(loss=value, gradient_norm=norm, seconds=time.perf_counter()-started)
                    if device.type == 'cuda':
                        try:
                            row['process_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
                            row['process_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
                        except Exception as error:
                            row['memory_measurement_error'] = str(error)
                    history.append(row)
                    stage_report['completed_updates'] = update
                    report['completed_updates'] += 1
                    # Save progress before validation and before time checks.
                    persist()
                    run.check()
                    if not probe and phase != 'TCM' and (update % ref['validate_every_updates'] == 0 or update == updates):
                        validation_start = time.monotonic()
                        report['phase'] = phase + '_VALIDATION'
                        persist()
                        records = score_reference(model, normalizer, validation, device, ref['batch_tracks'], check=run.check)
                        selection = select_threshold(records)
                        row['selection'] = selection
                        row['validation_seconds'] = time.monotonic() - validation_start
                        if best is None or selection['macro_f1'] > best['validation_macro_f1'] + 1e-12:
                            best = {'stage': phase, 'update': update, 'association_update': association_updates,
                                    'threshold': selection['threshold'], 'validation_macro_f1': selection['macro_f1'],
                                    'validation': evaluate_scenes(records, selection['threshold'])}
                            _save_checkpoint(checkpoint, model, model_config, normalizer, binding, best)
                            report['checkpoint_written'] = True
                        report['phase'] = phase
                        persist()
                        run.check()
                stage_report.update(after_hashes=module_hashes(model), elapsed_seconds=time.monotonic()-stage_start)
                stage_report['changed_modules'] = [name for name in before if before[name] != stage_report['after_hashes'][name]]
                frozen_modules = ('input_embedding', 'tam_encoder', 'ce_classifier') if phase == 'TCM' else (
                                 ('input_embedding', 'tcm_encoder', 'tcm_decoder') if phase == 'TAM_FROZEN' else ())
                if any(name in stage_report['changed_modules'] for name in frozen_modules):
                    raise RuntimeError('a frozen stage unexpectedly changed protected model parameters')
                report['phase_seconds'][phase] = stage_report['elapsed_seconds']
                if cache is not None:
                    stage_report['completion_cache'] = cache.stats()
                    cache.clear()
                    cache = None
                persist()
            summary = {**binding, 'purpose': ref['purpose'], 'eligible_reference': not probe,
                       'model_config': asdict(model_config), 'normalizer': normalizer,
                       'training_scene_ids': report['training_scene_ids'],
                       'training_splits': ['T_core'] + (['T_head'] if role == 'A_full' else []),
                       'test_opened': False, 'sampler': sampler.describe(),
                       'history_file': 'history.json', 'stages': stage_reports,
                       'phase_seconds': report['phase_seconds'],
                       'lr_schedule': 'fixed learning rate with fresh Adam per stage',
                       'reference_status': report['scope']}
            if not probe:
                payload = torch.load(checkpoint, map_location=device, weights_only=True)
                model.load_state_dict(payload['state_dict'])
                model.eval()
                summary.update(checkpoint_sha256=runtime.digest(checkpoint), threshold=best['threshold'],
                               validation=best['validation'], selected_stage=best['stage'],
                               selected_update=best['update'], selected_association_update=best['association_update'],
                               selection_split='V_select', joint_completion=bool(ref['joint_updates']),
                               completion_weight=ref['completion_weight'])
                report['phase'] = 'final_caches'
                persist()
                cache_start = time.monotonic()
                head = load_split(archive_path, manifest, 'T_head', model_config.length)
                for split, data in (('T_head', head), ('V_select', validation)):
                    run.check()
                    records = score_reference(model, normalizer, data, device, ref['batch_tracks'], check=run.check)
                    runtime.save_records(run.path / split, records, cache_binding(summary, split))
                summary['cache_bindings'] = {split: cache_binding(summary, split) for split in ('T_head','V_select')}
                summary['coverage'] = {'training': _coverage(scenes), 'T_head': _coverage(head), 'V_select': _coverage(validation)}
                report['phase_seconds']['final_caches'] = time.monotonic()-cache_start
            runtime.write_json(run.path / 'summary.json', summary)
            run.check()
            report.update(status='COMPLETED', eligible_reference=not probe, phase='complete')
            persist()
            run.finish(eligible_reference=not probe, summary_sha256=runtime.digest(run.path / 'summary.json'),
                       **({'checkpoint_sha256': summary['checkpoint_sha256']} if not probe else {}))
            return summary
        except BaseException as error:
            report.update(status='FAILED', eligible_reference=False, failure_type=type(error).__name__, failure=str(error))
            persistence_errors = []
            try:
                persist()
            except Exception as persistence_error:
                persistence_errors.append('progress: ' + str(persistence_error))
            try:
                run.finish('FAILED', eligible_reference=False, failure_type=type(error).__name__, failure=str(error),
                           evidence_write_errors=persistence_errors)
            except Exception as persistence_error:
                persistence_errors.append('run: ' + str(persistence_error))
            if persistence_errors and hasattr(error, 'add_note'):
                error.add_note('; '.join(persistence_errors))
            raise
