"""Complete development inference on registered or native-time physical scenes.

One TCM pass feeds both association and fusion. References enter only the
evaluation call after predictions; this module does not register native tracks.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .full_scene_evaluation import evaluate_full_scene, simple_fusion_outputs, single_source_outputs
from .fusion_inference import group_predicted_tracks, group_tracks_by_identity
from .metrics import evaluate_scenes, select_threshold
from .reference import normalize_scene, stable_evaluation_parameters


def aggregate_fusion_reports(reports):
    """Pool error numerators and coverage counts, never average scene RMSEs."""
    result = {}
    for region in ('all', 'observed', 'missing'):
        result[region] = {}
        for feature in ('position', 'speed', 'heading'):
            rows = [report['full_target_metrics'][region][feature] for report in reports]
            count_keys = ('available_target_count', 'available_prediction_count',
                          'intersection_count', 'missing_prediction_count',
                          'prediction_without_target_count', 'available_target_trajectory_count',
                          'covered_target_trajectory_count', 'missing_target_trajectory_count')
            pooled = {key: sum(row[key] for row in rows) for key in count_keys}
            n = pooled['intersection_count']
            for key in ('mae', 'mse'):
                pooled[key] = sum(row[key] * row['intersection_count'] for row in rows
                                  if row['intersection_count']) / n if n else None
            pooled['rmse'] = math.sqrt(pooled['mse']) if n else None
            for ratio, numerator, denominator in (
                    ('coverage', 'intersection_count', 'available_target_count'),
                    ('trajectory_coverage', 'covered_target_trajectory_count', 'available_target_trajectory_count')):
                pooled[ratio] = pooled[numerator] / pooled[denominator] if pooled[denominator] else None
            pooled['unit'] = {'position': 'metres', 'speed': 'native_speed', 'heading': 'degrees'}[feature]
            result[region][feature] = pooled
    return result


def _registered(scene, length):
    if scene.get('split') != 'V_select':
        raise ValueError('complete development evaluation requires V_select; T_test is sealed')
    timed = scene.get('input_adapter') == 'native_time_v1'
    if timed:
        from .timed_inputs import validate_timed_scene
        validate_timed_scene(scene)
    values = np.asarray(scene['values'])
    observed = np.asarray(scene['observed'])
    grid = np.asarray(scene['grid'], dtype=np.float64)
    actual = np.asarray(scene['observation_times'], dtype=np.float64)
    if (values.ndim != 3 or values.shape[1:] != (length, 4)
            or observed.shape != values.shape[:2] or observed.dtype != bool
            or grid.shape != (length,) or actual.shape != observed.shape
            or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any()):
        raise ValueError('invalid registered scene shape, mask or physical grid')
    if len(values) and ((not timed and (~observed.any(axis=1)).any()) or not np.isfinite(values[observed]).all()):
        raise ValueError('every valid fragment must supply finite observations')
    if not np.allclose(actual[observed], np.broadcast_to(grid, observed.shape)[observed], rtol=0, atol=1e-8):
        raise ValueError('observation time differs from physical grid; legacy scene64 registration is not accepted')
    pairs = np.asarray(scene['pairs'])
    expected_pairs = np.column_stack(np.triu_indices(len(values), 1))
    if pairs.dtype.kind not in 'iu' or not np.array_equal(pairs, expected_pairs):
        raise ValueError('full-scene evaluation requires all canonical unordered candidate pairs')
    identities, labels = np.asarray(scene['identities']), np.asarray(scene['labels'])
    if (identities.shape != (len(values),) or labels.shape != (len(pairs),)
            or not np.array_equal(labels, identities[pairs[:, 0]] == identities[pairs[:, 1]])):
        raise ValueError('full-scene labels must match the true identities for every candidate')
    targets = scene.get('fusion_targets')
    if not isinstance(targets, dict):
        raise ValueError('fusion_targets must explicitly declare aligned references, possibly an empty dict')
    for target in targets.values():
        if (not isinstance(target, dict) or 'query_times' not in target
                or not np.array_equal(np.asarray(target['query_times'], dtype=np.float64), grid)):
            raise ValueError('every fusion target query_times must exactly match the physical scene grid')
    return values, observed


def _inference_device(device, model_device, fusion_device):
    """Resolve generic CUDA to the model's actual device without a GPU call."""
    requested, model_device, fusion_device = map(torch.device, (device, model_device, fusion_device))
    if requested.type == 'cuda' and requested.index is None and model_device.type == 'cuda':
        requested = model_device
    if model_device != requested or fusion_device != requested:
        raise ValueError('models must be on the requested device')
    return requested


@stable_evaluation_parameters
@torch.inference_mode()
def evaluate_complete(model, fusion, normalizer, scenes, *, device,
                      batch_tracks, pair_batch_size, select_on_validation,
                      threshold, singleton_policy, check=None):
    """Return full-V association plus predicted/oracle/simple/single-source fusion.

    ``singleton_policy`` is explicitly 'missing' or 'observed'. The latter
    passes through the singleton's aligned physical observations only; it does
    not decode a completed latent as a physical trajectory. References never
    pick an output, repair a predicted group or feed recursive fusion. The
    oracle, simple-fusion and single-source rows use true-identity grouping and
    are diagnostics, not independent association algorithms.

    Model modes are set to eval and left there. Training must explicitly restore
    its current stage. Cached latents live on CPU only for this call. Threshold
    fitting uses all supplied V scenes together, never per scene or on T_test.
    """
    for value in (batch_tracks, pair_batch_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('inference batch sizes must be positive integers')
    if singleton_policy not in ('missing', 'observed'):
        raise ValueError('singleton_policy must explicitly be missing or observed')
    if not isinstance(select_on_validation, bool):
        raise ValueError('select_on_validation must be boolean')
    if select_on_validation and threshold is not None:
        raise ValueError('provide threshold=None when fitting on V_select')
    if (model.config.length, model.config.d_model) != (fusion.config.length, fusion.config.d_model):
        raise ValueError('association and fusion dimensions must agree')
    scenes = list(scenes)
    if not scenes or len({str(s['scene_id']) for s in scenes}) != len(scenes):
        raise ValueError('a nonempty list of distinct validation scenes is required')
    adapter = getattr(model, 'input_adapter', 'registered_v1')
    if any(scene.get('input_adapter', 'registered_v1') != adapter for scene in scenes):
        raise ValueError('all validation scenes must match the model input adapter')
    for scene in scenes:
        _registered(scene, model.config.length)
    device = _inference_device(device, next(model.parameters()).device, next(fusion.parameters()).device)
    dtype = next(model.parameters()).dtype
    if next(fusion.parameters()).dtype != dtype:
        raise ValueError('association and fusion dtype must agree')
    started = time.monotonic()
    model.eval()
    fusion.eval()
    records, cached = [], []
    for scene in scenes:
        if check:
            check()
        x, mask = normalize_scene(scene, normalizer)
        completed, encoded = [], []
        for offset in range(0, len(x), batch_tracks):
            if check:
                check()
            latent = model.complete(torch.as_tensor(x[offset:offset+batch_tracks], device=device, dtype=dtype),
                                    torch.as_tensor(mask[offset:offset+batch_tracks], device=device))
            embedding = model.tam_encoder(latent).flatten(1)
            if not torch.isfinite(latent).all() or not torch.isfinite(embedding).all():
                raise FloatingPointError('nonfinite completed or association features')
            completed.append(latent.cpu())
            encoded.append(embedding.cpu())
        latents = torch.cat(completed) if completed else torch.empty(0, model.config.length, model.config.d_model, dtype=dtype)
        embeddings = torch.cat(encoded) if encoded else torch.empty(0, model.embedding_dim, dtype=dtype)
        pairs = np.asarray(scene['pairs'])
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu':
            raise ValueError('pairs must be integer [E,2]')
        if len(pairs) and ((pairs < 0).any() or (pairs >= len(x)).any() or (pairs[:, 0] >= pairs[:, 1]).any()):
            raise ValueError('invalid candidate pair')
        scores = []
        for offset in range(0, len(pairs), pair_batch_size):
            if check:
                check()
            nodes, inverse = np.unique(pairs[offset:offset+pair_batch_size], return_inverse=True)
            value = model.score_pairs(embeddings[nodes].to(device), inverse.reshape(-1, 2))
            scores.append(value.cpu().numpy())
        record = {key: scene[key] for key in ('scene_id', 'component_id', 'node_ids', 'identities', 'sources', 'intervals', 'labels')}
        record.update(pairs=pairs, scores=np.concatenate(scores) if scores else np.empty(0))
        records.append(record)
        cached.append(latents)
    selection = select_threshold(records) if select_on_validation else None
    if selection is not None:
        threshold = selection['threshold']
    association = evaluate_scenes(records, threshold)
    sources = sorted({str(source) for scene in scenes for source in scene['sources']})
    methods = {key: [] for key in ['predicted', 'oracle', 'simple_fusion_oracle']
               + ['single_source:' + source for source in sources]}
    per_scene = []
    fusion_steps = 0
    for scene, record, latent in zip(scenes, records, cached):
        targets = scene['fusion_targets']
        target_ids = list(targets)
        # Empty scenes still belong to the declared development population.
        # Nonempty nodes without target rows are rejected by evaluate_full_scene.
        truth = (np.stack([targets[identity]['states'] for identity in target_ids])
                 if target_ids else np.empty((0, model.config.length, 4)))
        valid = (np.stack([targets[identity]['valid'] for identity in target_ids])
                 if target_ids else np.empty((0, model.config.length, 4), dtype=bool))
        common = dict(node_identities=scene['identities'], target_identities=target_ids,
                      target_states=truth, target_valid=valid, node_observed=scene['observed'],
                      split=scene['split'], correspondence_policy='anchor_identity_first')
        outputs = {}
        groupings = {
            'predicted': group_predicted_tracks(len(latent), record['pairs'], record['scores'],
                threshold=threshold, grouping_policy='anchor_direct', node_order='input'),
            'oracle': group_tracks_by_identity(scene['identities'], node_order='input'),
        }
        for kind, groups in groupings.items():
            generated = []
            for members in groups:
                if check:
                    check()
                row = dict(node_indices=members, states=None)
                if len(members) == 1 and singleton_policy == 'observed':
                    row.update(states=np.asarray(scene['values'])[members[0]],
                               prediction_valid=np.repeat(np.asarray(scene['observed'])[members[0], :, None], 4, axis=1))
                elif len(members) > 1:
                    memory = latent[members[0]:members[0]+1].to(device)
                    for node in members[1:]:
                        if check:
                            check()
                        fused = fusion(memory, latent[node:node+1].to(device))
                        memory = fused['fused_features']
                        fusion_steps += 1
                    row['states'] = fused['states'][0].cpu().numpy()
                generated.append(row)
            outputs[kind] = evaluate_full_scene(generated, grouping_kind=kind, **common)
        node_valid = np.repeat(np.asarray(scene['observed'])[..., None], 4, axis=-1)
        simple = simple_fusion_outputs(scene['values'], node_valid, scene['identities'], scene['sources'])
        outputs['simple_fusion_oracle'] = evaluate_full_scene(simple, grouping_kind='oracle', **common)
        for source in sources:
            single = single_source_outputs(scene['values'], node_valid, scene['identities'], scene['sources'], source=source)
            outputs['single_source:' + source] = evaluate_full_scene(single, grouping_kind='oracle', **common)
        for key, report in outputs.items():
            methods[key].append(report)
        per_scene.append(dict(scene_id=str(scene['scene_id']), methods=outputs))
    if check:
        check()
    return dict(association=association, selection=selection, threshold=threshold, records=records,
                fusion={key: aggregate_fusion_reports(reports) for key, reports in methods.items()},
                per_scene=per_scene, singleton_policy=singleton_policy, feedback_policy='fc_memory',
                grouping_policy='anchor_direct', correspondence_policy='anchor_identity_first',
                elapsed_seconds=time.monotonic()-started, fusion_forward_count=fusion_steps,
                test_opened=False, scope='V_select development; oracle, simple-fusion and single-source are diagnostics')
