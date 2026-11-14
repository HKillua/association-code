"""Prediction-only inference with the frozen TMTAFF-style modules."""
from __future__ import annotations

from time import monotonic

import numpy as np
import torch

from association.evaluation.grouping import group_predicted_tracks
from association.training.common import normalize_scene


INPUT_ARRAYS = ('node_ids', 'sources', 'intervals', 'grid', 'values', 'observed',
                'observation_times', 'native_values', 'native_times', 'native_observed')


def _prepare(scene, normalizer):
    """Adapt the original measurement arrays without inspecting identity or reference."""
    prepared = {key: scene[key] for key in INPUT_ARRAYS}
    config = normalizer['adapter_config']
    if scene.get('split') != 'V_select':
        raise ValueError('TMTAFF blind replay requires the original V_select role')
    prepared.update(input_adapter='native_time_v1', split=scene['split'], adapter_config=config,
                    scene_id=scene['scene_id'],
                    time_origin_seconds=float(prepared['grid'][0]))
    prepared.update({key: config[key] for key in ('length', 'max_observations',
                                                   'time_unit_seconds', 'observation_interpolation')})
    if 'physical_units' in normalizer:
        prepared['physical_units'] = normalizer['physical_units']
    # The shared packer validates the length of this identifier vector only.
    prepared['identities'] = prepared['node_ids']
    return normalize_scene(prepared, normalizer)


def _encode_and_assign(model, fusion, normalizer, scene, device, threshold,
                       tracks_per_batch, pairs_per_batch):
    started = monotonic()
    values, observed = _prepare(scene, normalizer)
    pairs = np.asarray(scene['pairs'])
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu'
            or ((pairs < 0) | (pairs >= len(values))).any() or (pairs[:, 0] >= pairs[:, 1]).any()):
        raise ValueError('prediction needs archived ordered candidate pairs')
    template = next(model.parameters())
    latents, embeddings = [], []
    started_encoding = monotonic()
    for offset in range(0, len(values), tracks_per_batch):
        batch = slice(offset, offset + tracks_per_batch)
        latent = model.complete(torch.as_tensor(values[batch], dtype=template.dtype, device=device),
                                torch.as_tensor(observed[batch], device=device))
        if not torch.isfinite(latent).all():
            raise FloatingPointError('TCM produced nonfinite features')
        latents.append(latent)
        embedding = model.tam_encoder(latent).flatten(1)
        if not torch.isfinite(embedding).all():
            raise FloatingPointError('TAM produced nonfinite features')
        embeddings.append(embedding)
    encoded = monotonic()
    latents = torch.cat(latents) if latents else template.new_empty(
        (0, fusion.config.length, fusion.config.d_model))
    embeddings = torch.cat(embeddings) if embeddings else template.new_empty((0, model.embedding_dim))
    chunks = [model.score_pairs(embeddings, pairs[i:i + pairs_per_batch]).cpu().numpy()
              for i in range(0, len(pairs), pairs_per_batch)]
    scores = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    scored = monotonic()
    groups = group_predicted_tracks(len(values), pairs, scores, threshold=threshold,
                                    grouping_policy='anchor_direct', node_order='input')
    states = np.full((len(groups), fusion.config.length, 4), np.nan)
    valid = np.zeros_like(states, dtype=bool)
    for i, group in enumerate(groups):
        if len(group) == 1:
            states[i] = scene['values'][group[0]]
            valid[i] = scene['observed'][group[0], :, None]
    result = dict(groups=groups, states=states, valid=valid, scores=scores,
                  steps=sum(len(group)-1 for group in groups),
                  timing_seconds=dict(preparation=started_encoding-started, encoding=encoded-started_encoding,
                                      tam_pairs=scored-encoded, grouping=monotonic()-scored))
    return result, latents


def _fuse_outputs(fusion, predictions, latents, fusion_batch_size):
    """Batch independent group streams at each depth, preserving each stream's order."""
    active = [(i, j, group, values[group[0]]) for i, (p, values) in enumerate(zip(predictions, latents, strict=True))
              for j, group in enumerate(p['groups']) if len(group) > 1]
    depth, calls, decodes = 1, 0, 0
    while active:
        remaining = []
        for start in range(0, len(active), fusion_batch_size):
            batch = active[start:start + fusion_batch_size]
            left = torch.stack([memory for _, _, _, memory in batch])
            right = torch.stack([latents[i][group[depth]] for i, _, group, _ in batch])
            fused = fusion.fuse_features(left, right)
            calls += 1
            finished = [row for row, (_, _, group, _) in enumerate(batch) if len(group) == depth + 1]
            if finished:
                done = fusion.decode_features(fused[finished]).cpu().numpy()
                decodes += 1
            else:
                done = ()
            for row, states in zip(finished, done, strict=True):
                i, j, _, _ = batch[row]
                predictions[i]['states'][j] = states
                predictions[i]['valid'][j] = True
            for row, (i, j, group, _) in enumerate(batch):
                if len(group) > depth + 1:
                    remaining.append((i, j, group, fused[row]))
        active = remaining
        depth += 1
    return calls, decodes


@torch.inference_mode()
def predict_scenes(snapshot, scenes, *, threshold, device, batch_tracks, pair_batch_size,
                   fusion_batch_size=16):
    """Return scene outputs and shared TFM runtime for a measurement-only batch.

    Encodes each scene with the original track batch size. Pair scores and
    grouping come from that scene's own TAM. Only independent TFM streams may
    share a batch. Group members and recursive FC memory order stay fixed.
    The physical decoder runs on terminal FC memory, since only that output is
    used. Scores, logical TFM pair steps, and native-unit values/masks are explicit.
    """
    for value in (batch_tracks, pair_batch_size, fusion_batch_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('inference batch sizes must be positive integers')
    model, fusion = snapshot['model'], snapshot['fusion']
    model.requires_grad_(False).eval()
    fusion.requires_grad_(False).eval()
    normalizer = snapshot['normalizer']
    model.validate_normalizer(normalizer)
    predictions, latents = [], []
    for scene in scenes:
        result, latent = _encode_and_assign(model, fusion, normalizer, scene, device, threshold,
                                           batch_tracks, pair_batch_size)
        predictions.append(result)
        latents.append(latent)
    started = monotonic()
    calls, decodes = _fuse_outputs(fusion, predictions, latents, fusion_batch_size)
    return predictions, dict(fusion_calls=calls, fusion_decodes=decodes, fusion_seconds=monotonic()-started)


def predict_scene(snapshot, scene, *, threshold, device, batch_tracks, pair_batch_size,
                  fusion_batch_size=16):
    """Predict an isolated scene and include its TFM call count and elapsed time."""
    (prediction,), runtime = predict_scenes(snapshot, [scene], threshold=threshold,
                                           device=device, batch_tracks=batch_tracks,
                                           pair_batch_size=pair_batch_size, fusion_batch_size=fusion_batch_size)
    prediction['fusion_calls'] = runtime['fusion_calls']
    prediction['fusion_decodes'] = runtime['fusion_decodes']
    prediction['timing_seconds']['tfm'] = runtime['fusion_seconds']
    return prediction
