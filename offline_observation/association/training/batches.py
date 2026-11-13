"""Pair batches and a stage-local cache for frozen completion features."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
import torch

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
        self.completion_policy = model.config.completion_policy
        self.weight_versions = tuple((id(p), p._version) for p in
                                     model.completion_parameters(include_embedding=True))

    def clear(self):
        self.entries.clear()
        self.bytes = 0

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    bytes=self.bytes, peak_bytes=self.peak_bytes, max_bytes=self.max_bytes)

    def get(self, keys):
        if self.model.config.completion_policy != self.completion_policy:
            raise ValueError('completion policy changed; discard the stale stage cache')
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
