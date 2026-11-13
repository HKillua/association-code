"""T-only discrete selection; inference remains the original context residual head."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from association.models.association_context import EdgeResidual, edge_inputs, normalize
from tools.analysis.static_partition_pilot import agglomerate, THRESHOLD

SELECTION_THRESHOLD = 2.0085254541993995


def balanced_bce(logits, labels):
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    positive = labels.bool()
    if positive.any() and (~positive).any():
        return (losses[positive].mean() + losses[~positive].mean()) / 2
    return losses.mean()


def selected_bce(logits, labels, mask):
    selected = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    return balanced_bce(logits[selected], labels[selected]) if selected.any() else logits.sum() * 0


def matched_masks(dec_logits, hard_logits, pairs, labels, node_count,
                  threshold=SELECTION_THRESHOLD):
    """Match class counts, changing only which edges supply auxiliary gradients."""
    scores = dec_logits.detach().cpu().numpy()
    target = labels.detach().cpu().numpy().astype(bool)
    membership = agglomerate(node_count, pairs, scores + THRESHOLD - threshold, 'WS')[0]
    decision = (membership[pairs[:, 0]] == membership[pairs[:, 1]]) != target
    losses = F.binary_cross_entropy_with_logits(hard_logits.detach(), labels, reduction='none').cpu().numpy()
    hard = np.zeros(len(pairs), bool)
    for positive in (False, True):
        indices = np.flatnonzero(target == positive)
        count = int(np.count_nonzero(decision[indices]))
        order = np.argsort(-losses[indices], kind='stable')
        hard[indices[order[:count]]] = True
    return {'DEC': decision, 'HARD': hard}


def train(caches, initial, normalizer, device, check, epochs=4):
    heads = {k: EdgeResidual().to(device) for k in ('HARD', 'DEC')}
    for head in heads.values():
        head.load_state_dict(initial)
    optimizers = {k: torch.optim.Adam(h.parameters(), lr=.001) for k, h in heads.items()}
    rng = np.random.default_rng(20)
    curve, draws, packed = [], [], {}
    counts = np.zeros(len(caches), int)
    presented = 0
    for epoch in range(epochs):
        for i in rng.permutation(len(caches)):
            cache = caches[i]
            indices = np.arange(len(cache['pairs']))
            features = torch.as_tensor(normalize(edge_inputs(cache, indices, 'CTX'), normalizer), device=device)
            labels = torch.as_tensor(cache['labels'], device=device, dtype=torch.float64)
            base = torch.as_tensor(cache['scores'], device=device)
            logits = {k: base + h(features).double() for k, h in heads.items()}
            masks = matched_masks(logits['DEC'], logits['HARD'], cache['pairs'], labels, len(cache['identities']))
            losses = {}
            step = len(draws) + 1
            for key, head in heads.items():
                optimizers[key].zero_grad(set_to_none=True)
                bce = balanced_bce(logits[key], labels)
                auxiliary = selected_bce(logits[key], labels, masks[key])
                loss = bce + auxiliary
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizers[key].step()
                mask = masks[key]
                losses[key] = dict(loss=float(loss.detach().cpu()), bce=float(bce.detach().cpu()),
                                   auxiliary=float(auxiliary.detach().cpu()), unclipped_gradient_norm=float(norm),
                                   selected_positive=int(np.count_nonzero(mask & cache['labels'])),
                                   selected_negative=int(np.count_nonzero(mask & ~cache['labels'])))
                packed[f'{step:04d}_{key}'] = np.packbits(mask)
            if any(losses['DEC'][key] != losses['HARD'][key] for key in ('selected_positive', 'selected_negative')):
                raise ValueError('unmatched auxiliary exposure')
            counts[i] += 1
            presented += len(indices)
            draws.append(cache['scene_id'])
            curve.append(dict(step=step, epoch=epoch+1, scene_id=cache['scene_id'], pairs=len(indices), heads=losses,
                              selected_intersection=int(np.count_nonzero(masks['HARD'] & masks['DEC'])),
                              selected_union=int(np.count_nonzero(masks['HARD'] | masks['DEC']))))
            if step == 1 or step % 16 == 0:
                check(phase='T_fit', optimizer_updates=2*step)
    for head in heads.values():
        head.eval()
        head.requires_grad_(False)
    sampling = dict(scene_draws=draws, epochs=epochs, per_scene_passes=counts.tolist(),
                    unique_pairs=sum(len(c['pairs']) for c in caches),
                    pair_presentations_per_head=presented, updates_per_head=len(draws))
    return heads, curve, sampling, packed
