"""One-pass CDF1 contextual statistics and capacity-matched residual controls.

Statistics use only a frozen symmetric score graph. Missing edges are excluded,
never interpreted as negative identity evidence. Computation is O(N^3) in the
worst case with O(N^2) storage; no N x N x N tensor is constructed.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _graph(z, mask):
    z = np.asarray(z, dtype=np.float64)
    mask = np.asarray(mask)
    if z.ndim != 2 or z.shape[0] != z.shape[1] or mask.shape != z.shape:
        raise ValueError("z and mask must be equally sized square matrices")
    if mask.dtype != np.bool_:
        raise ValueError("mask must be boolean")
    if np.any(np.diag(mask)) or not np.array_equal(mask, mask.T):
        raise ValueError("mask must be symmetric and exclude self edges")
    if not np.isfinite(z[mask]).all():
        raise ValueError("visible logits must be finite")
    if not np.allclose(z[mask], z.T[mask], rtol=1e-7, atol=1e-8):
        raise ValueError("visible logits must be symmetric")
    pairs = np.argwhere(np.triu(mask, k=1)).astype(np.int64)
    q = np.zeros_like(z)
    # Stable sigmoid, including very large finite logits.
    exp_negative = np.exp(-np.abs(z[mask]))
    q[mask] = np.where(z[mask] >= 0, 1 / (1 + exp_negative),
                       exp_negative / (1 + exp_negative))
    return z, mask, pairs, q


def context_features(z, mask, shuffle=False, seed=0):
    """Return unscaled NumPy features in lexicographic upper-edge order.

    For the shuffle intervention only the j-side common-neighbor ordering is
    permuted. Endpoint means/variances and K are calculated before shuffling and
    are exactly shared with the unshuffled features. A nonidentity permutation
    is used whenever K > 1; equal scores can nevertheless make it ineffectual.
    The *unshuffled* operator is node-permutation equivariant. Shuffle seeds are
    tied to matrix indices; callers should derive ``seed`` from a stable scene
    identifier and split/epoch seed, not from Python's randomized hash().
    """
    z, mask, pairs, q = _graph(z, mask)
    features = np.zeros((len(pairs), 8), dtype=np.float32)
    counts = np.zeros(len(pairs), dtype=np.int64)
    support = np.zeros(len(pairs), dtype=np.float64)
    contradiction = np.zeros(len(pairs), dtype=np.float64)
    shuffle_effective = np.zeros(len(pairs), dtype=bool)
    for edge, (i, j) in enumerate(pairs):
        neighbors = np.flatnonzero(mask[i] & mask[j])
        k = len(neighbors)
        counts[edge] = k
        features[edge, 0] = z[i, j]
        if not k:
            continue
        left, right = q[i, neighbors], q[j, neighbors]
        mu_i, mu_j = left.mean(), right.mean()
        var_i, var_j = left.var(), right.var()
        shuffled = right
        if shuffle and k > 1:
            digest = hashlib.sha256(f"{int(seed)}:{i}:{j}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            order = rng.permutation(k)
            if np.array_equal(order, np.arange(k)):
                order = np.roll(order, 1)
            shuffled = right[order]
            shuffle_effective[edge] = not np.array_equal(shuffled, right)
        s = float(np.mean(left * shuffled))
        c = float(mu_i + mu_j - 2 * s)
        support[edge], contradiction[edge] = s, c
        features[edge] = [z[i, j], s, c, mu_i + mu_j, abs(mu_i - mu_j),
                          math.log1p(k), var_i + var_j, abs(var_i - var_j)]
    if not np.isfinite(features).all():
        raise ValueError("features cannot be represented as finite float32")
    return {"features": features, "pairs": pairs, "counts": counts,
            "support": support, "contradiction": contradiction,
            "shuffleable": counts > 1, "shuffle_effective": shuffle_effective}


def scene_pool_features(z, mask):
    """Own calibrated logit plus q mean/variance and fixed [0,1] 10-bin mass."""
    z, _, pairs, q = _graph(z, mask)
    result = np.zeros((len(pairs), 13), dtype=np.float32)
    if len(pairs):
        values = q[pairs[:, 0], pairs[:, 1]]
        counts, _ = np.histogram(values, bins=np.linspace(0, 1, 11))
        result[:, 0] = z[pairs[:, 0], pairs[:, 1]]
        result[:, 1:] = np.r_[values.mean(), values.var(), counts / len(values)]
    if not np.isfinite(result).all():
        raise ValueError("scene features must be finite float32")
    return result


HEAD_DIMENSIONS = {"B": (8, 32, 16), "B_shuffle": (8, 32, 16),
                   "PairMLP": (1, 44, 16),
                   "EndpointMarginal": (6, 35, 16),
                   "ScenePool": (13, 27, 16)}


def head_features(kind, context, scene_features=None):
    """Select *raw* features; fit a separate scaler per head on T_head only."""
    features = np.asarray(context["features"], dtype=np.float32)
    if kind in ("B", "B_shuffle"):
        return features
    if kind == "PairMLP":
        return features[:, :1]
    if kind == "EndpointMarginal":
        return features[:, [0, 3, 4, 5, 6, 7]]
    if kind == "ScenePool":
        if scene_features is None:
            raise ValueError("ScenePool requires scene_pool_features")
        result = np.asarray(scene_features, dtype=np.float32)
        if result.shape != (len(features), 13):
            raise ValueError("ScenePool feature shape must be [E,13]")
        return result
    raise ValueError(f"Unknown head kind: {kind}")


class MonotoneCalibrator(nn.Module):
    """Platt logit z = a*s+b, a=softplus(raw_a)+1e-6 > 0, initially identity."""

    def __init__(self):
        super().__init__()
        self.raw_a = nn.Parameter(torch.tensor(math.log(math.expm1(1 - 1e-6))))
        self.bias = nn.Parameter(torch.tensor(0.0))

    @property
    def slope(self):
        return F.softplus(self.raw_a) + 1e-6

    def forward(self, raw):
        if not torch.isfinite(raw).all():
            raise ValueError("calibrator input must be finite")
        result = self.slope * raw + self.bias
        if not torch.isfinite(result).all():
            raise ValueError("calibrated logits must be finite")
        return result


class FeatureScaler(nn.Module):
    """Frozen population mean/std; fitting outside T_head is refused explicitly."""

    def __init__(self, n_features):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("scale", torch.ones(n_features))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, features, split="T_head"):
        if split != "T_head":
            raise ValueError("FeatureScaler may only fit T_head")
        if self.fitted.item():
            raise ValueError("FeatureScaler is already frozen")
        x = torch.as_tensor(features, device=self.mean.device, dtype=self.mean.dtype)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not len(x):
            raise ValueError("fit needs a nonempty [E,D] feature matrix")
        if not torch.isfinite(x).all():
            raise ValueError("fit features must be finite")
        with torch.no_grad():
            # Float64 accumulation reduces sensitivity to large/constant logits.
            avg = x.double().mean(dim=0)
            std = x.double().std(dim=0, unbiased=False)
            self.mean.copy_(avg)
            self.scale.copy_(torch.where(std >= torch.finfo(self.scale.dtype).tiny,
                                         std, torch.ones_like(std)))
            self.fitted.fill_(True)
        return self

    def forward(self, features):
        if not self.fitted.item():
            raise ValueError("FeatureScaler must be fitted before use")
        if features.ndim != 2 or features.shape[1] != len(self.mean):
            raise ValueError("feature dimension does not match scaler")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        result = (features - self.mean) / self.scale
        if not torch.isfinite(result).all():
            raise ValueError("scaled features must be finite")
        return result


class ResidualHead(nn.Module):
    """Bounded residual, NOT final logits. All arms return zero where K=0.

    Active parameter counts: B/B_shuffle 833, PairMLP 825,
    EndpointMarginal 838, ScenePool 843. No unused padding parameters.
    """

    def __init__(self, kind="B", delta=1.0):
        super().__init__()
        if kind not in HEAD_DIMENSIONS:
            raise ValueError(f"Unknown head kind: {kind}")
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and positive")
        self.kind = kind
        self.delta = float(delta)
        n_input, hidden, second = HEAD_DIMENSIONS[kind]
        self.network = nn.Sequential(nn.Linear(n_input, hidden), nn.ReLU(),
                                     nn.Linear(hidden, second), nn.ReLU(),
                                     nn.Linear(second, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, features, counts):
        if features.ndim != 2 or features.shape[1] != HEAD_DIMENSIONS[self.kind][0]:
            raise ValueError("head feature dimension mismatch")
        counts = torch.as_tensor(counts, device=features.device)
        if counts.shape != (len(features),) or not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("counts must be a nonnegative finite [E] vector")
        if not torch.isfinite(features).all():
            raise ValueError("head inputs must be finite")
        residual = self.delta * torch.tanh(self.network(features).squeeze(-1))
        if not torch.isfinite(residual).all():
            raise ValueError("head residual must be finite")
        return torch.where(counts > 0, residual, torch.zeros_like(residual))


def fixed_rule(context, delta=1.0):
    """Raw, unscaled S-C control, with the same residual bound and K=0 rule."""
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    correction = delta * np.tanh(context["support"] - context["contradiction"])
    return np.where(context["counts"] > 0, correction, 0.0)
