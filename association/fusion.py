"""Pairwise TFM adaptation on completed embeddings, with native-state loss.

This module does not choose reference alignment, target groups, or recursive
fusion. The undisclosed FC width and interpretation of Eq. 20 are explicit.
See ``docs/specs/tmtaff-fusion.md`` for the reproduction boundary.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models import _causal_mask


@dataclass(frozen=True)
class PairFusionConfig:
    fc_hidden: int
    length: int = 64
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 256
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("fc_hidden", "length", "d_model", "nhead", "num_layers",
                     "dim_feedforward"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")


class PairFusion(nn.Module):
    """Two independent encoders, dense fusion memory, and a latent decoder.

    Input embeddings already contain the TCM's positional information. A
    learned, initially zero BOS starts the decoder; subsequent inputs are its
    own latent outputs. There is no reference/teacher-forcing input. The state
    projection emits ``lat, lon, vel, cou`` in native units, using softplus for
    nonnegative speed. This output constraint is an explicit adaptation.
    """

    def __init__(self, config: PairFusionConfig):
        super().__init__()
        self.config = config

        def encoder():
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model, nhead=config.nhead,
                dim_feedforward=config.dim_feedforward, dropout=config.dropout,
                activation="gelu", batch_first=True,
            )
            return nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)

        self.left_encoder = encoder()
        self.right_encoder = encoder()
        flattened = config.length * config.d_model
        self.fc1 = nn.Linear(2 * flattened, config.fc_hidden)
        self.fc_activation = nn.LeakyReLU()
        self.fc2 = nn.Linear(config.fc_hidden, flattened)
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward, dropout=config.dropout,
            activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, config.num_layers)
        self.bos = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.output_dropout = nn.Dropout(config.dropout)
        self.state_projection = nn.Linear(config.d_model, 4)
        # Transformer containers clone their layers. Initialize each depth and
        # each side independently, following AssociationModel's convention.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

    def _validate_features(self, values: Tensor):
        expected = (self.config.length, self.config.d_model)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(f"completed embeddings must have shape [N, {expected[0]}, {expected[1]}]")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("completed embeddings must be finite floating point")
        if values.device != self.bos.device or values.dtype != self.bos.dtype:
            raise ValueError("completed embeddings and PairFusion must use the same device and dtype")

    def forward(self, left: Tensor, right: Tensor) -> dict[str, Tensor]:
        """Return physical ``states`` and the separate latent ``fused_features``.

        ``fused_features`` is the FC fusion memory, not a physical prediction or
        a claimed implementation of the paper's ambiguous recursive feedback.
        """
        self._validate_features(left)
        self._validate_features(right)
        if left.shape != right.shape:
            raise ValueError("left and right must have identical shapes")
        if len(left) == 0:
            # Preserve a gradient path and well-defined shapes for no pairs.
            return {"states": left.sum(-1, keepdim=True).expand(-1, -1, 4),
                    "fused_features": left + right}
        encoded = torch.cat((self.left_encoder(left), self.right_encoder(right)), dim=1)
        memory = self.fc2(self.fc_activation(self.fc1(encoded.flatten(start_dim=1))))
        memory = memory.reshape(len(left), self.config.length, self.config.d_model)
        history = self.bos.expand(len(left), -1, -1)
        generated = []
        for step in range(self.config.length):
            decoded = self.decoder(history, memory, tgt_mask=_causal_mask(history.shape[1], history.device))
            latent = decoded[:, -1:]
            generated.append(latent)
            if step + 1 < self.config.length:
                history = torch.cat((history, latent), dim=1)
        raw = self.state_projection(self.output_dropout(torch.cat(generated, dim=1)))
        states = torch.cat((raw[..., :2], F.softplus(raw[..., 2:3]), raw[..., 3:]), dim=-1)
        if not torch.isfinite(states).all() or not torch.isfinite(memory).all():
            raise ValueError("fusion produced non-finite output")
        return {"states": states, "fused_features": memory}

    def forward_pairs(self, completed_tracks: Tensor, pairs: Tensor) -> dict[str, Tensor]:
        """Gather pairs from TCM output without detaching its computation graph."""
        self._validate_features(completed_tracks)
        pairs = torch.as_tensor(pairs, device=completed_tracks.device)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape [E, 2]")
        if pairs.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("pairs must contain integer indices")
        if pairs.numel() and ((pairs < 0).any() or (pairs >= len(completed_tracks)).any()):
            raise ValueError("pair index outside completed tracks")
        pairs = pairs.long()
        return self(completed_tracks[pairs[:, 0]], completed_tracks[pairs[:, 1]])


class FusionLoss(nn.Module):
    """Eq. 18--21 in native units with explicit position-formula interpretation.

    ``losses`` and the ``variance`` vector are ordered lat/lon/vel/cou. Masked
    values are indexed out *before* nonlinear arithmetic; NaN is allowed only
    outside valid supervision. Each feature must have some valid supervision.
    """

    feature_names = ("lat", "lon", "vel", "cou")

    def __init__(self, *, position_loss_mode: str):
        super().__init__()
        if position_loss_mode not in ("paper_printed", "standard_smooth_l1"):
            raise ValueError("position_loss_mode must be paper_printed or standard_smooth_l1")
        self.position_loss_mode = position_loss_mode
        self.log_variance = nn.Parameter(torch.zeros(4))

    def _position_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.position_loss_mode == "standard_smooth_l1":
            return F.smooth_l1_loss(prediction, target, beta=0.001)
        error = prediction - target
        absolute = error.abs()
        small = absolute < 0.001
        # Eq. 20 literally has a discontinuity. Index each branch so an unused
        # quadratic branch cannot overflow and poison a linear-branch gradient.
        return (0.5 * error[small].square().sum() + (absolute[~small] - 0.0005).sum()) / error.numel()

    def forward(self, prediction: Tensor, target: Tensor, valid: Tensor) -> dict:
        if prediction.ndim != 3 or prediction.shape[-1] != 4 or target.shape != prediction.shape:
            raise ValueError("prediction and target must have identical [B, L, 4] shapes")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise ValueError("prediction and target must be floating point")
        if valid.shape != prediction.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B, L, 4] mask")
        if prediction.device != target.device or prediction.device != valid.device or prediction.device != self.log_variance.device:
            raise ValueError("loss inputs and FusionLoss must use the same device")
        if prediction.dtype != target.dtype:
            raise ValueError("prediction and target must use the same dtype")
        losses = {}
        for index, name in enumerate(self.feature_names):
            mask = valid[..., index]
            if not mask.any():
                raise ValueError(f"no valid supervision for {name}")
            pred = prediction[..., index][mask]
            truth = target[..., index][mask]
            if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
                raise ValueError(f"valid prediction and target must be finite for {name}")
            if index < 2:
                loss = self._position_loss(pred, truth)
            elif index == 2:
                if (pred < 0).any() or (truth < 0).any():
                    raise ValueError("valid speeds must be nonnegative")
                loss = ((pred + 1e-6).log() - (truth + 1e-6).log()).square().mean()
            else:
                loss = (1 - torch.cos((pred - truth) * (math.pi / 180))).mean()
            losses[name] = loss
        components = torch.stack(tuple(losses.values()))
        variance = self.log_variance.exp()
        total = (0.5 * (-self.log_variance).exp() * components + F.softplus(self.log_variance)).sum()
        if not torch.isfinite(components).all() or not torch.isfinite(variance).all() or not torch.isfinite(total):
            raise ValueError("fusion loss or learned variance is non-finite")
        return {"total": total, "losses": losses, "variance": variance}
