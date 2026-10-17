"""TSA-cTFER's offline encoder and contrastive objective (RS 16, 3380).

The paper's online assignment procedure is not part of this module. Coordinates
must already be standardized using training-only statistics; timestamp mapping
belongs to the data adapter because the paper does not specify its vocabulary.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TSAConfig:
    length: int = 128
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    time_bins: int = 4096

    def __post_init__(self):
        for name in ("length", "d_model", "nhead", "num_layers",
                     "dim_feedforward", "time_bins"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if (isinstance(self.dropout, bool)
                or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout)
                or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in [0, 1)")


class TSActfer(nn.Module):
    def __init__(self, config: TSAConfig):
        super().__init__()
        self.config = config
        self.continuous = nn.Linear(2, config.d_model, bias=False)
        self.time_embedding = nn.Embedding(config.time_bins, config.d_model)
        self.padding = nn.Parameter(torch.zeros(config.d_model))
        self.input_projection = nn.Sequential(
            nn.ReLU(), nn.Linear(config.d_model, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False)
        self.heads = nn.ModuleDict({
            task: nn.Linear(config.d_model, config.d_model, bias=False)
            for task in ("itsa", "mstsa")})

        positions = torch.arange(config.length, dtype=torch.float32)[:, None]
        rates = torch.exp(torch.arange(0, config.d_model, 2, dtype=torch.float32)
                          * (-math.log(10000.0) / config.d_model))
        positional = torch.zeros(config.length, config.d_model)
        positional[:, 0::2] = torch.sin(positions * rates)
        positional[:, 1::2] = torch.cos(
            positions * rates[:positional[:, 1::2].shape[1]])
        self.register_buffer("positional", positional)

    def _validate_inputs(self, positions: Tensor, time_ids: Tensor,
                         observed: Tensor) -> None:
        if not all(isinstance(value, Tensor)
                   for value in (positions, time_ids, observed)):
            raise ValueError("positions, time_ids and observed must be tensors")
        if positions.ndim != 3 or positions.shape[2] != 2:
            raise ValueError("positions must have shape [B, L, 2]")
        batch, length, _ = positions.shape
        if batch < 1 or not 1 <= length <= self.config.length:
            raise ValueError("nonempty batch and 1 <= L <= config.length required")
        if time_ids.shape != (batch, length) or observed.shape != (batch, length):
            raise ValueError("time_ids and observed must have shape [B, L]")
        if time_ids.dtype != torch.long or observed.dtype != torch.bool:
            raise ValueError("time_ids must be long and observed must be bool")
        weight = self.continuous.weight
        if positions.dtype != weight.dtype or not positions.is_floating_point():
            raise ValueError("positions must have the model's floating dtype")
        if any(value.device != weight.device
               for value in (positions, time_ids, observed)):
            raise ValueError("all inputs must be on the model's device")
        if not observed.any(dim=1).all().item():
            raise ValueError("every trajectory must have at least one observed point")
        if not torch.isfinite(positions[observed]).all().item():
            raise ValueError("observed positions must be finite")
        valid_ids = time_ids[observed]
        if ((valid_ids < 0) | (valid_ids >= self.config.time_bins)).any().item():
            raise ValueError("observed time_ids are outside the timestamp vocabulary")

    def encode(self, positions: Tensor, time_ids: Tensor, observed: Tensor) -> Tensor:
        self._validate_inputs(positions, time_ids, observed)
        safe_positions = positions.masked_fill(~observed[..., None], 0)
        safe_time_ids = time_ids.masked_fill(~observed, 0)
        embedded = self.continuous(safe_positions) + self.time_embedding(safe_time_ids)
        embedded = torch.where(observed[..., None], embedded, self.padding)
        embedded = (self.input_projection(embedded)
                    + self.positional[:positions.shape[1]])
        encoded = self.encoder(embedded, src_key_padding_mask=~observed)
        # Eq. 25 masks padding; divide by actual support to avoid length scaling.
        # A bias-free projection followed by cosine cancels that scaling anyway.
        return (encoded.masked_fill(~observed[..., None], 0).sum(dim=1)
                / observed.sum(dim=1, keepdim=True))

    def project(self, pooled: Tensor, task: str) -> Tensor:
        if task not in self.heads:
            raise ValueError("task must be 'itsa' or 'mstsa'")
        if (not isinstance(pooled, Tensor) or pooled.ndim != 2
                or pooled.shape[0] < 1 or pooled.shape[1] != self.config.d_model):
            raise ValueError("pooled must have shape [B, d_model], B >= 1")
        weight = self.heads[task].weight
        if pooled.dtype != weight.dtype or pooled.device != weight.device:
            raise ValueError("pooled must have the model's dtype and device")
        if not torch.isfinite(pooled).all().item():
            raise ValueError("pooled representations must be finite")
        return self.heads[task](pooled)

    def forward(self, positions: Tensor, time_ids: Tensor, observed: Tensor,
                task: str = "itsa") -> Tensor:
        return self.project(self.encode(positions, time_ids, observed), task)


def symmetric_info_nce(embeddings: Tensor, temperature: float = 0.07) -> Tensor:
    """Eq. 27--29 for [left_1..left_B, right_1..right_B], B >= 2.

    Every non-self, non-partner entry is a negative. The data adapter must avoid
    repeated identities within a batch; this loss does not redefine negatives.
    """
    if (not isinstance(embeddings, Tensor) or embeddings.ndim != 2
            or embeddings.shape[0] < 4 or embeddings.shape[0] % 2
            or embeddings.shape[1] < 1 or not embeddings.is_floating_point()):
        raise ValueError("embeddings must be floating [2B, D], B >= 2 and D >= 1")
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("embeddings must be finite")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be finite and positive")
    normalized = F.normalize(embeddings, p=2, dim=1)
    logits = normalized @ normalized.T / temperature
    count = embeddings.shape[0]
    indices = torch.arange(count, device=embeddings.device)
    logits = logits.masked_fill(torch.eye(count, dtype=torch.bool,
                                         device=embeddings.device), -torch.inf)
    loss = F.cross_entropy(logits, (indices + count // 2) % count)
    if not torch.isfinite(loss).item():
        raise ValueError("InfoNCE produced a nonfinite loss")
    return loss
