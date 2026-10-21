"""TMTAFF-style embedded endpoint completion and Siamese association.

This is an independently specified association reference, not an author-code
reproduction. See docs/model_conformance.md for the paper ambiguities and the
deliberate decisions. Values must already use TRAINING-fitted normalization.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 4
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 256
    dropout: float = 0.1
    length: int = 64
    margin: float = 1.0
    contrastive_positive_power: int = 1
    heading_representation: str = "linear_zscore"
    # None preserves the historical shared dropout. An explicit value changes
    # only TAM, so frozen completion can retain its original configuration.
    tam_dropout: float | None = None
    # Explicit objective adaptation; the printed paper formula remains default.
    contrastive_negative_scale: str = "paper"

    def __post_init__(self):
        for name in ("input_dim", "d_model", "nhead", "num_layers", "dim_feedforward", "length"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.tam_dropout is not None and (
                isinstance(self.tam_dropout, bool)
                or not isinstance(self.tam_dropout, (float, int))
                or not math.isfinite(self.tam_dropout) or not 0 <= self.tam_dropout < 1):
            raise ValueError("tam_dropout must be None or a finite value in [0, 1)")
        if not math.isfinite(self.margin) or self.margin <= 0:
            raise ValueError("margin must be finite and positive")
        if self.contrastive_positive_power not in (1, 2):
            raise ValueError("contrastive_positive_power must be 1 or 2")
        if self.contrastive_negative_scale not in ("paper", "inverse_margin"):
            raise ValueError("contrastive_negative_scale must be paper or inverse_margin")
        if self.contrastive_negative_scale == "inverse_margin" and self.contrastive_positive_power != 1:
            raise ValueError("inverse_margin requires the linear contrastive positive term")
        if self.heading_representation not in ("linear_zscore", "circular_sincos"):
            raise ValueError("heading_representation must be linear_zscore or circular_sincos")

    @property
    def effective_tam_dropout(self) -> float:
        return self.dropout if self.tam_dropout is None else self.tam_dropout


def _sinusoidal_positions(length: int, dimension: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    result = torch.zeros(length, dimension)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies[: dimension // 2])
    return result


def _causal_mask(length: int, device: torch.device) -> Tensor:
    """True entries are inaccessible to attention, including cross attention."""
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


class AssociationModel(nn.Module):
    """Actual TCM encoder/decoder and separate, shared Siamese TAM encoder.

    encode_tracks returns [N, length*d_model], preserving the paper's complete
    temporal representation. ``embedding_dim`` reports that dimension; no mean
    pooling or independent pair-specific encoder is used.
    """

    def __init__(self, config: ModelConfig, branch: str = "ce"):
        super().__init__()
        if branch not in ("ce", "contrastive"):
            raise ValueError("branch must be 'ce' or 'contrastive'")
        if config.heading_representation != "linear_zscore":
            raise ValueError("circular heading representation requires the native-time model")
        self.config = config
        self.branch = branch
        self._completion_frozen = False
        self.embedding_dim = config.length * config.d_model
        self.input_embedding = nn.Linear(config.input_dim, config.d_model, bias=False)
        self.register_buffer("positions", _sinusoidal_positions(config.length, config.input_dim))

        def encoder(dropout):
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model, nhead=config.nhead,
                dim_feedforward=config.dim_feedforward, dropout=dropout,
                activation="gelu", batch_first=True,
            )
            return nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)

        self.tcm_encoder = encoder(config.dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward, dropout=config.dropout,
            activation="gelu", batch_first=True,
        )
        self.tcm_decoder = nn.TransformerDecoder(decoder_layer, config.num_layers)
        self.tam_encoder = encoder(config.effective_tam_dropout)
        self.output_dropout = nn.Dropout(config.effective_tam_dropout)
        # Eq.16 specifies an absolute difference and a single linear projection.
        self.ce_classifier = nn.Linear(self.embedding_dim, 1, bias=False) if branch == "ce" else None
        # Transformer containers clone layers; independently initialize their
        # weights, rather than leave every depth with identical initial values.
        self._reset_weights()

    def completion_parameters(self, include_embedding: bool = False):
        """Parameters for TCM pretraining; embedding targets may remain frozen."""
        if include_embedding:
            yield from self.input_embedding.parameters()
        yield from self.tcm_encoder.parameters()
        yield from self.tcm_decoder.parameters()

    def set_completion_trainable(self, trainable: bool, include_embedding: bool = True):
        """Choose an explicit frozen or end-to-end association training stage.

        Frozen TCM also disables its dropout, including after model.train().
        This does not toggle the TAM or change the current model training mode.
        """
        self._completion_frozen = not trainable
        for parameter in self.completion_parameters(include_embedding=include_embedding):
            parameter.requires_grad_(trainable)
        self.tcm_encoder.train(self.training and trainable)
        self.tcm_decoder.train(self.training and trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._completion_frozen:
            self.tcm_encoder.eval()
            self.tcm_decoder.eval()
        return self

    def _reset_weights(self):
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

    def _validate_inputs(self, values: Tensor, observed: Tensor) -> Tensor:
        expected = (self.config.length, self.config.input_dim)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(f"values must have shape [N, {expected[0]}, {expected[1]}]")
        if observed.shape != values.shape[:2] or observed.dtype != torch.bool:
            raise ValueError("observed must be a boolean [N, length] mask")
        if values.device != observed.device:
            raise ValueError("values and observed must use the same device")
        if not values.is_floating_point():
            raise ValueError("values must be floating point")
        if not torch.isfinite(values[observed]).all():
            raise ValueError("observed values must be finite")
        counts = observed.sum(dim=1)
        if (counts == 0).any():
            raise ValueError("each track must contain at least one observed grid point")
        # Endpoint completion is not an internal-gap imputer. Data ingestion
        # determines interpolation inside support and records that policy.
        if len(observed):
            indices = torch.arange(self.config.length, device=observed.device)
            first = torch.where(observed, indices, self.config.length).min(dim=1).values
            last = torch.where(observed, indices, -1).max(dim=1).values
            if ((last - first + 1) != counts).any():
                raise ValueError("internal gaps in observed support are unsupported; resample explicitly")
        return observed

    def embed_inputs(self, values: Tensor, observed: Tensor) -> Tensor:
        """ReLU(W(values + positional vector)); absent input values are ignored."""
        self._validate_inputs(values, observed)
        safe = torch.where(observed[..., None], values, torch.zeros_like(values))
        return F.relu(self.input_embedding(safe + self.positions.to(dtype=values.dtype)))

    def _support_groups(self, observed: Tensor):
        """One device synchronization; batch equal-support tracks together."""
        groups = {}
        for track_index, row in enumerate(observed.detach().cpu().tolist()):
            start = row.index(True)
            end = len(row) - list(reversed(row)).index(True)
            groups.setdefault((start, end), []).append(track_index)
        for (start, end), members in groups.items():
            yield start, end, torch.tensor(members, dtype=torch.long, device=observed.device)

    def _decode_causal_history(self, history: Tensor, memory: Tensor) -> Tensor:
        """Use the same prefix visibility in teacher forcing and endpoint rollout.

        Memory contains only the observed sequence. Once history grows with
        generated points, cross attention is rectangular: observed prefix query
        i can access memory j <= i, while generated queries can access all
        observed memory. Earlier decoder queries must remain causal too, since
        later decoder layers attend to their outputs.
        """
        mask = _causal_mask(history.shape[1], history.device)
        memory_mask = (
            torch.arange(memory.shape[1], device=history.device)[None, :]
            > torch.arange(history.shape[1], device=history.device)[:, None]
        )
        return self.tcm_decoder(history, memory, tgt_mask=mask, memory_mask=memory_mask)

    def _teacher_predictions(self, history: Tensor) -> Tensor:
        """Predict each next embedding with access only to its historical prefix.

        Both encoder and decoder cross-attention require causal masks. Masking
        only decoder self-attention would leak later teacher-forcing targets
        through the encoder's memory.
        """
        mask = _causal_mask(history.shape[1], history.device)
        memory = self.tcm_encoder(history, mask=mask)
        return self._decode_causal_history(history, memory)

    def completion_loss(self, values: Tensor, observed: Tensor) -> Tensor:
        """Bidirectional next-embedding MSE on observed points only.

        No measurements from other fragments or reference state are targets.
        Targets are detached; the caller should freeze input_embedding during
        completion-only pretraining to prevent a moving/collapsing target space.
        Singleton supports have no supervised transition and contribute zero.
        """
        embedded = self.embed_inputs(values, observed)
        # Retain a zero gradient path even if the embedding is frozen and a
        # minibatch contains only singleton supports (no temporal targets).
        total = embedded.sum() * 0.0 + next(self.tcm_encoder.parameters()).reshape(-1)[0] * 0.0
        target_elements = 0
        for start, end, members in self._support_groups(observed):
            if end - start < 2:
                continue
            original = embedded[members, start:end]
            # Batch both directions in the same Transformer call.
            sequences = torch.cat((original, original.flip(1)), dim=0)
            prediction = self._teacher_predictions(sequences[:, :-1])
            targets = sequences[:, 1:].detach()
            total = total + F.mse_loss(prediction, targets, reduction="sum")
            target_elements += targets.numel()
        return total / max(target_elements, 1)

    def _extrapolate(self, sequence: Tensor, steps: int) -> Tensor:
        """Predict future embeddings autoregressively from known context only."""
        if steps == 0:
            return sequence[:, :0]
        # Match teacher forcing's prefix representations, including earlier
        # memory tokens. Merely hiding future targets in the final decoder query
        # would not make a multilayer decoder's training/rollout paths agree.
        memory = self.tcm_encoder(sequence, mask=_causal_mask(sequence.shape[1], sequence.device))
        history = sequence
        predictions = []
        for _ in range(steps):
            decoded = self._decode_causal_history(history, memory)
            prediction = decoded[:, -1:]
            predictions.append(prediction)
            history = torch.cat((history, prediction), dim=1)
        return torch.cat(predictions, dim=1)

    def complete(self, values: Tensor, observed: Tensor) -> Tensor:
        """Complete only temporal endpoints, returning [N,L,d_model].

        Pre-completion reverses the already embedded observed sequence, predicts
        toward earlier times, and reverses the predictions back. Observed input
        embeddings remain bit-for-bit unchanged. The operation is differentiable.
        """
        embedded = self.embed_inputs(values, observed)
        if values.shape[0] == 0:
            return embedded
        completed = embedded.clone()
        for start, end, members in self._support_groups(observed):
            sequence = embedded[members, start:end]
            if start:
                before = self._extrapolate(sequence.flip(1), start).flip(1)
                completed[members, :start] = before
            if end < self.config.length:
                after = self._extrapolate(sequence, self.config.length - end)
                completed[members, end:] = after
        return completed

    def encode_tracks(self, values: Tensor, observed: Tensor) -> Tensor:
        complete = self.complete(values, observed)
        if values.shape[0] == 0:
            return complete.reshape(0, self.embedding_dim)
        return self.tam_encoder(complete).flatten(start_dim=1)

    def score_pairs(self, embeddings: Tensor, pairs: Tensor) -> Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"embeddings must have shape [N, {self.embedding_dim}]")
        pairs = torch.as_tensor(pairs, dtype=torch.long, device=embeddings.device)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape [E, 2]")
        if pairs.numel() and ((pairs < 0).any() or (pairs >= len(embeddings)).any()):
            raise ValueError("pair index outside embeddings")
        difference = embeddings[pairs[:, 0]] - embeddings[pairs[:, 1]]
        if self.branch == "ce":
            return self.ce_classifier(self.output_dropout(difference.abs())).squeeze(-1)
        # Raw score increases with similarity; no learned probability claim.
        return -torch.linalg.vector_norm(difference, ord=2, dim=-1)

    def association_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        labels = torch.as_tensor(labels, dtype=scores.dtype, device=scores.device)
        if scores.ndim != 1 or scores.shape != labels.shape:
            raise ValueError("scores and labels must have identical [E] shape")
        if not torch.isfinite(scores).all() or not torch.isfinite(labels).all():
            raise ValueError("scores and labels must be finite")
        if ((labels != 0) & (labels != 1)).any():
            raise ValueError("association labels must be binary")
        if not scores.numel():
            return scores.sum() * 0.0
        if self.branch == "ce":
            return F.binary_cross_entropy_with_logits(scores, labels)
        if (scores > 0).any():
            raise ValueError("contrastive raw scores must be negative Euclidean distances")
        distance = -scores
        positive = distance.pow(self.config.contrastive_positive_power)
        negative = F.relu(self.config.margin - distance).square()
        if self.config.contrastive_negative_scale == "inverse_margin":
            negative = negative / self.config.margin
        return 0.5 * (labels * positive + (1 - labels) * negative).mean()
