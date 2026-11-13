"""Report-time motion means adapted from the NEMOT recurrent transition."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


HISTORY_STEPS = 6
MAX_NEIGHBORS = 4
FEATURE_COUNT = 5
HIDDEN_SIZE = 32
RESIDUAL_SPEED_MPS = 12.0
RESIDUAL_SCALE_M = 100.0


@dataclass(frozen=True)
class MotionBatch:
    """Left-aligned real histories and their report-time neighbor snapshots.

    Own features are displacement east/north, elapsed seconds, and velocity
    east/north. Neighbor features are relative east/north position, east/north
    velocity, and age in seconds. Coordinates and velocities use SI units.
    Absent neighbors have zero weight. The center is the common KF prediction;
    targets are supplied only for training or selection.
    """

    own: Tensor
    neighbors: Tensor
    neighbor_weights: Tensor
    lengths: Tensor
    horizon: Tensor
    center: Tensor
    target: Tensor | None = None

    def __post_init__(self):
        count = len(self.lengths)
        own_shape = (count, HISTORY_STEPS, FEATURE_COUNT)
        neighbor_shape = (count, HISTORY_STEPS, MAX_NEIGHBORS, FEATURE_COUNT)
        if (self.own.shape != own_shape or self.neighbors.shape != neighbor_shape
                or self.neighbor_weights.shape != neighbor_shape[:-1]
                or self.lengths.shape != (count,) or self.horizon.shape != (count,)
                or self.center.shape != (count, 2)
                or (self.target is not None and self.target.shape != (count, 2))):
            raise ValueError("invalid NEMOT motion batch shape")

    def __len__(self):
        return len(self.lengths)

    def take(self, indices: Tensor | slice):
        target = None if self.target is None else self.target[indices]
        return MotionBatch(self.own[indices], self.neighbors[indices], self.neighbor_weights[indices],
                           self.lengths[indices], self.horizon[indices], self.center[indices], target)


class NEMOTMotion(nn.Module):
    """Encode individual and inverse-distance neighbor features with two GRUs.

    The transition follows the structure of NEMOT §IV-B. A horizon-conditioned
    decoder predicts a bounded velocity residual to the shared KF mean.
    """

    def __init__(self, use_neighbors: bool):
        super().__init__()
        self.use_neighbors = use_neighbors
        self.register_buffer("own_scale", torch.tensor((100.0, 100.0, 20.0, 10.0, 10.0)), persistent=False)
        self.register_buffer("neighbor_scale", torch.tensor((4000.0, 4000.0, 10.0, 10.0, 60.0)), persistent=False)
        self.own_encoder = nn.Sequential(nn.Linear(FEATURE_COUNT, HIDDEN_SIZE), nn.SiLU(),
                                         nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU())
        self.neighbor_encoder = nn.Sequential(nn.Linear(FEATURE_COUNT, HIDDEN_SIZE), nn.SiLU(),
                                              nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU())
        self.own_step = nn.GRUCell(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fused_step = nn.GRUCell(2 * HIDDEN_SIZE, HIDDEN_SIZE)
        self.decode = nn.Sequential(nn.Linear(HIDDEN_SIZE + 1, HIDDEN_SIZE), nn.SiLU(), nn.Linear(HIDDEN_SIZE, 2))

    def forward(self, batch: MotionBatch) -> Tensor:
        own = self.own_encoder(batch.own / self.own_scale)
        if self.use_neighbors:
            features = self.neighbor_encoder(batch.neighbors / self.neighbor_scale)
            neighbors = (features * batch.neighbor_weights.unsqueeze(-1)).sum(dim=2)
        else:
            neighbors = torch.zeros_like(own)

        hidden = own.new_zeros((len(batch), HIDDEN_SIZE))
        for step in range(HISTORY_STEPS):
            individual = self.own_step(own[:, step], hidden)
            fused = self.fused_step(torch.cat((individual, neighbors[:, step]), dim=1), hidden)
            hidden = torch.where((batch.lengths > step).unsqueeze(1), fused, hidden)

        horizon = batch.horizon.unsqueeze(1)
        time_feature = torch.log1p(horizon) / math.log(61.0)
        velocity = RESIDUAL_SPEED_MPS * torch.tanh(self.decode(torch.cat((hidden, time_feature), dim=1)))
        enabled = (batch.lengths >= 2).unsqueeze(1) & (horizon > 0)
        return torch.where(enabled, horizon * velocity, torch.zeros_like(velocity))


@torch.inference_mode()
def predict_residuals(model: NEMOTMotion, batch: MotionBatch) -> Tensor:
    """Return east/north residuals in metres for addition to the original KF means."""
    return model(batch)


def _targets(batch: MotionBatch) -> Tensor:
    return (batch.target - batch.center).to(dtype=batch.own.dtype)


@torch.inference_mode()
def _selection_mae(model: NEMOTMotion, selection: MotionBatch, batch_size: int) -> float:
    error_sum = 0.0
    for start in range(0, len(selection), batch_size):
        batch = selection.take(slice(start, start + batch_size))
        errors = torch.linalg.vector_norm(model(batch) - _targets(batch), dim=1)
        error_sum += errors.double().sum().item()
    return error_sum / len(selection)


def fit_motion(
    train: MotionBatch,
    selection: MotionBatch,
    *,
    use_neighbors: bool,
    epochs: int = 6,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 27,
) -> tuple[NEMOTMotion, dict]:
    """Fit one fixed-budget CPU arm and restore its best selection checkpoint."""
    if not len(train) or not len(selection) or train.target is None or selection.target is None:
        raise ValueError("training and selection require nonempty batches with targets")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = NEMOTMotion(use_neighbors)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sampler = torch.Generator().manual_seed(seed)
    best = None
    best_epoch = None
    best_mae = math.inf
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        updates = 0
        for indices in torch.randperm(len(train), generator=sampler).split(batch_size):
            batch = train.take(indices)
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(batch) / RESIDUAL_SCALE_M, _targets(batch) / RESIDUAL_SCALE_M)
            if not math.isfinite(loss.item()):
                raise FloatingPointError("nonfinite NEMOT training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            updates += 1

        model.eval()
        mae = _selection_mae(model, selection, batch_size)
        if not math.isfinite(mae):
            raise FloatingPointError("nonfinite NEMOT selection error")
        history.append(dict(epoch=epoch, train_samples=len(train), updates=updates,
                            train_smooth_l1=total_loss / len(train), selection_samples=len(selection),
                            selection_mae_m=mae))
        if mae < best_mae:
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch, best_mae = epoch, mae

    model.load_state_dict(best)
    return model, dict(selected_epoch=best_epoch, selected_mae_m=best_mae, epochs=history)
