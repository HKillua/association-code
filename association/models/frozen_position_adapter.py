"""090 visible-scope residual adaptation of the strong final072C model."""
from __future__ import annotations

import torch
from torch import nn

from association.models.sequence_position import FEATURE_WIDTH, SequencePosition, tangent_position

SCOPE = 'frozen-position-adapter-090-v1'
ROUTING = 'teacher-final-tangent-increment-visible-anchor-support-v1'
MODES = ('all_queries', 'anchor_unsupported', 'full_finetune')


class FrozenPositionAdapter(nn.Module):
    def __init__(self, *, mode, adapter_hidden_size=32):
        super().__init__()
        if mode not in MODES or type(adapter_hidden_size) is not int or adapter_hidden_size != 32:
            raise ValueError('090 requires one declared mode and hidden32 adapter')
        self.mode = mode
        self.motion_observation_policy = 'canonical_all'
        self.routing_semantics = ROUTING
        self.teacher = SequencePosition(backbone='transformer')
        self.adapter = None
        if mode != 'full_finetune':
            self.teacher.requires_grad_(False)
            self.adapter = nn.Sequential(nn.Linear(FEATURE_WIDTH+64, 32), nn.SiLU(), nn.Linear(32, 2))
            nn.init.xavier_uniform_(self.adapter[0].weight)
            nn.init.zeros_(self.adapter[0].bias)
            nn.init.zeros_(self.adapter[2].weight)
            nn.init.zeros_(self.adapter[2].bias)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if self.mode != 'full_finetune':
            self.teacher.eval()
        return self

    def forward(self, base_states, features, mean_states, anchor_support):
        if (anchor_support.dtype != torch.bool or anchor_support.shape != (len(base_states),)
                or anchor_support.device != base_states.device
                or features.shape != (len(base_states), FEATURE_WIDTH)
                or features.dtype != torch.float32 or not torch.isfinite(features).all()
                or not torch.equal(features[:, 57], anchor_support.to(torch.float32))
                or mean_states.shape != base_states.shape or mean_states.dtype != torch.float64
                or not torch.isfinite(mean_states).all()
                or not torch.equal(base_states[:, 2:], mean_states[:, 2:])):
            raise ValueError('090 requires original visible features, Mean and explicit anchor support')
        if self.mode == 'full_finetune':
            return self.teacher(base_states, features, mean_states)
        with torch.no_grad():
            hidden = self.teacher.encoder(self.teacher.input_projection(features).unsqueeze(0)).squeeze(0)
            teacher_residual = self.teacher.output(hidden).to(torch.float64)*1000.
            teacher_states = tangent_position(base_states, teacher_residual)
        raw = self.adapter(torch.cat((features, hidden.detach()), -1)).to(torch.float64)*1000.
        allowed = torch.ones_like(anchor_support) if self.mode == 'all_queries' else ~anchor_support
        delta = raw * allowed[:, None]
        states = tangent_position(teacher_states, delta)
        return dict(states=states, base_states=teacher_states, mean_states=mean_states,
            residual_m=delta, raw_increment_m=raw, teacher_states=teacher_states,
            teacher_residual_m=teacher_residual, update_allowed=allowed)


def initialize_from_teacher(model, state, *, branch_seed):
    """Copy every source-final tensor; A/B share independent new-branch tensors."""
    model.teacher.load_state_dict(state, strict=True)
    if model.adapter is not None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(branch_seed)
            nn.init.xavier_uniform_(model.adapter[0].weight)
            nn.init.zeros_(model.adapter[0].bias)
            nn.init.zeros_(model.adapter[2].weight)
            nn.init.zeros_(model.adapter[2].bias)
    if any(not torch.equal(v.detach().cpu(), state[k]) for k, v in model.teacher.state_dict().items()):
        raise ValueError('090 source-final initialization must preserve all teacher tensors exactly')


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
