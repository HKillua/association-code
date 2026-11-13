"""100: paired native-event residuals and an added-budget strong-model control."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from association.data.source_events import group_events
from association.models.native_query import NativeQueryPosition
from association.models.sequence_position import SequencePosition, sequence_inputs, sequence_objectives
from association.training.evidence_fusion import prepare_evidence_scenes, training_members, TRAIN_MODES
from association.utils.runtime import write_json

SCOPE = 'native-query-fusion-100-v1'
ARMS = ('budget', 'mean', 'query')


class NativeExperiment(nn.Module):
    def __init__(self, base, arm, model_config):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('unknown native experiment arm')
        self.arm = arm
        self.base = deepcopy(base).eval().requires_grad_(False)
        self.branch = (deepcopy(base).requires_grad_(True) if arm == 'budget' else
                       NativeQueryPosition(pooling=arm, **model_config))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, motion, members):
        device = next(self.parameters()).device
        inputs = sequence_inputs(motion, members, device=device)
        with torch.no_grad():
            base = self.base(*inputs)['states']
        if self.arm == 'budget':
            result = self.branch(*inputs)
        else:
            result = self.branch(group_events(motion, members), base, motion['query_seconds'])
        return dict(result, frozen_base_states=base)


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('scope') != SCOPE or saved.get('step') != saved.get('actual_updates')
            or not 0 <= saved['step'] <= saved['configuration']['training']['updates']):
        raise ValueError('native checkpoint scope or successful update mismatch')
    base = SequencePosition(**saved['base_model_config'])
    base.load_state_dict(saved['base_state'], strict=True)
    config = saved['configuration']
    model = NativeExperiment(base, config['arm'], config['model'])
    model.branch.load_state_dict(saved['branch_state'], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite native checkpoint')
    return model.to(device), saved


def fit(train_raw, config, base, base_model_config, replay_rows, output, *, source, device='cpu', check=None):
    """Replay actual successful source samples; no random resampling or V selection.

    A tiny fixture can call this function, but only the CLI may declare a real
    Level1 run. Its source binding checks the full original protocol first.
    """
    if config.get('scope') != SCOPE or config.get('arm') not in ARMS:
        raise ValueError('native training requires an explicit scope and arm')
    train_raw = list(train_raw)
    if not train_raw or any(s.get('split') != 'T_core' for s in train_raw):
        raise ValueError('only T_core may enter native training')
    training = config['training']; steps = training['updates']
    if len(replay_rows) != steps:
        raise ValueError('actual source replay length differs from fixed budget')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(training['seed'])
    model = NativeExperiment(base, config['arm'], config['model']).to(device)
    parameters = list(model.branch.parameters())
    optimizer = torch.optim.Adam(parameters, lr=training['learning_rate'])
    frozen = {k: v.detach().cpu().clone() for k, v in model.base.state_dict().items()}
    exposure = dict(successful_updates=0, sampled_anchors=0, valid_position_points=0,
                    scenes={}, anchors={}, modes={m: dict(updates=0, anchors=0, valid_position_points=0) for m in TRAIN_MODES})
    state = dict(scope=SCOPE, status='RUNNING', configuration=config, optimizer_updates=0,
        training_scene_ids=[str(s['scene_id']) for s in train_raw], exposure=exposure,
        source=source, test_opened=False, selection_policy='fixed final; no V in fit',
        sampling='exact replay of source successful scene/anchor/member batches',
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameter_count=sum(p.numel() for p in parameters),
        frozen_base_parameter_count=sum(p.numel() for p in model.base.parameters()),
        total_source_training_updates=source.get('source_total_updates'), additional_updates=steps,
        gradient_clipping=dict(clipped_updates=0), first100=None)
    write_json(output/'run.json', state)

    def save(name, step):
        torch.save(dict(scope=SCOPE, configuration=config, step=step, actual_updates=step,
            base_model_config=base_model_config, base_state=frozen, branch_state=model.branch.state_dict(),
            optimizer=optimizer.state_dict(), source=source,
            torch_rng=torch.get_rng_state()), output/name)

    try:
        save('step-0.pt', 0)
        prepared = prepare_evidence_scenes(train_raw, check=check, motion_observation_policy='canonical_all',
                                           retain_native_events=config['arm'] != 'budget')
        train = {str(s['scene_id']): s for s in prepared}
        if len(train) != len(prepared):
            raise ValueError('duplicate training scene')
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        fit_started = time.monotonic()
        with (output/'training.jsonl').open('w') as log:
            for step, old in enumerate(replay_rows, 1):
                if check: check()
                if (old['step'] != step or old['mode'] != TRAIN_MODES[(step-1) % 3]
                        or len(old['batch']) != training['batch_size']):
                    raise ValueError('source update/mode/batch differs')
                tick = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
                batch, losses = [], []
                for sample in old['batch']:
                    scene = train[sample['scene_id']]; anchor = sample['anchor_index']
                    members = training_members(scene, anchor, old['mode'])
                    result = model(scene['motion'], members)
                    target = torch.as_tensor(scene['target'][anchor], dtype=torch.float64, device=device)
                    valid = torch.as_tensor(scene['valid'][anchor], dtype=torch.bool, device=device)
                    objectives, diagnostics = sequence_objectives(result, target, valid)
                    objective = objectives['linear']
                    if not torch.isfinite(objective): raise FloatingPointError('nonfinite native objective')
                    eligible = ~scene['motion']['support'][anchor] & scene['motion']['support'][members[1:]].any(0)
                    actual = dict(scene_id=str(scene['scene_id']), anchor_index=anchor,
                        node_id=str(scene['node_ids'][anchor]), members=members,
                        valid_position_points=diagnostics['valid_position_points'],
                        eligible_query_points=int(eligible.sum()),
                        eligible_valid_position_points=int((eligible & valid[:, :2].all(-1).cpu().numpy()).sum()))
                    if actual != sample:
                        raise ValueError(f'actual input/target exposure differs before update {step}')
                    (objective/training['batch_size']).backward()
                    batch.append(actual); losses.append(float(objective.detach()))
                norm = torch.nn.utils.clip_grad_norm_(parameters, training['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                state['optimizer_updates'] = step
                if not all(torch.isfinite(p).all() for p in parameters):
                    raise FloatingPointError('nonfinite native parameters after optimizer update')
                exposure['successful_updates'] += 1
                exposure['sampled_anchors'] += len(batch)
                exposure['modes'][old['mode']]['updates'] += 1
                for sample in batch:
                    count = sample['valid_position_points']
                    exposure['valid_position_points'] += count
                    exposure['modes'][old['mode']]['anchors'] += 1
                    exposure['modes'][old['mode']]['valid_position_points'] += count
                    for key, value in (('scenes', sample['scene_id']), ('anchors', sample['node_id'])):
                        exposure[key][value] = exposure[key].get(value, 0)+1
                clipped = bool(norm > training['gradient_clip'])
                state['gradient_clipping']['clipped_updates'] += int(clipped)
                row = dict(step=step, mode=old['mode'], batch=batch, loss=float(np.mean(losses)),
                           gradient_norm=float(norm), gradient_clipped=clipped, seconds=time.monotonic()-tick)
                log.write(json.dumps(row)+'\n'); log.flush()
                if step == min(100, steps):
                    elapsed = time.monotonic()-fit_started
                    state['first100'] = dict(updates=step, seconds=elapsed, seconds_per_update=elapsed/step,
                        projected_fit_seconds=elapsed/step*steps, projection_is_not_guarantee=True)
                if step % 100 == 0 or step == steps:
                    state.update(elapsed_seconds=time.monotonic()-started)
                    write_json(output/'run.json', state)
        if any(not torch.equal(v, model.base.state_dict()[k].detach().cpu()) for k, v in frozen.items()):
            raise ValueError('frozen strong base changed')
        save('final.pt', steps)
        state.update(status='COMPLETED', final_checkpoint='final.pt', frozen_base_unchanged=True,
            actual_successful_batches_matched=steps, elapsed_seconds=time.monotonic()-started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == 'cuda' else None)
        write_json(output/'run.json', state)
        return state
    except BaseException as error:
        # Preserve this attempt's last complete optimizer update; never auto-resume.
        save('interrupted.pt', state['optimizer_updates'])
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise
