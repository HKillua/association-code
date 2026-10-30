"""148: real T8-only bounded training connection, not a trained fair baseline.

Scratch seed20, eight updates of four groups, fixed u0/u8 probes; no V/test,
old peer sampler/weights or further automatic training. Separate attempt per
method. PTFM uses two sensor branches; DeepAF uses its full P/V core.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.models import deepaf_peer as deep
from association.models.ptfm import PTFM, objectives
from association.models.fusion_peer_inputs import sensor_segments, ptfm_segment_inputs, deepaf_visible_inputs, group_mean
from association.models.evidence_fusion import physical_errors
from association.utils.runtime import read_json, write_json

IDEA = 'IDEA-FUSION-PEER-T-CONNECTION-20261030'


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def training_pool(native, manifest):
    run = read_json(native/'run.json')
    cache = read_json(native/'native-cache.json')
    ids = manifest['splits']['T_core'][:8]
    if (run['status'] != 'COMPLETED' or run['exit_code'] != 0
            or cache['native_motion_policy'] != 'canonical_all'
            or cache['scene_ids']['T_core'][:8] != ids):
        raise ValueError('completed original canonical_all T8 cache required')
    pool, counts = [], {}
    for sid in ids:
        motion = load(native/'private/native-motion/T_core'/f'scene-{sid}.npz')
        ev = load(native/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
        if not np.array_equal(motion['node_ids'], ev['node_ids']):
            raise ValueError('T node order mismatch')
        motion['states'] = motion['states'].copy()
        motion['states'][..., 2] *= 1852/3600
        registry = {str(t): i for i, t in enumerate(ev['target_ids'])}
        before = len(pool)
        for identity in dict.fromkeys(ev['identities'].tolist()):
            members = np.flatnonzero(ev['identities'] == identity)
            segments = sensor_segments(motion, members)
            if not segments or str(identity) not in registry:
                continue
            # Fixed longest common segment; ties keep earliest. No V selection.
            segment = max(segments, key=lambda s: len(s['query_indices']))
            target, valid = ev['target_states'][registry[str(identity)]], ev['target_valid'][registry[str(identity)]]
            if not valid[segment['query_indices'], :2].all(-1).any():
                continue
            pool.append(dict(scene_id=sid, members=members, motion=motion, segment=segment,
                             target=target, valid=valid))
        counts[sid] = len(pool)-before
    if not pool:
        raise ValueError('no T8 eligible groups')
    return pool, counts


def fit_normalizer(pool):
    arrays = []
    for row in pool:
        a, b, _, _ = ptfm_segment_inputs(row['segment'])
        arrays.extend((a.numpy(), b.numpy()))
    values = np.concatenate(arrays)
    return dict(mean=values.mean(0).tolist(), scale=np.maximum(values.std(0), 1e-6).tolist(),
                split='T_core', truth_used=False, count=len(values), speed_unit='mps')


def ptfm_loss(model, rows, device):
    losses, diagnostics = [], []
    for row in rows:
        indices = row['segment']['query_indices']
        result = model(*ptfm_segment_inputs(row['segment'], device=device))
        loss, detail = objectives(model, result,
            torch.as_tensor(row['target'][indices], dtype=torch.float64, device=device),
            torch.as_tensor(row['valid'][indices], dtype=torch.bool, device=device))
        losses.append(loss); diagnostics.append(detail['task_valid_counts'])
    return torch.stack(losses).mean(), dict(task_valid_counts=np.sum(diagnostics, axis=0).tolist())


def deepaf_loss(model, rows, device):
    batch = deep.deepaf_batch_inputs([deepaf_visible_inputs(r['motion'], r['members'], device=device) for r in rows])
    rollout = model.forward_batch(batch, cache=True)
    target = torch.as_tensor(np.stack([r['target'] for r in rows]), dtype=torch.float64, device=device)
    valid = torch.as_tensor(np.stack([r['valid'] for r in rows]), dtype=torch.bool, device=device)
    supervision = deep.deepaf_supervision(rollout, target, valid)
    # B*Q=256 queries fit in this bounded probe; TBPTT1 cache keeps own history.
    losses, _ = deep.deepaf_objectives(model.replay_batch(rollout), supervision)
    return losses['total'], dict(valid_position_points=sum(supervision['valid_position_points']),
                                valid_velocity_points=sum(supervision['valid_velocity_points']))


@torch.no_grad()
def probe(model, rows, method, device):
    errors, control, swaps = [], [], []
    for row in rows:
        mean = group_mean(row['motion'], row['members'])
        if method == 'ptfm':
            states = torch.as_tensor(mean, dtype=torch.float64, device=device).clone()
            result = model(*ptfm_segment_inputs(row['segment'], device=device))
            indices = row['segment']['query_indices']
            states[indices] = result['states']
            swap = model(*ptfm_segment_inputs(row['segment'], device=device, swapped=True))
            swaps.extend(physical_errors(result['states'], swap['states'])[:, 0].cpu().tolist())
        else:
            states = model(deepaf_visible_inputs(row['motion'], row['members'], device=device), cache=False)['states']
        target = torch.as_tensor(row['target'], dtype=torch.float64, device=device)
        valid = torch.as_tensor(row['valid'], dtype=torch.bool, device=device)
        safe = torch.where(valid, target, torch.zeros_like(target))
        use = valid[:, :2].all(-1)
        errors.extend(physical_errors(states, safe)[:, 0][use].cpu().tolist())
        control.extend(physical_errors(torch.as_tensor(mean, dtype=torch.float64, device=device), safe)[:, 0][use].cpu().tolist())
    def stats(values):
        x = np.asarray(values)
        return dict(count=len(x), mae_m=float(x.mean()), rmse_m=float(np.sqrt(np.mean(x*x))),
                    p95_m=float(np.quantile(x, .95)), p99_m=float(np.quantile(x, .99))) if len(x) else None
    return dict(position=stats(errors), mean_control=stats(control), ptfm_swap_displacement=stats(swaps),
                scope='fixed first sampled T batch; PTFM core plus Mean outside longest common segment; not V or fair ranking')


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    (output/'private').mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id=IDEA, plan_version='1.0', method=args.method,
        started_at=datetime.now(timezone.utc).isoformat(), optimizer_updates=0, seed=20, batch_groups=4,
        planned_updates=8, native_run=str(args.native_run.resolve()), manifest=str(args.manifest.resolve()),
        test_opened=False, validation_opened=False, raw_reference_read=False,
        old_weights_loaded=False, source_sampler_replayed=False, input_speed_unit='mps',
        prediction_speed_unit='mps', time_policy='advisory', device=args.device)
    def checkpoint(**extra):
        state.update(extra, elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if sys.platform.startswith('linux') else 1))
        write_json(output/'run.json', state)
        if state['peak_rss_bytes'] > 6*1024**3:
            raise MemoryError('6GiB resident budget exceeded')
    try:
        torch.set_num_threads(4)
        torch.manual_seed(20)
        checkpoint(phase='prepare')
        pool, counts = training_pool(args.native_run.resolve(), read_json(args.manifest))
        normalizer = fit_normalizer(pool) if args.method == 'ptfm' else None
        rng = np.random.default_rng(20)
        plan = rng.integers(0, len(pool), (8, 4))
        write_json(output/'private/sample-plan.json', dict(indices=plan.tolist(),
            pool=[dict(scene_id=r['scene_id'], members=r['members'].tolist(), query_indices=r['segment']['query_indices'].tolist()) for r in pool]))
        model = (PTFM(normalizer, sensor_branches=True) if args.method == 'ptfm' else deep.DeepAFPeer()).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        selected = [pool[i] for i in plan[0]]
        model.eval()
        u0 = probe(model, selected, args.method, args.device)
        checkpoint(phase='training', training_scene_ids=list(counts), eligible_groups_by_scene=counts,
            eligible_groups=len(pool), parameter_count=sum(p.numel() for p in model.parameters()))
        loss_fn = ptfm_loss if args.method == 'ptfm' else deepaf_loss
        logs = []
        for step, batch in enumerate(plan, 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, detail = loss_fn(model, [pool[i] for i in batch], args.device)
            if args.method == 'ptfm':
                loss = loss+1e-4*model.l2()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            logs.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), **detail))
            checkpoint(optimizer_updates=step)
        model.eval()
        u8 = probe(model, selected, args.method, args.device)
        final = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        changes = {name: sum(not torch.equal(value, initial[k]) for k, value in final.items() if k.startswith(name))
            for name in (('anchor', 'helper') if args.method == 'ptfm' else ('position_network', 'velocity_network'))}
        if not all(changes.values()):
            raise ValueError('required independent branches did not update')
        torch.save(dict(scope='fusion-peer-t8-new-unit-v1', method=args.method, model=final,
            normalizer=normalizer, optimizer=optimizer.state_dict(), updates=8), output/'private/final.pt')
        write_json(output/'training.json', logs)
        write_json(output/'summary.json', dict(status='COMPLETED', idea_id=IDEA, method=args.method,
            u0=u0, u8=u8, changed_tensors_by_branch=changes, optimizer_updates=8,
            training_group_presentations=32, train_probe_only=True, normalization=normalizer,
            limitations=['T8 only, not full2100 or V300; no baseline performance qualification',
                'PTFM same-source pooling/longest common segment/output projection/loss interpretations are disclosed adaptations',
                'DeepAF offline OLS anchor/actual dt/TBPTT1/finite difference V are disclosed adaptations']))
        checkpoint(status='COMPLETED', exit_code=0, phase='completed',
            finished_at=datetime.now(timezone.utc).isoformat(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0)
    except BaseException as error:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('ptfm', 'deepaf'), required=True)
    parser.add_argument('--native-run', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    run(parser.parse_args())
