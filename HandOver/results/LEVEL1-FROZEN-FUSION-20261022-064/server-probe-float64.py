"""Read-only FDE probe for the single 064 utility violation at [46,8,1] (direct_only, scene 1).

Uses only the preserved snapshot (complete motion + groups + old/new routing) and the
frozen 057 checkpoint. CPU only, no GPU, no writes outside /tmp, no tolerance change,
no task re-run. Compares the same forward in float32 and float64 from identical
float32 features, and reports the cancellation amplification of the utility head.
"""
import sys
from pathlib import Path
sys.path.insert(0, '/home/wangguangjie/djs/association-code-run064')
import numpy as np
import torch

from association.models.evidence_fusion import fusion_inputs
from association.training.evidence_fusion import load_checkpoint

torch.set_num_threads(4)
snap = np.load('/home/wangguangjie/djs/association-code/runs/LEVEL1-FROZEN-FUSION-20261022-064-attempt-1/replay-failure/snapshot.npz')
motion = {k[len('motion_'):]: snap[k] for k in snap.files if k.startswith('motion_')}
print('motion字段', sorted(motion), flush=True)
offsets, flat = snap['group_offsets'], snap['group_node_indices']
groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
print('OLS组', groups, flush=True)

routing_anchor = snap['routing_helper_anchor']
routing_node = snap['routing_helper_node']
cur_u = snap['routing_utility']
old_u = snap['old_ols_routing_utility']
idx = (46, 8, 1)
print('越界元素', idx, 'current', float(cur_u[idx]), 'previous(057,GPU)', float(old_u[idx]),
      'diff', float(abs(cur_u[idx] - old_u[idx])), flush=True)
anchor, helper = int(routing_anchor[46]), int(routing_node[46])
print('该pair anchor/helper', anchor, helper, flush=True)
group = next(g for g in groups if anchor in g)
members = [anchor] + [n for n in group if n != anchor]
helper_slot = members[1:].index(helper)
print('组成员', members, 'helper在members[1:]中的槽位', helper_slot, flush=True)

ckpt = Path('/home/wangguangjie/djs/association-code/runs/LEVEL1-FUSION-20261022-057-attempt-1/direct-only/fit/final.pt')
model32, snapshot = load_checkpoint(ckpt, device='cpu')
model32.eval().requires_grad_(False)
states, features, exact = fusion_inputs(motion, members, device='cpu')
print('states', tuple(states.shape), states.dtype, '| features', tuple(features.shape), features.dtype,
      '| exact', tuple(exact.shape), exact.dtype, flush=True)

with torch.no_grad():
    out32 = model32(states, features, exact)
u32 = out32['utility'][helper_slot, 8, 1].item()
print('CPU float32 重算 utility =', repr(u32), ' 与 snapshot current 逐位相同 =',
      bool(np.float32(u32) == np.float32(cur_u[idx])), flush=True)

model64 = load_checkpoint(ckpt, device='cpu')[0]
model64.eval().requires_grad_(False).double()
with torch.no_grad():
    out64 = model64(states, features.double(), exact)
u64 = out64['utility'][helper_slot, 8, 1].item()
prev = float(old_u[idx])
print('CPU float64 重算 utility =', repr(u64), flush=True)
print(f'  |f64 - f32(CPU)|   = {abs(u64 - u32):.6e}')
print(f'  |f64 - previous|   = {abs(u64 - prev):.6e}   (previous 来自 057 的 cuda:0 前向)')
print(f'  |f32 - previous|   = {abs(u32 - prev):.6e}   (这就是被判越界的 1.63e-06)')
print(f'  f64 是否落在 f32 与 previous 之间: {bool(min(u32, prev) <= u64 <= max(u32, prev))}', flush=True)

# 末层线性组合的抵消放大：utility = w·a + b，a 为 64 个 SiLU 激活
with torch.no_grad():
    h = model32.utility[0](features[helper_slot:helper_slot+1, 8:9, :])
    a32 = model32.utility[1](h)
    h2 = model32.utility[2](a32)
    a64 = model32.utility[3](h2)
    w, b = model32.utility[4].weight.double(), model32.utility[4].bias.double()
    act = a64.double()
    terms = (act * w)
    total = float(terms.sum() + b[1])
print(f'末层 64 项：sum|w_i*a_i| = {float(terms.abs().sum()):.6f}，代数和 = {total:.9f}，'
      f'bias = {float(b[1]):.6f}', flush=True)
print(f'抵消放大倍数 sum|term|/|result| = {float(terms.abs().sum()) / max(abs(total), 1e-30):.1f}', flush=True)
print(f'float32 在该量级的 1 ULP ≈ {np.spacing(np.float32(float(terms.abs().sum()))):.3e}；'
      f'64 项累加的预期绝对误差量级 ≈ {float(terms.abs().sum()) * 1.19e-7 * 8:.3e}', flush=True)
print(f'第一层输入特征绝对值最大 = {float(features[helper_slot, 8].abs().max()):.6f}，'
      f'第一层输出绝对值最大 = {float(a32.abs().max()):.6f}，'
      f'第二层输出绝对值最大 = {float(a64.abs().max()):.6f}', flush=True)
