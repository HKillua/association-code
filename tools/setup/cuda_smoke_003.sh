#!/usr/bin/env bash
# The server AI selects one currently free card under the lab's FIFO policy.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
case "${1-}" in 0|1|2|3) task_gpu=$1 ;; *) exit 2 ;; esac
task_python=/home/wangguangjie/djs/mtad-env-py311-cu121-env003/bin/python
task_out=HandOver/results/ENV-20261016-003
test "$(cat "$task_out/setup_exit_code.txt")" = 0
test "$(cat "$task_out/setup_wrapper_exit_code.txt")" = 0
test ! -e runs/SMOKE-ENV-20261016-003
mkdir "$task_out/cuda"
exec >"$task_out/cuda/smoke.log" 2>&1
trap 'task_rc=$?; printf "%s\n" "$task_rc" > "$task_out/cuda/exit_code.txt"; date -u +%FT%TZ > "$task_out/cuda/ended_at.txt"' EXIT
date -u +%FT%TZ > "$task_out/cuda/started_at.txt"
git rev-parse HEAD > "$task_out/cuda/actual_commit.txt"
printf '%s\n' "$task_gpu" > "$task_out/cuda/physical_gpu.txt"
# Bind the nvidia-smi physical card by UUID; CUDA ordinals may differ.
task_gpu_uuid=$(nvidia-smi -i "$task_gpu" --query-gpu=uuid --format=csv,noheader)
[[ "$task_gpu_uuid" =~ ^GPU-[a-fA-F0-9-]+$ ]]
printf '%s\n' "$task_gpu_uuid" > "$task_out/cuda/physical_gpu_uuid.txt"
nvidia-smi -i "$task_gpu" --query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu --format=csv > "$task_out/cuda/allocation_snapshot.csv"
export CUDA_VISIBLE_DEVICES="$task_gpu_uuid"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
"$task_python" tools/pilot/server_check.py --workspace . --output "$task_out/cuda/environment.json"
"$task_python" - <<'CHECK'
import torch
assert torch.__version__ == '2.5.1+cu121'
assert torch.cuda.is_available(), 'CUDA is unavailable; no CPU fallback'
assert torch.cuda.device_count() == 1, 'Expected exactly the allocated GPU'
assert '3090' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
CHECK
"$task_python" -m association.cli smoke --config configs/smoke.json --device cuda:0 --cpu-threads 1 --output runs/SMOKE-ENV-20261016-003
cp runs/SMOKE-ENV-20261016-003/pilot/run.json "$task_out/cuda/pilot_run.json"
