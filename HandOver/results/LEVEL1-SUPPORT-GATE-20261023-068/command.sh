set -euo pipefail
: "${task_gpu_uuid:?请先按现场资源选择并设置实际GPU UUID}"
task_repo=/home/wangguangjie/djs/association-code
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_code=/home/wangguangjie/djs/association-code-run068
task_archive=/home/wangguangjie/djs/data/mtad
task_out="$task_repo/runs/LEVEL1-SUPPORT-GATE-20261023-068-attempt-1"
cd "$task_repo"
git fetch origin
test ! -e "$task_code"
test ! -e "$task_out"
test ! -e "$task_out-launch"
git worktree add --detach "$task_code" 11a70a5
"$task_python" "$task_code/tools/experiments/launch_detached.py" \
  --cwd "$task_code" --record-dir "$task_out-launch" --time-policy advisory \
  --max-seconds 14400 --grace-seconds 30 \
  --env CUDA_VISIBLE_DEVICES="$task_gpu_uuid" \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  -- "$task_python" "$task_code/tools/experiments/support_gate_fit.py" \
  --source-system "$task_repo/runs/LEVEL1-ASSOCIATION-20261022-056-attempt-1/system" \
  --manifest "$task_code/configs/data/mtad-level1-scene3000.json" \
  --alignment "$task_code/configs/data/mtad-level1-alignment.json" \
  --config "$task_code/configs/support-gate-level1.json" \
  --archive "$task_archive/数据集level1.zip.001.zip" "$task_archive/数据集level1.zip.002.zip" "$task_archive/数据集level1.zip.003.zip" \
  --output "$task_out" --device cuda:0 --cpu-threads 4
