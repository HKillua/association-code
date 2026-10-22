set -euo pipefail
task_repo=/home/wangguangjie/djs/association-code
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_code=/home/wangguangjie/djs/association-code-run069
task_out="$task_repo/runs/LEVEL1-GATE-DIAGNOSTICS-20261023-069-attempt-1"
cd "$task_repo"
git fetch origin
test ! -e "$task_code"
test ! -e "$task_out"
test ! -e "$task_out-launch"
git worktree add --detach "$task_code" 4d7eee5
"$task_python" "$task_code/tools/experiments/launch_detached.py" \
  --cwd "$task_code" --record-dir "$task_out-launch" --time-policy advisory \
  --max-seconds 3600 --grace-seconds 30 \
  --env CUDA_VISIBLE_DEVICES= \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  -- "$task_python" "$task_code/tools/analysis/support_gate_diagnostics.py" \
  --source-068 "$task_repo/runs/LEVEL1-SUPPORT-GATE-20261023-068-attempt-1" \
  --output "$task_out" --cpu-threads 4
