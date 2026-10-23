set -euo pipefail
task_repo=/home/wangguangjie/djs/association-code
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_code=/home/wangguangjie/djs/association-code-run075
task_out="$task_repo/runs/LEVEL1-TSA-SCORE-DIAGNOSTICS-20261024-075-attempt-1"
cd "$task_repo"
git fetch origin
test ! -e "$task_code"
test ! -e "$task_out"
test ! -e "$task_out-launch"
git worktree add --detach "$task_code" 72caeac
"$task_python" "$task_code/tools/experiments/launch_detached.py" \
  --cwd "$task_code" --record-dir "$task_out-launch" --time-policy advisory \
  --max-seconds 3600 --grace-seconds 30 \
  --env CUDA_VISIBLE_DEVICES= \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  -- "$task_python" "$task_code/tools/analysis/tsactfer_score_diagnostics.py" \
  --source-074d "$task_repo/runs/LEVEL1-TSA-NATIVE-20261023-074D-attempt-1" \
  --output "$task_out" --cpu-threads 4
