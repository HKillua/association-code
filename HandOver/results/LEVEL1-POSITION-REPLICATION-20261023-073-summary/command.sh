set -euo pipefail
task_repo=/home/wangguangjie/djs/association-code
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_code=/home/wangguangjie/djs/association-code-run073A
task_out="$task_repo/runs/LEVEL1-POSITION-REPLICATION-20261023-073-summary-attempt-1"
test ! -e "$task_out"
test ! -e "$task_out-launch"
"$task_python" "$task_code/tools/experiments/launch_detached.py" \
  --cwd "$task_code" --record-dir "$task_out-launch" --time-policy advisory \
  --max-seconds 3600 --grace-seconds 30 \
  --env CUDA_VISIBLE_DEVICES= \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  -- "$task_python" "$task_code/tools/analysis/compare_position_replication.py" \
  --gate-run073A "$task_repo/runs/LEVEL1-GATE-LINEAR-20261023-073A-attempt-1" \
  --gate-run073B "$task_repo/runs/LEVEL1-GATE-LINEAR-20261023-073B-attempt-1" \
  --transformer-run073C "$task_repo/runs/LEVEL1-SEQUENCE-TRANSFORMER-20261023-073C-attempt-1" \
  --transformer-run072C "$task_repo/runs/LEVEL1-SEQUENCE-TRANSFORMER-20261023-072C-attempt-1" \
  --source071B "$task_repo/runs/LEVEL1-GATE-LINEAR-20261023-071B-attempt-1" \
  --source070 "$task_repo/runs/LEVEL1-LINEAR-GATE-20261023-070-attempt-1" \
  --output "$task_out" --cpu-threads 4
