set -euo pipefail
task_repo=/home/wangguangjie/djs/association-code
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_code=/home/wangguangjie/djs/association-code-run072A
task_out="$task_repo/runs/LEVEL1-POSITION-BENCHMARK-20261023-072-summary-attempt-1"
test ! -e "$task_out"
test ! -e "$task_out-launch"
"$task_python" "$task_code/tools/experiments/launch_detached.py" \
  --cwd "$task_code" --record-dir "$task_out-launch" --time-policy advisory \
  --max-seconds 3600 --grace-seconds 30 \
  --env CUDA_VISIBLE_DEVICES= \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  -- "$task_python" "$task_code/tools/analysis/compare_position_benchmarks.py" \
  --log-run071A "$task_repo/runs/LEVEL1-GATE-LOG-20261023-071A-attempt-1" \
  --linear-run071B "$task_repo/runs/LEVEL1-GATE-LINEAR-20261023-071B-attempt-1" \
  --lstm-run071C "$task_repo/runs/LEVEL1-SEQUENCE-LSTM-20261023-071C-attempt-1" \
  --transformer-run071D "$task_repo/runs/LEVEL1-SEQUENCE-TRANSFORMER-20261023-071D-attempt-1" \
  --source-070 "$task_repo/runs/LEVEL1-LINEAR-GATE-20261023-070-attempt-1" \
  --constant-run072A "$task_repo/runs/LEVEL1-CONSTANT-GATE-20261023-072A-attempt-1" \
  --lstm-run072B "$task_repo/runs/LEVEL1-SEQUENCE-LSTM-20261023-072B-attempt-1" \
  --transformer-run072C "$task_repo/runs/LEVEL1-SEQUENCE-TRANSFORMER-20261023-072C-attempt-1" \
  --output "$task_out" --cpu-threads 4
