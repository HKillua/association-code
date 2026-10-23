set -euo pipefail
: "${task_arm:?设置为071A或071B或071C或071D}"
: "${task_gpu_uuid:?先按FIFO确认并设置该臂独占GPU UUID}"
task_repo=/home/wangguangjie/djs/association-code
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_code="/home/wangguangjie/djs/association-code-run${task_arm}"
task_archive=/home/wangguangjie/djs/data/mtad
case "$task_arm" in
  071A) task_name=LEVEL1-GATE-LOG-20261023-071A-attempt-1
        task_config=support-gate-replica-log-level1.json
        task_entry=("$task_python" "$task_code/tools/experiments/support_gate_replication.py" fit) ;;
  071B) task_name=LEVEL1-GATE-LINEAR-20261023-071B-attempt-1
        task_config=support-gate-replica-linear-level1.json
        task_entry=("$task_python" "$task_code/tools/experiments/support_gate_replication.py" fit) ;;
  071C) task_name=LEVEL1-SEQUENCE-LSTM-20261023-071C-attempt-1
        task_config=sequence-position-lstm-level1.json
        task_entry=("$task_python" "$task_code/tools/experiments/sequence_position_fit.py") ;;
  071D) task_name=LEVEL1-SEQUENCE-TRANSFORMER-20261023-071D-attempt-1
        task_config=sequence-position-transformer-level1.json
        task_entry=("$task_python" "$task_code/tools/experiments/sequence_position_fit.py") ;;
  *) echo '未知071任务，停止' >&2; exit 2 ;;
esac
task_out="$task_repo/runs/$task_name"
cd "$task_repo"
git fetch origin
test ! -e "$task_code"
test ! -e "$task_out"
test ! -e "$task_out-launch"
git worktree add --detach "$task_code" b7ad49c
"$task_python" "$task_code/tools/experiments/launch_detached.py" \
  --cwd "$task_code" --record-dir "$task_out-launch" --time-policy advisory \
  --max-seconds 14400 --grace-seconds 30 \
  --env CUDA_VISIBLE_DEVICES="$task_gpu_uuid" \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  -- "${task_entry[@]}" \
  --source-system "$task_repo/runs/LEVEL1-ASSOCIATION-20261022-056-attempt-1/system" \
  --manifest "$task_code/configs/data/mtad-level1-scene3000.json" \
  --alignment "$task_code/configs/data/mtad-level1-alignment.json" \
  --config "$task_code/configs/$task_config" \
  --archive "$task_archive/数据集level1.zip.001.zip" "$task_archive/数据集level1.zip.002.zip" "$task_archive/数据集level1.zip.003.zip" \
  --output "$task_out" --device cuda:0 --cpu-threads 4 --time-policy advisory
