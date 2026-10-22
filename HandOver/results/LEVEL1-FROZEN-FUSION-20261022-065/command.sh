set -euo pipefail
task_repo=/home/wangguangjie/djs/association-code
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_code=/home/wangguangjie/djs/association-code-run065
task_archive=/home/wangguangjie/djs/data/mtad
task_out="$task_repo/runs/LEVEL1-FROZEN-FUSION-20261022-065-attempt-1"
cd "$task_repo"
git fetch origin
test ! -e "$task_code"
test ! -e "$task_out"
test ! -e "$task_out-launch"
git worktree add --detach "$task_code" 606edd5
"$task_python" "$task_code/tools/experiments/launch_detached.py" \
  --cwd "$task_code" --record-dir "$task_out-launch" --time-policy advisory \
  --max-seconds 14400 --grace-seconds 30 \
  --env CUDA_VISIBLE_DEVICES= --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  -- "$task_python" "$task_code/tools/analysis/frozen_fusion_transfer.py" \
  --source-system "$task_repo/runs/LEVEL1-ASSOCIATION-20261022-056-attempt-1/system" \
  --source-fusion "$task_repo/runs/LEVEL1-FUSION-20261022-057-attempt-1" \
  --manifest "$task_code/configs/data/mtad-level1-scene3000.json" \
  --alignment "$task_code/configs/data/mtad-level1-alignment.json" \
  --archive "$task_archive/数据集level1.zip.001.zip" "$task_archive/数据集level1.zip.002.zip" "$task_archive/数据集level1.zip.003.zip" \
  --output "$task_out" --cpu-threads 4 --utility-replay-domain relu
