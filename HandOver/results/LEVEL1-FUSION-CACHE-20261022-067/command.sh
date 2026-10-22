set -euo pipefail
task_repo=/home/wangguangjie/djs/association-code
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_code=/home/wangguangjie/djs/association-code-run067
task_out="$task_repo/runs/LEVEL1-FUSION-CACHE-20261022-067-attempt-1"
cd "$task_repo"
git fetch origin
test ! -e "$task_code"
test ! -e "$task_out"
test ! -e "$task_out-launch"
git worktree add --detach "$task_code" a208767
"$task_python" "$task_code/tools/experiments/launch_detached.py" \
  --cwd "$task_code" --record-dir "$task_out-launch" --time-policy advisory \
  --max-seconds 3600 --grace-seconds 30 \
  --env CUDA_VISIBLE_DEVICES= --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
  -- "$task_python" "$task_code/tools/analysis/fusion_cache_attribution.py" \
  --source-066 "$task_repo/runs/LEVEL1-FROZEN-FUSION-20261022-066-attempt-1" \
  --source-system "$task_repo/runs/LEVEL1-ASSOCIATION-20261022-056-attempt-1/system" \
  --source-fusion "$task_repo/runs/LEVEL1-FUSION-20261022-057-attempt-1" \
  --manifest "$task_code/configs/data/mtad-level1-scene3000.json" \
  --output "$task_out" --cpu-threads 4
