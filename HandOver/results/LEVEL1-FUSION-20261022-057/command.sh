#!/bin/bash
set -euo pipefail
task_python=$1
task_code=$2
task_out=$3
task_repo=$4
task_archive=/home/wangguangjie/djs/data/mtad
task_parts=("$task_archive/数据集level1.zip.001.zip" "$task_archive/数据集level1.zip.002.zip" "$task_archive/数据集level1.zip.003.zip")
task_wp="$task_repo/runs/LEVEL1-SYSTEM-20261021-053A-attempt-1/ols/ols-working-point.json"
for task_arm in helper-quality direct-only mixed-action; do
  case "$task_arm" in
    helper-quality) task_config=evidence-fusion-level1.json ;;
    direct-only) task_config=evidence-fusion-level1-direct-only.json ;;
    mixed-action) task_config=evidence-fusion-level1-mixed-action.json ;;
  esac
  "$task_python" "$task_code/tools/experiments/evidence_fusion_fit.py" --mode fit \
    --archive "${task_parts[@]}" --manifest "$task_code/configs/data/mtad-level1-scene3000.json" \
    --alignment "$task_code/configs/data/mtad-level1-alignment.json" \
    --config "$task_code/configs/$task_config" --ols-working-point "$task_wp" \
    --device cuda:0 --cpu-threads 4 --output "$task_out/$task_arm/fit"
  "$task_python" "$task_code/tools/experiments/level1_system_evaluate.py" \
    --archive "${task_parts[@]}" --manifest "$task_code/configs/data/mtad-level1-scene3000.json" \
    --alignment "$task_code/configs/data/mtad-level1-alignment.json" --ols-working-point "$task_wp" \
    --fusion-checkpoint "$task_out/$task_arm/fit/final.pt" --fusion-run "$task_out/$task_arm/fit/run.json" \
    --ordinary-controls anchor mean_supported attribute_hybrid \
    --device cuda:0 --cpu-threads 4 --output "$task_out/$task_arm/system"
done
