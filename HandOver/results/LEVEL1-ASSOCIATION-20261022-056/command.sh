#!/bin/bash
set -euo pipefail
task_python=$1
task_code=$2
task_out=$3
task_repo=$4
task_archive=/home/wangguangjie/djs/data/mtad
task_parts=("$task_archive/数据集level1.zip.001.zip" "$task_archive/数据集level1.zip.002.zip" "$task_archive/数据集level1.zip.003.zip")
"$task_python" "$task_code/tools/experiments/evidence_association_fit.py" fit \
  --archive "${task_parts[@]}" --manifest "$task_code/configs/data/mtad-level1-scene3000.json" \
  --config "$task_code/configs/evidence-association-level1.json" \
  --device cuda:0 --cpu-threads 4 --output "$task_out/association"
exec "$task_python" "$task_code/tools/experiments/level1_system_evaluate.py" \
  --archive "${task_parts[@]}" --manifest "$task_code/configs/data/mtad-level1-scene3000.json" \
  --alignment "$task_code/configs/data/mtad-level1-alignment.json" \
  --ols-working-point "$task_repo/runs/LEVEL1-SYSTEM-20261021-053A-attempt-1/ols/ols-working-point.json" \
  --association-scores "$task_out/association/final-scores.npz" \
  --association-summary "$task_out/association/summary.json" \
  --association-data-audit "$task_out/association/data_audit.json" \
  --association-run "$task_out/association/run.json" \
  --association-checkpoint "$task_out/association/final.pt" --association-endpoint final \
  --ordinary-controls anchor mean_supported attribute_hybrid \
  --device cuda:0 --cpu-threads 4 --output "$task_out/system"
