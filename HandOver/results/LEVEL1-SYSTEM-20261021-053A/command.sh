#!/bin/bash
set -euo pipefail
task_python=$1
task_code=$2
task_out=$3
task_archive=/home/wangguangjie/djs/data/mtad
"$task_python" "$task_code/tools/experiments/level1_ols_working_point.py" \
  --archive "$task_archive/数据集level1.zip.001.zip" "$task_archive/数据集level1.zip.002.zip" "$task_archive/数据集level1.zip.003.zip" \
  --manifest "$task_code/configs/data/mtad-level1-scene3000.json" \
  --output "$task_out/ols"
exec "$task_python" "$task_code/tools/experiments/level1_system_evaluate.py" \
  --archive "$task_archive/数据集level1.zip.001.zip" "$task_archive/数据集level1.zip.002.zip" "$task_archive/数据集level1.zip.003.zip" \
  --manifest "$task_code/configs/data/mtad-level1-scene3000.json" \
  --alignment "$task_code/configs/data/mtad-level1-alignment.json" \
  --ols-working-point "$task_out/ols/ols-working-point.json" \
  --output "$task_out/system" --device cpu --cpu-threads 4
