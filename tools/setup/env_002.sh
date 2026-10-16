#!/usr/bin/env bash
# ENV-002 installation/CPU checks only. GPU execution is a separate handover step.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
task_env=/home/wangguangjie/djs/mtad-env-py311-cu121
task_base=/home/wangguangjie/anaconda3/bin/python
task_out=HandOver/results/ENV-20261016-002
mkdir "$task_out"
exec >"$task_out/setup.log" 2>&1
trap 'task_rc=$?; printf "%s\n" "$task_rc" > "$task_out/setup_exit_code.txt"; date -u +%FT%TZ > "$task_out/setup_ended_at.txt"' EXIT
date -u +%FT%TZ > "$task_out/setup_started_at.txt"
git rev-parse HEAD > "$task_out/actual_commit.txt"
test ! -e "$task_env"
"$task_base" - <<'CHECK'
import platform, shutil
assert platform.python_version() == '3.11.7', platform.python_version()
assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
assert shutil.disk_usage('/home/wangguangjie/djs').free >= 20 * 2**30
CHECK
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
"$task_base" -m venv "$task_env"
task_python="$task_env/bin/python"
"$task_python" -m pip install --retries 0 --timeout 60 --no-cache-dir --only-binary=:all: numpy==1.26.4
"$task_python" -m pip install --retries 0 --timeout 60 --no-cache-dir --only-binary=:all: torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$task_python" -m pip install --retries 0 --timeout 60 --no-cache-dir --no-deps -e .
"$task_python" -m pip check > "$task_out/pip_check.txt"
"$task_python" -m pip freeze > "$task_out/pip_freeze.txt"
"$task_python" - <<'CHECK' > "$task_out/import_check.json"
import json, sys, numpy, torch, association
assert sys.version_info[:2] == (3, 11)
assert numpy.__version__ == '1.26.4'
assert torch.__version__ == '2.5.1+cu121', torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
assert torch.cuda.device_count() == 0  # Deliberately hidden until allocated.
print(json.dumps({'python': sys.version, 'numpy': numpy.__version__,
                  'torch': torch.__version__, 'build_cuda': torch.version.cuda,
                  'gpu_hidden_for_setup': True}, indent=2))
CHECK
"$task_python" -m unittest discover -s tests > "$task_out/tests.log" 2>&1
"$task_python" -m unittest discover -s tools/pilot -p test_audit_mtad_level2.py > "$task_out/legacy_tests.log" 2>&1
"$task_python" -m association.cli --help > "$task_out/cli_help.txt"
