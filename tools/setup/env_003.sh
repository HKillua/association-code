#!/usr/bin/env bash
# Run only through the bounded wrapper in L2S-20261016-003.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
task_env=/home/wangguangjie/djs/mtad-env-py311-cu121-env003
task_base=/home/wangguangjie/anaconda3/bin/python
task_store=/home/wangguangjie/djs/package-cache/ENV-20261016-003
task_out=HandOver/results/ENV-20261016-003
task_stage=preflight
mkdir "$task_out"
exec >"$task_out/setup.log" 2>&1
task_finish() {
  task_rc=$?
  printf '%s\n' "$task_rc" > "$task_out/setup_exit_code.txt"
  date -u +%FT%TZ > "$task_out/setup_ended_at.txt"
  printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$task_stage" "exit=$task_rc" >> "$task_out/stages.tsv"
  # Record only files actually saved, including after a later download fails.
  "$task_base" tools/setup/wheel_inventory.py "$task_store/wheels" > "$task_out/wheels.json" || true
}
trap task_finish EXIT
task_begin() {
  task_stage=$1
  printf '%s\t%s\tstart\n' "$(date -u +%FT%TZ)" "$task_stage" >> "$task_out/stages.tsv"
}
task_done() {
  printf '%s\t%s\tcomplete\n' "$(date -u +%FT%TZ)" "$task_stage" >> "$task_out/stages.tsv"
}
date -u +%FT%TZ > "$task_out/setup_started_at.txt"
git rev-parse HEAD > "$task_out/actual_commit.txt"
task_begin preflight
test ! -e "$task_env"
test ! -e "$task_store"
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
"$task_base" -m venv "$task_env"
task_python="$task_env/bin/python"
mkdir -p "$(dirname "$task_store")"
mkdir "$task_store" "$task_store/wheels" "$task_store/pip-cache"
task_wheels="$task_store/wheels"
"$task_python" -m pip --version > "$task_out/pip_version.txt"
task_done

# Clear only this child's pip configuration; leave the user's configuration alone.
# This prevents a user/global index, target, or no-cache setting changing the plan.
task_pip() {
  "$task_python" - --require-virtualenv --disable-pip-version-check --no-input \
    --retries 2 --timeout 120 --cache-dir "$task_store/pip-cache" \
    --log "$task_store/pip.log" "$@" <<'PIP'
import os, subprocess, sys
env = {key: value for key, value in os.environ.items() if not key.startswith('PIP_')}
env['PIP_CONFIG_FILE'] = os.devnull
raise SystemExit(subprocess.call([sys.executable, '-m', 'pip', *sys.argv[1:]], env=env))
PIP
}

task_begin download_bootstrap
task_pip download --only-binary=:all: --progress-bar off --dest "$task_wheels" \
  --index-url https://pypi.org/simple numpy==1.26.4 'setuptools>=68' wheel
task_done
task_begin download_torch_primary
task_pip download --only-binary=:all: --progress-bar off --no-deps --dest "$task_wheels" \
  --index-url https://download.pytorch.org/whl/cu121 'torch==2.5.1+cu121'
task_done
task_begin download_torch_dependencies
# The primary wheel has been saved by the completed --no-deps command above.
# Pass its local path so another dependency's metadata failure cannot discard it.
task_torch="$task_wheels/torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl"
test -s "$task_torch"
task_pip download --only-binary=:all: --progress-bar off --dest "$task_wheels" \
  --find-links "$task_wheels" --index-url https://download.pytorch.org/whl/cu121 "$task_torch"
task_done
task_begin offline_install
task_pip install --no-index --find-links "$task_wheels" --only-binary=:all: \
  numpy==1.26.4 'torch==2.5.1+cu121' 'setuptools>=68' wheel
task_pip install --no-index --no-deps --no-build-isolation -e .
task_pip check > "$task_out/pip_check.txt"
task_pip freeze --all > "$task_out/pip_freeze.txt"
task_done
task_begin import_check
"$task_python" - <<'CHECK' > "$task_out/import_check.json"
import json, sys, numpy, torch, association
assert sys.version_info[:2] == (3, 11)
assert numpy.__version__ == '1.26.4'
assert torch.__version__ == '2.5.1+cu121'
assert torch.version.cuda == '12.1'
assert torch.cuda.device_count() == 0
print(json.dumps({'python': sys.version, 'numpy': numpy.__version__, 'torch': torch.__version__,
                  'cuda_build': torch.version.cuda, 'visible_gpu_count': torch.cuda.device_count(),
                  'association': association.__file__}, indent=2))
CHECK
task_done
task_begin cpu_tests
"$task_python" -m unittest discover -s tests > "$task_out/tests.log" 2>&1
"$task_python" -m unittest discover -s tools/pilot -p test_audit_mtad_level2.py > "$task_out/legacy_tests.log" 2>&1
"$task_python" -m association.cli --help > "$task_out/cli_help.txt"
task_done
task_stage=completed
