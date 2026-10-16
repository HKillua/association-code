#!/usr/bin/env bash
# One server-selected free card only, under the bounded ENV-004 CUDA wrapper.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
case "${1-}" in 0|1|2|3) task_gpu=$1 ;; *) exit 2 ;; esac
case "${2-1}" in 1|2) task_attempt=${2-1} ;; *) exit 2 ;; esac
case "${3-$task_attempt}" in 1|2) task_cpu_attempt=${3-$task_attempt} ;; *) exit 2 ;; esac
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_out="HandOver/results/ENV-20261016-004/attempt-$task_attempt"
task_cpu_out="HandOver/results/ENV-20261016-004/attempt-$task_cpu_attempt"
task_run="runs/SMOKE-ENV-20261016-004-attempt-$task_attempt"
mkdir -p "$task_out"
mkdir "$task_out/cuda"
exec > "$task_out/cuda/smoke.log" 2>&1
source tools/setup/run_status.sh
task_status_init "$task_out/cuda" ''
git rev-parse HEAD > "$task_out/cuda/actual_commit.txt"
printf '%s\n' "$task_cpu_out" > "$task_out/cuda/cpu_source_directory.txt"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME

task_begin preflight
test "$(cat "$task_cpu_out/setup_exit_code.txt")" = 0
test "$(cat "$task_cpu_out/setup_wrapper_exit_code.txt")" = 0
test "$(cat "$task_cpu_out/setup_completed.txt")" = completed
test ! -e "$task_run"
"$task_python" - "$task_cpu_out" "$task_out/cuda" <<'CHECK'
import json, sys
from pathlib import Path
from association.runtime import code_fingerprint, digest
out, cuda_out = map(Path, sys.argv[1:])
report = json.loads((out / 'validation.json').read_text())
assert report['status'] == 'COMPLETED'
cpu_commit = (out / 'actual_commit.txt').read_text().strip()
assert report['actual_commit'] == cpu_commit, 'CPU commit evidence mismatch'
assert (cuda_out / 'actual_commit.txt').read_text().strip() == cpu_commit, 'Commit changed after CPU verification'
assert report['import_check']['code_sha256'] == code_fingerprint(), 'Source changed after CPU verification'
for name, expected in json.loads((out / 'source_files.json').read_text()).items():
    assert digest(Path(name)) == expected, 'Source/config changed after CPU verification: ' + name
for name in ('pip_check', 'pip_freeze', 'tests', 'legacy_tests', 'cli_help'):
    assert report['checks'][name] == 0
    assert (out / (name + '_exit_code.txt')).read_text().strip() == '0'
(cuda_out / 'cpu_source.json').write_text(json.dumps({'directory': str(out.resolve()),
    'actual_commit': cpu_commit, 'validation_sha256': digest(out / 'validation.json'),
    'source_files_sha256': digest(out / 'source_files.json')}, indent=2) + '\n')
CHECK
task_done

task_begin allocation
printf '%s\n' "$task_gpu" > "$task_out/cuda/physical_gpu.txt"
task_gpu_uuid=$(timeout --signal=TERM --kill-after=2s 15s nvidia-smi -i "$task_gpu" --query-gpu=uuid --format=csv,noheader)
[[ "$task_gpu_uuid" =~ ^GPU-[a-fA-F0-9-]+$ ]]
printf '%s\n' "$task_gpu_uuid" > "$task_out/cuda/physical_gpu_uuid.txt"
timeout --signal=TERM --kill-after=2s 15s nvidia-smi -i "$task_gpu_uuid" \
  --query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu --format=csv > "$task_out/cuda/allocation_snapshot.csv"
# Fail closed on unsupported/failed queries or any compute process on this UUID.
task_query_rc=0
timeout --signal=TERM --kill-after=2s 15s nvidia-smi -i "$task_gpu_uuid" \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader \
  > "$task_out/cuda/compute_apps.csv" 2> "$task_out/cuda/compute_apps.stderr" || task_query_rc=$?
printf '%s\n' "$task_query_rc" > "$task_out/cuda/compute_apps_exit_code.txt"
if (( task_query_rc != 0 )); then
  printf 'allocation_query_failed\n' > "$task_out/cuda/allocation_status.txt"
  exit "$task_query_rc"
fi
if [[ -n "$(tr -d '[:space:]' < "$task_out/cuda/compute_apps.csv")" ]]; then
  printf 'deferred_gpu_busy\n' > "$task_out/cuda/allocation_status.txt"
  exit 3
fi
printf 'free_at_prelaunch_check\n' > "$task_out/cuda/allocation_status.txt"
task_done
export CUDA_VISIBLE_DEVICES="$task_gpu_uuid"

task_run_logged environment_inventory "$task_out/cuda/environment.log" "$task_python" tools/pilot/server_check.py --workspace . --output "$task_out/cuda/environment.json"
task_begin cuda_assertions
"$task_python" - <<'CHECK' > "$task_out/cuda/import_check.json"
import json, platform, torch, numpy
assert platform.python_version() == '3.11.7'
assert numpy.__version__ == '1.26.4'
assert torch.__version__ == '2.5.1+cu121'
assert torch.version.cuda == '12.1'
assert torch.cuda.is_available(), 'CUDA is unavailable; no CPU fallback'
assert torch.cuda.device_count() == 1, 'Expected exactly the allocated GPU'
assert '3090' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print(json.dumps({'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                  'visible_gpu_count': torch.cuda.device_count(), 'name': torch.cuda.get_device_name(0)}, indent=2))
CHECK
task_done
task_run_logged smoke_command "$task_out/cuda/smoke_command.log" "$task_python" -m association.cli smoke --config configs/smoke.json --device cuda:0 --cpu-threads 1 --output "$task_run"
task_begin acceptance
"$task_python" - "$task_run" "$task_out/cuda" "$task_gpu_uuid" <<'CHECK'
import json, sys
from pathlib import Path
from association.runtime import digest
run, out = map(Path, sys.argv[1:3])
metadata = json.loads((run / 'pilot' / 'run.json').read_text())
assert metadata['status'] == 'COMPLETED'
assert metadata['provenance']['device'] == 'cuda:0'
assert metadata['provenance']['cuda_visible_devices'] == sys.argv[3]
assert metadata['cuda_process_peak_allocated_bytes'] > 0
report_path = run / 'pilot' / 'report.json'
assert digest(report_path) == metadata['report_sha256']
assert json.loads(report_path.read_text())['protocol'] == 'synthetic-smoke-only-not-research-result'
(out / 'acceptance.json').write_text(json.dumps({'status': 'COMPLETED', 'run': str(run),
    'pilot_run_sha256': digest(run / 'pilot' / 'run.json'), 'report_sha256': digest(report_path)}, indent=2) + '\n')
CHECK
cp "$task_run/pilot/run.json" "$task_out/cuda/pilot_run.json"
task_done
task_complete
