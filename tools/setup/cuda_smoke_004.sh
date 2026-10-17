#!/usr/bin/env bash
# One dynamically selected CUDA device; co-tenancy authorized by the user on 2026-10-17.
# Arguments: <auto|index|GPU-UUID> [attempt=1] [cpu_attempt=attempt] [run_id=ENV-20261016-004] [cpu_run_id=run_id]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
(( $# <= 5 )) || exit 2
task_gpu_request=${1-}
[[ "$task_gpu_request" = auto || "$task_gpu_request" =~ ^[0-9]+$ || "$task_gpu_request" =~ ^GPU-[a-fA-F0-9-]+$ ]] || exit 2
case "${2-1}" in 1|2|3) task_attempt=${2-1} ;; *) exit 2 ;; esac
case "${3-$task_attempt}" in 1|2) task_cpu_attempt=${3-$task_attempt} ;; *) exit 2 ;; esac
task_python=/home/wangguangjie/djs/mtad-field-20261016/bin/python
task_run_id=${4-ENV-20261016-004}
task_cpu_run_id=${5-$task_run_id}
[[ "$task_run_id" =~ ^ENV-[0-9]{8}-[0-9]{3}$ && "$task_cpu_run_id" =~ ^ENV-[0-9]{8}-[0-9]{3}$ ]] || exit 2
task_out="HandOver/results/$task_run_id/attempt-$task_attempt"
task_cpu_out="HandOver/results/$task_cpu_run_id/attempt-$task_cpu_attempt"
task_run="runs/SMOKE-$task_run_id-attempt-$task_attempt"
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
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null

task_begin preflight
test "$(cat "$task_cpu_out/setup_exit_code.txt")" = 0
test "$(cat "$task_cpu_out/setup_wrapper_exit_code.txt")" = 0
test "$(cat "$task_cpu_out/setup_completed.txt")" = completed
test ! -e "$task_run"
"$task_python" - "$task_cpu_out" "$task_out/cuda" <<'CHECK'
import json, sys
from pathlib import Path
from tools.setup.verification_contract import capture_environment, validate_cpu_receipt
out, cuda_out = map(Path, sys.argv[1:])
root = Path.cwd().resolve()
environment = capture_environment(root)
(cuda_out / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
source = validate_cpu_receipt(root, out, environment)
if (cuda_out / 'actual_commit.txt').read_text().strip() != source['current_commit']:
    raise RuntimeError('Commit changed during CUDA preflight')
(cuda_out / 'cpu_source.json').write_text(json.dumps(source, indent=2) + '\n')
CHECK
task_done
task_run_logged pip_check "$task_out/cuda/pip_check.txt" "$task_python" -m pip --disable-pip-version-check check

task_begin allocation
printf '%s\n' "$task_gpu_request" > "$task_out/cuda/requested_gpu.txt"
task_gpu_uuid=$("$task_python" tools/setup/gpu_allocation.py \
  --device "$task_gpu_request" --output "$task_out/cuda/allocation.json")
[[ "$task_gpu_uuid" =~ ^GPU-[a-fA-F0-9-]+$ ]]
printf '%s\n' "$task_gpu_uuid" > "$task_out/cuda/physical_gpu_uuid.txt"
"$task_python" - "$task_out/cuda/allocation.json" <<'CHECK' > "$task_out/cuda/physical_gpu.txt"
import json, sys
print(json.load(open(sys.argv[1]))['selected']['index'])
CHECK
printf 'selected_with_sharing_allowed\n' > "$task_out/cuda/allocation_status.txt"
task_done
export CUDA_VISIBLE_DEVICES="$task_gpu_uuid"

task_run_logged environment_inventory "$task_out/cuda/environment.log" "$task_python" tools/pilot/server_check.py --workspace . --output "$task_out/cuda/inventory.json"
task_begin cuda_assertions
"$task_python" - <<'CHECK' > "$task_out/cuda/import_check.json"
import json, platform, torch, numpy
assert platform.python_version() == '3.11.7'
assert numpy.__version__ == '1.26.4'
assert torch.__version__ == '2.5.1+cu121'
assert torch.version.cuda == '12.1'
assert torch.cuda.is_available(), 'CUDA is unavailable; no CPU fallback'
assert torch.cuda.device_count() == 1, 'Expected exactly the allocated GPU'
print(json.dumps({'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                  'visible_gpu_count': torch.cuda.device_count(), 'name': torch.cuda.get_device_name(0),
                  'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory}, indent=2))
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
    'pilot_run_sha256': digest(run / 'pilot' / 'run.json'), 'report_sha256': digest(report_path),
    'allocation_sha256': digest(out / 'allocation.json'), 'gpu_policy': 'sharing_allowed',
    'performance_scope': 'concurrent_load_not_controlled'}, indent=2) + '\n')
CHECK
cp "$task_run/pilot/run.json" "$task_out/cuda/pilot_run.json"
task_done
task_complete
