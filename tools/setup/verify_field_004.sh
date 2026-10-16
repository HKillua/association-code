#!/usr/bin/env bash
# Verification only. Run through the current handover budget wrapper; never installs packages.
# Arguments: [attempt=1] [run_id=ENV-20261016-004]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
task_repo=$(pwd -P)
(( $# <= 2 )) || exit 2
case "${1-1}" in 1|2) task_attempt=${1-1} ;; *) exit 2 ;; esac
task_env=/home/wangguangjie/djs/mtad-field-20261016
task_python="$task_env/bin/python"
task_run_id=${2-ENV-20261016-004}
[[ "$task_run_id" =~ ^ENV-[0-9]{8}-[0-9]{3}$ ]] || exit 2
task_root="HandOver/results/$task_run_id"
task_out="$task_root/attempt-$task_attempt"
mkdir -p "$task_root"
mkdir "$task_out"
exec > "$task_out/setup.log" 2>&1
source tools/setup/run_status.sh
task_status_init "$task_out" setup
git rev-parse HEAD > "$task_out/actual_commit.txt"
printf '%s\n' "$task_attempt" > "$task_out/attempt.txt"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME
# These commands need no index or network. Do not alter the existing environment.
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INDEX=1 PIP_CONFIG_FILE=/dev/null

task_begin preflight
test -x "$task_python"
"$task_python" - "$task_env" <<'CHECK'
from pathlib import Path
import platform, sys
assert platform.system() == 'Linux', platform.system()
assert platform.machine() == 'x86_64', platform.machine()
assert platform.python_version() == '3.11.7', platform.python_version()
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve(), sys.prefix
assert sys.prefix != sys.base_prefix, 'Expected the existing FIELD virtual environment'
CHECK
task_done

task_begin evidence
"$task_python" - "$task_repo" "$task_out" <<'EVIDENCE'
import hashlib, json, shutil, sys, sysconfig
from pathlib import Path

root, out = map(Path, sys.argv[1:])
def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()
def record(path):
    entry = {'path': str(path), 'exists': path.is_file()}
    if entry['exists']:
        entry.update(bytes=path.stat().st_size, sha256=digest(path))
    return entry

destination = out / 'field_evidence'
destination.mkdir()
field = Path('/home/wangguangjie/djs/package-cache/FIELD-20261016')
names = ('pip_check.txt', 'pip_freeze.txt', 'import_check.json', 'tests.log',
         'legacy_tests.log', 'cli_help.txt', 'stages_ipv4.tsv', 'stages_deps.tsv', 'stages_tuna.tsv')
records = []
for name in names:
    path = field / name
    entry = record(path)
    if entry['exists'] and entry['bytes'] <= 256 * 1024:
        shutil.copyfile(path, destination / name)
        entry['copy'] = str(destination / name)
    records.append(entry)
# Preserve source/log identity without copying potentially verbose transport logs.
records.extend(record(path) for path in (
    field / 'field.log', Path('/home/wangguangjie/djs/field_env_20261016.sh')))
custom = sys.modules.get('sitecustomize')
custom_loaded = bool(custom and getattr(custom, '__file__', None))
custom_path = (Path(custom.__file__) if custom_loaded else
               Path(sysconfig.get_path('purelib')) / 'sitecustomize.py').resolve()
custom_record = record(custom_path)
custom_record['loaded'] = custom_loaded
if custom_record['exists']:
    assert custom_path.is_relative_to(Path(sys.prefix).resolve()), 'sitecustomize is outside FIELD'
    if custom_path.suffix == '.py' and custom_record['bytes'] <= 64 * 1024:
        shutil.copyfile(custom_path, destination / 'sitecustomize.py')
        custom_record['copy'] = str(destination / 'sitecustomize.py')
payload = {'field_artifacts': records, 'sitecustomize': custom_record}
(out / 'field_evidence.json').write_text(json.dumps(payload, indent=2) + '\n')
from tools.setup.verification_contract import source_manifest
(out / 'source_files.json').write_text(json.dumps(source_manifest(root), indent=2) + '\n')
EVIDENCE
task_done

task_run_logged pip_check "$task_out/pip_check.txt" "$task_python" -m pip --disable-pip-version-check check
task_run_logged pip_freeze "$task_out/pip_freeze.txt" "$task_python" -m pip --disable-pip-version-check freeze --all
task_begin import_check
"$task_python" - "$task_repo" <<'CHECK' > "$task_out/import_check.json"
import json, platform, sys
from pathlib import Path
import numpy, torch, association
from association.runtime import code_fingerprint
root = Path(sys.argv[1]).resolve()
assert platform.python_version() == '3.11.7'
assert numpy.__version__ == '1.26.4', numpy.__version__
assert torch.__version__ == '2.5.1+cu121', torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
assert torch.cuda.device_count() == 0, 'CPU verification must hide GPUs'
assert Path(association.__file__).resolve() == root / 'association' / '__init__.py', association.__file__
for package in (numpy, torch):
    assert Path(package.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), package.__file__
print(json.dumps({'python': platform.python_version(), 'python_executable': str(Path(sys.executable).absolute()),
                  'prefix': str(Path(sys.prefix).resolve()), 'base_prefix': str(Path(sys.base_prefix).resolve()), 'numpy': numpy.__version__,
                  'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                  'visible_gpu_count': torch.cuda.device_count(), 'code_sha256': code_fingerprint(),
                  'package_sources': {p.__name__: str(Path(p.__file__).resolve()) for p in (numpy, torch, association)}}, indent=2))
CHECK
task_done
task_begin environment_snapshot
"$task_python" - "$task_repo" <<'CHECK' > "$task_out/environment.json"
import json, sys
from pathlib import Path
from tools.setup.verification_contract import capture_environment
print(json.dumps(capture_environment(Path(sys.argv[1])), indent=2))
CHECK
task_done
task_run_logged tests "$task_out/tests.log" "$task_python" -m unittest discover -s tests
task_run_logged legacy_tests "$task_out/legacy_tests.log" "$task_python" -m unittest discover -s tools/pilot -p test_audit_mtad_level2.py
task_run_logged cli_help "$task_out/cli_help.txt" "$task_python" -m association.cli --help
task_begin acceptance
"$task_python" - "$task_out" <<'CHECK'
import hashlib, json, sys
from pathlib import Path
from tools.setup.verification_contract import capture_environment, source_manifest
out = Path(sys.argv[1])
root = Path.cwd().resolve()
if json.loads((out / 'source_files.json').read_text()) != source_manifest(root):
    raise RuntimeError('Source/config changed during CPU verification')
if json.loads((out / 'environment.json').read_text()) != capture_environment(root):
    raise RuntimeError('Environment fields changed during CPU verification')
def evidence_hash(name):
    return hashlib.sha256((out / name).read_bytes()).hexdigest()
checks = ('pip_check', 'pip_freeze', 'tests', 'legacy_tests', 'cli_help')
assert all((out / (name + '_exit_code.txt')).read_text().strip() == '0' for name in checks)
report = {'status': 'COMPLETED', 'kind': 'existing_FIELD_environment_verification',
          'actual_commit': (out / 'actual_commit.txt').read_text().strip(),
          'environment_snapshot': 'environment.json',
          'environment_rechecked': True,
          'source_files_sha256': evidence_hash('source_files.json'),
          'environment_sha256': evidence_hash('environment.json'),
          'checks': {name: 0 for name in checks},
          'import_check': json.loads((out / 'import_check.json').read_text()),
          'cuda_smoke': 'not_run_by_this_script'}
(out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
CHECK
task_done
task_complete
