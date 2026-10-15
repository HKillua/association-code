#!/usr/bin/env python3
"""Read-only environment inventory for the MTAD pilot; writes one JSON report.

Run with the Python environment intended for training. This script does not
install packages, download data, train a model, or inspect credentials.
"""
import argparse
import csv
import datetime
import io
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys


def run(command, timeout=20):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()[:2500],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"exit_code": None, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=".", help="Existing directory intended for pilot work")
    parser.add_argument("--output", help="New JSON output file; existing files are never overwritten")
    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        parser.error("--workspace must be an existing directory")
    now = datetime.datetime.now(datetime.timezone.utc)
    output = Path(args.output or ("server_check_" + now.strftime("%Y%m%dT%H%M%SZ") + ".json")).expanduser()
    if output.exists():
        parser.error("Output exists; choose a new --output filename")
    disk = shutil.disk_usage(workspace)
    fields = ["index", "name", "memory.total", "memory.used", "utilization.gpu", "driver_version"]
    gpu = run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
    if gpu.get("exit_code") == 0:
        gpu["devices"] = [dict(zip(fields, [v.strip() for v in row])) for row in csv.reader(io.StringIO(gpu["stdout"]))]
        gpu["units"] = {"memory.total": "MiB", "memory.used": "MiB", "utilization.gpu": "%"}
    torch_probe = run([sys.executable, "-c", """
import json
import torch
r = {'torch_version': str(torch.__version__), 'build_cuda': torch.version.cuda,
     'cuda_available': torch.cuda.is_available(), 'visible_device_count': torch.cuda.device_count()}
if r['cuda_available']:
    r['visible_devices'] = [{'index': i, 'name': torch.cuda.get_device_name(i),
        'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())]
print(json.dumps(r))
"""], timeout=30)
    if torch_probe.get("exit_code") == 0:
        try:
            torch_probe["details"] = json.loads(torch_probe["stdout"])
        except json.JSONDecodeError:
            torch_probe["parse_note"] = "See stdout; a dependency may have printed additional messages"
    report = {
        "purpose": "MTAD seven-day pilot environment inventory; no training or benchmark performed",
        "checked_at_utc": now.isoformat(),
        "os": platform.system(),
        "machine_architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "workspace": str(workspace),
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "nvidia_smi": gpu,
        "pytorch": torch_probe,
        "notes": [
            "nvidia-smi may list more devices than this Python process is allowed to use.",
            "Reported GPU memory/use is only an instantaneous snapshot, not a reservation.",
            "A missing torch installation is recorded; this script never changes the environment.",
        ],
    }
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("Report saved to:", output.resolve())
    print("Workspace free disk: %.1f GiB" % (disk.free / 1024**3))
    print("nvidia-smi exit code:", gpu.get("exit_code"))
    print("PyTorch probe exit code:", torch_probe.get("exit_code"))


if __name__ == "__main__":
    main()
