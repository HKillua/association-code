"""Record the exact checkout, timing and terminal state of one fixed audit."""

import argparse
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from tools.analysis.offline_baseline_restart import write


def execute(audit):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-policy", choices=["advisory"], required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.monotonic()

    def stamp():
        return datetime.now(timezone.utc).isoformat()

    record = dict(
        status="RUNNING",
        started=stamp(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        training_updates=0,
        test_opened=False,
    )
    write(args.output / "run.json", record)
    try:
        audit(args.runs, args.output)
        record.update(status="COMPLETED", exit_code=0)
    except BaseException as exc:
        record.update(
            status="FAILED", exit_code=1, error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        record.update(
            finished=stamp(),
            seconds=time.monotonic() - start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        write(args.output / "run.json", record)
