"""Record completed wheel files without treating caches as installed packages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def inventory(directory: Path) -> dict:
    files = []
    for path in sorted(directory.glob("*.whl")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return {"directory": str(directory), "exists": directory.is_dir(), "files": files}


if __name__ == "__main__":
    print(json.dumps(inventory(Path(sys.argv[1])), indent=2))
