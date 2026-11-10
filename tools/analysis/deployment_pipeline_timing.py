"""Compatibility entry for :mod:`tools.online.pipeline_timing`."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.online import pipeline_timing as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    # Keep legacy imports, globals and monkeypatches on the same module object.
    sys.modules[__name__] = _implementation
