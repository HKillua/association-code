"""Compatibility entry point for :mod:`tools.offline.system_evaluate`."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.offline import system_evaluate as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())

# Preserve imports and patch targets as the same module object, including
# private helpers used by existing evaluation and contract checks.
sys.modules[__name__] = _implementation
