"""Run online observation commands from the source checkout."""
from __future__ import annotations

import argparse
import sys

from . import causal_replay


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='python -m online_observation', description=__doc__)
    parser.add_argument('command', choices=('causal-replay',), help='1 Hz measurement replay control')
    parser.parse_args(args[:1])
    return causal_replay.main(args[1:], prog=f'{parser.prog} causal-replay')


if __name__ == '__main__':
    raise SystemExit(main())
