"""Dispatch the historical offline pipeline timing command."""
from __future__ import annotations

import argparse
from importlib import import_module
import sys


COMMANDS = {
    'pipeline-timing': 'tools.online.pipeline_timing',
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog='python -m tools.online', description=__doc__,
        epilog='Pass COMMAND --help for its original arguments. '
               'Prefix replay does not certify sustained 1 Hz operation.')
    parser.add_argument('command', choices=COMMANDS,
                        help='pipeline-timing: bounded prefix timing')
    command = parser.parse_args(argv[:1]).command
    return import_module(COMMANDS[command]).main(
        argv[1:], prog=f'{parser.prog} {command}')


if __name__ == '__main__':
    raise SystemExit(main())
