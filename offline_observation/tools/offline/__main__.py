"""Run offline evaluation, training and historical pipeline timing."""
from __future__ import annotations

import argparse
from importlib import import_module
import sys


COMMANDS = {
    'system-evaluate': 'tools.offline.system_evaluate',
    'tmtaff-evaluate': 'tools.experiments.tmtaff_complete_evaluate',
    'peer-train': 'tools.offline.peer_fit',
    'pipeline-timing': 'tools.offline.pipeline_timing',
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog='python -m tools.offline', description=__doc__,
        epilog='Pass COMMAND --help for its original arguments. '
               'These are project protocols, not verified author reproductions.')
    parser.add_argument('command', choices=COMMANDS,
                        help='system-evaluate: frozen Level1 V evaluation; '
                             'tmtaff-evaluate: full-chain TMTAFF snapshot evaluation; '
                             'peer-train: fixed-budget PTFM/DeepAF training and V evaluation; '
                             'pipeline-timing: reproduce historical measurement-prefix timing')
    # Parse only the command so subcommand flags, including --help, retain
    # their existing meaning and validation in the implementation's parser.
    command = parser.parse_args(args[:1]).command
    return import_module(COMMANDS[command]).main(
        args[1:], prog=f'{parser.prog} {command}')


if __name__ == '__main__':
    raise SystemExit(main())
