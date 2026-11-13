"""Run the standalone replay under its historical command name."""

import argparse
import sys

from online_observation.causal_replay import main as replay_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m tools.online", description=__doc__)
    parser.add_argument("command", choices=("causal-replay",))
    command = parser.parse_args(args[:1]).command
    return replay_main(args[1:], prog=f"{parser.prog} {command}")


if __name__ == "__main__":
    raise SystemExit(main())
