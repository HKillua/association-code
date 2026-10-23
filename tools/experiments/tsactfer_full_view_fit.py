"""082: paired full-native training-view intervention; same074D sampler and final budget."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.experiments import tsactfer_level1_fit as shared


def run(args):
    return shared.run(args, profile='082')


def main(argv=None):
    return shared.main(argv, profile='082')


if __name__ == '__main__':
    raise SystemExit(main())
