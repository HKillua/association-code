"""Compatibility entry for the standalone online replay command."""
from online_observation.causal_replay import main


if __name__ == '__main__':
    raise SystemExit(main())
