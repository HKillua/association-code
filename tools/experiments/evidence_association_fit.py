"""Train one bounded native-OLS evidence association arm or a synthetic check.

Native fits use only T_core identity labels and V_select development selection.
Reference targets, learned completion weights and T_test are never requested.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.data.motion_evidence import DATA, build_motion_evidence
from association.models.evidence_association import VARIANTS
from association.training.evidence_association import SCOPE, fit_evidence_association
from association.utils import runtime


def synthetic_scenes():
    """Tiny disjoint train/V identities, including positive nonoverlap pairs."""
    records = []
    for name, split, offset in (('train-a', 'T_core', 0.), ('train-b', 'T_core', .002),
                                ('validation', 'V_select', .001)):
        n, maximum = 6, 8
        values, times = np.zeros((n, maximum, 4)), np.zeros((n, maximum))
        observed = np.zeros((n, maximum), bool)
        query = np.linspace(0., 600., DATA['length'])
        for node in range(n):
            t = np.linspace(0., 240., maximum) if node % 2 == 0 else np.linspace(360., 600., maximum)
            times[node], observed[node] = t, True
            values[node, :, 0] = 30.+offset+(node//2)*.1+t*.00001
            values[node, :, 1] = 120.+(node//2)*.03+t*.00002
            values[node, :, 2] = 5.+(node//2)+t*.0001
            values[node, :, 3] = (359.+(node//2)*30.+t*.002) % 360.
        sources = np.asarray(['9001', '9002']*3)
        identities = np.asarray([f'{name}-identity-{node//2}' for node in range(n)])
        pairs = np.column_stack(np.triu_indices(n, 1))
        records.append(dict(scene_id=name, component_id=name+'-component', split=split,
            node_ids=[f'{name}-{node}' for node in range(n)], sources=sources, identities=identities,
            pairs=pairs, labels=(identities[pairs[:, 0]] == identities[pairs[:, 1]]).astype(np.float32),
            motion=build_motion_evidence(values, times, observed, query, sources)))
    return records[:2], records[2:]


def synthetic_config(variant):
    return dict(scope=SCOPE, variant=variant, seed=20, data=dict(DATA),
        training=dict(updates=3, batch_size=8, learning_rate=1e-3,
                      gradient_clip=1., validate_every=1, inference_batch_size=16,
                      advisory_hours=1.))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'fit'))
    parser.add_argument('--device', required=True, help='cpu or explicit cuda:0; no automatic fallback')
    parser.add_argument('--output', required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--variant', choices=VARIANTS, help='synthetic check only; fit reads config')
    parser.add_argument('--archive', nargs='+', help='one ZIP or its ordered continuous parts')
    parser.add_argument('--prefix-reference', type=Path,
                        help='060 only: completed056 association directory; compare final1000, never resume')
    for name in ('manifest', 'config'):
        parser.add_argument('--'+name)
    args = parser.parse_args(argv)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError('SIGTERM; fixed association fit interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.cpu_threads < 1:
            raise ValueError('cpu-threads must be positive')
        torch.set_num_threads(args.cpu_threads)
        if args.mode == 'fit':
            if not all((args.archive, args.manifest, args.config)) or args.variant is not None:
                raise ValueError('fit needs archive, manifest, config; variant comes from config')
            kwargs = dict(archive_path=args.archive, manifest=runtime.read_json(args.manifest),
                          prefix_reference=args.prefix_reference)
            config = runtime.read_json(args.config)
        else:
            if any((args.archive, args.manifest, args.config, args.prefix_reference)):
                raise ValueError('synthetic check does not accept native archive/manifest/config')
            train, validation = synthetic_scenes()
            kwargs = dict(train_scenes=train, validation_scenes=validation, synthetic=True)
            config = synthetic_config(args.variant or 'full_evidence')
        command = [sys.executable, str(Path(__file__).resolve()),
                   *(sys.argv[1:] if argv is None else argv)]
        result = fit_evidence_association(config, args.output, args.device, command=command, **kwargs)
        print(f"{result['status']}: {Path(args.output).resolve()}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
