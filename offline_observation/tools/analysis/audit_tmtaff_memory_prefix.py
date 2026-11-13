"""Recount C96 saved per-query responses using scalar aggregation."""

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np


def percentile(values, fraction):
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def recount(root):
    started = time.monotonic()
    summary = json.loads((root / 'summary.json').read_text())
    with np.load(root / 'private/masks.npz', allow_pickle=False) as saved:
        valid, support = saved['valid'], saved['support']
    assert valid.shape == (16, 64, 4) and support.shape == (16, 64)
    periods = {'all': range(64), 'bos': range(1), 'early': range(1, 32),
               'late': range(32, 64), 'last': range(63, 64)}
    checks = 0
    max_deltas = {'float32': 0., 'float64': 0.}
    for name, snapshot in summary['snapshots'].items():
        with np.load(root / 'private' / f'{name}.npz', allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        for target, strata in snapshot['response'].items():
            for period, subsets in strata.items():
                for subset, metrics in subsets.items():
                    selected = []
                    for i, scene in enumerate(summary['scenes']):
                        if target not in (scene['scene'], 'distinct' if scene['distinct_targets'] else 'same'):
                            continue
                        for pair in (2 * i, 2 * i + 1):
                            for q in periods[period]:
                                both = (pair, pair ^ 1)
                                if subset != 'blind' and not all(valid[k, q, :2].all() for k in both):
                                    continue
                                if subset == 'valid_dual_both' and not all(support[k, q] == 2 for k in both):
                                    continue
                                selected.append((pair, q))
                    for key, stored in metrics.items():
                        data = arrays[key]
                        values = sorted(float(data[a, b]) for a, b in selected)
                        assert len(values) == stored['count']
                        expected = [statistics.fmean(values), math.sqrt(statistics.fmean(x*x for x in values)),
                                    percentile(values, .95), percentile(values, .99), values[-1]] if values else [None]*5
                        dtype = str(data.dtype)
                        for metric, actual in zip(('mae', 'rmse', 'p95', 'p99', 'max'), expected, strict=True):
                            reported = stored[metric]
                            if actual is None:
                                assert reported is None
                                continue
                            max_deltas[dtype] = max(max_deltas[dtype], abs(actual - reported))
                            # The production summary reduces latitude in float32.
                            tolerance = 1e-6 if dtype == 'float32' else 1e-10
                            assert math.isclose(actual, reported, rel_tol=tolerance, abs_tol=1e-10), (
                                name, target, period, subset, key, metric, actual, reported)
                        checks += 6
        for key in ('prefix_latent_rms', 'prefix_standardized_rms', 'prefix_physical_m'):
            assert np.count_nonzero(arrays[key][:, 0]) == 0
        np.testing.assert_array_equal(arrays['both_physical_m'], arrays['donor_physical_m'])
    return dict(status='VERIFIED', statistics=checks, max_absolute_delta_by_storage_dtype=max_deltas,
                model_forwards=0, seconds=time.monotonic() - started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(recount(parser.parse_args().run), allow_nan=False))
