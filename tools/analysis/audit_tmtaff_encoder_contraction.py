"""Independently aggregate C99 trace and rederive responses from frozen arrays."""

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np

from tools.analysis.audit_tmtaff_memory_prefix import percentile


def read_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def rms_difference(first, second):
    a, b = first.astype(np.float64), second.astype(np.float64)
    absolute = np.linalg.norm(a - b, axis=-1) / np.sqrt(a.shape[-1])
    amplitude = np.sqrt((np.linalg.norm(a, axis=-1) ** 2 + np.linalg.norm(b, axis=-1) ** 2)
                        / (2 * a.shape[-1]))
    relative = np.divide(absolute, amplitude, out=np.zeros_like(absolute), where=amplitude != 0)
    return absolute, relative


def distance(first, second):
    a, b = (np.deg2rad(array[..., :2].astype(np.float64)) for array in (first, second))
    delta = a - b
    haversine = (np.sin(delta[..., 0] / 2) ** 2 + np.sin(delta[..., 1] / 2) ** 2
                 * np.cos(a[..., 0]) * np.cos(b[..., 0]))
    return 12742000. * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))


class Checks:
    def __init__(self):
        self.count = 0
        self.max_absolute_delta = 0.

    def scalar(self, actual, expected):
        self.count += 1
        if actual is None:
            assert expected is None
        else:
            delta = abs(actual - expected)
            self.max_absolute_delta = max(self.max_absolute_delta, delta)
            assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7), (actual, expected, delta)

    def stats(self, data, expected):
        values = sorted(map(float, data))
        assert len(values) == expected['count']
        self.count += 1
        metrics = (statistics.fmean(values), math.sqrt(statistics.fmean(x*x for x in values)),
                   percentile(values, .95), percentile(values, .99), values[-1]) if values else (None,) * 5
        for key, value in zip(('mae', 'rmse', 'p95', 'p99', 'max'), metrics, strict=True):
            self.scalar(value, expected[key])


def audit(root):
    summary = json.loads((root / 'summary.json').read_text())
    selection = json.loads((root / 'selection.json').read_text())
    data = read_arrays(root / 'private/population.npz')
    valid = data['valid'][..., :2].all(-1)
    support = data['support'] == 2
    donor = np.arange(len(valid)) ^ 1
    labels = np.repeat([row['scene'] for row in summary['scenes']], 2)
    distinct = np.repeat([row['distinct_targets'] for row in summary['scenes']], 2)
    selected = {'blind': np.ones_like(valid), 'valid_both': valid & valid[donor],
                'valid_dual_both': valid & valid[donor] & support & support[donor]}
    periods = {'all': slice(None), 'bos': slice(0, 1), 'early': slice(1, 32),
               'late': slice(32, None), 'last': slice(63, 64)}
    groups = {'distinct': distinct, 'same': ~distinct, **{label: labels == label for label in labels}}
    checks = Checks()
    for snapshot, row in summary['snapshots'].items():
        trace = read_arrays(root / 'private' / f'{snapshot}-trace.npz')
        for group, recorded in row['trace'].items():
            for name, result in recorded.items():
                checks.stats(trace[name][groups[group]].reshape(-1), result)
        if snapshot == 'original_u32':
            ranked = []
            for candidate in selection['candidates']:
                prefix = f"{candidate['side']}.{candidate['layer']}."
                means = [statistics.fmean(trace[prefix + candidate[key] + '/relative'][distinct].flat)
                         for key in ('entry', 'exit')]
                ratio = means[1] / means[0] if means[0] else None
                checks.scalar(ratio, candidate['ratio'])
                if ratio is not None and ratio < 1:
                    ranked.append((ratio, candidate))
            chosen = min(ranked, key=lambda row: row[0])[1] if ranked else None
            assert chosen == selection['boundary'] == summary['boundary']
        if summary['boundary'] is None:
            continue
        replay = read_arrays(root / 'private' / f'{snapshot}-replay.npz')
        original = replay['native_boundary'].astype(np.float64)
        patched = replay['patched_boundary'].astype(np.float64)
        checks.scalar(float(np.max(np.abs((original + original[donor] - patched - patched[donor]) / 2))),
                      row['midpoint_max_abs'])
        responses = {}
        for arm in ('native', 'patched'):
            for name in ('boundary', 'memory', 'latent', 'standardized'):
                array = replay[f'{arm}_{name}']
                responses[f'{arm}_{name}_pair_relative'] = rms_difference(array, array[donor])[1]
            states = replay[f'{arm}_states']
            responses[f'{arm}_pair_m'] = distance(states, states[donor])
            error = distance(states, data['truth'])
            for scope, mask in (('all', valid), ('both_supported', valid & support)):
                checks.stats(error[mask], row[f'{arm}_metrics'][scope]['position'])
        for name in ('boundary', 'memory', 'latent', 'standardized'):
            absolute, relative = rms_difference(replay[f'patched_{name}'], replay[f'native_{name}'])
            responses[f'{name}_response_rms'] = absolute
            responses[f'{name}_response_relative'] = relative
        responses['physical_response_m'] = distance(replay['patched_states'], replay['native_states'])
        for group, slices in row['replay'].items():
            for period, subsets in slices.items():
                for subset, metrics in subsets.items():
                    mask = (selected[subset] & groups[group][:, None])[:, periods[period]]
                    for key, expected in metrics.items():
                        checks.stats(responses[key][:, periods[period]][mask], expected)
    return dict(status='VERIFIED', checks=checks.count, max_absolute_delta=checks.max_absolute_delta,
                model_forwards=0, selection=summary['boundary'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().run), allow_nan=False, indent=2))
