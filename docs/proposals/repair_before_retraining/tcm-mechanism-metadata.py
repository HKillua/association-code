"""Run from association-code root; metadata/results only, no model or archive read."""
from pathlib import Path
import gzip
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_suffix('.json')

def read(relative):
    with gzip.open(ROOT / relative, 'rt') as handle:
        return json.load(handle)

def quantiles(values):
    return [float(x) for x in np.percentile(values, [0, 25, 50, 75, 95, 100])]

contracts = read('HandOver/results/STAGEWISE-COMPLETION-20261019-028/021/run.json.gz')['scene_contract']
scenes = [scene for group in contracts for scene in group]
evaluation = read('HandOver/results/BASELINE-TCM-COVERAGE-20261020-033/completion-evaluation.json.gz')
lookup = {row['node_id']: row for row in evaluation['per_fragment']}
validation_nodes = {node for scene in scenes if scene['split'] == 'V_select' for node in scene['node_ids']}
assert validation_nodes == set(lookup)
for scene in scenes:
    if scene['split'] == 'V_select':
        for node, count in zip(scene['node_ids'], scene['native_observed_counts']):
            assert lookup[node]['native_count'] == count

result = {}
for split in ['T_core', 'V_select']:
    selected = [scene for scene in scenes if scene['split'] == split]
    means, spans, before_steps, after_steps, horizons, counts = [], [], [], [], [], []
    for scene in selected:
        query = np.asarray(scene['grid'])
        spans.append(query[-1] - query[0])
        for count, (start, stop) in zip(scene['native_observed_counts'], scene['intervals']):
            means.append((stop - start) / (count - 1))
            counts.append(count)
            before_steps.append(int((query < start).sum()))
            after_steps.append(int((query > stop).sum()))
            horizons.extend(start - query[query < start])
            horizons.extend(query[query > stop] - stop)
    result[split] = {
        'scenes': len(selected), 'fragments': len(counts),
        'scene_span_q0_25_50_75_95_100': quantiles(spans),
        'fragment_mean_native_interval_q0_25_50_75_95_100': quantiles(means),
        'before_rollout_steps_q0_25_50_75_95_100': quantiles(before_steps),
        'after_rollout_steps_q0_25_50_75_95_100': quantiles(after_steps),
        'all_outside_endpoint_horizon_q0_25_50_75_95_100': quantiles(horizons),
        'native_total': sum(counts),
        'n_le4_fragments': sum(count <= 4 for count in counts),
        'n_le4_transition_share': sum(count - 1 for count in counts if count <= 4) / sum(count - 1 for count in counts),
    }

OUT.write_text(json.dumps(result, indent=2) + '\n')
print(OUT)

# Per-fragment mean intervals, not individual-transition quantiles.
# Horizons include queries without reference; no target error is implied.
