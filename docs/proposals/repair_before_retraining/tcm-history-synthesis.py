"""Read-only synthesis of already-returned TCM logs and fragment statistics."""
from pathlib import Path
import gzip
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
    '020_history': 'HandOver/results/STAGEWISE-COMPLETION-20261019-028/020/history.json.gz',
    '033_history': 'HandOver/results/BASELINE-TCM-COVERAGE-20261020-033/history.json.gz',
    'common_scene_contract': 'HandOver/results/STAGEWISE-COMPLETION-20261019-028/021/run.json.gz',
    '033_evaluation': 'HandOver/results/BASELINE-TCM-COVERAGE-20261020-033/completion-evaluation.json.gz',
    'loss_implementation': 'association/models/timed.py',
}
BUCKETS = [(2, 4), (5, 15), (16, 31), (32, 63), (64, 128)]

def read(key):
    with gzip.open(ROOT / SOURCES[key], 'rt') as handle:
        return json.load(handle)

def window(rows):
    losses = sorted(row['loss'] for row in rows)
    gradients = [row['gradient_norm'] for row in rows]
    top_count = max(1, round(len(losses) * .01))
    return dict(updates=len(rows), loss_mean=statistics.mean(losses),
                loss_median=statistics.median(losses), loss_max=max(losses),
                gradient_mean=statistics.mean(gradients),
                gradient_median=statistics.median(gradients), gradient_max=max(gradients),
                clip1_exceeded=sum(value > 1 for value in gradients),
                top_one_percent_batches=top_count,
                top_one_percent_loss_share=sum(losses[-top_count:]) / sum(losses),
                descriptive_trimmed_mean=statistics.mean(losses[:-top_count]),
                loss_above_point1=sum(value > .1 for value in losses))

result = dict(sources=SOURCES, scope='Existing returned artifacts only; no inference, training, ZIP scan, or T_test.',
              training={}, population={}, validation_buckets={},
              caveats=[
                  '021 scene_contract is common fixed data metadata, not candidate training results used as baseline evidence.',
                  'Supervision shares count full-library native transitions, not exact cumulative SGD gradient weights; per-batch denominators vary.',
                  'Bucket comparisons are descriptive and confounded by source, scene, observed support and fragment information.',
                  'Better/worse comparisons below are 033 versus state_hold, not 4000 versus 1000 steps.',
                  'Top-tail loss summaries describe batch variation; logs lack per-batch identities and attribute losses.',
                  'Temporary trimmed means diagnose tail influence only; no batch is excluded from training or official metrics.',
              ])
for experiment in ('020', '033'):
    rows = [row for row in read(experiment + '_history') if row['stage'] == 'TCM']
    assert all(math.isfinite(row['loss']) and math.isfinite(row['gradient_norm']) for row in rows)
    result['training'][experiment] = dict(overall=window(rows), windows={})
    for start, stop in [(1, 100), (101, 400), (401, 1000), (1001, 2000), (2001, 3000), (3001, 4000)]:
        part = rows[start - 1:stop]
        if part:
            result['training'][experiment]['windows'][f'{start}-{stop}'] = window(part)

for scenes in read('common_scene_contract')['scene_contract']:
    counts = [count for scene in scenes for count in scene['native_observed_counts']]
    transitions = sum(count - 1 for count in counts)
    buckets = {}
    for lower, upper in BUCKETS:
        selected = [count for count in counts if lower <= count <= upper]
        buckets[f'{lower}-{upper}'] = dict(fragment_count=len(selected), fragment_share=len(selected)/len(counts),
            native_transitions=sum(count - 1 for count in selected),
            native_transition_share=sum(count - 1 for count in selected)/transitions)
    assert sum(row['fragment_count'] for row in buckets.values()) == len(counts)
    assert sum(row['native_transitions'] for row in buckets.values()) == transitions
    result['population'][scenes[0]['split']] = dict(scene_count=len(scenes), fragment_count=len(counts),
        median_native_count=statistics.median(counts), native_transitions=transitions, buckets=buckets)

fragments = read('033_evaluation')['per_fragment']
for lower, upper in BUCKETS:
    rows = [row for row in fragments if lower <= row['native_count'] <= upper]
    item = dict(fragment_count=len(rows), metrics={})
    for metric in ('one_step', 'query', 'outside_query'):
        sums = {method: dict(squared_error_sum=0., feature_elements=0) for method in ('model', 'state_hold')}
        changes = dict(better=0, equal=0, worse=0, not_evaluable=0)
        for row in rows:
            values = {}
            for method in sums:
                if metric == 'one_step':
                    parts = [row['one_step'][method]['all']]
                else:
                    regions = ['all'] if metric == 'query' else ['before_support', 'after_support']
                    parts = [row['query'][method]['regions'][region] for region in regions]
                sse = sum(part['squared_error_sum'] for part in parts)
                elements = sum(part['feature_elements'] for part in parts)
                sums[method]['squared_error_sum'] += sse
                sums[method]['feature_elements'] += elements
                values[method] = sse/elements if elements else None
            if any(value is None for value in values.values()):
                changes['not_evaluable'] += 1
            elif values['model'] < values['state_hold'] - 1e-14:
                changes['better'] += 1
            elif values['model'] > values['state_hold'] + 1e-14:
                changes['worse'] += 1
            else:
                changes['equal'] += 1
        for summary in sums.values():
            summary['feature_mse'] = summary['squared_error_sum']/summary['feature_elements'] if summary['feature_elements'] else None
        assert sums['model']['feature_elements'] == sums['state_hold']['feature_elements']
        assert sum(changes.values()) == len(rows)
        model, hold = sums['model']['feature_mse'], sums['state_hold']['feature_mse']
        item['metrics'][metric] = dict(**sums, relative_reduction=1-model/hold if hold else None,
                                      fragment_changes=changes, equality_tolerance=1e-14)
    result['validation_buckets'][f'{lower}-{upper}'] = item
assert sum(row['fragment_count'] for row in result['validation_buckets'].values()) == len(fragments)
output = Path(__file__).with_suffix('.json')
output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(str(output))
