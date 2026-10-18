"""Summarize the actual inspect-only CLI output; does not reopen the ZIP."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
source = ROOT / 'runs/LOCAL-NATIVE-INSPECT-20261018-012-attempt-1/preparation.json'
raw = json.loads(source.read_text())
data = raw['dataset']
manifest = json.loads((ROOT / 'HandOver/results/EVIDENCE-20261018-012/prepared007/manifest.json').read_text())
assert raw['status'] == 'INSPECTED' and not raw['model_constructed'] and not raw['gpu_training_started']
assert not data['test_opened'] and not data['resampled_with_legacy_scene64']
assert data['training_scene_ids'] == manifest['splits']['T_core']
assert data['validation_scene_ids'] == manifest['splits']['V_select']
allowed = set(data['training_scene_ids'] + data['validation_scene_ids'])
for members in (data['observation_members_read'], data['reference_members_read']):
    assert {row['scene_id'] for row in members} == allowed
    assert not {row['scene_id'] for row in members} & set(manifest['splits']['T_test'])
counts = {}
for split in ('T_core', 'V_select'):
    rows = [row for row in data['scenes'] if row['split'] == split]
    assert [row['scene_id'] for row in rows] == manifest['splits'][split]
    counts[split] = dict(scene_count=len(rows), **{
        key: sum(row[key] for row in rows) for key in ('node_count', 'native_point_count',
        'candidate_count', 'positive_count', 'target_count', 'target_without_input_count',
        'target_without_reference_count')}, reference_valid_points_per_feature=[
        sum(row['reference_valid_points_per_feature'][i] for row in rows) for i in range(4)])
assert (counts['T_core']['node_count'], counts['V_select']['node_count']) == (54792, 1635)
assert (counts['T_core']['candidate_count'], counts['V_select']['candidate_count']) == (18249384, 29008)
result = dict(source=str(source.relative_to(ROOT)), source_bytes=source.stat().st_size,
    cli_exit_code=0, status=raw['status'], elapsed_seconds_before_result_io=raw['elapsed_seconds'],
    configuration=raw['configuration'], cpu_threads=raw['cpu_threads'], counts=counts,
    observation_members_read=len(data['observation_members_read']),
    reference_members_read=len(data['reference_members_read']), test_opened=False,
    model_constructed=False, gpu_training_started=False,
    raw_report_policy='122 MB report retained in local ignored runs; no hashes or raw test data')
Path(__file__).with_name('native_inspect_summary.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(status=result['status'], counts=counts), ensure_ascii=False))
