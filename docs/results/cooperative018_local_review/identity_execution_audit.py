"""Read-only 018 identity/execution audit, independent of production metric helpers."""
from collections import Counter
from pathlib import Path
import gzip
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'HandOver/results/COOPERATIVE-DEVELOPMENT-20261018-018'
FILES = SOURCE / 'files'
OUT = Path(__file__).with_suffix('.json')
ROUTES = ('plain', 'identity', 'utility', 'learned')

def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)

def metrics(labels, scores, threshold):
    y, s = np.asarray(labels, bool), np.asarray(scores, float)
    assert y.shape == s.shape and y.ndim == 1 and np.isfinite(s).all()
    pred = s >= threshold
    tp, fp, fn, tn = [int(v.sum()) for v in (pred & y, pred & ~y, ~pred & y, ~pred & ~y)]
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
    order = np.argsort(s, kind='stable')
    _, starts, counts = np.unique(s[order], return_index=True, return_counts=True)
    ranks = np.repeat(starts+(counts+1)/2, counts)
    positives = int(y.sum()); negatives = len(y)-positives
    auc = float((ranks[y[order]].sum()-positives*(positives+1)/2)/(positives*negatives)) if positives and negatives else None
    order = np.argsort(-s, kind='stable')
    ends = np.r_[np.flatnonzero(s[order][1:] != s[order][:-1]), len(s)-1]
    cumulative_tp = y[order].cumsum()[ends]
    ap = float(np.sum(np.diff(np.r_[0, cumulative_tp])*cumulative_tp/(ends+1))/positives) if positives else None
    return dict(n=len(y), positive_count=positives, prior=positives/len(y), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1, roc_auc=auc, average_precision=ap)

def finite_tree(obj):
    if isinstance(obj, dict): return all(finite_tree(v) for v in obj.values())
    if isinstance(obj, list): return all(finite_tree(v) for v in obj)
    return math.isfinite(obj) if isinstance(obj, (int, float)) else True

run = read(FILES/'run.json.gz'); history = read(FILES/'history.json')
progress = read(FILES/'progress.json'); metadata = read(FILES/'checkpoint_scale_metadata.json')
actual = read(SOURCE/'actual_config.json'); approved = read(ROOT/'configs/cooperative-development-018.json')
assert actual == approved
assert read(FILES/'execution_extract.json')['configuration'] == {k:actual[k] for k in ('model','training')}
assert run['training'] == actual['training'] and run['model_config'] == actual['model']
assert run['status'] == progress['status'] == 'COMPLETED'
assert run['completed_updates'] == progress['completed_updates'] == len(history) == 600
assert set(run['completed_routes']) == set(ROUTES)
assert all(p['status'] == 'COMPLETED' for p in run['phases'].values())
assert finite_tree(history)
phase_rows = {}
for row in history:
    key = row['stage'] if row['stage'] != 'adapt' else 'adapt:'+row['routing']
    phase_rows.setdefault(key, []).append(row)
assert set(phase_rows) == {'teacher', 'utility'} | {'adapt:'+r for r in ROUTES}
for key, rows in phase_rows.items():
    assert [row['update'] for row in rows] == list(range(1, 101))
    assert run['phases'][key]['completed_updates'] == 100

checkpoints = {}
scale = run['support_time_scaling']['scale_tau']
assert math.isfinite(scale) and scale >= 1
assert run['support_time_scaling']['fit_splits'] == ['T_core']
for name, info in metadata.items():
    if name == 'consistency_check': continue
    assert info['model_config.support_time_policy'] == 'train_absmax_v1'
    assert info['support_time_scale_tau'] == info['model_state[support_time_scale]'] == scale
    assert info['dtype'] == 'torch.float32' and info['numel'] == 1 and info['completed_updates'] == 100
    checkpoints[name] = dict(stage=info['stage'], route=info['route'], updates=info['completed_updates'], scale=scale)
assert len(checkpoints) == 6

training, validation = run['scene_contract']
populations = {}
for split, scenes in [('T_core', training), ('V_select', validation)]:
    assert all(c['split'] == split for c in scenes)
    assert len(set(c['scene_id'] for c in scenes)) == len(scenes)
    populations[split] = dict(scenes=len(scenes), tracks=sum(len(c['node_ids']) for c in scenes),
                             pairs=sum(c['pair_count'] for c in scenes), positive=sum(c['positive_count'] for c in scenes),
                             targets=sum(len(c['target_ids']) for c in scenes))
assert populations['V_select'] == dict(scenes=64, tracks=1635, pairs=29008, positive=3297, targets=377)
assert populations['T_core']['scenes'] == 512 and populations['T_core']['tracks'] == 54792

archives = {r: np.load(FILES/f'{r}-scores.npz', allow_pickle=False) for r in ROUTES}
base = archives['plain']
array_equalities = {}
for route in ROUTES:
    z = archives[route]
    assert set(z.files) == set(base.files)
    for key in z.files:
        assert z[key].dtype == base[key].dtype and np.array_equal(z[key], base[key])
    array_equalities[route] = len(z.files)
threshold = run['threshold']
scene_rows = []; collected = {k: ([], []) for k in ('all', 'itsa', 'mstsa')}
for i, c in enumerate(validation):
    pairs, scores = base[f'pairs_{i}'], base[f'scores_{i}']
    assert pairs.shape == (c['pair_count'], 2) and len(np.unique(pairs, axis=0)) == len(pairs)
    assert (pairs[:, 0] < pairs[:, 1]).all() and pairs.min() >= 0 and pairs.max() < len(c['node_ids'])
    identities = np.array(c['identities']); sources = np.array(c['sources'])
    labels = identities[pairs[:, 0]] == identities[pairs[:, 1]]
    assert int(labels.sum()) == c['positive_count']
    same_source = sources[pairs[:, 0]] == sources[pairs[:, 1]]
    for kind, mask in [('all', np.ones(len(labels), bool)), ('itsa', same_source), ('mstsa', ~same_source)]:
        collected[kind][0].append(labels[mask]); collected[kind][1].append(scores[mask])
    scene_rows.append(dict(scene_id=c['scene_id'], **metrics(labels, scores, threshold)))
pooled = {k:metrics(np.concatenate(y), np.concatenate(s), threshold) for k,(y,s) in collected.items()}
all_scores = np.concatenate(collected['all'][1])
assert len(np.unique(all_scores))+1 == run['identity_selection']['candidate_count']
macro_f1 = float(np.mean([s['f1'] for s in scene_rows]))
assert abs(macro_f1-run['identity_selection']['macro_f1']) < 1e-14

predictions = {}; evaluations = {}; base_groups = None
for route in ROUTES:
    ev = read(FILES/f'{route}-evaluation.json.gz'); evaluations[route] = ev
    assert ev['threshold'] == threshold and ev['selection'] is None and ev['test_opened'] is False
    assert ev['identity_score_source'] == 'fixed_records' and not ev['use_utility_identity']
    assert len(ev['per_scene']) == 64
    for key in ('tp','fp','fn','tn'):
        assert ev['association'][key] == pooled['all'][key]
    for evkey, key in [('micro_precision','precision'),('micro_recall','recall'),('micro_f1','f1')]:
        assert abs(ev['association'][evkey]-pooled['all'][key]) < 1e-14
    assert abs(ev['association']['macro_f1']-macro_f1) < 1e-14
    for original, recomputed in zip(ev['association']['per_scene'], scene_rows):
        assert original['scene_id'] == recomputed['scene_id']
        for key in ('tp','fp','fn','tn','f1','precision','recall'): assert abs(original[key]-recomputed[key]) < 1e-14
    groups = [s['predicted_groups'] for s in ev['per_scene']]
    if base_groups is None: base_groups = groups
    assert groups == base_groups
    folder = FILES/f'{route}-predictions-index'; idx = read(folder/'index.json')
    assert idx['test_opened'] is False and len(idx['scenes']) == 64
    for i, item in enumerate(idx['scenes']):
        c = validation[i]; info = read(folder/item['metadata'])
        assert item['scene_index'] == info['scene_index'] == i
        assert item['scene_id'] == info['scene_id'] == c['scene_id']
        assert info['test_opened'] is False and info['split'] == 'V_select'
        assert info['input_adapter'] == c['input_adapter'] == 'native_time_v1'
        for method, record in info['methods'].items():
            assert record['target_count'] == len(c['target_ids'])
        assert info['methods']['predicted']['group_count'] == len(groups[i])
        assert info['methods']['oracle']['group_count'] == len(c['target_ids'])
    predictions[route] = dict(index_scenes=64, local_metadata_files=64,
        labelled_targets=377, predicted_groups=sum(map(len, groups)),
        anchor_matched_targets=sum(s['methods']['predicted']['anchor_matched_target_count'] for s in ev['per_scene']),
        duplicate_prediction_groups=sum(s['methods']['predicted']['duplicate_prediction_group_count'] for s in ev['per_scene']),
        local_physical_npz_files=len(list(folder.glob('*.npz'))))
assert run['test_opened'] is False
assert read(FILES/'execution_extract.json')['preparation_scalars']['test_opened'] is False
assert int((SOURCE/'exit_code.txt').read_text()) == 0
outer_seconds = float((SOURCE/'elapsed_seconds.txt').read_text())
assert 0 < outer_seconds <= 7200

result = dict(scope='CPU read-only original arrays/JSON; no model, inference, threshold selection or hash gate',
    status='PASS', config_matches_approved=True, outer_seconds=outer_seconds,
    progress_test_opened=progress.get('test_opened', 'FIELD_ABSENT'),
    outer_gpu_hours=outer_seconds/3600, run_seconds=run['elapsed_seconds'],
    phase_updates={k:len(v) for k,v in phase_rows.items()}, all_history_numbers_finite=True,
    checkpoint_metadata=checkpoints, checkpoint_limit='Server-extracted metadata checked; actual weight files remain on server, not locally loaded',
    support_time_scale=run['support_time_scaling'], populations=populations,
    score_array_equalities=array_equalities, threshold=threshold,
    score_range=[float(all_scores.min()),float(all_scores.max())], unique_scores=int(len(np.unique(all_scores))),
    threshold_candidate_count=run['identity_selection']['candidate_count'],
    pooled=pooled, macro_f1=macro_f1,
    scene_macro_auc=float(np.mean([s['roc_auc'] for s in scene_rows if s['roc_auc'] is not None])),
    scene_macro_ap=float(np.mean([s['average_precision'] for s in scene_rows if s['average_precision'] is not None])),
    scene_macro_prior=float(np.mean([s['prior'] for s in scene_rows])),
    scene_auc_above_half=sum(s['roc_auc']>.5 for s in scene_rows if s['roc_auc'] is not None),
    predictions=predictions, single_group_scenes=sum(len(x)==1 for x in base_groups),
    group_diagnostics={k:sum(s['methods']['predicted']['group_diagnostics'][k] for s in evaluations['plain']['per_scene'])
        for k in evaluations['plain']['per_scene'][0]['methods']['predicted']['group_diagnostics']
        if isinstance(evaluations['plain']['per_scene'][0]['methods']['predicted']['group_diagnostics'][k], int)},
    per_scene_identity=scene_rows,
    physical_array_limit='256 complete physical prediction NPZ files remain on server; local verification covers indices/metadata/evaluation populations, not their pointwise contents')
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','outer_seconds','phase_updates','pooled','macro_f1','scene_macro_auc','scene_macro_ap','scene_macro_prior','scene_auc_above_half')},indent=2))
