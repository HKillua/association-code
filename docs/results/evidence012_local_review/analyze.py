"""Review returned 012 evidence and extend 011 cache comparisons; CPU, no fit.

Reuses existing 011 metric/subset helpers. Original thresholds are fixed; no
checkpoint/model inference, raw ZIP/reference/T_test data reads, or hashes.
"""
from collections import Counter
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('prior_011_analysis', ROOT/'docs/results/compare011_interim/analyze.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
from association.runtime import load_records, validate_record_split
from association.metrics import evaluate_scenes, compare_predictions


def read(path):
    return json.loads(path.read_text())


def inventory(path):
    files = [p for p in path.rglob('*') if p.is_file()]
    return dict(files=len(files), bytes=sum(p.stat().st_size for p in files))


def main():
    started = time.monotonic()
    new = ROOT/'HandOver/results/EVIDENCE-20261018-012'
    old = ROOT/'HandOver/results/COMPARE-20261017-011'
    expected_arms = ('A_prime', 'B', 'B_shuffle', 'EndpointMarginal', 'FixedRule', 'PairMLP', 'ScenePool')
    assert set(p.name for p in (new/'cdf/V_select').iterdir()) == set(expected_arms)
    copy_rc = int((new/'copy_process_exit_code.txt').read_text())
    item_rcs = {k:int(v) for k,v in (line.split('=') for line in (new/'copy.log').read_text().splitlines())}
    assert copy_rc == 0 and item_rcs == dict(cdf_copy_rc=0, manifest_copy_rc=0, history_copy_rc=0)
    timestamps = [(new/f).read_text().strip() for f in ('started_at.txt','copy_ended_at.txt')]
    copy_timestamp_delta = (datetime.fromisoformat(timestamps[1])-datetime.fromisoformat(timestamps[0])).total_seconds()
    # Second-resolution equal timestamps establish recorded delta=0, not a
    # sub-second monotonic duration or the duration of all content verification.
    counts = {name: inventory(new/name) for name in ('cdf','prepared007','reference010')}
    assert sum(v['files'] for v in counts.values()) == 457
    assert sum(v['bytes'] for v in counts.values()) == 3940668

    manifest = read(new/'prepared007/manifest.json')
    splits = manifest['splits']
    assert {k:len(v) for k,v in splits.items()} == dict(T_core=512,T_head=128,V_select=64,T_test=128)
    all_selected = [sid for ids in splits.values() for sid in ids]
    assert len(set(all_selected)) == len(all_selected)
    assert set(all_selected).issubset(manifest['all_scenes'])
    component_sets = {side:{manifest['components'][sid] for sid in ids} for side,ids in splits.items()}
    for side, ids in splits.items():
        assert len(ids) == len(set(ids))
        assert len(component_sets[side]) == manifest['split_counts'][side]['selected_components']
        assert all(manifest['component_splits'][manifest['components'][sid]] == side for sid in ids)
        for other in splits:
            if side != other:
                assert not component_sets[side] & component_sets[other]
    for component, members in manifest['component_members'].items():
        assert all(manifest['components'][sid] == component for sid in members)
    prior_lists = read(ROOT/'docs/results/frozen_clock_audit_20261018/frozen_scene_lists_20261018.json')
    for side, ids in prior_lists['scene_ids'].items():
        assert set(map(str, ids)) == set(splits[side]), side
    a0_summary = read(old/'reference010/summary_metrics.json')
    selection = read(old/'cdf/files/heads/selection.json')
    report = read(old/'cdf/files/validation.json')
    assert a0_summary['training_scene_ids'] == splits['T_core']
    assert selection['training_scene_ids'] == splits['T_head']
    assert selection['validation_scene_ids'] == splits['V_select']

    rows = {'A0':load_records(old/'reference010/V_select'),
            'TSA':load_records(old/'tsa/files/V_select'),
            'A_full':load_records(old/'afull/files/V_select')}
    thresholds = dict(A0=a0_summary['threshold'],
                      TSA=read(old/'tsa/files/summary.json')['threshold'],
                      A_full=read(old/'afull/files/summary_metrics.json')['threshold'])
    arm_inventory = {}
    for name in expected_arms:
        directory = new/'cdf/V_select'/name
        index = read(directory/'index.json')
        assert index['schema'] == 1 and index['binding'] == dict(split='V_select',arm=name,seed=20)
        assert len(index['records']) == 64
        assert {p.name for p in directory.iterdir()} == {'index.json'} | {r['file'] for r in index['records']}
        arm_inventory[name] = inventory(directory)
        assert arm_inventory[name]['files'] == 65
        rows[name] = load_records(directory, expected_binding=dict(split='V_select',arm=name,seed=20))
        thresholds[name] = selection['arms'][name]['threshold']

    for name, records in rows.items():
        validate_record_split(records, manifest, 'V_select')
        assert [r['scene_id'] for r in records] == splits['V_select']
        for a,b in zip(rows['A0'], records):
            for key in ('scene_id','component_id','node_ids'):
                assert a[key] == b[key], (name,key)
            for key in ('pairs','labels','identities','sources','intervals'):
                assert np.array_equal(a[key],b[key]), (name,key)
            n = len(b['sources'])
            canonical = np.column_stack(np.triu_indices(n,1))
            assert np.array_equal(b['pairs'], canonical), (name,b['scene_id'])

    full = {name:evaluate_scenes(records, thresholds[name]) for name,records in rows.items()}
    for name, records in rows.items():
        # Independent NumPy confusion arithmetic, in addition to the reused
        # metric function and comparison against the original summary.
        labels = np.concatenate([r['labels'] for r in records]).astype(bool)
        scores = np.concatenate([r['scores'] for r in records])
        decisions = np.zeros(len(scores),dtype=bool) if thresholds[name] is None else scores>=thresholds[name]
        direct = dict(tp=int((labels & decisions).sum()),fp=int((~labels & decisions).sum()),
                      fn=int((labels & ~decisions).sum()),tn=int((~labels & ~decisions).sum()))
        assert all(full[name][k]==v for k,v in direct.items()), name
    for name in ('A0',)+expected_arms:
        assert full[name] == report['arms'][name]['validation'], name
        if name != 'A0':
            assert report['arms'][name] == selection['arms'][name]
    errors_to_b = {name:compare_predictions(rows[name], rows['B'], thresholds[name], thresholds['B'])
                   for name in report['error_changes']}
    assert errors_to_b == report['error_changes']
    masks = [prior.scope_mask(r) for r in rows['A0']]
    restricted_rows = {name:prior.subset(records,masks) for name,records in rows.items()}
    restricted = {name:evaluate_scenes(records,thresholds[name]) for name,records in restricted_rows.items()}
    original_011 = read(ROOT/'docs/results/compare011_interim/analysis.json')
    for name in ('A0','TSA','A_full'):
        assert prior.compact(full[name]) == original_011['full'][name]
        assert prior.compact(restricted[name]) == original_011['common_paper_scope_original_thresholds'][name]
    changes_a0 = {name:compare_predictions(rows['A0'],records,thresholds['A0'],thresholds[name])
                  for name,records in rows.items() if name!='A0'}
    changes_common = {name:compare_predictions(restricted_rows['A0'],records,thresholds['A0'],thresholds[name])
                      for name,records in restricted_rows.items() if name!='A0'}
    win_tie_loss = {}
    for name in expected_arms:
        diff = np.array([b['f1']-a['f1'] for a,b in zip(full['A0']['per_scene'],full[name]['per_scene'])])
        win_tie_loss[name] = dict(win=int((diff>1e-12).sum()),tie=int((abs(diff)<=1e-12).sum()),loss=int((diff < -1e-12).sum()))

    history = read(new/'reference010/history.json')
    assert len(history) == 220
    assert Counter(r['phase'] for r in history) == dict(TCM=100,TAM_FROZEN=100,JOINT=20)
    selection_rows = [r for r in history if 'selection' in r]
    assert selection_rows == read(old/'reference010/history_with_selection.json')
    history_report = {}
    pair_union = set()
    for phase,count in (('TCM',100),('TAM_FROZEN',100),('JOINT',20)):
        phase_rows = [r for r in history if r['phase']==phase]
        assert [r['update'] for r in phase_rows] == list(range(1,count+1))
        loss_key = 'loss' if phase=='TCM' else 'association_loss'
        losses = np.array([r[loss_key] for r in phase_rows])
        gradients = np.array([r['gradient_norm'] for r in phase_rows])
        assert np.isfinite(losses).all() and np.isfinite(gradients).all()
        stage = dict(updates=count, loss_field=loss_key, first10_loss_mean=float(losses[:10].mean()),
                     last10_loss_mean=float(losses[-10:].mean()),gradient_min=float(gradients.min()),
                     gradient_max=float(gradients.max()),gradient_above_1_count=int((gradients>1).sum()),
                     all_gradients_nonzero=bool((gradients>0).all()),
                     step_seconds=sum(r['seconds'] for r in phase_rows),
                     validation_seconds=sum(r.get('validation_seconds',0) for r in phase_rows))
        if phase=='TCM':
            tracks = {tuple(k) for r in phase_rows for k in r['track_keys']}
            assert all(0<=s<len(splits['T_core']) and n>=0 for s,n in tracks)
            stage.update(track_presentations=sum(r['track_count'] for r in phase_rows),
                         unique_tracks=len(tracks), unique_scenes=len({s for s,n in tracks}))
        else:
            pairs = set()
            for r in phase_rows:
                assert r['pair_count']==64 and r['positive_pairs']==r['negative_pairs']==32
                assert r['scene_ids']==r['sampling']['scene_ids']
                for sid,indices,group in zip(r['scene_ids'],r['pair_indices'],r['sampling']['groups']):
                    assert sid==group['scene_id']==splits['T_core'][group['scene_index']]
                    assert len(indices)==group['pair_count']
                    pairs.update((sid,int(i)) for i in indices)
            pair_union |= pairs
            stage.update(pair_presentations=sum(r['pair_count'] for r in phase_rows),
                         unique_pairs=len(pairs),unique_scenes=len({s for s,i in pairs}))
        history_report[phase] = stage
    assert selection_rows[-1]['selection']['threshold']==a0_summary['threshold']
    assert np.isclose(selection_rows[-1]['selection']['macro_f1'],full['A0']['macro_f1'],rtol=0,atol=1e-12)

    result = dict(
        scope='012 original-copy/manifest/history verification; all CDF edges and common subset evaluated at unchanged original thresholds',
        source_directory=str(new.relative_to(ROOT)),
        copy=dict(outer_exit_code=copy_rc, item_exit_codes=item_rcs, recorded_timestamps=timestamps,
                  recorded_timestamp_delta_seconds=copy_timestamp_delta,
                  timing_limit='equal second-resolution timestamps do not measure total collection/content-check time',
                  payload_inventory=counts, total_payload_files=457,total_payload_bytes=3940668,
                  cdf_arm_inventory=arm_inventory, missing_items=[]),
        manifest=dict(protocol=manifest['protocol'],seed=manifest['seed'],split_counts={k:len(v) for k,v in splits.items()},
                      selected_component_counts={k:len(v) for k,v in component_sets.items()},
                      all_scene_count=len(manifest['all_scenes']),development_scene_count=len(manifest['development_scenes']),
                      selected_scenes_and_components_disjoint=True,
                      prior_frozen_scene_lists_match=True, original_010_and_CDF_train_and_validation_order_match=True,
                      registration_policy=manifest['registration_policy'],
                      content_scope='T_test split metadata only; no T_test measurements/reference opened; no new split',
                      distinction='manifest stores original historical scene64 policy; copying it does not endorse old input registration for new native-time experiments'),
        identity_checks=dict(all_ten_methods_same_scenes_components_nodes_pairs_labels_identities_sources_intervals=True,
                             all_pairs_canonical_and_complete=True, all_cdf_metrics_match_original_dicts=True,
                             independent_numpy_confusion_counts_match_all_ten_methods=True,
                             all_cdf_error_changes_match_every_original_edge_count=True,
                             no_threshold_selection_called=True, no_hash_computed=True),
        thresholds=thresholds,
        full={k:prior.compact(v) for k,v in full.items()},
        by_source={k:{s:prior.compact(v[s]) for s in ('itsa','mstsa')} for k,v in full.items()},
        common_paper_scope_original_thresholds={k:prior.compact(v) for k,v in restricted.items()},
        paper_scope_coverage=prior.paper_scope_coverage(rows['A0']),
        score_rank_metrics={k:prior.rank_metrics(v) for k,v in rows.items()},
        error_changes_baseline_to_B=errors_to_b,error_changes_A0_to_each=changes_a0,
        common_scope_error_changes_A0_to_each=changes_common, scene_f1_win_tie_loss_vs_A0=win_tie_loss,
        history010=dict(total_rows=len(history),selection_rows=selection_rows,
                        exact_match_to_previous_two_selection_records=True,
                        phase_diagnostics=history_report, unique_association_pairs_across_TAM_JOINT=len(pair_union),
                        recorded_scope='historical scene64 CE association experiment; not native-time complete fusion training'),
        limits=['V_select selected models and thresholds; no independent-test or multi-seed inference.',
                'Common TSA applicability subset is a supplementary common population, never a replacement for all pairs.',
                'History proves recorded updates/samples, not convergence or physical completion/fusion quality.',
                'File contents and receipts support consistency, not an independent reconstruction of every server action.'],
        analysis_seconds=time.monotonic()-started)
    (OUTPUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('copy','manifest','identity_checks','full','common_paper_scope_original_thresholds','score_rank_metrics','scene_f1_win_tie_loss_vs_A0','analysis_seconds')},ensure_ascii=False,indent=2))
    print('history010',json.dumps(history_report,ensure_ascii=False))
    print('A0 to B full', {k:v for k,v in changes_a0['B'].items() if k!='per_scene'})
    print('A0 to B common', {k:v for k,v in changes_common['B'].items() if k!='per_scene'})


if __name__=='__main__':
    main()
