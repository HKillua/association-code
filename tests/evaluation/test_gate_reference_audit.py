"""069 fixed reference brackets never filter, infer, or read original data."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from association.data.archive import DataError
from association.data.targets import align_reference_states
from association.evaluation.gate_reference_audit import ARMS, FOCAL, _reader, audit_reference


def write_gzip(path, value):
    with gzip.open(path, 'wt') as stream:
        json.dump(value, stream)


class GateReferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'preflight').mkdir()
        (self.root/'preflight/report.json').write_text(json.dumps(dict(status='PASS', data_contract_reproduced=True)))
        self.rows, self.caches = [], {}
        for sid, spec in FOCAL.items():
            identity = spec['target']
            times = sorted(set(t for pair in spec['intervals'] for t in pair))
            raw = dict(scene_id=sid, member=f'data/真实场景/场景-{sid}.csv', rows=[
                dict(MMSI=identity, time=str(t), lat=str(19.+i*.1), lon='130.', vel='7.', cou='10.')
                for i, t in enumerate(times)])
            # One query lies in first interval; one exact point; no query in huge short interval.
            query = np.array([times[0]-1, (times[0]+times[1])/2, times[2], times[-1]+1])
            aligned = align_reference_states(_reader(raw, sid), identity, query,
                time_offset_seconds=0., max_gap_seconds=600., longitude_policy='shortest_arc')
            truth, valid = aligned['states'][None], aligned['valid'][None]
            n = max(spec['nodes'])+1
            offsets, indices = [0], []
            focal_groups = [[4,133],[18]] if sid=='3035' else [[6,14]]
            groups = focal_groups + [[i] for i in range(n) if i not in spec['nodes']]
            for group in groups:
                indices.extend(group)
                offsets.append(len(indices))
            cache = dict(query_seconds=query, target_ids=np.array([identity]), complete_truth=truth,
                complete_truth_valid=valid, selected_group_by_target=np.array([0]),
                group_offsets=np.array(offsets), group_node_indices=np.array(indices),
                node_ids=np.array([f'{sid}:9001:{i}' for i in range(n)]), identities=np.array([identity]*n),
                native_states=np.broadcast_to(np.array([19., 130., 5., 10.]), (n, len(query), 4)).copy(),
                support=np.ones((n,len(query)), bool), nearest_seconds=np.zeros((n,len(query))))
            for node in spec['nodes']:
                cache[f'canonical_times_{node}'] = np.array([0., 10., 20.])
                cache[f'canonical_values_{node}'] = np.array([[19.,130.,5.,10.], [20.,130.,80000.,10.], [19.,130.,5.,10.]])
                cache['support'][node] = (query >= 0.) & (query <= 20.)
                cache['nearest_seconds'][node] = np.abs(cache[f'canonical_times_{node}'][:,None]-query[None,:]).min(0)
            self.caches[sid] = cache
            np.savez_compressed(self.root/f'preflight/scene-{sid}-evidence.npz', **cache)
            write_gzip(self.root/f'preflight/scene-{sid}-raw-reference.json.gz', raw)
            write_gzip(self.root/f'preflight/scene-{sid}.json.gz', dict(scene_id=sid,
                contracts_reproduced=True, alignment=dict(time_offset_seconds=0., max_gap_seconds=600.)))
            mask = valid[..., :2].all(-1)
            errors = {arm: np.where(mask, 20., np.nan) for arm in ARMS}
            errors['final'][0,1], errors['final'][0,2] = 10., 25.
            self.rows.append(dict(scene_id=sid, target_ids=cache['target_ids'], query_seconds=query,
                errors=errors, regions={'all':np.ones(mask.shape,bool)}, truth=truth, truth_valid=valid,
                selected_group_by_target=cache['selected_group_by_target'], groups=groups,
                native_inputs={k:cache[k] for k in ('node_ids','identities','native_states','support','nearest_seconds','query_seconds')}))
        # Denominator contains another scene and cannot silently reduce to named targets.
        errors = {arm:np.array([[30.,np.nan]]) for arm in ARMS}
        errors['final'] = np.array([[35.,np.nan]])
        self.rows.append(dict(scene_id='9',target_ids=np.array(['other-0']), query_seconds=np.array([0.,1.]),
                              errors=errors,regions={'all':np.ones((1,2),bool)}))

    def test_fixed_brackets_truth_binding_full_population_and_no_forward(self):
        with patch('torch.load', side_effect=AssertionError('checkpoint forbidden')), \
             patch('torch.nn.Module.__call__', side_effect=AssertionError('forward forbidden')), \
             patch('association.data.archive.MTADArchive', side_effect=AssertionError('original archives forbidden')):
            report = audit_reference(self.root, self.rows)
        self.assertEqual(report['full_population_position_points'],5)
        self.assertEqual(report['scenes'][0]['named_bracket_query_indices'],[1])
        # The more abrupt named jump does not automatically become an evaluated query.
        short = report['scenes'][0]['named_intervals'][1]
        self.assertEqual(short['interior_query_indices'],[])
        self.assertEqual(short['exact_endpoint_query_indices'],[2])
        both = report['named_bracket_union_contributions']['initial']
        self.assertEqual(both['count'],2)
        self.assertEqual(both['signed_error_change_sum_m'],-20.)
        self.assertEqual(both['gross_improvement_sum_m'],20.)
        self.assertEqual(both['contribution_to_full_population_mae_change_m'],-4.)
        self.assertEqual(both['complement_sensitivity']['count'],3)
        self.assertEqual(both['complement_sensitivity']['mae_change_m'],5.)
        self.assertEqual(both['signed_squared_error_change_sum_m2'],-600.)
        canonical = report['scenes'][0]['canonical_tracks'][0]
        self.assertEqual(canonical['native_velocity_range'],[5.,80000.])
        self.assertIn('unverified',canonical['native_velocity_units'])
        self.assertTrue(canonical['is_unique_adopted_representative'])
        self.assertFalse(report['scenes'][0]['canonical_tracks'][1]['is_unique_adopted_representative'])
        self.assertFalse(report['scenes'][0]['canonical_tracks'][1]['in_unique_adopted_group'])
        self.assertTrue(report['scenes'][0]['canonical_tracks'][1]['is_group_representative'])
        self.assertFalse(report['anomalies_removed'])
        json.dumps(report,allow_nan=False)

    def test_tampered_cached_truth_or_masks_rejected(self):
        for key in ('complete_truth','complete_truth_valid'):
            cached = deepcopy(self.caches['3035'])
            cached[key][0,1,0] = (cached[key][0,1,0]+1 if key=='complete_truth' else False)
            np.savez_compressed(self.root/'preflight/scene-3035-evidence.npz', **cached)
            with self.subTest(key=key), self.assertRaises(DataError):
                audit_reference(self.root,self.rows)
        np.savez_compressed(self.root/'preflight/scene-3035-evidence.npz', **self.caches['3035'])

    def test_named_adjacent_rows_and_missing_inputs_fail_without_invention(self):
        path=self.root/'preflight/scene-3035-raw-reference.json.gz'
        with gzip.open(path,'rt') as stream: raw=json.load(stream)
        original=deepcopy(raw)
        inserted=dict(raw['rows'][0],time='5800.0')
        raw['rows'].insert(1,inserted)
        write_gzip(path,raw)
        with self.assertRaises(DataError): audit_reference(self.root,self.rows)
        write_gzip(path,original)
        (self.root/'preflight/scene-3035-evidence.npz').unlink()
        with self.assertRaises(FileNotFoundError): audit_reference(self.root,self.rows)

    def test_changed_endpoint_population_is_not_a_better_subset(self):
        rows=deepcopy(self.rows)
        rows[0]['errors']['final'][0,1]=np.nan
        with self.assertRaises(DataError): audit_reference(self.root,rows)
        with self.assertRaises(DataError): audit_reference(self.root,self.rows[1:])

    def test_preflight_and_endpoint_inputs_cannot_be_mixed(self):
        for key in ('support','native_states','identities'):
            rows=deepcopy(self.rows)
            value=rows[0]['native_inputs'][key]
            value.flat[0]=not value.flat[0] if key=='support' else (value.flat[0]+1 if key=='native_states' else 'wrong-0')
            with self.subTest(key=key),self.assertRaises(DataError): audit_reference(self.root,rows)

    def test_zero_effective_query_subset_has_zero_counts_not_zero_denominator(self):
        # A legal smaller gap masks interpolation but keeps the exact endpoint.
        for sid in FOCAL:
            p=self.root/f'preflight/scene-{sid}.json.gz'
            with gzip.open(p,'rt') as stream: report=json.load(stream)
            report['alignment']['max_gap_seconds']=1.
            write_gzip(p,report)
            cache=deepcopy(self.caches[sid])
            cache['complete_truth'][0,1]=np.nan
            cache['complete_truth_valid'][0,1]=False
            np.savez_compressed(self.root/f'preflight/scene-{sid}-evidence.npz',**cache)
            row=next(r for r in self.rows if r['scene_id']==sid)
            row['truth']=cache['complete_truth'];row['truth_valid']=cache['complete_truth_valid']
            for arm in ARMS: row['errors'][arm][0,1]=np.nan
        result=audit_reference(self.root,self.rows)
        c=result['named_bracket_union_contributions']['initial']
        self.assertEqual(c['count'],0)
        self.assertIsNone(c['subset_mae_change_m'])
        self.assertEqual(c['full_population_count'],3)
        json.dumps(result,allow_nan=False)


if __name__=='__main__': unittest.main()
