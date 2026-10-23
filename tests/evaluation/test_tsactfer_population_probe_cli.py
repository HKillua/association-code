"""080 reads and analyses frozen score caches without loading a model or raw data."""
from argparse import Namespace
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from association.evaluation.tsactfer_input_probe import DEFAULT as PROBE_CONFIG, probe, select_roster
from association.utils.runtime import read_json, write_json, write_json_gzip
from baselines.tsactfer import TSAConfig, TSActfer
from baselines.tsactfer_data import fit_normalizer
from baselines.tsactfer_level1 import compact_native, pack_compact
from tests.test_tsactfer_level1 import native_scene, tiny_config


def cached_sources(root):
    """Make actual079 score/view artifacts from small native tracks.

    074D binding is separately covered; only that large source is substituted in
    the CLI test. The080 source loader and population computations remain real.
    """
    source074 = root/'LEVEL1-TSA-NATIVE-20261023-074D-attempt-1'
    source079 = root/'LEVEL1-TSA-INPUT-PROBE-20261024-079-attempt-1'
    source074.mkdir(); source079.mkdir(); (source079/'scores').mkdir()
    manifest = dict(splits={'T_core': [f't{i}' for i in range(40)],
                           'V_select': [f'v{i}' for i in range(40)], 'T_test': ['sealed']})
    manifest['components'] = {sid: 'component-'+sid for rows in manifest['splits'].values() for sid in rows}
    tsa = tiny_config()
    roster = select_roster(manifest, PROBE_CONFIG)
    views = [compact_native(native_scene(sid), tsa['data']['max_points']) for sid in roster]
    normalizer = fit_normalizer(views)
    normalizer['training_splits'] = ['T_core']
    scenes = [pack_compact(view, normalizer, tsa) for view in views]
    torch.manual_seed(71)
    model = TSActfer(TSAConfig(**tsa['model']))
    summary, saved = probe(model, scenes, tsa, PROBE_CONFIG)
    summary.update(status='COMPLETED', configuration=PROBE_CONFIG, scene_ids=roster,
                   reference_read=False, validation_observations_read=False, test_opened=False,
                   device='cpu', elapsed_seconds=1., peak_cuda_memory_allocated_bytes=0)
    run = dict(scope=PROBE_CONFIG['scope'], status='COMPLETED', configuration=PROBE_CONFIG,
               source_074d=str(source074), scene_ids=roster, device='cpu',
               optimizer_updates=0, model_forward=True, test_opened=False,
               validation_observations_read=False, reference_read=False,
               threshold_selected=False, groups_recomputed=False, retries=0,
               advisory_reached=False, old066_acceptance='REJECTED', phase='completed', elapsed_seconds=1.)
    write_json(source079/'run.json', run)
    write_json(source079/'summary.json', summary)
    write_json(source079/'source-verification.json', dict(source=str(source074),
        checkpoint=str(source074/'fit/final.pt'), endpoint_steps=1000,
        fit_exposure_verified=True, normalizer_refit=False, source_final_bound=True,
        validation_metadata_read=True, validation_arrays_read=False))
    reads = [dict(scene_id=sid, kind=kind, member=f'{sid}/{kind}.json')
             for sid in roster for kind in ('labels', 'measurements')]
    write_json(source079/'data-audit.json', dict(scene_ids=roster, archive_members_read=reads,
        packing=[scene['packing_audit'] for scene in scenes], reference_read=False,
        validation_read=False, test_opened=False))
    write_json(source079/'execution.json', dict(arguments=dict(source_074d=str(source074)),
        configuration=PROBE_CONFIG, selected_training_scene_ids=roster,
        normalizer_refit=False, reference_read=False, validation_observations_read=False,
        test_opened=False, training=False))
    for head, item in saved.items():
        np.savez_compressed(source079/'scores'/f'{head.lower()}.npz', **item['arrays'])
        write_json_gzip(source079/'scores'/f'{head.lower()}-views.json.gz', item['rows'])
    records = []
    for index, sid in enumerate(manifest['splits']['V_select']):
        row = native_scene(sid, 'V_select')
        row.pop('tracks')
        row['node_ids'] = np.asarray(row['node_ids'])
        row['sources'] = np.asarray(row['sources'])
        row['scores'] = np.where(row['labels'], .8, .2).astype(np.float32)
        # One observable negative tail, without changing the frozen labels.
        row['scores'][np.flatnonzero(row['labels'] == 0)[0]] = .9
        records.append(row)
    return source074, source079, manifest, records, tsa


class TSAPopulationProbeCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_cached_probe_to_cli_and_failed_run_preservation(self):
        from tools.analysis import tsactfer_population_probe as entry
        from association.evaluation.tsactfer_population_probe import DEFAULT
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source074, source079, manifest, records, tsa = cached_sources(root)
            write_json(root/'manifest.json', manifest); write_json(root/'config.json', DEFAULT)
            args = Namespace(source_074d=str(source074), source_079=str(source079),
                manifest=str(root/'manifest.json'), config=str(root/'config.json'),
                output=str(root/'result'), cpu_threads=4)
            bound074 = (records, .5, tsa, manifest['splits']['T_core'],
                        dict(source_metrics_reproduced=True, checkpoints_opened=False))
            with ExitStack() as stack:
                stack.enter_context(patch.object(entry, 'production_contract'))
                stack.enter_context(patch.object(entry, 'bind_source', return_value=bound074))
                for target in ('torch.load', 'torch.optim.Adam', 'baselines.tsactfer.TSActfer.forward',
                               'association.data.archive.MTADArchive.__init__'):
                    stack.enter_context(patch(target, side_effect=AssertionError('080 must use cached arrays only')))
                result = entry.run(args)
                record = read_json(root/'result/run.json')
                self.assertEqual(record['status'], 'COMPLETED')
                self.assertFalse(record['test_opened'])
                self.assertEqual(result['model_forwards'], 0)
                self.assertFalse(result['raw_archives_read'])
                for split in ('train_balanced_full', 'validation_balanced_full'):
                    for head in ('ITSA', 'MSTSA'):
                        metric = result[split][head]['pooled']
                        self.assertGreater(metric['positive_count'], 0)
                        self.assertEqual(metric['positive_count'], metric['negative_count'])
                        self.assertAlmostEqual(metric['average_precision'], metric['average_precision_prior05'])
                self.assertEqual(result['validation_full_candidates']['scene_count'], 32)
                self.assertTrue((root/'result/summary.json').is_file())
                self.assertTrue((root/'result/report.md').is_file())
                self.assertTrue((root/'result/selected-views.json.gz').is_file())
                old = (root/'result/run.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    entry.run(args)
                self.assertEqual((root/'result/run.json').read_bytes(), old)
                args.output = str(root/'failed')
                with patch.object(entry, 'analyze_population', side_effect=RuntimeError('controlled failure')):
                    with self.assertRaisesRegex(RuntimeError, 'controlled failure'):
                        entry.run(args)
                self.assertEqual(read_json(root/'failed/run.json')['status'], 'FAILED')
                self.assertTrue((root/'failed/execution.json').is_file())
                args.output = str(source079/'forbidden-child')
                with self.assertRaisesRegex(ValueError, 'overlaps'):
                    entry.run(args)
                self.assertFalse((source079/'forbidden-child').exists())

    def test_079_binding_rejects_incomplete_read_scope_and_wrong_source(self):
        from tools.analysis import tsactfer_population_probe as entry
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source074, source079, manifest, _, _ = cached_sources(root)
            summary, saved, binding = entry.bind_probe(source079, source074, manifest)
            self.assertEqual(set(saved), {'ITSA', 'MSTSA'})
            for filename, change in (
                    ('run.json', lambda row: row.update(status='FAILED')),
                    ('run.json', lambda row: row.update(source_074d=str(root/'different'))),
                    ('summary.json', lambda row: row.update(frozen_weights_unchanged=False)),
                    ('data-audit.json', lambda row: row['archive_members_read'][0].update(kind='reference')),
                    ('execution.json', lambda row: row.update(training=True))):
                path = source079/filename; original = read_json(path); bad = deepcopy(original); change(bad)
                write_json(path, bad)
                with self.assertRaises(ValueError, msg=filename):
                    entry.bind_probe(source079, source074, manifest)
                write_json(path, original)
