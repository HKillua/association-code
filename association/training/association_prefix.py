"""Read-only 060 prefix comparison; never restores model, optimizer or RNG state."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from association.evaluation.association import evaluate_scenes, predict, select_threshold
from association.models.evidence_association import FEATURE_NAMES
from association.utils.runtime import read_json


WEIGHT_TOLERANCE = dict(atol=1e-7, rtol=1e-5)
SCORE_TOLERANCE = dict(atol=1e-6, rtol=1e-5)
SCORE_METADATA = ('pairs', 'labels', 'node_ids', 'sources', 'joint_support_queries')


def _numeric(actual, reference, tolerance):
    if isinstance(actual, torch.Tensor):
        actual = actual.detach().cpu().numpy()
    if isinstance(reference, torch.Tensor):
        reference = reference.detach().cpu().numpy()
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('prefix reference has incompatible shape/dtype or nonfinite numeric state')
    delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
    close = np.isclose(a, b, **tolerance)
    return dict(exact=a.tobytes(order='C') == b.tobytes(order='C'), within_tolerance=bool(close.all()),
                value_count=a.size, outside_tolerance_count=int((~close).sum()),
                max_absolute_difference=float(delta.max()) if delta.size else 0.)


def _state_comparison(actual, reference):
    leaves = {}

    def visit(a, b, name):
        if isinstance(b, dict):
            if not isinstance(a, dict) or a.keys() != b.keys():
                raise ValueError('prefix reference state keys differ: ' + name)
            for key in b:
                visit(a[key], b[key], name + '/' + str(key))
        elif isinstance(b, (list, tuple)):
            if not isinstance(a, type(b)) or len(a) != len(b):
                raise ValueError('prefix reference state structure differs: ' + name)
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, name + '/' + str(i))
        elif isinstance(b, (torch.Tensor, np.ndarray, int, float, bool)):
            leaves[name] = _numeric(a, b, WEIGHT_TOLERANCE)
        elif a != b:
            raise ValueError('prefix reference state metadata differs: ' + name)

    visit(actual, reference, 'state')
    return dict(exact=all(x['exact'] for x in leaves.values()),
                within_tolerance=all(x['within_tolerance'] for x in leaves.values()),
                tolerance=dict(WEIGHT_TOLERANCE), leaves=leaves)


class PrefixReference:
    """Bind a completed ordinary fit, then compare its final state to a new prefix.

    Native use is fixed to 056's 1000-update endpoint and a fresh 3000-update
    fit. Synthetic checks may use shorter positive endpoints. No selected
    checkpoint is read, and no model is instantiated here.
    """

    def __init__(self, directory, config, *, expected_precision, synthetic=False):
        self.path = Path(directory).resolve()
        self.run = read_json(self.path/'run.json')
        self.summary = read_json(self.path/'summary.json')
        self.audit = read_json(self.path/'data_audit.json')
        self.sampling = read_json(self.path/'sampling_audit.json')
        self.baseline = read_json(self.path/'baseline.json')
        self.checkpoint = torch.load(self.path/'final.pt', map_location='cpu', weights_only=False)
        reference_config = self.run['config']
        self.update = reference_config['training']['updates']
        self.target_updates = config['training']['updates']
        expected = deepcopy(reference_config)
        expected['training']['updates'] = config['training']['updates']
        if (isinstance(self.update, bool) or not isinstance(self.update, int) or self.update < 1
                or config != expected or 'sampling' in config
                or config['training']['updates'] <= self.update
                or self.update % config['training']['validate_every']
                or (not synthetic and (self.update != 1000 or self.target_updates != 3000
                    or config['seed'] != 20261021 or config['variant'] != 'full_evidence'
                    or config.get('motion_observation_policy') != 'canonical_all'
                    or len(self.audit['training_scene_ids']) != 2100
                    or len(self.audit['validation_scene_ids']) != 300))):
            raise ValueError('prefix reference requires ordinary same-config training; only update budget may change')
        for document in (self.run, self.summary):
            if (document.get('status') != 'COMPLETED' or document.get('optimizer_updates') != self.update
                    or document.get('synthetic') is not synthetic or document.get('test_opened') is not False
                    or document.get('reference_read') is not False):
                raise ValueError('prefix reference must be a completed final fit with matching source role')
        if (self.checkpoint.get('config') != reference_config or self.checkpoint.get('update') != self.update
                or self.checkpoint.get('scope') != config['scope']
                or self.summary.get('scope') != config['scope']
                or self.summary.get('variant') != config['variant']
                or any(document.get('precision') != expected_precision
                       for document in (self.checkpoint, self.summary, self.run))
                or any(document.get('feature_names') != list(FEATURE_NAMES)
                       for document in (self.checkpoint, self.summary))
                or any(document.get('motion_observation_policy', 'selected_cap128')
                       != config.get('motion_observation_policy', 'selected_cap128')
                       for document in (self.checkpoint, self.summary))
                or self.checkpoint.get('selection') != self.summary['final_selection']
                or self.checkpoint.get('normalizer') != self.audit['normalizer']
                or self.summary.get('population') != self.audit['validation_population']
                or self.sampling.get('actual_optimizer_updates') != self.update
                or self.sampling.get('completed_batch_count') != self.update
                or self.sampling.get('configured_updates') != self.update
                or self.sampling.get('split') != 'T_core'
                or self.sampling.get('seed') != config['seed']
                or self.sampling.get('batch_size') != config['training']['batch_size']
                or self.sampling.get('sampled_pair_count') != self.update*config['training']['batch_size']
                or any(self.sampling.get('sampled_'+kind+'_count')
                       != self.update*config['training']['batch_size']//2 for kind in ('positive', 'negative'))):
            raise ValueError('prefix reference final checkpoint, population or sampling audit disagrees')
        self.bound = False

    def _records(self, filename, validation):
        result = []
        with np.load(self.path/filename, allow_pickle=False) as archive:
            if not np.array_equal(archive['scene_ids'], [s['scene_id'] for s in validation]):
                raise ValueError('prefix reference score scene population/order differs')
            for scene in validation:
                sid = scene['scene_id']
                for key in SCORE_METADATA:
                    if not np.array_equal(archive[f'{sid}_{key}'], scene[key]):
                        raise ValueError('prefix reference score population differs: ' + key)
                scores = archive[f'{sid}_scores']
                if scores.shape != scene['base_scores'].shape or not np.isfinite(scores).all():
                    raise ValueError('prefix reference scores are incomplete or nonfinite')
                result.append({**{k: scene[k] for k in ('scene_id', 'component_id', *SCORE_METADATA)},
                               'scores': scores.copy()})
        return result

    def bind(self, audit, validation, model, optimizer):
        for key in ('split_unit', 'training_scene_ids', 'validation_scene_ids', 'normalizer',
                    'sampler', 'data', 'motion_observation_policy', 'validation_population',
                    'candidate_policy', 'training_labels', 'test_opened', 'reference_read',
                    'inference_uses_identity'):
            if audit.get(key) != self.audit.get(key):
                raise ValueError('prefix reference data/normalizer/sampler protocol differs: ' + key)
        if self.checkpoint['optimizer_state']['param_groups'] != optimizer.state_dict()['param_groups']:
            raise ValueError('prefix reference optimizer settings differ')
        # Validate parameter structure/finiteness without installing any values.
        _state_comparison(model.state_dict(), self.checkpoint['state_dict'])
        if not torch.equal(model.feature_mask.detach().cpu(), self.checkpoint['state_dict']['feature_mask']):
            raise ValueError('prefix reference fixed feature mask disagrees with its variant')
        baseline = self._records('baseline-scores.npz', validation)
        if any(not np.array_equal(row['scores'], scene['base_scores']) for row, scene in zip(baseline, validation)):
            raise ValueError('prefix reference original OLS base scores differ')
        if select_threshold(baseline) != self.baseline['selection']:
            raise ValueError('prefix reference original OLS selection differs from its scores')
        self.records = self._records('final-scores.npz', validation)
        if (select_threshold(self.records) != self.summary['final_selection']
                or evaluate_scenes(self.records, self.summary['final_selection']['threshold'])
                != self.summary['final_result']['metrics']):
            raise ValueError('prefix reference final score selection/metrics disagree')
        self.bound = True
        return dict(source_directory=str(self.path), source_endpoint='final', prefix_update=self.update,
            source_updates=self.update, target_updates=self.target_updates,
            protocol_valid=True, normalizer_exact=True, population_exact=True, ols_scores_exact=True,
            sampler_policy_exact=True, restores_parameters=False, restores_optimizer=False,
            restores_rng=False, source_provenance=self.run.get('provenance'))

    def compare(self, model, optimizer, records, selection, sampling):
        if not self.bound:
            raise ValueError('prefix reference must be bound before comparison')
        current_sampling, reference_sampling = deepcopy(sampling), deepcopy(self.sampling)
        current_sampling.pop('configured_updates', None)
        reference_sampling.pop('configured_updates', None)
        if current_sampling != reference_sampling:
            raise ValueError('prefix successful-update sampling exposure differs from reference')
        weights = _state_comparison(model.state_dict(), self.checkpoint['state_dict'])
        adam = _state_comparison(optimizer.state_dict(), self.checkpoint['optimizer_state'])
        per_scene = []
        old_threshold = self.summary['final_selection']['threshold']
        new_threshold = selection['threshold']
        for new, old in zip(records, self.records):
            if new['scene_id'] != old['scene_id']:
                raise ValueError('prefix endpoint scene order differs')
            stat = _numeric(new['scores'], old['scores'], SCORE_TOLERANCE)
            stat.update(scene_id=new['scene_id'], own_threshold_changed_decisions=int(np.count_nonzero(
                predict(new['scores'], new_threshold) != predict(old['scores'], old_threshold))),
                reference_threshold_changed_decisions=int(np.count_nonzero(
                    predict(new['scores'], old_threshold) != predict(old['scores'], old_threshold))))
            per_scene.append(stat)
        current_metrics = evaluate_scenes(records, new_threshold)
        old_metrics = deepcopy(self.summary['final_result']['metrics'])
        metric_values = {key: dict(reference=old_metrics[key], prefix=current_metrics[key],
            difference=(current_metrics[key]-old_metrics[key]
                        if current_metrics[key] is not None and old_metrics[key] is not None else None))
            for key in ('tp', 'fp', 'fn', 'tn', 'macro_f1', 'micro_f1', 'micro_precision', 'micro_recall')}
        current_metrics.pop('threshold')
        old_metrics.pop('threshold')
        metrics_exact = current_metrics == old_metrics
        score_exact = all(x['exact'] for x in per_scene)
        score_close = all(x['within_tolerance'] for x in per_scene)
        changed = sum(x['own_threshold_changed_decisions'] for x in per_scene)
        valid = weights['within_tolerance'] and adam['within_tolerance'] and score_close and not changed and metrics_exact
        exact = weights['exact'] and adam['exact'] and score_exact and selection == self.summary['final_selection']
        return dict(comparison_status='EXACT_MATCH' if exact and valid else 'TOLERANT_MATCH' if valid else 'MISMATCH',
            causal_attribution_valid=bool(valid), bitwise_state_and_scores_equal=weights['exact'] and adam['exact'] and score_exact,
            prefix_update=self.update, source_directory=str(self.path), source_endpoint='final',
            protocol_valid=True, population_exact=True, normalizer_exact=True, sampling_audit_exact=True,
            weights=weights, optimizer=adam,
            scores=dict(exact=score_exact, within_tolerance=score_close, tolerance=dict(SCORE_TOLERANCE),
                max_absolute_difference=max(row['max_absolute_difference'] for row in per_scene),
                outside_tolerance_count=sum(row['outside_tolerance_count'] for row in per_scene), per_scene=per_scene),
            thresholds=dict(reference=old_threshold, prefix=new_threshold, exact=old_threshold == new_threshold),
            own_threshold_changed_decisions=changed,
            reference_threshold_changed_decisions=sum(x['reference_threshold_changed_decisions'] for x in per_scene),
            edge_metrics_exact=metrics_exact, edge_metric_differences=metric_values,
            action='continue the original fixed update budget; never restore, retune or extend',
            interpretation='tolerance agreement is not bitwise reproduction; mismatches invalidate pure step-count attribution')
