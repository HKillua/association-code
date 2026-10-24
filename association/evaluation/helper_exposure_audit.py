"""084: read-only successful-training exposure accounting; no inference or fitting.

A sample contributes its mean valid-query loss to the batch mean. Consequently
pooled valid-query frequencies are not its loss coefficients. Zero-supervision
samples retain their batch denominator and contribute zero coefficient mass.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import gzip
import json
import math
from pathlib import Path

BUCKETS = ('zero', 'one', 'two_or_more')
SOURCE_BUCKETS = ('zero', 'one', 'multiple')
POPULATIONS = ('all_queries', 'valid_position_queries')
MODES = ('clean', 'clean_plus_wrong', 'wrong_only')
WEIGHTINGS = ('point_equal', 'scene_target_equal', 'scene_equal_point_mean', 'scene_equal_target_mean')
LOSS_SEMANTICS = 'mean_valid_position_linear_haversine_m_div_1000_all_queries_no_auxiliary'
LIMITATIONS = [
    'Loss coefficients describe sample/query averaging, not gradient contribution, Adam updates or clipping effects.',
    'Scene-anchor-mode deduplication keeps distinct corruption modes separate; it is descriptive, not a new training objective.',
    'Training and validation populations differ in scenes, anchors, groups and repeated exposure; differences are not causal attribution.',
    'Validation aggregate counts cannot recover per-anchor loss coefficients or identity-by-support joint counts.',
    'Anchor/node mappings and repeated memberships are checked within saved logs; raw node rosters and per-query arrays are not read.',
    'No original archives, checkpoints, model forwards, training, threshold selection or T_test are accessed.',
]


def _fail(message):
    raise ValueError(message)


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        _fail(label + ' must be an integer >= ' + str(minimum))
    return value


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        _fail(label + ' must be finite and nonnegative')
    return value


def _no_constant(value):
    _fail('nonfinite JSON constant: ' + value)


def read_json(path):
    """Only JSON/JSON.gz reports are accepted; never open a data archive."""
    path = Path(path)
    if not (path.name.endswith('.json') or path.name.endswith('.json.gz')):
        _fail('expected a .json or .json.gz report')
    opener = gzip.open if path.name.endswith('.json.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle, parse_constant=_no_constant)


def _counts(value, label, *, source=True):
    keys = SOURCE_BUCKETS if source else BUCKETS
    if not isinstance(value, dict) or set(value) != set(keys):
        _fail(label + ' requires exactly three helper buckets')
    return {dst: _integer(value[src], label + '/' + src) for dst, src in zip(BUCKETS, keys)}


def _same_record(left, right):
    # Python considers True == 1; provenance counters must not accept that.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _fractions(counts, denominator):
    return {k: counts[k] / denominator if denominator else None for k in BUCKETS}


@dataclass
class _Totals:
    samples: int = 0
    supervised: int = 0
    counts: dict = field(default_factory=lambda: {p: Counter({k: 0 for k in BUCKETS}) for p in POPULATIONS})
    coefficients: dict = field(default_factory=lambda: {k: 0. for k in BUCKETS})
    eligible_valid: int = 0
    eligible_coefficients: float = 0.

    def add(self, counts, valid, eligible_valid):
        self.samples += 1
        self.supervised += int(valid > 0)
        for p in POPULATIONS:
            self.counts[p].update(counts[p])
        if valid:
            for k in BUCKETS:
                self.coefficients[k] += counts['valid_position_queries'][k] / valid
            self.eligible_coefficients += eligible_valid / valid
        self.eligible_valid += eligible_valid

    def describe(self):
        return dict(sample_count=self.samples, supervised_sample_count=self.supervised,
            zero_supervision_sample_count=self.samples-self.supervised,
            point_exposure={p: dict(total=sum(c.values()), counts=dict(c),
                fractions=_fractions(c, sum(c.values()))) for p, c in self.counts.items()},
            sample_equal_loss_coefficients=dict(denominator_samples=self.samples,
                shares=_fractions(self.coefficients, self.samples),
                coefficient_mass=self.supervised/self.samples if self.samples else None,
                zero_supervision_in_denominator=True),
            anchor_outside_helper_supported=dict(valid_query_count=self.eligible_valid,
                sample_equal_loss_coefficient=self.eligible_coefficients/self.samples if self.samples else None))


def _scene_equal(scenes):
    n = len(scenes)
    if not n:
        return dict(scene_count=0, sample_equal_loss_coefficients=None, scene_point_fractions={})
    coefficients = {k: sum(s.coefficients[k]/s.samples for s in scenes.values())/n for k in BUCKETS}
    point_fractions = {}
    for p in POPULATIONS:
        valid_scenes = [s for s in scenes.values() if sum(s.counts[p].values())]
        point_fractions[p] = dict(denominator_scenes=len(valid_scenes), zero_point_scenes=n-len(valid_scenes),
            shares={k: sum(s.counts[p][k]/sum(s.counts[p].values()) for s in valid_scenes)/len(valid_scenes)
                    if valid_scenes else None for k in BUCKETS})
    return dict(scene_count=n, sample_equal_loss_coefficients=dict(shares=coefficients,
        coefficient_mass=sum(s.supervised/s.samples for s in scenes.values())/n,
        zero_supervision_scenes_in_denominator=True), scene_point_fractions=point_fractions)


class _Strata:
    def __init__(self):
        self.total = _Totals()
        self.scenes = {}
        self.modes = {m: _Totals() for m in MODES}
        self.mode_scenes = {m: {} for m in MODES}

    def add(self, sid, mode, counts, valid, eligible_valid):
        for destination in (self.total, self.modes[mode],
                self.scenes.setdefault(sid, _Totals()), self.mode_scenes[mode].setdefault(sid, _Totals())):
            destination.add(counts, valid, eligible_valid)

    def describe(self):
        return dict(**self.total.describe(), scene_equal=_scene_equal(self.scenes),
            modes={m: dict(**self.modes[m].describe(), scene_equal=_scene_equal(self.mode_scenes[m])) for m in MODES})


def audit_fit(fit, *, allowed_training_scene_ids, expected_configuration=None):
    """Audit one completed fit using its full successful-update JSONL only.

    The caller owns the immutable legal T list. Production additionally binds
    the full configuration to 078B or 081A; tests may supply a tiny actual fit.
    """
    fit = Path(fit).resolve()
    record = read_json(fit/'run.json')
    config = record.get('configuration', {})
    training = config.get('training', {})
    seed = _integer(training.get('seed'), 'seed', 1)
    updates = _integer(training.get('updates'), 'configured updates', 1)
    batch_size = _integer(training.get('batch_size'), 'batch size', 1)
    length = _integer(config.get('data', {}).get('length'), 'query length', 1)
    clip = _finite(training.get('gradient_clip'), 'gradient clip')
    legal = list(allowed_training_scene_ids)
    if (not legal or any(not isinstance(s, str) or not s for s in legal) or len(set(legal)) != len(legal)
            or record.get('training_scene_ids') != legal):
        _fail('training_scene_ids must exactly match the legal T_core list')
    if (record.get('status') != 'COMPLETED' or record.get('test_opened') is not False
            or record.get('optimizer_updates') != updates or type(record.get('optimizer_updates')) is not int
            or config.get('train_splits') != ['T_core'] or config.get('position_loss') != 'linear'
            or config.get('scope') != 'helper-sequence-position-v1' or record.get('scope') != config['scope']
            or record.get('loss_semantics') != LOSS_SEMANTICS
            or config.get('motion_observation_policy') != 'canonical_all'
            or record.get('motion_observation_policy') != 'canonical_all'
            or record.get('helper_aggregation') != 'pre_pool' or config.get('model', {}).get('aggregation') != 'pre_pool'
            or record.get('final_checkpoint') != 'final.pt'):
        _fail('source must be a completed T_core pre_pool linear fit with a fixed final endpoint')
    if expected_configuration is not None and config != expected_configuration:
        _fail('source configuration differs from the frozen fit')
    paired = record.get('paired_control', {})
    if (paired.get('status') != 'COMPLETED' or paired.get('initial_parameters_exact') is not True
            or paired.get('initial_sampler_rng_exact') is not True or paired.get('sampling_exposure_exact') is not True
            or type(paired.get('actual_successful_batches_matched')) is not int
            or paired.get('actual_successful_batches_matched') != updates):
        _fail('source must retain its completed paired-control audit')

    legal = set(legal)
    actual, unique = _Strata(), _Strata()
    signatures, anchor_nodes, node_anchors, node_valid = {}, {}, {}, {}
    scene_counts, anchor_counts = Counter(), Counter()
    mode_counts = {m: dict(updates=0, anchors=0, valid_position_points=0) for m in MODES}
    gradient_updates, clipped, rows_read = 0, 0, 0
    helper_gradients = dict(phi_nonzero_updates=0, scorer_nonzero_updates=0)
    with (fit/'training.jsonl').open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                _fail('blank training JSONL line ' + str(line_number))
            row = json.loads(line, parse_constant=_no_constant)
            step = line_number
            mode = MODES[(step-1) % len(MODES)]
            if (step > updates or type(row.get('step')) is not int or row['step'] != step or row.get('mode') != mode
                    or not isinstance(row.get('batch'), list) or len(row['batch']) != batch_size
                    or not isinstance(row.get('helper_support_batch'), list)
                    or len(row['helper_support_batch']) != batch_size):
                _fail('successful batch step/mode/count mismatch at update ' + str(step))
            norm = _finite(row.get('gradient_norm'), 'gradient norm')
            _finite(row.get('loss'), 'loss')
            if type(row.get('gradient_clipped')) is not bool or row['gradient_clipped'] != (norm > clip):
                _fail('gradient-clipping record differs from the successful update')
            gradient_updates += int(norm > 0)
            clipped += int(row['gradient_clipped'])
            gradients = row.get('helper_gradients', {})
            if set(gradients) != set(helper_gradients) or any(type(v) is not bool for v in gradients.values()):
                _fail('helper gradient update flags must be explicit booleans')
            for k, v in gradients.items():
                helper_gradients[k] += int(v)
            batch_valid = 0
            for sample, helper in zip(row['batch'], row['helper_support_batch']):
                sid, node = sample.get('scene_id'), sample.get('node_id')
                anchor = _integer(sample.get('anchor_index'), 'anchor index')
                if sid not in legal or not isinstance(node, str) or not node.startswith(sid+':') or node == sid+':':
                    _fail('sample scene/node is outside legal T_core or lacks its scene prefix')
                if (anchor_nodes.setdefault((sid, anchor), node) != node
                        or node_anchors.setdefault((sid, node), anchor) != anchor):
                    _fail('anchor-node mapping changed across actual successful samples')
                members = sample.get('members')
                if (not isinstance(members, list) or not members or members[0] != anchor
                        or any(type(x) is not int or x < 0 for x in members) or len(set(members)) != len(members)):
                    _fail('members must be distinct nonnegative nodes with the original anchor first')
                valid = _integer(sample.get('valid_position_points'), 'sample valid positions')
                eligible = _integer(sample.get('eligible_query_points'), 'eligible queries')
                eligible_valid = _integer(sample.get('eligible_valid_position_points'), 'eligible valid queries')
                if valid > length or eligible > length or eligible_valid > min(valid, eligible):
                    _fail('sample valid/eligible denominator exceeds its query population')
                if node_valid.setdefault((sid, node), valid) != valid:
                    _fail('the same anchor changes its target valid-point denominator')
                if not isinstance(helper, dict) or set(helper) != set(POPULATIONS):
                    _fail('helper_support_batch requires all and valid query counts')
                counts = {p: _counts(helper[p], 'helper_support_batch/'+p) for p in POPULATIONS}
                a, v = counts['all_queries'], counts['valid_position_queries']
                if sum(a.values()) != length or sum(v.values()) != valid or any(v[k] > a[k] for k in BUCKETS):
                    _fail('helper bucket sum/subset differs from the valid-position denominator')
                if (eligible > a['one']+a['two_or_more'] or eligible_valid > v['one']+v['two_or_more']
                        or (len(members) == 1 and a['zero'] != length)
                        or (len(members) <= 2 and a['two_or_more'] != 0)):
                    _fail('helper counts or eligible queries are impossible for the saved membership')
                key = (sid, node, mode)
                signature = (anchor, tuple(members), valid, eligible, eligible_valid,
                             tuple(a[k] for k in BUCKETS), tuple(v[k] for k in BUCKETS))
                if key in signatures:
                    if signatures[key] != signature:
                        _fail('repeated scene-anchor-mode membership/support counts changed')
                else:
                    signatures[key] = signature
                    unique.add(sid, mode, counts, valid, eligible_valid)
                actual.add(sid, mode, counts, valid, eligible_valid)
                scene_counts[sid] += 1
                anchor_counts[node] += 1
                batch_valid += valid
            objective_points = row.get('objectives', {}).get('valid_position_points')
            if type(objective_points) is not int or objective_points != batch_valid:
                _fail('batch objective valid-point count differs from successful samples')
            mode_counts[mode]['updates'] += 1
            mode_counts[mode]['anchors'] += batch_size
            mode_counts[mode]['valid_position_points'] += batch_valid
            rows_read += 1
    if rows_read != updates:
        _fail('training.jsonl does not contain every successful configured update')
    expected_exposure = dict(successful_updates=updates, sampled_anchors=updates*batch_size,
        valid_position_points=sum(actual.total.counts['valid_position_queries'].values()),
        nonzero_gradient_updates=gradient_updates, scenes=dict(scene_counts), anchors=dict(anchor_counts), modes=mode_counts)
    if not _same_record(record.get('exposure'), expected_exposure):
        _fail('saved exposure does not match the full successful-update log')
    saved_support = record.get('helper_support_exposure', {})
    if set(saved_support) != set(POPULATIONS) or any(
            _counts(saved_support[p], 'run helper exposure/'+p) != dict(actual.total.counts[p]) for p in POPULATIONS):
        _fail('saved helper exposure does not match the full successful-update log')
    if (type(record.get('trained_scene_count')) is not int or record.get('trained_scene_count') != len(scene_counts)
            or not _same_record(record.get('helper_gradients'), helper_gradients)
            or type(record.get('gradient_clipping', {}).get('clipped_updates')) is not int
            or record.get('gradient_clipping', {}).get('clipped_updates') != clipped
            or not math.isclose(record.get('gradient_clipping', {}).get('successful_update_fraction', -1), clipped/updates,
                                rel_tol=1e-12, abs_tol=1e-12)):
        _fail('source trained-scene/gradient counters do not match successful updates')
    result = dict(status='VERIFIED', source_fit=str(fit), seed=seed, configuration=config,
        successful_updates=updates, legal_training_scene_count=len(legal), observed_training_scene_count=len(scene_counts),
        observed_scene_anchor_count=len(anchor_nodes), mode_updates={m: mode_counts[m]['updates'] for m in MODES},
        actual_exposure=actual.describe(), unique_scene_anchor_mode=unique.describe(),
        repeated_scene_anchor_mode_exposures=actual.total.samples-len(signatures),
        checks=dict(full_successful_log=True, legal_T_only=True, counters_exact=True,
            stable_anchor_node_mapping=True, stable_target_denominators=True, stable_repeated_scene_anchor_mode=True),
        paired_control_status_recorded=paired['status'])
    return result


def validation_description(path, *, allowed_validation_scene_ids):
    """Copy descriptive V counts/error weightings; never infer V sample weights."""
    data = read_json(path)
    ids = list(allowed_validation_scene_ids)
    if (data.get('status') != 'COMPLETED' or data.get('profile') != 'replica081'
            or data.get('test_opened') is not False or data.get('complete_profile') is not True
            or data.get('scene_ids') != ids or data.get('scene_count') != len(ids)
            or not ids or len(set(ids)) != len(ids)):
        _fail('expected completed 081 comparison with the exact legal V_select population')
    names = ('all_output_group_queries', 'all_output_group_valid_position_queries',
             'unique_selected_valid_position_queries')
    raw = data.get('helper_population', {})
    if set(raw) != set(names):
        _fail('V requires all three distinct helper populations')
    populations = {}
    previous = None
    for name in names:
        value = raw[name]
        total = _integer(value.get('total'), 'V helper total')
        counts = _counts(value.get('counts'), 'V helper counts', source=False)
        if sum(counts.values()) != total or (previous and any(counts[k] > previous[k] for k in BUCKETS)):
            _fail('V helper counts fail population sum/subset checks')
        populations[name] = dict(total=total, counts=counts, fractions=_fractions(counts, total))
        previous = counts
    position, population = data.get('position_population', {}), data.get('population', {})
    n = _integer(position.get('covered_target_points'), 'V covered points')
    trajectories = _integer(position.get('covered_target_trajectories'), 'V covered trajectories')
    groups = _integer(population.get('predicted_group_count'), 'V output groups')
    if populations[names[-1]]['total'] != n or populations[names[0]]['total'] != groups*64:
        _fail('V helper population differs from complete-system population')
    metrics = {}
    for name, values in data.get('position_weightings', {}).get('all', {}).items():
        if set(values) != set(WEIGHTINGS):
            _fail('V method requires the four explicitly named error weightings')
        metrics[name] = {}
        for key, units in zip(WEIGHTINGS, (n, trajectories, len(ids), len(ids))):
            value = values[key]
            if _integer(value.get('unit_count'), 'V weighting units') != units:
                _fail('V metric weighting denominator differs from its named population')
            kept = {k: _finite(value.get(k), 'V '+name+'/'+key+'/'+k) for k in ('mae', 'mse', 'rmse')}
            if not math.isclose(kept['rmse']**2, kept['mse'], rel_tol=1e-10, abs_tol=1e-8):
                _fail('V RMSE/MSE inconsistency')
            metrics[name][key] = dict(unit_count=units, **kept)
    if not {'linear', 'pre_pool', 'weighted_pool'} <= set(metrics):
        _fail('V comparison lacks the declared linear/pre/weighted methods')
    return dict(source=str(Path(path).resolve()), descriptive_only=True, scene_count=len(ids),
        helper_population=populations, position_error_weightings=metrics,
        unavailable=['V per-anchor training-style loss coefficients', 'identity-by-support joint counts',
                     'causal explanation of T/V differences'])
