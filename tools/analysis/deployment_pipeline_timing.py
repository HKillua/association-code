"""Application-boundary timing from received measurement CSV to response bytes.

Engineering replay, not accuracy validation or an optimized streaming service.
No label/reference payload is opened. Online buffers contain only timestamps
at or before their declared arrival cutoff. Existing trained models stay frozen.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import resource
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.data.archive import MTADArchive, HEADERS
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from association.data.motion_evidence import DATA, scene_motion_evidence
from association.evaluation import history_candidates as history
from association.evaluation import source_risk as risk
from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.local_source_evidence import ResidualScore, predict_residual
from association.evaluation.native_temporal import predict as quadratic_predict, spherical_position
from association.models.association_context import EdgeResidual, edge_inputs, normalize, context_cache
from association.models.evidence_association import features_from_motion
from association.models.ptfm import PTFM
from association.models.deepaf_peer import DeepAFPeer
from association.training.evidence_association import load_checkpoint, normalized_features
from tools.analysis.static_partition_pilot import agglomerate, THRESHOLD
from tools.analysis.source_risk_pilot import MOTION_KEYS
from tools.experiments.fusion_peer_full_fit import predict_group

SCENES = ('692', '3981', '1187')  # fixed small, median-sized, larger original V32 scenes
METHODS = ('WS_PTFM', 'WS_DEEPAF', 'HARD_O178', 'STRUCT_O178')


def read(path):
    return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def csv_rows(payload):
    try:
        text = payload.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = payload.decode('gb18030')
    reader = csv.DictReader(io.StringIO(text, newline=''), strict=True)
    if reader.fieldnames != HEADERS['measurements']:
        raise ValueError('measurement header differs')
    rows = []
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('wrong measurement column count')
        rows.append({k: v.strip() for k, v in row.items()})
    return rows


class ReceivedMeasurements(MTADArchive):
    """Reuse canonical measurement checks with anonymous, per-node metadata.

    The legacy loader requires label rows to describe fragments. This adapter
    derives the same source/batch boundaries from sensor messages themselves;
    its distinct anonymous IDs have no cross-node identity information.
    """
    def __init__(self, payload):
        self.rows = csv_rows(payload)

    def _rows(self, scene_id, kind):
        if kind != 'measurements':
            raise AssertionError('label/reference payload access forbidden')
        return self.rows

    def read_labels(self, scene_id):
        bounds = {}
        for row in self.rows:
            key = (row['source'], row['batch'])
            stamp = float(row['time'])
            low, high = bounds.get(key, (stamp, stamp))
            bounds[key] = (min(low, stamp), max(high, stamp))
        return [dict(source=s, batch=b, t_s=bounds[s, b][0], t_e=bounds[s, b][1],
                     mmsi=f'{i}-anonymous-0')
                for i, (s, b) in enumerate(sorted(bounds))]


def preprocess(payload, sid, component, cutoff):
    source = ReceivedMeasurements(payload)
    stamps = np.asarray([float(r['time']) for r in source.rows])
    if not len(stamps) or (stamps > cutoff).any():
        raise AssertionError('future observation entered the received buffer')
    native = _record(source.load_scene(sid), component, 'V_select', 128)
    if len(native['tracks']) < 2:
        raise ValueError('at least two valid received fragments required')
    prepared = prepare_timed_scene(native, **DATA)
    motion = scene_motion_evidence(prepared, motion_observation_policy='canonical_all')
    motion['node_ids'] = np.asarray(prepared['node_ids'], str)
    if motion['query_seconds'][-1] > cutoff:
        raise AssertionError('output grid extends beyond arrival cutoff')
    return motion, native['pairs'], dict(
        input_bytes=len(payload), received_rows=len(stamps),
        earliest_timestamp=float(stamps.min()), latest_timestamp=float(stamps.max()),
        arrival_cutoff=cutoff, future_observation_rows=0,
        valid_nodes=len(native['tracks']), quarantined_nodes=native['audit']['quarantined_nodes'],
        query_points=64, query_start=float(motion['query_seconds'][0]),
        query_end=float(motion['query_seconds'][-1]))


class ResidentRisk:
    def __init__(self, model):
        self.model = model
        self.net = ResidualScore(model['dimension'])
        self.net.load_state_dict(model['weights'])
        self.net.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x):
        model = self.model
        z = torch.as_tensor(np.clip((x-model['mean'])/model['std'], -8, 8), dtype=torch.float32)
        return quadratic_predict(model['base'], x)+self.net(z).numpy()


def only_O(motion, groups, count, scores, base):
    output = base.copy()
    for gid, members in enumerate(groups):
        for j in np.flatnonzero(count[gid] <= 1):
            selected = members[scores[members, j] == scores[members, j].min()]
            mass = history.source_balanced(selected, motion['sources'])
            output[gid, j, :2] = spherical_position(motion['states'][selected, j, :2], mass)
    return output


class Pipeline:
    def __init__(self, root):
        self.root = root
        self.teacher, ck = load_checkpoint(root/'LEVEL1-ASSOCIATION-20261022-056-attempt-1/association/final.pt', torch.device('cpu'))
        self.normalizer = ck['normalizer']
        self.teacher.eval().requires_grad_(False)
        parent = root/'LEVEL1-HISTORY-CANDIDATES-20261031-178-attempt-1'
        self.parent = read(parent/'run.json')
        h175 = torch.load(Path(self.parent['parent_run'])/'private/heads.pt', map_location='cpu', weights_only=False)
        r175 = read(Path(self.parent['parent_run'])/'run.json')
        h173 = torch.load(Path(r175['parent_run'])/'private/heads.pt', map_location='cpu', weights_only=False)
        self.context_norm = h175['normalizer']
        self.heads = {}
        for name, weights in [('HARD', h175['heads']['HARD']), ('STRUCT', h173['heads']['STRUCT'])]:
            net = EdgeResidual(); net.load_state_dict(weights)
            self.heads[name] = net.eval().requires_grad_(False)
        r159 = torch.load(Path(r175['expression_run'])/'private/model-full.pt', map_location='cpu', weights_only=False)
        self.baseline = ResidentRisk(r159['netold'])
        self.history = ResidentRisk(torch.load(parent/'private/model-full.pt', map_location='cpu', weights_only=False))
        self.peers = {}
        for method, name in [('ptfm', 'PTFM-20261030-152A'), ('deepaf', 'DEEPAF-20261030-152B')]:
            p = root/('LEVEL1-FUSION-PEER-FULL-'+name+'-attempt-1')
            if read(p/'run.json')['status'] != 'COMPLETED':
                raise ValueError('completed peer endpoint required')
            ck = torch.load(p/'private/last-epoch.pt', map_location='cpu', weights_only=False)
            model = PTFM(ck['normalizer'], sensor_branches=True) if method == 'ptfm' else DeepAFPeer()
            model.load_state_dict(ck['model']); self.peers[method] = model.eval().requires_grad_(False)

    def base_scores(self, motion, pairs, need_context):
        scores = np.empty(len(pairs), np.float64)
        hidden = np.empty((len(pairs), 64), np.float32) if need_context else None
        for start in range(0, len(pairs), 4096):
            stop = min(start+4096, len(pairs))
            f, b = features_from_motion(motion, pairs[start:stop])
            x = torch.from_numpy(normalized_features(f.astype(np.float32), self.normalizer))
            if need_context:
                hidden[start:stop] = self.teacher.network[:-1](x).numpy()
            scores[start:stop] = self.teacher(x, torch.as_tensor(b, dtype=torch.float64)).numpy()
        return scores, hidden

    @torch.inference_mode()
    def call(self, payload, sid, component, cutoff, method, *, verify=False):
        stage = {}; began = last = time.perf_counter()
        def mark(name):
            nonlocal last
            now = time.perf_counter(); stage[name] = now-last; last = now
        m, pairs, audit = preprocess(payload, sid, component, cutoff)
        mark('decode_clean_timed_motion')
        ours = method.endswith('O178')
        scores, hidden = self.base_scores(m, pairs, ours)
        mark('pair_features_and_base_association')
        if ours:
            front = method.split('_')[0]
            cache = context_cache(hidden, pairs, scores, len(m['states']), THRESHOLD)
            corrected = scores.copy()
            for start in range(0, len(pairs), 4096):
                ix = np.arange(start, min(start+4096, len(pairs)))
                x = normalize(edge_inputs(cache, ix, 'CTX'), self.context_norm)
                corrected[ix] += self.heads[front](torch.as_tensor(x)).numpy().astype(float)
            scores = corrected+THRESHOLD-self.parent['thresholds'][front]
        mark('context_and_residual_association')
        membership, _, _ = agglomerate(len(m['states']), pairs, scores, 'WS')
        groups = [np.flatnonzero(membership == k) for k in np.unique(membership)]
        mark('grouping')
        if ours:
            mean = ordinary_fusion(m['states'], m['nearest_seconds'], m['support'], groups)['predictions']['mean_supported'][[g[0] for g in groups]]
            pack = risk.build_source_features(**{k:m[k] for k in MOTION_KEYS}, groups=groups)
            s = self.baseline(pack['features'].reshape(-1, 10)).reshape(pack['available'].shape)
            base = risk.recover_source_policy(states=m['states'], support=m['support'], groups=groups,
                pack=pack, scores=s, kind='hard', mean_states=mean)['group_states']
            mark('source_features_and_base_recovery159')
            x, count = history.descriptors(m, groups)
            sc = self.history(x.reshape(-1, 12)).reshape(m['support'].shape)
            output = only_O(m, groups, count, sc, base)
            mark('history_features_score_and_recovery178')
            if verify:
                np.testing.assert_array_equal(self.history(x.reshape(-1, 12)), predict_residual(self.history.model, x.reshape(-1, 12), 'cpu'))
                np.testing.assert_array_equal(output, history.readouts(m, groups, count, sc, base)[0]['O'])
            output[..., 2] *= 1852/3600
        else:
            m = dict(m, states=m['states'].copy()); m['states'][..., 2] *= 1852/3600
            kind = 'ptfm' if method == 'WS_PTFM' else 'deepaf'
            output = np.stack([predict_group(self.peers[kind], m, g, kind, 'cpu')[0] for g in groups])
            mark('fusion_input_and_peer_recovery')
        emitted = emit_group_outputs(output, groups, node_count=len(m['states']))
        response = json.dumps(dict(query_seconds=m['query_seconds'].tolist(), speed_unit='mps',
            groups=[dict(node_ids=m['node_ids'][row['node_indices']].tolist(), states=row['states'].tolist())
                    for row in emitted]), allow_nan=False, separators=(',', ':')).encode()
        mark('final_output_serialization')
        # Verification runs are deliberately excluded from measured observations.
        result = dict(method=method, seconds=time.perf_counter()-began, stages_seconds=stage,
                      group_count=len(groups), response_bytes=len(response), **audit)
        return result, m, membership, output


def run(args):
    out = args.output; out.mkdir(exist_ok=False); (out/'private').mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', engineering_only=True, trained_model_fits=0, optimizer_updates=0,
                 native_OLS_recomputed_in_preprocessing=True,
                 real_labels_opened=False, reference_opened=False, test_opened=False,
                 planned_scenes=list(SCENES), methods=list(METHODS), completed_calls=0,
                 online_update_interval_seconds=1., time_policy='advisory')
    rows = []
    def save(**kw):
        state.update(kw, elapsed_seconds=time.monotonic()-started,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (out/'run.json').write_text(json.dumps(state, indent=2)+'\n')
        (out/'measurements.json').write_text(json.dumps(rows, indent=2)+'\n')
        if state['peak_rss_bytes'] > 4*1024**3:
            raise MemoryError('RSS4GiB budget')
    try:
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        root = args.root; pipeline = Pipeline(root)
        meta = read(root/'LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1/run.json')
        manifest = read(pipeline.parent['manifest'])
        save(phase='read_sensor_payloads')
        with MTADArchive(meta['archive_paths']) as archive:
            for sid in SCENES:
                if sid not in pipeline.parent['scene_ids']:
                    raise ValueError('fixed scene outside original V32')
                info, _ = archive._members['measurements'][sid]
                payload = archive._zip.read(info)  # transport replay only; no labels/reference
                raw = csv_rows(payload); stamps = np.asarray([float(x['time']) for x in raw])
                lo, hi = float(stamps.min()), float(stamps.max()); middle = lo+(hi-lo)/2
                cases = [('full_received_window', hi), ('online_t', middle), ('online_t_plus_1s', middle+1.)]
                for case, cutoff in cases:
                    if case == 'full_received_window':
                        received = payload
                    else:
                        text = io.StringIO(newline=''); writer = csv.DictWriter(text, fieldnames=HEADERS['measurements'])
                        writer.writeheader(); writer.writerows(x for x in raw if float(x['time']) <= cutoff)
                        received = text.getvalue().encode()
                    for method in METHODS:
                        save(phase='warmup_and_binding', scene_id=sid, case=case, method=method)
                        warm, m, membership, output = pipeline.call(received, sid, manifest['components'][sid], cutoff, method, verify=True)
                        if case == 'full_received_window':
                            old = load_npz(root/'LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1/private/native-motion/V_select'/f'scene-{sid}.npz')
                            np.testing.assert_array_equal(m['node_ids'], old['node_ids'])
                            for key in ('support', 'nearest_seconds', 'states', 'ols_velocity_mps'):
                                actual = m[key].copy()
                                if method.startswith('WS_') and key == 'states': actual[..., 2] /= 1852/3600
                                np.testing.assert_allclose(actual, old[key], rtol=0, atol=1e-10)
                            if method.endswith('O178'):
                                saved = load_npz(root/'LEVEL1-HISTORY-CANDIDATES-20261031-178-attempt-1/private'/f'partitions-V_select-{sid}.npz')
                                np.testing.assert_array_equal(membership, saved[method.split('_')[0]])
                        for repeat in range(2):
                            row, _, member, result = pipeline.call(received, sid, manifest['components'][sid], cutoff, method)
                            row.update(scene_id=sid, case=case, repeat=repeat, software_binding_passed=True)
                            rows.append(row)
                            np.savez_compressed(out/'private'/f'{sid}-{case}-{method}-{repeat}.npz', membership=member, states=result)
                            save(phase='timing', completed_calls=len(rows))
        save(status='COMPLETED', exit_code=0, phase='completed',
             finished_at=datetime.now(timezone.utc).isoformat(), planned_timed_calls=72)
    except BaseException as error:
        save(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    run(parser.parse_args())
