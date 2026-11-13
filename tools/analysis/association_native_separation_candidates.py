"""C141: compare three budget-matched ways to seed cuts in frozen MTAD groups."""

import argparse
from collections import Counter, defaultdict, deque
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np

from association.data.archive import MTADArchive
from association.evaluation.mixture_opportunity import _distance
from tools.analysis import association_partition_disagreement as c98
from tools.analysis import offline_association_tail as tail
from tools.analysis.association_partition_fit_qualification import write_gzip
from tools.analysis.held_fusion_population import _canonical_tracks, validate_measurement_archive

C139 = 'LEVEL1-OFFLINE-PARTITION-FIT-QUALIFICATION-20261114-attempt-1'
ARMS = ('SCORE', 'OVERLAP', 'NATIVE')
MODES = ('budget', 'matched')
SLOTS = 2
MATCH_S = 10.
MIN_SEPARATION_S = 30.
MAX_SEPARATION_S = 120.
SPEED_MPS = 50.
ERROR_PER_ROW_M = 250.
SENSITIVITY = (100., 500.)


def read(path):
    return tail.audit.read_json(path)


def position_signature(candidate, ws):
    nodes = candidate['nodes']
    original = np.asarray(ws)[nodes]
    groups = c98.group_nodes(candidate['partition'])
    changed = []
    for group in groups:
        members = tuple(nodes[i] for i in group)
        same_old = (original[group] == original[group[0]]).all()
        if not same_old or len(group) != (original == original[group[0]]).sum():
            changed.append(members)
    return tuple(changed)


def separated(time, margin):
    """Return the largest minimum margin over two disjoint near-time matches."""
    positive = np.flatnonzero(margin > 0.)
    pending = 0
    maxima = deque()
    best = None
    for current in positive:
        while pending < len(positive) and time[positive[pending]] <= time[current]-MIN_SEPARATION_S:
            prior = positive[pending]
            while maxima and margin[maxima[-1]] < margin[prior]:
                maxima.pop()
            maxima.append(prior)
            pending += 1
        while maxima and time[maxima[0]] < time[current]-MAX_SEPARATION_S:
            maxima.popleft()
        if maxima:
            prior = maxima[0]
            value = float(min(margin[prior], margin[current]))
            if best is None or value > best[0]:
                best = value, int(prior), int(current)
    return best


def pair_evidence(a, b):
    times_a, pos_a = a['canonical_times'], a['canonical_values'][:, :2]
    times_b, pos_b = b['canonical_times'], b['canonical_values'][:, :2]
    start, end = max(times_a[0], times_b[0]), min(times_a[-1], times_b[-1])
    if end <= start:
        return None, None, False
    ai = np.flatnonzero((times_a >= start) & (times_a <= end))
    lo = np.searchsorted(times_b, np.maximum(start, times_a[ai]-MATCH_S), side='left')
    hi = np.searchsorted(times_b, np.minimum(end, times_a[ai]+MATCH_S), side='right')
    count = hi-lo
    i = np.repeat(ai, count)
    j = np.repeat(lo-np.cumsum(count)+count, count)+np.arange(count.sum())
    delta = np.abs(times_a[i]-times_b[j])
    if np.any((delta == 0.) & (pos_a[i] == pos_b[j]).all(axis=1)):
        return float(end-start), None, True
    if len(i) < 2:
        return float(end-start), None, False
    order = np.argsort((times_a[i]+times_b[j])/2., kind='stable')
    i, j, delta = i[order], j[order], delta[order]
    time = (times_a[i]+times_b[j])/2.
    distance = _distance(pos_a[i], pos_b[j])
    main = separated(time, distance-SPEED_MPS*delta-2.*ERROR_PER_ROW_M)
    stress = separated(time, distance-SENSITIVITY[0]*delta-2.*SENSITIVITY[1])
    if main is None:
        return float(end-start), None, False
    score, x, y = main
    pairs = [[int(i[k]), int(j[k]), float(delta[k]), float(distance[k]), float(time[k])] for k in (x, y)]
    assert i[x] != i[y] and j[x] != j[y]
    return float(end-start), dict(score=score, pairs=pairs, stringent=stress is not None), False


def cut_candidates(nodes, heads, margins, seeds, counters):
    """Minimize nonnegative frozen pair capacity with the original heads retained."""
    weights = np.maximum(0., margins[np.ix_(nodes, nodes)])
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)+2))
    left, right = len(nodes), len(nodes)+1
    i, j = np.nonzero(np.triu(weights, 1))
    graph.add_weighted_edges_from(zip(i.tolist(), j.tolist(), weights[i, j].tolist()), weight='capacity')
    hard = float(weights.sum()+1.)
    node_index = {node: i for i, node in enumerate(nodes)}
    head_limit = {i for i, node in enumerate(nodes) if len(heads) == 2 and node < heads[1]}
    for pair in seeds:
        a, b = (node_index[k] for k in pair)
        best = None
        for first, second in (((a, b),) if len(heads) == 1 else ((a, b), (b, a))):
            left_nodes, right_nodes = {first}, {second}
            if len(heads) == 2:
                left_nodes |= head_limit
                right_nodes.add(node_index[heads[1]])
            if left_nodes & right_nodes:
                continue
            edges = [(left, i) for i in sorted(left_nodes)]+[(right, j) for j in sorted(right_nodes)]
            graph.add_edges_from(edges, capacity=hard)
            value, (side, other) = nx.minimum_cut(graph, left, right, flow_func=nx.algorithms.flow.edmonds_karp)
            graph.remove_edges_from(edges)
            assert left_nodes <= side and right_nodes <= other
            labels = np.array([int(i in other) for i in range(len(nodes))])
            calculated = weights[np.triu(labels[:, None] != labels[None, :], 1)].sum()
            np.testing.assert_allclose(value, calculated, atol=1e-9, rtol=1e-12)
            if len(nodes) <= 8 and counters['exhaustive_cut_checks'] < 64:
                feasible = []
                free = sorted(set(range(len(nodes)))-left_nodes-right_nodes)
                for mask in range(1 << len(free)):
                    selected = left_nodes | {u for k, u in enumerate(free) if mask & (1 << k)}
                    rest = sorted(set(range(len(nodes)))-selected)
                    feasible.append(weights[np.ix_(sorted(selected), rest)].sum())
                np.testing.assert_allclose(value, min(feasible), atol=1e-9, rtol=1e-12)
                counters['exhaustive_cut_checks'] += 1
            candidate = (float(value), tuple(c98.frozen.canonical_membership(labels, len(nodes))))
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            yield pair, best


def make_candidate(nodes, labels, ws, motion, margins):
    a, b = np.triu_indices(len(nodes), 1)
    old = ws[nodes]
    labels = np.asarray(labels)
    before = old[a] == old[b]
    after = labels[a] == labels[b]
    la, lb = np.asarray(nodes)[a], np.asarray(nodes)[b]
    cut = before & ~after
    add = ~before & after
    if not (cut.any() or add.any()):
        return None
    support = motion['support']
    heads = [nodes[g[0]] for g in c98.group_nodes(old)]
    new_heads = [nodes[g[0]] for g in c98.group_nodes(labels)]
    assert all(h in new_heads for h in heads)
    return dict(nodes=list(nodes), partition=labels.tolist(), old_gain=float(
        margins[la[add], lb[add]].sum()-margins[la[cut], lb[cut]].sum()),
        supplier_unsafe_edges=int((cut & (support[la] != support[lb]).any(axis=1)).sum()),
        anchor_exact=heads == new_heads)


def seed_rows(nodes, source, margins, tracks, cache, stats):
    ranked = {arm: [] for arm in ARMS}
    for a, b in combinations(nodes, 2):
        if source[a] != source[b]:
            continue
        pair = (a, b)
        ranked['SCORE'].append((-float(margins[a,b]), pair))
        if pair not in cache:
            cache[pair] = pair_evidence(tracks[a], tracks[b])
            overlap, evidence, duplicate = cache[pair]
            stats['compared_unique_same_source_pairs'] += 1
            stats['overlapping_unique_pairs'] += int(overlap is not None)
            stats['exact_shared_report_pairs'] += int(duplicate)
            stats['native_unique_pairs'] += int(evidence is not None)
            stats['stringent_unique_pairs'] += int(evidence is not None and evidence['stringent'])
        overlap, evidence, _ = cache[pair]
        if overlap is not None:
            ranked['OVERLAP'].append((overlap, pair))
        if evidence is not None:
            ranked['NATIVE'].append((evidence['score'], pair))
    return {arm: [p for _, p in sorted(ranked[arm], key=lambda x: (-x[0],x[1]))[:SLOTS]] for arm in ARMS}


def actual_choice(candidates, indexes):
    winner, gain = -1, 0.
    for i in indexes:
        c = candidates[i]
        if c['anchor_exact'] and c['supplier_unsafe_edges'] == 0 and c['old_gain'] > gain+c98.EPS:
            winner, gain = i, c['old_gain']
    return winner


def materialize(ws, blocks, ids):
    output = ws.copy()
    offset = int(ws.max())+1
    for block, i in zip(blocks, ids, strict=True):
        if i >= 0:
            candidate = block['candidates'][i]
            p = np.asarray(candidate['partition'])
            output[candidate['nodes']] = offset+p
            offset += int(p.max())+1
    return c98.frozen.canonical_membership(output, len(ws))


def freeze_scene(args, sid, motion, archive, counters, thresholds):
    parent = read(args.runs/C139/'private/blind'/f'{sid}.json.gz')
    ws = np.asarray(parent['ws'])
    n = len(ws)
    scores = np.load(args.runs/c98.PREP/'private'/f'scores-{sid}.npy', allow_pickle=False)
    a,b = np.triu_indices(n,1)
    c98.require(scores.shape == a.shape and np.isfinite(scores).all(), '056 population changed')
    margins = np.zeros((n,n))
    margins[a,b] = margins[b,a] = scores-thresholds['056']
    checked_ws, _, _ = tail.ws.agglomerate(n, np.column_stack((a,b)), scores, 'WS')
    np.testing.assert_array_equal(ws, checked_ws)
    sources = motion['sources']
    c98.require(n == len(sources) == len(motion['node_ids']) == len(motion['native_count'])
                and motion['support'].shape == (n,64), 'motion shape changed')
    heads = {int(ws[g[0]]): int(g[0]) for g in c98.group_nodes(ws)}
    members = {int(ws[g[0]]): g.tolist() for g in c98.group_nodes(ws)}
    scopes = []
    groups_in_blocks = set()
    for block in parent['blocks']:
        keys = sorted({ws[i] for i in block['nodes']}, key=heads.get)
        scopes.append((block, [(k,) for k in keys]+list(combinations(keys,2))))
        c98.require(not (groups_in_blocks & set(keys)), 'C139 old groups overlap blocks')
        groups_in_blocks.update(keys)
    scopes.extend((None, [(k,)]) for k in members if k not in groups_in_blocks)
    stats = Counter()
    eligible = any(any(len(set(sources[sorted(i for k in scope for i in members[k])])) <
                       sum(len(members[k]) for k in scope) for scope in group_scopes)
                   for _, group_scopes in scopes)
    tracks = None
    if eligible:
        tracks = _canonical_tracks(archive._rows(sid, 'measurements'), sid, motion['node_ids'], sources)
        np.testing.assert_array_equal([len(t['canonical_times']) for t in tracks], motion['native_count'])
    cache = {}
    blocks = []
    for parent_block, group_scopes in scopes:
        base = [] if parent_block is None else [dict(c, nodes=parent_block['nodes'])
                                                for c in parent_block['candidates'][1:]]
        base_keys = {position_signature(c,ws) for c in base}
        candidates = list(base)
        index = {position_signature(c,ws): i for i,c in enumerate(base)}
        proposed = {mode: {arm: [] for arm in ARMS} for mode in MODES}
        witnesses = []
        scope_rows = []
        for scope in group_scopes:
            scope_type = 'single' if len(scope) == 1 else 'pair'
            stats[f'{scope_type}_scopes'] += 1
            nodes = sorted(i for k in scope for i in members[k])
            if len(set(sources[nodes])) == len(nodes):
                continue
            stats[f'eligible_{scope_type}_scopes'] += 1
            seeds = seed_rows(nodes,sources,margins,tracks,cache,stats)
            cuts = dict(cut_candidates(nodes,[heads[k] for k in scope],margins,
                                        sorted({p for values in seeds.values() for p in values}),counters))
            by_arm = {arm: [] for arm in ARMS}
            for arm in ARMS:
                stats[f'{arm}_seeded_{scope_type}'] += len(seeds[arm])
                for pair in seeds[arm]:
                    if pair not in cuts:
                        stats[f'{arm}_infeasible_cut'] += 1
                        continue
                    capacity, labels = cuts[pair]
                    c = make_candidate(nodes,labels,ws,motion,margins)
                    if c is None:
                        stats[f'{arm}_unchanged_cut'] += 1
                        continue
                    key = position_signature(c,ws)
                    if key in base_keys:
                        stats[f'{arm}_legacy_cut'] += 1
                        continue
                    if key not in index:
                        index[key] = len(candidates)
                        candidates.append(c)
                    i = index[key]
                    if i not in by_arm[arm]:
                        by_arm[arm].append(i)
                        witnesses.append(dict(arm=arm, candidate=i, seed=list(pair), cut_capacity=capacity,
                                              native=cache[pair][1] if arm == 'NATIVE' else None))
            matched = min(map(len,by_arm.values()))
            stats['matched_candidate_slots'] += matched
            stats['matched_active_scopes'] += int(matched > 0)
            if any(seeds.values()):
                scope_rows.append(dict(nodes=nodes, heads=[heads[k] for k in scope],
                                       seeds={k:[list(pair) for pair in v] for k,v in seeds.items()},
                                       novel=by_arm, matched=matched))
            for arm in ARMS:
                proposed['budget'][arm].extend(by_arm[arm])
                proposed['matched'][arm].extend(by_arm[arm][:matched])
        if not candidates and not scope_rows:
            continue
        base_ids = list(range(len(base)))
        pools = {'C139': base_ids}
        for mode,arms in proposed.items():
            unique = {arm:list(dict.fromkeys(indexes)) for arm,indexes in arms.items()}
            if mode == 'matched':
                count = min(map(len,unique.values()))
                stats['matched_unique_candidates_per_arm'] += count
                unique = {arm:indexes[:count] for arm,indexes in unique.items()}
            for arm,indexes in unique.items():
                pools[f'{mode}_{arm}'] = base_ids+indexes
        chosen = {pool: actual_choice(candidates,indexes) for pool,indexes in pools.items()}
        blocks.append(dict(candidates=candidates,pools=pools,chosen=chosen,witnesses=witnesses,scopes=scope_rows))
    keys = ('C139', *(f'{mode}_{arm}' for mode in MODES for arm in ARMS))
    groups = {key: materialize(ws,blocks,[b['chosen'][key] for b in blocks]).tolist() for key in keys}
    original_blocks = [block['nodes'] for block in parent['blocks']]
    return dict(ws=ws.tolist(),blocks=blocks,actual=groups,counts=dict(stats),original_blocks=original_blocks)


def evaluate_candidates(ws, candidates, identities, motion, baseline_head_identities):
    by_scope = defaultdict(list)
    for i, candidate in enumerate(candidates):
        by_scope[tuple(candidate['nodes'])].append(i)
    labels = {}
    for nodes, indexes in by_scope.items():
        rows = [candidates[i] for i in indexes]
        selected = c98.label_block(dict(nodes=nodes, candidates=rows), ws, identities,
                                   motion['support'], motion['sources'])
        old = ws[list(nodes)]
        old_heads = {nodes[g[0]] for g in c98.group_nodes(old)}
        for i, candidate, label in zip(indexes, rows, selected, strict=True):
            new_heads = {nodes[g[0]] for g in c98.group_nodes(candidate['partition'])}
            extra = [identities[j] for j in new_heads-old_heads]
            duplicates = len(extra)-len(set(extra)-baseline_head_identities)
            delta = label['delta']
            nonworsening = all(v <= 0 for row in delta.values() for v in row.values())
            protected = (old_heads <= new_heads and not duplicates and not label['same_ship_lost'])
            strict = candidate['anchor_exact']
            dual = protected and nonworsening and all(v < 0 for v in delta['mstsa'].values())
            only_fp = protected and nonworsening and delta['mstsa']['fn'] == 0 and delta['mstsa']['fp'] < 0
            labels[i] = dict(label, dual=bool(dual), strict_dual=bool(strict and dual),
                             only_fp=bool(only_fp), new_duplicate_heads=duplicates,
                             extra_identities=set(extra),
                             safe_positive=bool(dual and candidate['old_gain'] > c98.EPS),
                             blind_eligible=bool(dual and strict and candidate['supplier_unsafe_edges'] == 0))
    return labels


def evaluate_scene(frozen, motion, truth):
    ws = np.asarray(frozen['ws'])
    blocks = frozen['blocks']
    original_heads = [g[0] for g in c98.group_nodes(ws)]
    head_ids = set(truth[original_heads])
    a,b = np.triu_indices(len(ws),1)
    target = truth[a] == truth[b]
    same = motion['sources'][a] == motion['sources'][b]
    before = c98.edge_rows(target,ws[a] == ws[b],same)
    original_block = np.full(len(ws),-1)
    for j,nodes in enumerate(frozen['original_blocks']):
        original_block[nodes] = j
    scope = (original_block[a] >= 0) & (original_block[a] == original_block[b])
    missed = target & ~same & (ws[a] != ws[b])
    reachability = dict(cross_fn_inside_c139_scopes=int((missed & scope).sum()),
                        cross_fn_outside_c139_scopes=int((missed & ~scope).sum()))
    stats = {k:Counter() for k in frozen['actual']}
    choices = {k:[] for k in stats}
    new_head_ids = {k:set() for k in stats}
    safe_pairs = []
    for block in blocks:
        candidates = block['candidates']
        labels = evaluate_candidates(ws,candidates,truth,motion,head_ids)
        common = block['pools']['C139']
        signature = {name: tuple(inds) for name, inds in block['pools'].items()}
        for w in block['witnesses']:
            a0,b0 = w['seed']
            stats[f'budget_{w["arm"]}']['witness_seed_uses'] += 1
            stats[f'budget_{w["arm"]}']['same_ship_witness_seed_uses'] += int(truth[a0] == truth[b0])
        for name, indexes in signature.items():
            pool = stats[name]
            new = indexes[len(common):] if name != 'C139' else []
            label, chosen = labels.get(block['chosen'][name]), block['chosen'][name]
            safe = [i for i in indexes if labels[i]['dual']]
            positive = [i for i in safe if labels[i]['safe_positive']]
            eligible = [i for i in positive if labels[i]['blind_eligible']]
            selected_safe = bool(label is not None and label['dual'])
            original_safe = any(labels[i]['dual'] for i in common)
            jointly_safe = [i for i in safe if not labels[i]['extra_identities'] & new_head_ids[name]]
            best = max(jointly_safe,key=lambda i: labels[i]['value'],default=-1)
            if best >= 0:
                new_head_ids[name].update(labels[best]['extra_identities'])
            choices[name].append(best)
            pool.update(blocks=1, candidates=len(indexes), new_candidates=len(new),
                        safe_candidates=len(safe), new_safe_candidates=sum(labels[i]['dual'] for i in new),
                        strict_safe_candidates=sum(labels[i]['strict_dual'] for i in safe),
                        safe_blocks=int(bool(safe)), new_safe_blocks=int(bool(safe) and not original_safe),
                        protected_false_only_candidates=sum(labels[i]['only_fp'] for i in indexes),
                        selected=int(chosen >= 0), selected_safe=int(selected_safe))
            if not safe:
                pool['blocked_no_safe'] += 1
            elif not selected_safe:
                reason = 'nonpositive_j' if not positive else 'protection' if not eligible else 'higher_j'
                pool[f'blocked_{reason}'] += 1
        safe_pairs.append([int(any(labels[i]['dual'] for i in inds)) for inds in signature.values()])
    results = {}
    for arm, selected in frozen['actual'].items():
        actual = np.asarray(selected)
        oracle = materialize(ws,blocks,choices[arm])
        outcomes = {}
        for label, group in [('actual',actual),('oracle',oracle)]:
            edges = c98.edge_rows(target,group[a] == group[b],same)
            cov = tail.coverage(ws,group,truth,motion['support'])
            c98.require(not (cov['identities_lost'] or cov['duplicate_increase_scenes'] or cov['same_ship_lost']),
                        f'{arm} {label} failed protected full-scene coverage')
            outcomes[label] = dict(
                decoded=edges, coverage=cov, changed_groups=int(not np.array_equal(ws,group)),
                delta={scope:{key: edges[scope][key]-before[scope][key] for key in ('fn','fp')}
                       for scope in c98.SCOPES})
        results[arm] = dict(counts=dict(stats[arm]), **outcomes)
    outcome_key = hashlib.sha256(json.dumps(safe_pairs,separators=(',',':')).encode()).hexdigest()
    return dict(counts=frozen['counts'], baseline=before, arms=results, reachability=reachability,
                safe_block_pattern_sha256=outcome_key)


def population(rows):
    freeze = sum((Counter(r['counts']) for r in rows),Counter())
    arms = {}
    for arm in rows[0]['arms']:
        arm_rows = [r['arms'][arm] for r in rows]
        counts = sum((Counter(r['counts']) for r in arm_rows),Counter())
        metrics = {}
        for name in ('actual','oracle'):
            coverage = Counter()
            delta = {scope:Counter() for scope in c98.SCOPES}
            confusion = {scope:Counter() for scope in c98.SCOPES}
            for row in arm_rows:
                coverage.update(row[name]['coverage'])
                for scope in c98.SCOPES:
                    delta[scope].update(row[name]['delta'][scope])
                    values = row[name]['decoded'][scope]
                    confusion[scope].update({key:values[key] for key in
                                            ('tp','fp','fn','tn','pair_count','positive_count')})
            metrics[name] = dict(
                changed_scenes=sum(r[name]['changed_groups'] for r in arm_rows),
                coverage=dict(coverage), delta={k:dict(v) for k,v in delta.items()},
                decoded={k:dict(v) for k,v in confusion.items()},
                macro={scope:c98._aggregate([r[name]['decoded'][scope] for r in arm_rows])
                       for scope in c98.SCOPES})
        arms[arm] = dict(counts=dict(counts),**metrics)
    reachability = sum((Counter(r['reachability']) for r in rows),Counter())
    return dict(scenes=len(rows),components=len({r['component'] for r in rows}),counts=dict(freeze),
                arms=arms,reachability=dict(reachability))


def gates(full, remaining, components, largest):
    def counts(pop, name):
        return pop['arms'][name]['counts'].get('safe_blocks',0)
    result = {}
    for arm in ARMS[:2]:
        native, control = 'matched_NATIVE', f'matched_{arm}'
        delta = counts(full,native)-counts(full,control)
        rem = counts(remaining,native)-counts(remaining,control)
        positive = [c for c,p in components.items() if counts(p,native)>counts(p,control)]
        result[arm] = dict(additional_safe_blocks=delta, positive_components=len(positive),
                           without_largest=rem, positive_without_largest=len(set(positive)-{largest}),
                           passes=delta>=10 and len(positive)>=3 and rem>=5 and len(set(positive)-{largest})>=2)
    return result


def run(args, start):
    output = args.output
    old = args.runs/C139
    data = args.runs/c98.DATA
    prep = args.runs/c98.PREP
    roles = read(prep/'frozen-split.json')
    ids = roles['fit']
    manifest = read(args.manifest)
    parent = read(old/'freeze.json')
    components = {sid:manifest['components'][sid] for sid in ids}
    index = read(data/'private/index.json')['T_core']
    require = c98.require
    require(all(read(p/'run.json')['status'] == 'COMPLETED' for p in (old,data,prep)), 'parent incomplete')
    require(roles == tail.scene_split(manifest['splits']['T_core'],manifest['components']), 'C92 roles changed')
    require(len(ids)==len(set(ids))==parent['scene_count']==1748, 'fit scenes changed')
    require(len(set(components.values()))==parent['component_count']==76, 'fit components changed')
    require(parent['scene_order_sha256'] == hashlib.sha256('\n'.join(ids).encode()).hexdigest(), 'fit order changed')
    require(all(index[sid]['component_id'] == components[sid] for sid in ids), 'fit component binding changed')
    private = output/'private'
    private.mkdir()
    counters = Counter()
    with MTADArchive(read(data/'run.json')['archive']) as archive:
        validate_measurement_archive(args.runs,archive)
        for i,sid in enumerate(ids,1):
            m = tail.read_motion(index[sid])
            frozen = freeze_scene(args,sid,m,archive,counters,parent['thresholds'])
            write_gzip(private/f'{sid}.json.gz',frozen)
            if i % 50 == 0:
                tail.summary_state(output,start,status='RUNNING',phase='freeze',scenes=i,**counters)
        require(all(r['kind']=='measurements' and r['scene_id'] in components for r in archive.members_read),
                'archive role changed')
        archive_reads = len(archive.members_read)
    require(len(list(private.iterdir()))==len(ids), 'incomplete frozen artifacts')
    tail.write(output/'freeze.json',dict(
        scenes=len(ids), components=len(set(components.values())), raw_measurement_members=archive_reads,
        c139_scene_order_sha256=parent['scene_order_sha256'], no_identity_before_freeze=True,
        cutoff=dict(slots=SLOTS, near_seconds=MATCH_S, witness_min_seconds=MIN_SEPARATION_S,
                    witness_max_seconds=MAX_SEPARATION_S, speed_mps=SPEED_MPS,
                    per_report_error_m=ERROR_PER_ROW_M, sensitivity=list(SENSITIVITY)),
        **counters))
    rows = []
    for i,sid in enumerate(ids,1):
        m = tail.read_motion(index[sid])
        truth = tail.read_identity(index[sid],m)
        row = evaluate_scene(read(private/f'{sid}.json.gz'),m,truth)
        rows.append(dict(scene=sid,component=components[sid],**row))
        if i % 50 == 0:
            tail.summary_state(output,start,status='RUNNING',phase='evaluation',scenes=i,**counters)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['component']].append(row)
    largest = min(grouped,key=lambda c:(-len(grouped[c]),c))
    full = population(rows)
    remaining = population([r for r in rows if r['component'] != largest])
    by_component = {c:population(group) for c,group in grouped.items()}
    gate = gates(full,remaining,by_component,largest)
    write_gzip(output/'fit-scenes.json.gz',rows)
    tail.write(output/'summary.json',dict(full=full, without_largest=remaining, by_component=by_component,
                                         largest_scene_component=largest, gates=gate,
                                         candidate_qualified=all(g['passes'] for g in gate.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runs','manifest','output'):
        parser.add_argument(f'--{key}',type=Path,required=True)
    parser.add_argument('--time-policy',choices=('advisory',),required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    tail.summary_state(args.output,start,status='RUNNING',phase='preflight')
    try:
        run(args,start)
    except Exception as exc:
        tail.summary_state(args.output,start,status='FAILED',error=repr(exc),optimizer_updates=0,model_forwards=0)
        raise
    tail.summary_state(args.output,start,status='COMPLETED',optimizer_updates=0,model_forwards=0,test_opened=False)


if __name__ == '__main__':
    main()
