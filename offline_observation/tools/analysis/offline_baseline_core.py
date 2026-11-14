"""C145 common-protocol geometric association from MTAD measurements only."""
from __future__ import annotations

import numpy as np

from association.data.physical_units import KNOT_TO_MPS
from association.evaluation.native_compatibility import advance, distance


ARMS = ('END_DTW', 'CV_DTW', 'END_SYNC')
MAX_GAP_SECONDS = 1200.0
MAX_ENDPOINT_METRES = 20_000.0
ENDPOINT_SLACK_METRES = 1000.0
ENDPOINT_SPEED_MPS = 15.0
MIN_COMMON_QUERIES = 2
CANDIDATE_METRES = 10_000.0
ACCEPT_METRES = 5000.0
DTW_BAND = 2


def _itsa_candidates(sources, native, values):
    _, times, positions, offsets = native
    starts, ends = offsets[:-1], offsets[1:] - 1
    gaps = times[starts][None, :] - times[ends][:, None]
    same_source = sources[:, None] == sources[None, :]
    left, right = np.nonzero(same_source & (gaps > 0) & (gaps <= MAX_GAP_SECONDS))
    gap = gaps[left, right]
    from_pos, to_pos = positions[ends[left]], positions[starts[right]]
    limit = np.minimum(MAX_ENDPOINT_METRES, ENDPOINT_SLACK_METRES + ENDPOINT_SPEED_MPS * gap)
    keep = distance(from_pos, to_pos) <= limit
    left, right, gap = left[keep], right[keep], gap[keep]
    from_pos, to_pos = from_pos[keep], to_pos[keep]
    delta = to_pos - from_pos
    endpoint = np.abs(delta[:, 0]) + np.abs((delta[:, 1] + 180.0) % 360.0 - 180.0)

    speed = values[:, 2] * KNOT_TO_MPS
    course = np.deg2rad(values[:, 3])
    velocity = speed[:, None] * np.column_stack((np.cos(course), np.sin(course)))
    from_midpoint = advance(from_pos, velocity[ends[left]], gap / 2)
    to_midpoint = advance(to_pos, velocity[starts[right]], -gap / 2)
    return left, right, endpoint, distance(from_midpoint, to_midpoint)


def _itsa_chains(node_count, left, right, scores):
    previous = np.full(node_count, -1, dtype=np.int64)
    following = previous.copy()
    links = []
    for index in np.lexsort((right, left, scores)):
        before, after = int(left[index]), int(right[index])
        if following[before] != -1 or previous[after] != -1:
            continue
        following[before], previous[after] = after, before
        links.append((before, after))

    chains = []
    for head in np.flatnonzero(previous == -1):
        node, chain = int(head), []
        while node != -1:
            chain.append(node)
            node = int(following[node])
        chains.append(tuple(chain))
    chains.sort(key=min)
    return chains, links


def _chain_queries(chains, support, states):
    """Read one original supporting member for each chain/query in input order."""
    coverage = np.zeros((len(chains), support.shape[1]), bool)
    positions = np.zeros((*coverage.shape, 2), np.float64)
    for index, chain in enumerate(chains):
        members = np.asarray(sorted(chain), np.int64)
        visible = support[members]
        queries = np.flatnonzero(visible.any(axis=0))
        selected = members[np.argmax(visible[:, queries], axis=0)]
        coverage[index, queries] = True
        positions[index, queries] = states[selected, queries, :2]
    return coverage, positions


def _dtw_path_mean(first, second):
    """Mean on the minimum-total-distance path inside a ±2 index band."""
    batch, size, _ = first.shape
    width = 2 * DTW_BAND + 1
    centre = DTW_BAND + 1
    column = np.arange(size)[:, None] + np.arange(-DTW_BAND, DTW_BAND + 1)
    local = distance(first[:, :, None], second[:, np.clip(column, 0, size - 1)])
    total = np.full((batch, size + 1, width + 2), np.inf)
    length = np.zeros(total.shape, np.int16)
    total[:, 0, centre] = 0.0
    for row, columns in enumerate(column, 1):
        for band, other in enumerate(columns, 1):
            if not 0 <= other < size:
                continue
            # Ties use diagonal, then vertical, then horizontal.
            previous = np.stack((total[:, row - 1, band], total[:, row - 1, band + 1],
                                 total[:, row, band - 1]), axis=1)
            steps = np.stack((length[:, row - 1, band], length[:, row - 1, band + 1],
                              length[:, row, band - 1]), axis=1)
            best = previous.argmin(axis=1, keepdims=True)
            total[:, row, band] = np.take_along_axis(previous, best, axis=1)[:, 0] + local[:, row - 1, band - 1]
            length[:, row, band] = np.take_along_axis(steps, best, axis=1)[:, 0] + 1
    return total[:, size, centre] / length[:, size, centre]


def _mstsa_candidates(chain_sources, coverage, positions):
    row, col = np.triu_indices(len(chain_sources), 1)
    cross = chain_sources[row] != chain_sources[col]
    rows, cols = row[cross], col[cross]
    time_qualified, candidates = 0, []
    # Keep pair-by-query temporaries bounded independently of the scene size.
    for start in range(0, len(rows), 1024):
        block = slice(start, start + 1024)
        left, right = rows[block], cols[block]
        joint = coverage[left] & coverage[right]
        counts = joint.sum(axis=1)
        eligible = counts >= MIN_COMMON_QUERIES
        time_qualified += int(eligible.sum())
        left, right, joint, counts = left[eligible], right[eligible], joint[eligible], counts[eligible]
        ends = distance(positions[left], positions[right])
        diagonal = np.sum(ends * joint, axis=1) / counts
        close = diagonal <= CANDIDATE_METRES
        for count in np.unique(counts[close]):
            selected = close & (counts == count)
            first, second = left[selected], right[selected]
            mask = joint[selected]
            first_positions = positions[first][mask].reshape(-1, count, 2)
            second_positions = positions[second][mask].reshape(-1, count, 2)
            dtw = _dtw_path_mean(first_positions, second_positions)
            candidates.extend(zip(first.tolist(), second.tolist(), dtw.tolist(), diagonal[selected].tolist()))
    return candidates, time_qualified


def _mstsa_partition(chains, chain_sources, candidates, node_count, score_index):
    parent = list(range(len(chains)))
    component_sources = [{source} for source in chain_sources.tolist()]

    def root(index):
        while index != parent[index]:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    links, paired = [], set()
    ranked = sorted(candidates, key=lambda candidate: (candidate[score_index], candidate[0], candidate[1]))
    for candidate in ranked:
        if candidate[score_index] > ACCEPT_METRES:
            break
        left, right = candidate[:2]
        source_pair = frozenset((chain_sources[left], chain_sources[right]))
        first, second = root(left), root(right)
        if ((source_pair, left) in paired or (source_pair, right) in paired
                or component_sources[first] & component_sources[second]):
            continue
        parent[second] = first
        component_sources[first].update(component_sources[second])
        paired.update(((source_pair, left), (source_pair, right)))
        links.append((chains[left], chains[right]))

    # Chains are ordered by first original node so membership is canonical.
    labels = {}
    membership = np.empty(node_count, dtype=np.int64)
    for index, chain in enumerate(chains):
        membership[list(chain)] = labels.setdefault(root(index), len(labels))
    return membership, links


def predict_arms(scene):
    """Return full node partitions, direct decisions and candidate accounting.

    Uses the measurement-only scene's ragged canonical events, source names,
    Q64 native support and native positions. Link indices use original node order.
    """
    sources = scene['sources']
    node_count = len(scene['nodes'])
    left, right, endpoint, constant_velocity = _itsa_candidates(sources, scene['native'], scene['values'])
    outputs = {}
    for scores, arms in ((endpoint, (('END_DTW', 2), ('END_SYNC', 3))),
                         (constant_velocity, (('CV_DTW', 2),))):
        chains, itsa_links = _itsa_chains(node_count, left, right, scores)
        chain_sources = sources[[chain[0] for chain in chains]]
        coverage, positions = _chain_queries(chains, scene['support'], scene['native_motion'])
        candidates, time_qualified = _mstsa_candidates(chain_sources, coverage, positions)
        for name, score_index in arms:
            membership, mstsa_links = _mstsa_partition(chains, chain_sources, candidates, node_count, score_index)
            outputs[name] = dict(
                membership=membership, itsa_links=itsa_links.copy(), mstsa_links=mstsa_links,
                audit=dict(itsa_candidates=len(left), itsa_accepted=len(itsa_links),
                           same_source_chains=len(chains), mstsa_time_qualified=time_qualified,
                           mstsa_candidates=len(candidates),
                           mstsa_under_accept_limit=sum(pair[score_index] <= ACCEPT_METRES for pair in candidates),
                           mstsa_accepted=len(mstsa_links), output_groups=len(np.unique(membership)),
                           singleton_groups=int((np.bincount(membership) == 1).sum())),
            )
    return {name: outputs[name] for name in ARMS}
