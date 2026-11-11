"""Score frozen neighbor evidence, with identity labels used only here."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, product
from math import dist
from statistics import median

from .causal_anchor_support import CONSISTENCY_M
from .expansion_evidence_metrics import TRUTH_GROUPS
from .neighbor_witness import WITNESS_TOLERANCE_M, summarize_votes


BIRTH_GROUPS = ("all", "sparse")


def _new_birth(arms):
    return dict(
        queries=Counter(), comparison=Counter(),
        open_edges={group: Counter() for group in TRUTH_GROUPS},
        wide_edges={group: Counter() for group in TRUTH_GROUPS},
        outcomes={arm: Counter() for arm in arms},
        paired={reference: {arm: Counter() for arm in arms if arm != reference}
                for reference in arms},
    )


def _new_summary(arms):
    return dict(
        birth={group: _new_birth(arms) for group in BIRTH_GROUPS},
        writes={group: Counter() for group in TRUTH_GROUPS},
        selections={arm: Counter() for arm in arms},
    )


def _edge_truth(pair, labels, seen):
    a, b = pair
    if labels[a] == labels[b]:
        return "true"
    if (b[0], labels[a]) in seen and (a[0], labels[b]) in seen:
        return "false_both_seen"
    return "false_unseen_truth"


def _incident(edges):
    index = defaultdict(list)
    for pair in edges:
        a, b = pair
        index[a].append((b, pair))
        index[b].append((a, pair))
    return index


def _true_independent_rows(rows, labels, excluded):
    """Keep one deterministic representative per real non-focal object."""
    representatives = {}
    for pair, row in sorted(rows, key=lambda item: item[0]):
        a, b = pair
        identity = labels[a]
        if identity == labels[b] and identity not in excluded:
            representatives.setdefault(identity, row)
    return tuple(representatives.values())


def _witness_truth(pairs, labels):
    n = sum(labels[a] == labels[b] for a, b in pairs)
    if not pairs:
        return "missing"
    if n == len(pairs):
        return "all_true"
    return "mixed" if n else "all_false"


def _support(fit):
    return fit.support if fit is not None else 0


def _signal(consensus, offset):
    """Classify strong positive, strong conflict and unresolved evidence."""
    if offset is None:
        return "missing_clock"
    best = consensus.best
    if _support(best) < 2:
        return "insufficient_support"
    if _support(consensus.competing) >= best.support:
        return "competing_offset"
    if best.assignment_ambiguous:
        return "ambiguous_assignment"
    if consensus.residual_m > WITNESS_TOLERANCE_M:
        return "conflict"
    return "compatible" if _support(consensus.candidate) >= 2 else "insufficient_at_candidate"


def _range_bin(count):
    return str(min(count, 2))


def _record_neighbor(counts, pair, evidence, labels, truth):
    counts["n"] += 1
    counts[f"clock_{evidence.clock}"] += 1
    counts["any_nearby_both"] += all(evidence.nearby_counts)
    counts["any_cotimed_both"] += all(evidence.cotimed_counts)
    counts["any_neighborhood_truncated"] += any(evidence.truncated)
    counts["both_neighborhoods_truncated"] += all(evidence.truncated)
    counts["bridge_available"] += evidence.bridge_offset is not None
    consensus = evidence.consensus
    best = consensus.best
    support = _support(best)
    counts[f"fragment_support_{_range_bin(support)}"] += 1
    counts[f"candidate_support_{_range_bin(_support(consensus.candidate))}"] += 1
    counts["best_two_or_more"] += support >= 2
    counts["best_assignment_ambiguous"] += bool(best and best.assignment_ambiguous)
    counts["best_tied_offset"] += support >= 2 and _support(consensus.competing) == support
    if consensus.competing is not None and support >= 2 and consensus.competing.support == support:
        counts["best_tied_offset_different_witnesses"] += set(consensus.competing.witnesses) != set(best.witnesses)
    if consensus.residual_m is not None:
        counts["offset_compared"] += 1
        counts["offset_within_400"] += consensus.residual_m <= WITNESS_TOLERANCE_M

    witnesses = best.witnesses if best is not None else ()
    witness_truth = _witness_truth(witnesses, labels)
    counts[f"best_witness_{witness_truth}"] += 1
    excluded = {labels[key] for key in pair}
    best_true_objects = {labels[a] for a, b in witnesses if labels[a] == labels[b] and labels[a] not in excluded}
    counts[f"best_true_independent_objects_{_range_bin(len(best_true_objects))}"] += 1
    counts["best_contains_focal_true_identity"] += any(labels[a] == labels[b] and labels[a] in excluded
                                                       for a, b in witnesses)
    signal = _signal(consensus, evidence.candidate_offset)
    counts[f"signal_{signal}"] += 1
    counts[f"{evidence.clock}_signal_{signal}"] += 1
    if truth != "true":
        return signal

    clean = _true_independent_rows(((vote.pair, vote) for vote in evidence.votes), labels, excluded)
    counts[f"clean_object_support_{_range_bin(len(clean))}"] += 1
    if evidence.candidate_offset is None:
        clean_signal = "missing_clock"
    elif len(clean) < 2:
        clean_signal = "insufficient_objects"
    else:
        clean_consensus = summarize_votes(clean, evidence.candidate_offset)
        clean_signal = _signal(clean_consensus, evidence.candidate_offset)
        if clean_signal == "insufficient_support":
            clean_signal = "disagreement"
    counts[f"clean_{clean_signal}"] += 1
    counts[f"{evidence.clock}_clean_{clean_signal}"] += 1
    has_false_support = witness_truth in {"mixed", "all_false"}
    counts["dirty_support_clean_insufficient"] += has_false_support and clean_signal == "insufficient_objects"
    counts["removable_pollution"] += has_false_support and signal == "conflict" and clean_signal == "compatible"
    return signal


def _record_legacy(counts, pair, edge, evidence, labels, truth):
    witness_truth = _witness_truth(edge.witnesses, labels)
    counts[f"legacy_witness_{witness_truth}"] += 1
    observed = edge.features["witness_residual_m"]
    counts["legacy_comparable"] += observed is not None
    counts["legacy_compatible"] += observed is not None and observed <= WITNESS_TOLERANCE_M
    if truth != "true":
        return
    clean = _true_independent_rows(edge.witness_offsets, labels, {labels[key] for key in pair})
    counts[f"legacy_clean_object_support_{_range_bin(len(clean))}"] += 1
    if evidence.strict_offset is None:
        signal = "missing_clock"
    elif not clean:
        signal = "insufficient_objects"
    elif any(dist(a, b) > CONSISTENCY_M for a, b in combinations(clean, 2)):
        signal = "disagreement"
    else:
        centre = tuple(map(median, zip(*clean, strict=True)))
        signal = "compatible" if dist(centre, evidence.strict_offset) <= WITNESS_TOLERANCE_M else "conflict"
    counts[f"legacy_clean_{signal}"] += 1
    dirty = witness_truth in {"mixed", "all_false"}
    counts["legacy_dirty_support_clean_insufficient"] += dirty and signal == "insufficient_objects"
    counts["legacy_removable_pollution"] += (dirty and observed is not None
                                            and observed > WITNESS_TOLERANCE_M and signal == "compatible")


def _birth_population(key, geometry, geometry_selected, all_ring, opened, seen, fresh_labels, labels):
    identity = labels[key]
    if not any(source != key[0] and other == identity for source, other in seen):
        return "unseen_truth"
    if not any(source != key[0] and identity in identities for source, identities in fresh_labels.items()):
        return "seen_truth_not_fresh"
    if any(labels[peer] == identity for peer, _ in geometry_selected[key]):
        return "geometry_correct"
    if any(labels[peer] == identity for peer, _ in opened[key]):
        return "true_open_ring"
    if any(labels[peer] == identity for peer, _ in geometry[key]):
        return "true_geometry_candidate_unselected"
    if any(labels[peer] == identity for peer, _ in all_ring[key]):
        return "true_ring_occupied"
    return "fresh_truth_outside_graph"


def _record_pair_comparison(counts, incident, signals, parallel, labels):
    true = [(peer, pair) for peer, pair in incident if labels[pair[0]] == labels[pair[1]]]
    false = [(peer, pair) for peer, pair in incident if labels[pair[0]] != labels[pair[1]]]
    counts["both_true_false_query"] += bool(true and false)
    counts["unique_false_no_true_query"] += not true and len(false) == 1
    for (true_peer, true_pair), (false_peer, false_pair) in product(true, false):
        value = parallel.get(tuple(sorted((true_peer, false_peer))))
        group = "parallel" if value is True else "other" if value is False else "unknown"
        counts[f"{group}_comparisons"] += 1
        true_signal, false_signal = signals[true_pair], signals[false_pair]
        counts[f"{group}_both_compatible"] += true_signal == false_signal == "compatible"
        counts[f"{group}_true_compatible_false_conflict"] += true_signal == "compatible" and false_signal == "conflict"
        counts[f"{group}_true_compatible_false_unresolved"] += (true_signal == "compatible"
                                                                  and false_signal not in {"compatible", "conflict"})
        counts[f"{group}_false_compatible_true_conflict"] += false_signal == "compatible" and true_signal == "conflict"


def _record_outcomes(target, selected, key, own_unseen, labels, seen):
    outcomes = {}
    for arm, incident in selected.items():
        choice = incident[key]
        if len(choice) > 1:
            raise ValueError("an arm selected more than one partner for a birth fragment")
        counts = target["outcomes"][arm]
        result = "unmatched"
        if choice:
            _, pair = choice[0]
            truth = _edge_truth(pair, labels, seen)
            result = "correct" if truth == "true" else "wrong"
            counts[truth] += 1
            counts["own_unseen_wrong"] += own_unseen and result == "wrong"
        counts[result] += 1
        outcomes[arm] = result
    for reference, arms in target["paired"].items():
        for arm, counts in arms.items():
            counts[f"{outcomes[reference]}_to_{outcomes[arm]}"] += 1


def score_neighbor_audit(recovery_frames, expansion_frames, probe_frames, labels, *, arms=()):
    """Return anonymous scene counts from co-timed frames frozen before label access."""
    arms = tuple(arms) or ("geometry", "wide")
    output = _new_summary(arms)
    seen = set()
    seen_selected = {arm: set() for arm in arms}
    seen_birth = {group: set() for group in BIRTH_GROUPS}
    seen_wide_birth = {group: set() for group in BIRTH_GROUPS}
    seen_writes = set()
    for frame, expansion, probe in zip(recovery_frames, expansion_frames, probe_frames, strict=True):
        if frame.stamp != expansion.stamp:
            raise ValueError("recovery and expansion frames must describe the same decision")
        seen.update((key[0], labels[key]) for key in frame.updated)
        base = set(frame.geometry_links)
        selected = {}
        for arm in arms:
            added = set(frame.additions.get(arm, ()))
            pairs = base | added
            selected[arm] = _incident(pairs)
            counts = output["selections"][arm]
            for pair in pairs:
                truth = _edge_truth(pair, labels, seen)
                counts[truth] += 1
                if pair in added:
                    counts[f"added_{truth}"] += 1
                if pair not in seen_selected[arm]:
                    counts[f"distinct_{truth}"] += 1
                    seen_selected[arm].add(pair)

        for pair, evidence in probe.writes.items():
            truth = _edge_truth(pair, labels, seen)
            counts = output["writes"][truth]
            counts["distinct_pairs"] += pair not in seen_writes
            seen_writes.add(pair)
            _record_neighbor(counts, pair, evidence, labels, truth)
        if not frame.born:
            continue

        geometry = _incident(frame.geometry_edges)
        geometry_selected = _incident(base)
        all_ring = _incident(item.pair for item in frame.evidence)
        opened = _incident(probe.ring)
        expansion_edges = {edge.pair: edge for edge in expansion.edges}
        wide = set(frame.additions["wide"])
        fresh_counts = Counter(key[0] for key in frame.fresh)
        fresh_labels = defaultdict(set)
        for key in frame.fresh:
            fresh_labels[key[0]].add(labels[key])

        for key in frame.born:
            groups = ("all", "sparse") if len(frame.fresh) - fresh_counts[key[0]] <= 3 else ("all",)
            population = _birth_population(key, geometry, geometry_selected, all_ring, opened, seen, fresh_labels, labels)
            for group in groups:
                target = output["birth"][group]
                counts = target["queries"]
                counts["n"] += 1
                counts[population] += 1
                counts["no_open_candidates"] += not opened[key]
                counts["unseen_truth_single_open_false"] += population == "unseen_truth" and len(opened[key]) == 1
                signals = {}
                for _, pair in opened[key]:
                    truth = _edge_truth(pair, labels, seen)
                    evidence, legacy = probe.ring[pair], expansion_edges[pair]
                    edge_counts = target["open_edges"][truth]
                    edge_counts["distinct_pairs"] += pair not in seen_birth[group]
                    seen_birth[group].add(pair)
                    signals[pair] = _record_neighbor(edge_counts, pair, evidence, labels, truth)
                    _record_legacy(edge_counts, pair, legacy, evidence, labels, truth)
                    if pair in wide:
                        edge_counts = target["wide_edges"][truth]
                        edge_counts["distinct_pairs"] += pair not in seen_wide_birth[group]
                        seen_wide_birth[group].add(pair)
                        _record_neighbor(edge_counts, pair, evidence, labels, truth)
                        _record_legacy(edge_counts, pair, legacy, evidence, labels, truth)
                _record_pair_comparison(target["comparison"], opened[key], signals, probe.parallel, labels)
                _record_outcomes(target, selected, key, population == "unseen_truth", labels, seen)
    return output


def _merge_counts(target, source):
    if isinstance(target, Counter):
        target.update(source)
        return
    for key, value in source.items():
        _merge_counts(target[key], value)


def pool_neighbor_audit(results):
    """Pool anonymous scene counts without deduplicating across scenes."""
    iterator = iter(results)
    first = next(iterator, None)
    if first is None:
        return _new_summary(())
    output = _new_summary(first["selections"])
    _merge_counts(output, first)
    for result in iterator:
        _merge_counts(output, result)
    return output
