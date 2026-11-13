"""Truth-free single-node refinement of a complete signed association graph.

J is only a within-group surrogate; improving it guarantees no physical risk.
No identities, source priorities, reference states, or training are accepted.
"""
from __future__ import annotations
import numpy as np


def refine_partition(node_count, pairs, scores, membership, *, threshold, max_moves=None,
                     move_policy='free',support=None,sources=None):
    n = int(node_count)
    pairs, scores = np.asarray(pairs), np.asarray(scores, float)
    membership = np.asarray(membership, np.int64).copy()
    if n <= 0 or membership.shape != (n,) or pairs.shape != (n*(n-1)//2, 2):
        raise ValueError('nonempty complete graph and one membership per node required')
    if pairs.dtype.kind not in 'iu' or scores.shape != (len(pairs),) or not np.isfinite(scores).all():
        raise ValueError('integer pairs and finite scores required')
    if (pairs < 0).any() or (pairs >= n).any() or (pairs[:,0] >= pairs[:,1]).any() or len(np.unique(pairs,axis=0)) != len(pairs):
        raise ValueError('complete unique unordered pairs required')
    if not np.isfinite(threshold) or (membership < 0).any(): raise ValueError('finite threshold/nonnegative groups')
    groups = sorted(set(membership), key=lambda g: int(np.flatnonzero(membership == g)[0]))
    membership = np.asarray([{g:i for i,g in enumerate(groups)}[v] for v in membership])
    if move_policy not in ('free','anchors','support'):raise ValueError('unknown move policy')
    anchors=np.asarray([np.flatnonzero(membership==g)[0] for g in range(len(groups))])
    original_group_count=len(groups)
    rejection_counts=dict(anchor_rejections=0,support_rejections=0)
    if move_policy=='support':
        support=np.asarray(support);sources=np.asarray(sources)
        if support.ndim!=2 or support.shape[0]!=n or support.dtype.kind!='b' or sources.shape!=(n,):
            raise ValueError('boolean query support and one source per node required')
        _,source_index=np.unique(sources,return_inverse=True)
        observation_counts=np.zeros((n,int(source_index.max())+1,support.shape[1]),np.int64)
        for i in range(n):observation_counts[membership[i],source_index[i]]+=support[i]
    budget = 4*n if max_moves is None else int(max_moves)
    if budget < 0: raise ValueError('nonnegative move budget')
    weights = scores-float(threshold)
    w = np.zeros((n,n)); w[pairs[:,0],pairs[:,1]] = weights; w += w.T
    sums = np.zeros((n,n)); counts = np.bincount(membership,minlength=n)
    for g in np.flatnonzero(counts): sums[:,g] = w[:,membership == g].sum(1)
    def objective(m): return float(weights[m[pairs[:,0]] == m[pairs[:,1]]].sum())
    initial = objective(membership); running = initial; trace = []; ties = 0
    def best_move():
        active = sorted(np.flatnonzero(counts), key=lambda g:int(np.flatnonzero(membership == g)[0]))
        free = np.flatnonzero(counts == 0)
        targets = np.asarray(active+([int(free[0])] if len(free) else []))
        gains = sums[:,targets]-sums[np.arange(n),membership,None]
        gains[membership[:,None] == targets] = -np.inf
        if move_policy!='free':
            for i,j in np.argwhere(gains>1e-10):
                destination=int(targets[j]);old=int(membership[i])
                if i==anchors[old] or destination>=original_group_count or i<anchors[destination]:
                    gains[i,j]=-np.inf;rejection_counts['anchor_rejections']+=1
                elif move_policy=='support' and np.any((observation_counts[old,source_index[i]]==1)&support[i]):
                    gains[i,j]=-np.inf;rejection_counts['support_rejections']+=1
        index = int(np.argmax(gains)); value = float(gains.flat[index])
        i,j = np.unravel_index(index,gains.shape)
        return value,int(i),int(targets[j]),int((gains == value).sum())
    for _ in range(budget):
        gain,node,destination,tie_count = best_move()
        if gain <= 1e-10: break
        old = int(membership[node]); ties += int(tie_count > 1)
        if move_policy=='support':
            observation_counts[old,source_index[node]]-=support[node]
            observation_counts[destination,source_index[node]]+=support[node]
        sums[:,old] -= w[:,node]; sums[:,destination] += w[:,node]
        counts[old] -= 1; counts[destination] += 1; membership[node] = destination
        running += gain
        trace.append(dict(node=node,from_label=old,to_label=destination,gain=gain,J=running,best_gain_tied=tie_count>1))
    remaining,_,_,_ = best_move()
    if not np.isfinite(remaining): remaining = 0.
    final = objective(membership)
    if not np.isclose(final,running,rtol=1e-11,atol=1e-6) or final < initial-1e-6:
        raise FloatingPointError('signed objective accounting mismatch')
    active = sorted(set(membership),key=lambda g:int(np.flatnonzero(membership == g)[0]))
    mapping = {g:i for i,g in enumerate(active)}
    canonical = np.asarray([mapping[v] for v in membership],np.int64)
    return canonical,dict(initial_J=initial,final_J=final,move_count=len(trace),max_moves=budget,
        locally_optimal=remaining<=1e-10,remaining_best_gain=remaining,tie_decisions=ties,
        **({} if move_policy=='free' else dict(move_policy=move_policy,**rejection_counts))),trace
