#!/usr/bin/env python3
"""
Correlation-based asset clustering -- new infrastructure, built for candidate 16
(correlation-cluster basket trend). Groups co-moving assets so a basket-level
trend signal can be traded as a unit, rather than each symbol independently.

Simplification (stated explicitly): clusters are computed ONCE from the
correlation structure over the full test window, not walk-forward/point-in-time.
This uses information from the whole window to decide grouping (a look-ahead
in the sense that day-1 clustering "knows" about correlation structure from
later in the window), which is fine for a first-pass test of whether the
CONCEPT has any edge at all, but is not how a live system would compute
clusters (that would need rolling/expanding-window reclustering). Flag this
before treating any T1 pass here as fully validated -- a walk-forward version
is the natural next refinement, not built here.
"""
from __future__ import annotations

import pandas as pd


def compute_static_clusters(panel: dict[str, pd.DataFrame], corr_threshold: float = 0.6,
                             min_cluster_size: int = 3, start: pd.Timestamp | None = None,
                             end: pd.Timestamp | None = None,
                             min_coverage: float = 0.5) -> dict[str, list[str]]:
    """Greedy correlation clustering (no external ML dependency): pick an unclustered
    seed, absorb every other unclustered symbol with correlation >= threshold to the
    seed, repeat. Clusters below min_cluster_size are pooled into 'unclustered'
    (traded flat -- no basket signal for symbols that don't cluster meaningfully).
    Not hierarchical/optimal clustering -- a simple, fast, defensible first pass."""
    returns = pd.DataFrame({sym: df["close"].pct_change() for sym, df in panel.items()})
    if start is not None or end is not None:
        returns = returns.loc[(returns.index >= (start or returns.index.min()))
                               & (returns.index <= (end or returns.index.max()))]
    coverage = returns.notna().mean()
    valid_symbols = coverage[coverage >= min_coverage].index
    returns = returns[valid_symbols]
    corr = returns.corr()

    # Seed order must be deterministic across process runs -- `set` iteration order for
    # strings depends on Python's per-process hash randomization (PYTHONHASHSEED is
    # randomized by default), so `next(iter(unclustered))` previously picked a different
    # seed on every invocation, silently producing different cluster groupings (and thus
    # different T5/T6/T7 accepted trade sets) run to run. Sorted seed order fixes this;
    # a `set` is still used for O(1) membership/removal within each seed's absorption pass.
    unclustered = set(corr.columns)
    seed_order = sorted(corr.columns)
    clusters: dict[str, list[str]] = {}
    cluster_id = 0
    for seed in seed_order:
        if seed not in unclustered:
            continue
        members = [seed]
        unclustered.discard(seed)
        for other in sorted(unclustered):
            if corr.loc[seed, other] >= corr_threshold:
                members.append(other)
                unclustered.discard(other)
        if len(members) >= min_cluster_size:
            clusters[f"cluster_{cluster_id}"] = members
            cluster_id += 1
        else:
            clusters.setdefault("unclustered", []).extend(members)

    return clusters
