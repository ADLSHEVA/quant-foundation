"""Equity graph construction: rolling-correlation top-k, optional sector boost.

The backbone is correlation top-k, which is N-agnostic and holds at any universe
size. Sector co-membership is an *optional* enhancement (skipped silently when no
sector data is present), not a hard dependency — so a small, sector-sparse
universe still yields a non-empty, connected graph. A `min_degree` fallback wires
any leftover isolated node to its highest-correlation peers; it does nothing once
the universe is large enough.

Leakage guard: the correlation window is strictly before `as_of`, asserted at
build time. Build the static graph from the training period only.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quant_alpha.graph.propagate import Topology


def build_equity_topology(
    panel: pd.DataFrame,
    sectors: Mapping[str, str] | None = None,
    *,
    as_of,
    return_col: str = "ret_1d",
    window: int = 60,
    top_k: int = 8,
    min_periods: int = 40,
    min_degree: int = 1,
) -> Topology:
    """Build one undirected correlation top-k graph as of `as_of`.

    Backbone: each node connects to its top_k peers by return correlation over the
    trailing `window` dates before `as_of`. If `sectors` is given, same-sector
    peers (top_k by correlation) are additionally connected. Any node still below
    `min_degree` is wired to its highest-correlation peers. Edge weight is the
    correlation.
    """
    times = panel.index.get_level_values(0)
    history = panel[times < as_of]
    recent_dates = sorted(history.index.get_level_values(0).unique())[-window:]
    if recent_dates:
        assert max(recent_dates) < as_of, "correlation window must precede as_of"
    window_panel = history[history.index.get_level_values(0).isin(recent_dates)]

    corr = window_panel[return_col].unstack(level=1).corr(min_periods=min_periods)
    nodes = tuple(corr.index)
    neighbours: dict[str, set[str]] = {n: set() for n in nodes}

    def connect(a: str, b: str) -> None:
        if a != b and not pd.isna(corr.loc[a, b]):
            neighbours[a].add(b)
            neighbours[b].add(a)

    for node in nodes:
        ranked = corr.loc[node].drop(labels=[node]).dropna().sort_values(ascending=False)
        for peer in ranked.index[:top_k]:  # correlation backbone
            connect(node, peer)
        if sectors and sectors.get(node) is not None:  # optional sector boost
            same = [p for p in ranked.index if sectors.get(p) == sectors.get(node)]
            for peer in same[:top_k]:
                connect(node, peer)

    for node in nodes:  # min_degree fallback against isolated nodes
        if len(neighbours[node]) >= min_degree:
            continue
        ranked = corr.loc[node].drop(labels=[node]).dropna().sort_values(ascending=False)
        for peer in ranked.index:
            if len(neighbours[node]) >= min_degree:
                break
            connect(node, peer)

    edges = tuple(
        sorted(
            (a, b, float(corr.loc[a, b]))
            for a in nodes
            for b in neighbours[a]
        )
    )
    return Topology(nodes=nodes, edges=edges)


def static_topology_for(
    panel: pd.DataFrame,
    sectors: Mapping[str, str] | None,
    as_of,
    **kwargs,
):
    """A `topology_for` callable returning one static graph built from data before
    `as_of` (use the train-period end), ignoring the query time."""
    topology = build_equity_topology(panel, sectors, as_of=as_of, **kwargs)
    return lambda _time: topology


def rolling_topology_for(
    panel: pd.DataFrame,
    sectors: Mapping[str, str] | None,
    **kwargs,
):
    """A `topology_for` callable that rebuilds the graph as of each query time.

    The dynamic upgrade over `static_topology_for`'s frozen train-period graph:
    every call uses only data strictly before the query time (asserted in the
    builder), so OOS snapshots get fresh graphs that are still point-in-time
    correct — in deployment the trailing correlation window is available at t.
    Early snapshots without enough history yield an edge-free topology rather
    than failing. Topologies are cached per timestamp."""
    cache: dict = {}

    def topology_for(time):
        if time not in cache:
            cache[time] = build_equity_topology(panel, sectors, as_of=time, **kwargs)
        return cache[time]

    return topology_for
