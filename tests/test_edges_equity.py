from __future__ import annotations

import numpy as np
import pandas as pd

from quant_alpha.graph.edges_equity import (
    build_equity_topology,
    rolling_topology_for,
    static_topology_for,
)

SECTORS = {
    "A1": "Tech", "A2": "Tech", "A3": "Tech", "A4": "Tech",
    "B1": "Fin", "B2": "Fin", "B3": "Fin",
    "C1": "Solo",
}


def _panel(n_days: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(3)
    rows = []
    for sym in SECTORS:
        ret = rng.normal(0, 0.01, n_days)
        for i, dt in enumerate(dates):
            rows.append({"date": dt, "symbol": sym, "ret_1d": float(ret[i])})
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()


def _degree(topo) -> dict[str, int]:
    deg: dict[str, int] = {}
    for src, _dst, _w in topo.edges:
        deg[src] = deg.get(src, 0) + 1
    return deg


def test_correlation_backbone_undirected_no_isolated_nodes() -> None:
    panel = _panel()
    as_of = panel.index.get_level_values(0).max()
    topo = build_equity_topology(panel, None, as_of=as_of, top_k=3, min_periods=20)

    for src, dst, w in topo.edges:
        assert (dst, src, w) in topo.edges  # undirected
    deg = _degree(topo)
    # backbone + min_degree leaves no node isolated, including the Solo singleton
    assert all(deg.get(n, 0) >= 1 for n in topo.nodes)


def test_works_without_sector_data() -> None:
    panel = _panel()
    as_of = panel.index.get_level_values(0).max()
    topo = build_equity_topology(panel, sectors=None, as_of=as_of, top_k=3, min_periods=20)
    assert len(topo.edges) > 0


def test_sector_boost_only_adds_edges() -> None:
    panel = _panel()
    as_of = panel.index.get_level_values(0).max()
    without = build_equity_topology(panel, None, as_of=as_of, top_k=2, min_periods=20)
    with_sectors = build_equity_topology(panel, SECTORS, as_of=as_of, top_k=2, min_periods=20)
    assert set(with_sectors.edges) >= set(without.edges)


def test_min_degree_fallback_when_no_backbone() -> None:
    panel = _panel()
    as_of = panel.index.get_level_values(0).max()
    topo = build_equity_topology(
        panel, None, as_of=as_of, top_k=0, min_periods=20, min_degree=1
    )
    deg = _degree(topo)
    assert all(deg.get(n, 0) >= 1 for n in topo.nodes)


def test_no_future_leak() -> None:
    panel = _panel()
    dates = sorted(panel.index.get_level_values(0).unique())
    as_of = dates[80]

    clean = build_equity_topology(panel, SECTORS, as_of=as_of, min_periods=20)
    corrupted = panel.copy()
    corrupted.loc[corrupted.index.get_level_values(0) >= as_of, "ret_1d"] = 999.0
    poisoned = build_equity_topology(corrupted, SECTORS, as_of=as_of, min_periods=20)

    assert clean.edges == poisoned.edges


def test_static_topology_for_is_constant() -> None:
    panel = _panel()
    as_of = panel.index.get_level_values(0).max()
    topology_for = static_topology_for(panel, SECTORS, as_of, min_periods=20)
    t0, t1 = panel.index.get_level_values(0)[0], panel.index.get_level_values(0)[-1]
    assert topology_for(t0) is topology_for(t1)


def test_rolling_topology_for_rebuilds_per_time_and_caches() -> None:
    panel = _panel()
    dates = sorted(panel.index.get_level_values(0).unique())
    topology_for = rolling_topology_for(panel, SECTORS, min_periods=20)

    # cached: same query time returns the same object
    assert topology_for(dates[80]) is topology_for(dates[80])
    # per-time: different query times are built independently and match a
    # direct build as of that time
    assert topology_for(dates[40]) is not topology_for(dates[-1])
    direct = build_equity_topology(panel, SECTORS, as_of=dates[40], min_periods=20)
    assert topology_for(dates[40]).edges == direct.edges


def test_rolling_topology_for_early_dates_degrade_gracefully() -> None:
    panel = _panel()
    dates = sorted(panel.index.get_level_values(0).unique())
    topology_for = rolling_topology_for(panel, SECTORS, min_periods=20)

    # first date: no history at all -> empty, edge-free topology, no crash
    first = topology_for(dates[0])
    assert first.edges == ()
    # below min_periods: nodes exist but correlations are NaN -> no edges
    early = topology_for(dates[5])
    assert early.edges == ()


def test_rolling_topology_for_no_future_leak() -> None:
    panel = _panel()
    dates = sorted(panel.index.get_level_values(0).unique())
    as_of = dates[80]

    clean = rolling_topology_for(panel, SECTORS, min_periods=20)(as_of)
    corrupted = panel.copy()
    corrupted.loc[corrupted.index.get_level_values(0) >= as_of, "ret_1d"] = 999.0
    poisoned = rolling_topology_for(corrupted, SECTORS, min_periods=20)(as_of)

    assert clean.edges == poisoned.edges
