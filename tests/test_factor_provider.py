from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_alpha.features.factor import (
    ExpressionFactorProvider,
    Factor,
    FactorProvider,
    GraphFactorProvider,
    apply_factors,
)
from quant_alpha.features.registry import make_equity_alpha_registry
from quant_alpha.graph.propagate import Topology, UniformMeanPropagator


def _equity_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    symbols = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(0)
    rows = []
    for sym in symbols:
        price = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
        for i, dt in enumerate(dates):
            close = float(price[i])
            rows.append(
                {
                    "date": dt,
                    "symbol": sym,
                    "adj_close": close,
                    "close": close,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "volume": float(1_000 + rng.integers(0, 500)),
                }
            )
    df = pd.DataFrame(rows)
    df["ret_1d"] = df.groupby("symbol")["adj_close"].transform(lambda s: s.pct_change())
    return df.set_index(["date", "symbol"])


def test_factor_rejects_bad_direction() -> None:
    with pytest.raises(ValueError):
        Factor("x", "fam", "hyp", expected_direction=0, compute=lambda p: p["adj_close"])


def test_providers_satisfy_protocol() -> None:
    expr = ExpressionFactorProvider(tuple(make_equity_alpha_registry()))
    graph = GraphFactorProvider(
        "g", "fam", "hyp", 1, UniformMeanPropagator("x"), lambda t: Topology((), ()), ("x",)
    )
    assert isinstance(expr, FactorProvider)
    assert isinstance(graph, FactorProvider)


def test_expression_provider_is_faithful_to_registry() -> None:
    # The provider path must reproduce each AlphaDefinition.compute exactly —
    # wrapping introduces no distortion (deepening #1/#2 characterisation).
    panel = _equity_panel()
    registry = make_equity_alpha_registry()

    result = apply_factors(panel, [ExpressionFactorProvider(tuple(registry))])

    for definition in registry:
        expected = definition.compute(panel)
        pd.testing.assert_series_equal(
            result[definition.name], expected, check_names=False
        )


def test_graph_provider_reproduces_cross_sectional_mean() -> None:
    # GraphFactorProvider + UniformMeanPropagator over a fully-connected
    # topology = the per-snapshot cross-sectional mean. Ties the propagate seam
    # (#3) to the provider seam (#2).
    panel = _equity_panel()[["adj_close"]].rename(columns={"adj_close": "x"})
    entities = panel.index.get_level_values(1).unique().tolist()
    fully_connected = Topology(
        nodes=tuple(entities),
        edges=tuple((s, d, 1.0) for s in entities for d in entities if s != d),
    )

    provider = GraphFactorProvider(
        name="alpha_graph_mean",
        family="relational",
        hypothesis="neighbour mean",
        expected_direction=1,
        propagator=UniformMeanPropagator("x", include_self=True),
        topology_for=lambda _t: fully_connected,
        feature_cols=("x",),
    )

    result = apply_factors(panel, [provider])
    expected = panel.groupby(level=0)["x"].transform("mean")
    pd.testing.assert_series_equal(
        result["alpha_graph_mean"], expected, check_names=False
    )
