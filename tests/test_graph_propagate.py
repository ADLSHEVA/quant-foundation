from __future__ import annotations

import pandas as pd

from quant_alpha.graph.propagate import (
    Propagator,
    Topology,
    UniformMeanPropagator,
)
from quant_alpha.ingestion.energy import generate_synthetic_power_market


def _fully_connected(nodes: list[str]) -> Topology:
    edges = tuple((src, dst, 1.0) for src in nodes for dst in nodes if src != dst)
    return Topology(nodes=tuple(nodes), edges=edges)


def test_uniform_mean_propagator_satisfies_protocol() -> None:
    assert isinstance(UniformMeanPropagator(feature="spot_price"), Propagator)


def test_uniform_mean_reproduces_cross_market_spot_mean() -> None:
    # Anchor test: the baseline adapter over a fully-connected topology must
    # reproduce the existing `cross_market_spot_mean` exactly. This pins the
    # island-vs-relational A/B — the GAT adapter is the same seam, swapped.
    market = generate_synthetic_power_market(
        ["DE_LU", "CZ", "FR"], "2024-01-01", "2024-01-15"
    )
    market = market.copy()
    market["cross_market_spot_mean"] = market.groupby("timestamp")["spot_price"].transform(
        "mean"
    )

    timestamp = market["timestamp"].iloc[0]
    snapshot = market[market["timestamp"] == timestamp]
    node_features = snapshot.set_index("market")[["spot_price"]]

    propagator = UniformMeanPropagator(feature="spot_price", include_self=True)
    out = propagator.propagate(node_features, _fully_connected(list(node_features.index)))

    expected = snapshot.set_index("market")["cross_market_spot_mean"]
    pd.testing.assert_series_equal(
        out.sort_index(), expected.sort_index(), check_names=False
    )


def test_uniform_mean_respects_edge_direction() -> None:
    # Directed topology = asymmetric transmission. A -> B only: B averages A and
    # itself; A has no in-neighbour, so it keeps its own value.
    node_features = pd.DataFrame({"x": [10.0, 20.0]}, index=["A", "B"])
    topology = Topology(nodes=("A", "B"), edges=(("A", "B", 1.0),))

    out = UniformMeanPropagator(feature="x", include_self=True).propagate(
        node_features, topology
    )

    assert out["A"] == 10.0
    assert out["B"] == 15.0
