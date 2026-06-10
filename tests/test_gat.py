from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from quant_alpha.features.factor import GraphFactorProvider, apply_factors
from quant_alpha.graph.propagate import GATPropagator, Topology
from quant_alpha.models.gat import (
    FactorGraphDataset,
    GATConfig,
    GATModel,
    build_sections,
    composite_series,
    fit,
    time_ordered_split,
)

FEATURES = ("f0", "f1", "f2")


def _panel(n_days: int = 40, entities: int = 6) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    syms = [f"S{i}" for i in range(entities)]
    rng = np.random.default_rng(7)
    rows = []
    for sym in syms:
        price = 100 + np.cumsum(rng.normal(0, 1, n_days))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "date": dt,
                    "symbol": sym,
                    "adj_close": float(price[i]),
                    "f0": float(rng.normal()),
                    "f1": float(rng.normal()),
                    "f2": float(rng.normal()),
                }
            )
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()


def _topology_for(panel: pd.DataFrame):
    entities = list(panel.index.get_level_values(1).unique())
    fully_connected = Topology(
        nodes=tuple(entities),
        edges=tuple((s, d, 1.0) for s in entities for d in entities if s != d),
    )
    return lambda _time: fully_connected


def test_gatmodel_forward_shape() -> None:
    import torch

    model = GATModel(GATConfig(in_dim=3, hidden_dim=8, heads=2, dropout=0.0)).eval()
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (5,)


def test_build_sections_shapes_and_mask() -> None:
    panel = _panel()
    sections = build_sections(panel, _topology_for(panel), FEATURES, k=2)
    assert len(sections) == 40
    sec = sections[0]
    assert sec.x.shape[1] == len(FEATURES)
    assert sec.mask.dtype == __import__("torch").bool
    # last k snapshots have no forward return -> fully masked out
    assert not bool(sections[-1].mask.any())


def test_fit_runs_and_saves(tmp_path) -> None:
    panel = _panel()
    ds = FactorGraphDataset(build_sections(panel, _topology_for(panel), FEATURES, k=2))
    cfg = GATConfig(in_dim=len(FEATURES), hidden_dim=8, heads=2, dropout=0.0, forward_k=2, epochs=2)
    out = tmp_path / "gat.pt"

    model = fit(ds, cfg, out_path=str(out))
    assert isinstance(model, GATModel)
    assert out.exists()

    train_idx, valid_idx, test_idx = time_ordered_split(len(ds), embargo=cfg.forward_k)
    assert set(train_idx).isdisjoint(valid_idx)
    assert min(valid_idx) - max(train_idx) >= cfg.forward_k


def test_gat_propagator_seam_and_provider() -> None:
    panel = _panel()
    topology_for = _topology_for(panel)
    model = GATModel(GATConfig(in_dim=len(FEATURES), hidden_dim=8, heads=2, dropout=0.0)).eval()

    propagator = GATPropagator(model=model, feature_cols=FEATURES)

    # direct seam call on one snapshot
    one_date = panel.index.get_level_values(0)[0]
    snapshot = panel.xs(one_date, level=0)
    out = propagator.propagate(snapshot, topology_for(one_date))
    assert list(out.index) == list(snapshot.index)
    assert np.isfinite(out.to_numpy()).all()

    # through the provider seam -> a relational factor column
    provider = GraphFactorProvider(
        name="alpha_gat_composite",
        family="relational",
        hypothesis="GAT propagation of island alphas",
        expected_direction=1,
        propagator=propagator,
        topology_for=topology_for,
        feature_cols=FEATURES,
    )
    result = apply_factors(panel, [provider])
    assert "alpha_gat_composite" in result.columns
    assert result["alpha_gat_composite"].notna().sum() > 0


def test_composite_series_aligns_to_panel_index() -> None:
    panel = _panel()
    ds = FactorGraphDataset(build_sections(panel, _topology_for(panel), FEATURES, k=2))
    model = GATModel(GATConfig(in_dim=len(FEATURES), hidden_dim=8, heads=2, dropout=0.0)).eval()

    composite = composite_series(model, ds, name="alpha_gat_composite")

    # same (date, symbol) index as the alpha panel — never reconstructed
    assert composite.index.names == ["date", "symbol"]
    assert set(composite.index) == set(panel.index)
    assert len(composite) == len(panel)
