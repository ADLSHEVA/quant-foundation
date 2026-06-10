from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from quant_alpha.config import ProjectConfig
from quant_alpha.features.alpha_factors import add_alpha_factors
from quant_alpha.run_gat_equity import (
    COMPOSITE_NAME,
    ISLAND_MEAN_NAME,
    UNIFORM_NAME,
    gat_equity_from_panel,
)

SECTORS = {
    "T1": "Tech", "T2": "Tech", "T3": "Tech", "T4": "Tech",
    "F1": "Fin", "F2": "Fin", "F3": "Fin", "F4": "Fin",
    "E1": "Energy", "E2": "Energy", "E3": "Energy", "E4": "Energy",
}


def _prices(n_days: int = 130) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(11)
    rows = []
    for sym in SECTORS:
        price = np.maximum(100 + np.cumsum(rng.normal(0, 1, n_days)), 5.0)
        for i, dt in enumerate(dates):
            close = float(price[i])
            rows.append(
                {
                    "date": dt,
                    "symbol": sym,
                    "adj_close": close,
                    "close": close,
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "volume": float(1_000_000 + rng.integers(0, 500_000)),
                }
            )
    return pd.DataFrame(rows)


def test_gat_equity_end_to_end(tmp_path) -> None:
    cfg = ProjectConfig()
    panel_flat = add_alpha_factors(_prices(), cfg)

    out = gat_equity_from_panel(
        panel_flat,
        SECTORS,
        cfg.backtest,
        k=5,
        window=40,
        top_k=4,
        epochs=2,
        train_ratio=0.6,
        out_path=str(tmp_path / "gat_equity.pt"),
    )

    # composite produced and merged on the alpha panel's own index
    assert COMPOSITE_NAME in out["panel"].columns
    assert out["panel"][COMPOSITE_NAME].notna().sum() > 0

    # the composite is evaluated through the same diagnostics as the singles
    assert COMPOSITE_NAME in set(out["diagnostics"]["alpha_name"])

    # all four gates are reported
    gates = out["gate_report"]
    for gate in ("value_added", "consistency", "uniqueness", "robustness"):
        assert gate in gates
        assert "passed" in gates[gate]
    assert (tmp_path / "gat_equity.pt").exists()

    # the no-learning anchors ride along through the same diagnostics
    assert out["panel"][ISLAND_MEAN_NAME].notna().sum() > 0
    assert out["panel"][UNIFORM_NAME].notna().sum() > 0
    ab = out["ab_report"]
    for key in ("gat", "uniform_mean", "island_mean", "attention_sharpe_value_add"):
        assert key in ab


def test_gat_equity_dynamic_graph_walk_forward(tmp_path) -> None:
    cfg = ProjectConfig()
    panel_flat = add_alpha_factors(_prices(), cfg)

    out = gat_equity_from_panel(
        panel_flat,
        SECTORS,
        cfg.backtest,
        k=5,
        window=40,
        top_k=4,
        epochs=1,
        train_ratio=0.6,
        graph="dynamic",
        retrain="walk-forward",
        oos_chunk=8,
        out_path=str(tmp_path / "gat_wf.pt"),
    )

    panel = out["panel"]
    assert COMPOSITE_NAME in panel.columns
    # walk-forward scores cover the OOS region (every snapshot gets a score)
    dates = sorted(panel["date"].unique())
    oos = panel[panel["date"] > dates[int(len(dates) * 0.6)]]
    assert oos[COMPOSITE_NAME].notna().all()
    assert (tmp_path / "gat_wf.pt").exists()
    assert "ab_report" in out
