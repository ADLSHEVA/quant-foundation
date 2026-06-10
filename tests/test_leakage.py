"""ADR-0003 leakage self-check, automated.

Two halves of one argument ("the pipeline produces no false positives"):

- **Negative control** — shuffle each snapshot's labels across nodes and
  retrain: valid IC must be ~0. A clearly positive IC means future information
  leaked into the features or the graph — audit ``build_sections``.
- **Positive control** — plant a recoverable signal in the label: the same
  fit loop must find it. This proves the negative control passes because
  there is nothing to learn, not because the trainer is broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch

from quant_alpha.graph.propagate import Topology
from quant_alpha.graph.training import is_constrained_split
from quant_alpha.models.gat import (
    FactorGraphDataset,
    GATConfig,
    build_sections,
    evaluate_ic,
    fit,
    ic_loss,
    mse_loss,
)

FEATURES = ("f0", "f1", "f2")
K = 2


def _panel(n_days: int = 60, entities: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(7)
    rows = []
    for i in range(entities):
        price = 100 + np.cumsum(rng.normal(0, 1, n_days))
        for d, dt in enumerate(dates):
            rows.append(
                {
                    "date": dt,
                    "symbol": f"S{i}",
                    "adj_close": float(price[d]),
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


LOSSES = {"mse": mse_loss, "ic": ic_loss}


def _fit_and_valid_ic(sections, epochs: int, loss: str, lr: float = 5e-3) -> float:
    ds = FactorGraphDataset(sections)
    train_idx, valid_idx = is_constrained_split(len(ds), embargo=K)
    assert len(valid_idx) > 0
    torch.manual_seed(0)
    cfg = GATConfig(
        in_dim=len(FEATURES), hidden_dim=8, heads=2, dropout=0.0,
        forward_k=K, epochs=epochs, lr=lr,
    )
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        model = fit(
            ds, cfg, loss_fn=LOSSES[loss], out_path=os.path.join(tmp, "leak.pt"),
            train_idx=train_idx, valid_idx=valid_idx,
        )
    # follow the trained model's device (fit defaults to cuda when available)
    device = next(model.parameters()).device
    return evaluate_ic(model, (ds[i] for i in valid_idx), device)


@pytest.mark.parametrize("loss", sorted(LOSSES))
def test_shuffled_labels_give_no_validation_ic(loss: str) -> None:
    panel = _panel()
    sections = build_sections(panel, _topology_for(panel), FEATURES, k=K)
    g = torch.Generator().manual_seed(123)
    for sec in sections:
        perm = torch.randperm(sec.label.shape[0], generator=g)
        sec.label = sec.label[perm]
        sec.mask = sec.mask[perm]

    ic = _fit_and_valid_ic(sections, epochs=5, loss=loss)
    assert abs(ic) < 0.25, f"shuffled labels but valid IC = {ic:.3f}: leakage suspected"


@pytest.mark.parametrize("loss", sorted(LOSSES))
def test_planted_signal_is_learned(loss: str) -> None:
    panel = _panel()
    sections = build_sections(panel, _topology_for(panel), FEATURES, k=K)
    for sec in sections:
        x0 = sec.x[:, 0]
        sec.label = (x0 - x0.mean()) / (x0.std() + 1e-8)
        sec.mask = torch.ones_like(sec.mask)

    # ic_loss converges cleanly at lr=1e-3 but is unstable early at 5e-3
    # (it eventually reaches IC ~0.99 there too, just past 40 epochs).
    ic = _fit_and_valid_ic(sections, epochs=40, loss=loss, lr={"mse": 5e-3, "ic": 1e-3}[loss])
    assert ic > 0.3, f"planted signal not recovered: valid IC = {ic:.3f}"
