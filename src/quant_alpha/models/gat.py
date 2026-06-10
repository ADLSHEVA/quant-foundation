"""GAT composite-alpha model (the torch network and its training loop).

This is the M2 modelling layer and the only torch-dependent module in the
project; it lives behind the ``[gnn]`` extra. The factor pipeline never imports
it eagerly — it reaches the trained network through the torch-free
``graph/propagate.py::GATPropagator`` adapter on the propagate seam.

Role within the established architecture:
    - features/   : the existing 10 equity + 8 energy alphas are the node
                    features (model input), not competitors to this model.
    - graph/      : the propagate seam and its pandas adapters. The trained
                    ``GATModel`` here is wrapped by ``GATPropagator`` so a
                    ``GraphFactorProvider`` can emit a relational composite alpha.
    - backtest/   : the composite score this model produces is fed to the
                    existing walk-forward IC and the four research gates.

Pipeline (as fixed in ADR-0001..0003):
    input   : one snapshot t, nodes = instruments, features = the alpha values
              at t (history only)
    graph   : the relation graph at t (sector / historical correlation /
              liquidity), built from data at or before t
    output  : one composite score per node
    label   : forward_return(t+k), cross-sectionally standardised, supervision
              only, never a feature
    loss    : MSE to bring the pipeline up, then IC loss to align with RankIC
    splits  : walk-forward + embargo (>= k), matching backtest/ for leakage safety
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from quant_alpha.graph.propagate import Topology
from quant_alpha.graph.training import cross_sectional_label, cross_sectional_median_fill

try:
    from torch_geometric.nn import GATConv
except ImportError:  # pragma: no cover
    GATConv = None


# --------------------------------------------------------------------------- #
# Config — mirrors the configs/ YAML convention; can be injected from
# second_foundation_project.yaml.
# --------------------------------------------------------------------------- #
@dataclass
class GATConfig:
    in_dim: int                 # number of input alphas (equity 10 / energy 8)
    hidden_dim: int = 64
    heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    forward_k: int = 10         # forward_return horizon; embargo must be >= this
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50


# --------------------------------------------------------------------------- #
# Data — one snapshot is one sample.
# --------------------------------------------------------------------------- #
@dataclass
class CrossSection:
    """A single snapshot t. x / edge_index use data at or before t; label uses t+k."""

    t: int                      # integer position, used only for ordering/splits
    x: torch.Tensor             # [N, F] node features = alpha matrix at t
    edge_index: torch.Tensor    # [2, E] relation graph at t (history only)
    label: torch.Tensor         # [N] forward_return, cross-sectionally standardised
    mask: torch.Tensor          # [N] bool, valid instruments (drop halted/missing/new)
    symbols: list[str] | None = None  # instrument codes, for backtest alignment
    time: object = None         # the real date/timestamp, for (date, symbol) alignment


class FactorGraphDataset(Dataset):
    """Alpha panel organised as a time-ordered sequence of snapshots.

    The caller (a builder; see ``build_sections``) must guarantee that x and
    edge_index use only data at or before t, that label is forward_return over
    close[t+k]/close[t]-1 standardised cross-sectionally, and that snapshots are
    ascending in t — the walk-forward split relies on that order.
    """

    def __init__(self, sections: list[CrossSection]):
        self.sections = sorted(sections, key=lambda s: s.t)

    def __len__(self) -> int:
        return len(self.sections)

    def __getitem__(self, idx: int) -> CrossSection:
        return self.sections[idx]


def build_sections(
    panel: pd.DataFrame,
    topology_for,
    feature_cols: tuple[str, ...],
    k: int,
    price_col: str = "adj_close",
    label_method: str = "zscore",
) -> list[CrossSection]:
    """Bridge a ``(time, entity)`` alpha panel into a list of CrossSection.

    Reuses the torch-free leakage-safe label from graph/training.py and the same
    topology source a GraphFactorProvider uses. A node is masked out when any of
    its features or its label is missing.
    """
    panel = panel.sort_index()
    label = cross_sectional_label(panel, k=k, price_col=price_col, method=label_method)
    panel = cross_sectional_median_fill(panel, tuple(feature_cols))

    sections: list[CrossSection] = []
    for t, (time, cross) in enumerate(panel.groupby(level=0)):
        nodes = list(cross.index.get_level_values(1))
        position = {node: i for i, node in enumerate(nodes)}

        x_np = cross[list(feature_cols)].to_numpy(dtype="float32")
        y_np = label.loc[cross.index].to_numpy(dtype="float32")
        valid = np.isfinite(x_np).all(axis=1) & np.isfinite(y_np)

        topology: Topology = topology_for(time)
        edges = [
            (position[s], position[d])
            for (s, d, _w) in topology.edges
            if s in position and d in position
        ]
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        sections.append(
            CrossSection(
                t=t,
                x=torch.from_numpy(np.nan_to_num(x_np)),
                edge_index=edge_index,
                label=torch.from_numpy(np.nan_to_num(y_np)),
                mask=torch.from_numpy(valid),
                symbols=nodes,
                time=time,
            )
        )
    return sections


# --------------------------------------------------------------------------- #
# Leakage-safe split — walk-forward + embargo, same discipline as backtest/.
# (Rolling folds live in graph/training.py::walk_forward_splits; this is the
#  single train/valid/test split a training run uses.)
# --------------------------------------------------------------------------- #
def time_ordered_split(
    n: int,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    embargo: int = 0,
) -> tuple[range, range, range]:
    """Strict time order, never shuffled. ``embargo`` should be >= forward_k so a
    train label window cannot overlap the next split's features."""
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = range(0, n_train)
    valid = range(n_train + embargo, n_train + embargo + n_valid)
    test = range(n_train + embargo + n_valid + embargo, n)
    return train, valid, test


def _subset(ds: FactorGraphDataset, idx: range) -> Iterator[CrossSection]:
    for i in idx:
        yield ds[i]


# --------------------------------------------------------------------------- #
# Model.
# --------------------------------------------------------------------------- #
class GATModel(nn.Module):
    """Stacked GAT over the snapshot graph; one scalar composite alpha per node."""

    def __init__(self, cfg: GATConfig):
        super().__init__()
        assert GATConv is not None, "torch_geometric is required (install the [gnn] extra)"
        self.cfg = cfg
        h, heads = cfg.hidden_dim, cfg.heads

        self.layers = nn.ModuleList()
        if cfg.num_layers >= 2:
            self.layers.append(GATConv(cfg.in_dim, h, heads=heads, dropout=cfg.dropout))
            for _ in range(cfg.num_layers - 2):
                self.layers.append(GATConv(h * heads, h, heads=heads, dropout=cfg.dropout))
            last_in = h * heads
        else:
            last_in = cfg.in_dim
        self.head = GATConv(last_in, 1, heads=1, concat=False, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.elu(layer(x, edge_index))
            x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        return self.head(x, edge_index).squeeze(-1)  # [N]


# --------------------------------------------------------------------------- #
# Losses — MSE to bring the pipeline up, then IC loss to align with RankIC.
# --------------------------------------------------------------------------- #
def mse_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, label)


def ic_loss(pred: torch.Tensor, label: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Within-snapshot negative Pearson IC. RankIC needs a differentiable soft
    rank; Pearson is the first step."""
    p = pred - pred.mean()
    y = label - label.mean()
    ic = (p * y).sum() / (p.norm() * y.norm() + eps)
    return -ic


# --------------------------------------------------------------------------- #
# Train / evaluate.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_ic(model: nn.Module, sections: Iterable[CrossSection], device) -> float:
    """Mean per-snapshot IC, matching the backtest/ walk-forward IC convention."""
    model.eval()
    ics = []
    for sec in sections:
        if int(sec.mask.sum()) < 2:  # need >= 2 nodes for a cross-sectional IC
            continue
        pred = model(sec.x.to(device), sec.edge_index.to(device))[sec.mask]
        ics.append(-ic_loss(pred, sec.label.to(device)[sec.mask]).item())
    return sum(ics) / max(len(ics), 1)


def train_one_epoch(model, sections, optimizer, device, loss_fn) -> float:
    model.train()
    total, n = 0.0, 0
    for sec in sections:
        if int(sec.mask.sum()) < 2:  # skip warmup / empty cross-sections
            continue
        pred = model(sec.x.to(device), sec.edge_index.to(device))[sec.mask]
        loss = loss_fn(pred, sec.label.to(device)[sec.mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def fit(
    ds: FactorGraphDataset,
    cfg: GATConfig,
    device=None,
    loss_fn=mse_loss,
    out_path: str = "data/warehouse/gat_best.pt",
    train_idx: range | None = None,
    valid_idx: range | None = None,
) -> GATModel:
    """Train, select the best-by-valid-IC epoch, and return that model.

    ``out_path`` always holds the state_dict of the returned model. Callers
    that also evaluate an OOS window (run_gat_equity) must pass
    ``train_idx``/``valid_idx`` constrained to the in-sample window
    (``graph.training.is_constrained_split``) so model selection never sees
    OOS data; the default ``time_ordered_split`` fallback is for standalone
    training only. When valid has no usable snapshot, selection is skipped and
    the final-epoch model is returned.

    Start with the default ``mse_loss`` to confirm the loss falls and there is
    no leak (see ``tests/test_leakage.py``); switch to ``ic_loss`` once the
    pipeline is validated.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_idx is None or valid_idx is None:
        train_idx, valid_idx, _ = time_ordered_split(len(ds), embargo=cfg.forward_k)

    model = GATModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    valid_usable = [i for i in valid_idx if int(ds[i].mask.sum()) >= 2]
    if not valid_usable:
        print("fit: no usable valid snapshot — best-epoch selection skipped")

    best_ic = float("-inf")
    best_state: dict | None = None
    for ep in range(cfg.epochs):
        tr = train_one_epoch(model, _subset(ds, train_idx), opt, device, loss_fn)
        va = (
            evaluate_ic(model, (ds[i] for i in valid_usable), device)
            if valid_usable
            else float("nan")
        )
        print(f"epoch {ep:02d}  train_loss={tr:.4f}  valid_IC={va:.4f}")
        if valid_usable and va > best_ic:
            best_ic = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_path)
    return model


@torch.no_grad()
def composite_series(
    model: GATModel,
    ds: FactorGraphDataset,
    device=None,
    name: str = "alpha_gat_composite",
) -> pd.Series:
    """The GAT composite as one ``(date, symbol)``-indexed column.

    This is the four-gates interface: the score is emitted on the same
    ``(date, symbol)`` index the alpha panel uses (from ``CrossSection.time`` and
    ``.symbols``, never reconstructed), so it can be appended as one more column
    and handed to ``evaluate_alpha_suite`` — which does the IS/OOS split itself,
    keeping Consistency aligned with the single factors. Scored over all
    snapshots; the honest read is the OOS slice that ``evaluate_alpha_suite``
    reports.
    """
    device = device or torch.device("cpu")
    model.eval()
    parts: list[pd.Series] = []
    for sec in ds:
        score = model(sec.x.to(device), sec.edge_index.to(device)).cpu().numpy()
        index = pd.MultiIndex.from_arrays(
            [[sec.time] * len(sec.symbols), sec.symbols], names=["date", "symbol"]
        )
        parts.append(pd.Series(score, index=index, name=name))
    return pd.concat(parts) if parts else pd.Series(name=name, dtype=float)


@torch.no_grad()
def predict_panel(
    model: GATModel, ds: FactorGraphDataset, idx: range, device
) -> dict[int, dict[str, float]]:
    """Composite alpha scores as {t: {symbol: score}} for the backtest layer to
    run long-short + walk-forward IC + the four gates."""
    model.eval()
    out: dict[int, dict[str, float]] = {}
    for i in idx:
        sec = ds[i]
        score = model(sec.x.to(device), sec.edge_index.to(device)).cpu()
        syms = sec.symbols or [str(j) for j in range(len(score))]
        out[sec.t] = {s: float(v) for s, v, m in zip(syms, score, sec.mask) if m}
    return out


# --------------------------------------------------------------------------- #
# Research-gate hooks (see docs/alpha_research.md):
#   Value-added : predict_panel -> backtest composite OOS Sharpe > best single factor
#   Consistency : IS IC and OOS IC same sign and comparable magnitude
#   Computed by the existing backtest/ module; only the interface point is marked
#   here, not re-implemented.
#
# Leakage self-check (run before trusting any result):
#   Shuffle each snapshot's label across nodes, retrain; valid IC should be ~0.
#   If it stays clearly positive, future information leaked into the features or
#   the graph — audit the section builder.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # sections = build_sections(panel, topology_for, feature_cols, k=10)
    sections: list[CrossSection] = []
    ds = FactorGraphDataset(sections)
    if len(ds) == 0:
        raise SystemExit("Provide sections: convert the alpha panel via build_sections().")
    cfg = GATConfig(in_dim=ds[0].x.shape[1])
    fit(ds, cfg)
