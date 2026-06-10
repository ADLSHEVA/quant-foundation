"""End-to-end equity GAT relational-factor run (the main axis).

A standalone entry that wires the already-built pieces together without
touching the existing ``pipeline.py`` (ADR-0004 isolates this from the live
pipeline). Flow:

    alpha panel -> graph (static train-period, or dynamic per-snapshot)
        -> build_sections -> GAT training (IC loss; MSE kept for A/B;
        single fit or walk-forward refits) -> composite + uniform/island
        baselines -> merge into factor matrix -> evaluate_alpha_suite
        (the four gates) + attention-vs-uniform A/B report.

``gat_equity_from_panel`` holds the orchestration and is unit-testable on a
small panel; ``run_gat_equity`` is the thin data-fetching wrapper used by the
CLI. Requires the ``[gnn]`` extra.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_alpha.backtest.diagnostics import alpha_correlation, evaluate_alpha_suite
from quant_alpha.config import BacktestConfig, load_project_config, load_universe
from quant_alpha.features.alpha_factors import BASE_FACTOR_COLUMNS, add_alpha_factors
from quant_alpha.features.factor import propagate_over_panel
from quant_alpha.graph.edges_equity import rolling_topology_for, static_topology_for
from quant_alpha.graph.propagate import UniformMeanPropagator
from quant_alpha.graph.training import cross_sectional_median_fill, is_constrained_split
from quant_alpha.models.gat import (
    FactorGraphDataset,
    GATConfig,
    build_sections,
    composite_series,
    fit,
    ic_loss,
    mse_loss,
    walk_forward_composite_series,
)

LOSSES = {"ic": ic_loss, "mse": mse_loss}

COMPOSITE_NAME = "alpha_gat_composite"
ISLAND_MEAN_NAME = "alpha_island_mean"
UNIFORM_NAME = "alpha_uniform_composite"


def _baseline_columns(
    indexed: pd.DataFrame, topology_for, feature_cols: tuple[str, ...]
) -> tuple[pd.Series, pd.Series]:
    """The two no-learning anchors for the attention A/B.

    ``alpha_island_mean`` is the equal-weight composite of the same input
    alphas (no propagation); ``alpha_uniform_composite`` propagates it with
    uniform neighbour averaging over the same topology the GAT uses. The GAT's
    claim to value is whatever it adds over these — same inputs, same graph,
    no learned attention (ADR-0001's adapter-swap A/B).
    """
    filled = cross_sectional_median_fill(indexed, tuple(feature_cols))
    island = filled[list(feature_cols)].mean(axis=1).rename(ISLAND_MEAN_NAME)
    uniform = propagate_over_panel(
        island.to_frame(),
        UniformMeanPropagator(feature=ISLAND_MEAN_NAME),
        topology_for,
        (ISLAND_MEAN_NAME,),
    ).rename(UNIFORM_NAME)
    return island, uniform


def ab_report(diagnostics: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Attention-vs-uniform A/B: same inputs, same topology — what does
    learned attention add over uniform averaging (and over no propagation)?"""
    indexed = diagnostics.set_index("alpha_name")

    def row(name: str) -> dict:
        return {
            "oos_ic_mean": float(indexed.loc[name, "oos_ic_mean"]),
            "oos_sharpe": float(indexed.loc[name, "oos_sharpe"]),
        }

    corr = alpha_correlation(panel, [COMPOSITE_NAME, UNIFORM_NAME])
    pair = corr[
        (corr["alpha_left"] == COMPOSITE_NAME) & (corr["alpha_right"] == UNIFORM_NAME)
    ]
    gat, uniform = row(COMPOSITE_NAME), row(UNIFORM_NAME)
    return {
        "gat": gat,
        "uniform_mean": uniform,
        "island_mean": row(ISLAND_MEAN_NAME),
        "attention_sharpe_value_add": gat["oos_sharpe"] - uniform["oos_sharpe"],
        "gat_uniform_spearman": (
            float(pair["spearman_corr"].iloc[0]) if not pair.empty else float("nan")
        ),
    }


def gate_report(
    diagnostics: pd.DataFrame,
    panel: pd.DataFrame,
    single_cols: list[str],
    composite_name: str = COMPOSITE_NAME,
) -> dict:
    """Score the GAT composite against the four research gates, reusing the
    existing diagnostics (value_added / consistency / robustness) plus a
    correlation-based uniqueness check."""
    indexed = diagnostics.set_index("alpha_name")
    comp = indexed.loc[composite_name]
    best_single = float(indexed.loc[single_cols, "oos_sharpe"].max())
    composite_sharpe = float(comp["oos_sharpe"])

    corr = alpha_correlation(panel, single_cols + [composite_name])
    pair = corr[(corr["alpha_left"] == composite_name) & (corr["alpha_right"].isin(single_cols))]
    max_abs_corr = float(pair["spearman_corr"].abs().max()) if not pair.empty else float("nan")

    same_sign = comp["is_oos_ic_same_sign"]
    return {
        "composite_oos_ic_mean": float(comp["oos_ic_mean"]),
        "composite_oos_ic_ir": float(comp["oos_ic_ir"]),
        "value_added": {
            "composite_oos_sharpe": composite_sharpe,
            "best_single_oos_sharpe": best_single,
            "sharpe_value_added": composite_sharpe - best_single,
            "passed": composite_sharpe > best_single,
        },
        "consistency": {
            "is_oos_ic_same_sign": None if same_sign is None else bool(same_sign),
            "consistency_score": float(comp["consistency_score"]),
            "passed": float(comp["consistency_score"]) >= 0.5,
        },
        "uniqueness": {
            "max_abs_corr_vs_single": max_abs_corr,
            "passed": max_abs_corr < 0.7,
        },
        "robustness": {
            "robustness_score": float(comp["robustness_score"]),
            "passed": float(comp["robustness_score"]) >= 0.5,
        },
    }


def gat_equity_from_panel(
    panel_flat: pd.DataFrame,
    sectors: dict[str, str],
    backtest_cfg: BacktestConfig,
    *,
    k: int | None = None,
    window: int = 60,
    top_k: int = 8,
    depth: int = 2,
    epochs: int = 50,
    train_ratio: float = 0.7,
    loss: str = "ic",
    graph: str = "static",
    retrain: str = "single",
    oos_chunk: int = 63,
    hidden_dim: int = 64,
    heads: int = 4,
    lr: float = 1e-3,
    out_path: str = "data/warehouse/gat_equity.pt",
) -> dict:
    """Graph -> train -> composite -> four gates, given an alpha panel.

    ``panel_flat`` is the flat add_alpha_factors output (date/symbol columns,
    the ``_rank`` columns, and forward_return). The label horizon ``k`` defaults
    to the backtest's forward_return_days so training and evaluation align.
    ``loss`` is ``"ic"`` (default, ADR-0003 step 2: aligned with the rank-IC
    metric now the pipeline is validated leak-free) or ``"mse"`` (the step-1
    bring-up objective, kept for A/B). ``graph`` is ``"static"`` (one graph
    frozen at the split date) or ``"dynamic"`` (rebuilt as of each snapshot).
    ``retrain`` is ``"single"`` (one fit) or ``"walk_forward"`` (refit every
    ``oos_chunk`` snapshots through the OOS window).
    """
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {sorted(LOSSES)}, got {loss!r}")
    if graph not in ("static", "dynamic"):
        raise ValueError(f"graph must be 'static' or 'dynamic', got {graph!r}")
    retrain = retrain.replace("-", "_")
    if retrain not in ("single", "walk_forward"):
        raise ValueError(f"retrain must be 'single' or 'walk_forward', got {retrain!r}")
    k = k or backtest_cfg.forward_return_days
    feature_cols = tuple(f"{name}_rank" for name in BASE_FACTOR_COLUMNS)
    indexed = panel_flat.set_index(["date", "symbol"]).sort_index()

    # Single source of truth for the IS/OOS boundary. The same split date
    # drives graph construction (as_of), model selection (train/valid inside
    # IS only), and the four-gate evaluation (split_date) — three places that
    # previously each assumed their own 0.7.
    dates = sorted(indexed.index.get_level_values(0).unique())
    n_is = int(len(dates) * train_ratio) + 1  # IS = dates[:n_is], OOS strictly after
    split_date = dates[n_is - 1]
    if graph == "dynamic":
        topology_for = rolling_topology_for(
            indexed, sectors, return_col="ret_1d", window=window, top_k=top_k
        )
    else:
        topology_for = static_topology_for(
            indexed, sectors, as_of=split_date, return_col="ret_1d", window=window, top_k=top_k
        )

    dataset = FactorGraphDataset(
        build_sections(indexed, topology_for, feature_cols, k=k, price_col="adj_close")
    )
    gcfg = GATConfig(
        in_dim=len(feature_cols), hidden_dim=hidden_dim, heads=heads,
        num_layers=depth, forward_k=k, lr=lr, epochs=epochs,
    )
    if retrain == "walk_forward":
        composite = walk_forward_composite_series(
            dataset, gcfg, n_is=n_is, oos_chunk=oos_chunk,
            loss_fn=LOSSES[loss], out_path=out_path, name=COMPOSITE_NAME,
        )
    else:
        # Snapshot index t maps 1:1 to dates[t]; valid sits at the end of IS with
        # an embargo of k on both sides so its labels never reach the OOS window.
        train_idx, valid_idx = is_constrained_split(n_is, embargo=k)
        if len(valid_idx):
            assert train_idx.stop + k <= valid_idx.start, "train labels reach into valid"
            assert valid_idx.stop + k <= n_is, "valid labels reach into the OOS window"
        model = fit(
            dataset, gcfg, loss_fn=LOSSES[loss], out_path=out_path,
            train_idx=train_idx, valid_idx=valid_idx,
        )
        composite = composite_series(model, dataset, name=COMPOSITE_NAME)

    panel = panel_flat.merge(
        composite.rename(COMPOSITE_NAME).reset_index(), on=["date", "symbol"], how="left"
    )
    island, uniform = _baseline_columns(indexed, topology_for, feature_cols)
    panel = panel.merge(island.reset_index(), on=["date", "symbol"], how="left")
    panel = panel.merge(uniform.reset_index(), on=["date", "symbol"], how="left")

    alpha_cols = list(BASE_FACTOR_COLUMNS) + [ISLAND_MEAN_NAME, UNIFORM_NAME, COMPOSITE_NAME]
    diagnostics, alpha_metrics, _ = evaluate_alpha_suite(
        panel, alpha_cols, backtest_cfg, split_date=str(split_date)
    )
    return {
        "panel": panel,
        "diagnostics": diagnostics,
        "alpha_metrics": alpha_metrics,
        "gate_report": gate_report(diagnostics, panel, list(BASE_FACTOR_COLUMNS)),
        "ab_report": ab_report(diagnostics, panel),
        "weights_path": out_path,
    }


def run_gat_equity(
    config_path: Path,
    root: Path,
    offline: bool = True,
    **kwargs,
) -> dict:
    """Fetch prices, compute island alphas, then run the GAT relational layer."""
    from quant_alpha.ingestion.yahoo import fetch_prices

    cfg = load_project_config(config_path, root=root)
    universe = load_universe(cfg.universe_path)
    prices = fetch_prices(cfg, universe, offline=offline)
    panel_flat = add_alpha_factors(prices, cfg)
    return gat_equity_from_panel(
        panel_flat,
        universe.sectors,
        cfg.backtest,
        out_path=str(cfg.duckdb_path.parent / "gat_equity.pt"),
        **kwargs,
    )
