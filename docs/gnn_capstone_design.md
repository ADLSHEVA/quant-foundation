# GNN Relational-Factor Capstone — Architecture Design (v1.0)

> Status: approved design draft. Supersedes the GNN sections of the
> existing `semester_plan.md` for the capstone scope. Implementation
> milestones in §8.
>
> **Implementation status (2026-06-10):** equity track M1+M2 complete and
> M3 first pass done on real data (3/4 gates pass; Value-added open) — see
> `gat_experiment_log.md` for the canonical experiment record and
> `CAPSTONE_STATUS.md` for the resume point. Scope narrowed to
> equity-first per ADR-0004 (energy deferred); the implemented module
> layout differs from §4 (it follows the ADR-0001/0002 seams:
> `graph/propagate.py`, `graph/training.py`, `graph/edges_equity.py`,
> `features/factor.py`, `models/gat.py`, `run_gat_equity.py`).

## 1. Thesis — "From Island Alphas to Relational Alphas"

Every factor in the current platform assumes **stocks (and markets) are
independent islands**. The equity registry (`features/registry.py`) is
entirely cross-sectional single-name math (`groupby(symbol)` time series),
and the energy track's cross-market signal
(`alpha_energy_cross_market_spread`) collapses the whole panel into a naive
`groupby(timestamp).mean()` — a *fully-connected, equally-weighted* graph.

This capstone introduces **Graph Attention Network (GAT) relational
factors** that turn the "pull-one-hair-move-the-whole-body" (Lead–Lag)
effect into computable graph signals, then uses the platform's existing
diagnostics framework to **prove the new factors carry orthogonal,
value-added alpha** over the island baselines.

The narrative axis (old island factors vs. new relational factors) maps
cleanly onto the existing A/B diagnostics (IC, value-added, turnover,
alpha-decay), which makes it a well-scoped 12-credit deliverable.

## 2. Two Heterogeneous Graphs, One Shared Kernel

GNN is applied to **both** existing tracks. They are genuinely different
graphs — heterogeneity is itself a capstone depth point.

| | Equity track | Energy track |
|---|---|---|
| Node | Stock (`symbol`) | Bidding zone / power market (`market`) |
| Cross-section | per `date` | per `timestamp` |
| Edge source | return correlation + sector co-membership (real supply-chain data = future work) | **ENTSO-E physical interconnector topology** + price correlation |
| Current baseline | 10 island cross-sectional factors | `cross_market_spread` = fully-connected mean |
| GNN upgrade | topology-weighted attention propagation | real grid topology replaces the naive mean |

The energy graph is the stronger physical story: power prices propagate
along **real cross-border transmission lines** between European zones, so
the adjacency matrix is grounded in physics, not estimated correlation.

A single GAT kernel (`graph/gat.py`) serves both tracks; only the
adjacency matrix and node features differ. One module absorbs one complete
capability — the interface to the rest of the pipeline does not change.

## 3. Decisions Locked

1. **Both markets** use GNN (equity + energy). A-shares are out of scope.
2. **GAT from the start** (dynamic attention), not a GCN warm-up phase.
   Implemented with **PyTorch Geometric (PyG)**.
3. **PyG ships as an optional extra** `pip install .[gnn]`, not a core
   dependency — keeps the traditional pipeline, CI, and day-to-day Docker
   image lightweight. See §6.
4. **Energy bidding zones expanded to ~15–30** (from the current 3:
   `DE_LU, CZ, FR`) via ENTSO-E `entsoe_domains` EIC codes. This is a
   **hard prerequisite** for GAT on the energy track — attention over 3
   nodes is statistically meaningless.

## 4. Module Layout

New, low-coupling package that reuses the existing panel contracts:

```
src/quant_alpha/graph/
├── edges_equity.py    # equity graph: correlation + sector co-membership
├── edges_energy.py    # energy graph: ENTSO-E interconnector topology
├── snapshot.py        # point-in-time graph slicing (anti ghost-edge), shared
├── gat.py             # PyG GAT kernel, shared by both tracks
├── train.py           # offline training loop, persists model weights
├── factors_equity.py  # wraps GAT output as AlphaDefinition
├── factors_energy.py  # appends GAT output into energy features
└── README.md
```

## 5. Integration Points (verified against current code)

The platform pipeline is:

```
fetch → add factors → backtest → DuckDB → dbt → Streamlit
              ↑
        GNN inserts ONLY here
```

Everything except the factor step is untouched and stays fast.

- **Equity** (`features/alpha_factors.py`): the panel is MultiIndex
  `[date, symbol]`. A GNN factor is just **one more column** computed per
  `(date, symbol)`. Wrap it as an `AlphaDefinition` (same frozen dataclass
  with a `compute` callable) and it flows through `run_long_short_backtest`,
  `evaluate_alpha_suite`, diagnostics, dbt marts, and the dashboard with
  **zero changes** downstream.
- **Energy** (`features/energy_alpha.py`): factors use
  `EnergyAlphaDefinition` (no `compute` callable — computed imperatively in
  `add_energy_alpha_features`). The GNN factor is appended as a column
  there. The energy pipeline already renames `timestamp→date`,
  `market→symbol` before calling the shared `run_long_short_backtest`, so
  the **backtest layer needs no changes** either.

## 6. Dependency & Performance Strategy

PyG + PyTorch is a heavy (~1 GB) dependency on a currently pure
pandas/numpy/duckdb stack. Containment plan:

- **Optional extra**: `[project.optional-dependencies] gnn = ["torch",
  "torch-geometric"]`. `pip install .` = lightweight traditional pipeline;
  `pip install .[gnn]` = full GNN stack.
- **Train / infer split** — the key to keeping production fast:
  - **Training** (learning attention weights): offline, infrequent
    (e.g. weekly/monthly retrain), persists model weights. This is the only
    genuinely slow step (minutes–tens of minutes; much faster on GPU).
  - **Inference** (emitting today's factor values): load saved weights →
    one forward pass per cross-section → seconds. The daily production
    pipeline does **inference only** and stays fast.
- CI suites for traditional factors and the lightweight K8s day-job
  container do **not** install PyG.

## 7. Risk Experiments (turn the article's pitfalls into deliverables)

Each becomes an explicit experiment in the final report:

- **Ghost edges** — edges must carry validity intervals and be sliced by
  date/timestamp. This is the reason `snapshot.py` exists.
- **Over-smoothing** — restrict to 1–2 hop neighbours; run a
  "layer depth vs. IC discrimination" decay experiment.
- **Asymmetric transmission** — directed graph; edges distinguish
  cost-side vs. demand-side (and upstream→downstream propagation that does
  not symmetrically reverse).

## 8. Milestones (align to semester weeks)

1. **M1 — Graph infrastructure**: expand energy bidding zones to ~15–30;
   `edges_equity.py` + `edges_energy.py` + `snapshot.py` + unit tests.
2. **M2 — GAT kernel**: `gat.py` (PyG) + `train.py` running; each track
   emits ≥1 GAT factor, registered into its registry.
3. **M3 — Strict A/B**: dual-track diagnostics proving GAT factors carry
   orthogonal value-add over the island baseline (equity) and the
   fully-connected-mean baseline (energy): value-added > 0, low correlation.
4. **M4 — Attention & risk**: attention-weight visualisation (the
   "macro-regime-adaptive" story) + the three risk experiments from §7.
5. **M5 — Report & dashboard**: final report (robustness, uniqueness,
   value-added, consistency) + a Streamlit "GAT vs. Baseline" comparison
   page per track.

## 9. Production Boundary (unchanged)

Research and portfolio demonstration only. Does not place orders, manage
broker state, or provide investment advice.
