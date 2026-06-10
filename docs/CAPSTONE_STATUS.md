# GNN Capstone — Progress Snapshot

Resume point for the GNN/GAT relational-factor extension. The original GitHub
repo (data-engineering platform) is prior work; everything below is the new
capstone contribution. Equity track first; energy deferred (ADR-0004).
Experiment record for the paper: `docs/gat_experiment_log.md`.

_Last updated: 2026-06-10._

## Status in one line

The equity end-to-end axis is **closed and tested**: alpha panel -> static graph
-> GAT training (MSE) -> composite -> four research gates, runnable via
`quant-alpha gat-equity`. **27 new-module tests passing.** Split hygiene is
locked (2026-06-10): one IS/OOS split date drives graph/training/evaluation,
valid sits inside IS with embargo on both sides, `fit` returns the
best-by-valid-IC model, and the shuffle-label leak check is an automated test
(`tests/test_leakage.py`, negative + positive control). See the ADR-0003
amendment.

## Decisions locked (see docs/adr/)

- **0001** — `propagate` seam: one snapshot in, one factor per node out;
  topology in, adapters weight internally; transform-only; directed; output is
  the factor (attention via a side method).
- **0002** — unified `Factor` + `FactorProvider` seam; canonical `(time, entity)`
  panel; energy wrapped as legacy provider; GNN factor is a provider.
- **0003** — GAT training objective: cross-sectionally standardised
  `forward_return(t+k)` label; MSE first, then IC loss; walk-forward + embargo
  (>= k); leakage-critical code is pure pandas + tested.
- **0004** — scope: equity end-to-end first; energy is documented extension + stubs.
- **0005** — equity graph: correlation top-k backbone + optional sector boost +
  `min_degree` fallback; node features = `_rank` alphas, cross-sectional median
  fill; label k in days; universe expanded to ~50 names with GICS sectors.

## New modules (all under src/quant_alpha/, English comments)

| File | Role |
|---|---|
| `graph/propagate.py` | `Propagator` seam; `UniformMeanPropagator` (baseline, A/B anchor); `GATPropagator` (wraps trained GATModel, torch lazy) |
| `graph/training.py` | torch-free leakage primitives: `cross_sectional_label`, `cross_sectional_median_fill`, `walk_forward_splits` (embargo), `rank_ic` |
| `graph/edges_equity.py` | `build_equity_topology` (corr top-k + sector + min_degree, leak-safe), `static_topology_for` |
| `features/factor.py` | unified `Factor`, `FactorProvider`, `apply_factors`, `propagate_over_panel`; `ExpressionFactorProvider`, `GraphFactorProvider`, `LegacyEnergyProvider` (stub) |
| `models/gat.py` | torch zone (needs `[gnn]`): `GATModel`, `GATConfig`, `CrossSection`, `FactorGraphDataset`, `build_sections`, `fit`, `composite_series`, `predict_panel`, losses, `time_ordered_split` |
| `run_gat_equity.py` | the main axis: `gat_equity_from_panel` (orchestration), `run_gat_equity` (CLI wrapper), `gate_report` (four gates) |

Config: `Universe.sectors` added (`config.py`); `configs/universe.yaml` = 50 names
+ sectors; `[gnn]` extra in `pyproject.toml` (torch, torch-geometric).

## Tests (tests/, run with the env note below)

`test_graph_propagate.py` (3) · `test_factor_provider.py` (4) · `test_training.py`
(6) · `test_edges_equity.py` (6) · `test_gat.py` (5) · `test_run_gat_equity.py` (1)
· `test_leakage.py` (2, shuffle + planted-signal controls)
= **27 passing** (torch tests `importorskip`, run here because torch is installed).

## How to run / resume

Environment quirks on this machine:
- Use the `py` launcher (Python 3.13). The bare `python` is a Microsoft Store stub.
- `torch` + `torch_geometric` ARE installed (CPU). `yfinance` installed
  2026-06-10 — the live `run_gat_equity` fetch path works (run it via a script,
  not the CLI: `cli.py` imports dlt/duckdb modules at the top). `duckdb` is
  still NOT installed, so duckdb-backed tests can't run here.
- Set `PYTHONPATH` to `src` (pytest also has `pythonpath=["src"]`).

```powershell
cd D:\AI_Models\quant-alpha-foundation
$env:PYTHONPATH = "D:\AI_Models\quant-alpha-foundation\src"
py -m pytest tests/test_run_gat_equity.py tests/test_gat.py tests/test_edges_equity.py tests/test_training.py tests/test_factor_provider.py tests/test_graph_propagate.py -q
```

CLI (needs yfinance for the live path): `quant-alpha gat-equity --offline --epochs 50`

## Demo result (synthetic prices, 50 names) — and how to read it

On random-walk synthetic data the gates correctly report **no edge**: Value-added
and Consistency FAIL (negative OOS IC ~ -0.06, Sharpe value-add negative);
Uniqueness and Robustness PASS. This is the desired behaviour — a leakage-safe
pipeline produces no false positives on noise. Real market data is where the GAT
gets a chance to add value.

## Real-data result (2026-06-10, yfinance 2021->now, 49 names, IC loss, 50 epochs)

**3 of 4 gates pass** — Uniqueness (max |corr| 0.257), Consistency (0.63),
Robustness (0.68); Value-added FAILS (composite OOS Sharpe 1.42 vs best
single 2.88, but it beats 9 of the 10 singles). Valid IC peaked 0.0756 at
epoch 32; best-epoch selection deployed that epoch (final epoch had decayed
to 0.0418 — the checkpoint fix nearly doubled the deployed valid IC).

**Full record — setup, training curve, all-factor diagnostics, limitations,
next experiments — lives in `docs/gat_experiment_log.md` (entry E5), the
canonical experiment log for the paper.** Artifacts:
`docs/results/2026-06-10_gat_real_run_diagnostics.csv` + the run script
alongside it; weights `data/warehouse/gat_equity.pt`.

## Architecture map

`infra.txt` (repo root) — full production framework as an OPM model (DOT).
Render: `dot -Tsvg infra.txt -o out.svg`.

## Next steps (prioritised)

1. ~~Switch MSE -> IC loss~~ **DONE** (2026-06-10): `loss="ic"` is the default
   in `gat_equity_from_panel` and the CLI (`--loss ic|mse`, MSE kept for A/B);
   leakage controls parametrised over both losses. IC loss converges cleanly at
   lr 1e-3 (planted-signal IC 0.99 vs MSE's 0.33) but is unstable early at 5e-3.
2. ~~Run on real data~~ **DONE** (2026-06-10): see "Real-data result" above.
   3/4 gates pass; Value-added is the open challenge. Remaining limitation to
   state honestly in the paper: the static graph gives early *training*
   snapshots a mild in-sample lookahead (graph built from correlations up to
   the split date), which does not affect OOS cleanliness — resolved by the
   dynamic graph in step 3.
3. **Attack the Value-added gate** — prioritised experiment queue in
   `docs/gat_experiment_log.md` ("Next experiments"): dynamic per-snapshot
   graph, walk-forward retraining, uniform-mean A/B anchor, seed/HP
   sensitivity, attention visualisation (M4).
4. Platform integration: composite into dbt marts; Streamlit "GAT vs
   Baseline" page.
5. Energy track (ADR-0004 deferred): hourly-return label variant, bidding-zone
   expansion, `edges_energy` interconnector topology.

## Timeline (proposed to advisor)

Final codebase by **2026-06-30**; paper first draft by **2026-07-15** (pending
advisor confirmation).
