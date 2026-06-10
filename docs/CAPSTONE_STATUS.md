# GNN Capstone — Progress Snapshot

Resume point for the GNN/GAT relational-factor extension. The original GitHub
repo (data-engineering platform) is prior work; everything below is the new
capstone contribution. Equity track first; energy deferred (ADR-0004).
Experiment record for the paper: `docs/gat_experiment_log.md`.

_Last updated: 2026-06-10._

## Status in one line

The equity end-to-end axis is **closed, tested, ablated, seed-qualified, and
HP-tuned on real data** (experiments E1-E9, all in
`docs/gat_experiment_log.md`): alpha panel -> graph (static default; dynamic
available, rejected by ablation) -> GAT training (IC loss default; single or
walk-forward refits) -> composite + no-learning anchors -> four research
gates + attention A/B, runnable via `quant-alpha gat-equity`. **35
new-module tests passing.** Split hygiene locked (ADR-0003 amendment);
leakage controls automated; HP selection used valid IC only and transferred
to OOS (E9). Headline: **attention value-add over the uniform anchor is
positive in 20/20 seeded runs**; best known setup (static graph + IC loss +
walk-forward + lr 3e-3/hidden 64/heads 2/layers 2) reads **OOS IC 0.0179 +/-
0.0145, OOS Sharpe 1.30 +/- 0.73** over 5 seeds. Value-added (vs best
single, 3.07) is the one open gate.

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
| `graph/propagate.py` | `Propagator` seam; `UniformMeanPropagator` (baseline, A/B anchor); `GATPropagator` (wraps trained GATModel, torch lazy; `last_attention` exposes head-layer softmax for M4) |
| `graph/training.py` | torch-free leakage primitives: `cross_sectional_label`, `cross_sectional_median_fill`, `walk_forward_splits` (embargo), `is_constrained_split` (valid inside IS), `rank_ic` |
| `graph/edges_equity.py` | `build_equity_topology` (corr top-k + sector + min_degree, leak-safe), `static_topology_for`, `rolling_topology_for` (dynamic, point-in-time) |
| `features/factor.py` | unified `Factor`, `FactorProvider`, `apply_factors`, `propagate_over_panel`; `ExpressionFactorProvider`, `GraphFactorProvider`, `LegacyEnergyProvider` (stub) |
| `models/gat.py` | torch zone (needs `[gnn]`): `GATModel` (+`forward_with_attention`), `GATConfig`, `CrossSection`, `FactorGraphDataset`, `build_sections`, `fit`, `composite_series`, `walk_forward_composite_series`, `predict_panel`, losses, `time_ordered_split` |
| `run_gat_equity.py` | the main axis: `gat_equity_from_panel` (orchestration; `loss`/`graph`/`retrain` switches), `run_gat_equity` (CLI wrapper), `gate_report` (four gates), `ab_report` + `_baseline_columns` (attention A/B anchors) |

Config: `Universe.sectors` added (`config.py`); `configs/universe.yaml` = 50 names
+ sectors; `[gnn]` extra in `pyproject.toml` (torch, torch-geometric).

## Tests (tests/, run with the env note below)

`test_graph_propagate.py` (3) · `test_factor_provider.py` (4) · `test_training.py`
(6) · `test_edges_equity.py` (9) · `test_gat.py` (7) · `test_run_gat_equity.py` (2)
· `test_leakage.py` (4, shuffle + planted-signal controls x both losses)
= **35 passing** (torch tests `importorskip`, run here because torch is installed).

## How to run / resume

Environment quirks on this machine:
- Use the `py` launcher (Python 3.13). The bare `python` is a Microsoft Store stub.
- `torch 2.12.0+cu126` + `torch_geometric` installed; **CUDA works** (RTX
  4060 Laptop 8GB, driver CUDA 12.6; swapped from the CPU wheel 2026-06-10).
  `fit` auto-selects cuda; `_dataset_to_device` pre-moves all sections once
  so per-snapshot transfers don't eat the gain. `yfinance` installed
  2026-06-10 — the live `run_gat_equity` fetch path works (run it via a script,
  not the CLI: `cli.py` imports dlt/duckdb modules at the top). `duckdb` is
  still NOT installed, so duckdb-backed tests can't run here.
- Set `PYTHONPATH` to `src` (pytest also has `pythonpath=["src"]`).
- Parallel experiment runs: 32 logical cores but only **15GB RAM** — a worker
  running the full `gat_equity_from_panel` (training + evaluate_alpha_suite)
  peaks ~2GB, so cap at **3 concurrent full-pipeline workers** (8 OOM'd with
  BrokenProcessPool). Train-only workers (HP grid) are light; 8 are fine.
  Cap `torch.set_num_threads(4-8)` per worker — on these tiny graphs fewer
  threads is *faster* (4-thread runs beat 32-thread by ~30%).

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

Done (all 2026-06-10, full records in `docs/gat_experiment_log.md`):
IC loss default (E4) · first real-data run (E5) · 2x2 graph-x-retraining
ablation with A/B anchors (E6) · seed sensitivity (E7) · GPU benchmark —
negative, CPU stays (E8) · HP grid by valid IC + OOS winner validation (E9)
· attention plumbing (`last_attention`) · split hygiene + automated leakage
controls (E2/E3).

Remaining, in order:

1. **Attention story (M4)** — the qualitative analysis on real data (which
   sectors/names attend to whom over time; regime shifts at fold
   boundaries), using the tested `GATPropagator.last_attention` hook. The
   paper's narrative centrepiece.
2. **Value-added gate variants** — the strict max-of-singles bar (3.07) is
   the one open gate; add mean-of-singles and marginal-contribution-to-a-
   multifactor-portfolio readings before concluding the composite adds
   nothing beyond the best island alpha.
3. **WF-vs-single significance** — more seeds or a paired-across-seeds test
   to upgrade "directionally helpful" (consistent in two paired comparisons,
   E7+E9) into a significance claim, if it holds.
4. **Platform integration (M5)** — composite into dbt marts; Streamlit
   "GAT vs Baseline" page (now has real content: E6 matrix, E7/E9 seed
   distributions, attention heatmaps).
5. **Paper assembly** — the evidence map (C1-C10) and narrative order are
   ready in the experiment log; limitations list in E5 + static-graph
   lookahead note. Data-source upgrade (survivorship-bias-free vendor) if
   time permits.
6. Energy track (ADR-0004 deferred): hourly-return label variant,
   bidding-zone expansion, `edges_energy` interconnector topology — confirm
   scope with the advisor before investing.

## Timeline (proposed to advisor)

Final codebase by **2026-06-30**; paper first draft by **2026-07-15** (pending
advisor confirmation).
