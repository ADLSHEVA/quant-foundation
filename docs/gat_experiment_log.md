# GAT Relational Factor — Experiment Log (paper reference)

The canonical record of every GAT-equity experiment: setup, integrity
controls, numbers, and interpretation. This is the primary source for the
capstone paper's results section. Companion docs: design rationale in
`gnn_capstone_design.md`, decisions in `adr/`, resume point in
`CAPSTONE_STATUS.md`.

_Maintained chronologically; never rewrite an entry — append corrections._

---

## E1 — Synthetic negative control (2026-06, MSE loss)

**Setup.** 50 synthetic random-walk names (`generate_synthetic_prices`),
static correlation top-k graph, GAT (2 layers, hidden 64, 4 heads), MSE loss,
50 epochs, `quant-alpha gat-equity --offline`.

**Result.** The four gates correctly report **no edge**: Value-added and
Consistency FAIL (OOS IC ~ -0.06, negative Sharpe value-add); Uniqueness and
Robustness PASS (structural properties, not signal claims).

**Reading.** On data with no learnable cross-sectional structure, a
leakage-safe pipeline must find nothing — and it does. This is the first half
of the no-false-positives argument (see E3 for the second half).

---

## E2 — Split-hygiene fixes (2026-06-10, pre-real-data)

Not an experiment but a precondition for trusting any later numbers; recorded
here because the paper must describe the evaluation protocol.

1. **One split date** (was: three independent 0.7 ratios that happened to
   coincide). A single IS/OOS boundary now drives graph construction
   (`as_of`), model selection (`fit(train_idx, valid_idx)`), and the
   four-gate evaluation (`evaluate_alpha_suite(split_date=...)`).
2. **Valid inside IS, embargoed on both sides**:
   `train | embargo(k) | valid | embargo(k) | OOS`
   (`graph.training.is_constrained_split`, asserted at both the helper and
   the call site). The trailing embargo matters: valid labels reach `t+k`,
   so without it best-epoch selection would peek at the OOS window.
3. **`fit` returns the best-by-valid-IC model** (was: saved best weights but
   returned the final epoch — checkpoint selection was dead code).
   `out_path` now always holds the returned model's weights.
4. **Seam-guaranteed deterministic inference**: `GATPropagator.propagate`
   calls `model.eval()` itself.

Full rationale: ADR-0003 amendment (2026-06-10).

---

## E3 — Leakage controls, automated (2026-06-10)

`tests/test_leakage.py`, parametrised over both losses (MSE, IC). Synthetic
panel: 60 days x 12 names, fully-connected topology, k=2, fixed seeds.

- **Negative control** — shuffle each snapshot's labels across nodes,
  retrain: valid IC must satisfy |IC| < 0.25. Passes for both losses
  (IC ~ 0). If this ever fails, future information has leaked into features
  or graph — audit `build_sections`.
- **Positive control** — plant a recoverable signal (label = standardised
  feature 0): the same fit loop must recover it (valid IC > 0.3). Passes for
  both losses. This proves the negative control passes because there is
  nothing to learn, not because the trainer is broken.

E1 + E3 together form the complete "this pipeline produces no false
positives" argument — worth a subsection in the paper before any positive
result is claimed.

---

## E4 — IC-loss convergence probe (2026-06-10)

Question: does the IC loss (`-Pearson(pred, label)` per snapshot) train as
reliably as MSE? Probe on the planted-signal panel (E3 setup), GAT hidden 8 /
heads 2 unless noted, best-epoch selection on valid IC:

| loss | lr | epochs | valid IC |
|---|---|---|---|
| mse | 1e-3 | 40 | 0.233 |
| mse | 5e-3 | 40 | 0.334 |
| mse | 5e-3 | 120 | 0.334 (plateau) |
| ic | 1e-3 | 40 | 0.409 |
| ic | 5e-3 | 40 | **0.134 (unstable early)** |
| ic | 5e-3 | 120 | 0.994 |
| ic | 1e-3 | 120, hidden 16 heads 4 | 0.997 |
| ic | 1e-3 | 120, 1 layer | 1.000 |

**Findings.** (1) IC loss ultimately recovers the planted signal far better
than MSE (0.99+ vs 0.33 plateau) — expected, since it optimises the
evaluation metric directly. (2) It is unstable in the early epochs at high
lr (5e-3): the loss bounces around 0 for ~40 epochs before converging. At
lr 1e-3 convergence is clean from the start. Default `GATConfig.lr=1e-3` is
therefore kept for IC-loss training. (3) Consequence for the pipeline:
`loss="ic"` is now the default (ADR-0003 step 2), MSE retained via
`--loss mse` for A/B.

---

## E5 — First real-data run (2026-06-10)

**Setup.**
- Data: yfinance daily bars, 2021-01-01 → 2026-06-10, universe
  `configs/universe.yaml` (50 names, GICS sectors). ORCL failed to download
  (yfinance local cache lock) → **49 names**, 1,364 trading days, 66,836
  panel rows. Pipeline is N-agnostic; no code change needed.
- Features: the 10 island alphas' `_rank` columns, cross-sectional median
  fill (ADR-0005).
- Graph: static correlation top-k backbone (window 60d, top_k 8) + sector
  boost + min_degree fallback, built strictly from data before the split
  date (ADR-0005).
- Label: forward_return(t+5) (backtest `forward_return_days=5`),
  cross-sectionally z-scored (ADR-0003).
- Training: IC loss, lr 1e-3, 50 epochs, 2 GAT layers, hidden 64, heads 4,
  dropout 0.1; split per E2 (train | 5 | valid | 5 | OOS at 70% of dates);
  best-epoch selection on valid IC.
- Command: `docs/results/2026-06-10_run_real.py` (wraps
  `run_gat_equity(offline=False, epochs=50, loss="ic")`); weights
  `data/warehouse/gat_equity.pt`; diagnostics CSV
  `docs/results/2026-06-10_gat_real_run_diagnostics.csv`.

**Training behaviour.** Textbook curve: train IC loss falls monotonically
(-0.02 → -0.134); valid IC rises from ~0.03 to a peak of **0.0756 at epoch
32**, then decays to 0.0418 by epoch 49 (overfitting). Best-epoch selection
returned epoch 32 — without the E2 checkpoint fix the deployed model would
have been the epoch-49 one, with barely half the valid IC. The fix paid for
itself on the first real run.

**Four-gate result: 3 of 4 PASS.**

| Gate | Value | Result |
|---|---|---|
| Value-added | composite OOS Sharpe 1.42 vs best single 2.88 (value-add -1.46) | **FAIL** |
| Consistency | IS/OOS IC same sign; score 0.63 | PASS |
| Uniqueness | max abs corr vs singles 0.257 | PASS |
| Robustness | score 0.68 | PASS |

Composite OOS IC mean 0.0066, OOS IC IR 0.025, OOS Sharpe 1.42.

**OOS diagnostics, all factors** (full table in
`docs/results/2026-06-10_gat_real_run_diagnostics.csv`):

| alpha | OOS IC | OOS Sharpe |
|---|---|---|
| alpha_wq_010_gap_quality | 0.0172 | **2.88** |
| alpha_wq_002_volume_price_divergence | 0.0235 | 1.15 |
| **alpha_gat_composite** | **0.0066** | **1.42** |
| alpha_wq_001_reversal_rank | 0.0045 | 0.19 |
| alpha_wq_007_price_to_ma_reversion | -0.0065 | 0.30 |
| alpha_wq_003_intraday_range_position | 0.0049 | -0.44 |
| alpha_liquidity_020_volume_shock | 0.0063 | -1.15 |
| alpha_trend_021_medium_momentum | -0.0107 | -1.42 |
| alpha_wq_009_volume_weighted_return | -0.0268 | -1.70 |
| alpha_wq_008_overnight_gap | -0.0296 | -1.91 |
| alpha_risk_020_low_volatility | -0.0628 | -4.05 |

**Honest reading.**

1. The GAT composite is a **real, unique, consistent but modest** signal:
   positive OOS Sharpe (1.42, second-best of eleven), low correlation with
   every island alpha (max 0.257), IS/OOS sign-stable. It is genuinely new
   information, not a repackaging of its inputs.
2. It does **not** beat the best single island alpha
   (`alpha_wq_010_gap_quality`, OOS Sharpe 2.88). The Value-added gate as
   defined — beat the *max* of ten singles — is the strictest possible bar;
   the composite does beat 9 of the 10.
3. **Valid IC 0.076 vs OOS IC 0.007** quantifies decay across the ~1.6-year
   OOS window. Candidate explanations, in testable order: (a) the static
   graph (correlations frozen at the split date) goes stale over OOS —
   motivates the dynamic per-snapshot graph; (b) single train-period
   weights face regime shift — motivates walk-forward retraining; (c) the
   five-day horizon's cross-sectional signal is simply weak in this period
   (the islands' OOS ICs are also small, median |IC| ~ 0.01).

**Limitations to state in the paper.**

- Static graph: early *training* snapshots see a graph built from
  correlations up to the split date (mild in-sample lookahead). OOS
  cleanliness is unaffected (graph data strictly precedes the OOS window).
  Resolved by the dynamic graph refinement.
- One run, one seed, one universe; no confidence intervals yet. A seed/
  hyperparameter sensitivity pass belongs in the robustness section.
- 49 of 50 names (ORCL download failure) — immaterial but record it.
- yfinance daily bars, no survivorship-bias-free vendor; universe is
  hand-picked current large caps, so the level (not the relative A/B) of
  all Sharpes is optimistic.

---

## Next experiments (priority order)

1. **Dynamic per-snapshot graph** — rolling `as_of=t` topology; directly
   targets the failed Value-added gate and explanation (a) of the
   valid→OOS decay. Compare static vs dynamic on identical splits.
2. **Walk-forward retraining** — `graph.training.walk_forward_splits` is
   already built and tested; targets explanation (b).
3. **Uniform-mean A/B anchor** — run `UniformMeanPropagator` through the
   same four gates: is *attention* adding anything over naive neighbour
   averaging? (The capstone's core A/B; cheap to run.)
4. **Seed/HP sensitivity** — 5 seeds x {lr, hidden, heads} grid on the real
   panel; report mean +/- std of OOS IC, not point estimates.
5. **Attention visualisation (M4)** — `GATPropagator.last_attention`;
   qualitative story for the paper (which sectors/names attend to whom).
