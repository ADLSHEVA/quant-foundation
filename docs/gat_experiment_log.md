# GAT Relational Factor — Experiment Log (paper reference)

The canonical record of every GAT-equity experiment: setup, integrity
controls, numbers, and interpretation. This is the primary source for the
capstone paper's results section. Companion docs: design rationale in
`gnn_capstone_design.md`, decisions in `adr/`, resume point in
`CAPSTONE_STATUS.md`.

_Maintained chronologically; never rewrite an entry — append corrections._

---

## Paper evidence map

Every claim the paper can make, with the experiment that supports it and the
artifact to cite. Update this table as entries land; a claim without a row
here is not yet supported.

| # | Claim | Evidence | Artifacts |
|---|---|---|---|
| C1 | The pipeline is leakage-safe and produces no false positives | E1 (synthetic negative control: four gates correctly find nothing on random walks) + E3 (automated shuffle-label negative control AND planted-signal positive control, both losses) + E2 (split protocol: one split date, valid inside IS, embargo both sides) | `tests/test_leakage.py`, ADR-0003 + amendment |
| C2 | IC loss is the right training objective: it directly optimises the evaluation metric and recovers signal far better than MSE (0.99 vs 0.33 planted-signal IC), at lr 1e-3 | E4 probe table | E4 entry; `LOSSES` in `run_gat_equity.py` |
| C3 | **Core thesis: learned attention adds value over naive relational propagation.** Same inputs, same topology: GAT beats the uniform-mean anchor in every E6 cell AND in **30/30 seeded runs across two HP configs and two devices** (+0.69 to +3.03 OOS Sharpe), with scores near-uncorrelated with the anchor (Spearman -0.29..-0.14) — it learns structure, not smoothing. Both no-learning anchors are negative OOS | E6 (all cells) + E7 + E9/E9b (all seeds) | `2026-06-10_matrix_summary.json`, `..._seed_sensitivity.csv`, `..._winner_validation_{gpu,cpu}.csv` |
| C4 | The relational composite is a real, unique, consistent signal: passes Uniqueness (max corr vs singles 0.26) and, with walk-forward, Consistency + Robustness — 3/4 gates | E5, E6 (static_wf) | matrix diagnostics CSVs |
| C5 | Honest negative: it does not (yet) beat the best single island alpha (the strictest Value-added bar); best config narrows the gap from -2.03 to -1.12 Sharpe while beating 9 of 10 singles | E5, E6 | gate reports in summary JSON |
| C6 | **Negative result: dynamic per-snapshot correlation graphs hurt** — per-date correlation estimates are noisy; the frozen train-period graph acts as a regulariser. Walk-forward retraining improves the mean in **all three** paired comparisons and reaches significance in the tuned config (E9b same-device: IC t~3.9, Sharpe t~2.4, n=5/arm) — claim "significant in the tuned config, directionally consistent everywhere" | E6 2x2 ablation + E7 + E9/E9b | E6/E7/E9 entries + artifacts |
| C7 | Model selection by valid IC matters in practice: valid IC peaked at epoch 32 (0.0756) then decayed to 0.0418 — best-epoch checkpointing nearly doubled deployed valid IC on the first real run | E5 training curve | E5 entry |
| C8 | Single-seed point estimates are not paper-grade: across 5 seeds OOS Sharpe spans 0.36-1.04 (single) and 0.13-1.95 (WF); same seed on a different device also diverges (E8). All headline numbers are reported as mean +/- std over >= 5 seeds; the long-short Sharpe is far more seed-stable in sign than the rank-IC | E7, E8 | `2026-06-10_seed_sensitivity.csv` |
| C9 | **The evaluation protocol is self-validating, with nuance**: HPs selected purely on the IS-internal valid IC transferred to OOS in the walk-forward arm (same-device: IC 0.0148 +/- 0.0043 vs default's 0.0055 +/- 0.0147 — 2.7x mean, 1/3 variance) but not in the single arm (a wash). Model selection never touched the OOS window. The initial both-arms read was a device artifact, caught and corrected (E9 correction) | E9 + E9b | `2026-06-10_hp_grid_valid_ic.csv` + `..._winner_validation_{gpu,cpu}.csv` |
| C10 | The HP surface is flat near the top (six configs within 0.10-0.115 valid IC); the only structural requirement is **2 GAT layers** (all top-6). Results are robust to reasonable HP choices — a robustness point, and a caution against HP-tuning theatre | E9 grid | `2026-06-10_hp_grid_valid_ic.csv` |

**Suggested paper narrative order** (each step cites the rows above):
methods & seams (ADRs) -> evaluation protocol & leakage controls (C1, C7) ->
training objective (C2) -> main A/B result (C3) -> gates & honest reading
(C4, C5) -> ablation & negative results (C6) -> robustness & variance (C8)
-> limitations (E5 list + static-graph in-sample lookahead) -> future work
(gate variants, attention story, energy track).

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

## E6 — 2x2 ablation: graph x retraining, with A/B anchors (2026-06-10)

**Question.** E5's valid->OOS IC decay had two candidate causes: stale graph
(frozen at the split date) or stale model (one training run). The 2x2
matrix {static, dynamic graph} x {single fit, walk-forward refit} attributes
the decay; every run also carries the two no-learning anchors
(`alpha_island_mean` = equal-weight composite of the same inputs, no
propagation; `alpha_uniform_composite` = uniform neighbour averaging of that
composite over the same topology the GAT uses).

**Setup.** One yfinance fetch (2021-01-01 → 2026-06-10, **all 50 names**
this time, 1,364 dates, 68,200 rows; panel pickled for reuse), identical
split/loss(IC)/epochs(50)/`torch.manual_seed(0)` across runs; walk-forward
refits every 63 snapshots. Script `.scratch/run_matrix.py`; per-run
diagnostics + summary JSON in `docs/results/2026-06-10_matrix_*`.

**Results** (gates: Value-added / Consistency / Uniqueness / Robustness;
`att_va` = GAT OOS Sharpe minus uniform-anchor OOS Sharpe):

| run | OOS IC | OOS Sharpe | gates V/C/U/R | att_va | runtime |
|---|---|---|---|---|---|
| static + single | -0.0051 | 1.04 | F/F/T/T | +2.09 | 179s |
| dynamic + single | -0.0348 | -1.01 | F/F/T/T | +0.18 | 207s |
| **static + walk-forward** | **+0.0200** | **1.95** | **F/T/T/T** | **+3.00** | 939s |
| dynamic + walk-forward | +0.0007 | 0.05 | F/T/T/T | +1.24 | 965s |

Anchors (identical across same-graph runs): island mean OOS IC -0.0225,
Sharpe -2.13; uniform composite OOS Sharpe -1.05 (static graph) / -1.19
(dynamic). Best single stays `alpha_wq_010_gap_quality` at OOS Sharpe 3.07.
GAT-vs-uniform Spearman is low and negative everywhere (-0.29 … -0.14).

**Findings.**

1. **Walk-forward retraining is the lever, not the dynamic graph.** The
   decay was *model* staleness: static+WF lifts OOS IC from -0.005 to +0.020
   and OOS Sharpe from 1.04 to 1.95, and Consistency flips to PASS (3/4
   gates). Hypothesis (a) from E5 is rejected as implemented: the
   per-snapshot correlation graph *hurts* in both arms — per-date correlation
   estimates are noisy, and the frozen train-period graph acts as a
   regulariser. (Refinements that might rescue dynamic graphs — longer
   windows, shrinkage, slower rebuild cadence — are future work, not
   currently planned.)
2. **The attention A/B — the capstone's core claim — is positive in every
   cell.** Both no-learning anchors are clearly negative OOS while the GAT is
   positive in 3 of 4 runs; attention adds +2.1 to +3.0 OOS Sharpe over
   uniform averaging on the *same* topology with the *same* inputs, and its
   scores are nearly uncorrelated with the uniform baseline (it is not just
   learning to smooth). "Relational learning beats naive relational
   averaging" holds regardless of which cell you read.
3. **Run-to-run variance is material — treat all point estimates as
   provisional.** static+single here reads -0.0051/1.04 vs E5's
   +0.0066/1.42 on a near-identical panel (deltas: ORCL restored, explicit
   seed 0 vs unseeded E5). Single-seed numbers cannot support paper claims;
   seed sensitivity (E7) must qualify every headline.
4. Value-added still fails in all cells against the strict max-of-singles
   bar, but static+WF narrows the gap to -1.12 (from -2.03).

**Current best config:** static graph + IC loss + walk-forward refits.

---

## E7 — Seed sensitivity, 5 seeds x {static_single, static_wf} (2026-06-10)

**Setup.** Same pickled E6 panel (only the torch seed varies), epochs 50,
IC loss, oos_chunk 63, seeds 0-4. Script `.scratch/run_seeds.py`; per-run
rows in `docs/results/2026-06-10_seed_sensitivity.csv`.

| arm | OOS IC (mean +/- std) | OOS Sharpe | attention value-add |
|---|---|---|---|
| static_single | -0.0009 +/- 0.0045 | 0.73 +/- 0.25 | 1.78 +/- 0.25 |
| static_wf | 0.0055 +/- 0.0147 | 1.15 +/- 0.75 | 2.20 +/- 0.75 |

Per-seed OOS Sharpe — single: 1.04, 0.67, 0.75, 0.83, 0.36;
walk-forward: 1.95, 1.81, 0.13, 0.84, 1.04 (seed 2 is a bad draw).

**Findings — several E6 headlines must be qualified.**

1. **The core claim survives intact and is now the strongest result:
   attention value-add over the uniform anchor is positive in 10 of 10 runs**
   (range +1.18 to +3.00 Sharpe). The uniform anchor is seed-independent
   (-1.05), so this is the GAT clearing a fixed negative bar in every draw.
   Likewise OOS Sharpe itself is positive in 10/10 runs (0.13-1.95).
2. **Walk-forward's advantage is suggestive, not conclusive.** Mean Sharpe
   1.15 vs 0.73 and mean IC +0.0055 vs -0.0009 favour WF, but the
   distributions overlap (Welch t ~ 1.2 on 5 seeds) and WF's variance is 3x
   single's. E6's "walk-forward is the lever" was a seed-0 read (1.95 was
   the luckiest WF draw); the honest paper claim is "walk-forward improves
   the mean but adds variance; not significant at n=5".
3. **The composite's rank-IC is fragile** (single-arm mean ~0) **while its
   long-short Sharpe is consistently positive** — the signal lives in the
   tails (top/bottom quantile selection) more than in the full
   cross-sectional ranking. Worth a paper paragraph: IC and L/S Sharpe
   measure different things and disagree here.
4. Paper protocol fixed by this entry: every headline number from now on is
   reported as mean +/- std over >= 5 seeds; single-run numbers (E5, E6) are
   retained as records but cited only with this caveat.

---

## E8 — Hardware note: GPU is a net loss at this graph size (2026-06-10)

Swapped `torch 2.12.0+cpu` -> `+cu126` (RTX 4060 Laptop 8GB, driver CUDA
12.6); pre-moved all dataset tensors to device once (`_dataset_to_device`)
so per-snapshot host->device transfers are not the bottleneck. Benchmark on
the E6 panel, seed 0, identical configs (`.scratch/bench_gpu.py`):

| arm | CPU (E7) | GPU | delta |
|---|---|---|---|
| static_single | 180s | 223s | +24% slower |
| static_wf | 955s | 1260s | +32% slower |

**Reading.** A 50-node, 10-feature snapshot graph is latency-bound: each
training step is dozens of microsecond-scale kernels, so GPU launch overhead
dominates and the 4060's throughput never engages. Decisions: (1) production
runs stay on CPU at the current universe size; (2) throughput for grids
comes from **process parallelism** (32 logical cores; ~180s per single run);
(3) the GPU becomes worthwhile only with a batched implementation (PyG
`Batch` packing ~950 snapshots/epoch into a few disconnected mega-graphs) or
a much larger universe — both future work.

### CPU vs GPU result divergence — magnitude, mechanism, impact

Same seed (0), same code, same data, different device:

| arm | CPU (OOS IC / Sharpe) | GPU (OOS IC / Sharpe) | CPU 5-seed Sharpe range (E7) |
|---|---|---|---|
| static_single | -0.0051 / 1.04 | +0.0056 / 1.35 | 0.36 - 1.04 |
| static_wf | +0.0200 / 1.95 | -0.0096 / 0.15 | 0.13 - 1.95 |

**Magnitude:** a same-seed device swap can swing the result by the *full
width of the seed distribution* (the wf row goes from the best CPU draw to
near the worst). A single run's number is device-dependent.

**Mechanism:** `torch.manual_seed` fixes the initial weights (init happens
on CPU), but training diverges immediately afterwards: (1) dropout masks
are drawn from each device's own RNG stream, so the per-step gradients
differ from step one; (2) GPU parallel reductions round in a different
order than CPU serial sums, and 1e-7-scale differences are amplified by 50
epochs of training dynamics. Cross-device same-seed is therefore another
draw from the same distribution, not the same experiment on new hardware.

**Impact on conclusions: none, given three disciplines already in place.**
(1) All claims are mean +/- std over >= 5 seeds — device variation behaves
like one more seed, and both GPU numbers fall inside or at the edge of the
CPU seed distribution (no evidence of systematic bias). The
attention-value-add result is device-insensitive (20/20 when this was
written; 30/30 after E9b). (2) Within-experiment
device consistency — **with one slip, caught post-hoc**: E5-E7 and the E9
grid ran on CPU, but the E9 winner validation silently ran on GPU (`fit`
auto-selects CUDA, and the CUDA wheel had just been installed). Fixed in
code the same day: `gat_equity_from_panel` now takes `device="cpu"` as an
explicit default (`"cuda"`/`"auto"` opt-in), so the documented
CPU-by-default decision is enforced rather than assumed; the E9 entry
carries the corresponding correction and a same-device rerun. (3)
Reproducibility statement for the paper: what is reproducible is the
*distribution* (given seeds + device class + torch version), not bit-exact
cross-device point estimates — the latter is unattainable on CUDA in
principle. Paper runs stay on CPU; the device is recorded per experiment.

---

## E9 — HP grid by valid IC, winner validated OOS (2026-06-10)

**Protocol (the hygiene point worth a paper paragraph).** HP selection used
**valid IC only** (the IS-internal, double-embargoed window; `fit` exposes it
as `model.best_valid_ic_`): 24 configs (lr {5e-4,1e-3,3e-3} x hidden {32,64}
x heads {2,4} x layers {1,2}) x 3 seeds = 72 train-only runs, static graph,
IC loss, no OOS metric computed anywhere in the grid. Only the selected
config then touched the OOS window, once, with fresh seeds. Scripts
`.scratch/run_hp_grid.py` + `run_winner.py`; artifacts
`2026-06-10_hp_grid_valid_ic.csv`, `2026-06-10_winner_validation_gpu.csv`
(the original run — see the device correction below) and
`..._winner_validation_cpu.csv` (same-device rerun).

**Grid result.** Winner: **lr=3e-3, hidden=64, heads=2, layers=2** (valid IC
0.1145 +/- 0.014). The surface is flat on top — six configs within
0.10-0.115, the default (lr 1e-3, heads 4) ranked 3rd at 0.1097 — and the
only clear structural signal is **all top-6 configs have 2 layers**: depth
matters, width/lr/heads barely do.

**Winner OOS validation** (5 seeds x both arms, vs the default config's E7
numbers in brackets):

| arm | OOS IC | OOS Sharpe | attention value-add |
|---|---|---|---|
| single | 0.0102 +/- 0.0179 [-0.0009 +/- 0.0045] | 1.03 +/- 0.84 [0.73 +/- 0.25] | 2.08 +/- 0.84 |
| walk-forward | **0.0179 +/- 0.0145** [0.0055 +/- 0.0147] | **1.30 +/- 0.73** [1.15 +/- 0.75] | 2.35 +/- 0.73 |

**Findings.**

1. **Valid-IC selection transferred to OOS**: the winner improves mean OOS
   IC and Sharpe over the default in *both* arms (directionally consistent,
   though within ~1 std). Positive evidence that the IS-internal valid
   window is informative — the protocol works, not just in principle.
2. **Attention value-add is now positive in 20 of 20 runs** (E7's 10 + E9's
   10; range +0.69 to +3.00 Sharpe). This is the paper's most robust result.
3. **Walk-forward beats single on the mean in both paired comparisons**
   (default: 1.15 vs 0.73; winner: 1.30 vs 1.03; and on IC 0.0179 vs
   0.0102) — two independent config draws agreeing strengthens E7's
   "directionally helpful", but per-comparison significance at n=5 remains
   out of reach; keep the qualified wording.
4. **Current best known setup**: static graph + IC loss + walk-forward +
   winner HPs -> OOS IC 0.0179 +/- 0.0145, OOS Sharpe 1.30 +/- 0.73, 9-10/10
   runs positive. Value-added vs the best single (3.07) stays open.

**Correction (same day): device confound in the winner-vs-default
comparison.** The winner validation silently ran on **GPU** (`fit`
auto-selected CUDA after the E8 wheel swap; the grid had passed CPU
explicitly). Within-E9 comparisons (single vs WF, all seeds) are
same-device and unaffected, but the bracketed E7 reference numbers are CPU
— so finding 1 ("valid-IC selection transferred to OOS") compared across
devices. Per E8, device behaves like another seed draw with no systematic
bias, so the directional read likely stands, but it is deconfounded by a
same-config CPU rerun (appended below). Code fixed so this cannot recur:
`gat_equity_from_panel(device="cpu")` is now explicit.

**E9b — same-device (CPU) rerun of the winner, 5 seeds
(2026-06-11, `2026-06-10_winner_validation_cpu.csv`):**

| arm | OOS IC | OOS Sharpe | attention value-add | default (E7, CPU) |
|---|---|---|---|---|
| single | -0.0004 +/- 0.0077 | 0.80 +/- 0.35 | 1.86 +/- 0.35 | IC -0.0009, Sharpe 0.73 |
| walk-forward | **0.0148 +/- 0.0043** | **1.37 +/- 0.39** | 2.42 +/- 0.39 | IC 0.0055, Sharpe 1.15 |

Deconfounded findings (these supersede finding 1 above):

1. **HP transfer is real in the walk-forward arm only.** Same-device,
   winner-vs-default: the single arm is a wash (0.80 vs 0.73 Sharpe, IC ~0
   both) — the GPU run's apparent single-arm improvement was a device/seed
   artifact. In the WF arm the winner is better on the mean AND much
   tighter: IC 0.0148 +/- 0.0043 vs 0.0055 +/- 0.0147 — a 3x variance
   reduction with 2.7x the mean.
2. **Best and most stable result to date: winner + walk-forward on CPU.**
   All 5 seeds positive on both IC (0.0109-0.0219) and Sharpe (0.98-1.98);
   the IC mean sits 3.4 std above zero.
3. **Walk-forward vs single reaches significance within the winner config**:
   Welch t ~ 3.9 on OOS IC, ~ 2.4 on OOS Sharpe (n=5 per arm). Combined
   with the two earlier paired comparisons (E7 default config, E9 GPU) both
   favouring WF, the walk-forward claim upgrades from "directionally
   helpful" to "significant in the tuned config, consistent everywhere".
4. Attention value-add: 10/10 positive again — **cumulative 30/30** across
   three run families (E7, E9-GPU, E9b-CPU).

---

## Next experiments (priority order)

1. ~~Dynamic per-snapshot graph~~ — **DONE, E6**: implemented
   (`rolling_topology_for`, `graph="dynamic"`) and rejected by the ablation;
   static graph stays the default.
2. ~~Walk-forward retraining~~ — **DONE, E6**: implemented
   (`walk_forward_composite_series`, `retrain="walk_forward"`) and adopted;
   it is the main OOS improvement so far.
3. ~~Uniform-mean A/B anchor~~ — **DONE, E6**: built into every run
   (`ab_report`); attention value-add is positive in every cell.
4. ~~Seed sensitivity~~ — **DONE, E7**; ~~HP grid~~ — **DONE, E9**: winner
   lr=3e-3/hidden=64/heads=2/layers=2 validated OOS; flat surface, depth-2
   is the only structural requirement.
5. **Attention visualisation (M4)** — plumbing done
   (`GATPropagator.last_attention`, head-layer softmax per snapshot, tested);
   remaining: the qualitative story (which sectors/names attend to whom over
   time) + a Streamlit "GAT vs Baseline" page.
6. **Value-added gate variants** — the strict max-of-singles bar is the only
   open gate; report alongside it the mean-of-singles and
   marginal-contribution-to-a-multifactor-portfolio readings before
   concluding the composite adds nothing.
7. **Significance for the WF-vs-single comparison** — more seeds (n=15-20
   per arm, cheap on the single arm) or a paired test across seeds, to
   upgrade "directionally helpful" if it holds.
