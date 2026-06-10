# GAT training objective: rank-IC target, leakage-safe by construction

The GAT learns to rank names cross-sectionally, not to hit absolute returns.
The label is `forward_return = price[t+k]/price[t] - 1` for a fixed k,
standardised per snapshot (z-score or rank). The loss starts as MSE — to prove
the pipeline runs and the loss falls — then switches to an IC/rank loss
(`-corr(pred, label)`) to align training with the rank-IC metric the platform
already reports. Two steps so loss design never blocks getting the pipeline
green.

Leakage is treated as the primary M2 risk and is prevented structurally:

- **Time-ordered walk-forward only**, never shuffled; valid always after train.
- **Features and edges at t use only data <= t.** `forward_return` uses t+k
  prices and is a label exclusively — never a feature. The last k snapshots per
  entity are therefore NaN and excluded.
- **Embargo >= k** snapshots between train and valid, so the last train label
  (reaching t+k) cannot overlap the first valid feature.

## Considered Options

- **Target** — cross-sectionally standardised forward return (chosen) over raw
  return regression: we care about who out-ranks whom, not absolute magnitude.
- **Loss** — MSE then IC loss (chosen) over IC loss from the start: MSE first
  de-risks the pipeline; IC loss aligns with the metric once it runs.
- **Validation** — walk-forward with embargo (chosen) over random k-fold: random
  splits leak future into past. Note the existing `backtest.walk_forward_ic` has
  no embargo — it is for factor evaluation, not model training; the training
  splitter (`graph.training.walk_forward_splits`) adds the embargo.

## Amendment (2026-06-10): one split date, valid inside IS

The IS/OOS boundary is a single source of truth computed once in
`run_gat_equity.gat_equity_from_panel` and passed explicitly to all three
consumers — graph construction (`as_of`), model selection
(`fit(train_idx, valid_idx)`), and the four-gate evaluation
(`evaluate_alpha_suite(split_date=...)`). Previously each assumed its own
0.7, which happened to coincide; changing any one ratio would have silently
leaked (e.g. a larger graph window absorbing OOS correlations, or valid-IC
model selection peeking at the evaluation OOS window).

Layout, enforced by asserts at the call site and in
`graph.training.is_constrained_split`:

    train | embargo(k) | valid | embargo(k) | OOS

The trailing embargo matters: valid labels reach `t + k`, so without it the
last valid labels would overlap the start of the OOS window and best-epoch
selection would be contaminated. `fit` now reloads the best-by-valid-IC
weights before returning (previously it saved them but returned the
final-epoch model); when valid is empty (small panels) selection is skipped.

The shuffle-label self-check is automated in `tests/test_leakage.py`:
a negative control (shuffled labels -> valid IC ~ 0) plus a positive control
(planted signal -> recovered), together arguing the pipeline produces no
false positives.

## Consequences

- The leakage-critical logic — label, splits, rank-IC — is pure pandas in
  `graph/training.py`, tested without torch (including a shuffle-label sanity
  check that drives IC to ~0). Only `fit_gat` needs the `[gnn]` extra.
- k is fixed first; multi-k, dynamic graphs, and attention visualisation are
  deferred until the single-k pipeline is validated.
