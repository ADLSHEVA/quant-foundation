"""GAT training: labels, leakage-safe splits, and the fit loop.

The pure, leakage-critical pieces (label construction, walk-forward splits with
embargo, rank-IC evaluation) live here as plain pandas — no torch — so they are
testable without the `[gnn]` extra. The GAT fit loop itself lives in
``quant_alpha.models.gat`` (which requires the extra).

Decisions (docs/adr/0003-gat-training-objective.md):

- **Label** — `forward_return = price[t+k]/price[t] - 1`, k fixed, then
  cross-sectionally standardised per snapshot (z-score or rank). Label only —
  it uses t+k prices and must never enter the features.
- **Loss** — MSE first (prove the pipeline runs and loss falls), then an
  IC/rank loss (`-corr(pred, label)`) to align with the rank-IC metric.
- **Leakage** — time-ordered walk-forward, never shuffled; an embargo of >= k
  steps between train and valid so train labels can't overlap valid features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def cross_sectional_label(
    panel: pd.DataFrame,
    k: int,
    price_col: str = "adj_close",
    method: str = "zscore",
) -> pd.Series:
    """Build the standardised forward-return label on a `(time, entity)` panel.

    LABEL ONLY: uses `t+k` prices. The last k snapshots per entity are NaN
    (no future), which keeps them out of training.
    """
    if method not in ("zscore", "rank"):
        raise ValueError(f"method must be 'zscore' or 'rank', got {method!r}")
    panel = panel.sort_index()
    fwd = panel.groupby(level=1)[price_col].transform(lambda s: s.shift(-k) / s - 1)

    if method == "rank":
        return fwd.groupby(level=0).rank(pct=True) - 0.5
    grouped = fwd.groupby(level=0)
    std = grouped.transform("std").replace(0, np.nan)
    return (fwd - grouped.transform("mean")) / std


def cross_sectional_median_fill(
    panel: pd.DataFrame, cols: tuple[str, ...]
) -> pd.DataFrame:
    """Fill missing feature values with the per-snapshot median (neutral position).

    In rank space the median is the no-information position, unlike a 0-fill which
    forces a node to a real rank and injects a spurious signal.
    """
    out = panel.copy()
    out[list(cols)] = panel.groupby(level=0)[list(cols)].transform(
        lambda s: s.fillna(s.median())
    )
    return out


@dataclass(frozen=True)
class Split:
    """A leakage-safe walk-forward fold: train times, then a gap, then valid."""

    train: tuple
    valid: tuple


def walk_forward_splits(
    times: list,
    is_size: int,
    oos_size: int,
    embargo: int,
    step: int | None = None,
) -> list[Split]:
    """Roll IS/OOS windows forward with an `embargo` gap between them.

    `embargo` must be >= k (the label horizon) so the last train snapshot's
    label (which reaches t+k) cannot overlap the first valid feature snapshot.
    Times are sorted; never shuffled.
    """
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    ordered = sorted(times)
    step = step or oos_size
    splits: list[Split] = []
    idx = 0
    while idx + is_size + embargo + oos_size <= len(ordered):
        train = tuple(ordered[idx : idx + is_size])
        valid_start = idx + is_size + embargo
        valid = tuple(ordered[valid_start : valid_start + oos_size])
        splits.append(Split(train=train, valid=valid))
        idx += step
    return splits


def is_constrained_split(
    n_is: int,
    embargo: int,
    valid_frac: float = 0.15,
) -> tuple[range, range]:
    """Train/valid snapshot ranges fitted entirely inside the in-sample window.

    Layout over snapshot indices: ``train | embargo | valid | embargo | OOS``
    where OOS starts at ``n_is``. The trailing embargo keeps the last valid
    label (which reaches ``t + embargo``) inside the in-sample window, so
    best-epoch selection on valid IC never sees OOS data — the precondition
    for the four-gate OOS numbers to be honest.

    Valid sits at the end of IS and is sized from the valid end backwards, so
    small panels degrade gracefully: when the window cannot fit
    ``train | embargo | valid | embargo``, valid is empty and the caller must
    skip checkpoint selection.
    """
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    valid_stop = n_is - embargo
    valid_len = max(int(n_is * valid_frac), 1)
    valid_start = valid_stop - valid_len
    train_stop = valid_start - embargo
    if train_stop < 1:
        return range(0, n_is), range(0, 0)
    assert train_stop + embargo <= valid_start, "train labels must not reach valid"
    assert valid_stop + embargo <= n_is, "valid labels must stay inside IS"
    return range(0, train_stop), range(valid_start, valid_stop)


def rank_ic(pred: pd.Series, label: pd.Series) -> float:
    """Mean per-snapshot rank IC between predictions and labels.

    The evaluation metric the IC loss targets, and the basis for the
    shuffle sanity check (shuffle labels within a snapshot -> IC ~ 0).
    """
    frame = pd.concat({"pred": pred, "label": label}, axis=1).dropna()

    def corr(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return np.nan
        return group["pred"].rank().corr(group["label"].rank())

    per_snapshot = frame.groupby(level=0).apply(corr, include_groups=False).dropna()
    return float(per_snapshot.mean()) if len(per_snapshot) else float("nan")
