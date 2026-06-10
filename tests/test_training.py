from __future__ import annotations

import numpy as np
import pandas as pd

from quant_alpha.graph.training import (
    Split,
    cross_sectional_label,
    cross_sectional_median_fill,
    rank_ic,
    walk_forward_splits,
)


def _panel(n_days: int = 30, entities: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    syms = [f"S{i}" for i in range(entities)]
    rng = np.random.default_rng(1)
    rows = []
    for sym in syms:
        price = 100 + np.cumsum(rng.normal(0, 1, n_days))
        for i, dt in enumerate(dates):
            rows.append({"date": dt, "symbol": sym, "adj_close": float(price[i])})
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()


def test_label_tail_is_nan_no_future_leak() -> None:
    k = 5
    label = cross_sectional_label(_panel(), k=k, method="zscore")
    # last k snapshots per entity have no t+k price -> NaN -> excluded from training
    per_entity_tail = label.groupby(level=1).apply(lambda s: s.iloc[-k:].isna().all())
    assert per_entity_tail.all()


def test_zscore_label_centered_per_snapshot() -> None:
    label = cross_sectional_label(_panel(), k=3, method="zscore").dropna()
    means = label.groupby(level=0).mean()
    assert np.allclose(means.to_numpy(), 0.0, atol=1e-9)


def test_rank_label_in_range() -> None:
    label = cross_sectional_label(_panel(), k=3, method="rank").dropna()
    assert label.between(-0.5, 0.5).all()


def test_median_fill_uses_per_snapshot_median() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2, freq="D"), ["A", "B", "C"]],
        names=["date", "symbol"],
    )
    panel = pd.DataFrame({"f": [1.0, 3.0, np.nan, 10.0, 20.0, np.nan]}, index=idx)
    filled = cross_sectional_median_fill(panel, ("f",))
    # snapshot 1 median of {1,3} = 2.0; snapshot 2 median of {10,20} = 15.0
    assert filled["f"].iloc[2] == 2.0
    assert filled["f"].iloc[5] == 15.0


def test_walk_forward_embargo_prevents_overlap() -> None:
    k = 5
    times = list(pd.date_range("2024-01-01", periods=60, freq="D"))
    splits = walk_forward_splits(times, is_size=20, oos_size=10, embargo=k, step=10)

    assert splits, "expected at least one split"
    for s in splits:
        assert isinstance(s, Split)
        # valid starts strictly after the last train label horizon (t+k)
        gap = times.index(s.valid[0]) - times.index(s.train[-1])
        assert gap >= k + 1
        assert set(s.train).isdisjoint(s.valid)
        assert min(s.valid) > max(s.train)


def test_rank_ic_perfect_and_shuffled() -> None:
    # Sanity checks the user called for: perfect ordering -> IC ~ 1;
    # labels shuffled within each snapshot -> IC ~ 0.
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    syms = [f"S{i}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    rng = np.random.default_rng(2)
    pred = pd.Series(rng.normal(size=len(idx)), index=idx)
    label = pred.copy()  # perfectly aligned

    assert rank_ic(pred, label) > 0.99

    shuffled = label.groupby(level=0, group_keys=False).apply(
        lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index)
    )
    assert abs(rank_ic(pred, shuffled)) < 0.15
