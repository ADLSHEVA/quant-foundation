"""Unified factor contract and the provider seam.

One `Factor` interface for both tracks (deepening #1) and a `FactorProvider`
seam so the apply step takes factors from any number of sources (deepening #2).
Island factors and relational (GNN) factors look identical at this layer — the
GNN factor is just a provider whose compute closes over a `Propagator`.

Panel contract: a factor's `compute` receives a panel indexed by a canonical
`(time, entity)` MultiIndex — `(date, symbol)` for equities, `(timestamp,
market)` for energy — carrying the raw and derived feature columns, and returns
a Series aligned to that index.

See docs/adr/0002-factor-provider-seam.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from quant_alpha.graph.propagate import Propagator, Topology

# panel: MultiIndex (time, entity) -> Series aligned to panel.index
FactorFn = Callable[[pd.DataFrame], pd.Series]
TopologyFor = Callable[[object], Topology]


@dataclass(frozen=True)
class Factor:
    """A predictive cross-sectional signal scored per node per snapshot.

    Unifies the equity `AlphaDefinition` and the energy `EnergyAlphaDefinition`:
    every factor — island or relational — carries metadata plus one `compute`.
    """

    name: str
    family: str
    hypothesis: str
    expected_direction: int
    compute: FactorFn
    expression: str | None = None

    def __post_init__(self) -> None:
        if self.expected_direction not in (-1, 1):
            raise ValueError(
                f"expected_direction must be -1 or 1, got {self.expected_direction!r}"
            )


@runtime_checkable
class FactorProvider(Protocol):
    """The provider seam — a source of `Factor`s for the apply step."""

    def factors(self) -> list[Factor]:
        ...


def apply_factors(panel: pd.DataFrame, providers: Sequence[FactorProvider]) -> pd.DataFrame:
    """Add one column per factor, drawn from every provider.

    `panel` is indexed by the canonical `(time, entity)` MultiIndex.
    """
    out = panel.copy()
    for provider in providers:
        for factor in provider.factors():
            out[factor.name] = factor.compute(panel)
    return out


def propagate_over_panel(
    panel: pd.DataFrame,
    propagator: Propagator,
    topology_for: TopologyFor,
    feature_cols: Sequence[str],
) -> pd.Series:
    """Run a `Propagator` snapshot-by-snapshot across a `(time, entity)` panel.

    This is the bridge between the propagate seam (one snapshot) and the factor
    contract (a whole panel). The panel loop lives here, not in the propagator.
    """
    pieces: list[pd.Series] = []
    for time, cross_section in panel.groupby(level=0):
        node_features = cross_section.droplevel(0)[list(feature_cols)]
        propagated = propagator.propagate(node_features, topology_for(time))
        propagated.index = pd.MultiIndex.from_arrays(
            [[time] * len(propagated), propagated.index],
            names=panel.index.names,
        )
        pieces.append(propagated)
    if not pieces:
        return pd.Series(dtype=float, index=panel.index)
    return pd.concat(pieces).reindex(panel.index)


@dataclass(frozen=True)
class ExpressionFactorProvider:
    """Adapter: wraps expression-style definitions as `Factor`s.

    The equity registry's `AlphaDefinition` already carries name, family,
    hypothesis, expected_direction, expression, and a `compute` over the
    `(date, symbol)` panel — so the mapping is near-identity.
    """

    definitions: tuple

    def factors(self) -> list[Factor]:
        return [
            Factor(
                name=d.name,
                family=d.family,
                hypothesis=d.hypothesis,
                expected_direction=d.expected_direction,
                compute=d.compute,
                expression=getattr(d, "expression", None),
            )
            for d in self.definitions
        ]


@dataclass(frozen=True)
class GraphFactorProvider:
    """Adapter: a relational (GNN) factor backed by a `Propagator`.

    `compute` closes over the propagator and a topology source, runs the
    per-snapshot propagation, and returns a panel-aligned Series — so at the
    registry/apply layer it is indistinguishable from an island factor.
    """

    name: str
    family: str
    hypothesis: str
    expected_direction: int
    propagator: Propagator
    topology_for: TopologyFor
    feature_cols: tuple[str, ...]

    def factors(self) -> list[Factor]:
        def compute(panel: pd.DataFrame) -> pd.Series:
            return propagate_over_panel(
                panel, self.propagator, self.topology_for, self.feature_cols
            )

        return [
            Factor(
                name=self.name,
                family=self.family,
                hypothesis=self.hypothesis,
                expected_direction=self.expected_direction,
                compute=compute,
            )
        ]


@dataclass(frozen=True)
class LegacyEnergyProvider:
    """Adapter: the imperative `add_energy_alpha_features` as 8 `Factor`s.

    Migration shim (F2: wrap now, migrate one-by-one later). The imperative
    function computes all 8 energy factors together with shared pre-computes
    (cross-market mean, diffs), so each Factor's compute should read its column
    from a single memoised run of `add_energy_alpha_features` over the panel's
    raw columns. Replace entries with native compute Factors over time.
    """

    def factors(self) -> list[Factor]:
        raise NotImplementedError(
            "Energy-pipeline wiring task: run add_energy_alpha_features once per "
            "panel (memoised), expose each of the 8 columns as a Factor.compute."
        )
