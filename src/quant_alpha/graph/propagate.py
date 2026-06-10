"""The propagate seam.

One cross-section in, one factor value per node out. This is the deep module
that the GNN/GAT relational-factor capstone turns on: the existing
fully-connected cross-market mean and the new learned-attention GAT factor are
both *adapters* on this one interface, so the island-vs-relational A/B is a
single adapter swap.

Design decisions (see docs/adr/0001-propagate-seam.md):

- **Granularity** — operates on a single snapshot (one date/timestamp). The
  panel loop lives outside; this stays a pure transform.
- **Weighting** — the seam takes a *topology* (who connects to whom); each
  adapter decides how to weight it. Baseline = uniform; GAT = learned
  attention.
- **State** — transform-only. Learned weights are bound at construction
  (`GATPropagator.from_weights`); training lives outside this interface.
- **Hops / direction** — topology is directed (asymmetric transmission). Hop
  depth is an adapter construction parameter, not a seam parameter.
- **Output** — returns only the propagated per-node factor. Attention weights
  are a GAT-adapter-specific capability (`last_attention`), not on the seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class Topology:
    """A directed graph over nodes for a single snapshot.

    Node labels match the index of the node-feature frame passed to
    ``propagate`` (``symbol`` for equities, ``market``/bidding zone for
    energy). Edges are directed ``(src, dst, base_weight)``; an undirected
    relationship is two edges. ``base_weight`` is the raw edge strength
    (interconnector capacity, correlation, supply-chain share); adapters
    decide how to use it.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, float], ...]

    def in_neighbours(self, node: str) -> list[tuple[str, float]]:
        """Sources of edges pointing at ``node``, with their base weights."""
        return [(src, w) for (src, dst, w) in self.edges if dst == node]


@runtime_checkable
class Propagator(Protocol):
    """The propagate seam.

    ``node_features`` is ``[N nodes x F features]`` for one snapshot; the
    return is ``[N]`` aligned to that index — one factor value per node.
    Stateless: any learned state is bound at construction.
    """

    def propagate(self, node_features: pd.DataFrame, topology: Topology) -> pd.Series:
        ...


@dataclass(frozen=True)
class UniformMeanPropagator:
    """Baseline adapter — average one feature over in-neighbours (and self).

    With a fully-connected topology and ``feature="spot_price"`` this
    reproduces the existing ``cross_market_spot_mean`` in
    ``features/energy_alpha.py`` exactly. That equivalence is the
    characterisation-test anchor for the A/B against the GAT adapter.
    """

    feature: str
    include_self: bool = True
    iters: int = 1

    def propagate(self, node_features: pd.DataFrame, topology: Topology) -> pd.Series:
        values = node_features[self.feature]
        for _ in range(self.iters):
            out = {}
            for node in topology.nodes:
                pool = [src for src, _w in topology.in_neighbours(node)]
                if self.include_self:
                    pool.append(node)
                pool = [n for n in pool if n in values.index]
                out[node] = values.reindex(pool).mean() if pool else float("nan")
            values = pd.Series(out).reindex(node_features.index)
        return values


@dataclass
class GATPropagator:
    """Learned-attention adapter — wraps a trained GATModel on the seam.

    Construction binds an already-trained model; training lives in
    ``quant_alpha.models.gat``. This adapter only bridges pandas <-> tensors so
    a GraphFactorProvider can emit a relational factor. torch is imported
    lazily, keeping this module importable without the ``[gnn]`` extra.
    """

    model: object  # quant_alpha.models.gat.GATModel in eval mode
    feature_cols: tuple[str, ...]

    @classmethod
    def from_weights(
        cls,
        path: str,
        config: object,
        feature_cols: tuple[str, ...],
    ) -> "GATPropagator":
        import torch

        from quant_alpha.models.gat import GATModel

        model = GATModel(config)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return cls(model=model, feature_cols=tuple(feature_cols))

    def propagate(self, node_features: pd.DataFrame, topology: Topology) -> pd.Series:
        import torch

        nodes = list(node_features.index)
        position = {node: i for i, node in enumerate(nodes)}
        x = torch.tensor(
            node_features[list(self.feature_cols)].to_numpy(dtype="float32")
        )
        edges = [
            (position[src], position[dst])
            for (src, dst, _w) in topology.edges
            if src in position and dst in position
        ]
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        self.model.eval()  # the seam guarantees deterministic inference (dropout off)
        with torch.no_grad():
            out = self.model(x, edge_index)
        return pd.Series(out.cpu().numpy(), index=node_features.index)

    def last_attention(self) -> pd.DataFrame:
        raise NotImplementedError(
            "M4: expose the learned per-edge attention weights for visualisation."
        )
