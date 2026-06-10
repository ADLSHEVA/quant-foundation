# quant-alpha-foundation

Quantitative alpha research platform with two tracks — US equities and
European power markets — now extended with GNN/GAT relational factors. This
glossary fixes the language the codebase and agent skills should use.

## Language

**Alpha**:
A predictive cross-sectional signal scored per node per snapshot; the
registries (`features/registry.py`, `features/energy_alpha.py`) define them.
_Avoid_: indicator, signal (as a synonym for the scored value).

**Factor**:
The unified declaration of an alpha — metadata plus one `compute(panel) →
Series` over the canonical `(time, entity)` panel. Spans both tracks and both
families (island, relational).
_Avoid_: alpha definition, signal spec.

**Factor provider**:
A source of `Factor`s for the apply step. Adapters: `ExpressionFactorProvider`,
`GraphFactorProvider`, `LegacyEnergyProvider`.
_Avoid_: factory, registry (when you mean the provider, not the list).

**Node**:
The unit a graph connects — a stock (`symbol`) on the equity track, a bidding
zone (`market`) on the energy track.
_Avoid_: vertex, entity, asset.

**Bidding zone**:
An energy-track node: a power market with its own price (e.g. `DE_LU`, `FR`).
_Avoid_: region, area, market (when you specifically mean the node).

**Snapshot**:
The point-in-time slice of node features and topology at one date/timestamp.
Propagation always operates on a snapshot, never the whole panel.
_Avoid_: frame, slice, window.

**Topology**:
The directed edge set connecting nodes within a snapshot (who connects to
whom, with a base weight). Directedness carries asymmetric transmission.
_Avoid_: graph, network, adjacency (when you mean the edges, not the matrix).

**Propagator**:
The seam mapping a snapshot's node features + topology to one factor value per
node. Adapters: `UniformMeanPropagator` (baseline), `GATPropagator` (learned
attention).
_Avoid_: model, layer, GNN (when you mean the seam).

**Island factor**:
A factor that scores each node from its own data alone — the existing
cross-sectional families. The baseline in the relational A/B.
_Avoid_: traditional factor, single-name factor.

**Relational factor**:
A factor produced by propagation over the topology — it scores a node using
its neighbours. The capstone's new family.
_Avoid_: GNN factor, graph factor (when you mean the resulting factor).
