# The propagate seam: one snapshot in, one factor per node out

To add GNN/GAT relational factors without forking the factor pipeline, we
introduce a single `Propagator` seam (`graph/propagate.py`). Both the existing
fully-connected cross-market mean and the new learned-attention GAT factor are
adapters on this one interface, so the island-vs-relational A/B that anchors
the capstone is a single adapter swap rather than two parallel code paths.

## Considered Options

The interface was settled across five forks; each was a real trade-off:

- **Granularity** — per-snapshot (chosen) over whole-panel: keeps the seam a
  pure, testable transform; the panel loop and point-in-time slicing live
  outside it.
- **Weighting** — topology in, adapter weights internally (chosen) over
  passing a pre-weighted adjacency matrix: keeps GAT's attention *behind* the
  seam so baseline and GAT genuinely share one interface.
- **State** — transform-only (chosen) over a fit+transform estimator: GAT
  weights are bound at construction (`from_weights`); training lives in
  `graph/train.py`. Matches the train/inference split in
  `docs/gnn_capstone_design.md §6`.
- **Hops & direction** — directed topology, hop depth as an adapter
  construction parameter (chosen) over a `hops=` seam parameter: GAT layers
  have per-layer parameters, so "call N times" is the wrong semantics; the
  over-smoothing experiment varies adapter `depth`.
- **Output** — propagated factor only (chosen) over a (factor, diagnostics)
  tuple: attention is a GAT-adapter capability (`last_attention`), not on the
  shared seam.

## Consequences

- `node_features` is `[N nodes x F features]`, output is `Series[N]` — GAT can
  use rich node features; the baseline adapter is configured with which single
  feature to average.
- `UniformMeanPropagator` with a fully-connected topology must reproduce the
  current `cross_market_spot_mean` exactly — that equivalence is the
  characterisation test for the A/B.
