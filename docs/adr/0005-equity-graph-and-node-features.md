# Equity relational factor: graph, node features, and label

The conservative, hard-to-get-wrong choices for the first end-to-end equity GAT.

## Graph

Rolling correlation top-k backbone, optional sector boost, static to start
(`graph/edges_equity.py`):

- **Backbone (main):** each node connects to its top-k peers (W=60 days return
  correlation, k=8-10; k=4 at the current small N), **undirected**. This is
  N-agnostic and holds at any universe size — it does not require any sector to
  have multiple members.
- **Sector boost (optional):** when sector data is present, same-sector peers are
  additionally connected; when absent, it is skipped silently. Sector
  co-membership adds economic structure a reviewer can explain but must not be a
  hard dependency — at small/sector-sparse N a sector-only graph is nearly empty,
  which degenerates GAT attention into mean pooling.
- **min_degree fallback:** any node still isolated is wired to its
  highest-correlation peers; a no-op once the universe is large enough.
- Static first: one graph reused across the window, built from the training
  period only. Per-snapshot / dynamic graphs are a later refinement.
- Leakage: the correlation window is strictly `[as_of - W, as_of - 1]`, asserted
  in the builder; the static graph uses train data only.

Earlier draft made sector co-membership the primary edge, which produced a
near-empty graph at the shipped N=10; correlation top-k is the primary edge
instead, with sector as enhancement.

## Node features

All alphas' `_rank` columns, cross-sectional median fill, no non-alpha features:

- `_rank` over raw or z-score: rank is robust to scale and outliers, aligns with
  the IC/RankIC metric, and stops different alphas' scales from fighting.
- Median fill over 0-fill: in rank space the median is the no-information
  position; 0-fill forces a real rank and injects a spurious signal.
- No returns/volume: those are the raw inputs the alphas already consume; feeding
  them lets the model shortcut and widens the leakage surface. The story is "GAT
  combines existing alphas" — keep the input pure.

## Label / k

Equity locks days + the existing `cross_sectional_label` (price-ratio):

- k in days (5 or 10), aligned with the existing backtest convention — no new
  horizon invented.
- Energy's hourly-return label and hour-unit k are out of scope here (ADR-0004).

## Amendment (2026-06-10): edge weights are diagnostic-only

`Topology.edges` carries the correlation as `base_weight`, but the GAT
consumes only `edge_index` — attention learns its own weighting, which is the
point of the model. This is a recorded design decision, not an omission:
feeding the correlation as `edge_attr` is an untested research variation, out
of scope for the capstone. The weights remain on the topology for diagnostics
and for adapters that do use them (per ADR-0001, adapters weight internally).

Caveat to record: the `min_degree` fallback ranks peers by *signed*
correlation descending (positive-first, which is the intended semantics —
a strong negative correlate is not a "similar" neighbour), but when a node's
only available peers are negatively correlated it will still connect them
rather than leave the node isolated. Acceptable at N≈50; revisit if the
universe shrinks.

## Universe expansion (decided)

N=10 is a structural problem for a GNN, not a polish item: correlation estimates
on 10 names are noisy, k cannot be large, and attention learns mostly noise — the
equity analogue of the energy bidding-zone expansion. So the equity universe is
expanded to ~50 liquid US names with GICS sectors (`configs/universe.yaml`). All
model/graph/diagnostics code is N-agnostic, so this is a config change only — a
bigger universe.yaml plus a sector column, no model code touched.
