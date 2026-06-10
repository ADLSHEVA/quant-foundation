# One Factor contract behind a provider seam

The two tracks declared factors two different ways: equity via
`AlphaDefinition` (metadata + a `compute` callable, applied generically) and
energy via `EnergyAlphaDefinition` (metadata only, applied by a ~150-line
imperative function). We unify them into one `Factor` (deepening #1) and add a
`FactorProvider` seam so the apply step draws factors from any number of
sources (deepening #2). The GNN/GAT factor then plugs in as just another
provider, so island and relational factors are indistinguishable at the
registry/apply layer.

## Considered Options

- **Panel contract** — canonical `(time, entity)` MultiIndex (chosen) over
  keeping two panel shapes: equity already uses `(date, symbol)` with
  `groupby(level=0/1)`, so it needs no change; energy re-indexes
  `(timestamp, market)` to match. One `compute(panel) -> Series` signature for
  both tracks.
- **Energy migration** — wrap the existing imperative function as a
  `LegacyEnergyProvider` now, migrate its 8 factors to native compute later
  (chosen) over rewriting all 8 up front: keeps tests green and unblocks the
  GNN work immediately.
- **Relational factors** — a `GraphFactorProvider` builds a `Factor` whose
  `compute` closes over a `Propagator` and a topology source and runs the
  per-snapshot loop internally (chosen) over a separate apply path for graph
  factors: preserves the single apply loop.

## Consequences

- `propagate_over_panel` is the bridge from the per-snapshot propagate seam
  (ADR-0001) to the whole-panel `compute` contract.
- Adopting this in the live pipelines is a follow-up: `add_alpha_factors` and
  `run_energy_pipeline` keep working unchanged until they are switched to
  `apply_factors([...providers])`.
