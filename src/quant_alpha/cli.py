from __future__ import annotations

from pathlib import Path

import typer

from quant_alpha.pipeline import run_pipeline
from quant_alpha.pipeline_energy import run_energy_pipeline
from quant_alpha.ingestion.entsoe import EntsoeError
from quant_alpha.ingestion.dlt_energy import run_dlt_energy_pipeline
from quant_alpha.ingestion.dlt_equity import run_dlt_equity_pipeline

app = typer.Typer(help="Quant alpha data engineering pipeline.", invoke_without_command=True)


def _run(config: Path, root: Path, offline: bool) -> None:
    result = run_pipeline(config, root.resolve(), offline=offline)
    typer.echo("Pipeline finished.")
    typer.echo(f"DuckDB: {result['duckdb_path']}")
    typer.echo(f"Rows: {result['rows']}")
    typer.echo(f"Metrics: {result['metrics']}")


def _run_energy(config: Path, root: Path, source: str | None = None) -> None:
    try:
        result = run_energy_pipeline(config, root.resolve(), source_override=source)
    except EntsoeError as exc:
        typer.echo(f"ENTSO-E ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Energy pipeline finished.")
    typer.echo(f"Data source: {result['data_source']}")
    typer.echo(f"DuckDB: {result['duckdb_path']}")
    if result.get("cloud_exports"):
        typer.echo(f"Cloud exports: {result['cloud_exports']}")
    typer.echo(f"Rows: {result['rows']}")
    typer.echo(f"Metrics: {result['metrics']}")


@app.callback()
def main(
    ctx: typer.Context,
    config: Path = typer.Option(Path("configs/project.yaml"), help="Project config YAML."),
    root: Path = typer.Option(Path("."), help="Project root."),
    offline: bool = typer.Option(False, help="Use deterministic synthetic prices."),
) -> None:
    """Run the pipeline directly, or use the `run` subcommand."""
    if ctx.invoked_subcommand is None:
        _run(config, root, offline)


@app.command("run")
def run_command(
    config: Path = typer.Option(Path("configs/project.yaml"), help="Project config YAML."),
    root: Path = typer.Option(Path("."), help="Project root."),
    offline: bool = typer.Option(False, help="Use deterministic synthetic prices."),
) -> None:
    """Run ingestion, factor computation, warehouse loading, and backtest."""
    _run(config, root, offline)


@app.command("energy-run")
def energy_run_command(
    config: Path = typer.Option(
        Path("configs/second_foundation_project.yaml"),
        help="Energy project config YAML.",
    ),
    root: Path = typer.Option(Path("."), help="Project root."),
    source: str | None = typer.Option(
        None,
        help="Override energy data source: synthetic or entsoe.",
    ),
) -> None:
    """Run the Second Foundation-inspired energy research pipeline."""
    _run_energy(config, root, source)


@app.command("dlt-energy")
def dlt_energy_command(
    root: Path = typer.Option(Path("."), help="Project root."),
    markets: str = typer.Option("DE_LU,CZ,FR", help="Comma-separated bidding zone list."),
    start: str = typer.Option("2023-01-01", help="Start date (YYYY-MM-DD)."),
    end: str | None = typer.Option(None, help="End date (YYYY-MM-DD), defaults to today."),
) -> None:
    """Run dlt-based energy ingestion pipeline (incremental, schema-managed)."""
    from quant_alpha.config import load_project_config

    cfg = load_project_config(Path("configs/second_foundation_project.yaml"), root=root.resolve())
    market_list = [m.strip() for m in markets.split(",")]
    info = run_dlt_energy_pipeline(cfg.duckdb_path, markets=market_list, start=start, end=end)
    typer.echo("dlt energy pipeline complete.")
    typer.echo(f"Dataset:  {info['dataset']} in {info['duckdb_path']}")
    typer.echo(f"Packages: {info['load_packages']}")


@app.command("dlt-equity")
def dlt_equity_command(
    config: Path = typer.Option(Path("configs/project.yaml"), help="Project config YAML."),
    root: Path = typer.Option(Path("."), help="Project root."),
    offline: bool = typer.Option(True, help="Use synthetic prices (no API key needed)."),
) -> None:
    """Run dlt-based equity ingestion pipeline (incremental, schema-managed)."""
    from quant_alpha.config import load_project_config, load_universe

    cfg = load_project_config(config, root=root.resolve())
    universe = load_universe(cfg.universe_path)
    info = run_dlt_equity_pipeline(cfg.duckdb_path, cfg=cfg, universe=universe, offline=offline)
    typer.echo("dlt equity pipeline complete.")
    typer.echo(f"Dataset:  {info['dataset']} in {info['duckdb_path']}")
    typer.echo(f"Packages: {info['load_packages']}")


@app.command("gat-equity")
def gat_equity_command(
    config: Path = typer.Option(Path("configs/project.yaml"), help="Project config YAML."),
    root: Path = typer.Option(Path("."), help="Project root."),
    offline: bool = typer.Option(True, help="Use deterministic synthetic prices."),
    epochs: int = typer.Option(50, help="GAT training epochs."),
    loss: str = typer.Option("ic", help="Training loss: ic (default) or mse."),
    graph: str = typer.Option("static", help="Graph mode: static or dynamic (per-snapshot)."),
    retrain: str = typer.Option("single", help="Fit mode: single or walk-forward."),
    oos_chunk: int = typer.Option(63, help="Walk-forward refit interval in snapshots."),
    device: str = typer.Option("cpu", help="Training device: cpu (default), cuda, or auto."),
) -> None:
    """Run the equity GAT relational-factor pipeline (requires the [gnn] extra)."""
    from quant_alpha.run_gat_equity import run_gat_equity

    out = run_gat_equity(
        config, root.resolve(), offline=offline, epochs=epochs,
        loss=loss, graph=graph, retrain=retrain, oos_chunk=oos_chunk, device=device,
    )
    gates = out["gate_report"]
    typer.echo("GAT equity pipeline finished.")
    typer.echo(f"Composite OOS IC mean: {gates['composite_oos_ic_mean']:.4f}")
    typer.echo(f"Value-added : {gates['value_added']}")
    typer.echo(f"Consistency : {gates['consistency']}")
    typer.echo(f"Uniqueness  : {gates['uniqueness']}")
    typer.echo(f"Robustness  : {gates['robustness']}")
    typer.echo(f"A/B anchor  : {out['ab_report']}")


@app.command("bruin-lineage")
def bruin_lineage_command(
    bruin_root: Path = typer.Option(Path("bruin"), help="Path to bruin/ directory."),
    asset: str | None = typer.Option(None, help="Show upstream/downstream for a specific asset."),
) -> None:
    """Print the Bruin asset lineage graph."""
    from quant_alpha.platform.bruin_graph import AssetGraph

    graph = AssetGraph(bruin_root.resolve())
    if asset:
        if asset not in graph.nodes:
            typer.echo(f"Asset '{asset}' not found.", err=True)
            raise typer.Exit(1)
        typer.echo(f"Upstream of '{asset}': {graph.upstream(asset)}")
        typer.echo(f"Downstream of '{asset}': {graph.downstream(asset)}")
    else:
        typer.echo(graph.lineage_report())


@app.command("bruin-run")
def bruin_run_command(
    bruin_root: Path = typer.Option(Path("bruin"), help="Path to bruin/ directory."),
    targets: str | None = typer.Option(
        None, help="Comma-separated asset names to run (with upstream). Default: all."
    ),
    dry_run: bool = typer.Option(False, help="Print execution plan without running."),
    project_root: Path = typer.Option(Path("."), help="PROJECT_ROOT env for asset runners."),
) -> None:
    """Run Bruin asset graph (topological order, upstream-aware)."""
    from quant_alpha.platform.bruin_graph import AssetGraph, AssetStatus

    graph = AssetGraph(bruin_root.resolve())
    target_list = [t.strip() for t in targets.split(",")] if targets else None
    env = {"PROJECT_ROOT": str(project_root.resolve())}

    typer.echo(f"Bruin asset graph — {len(graph.nodes)} assets loaded")
    if dry_run:
        typer.echo("(dry-run mode)")

    results = graph.run(targets=target_list, env=env, dry_run=dry_run)
    typer.echo(graph.status_report())

    failed = [n for n, s in results.items() if s == AssetStatus.FAILED]
    if failed:
        typer.echo(f"\nFailed assets: {failed}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
