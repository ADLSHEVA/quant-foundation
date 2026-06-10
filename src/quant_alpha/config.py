from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class BacktestConfig(BaseModel):
    forward_return_days: int = 5
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    transaction_cost_bps: float = 5.0
    periods_per_year: float = 252.0


class FactorHorizons(BaseModel):
    momentum: int = 21
    reversal: int = 5
    volatility: int = 20
    breakout: int = 55


class EntsoeConfig(BaseModel):
    token_env: str = "ENTSOE_API_KEY"
    base_url: str = "https://web-api.tp.entsoe.eu/api"
    timeout_seconds: int = 60


class CloudExportConfig(BaseModel):
    enabled: bool = False
    gcp_project_id: str | None = None
    gcs_bucket: str | None = None
    bigquery_dataset: str | None = None
    bigquery_location: str = "EU"
    gcs_prefix: str = "energy"
    write_disposition: str = "WRITE_TRUNCATE"


class ProjectConfig(BaseModel):
    project_name: str = "quant-alpha-foundation"
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    duckdb_path: Path = Path("data/warehouse/quant_alpha.duckdb")
    universe_path: Path = Path("configs/universe.yaml")
    start_date: str = "2021-01-01"
    end_date: str | None = None
    bar_interval: str = "1d"
    data_source: str = "synthetic"
    factor_horizons: FactorHorizons = Field(default_factory=FactorHorizons)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    entsoe: EntsoeConfig = Field(default_factory=EntsoeConfig)
    cloud: CloudExportConfig = Field(default_factory=CloudExportConfig)


class Universe(BaseModel):
    name: str
    timezone: str = "America/New_York"
    asset_class: str = "equities"
    symbols: list[str]
    benchmarks: list[str] = Field(default_factory=list)
    sectors: dict[str, str] = Field(default_factory=dict)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    resolved = (root / path).resolve()
    if not str(resolved).startswith(str(root.resolve())):
        raise ValueError(f"Path {path!r} escapes project root {root!r}")
    return resolved


def load_project_config(path: Path, root: Path | None = None) -> ProjectConfig:
    root = root or Path.cwd()
    cfg = ProjectConfig(**load_yaml(resolve_path(root, path)))
    if cfg.end_date is None:
        cfg.end_date = date.today().isoformat()
    cfg.raw_dir = resolve_path(root, cfg.raw_dir)
    cfg.processed_dir = resolve_path(root, cfg.processed_dir)
    cfg.duckdb_path = resolve_path(root, cfg.duckdb_path)
    cfg.universe_path = resolve_path(root, cfg.universe_path)
    return cfg


def load_universe(path: Path) -> Universe:
    return Universe(**load_yaml(path))


def ensure_project_dirs(cfg: ProjectConfig) -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
