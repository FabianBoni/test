"""Configuration objects for the Snowball backtest."""
from __future__ import annotations

import os
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, validator


DEFAULT_POOL = "0xc31e54c7a869b9fcbecc14363cf510d1c41fa443"  # Uniswap V3 WETH/USDC 0.05% on Arbitrum
DEFAULT_FEE_TIER = 500  # in bps * 100
DEFAULT_GRAPH_ENDPOINT = "https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-v3-arbitrum"


class BacktestConfig(BaseModel):
    """Runtime configuration for a single simulation."""

    start: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc) - timedelta(days=90))
    end: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    pool_address: str = Field(default=DEFAULT_POOL)
    fee_tier: Literal[500, 3000, 10000] = Field(default=DEFAULT_FEE_TIER)
    quote_token: Literal["USDC"] = "USDC"
    base_token: Literal["WETH"] = "WETH"
    rebalance_interval_minutes: int = Field(default=15, ge=1, le=240)
    chunk_size_days: int = Field(default=7, ge=1, le=30)
    graph_endpoint: str = Field(
        default_factory=lambda: os.environ.get("BACKTEST_GRAPH_ENDPOINT", DEFAULT_GRAPH_ENDPOINT)
    )
    graph_api_key: str | None = Field(default_factory=lambda: os.environ.get("BACKTEST_GRAPH_API_KEY"))
    cache_dir: Path = Field(default=Path("data/cache"))
    raw_data_dir: Path = Field(default=Path("data/raw"))
    processed_data_dir: Path = Field(default=Path("data/processed"))
    reports_dir: Path = Field(default=Path("reports"))
    gas_cost_usd: float = Field(default=0.03, ge=0.0)
    fee_threshold_multiple: float = Field(default=1.2, ge=1.0)
    tight_width: float = Field(default=0.001, gt=0.0, lt=1.0)
    base_width: float = Field(default=0.005, gt=0.0, lt=1.0)
    wide_width: float = Field(default=0.01, gt=0.0, lt=1.0)

    @validator("end")
    def _validate_window(cls, v: datetime, values: dict[str, datetime]) -> datetime:  # noqa: D401
        """Ensure end is after start."""

        start = values.get("start")
        if start and v <= start:
            raise ValueError("end must be greater than start")
        return v

    @validator("base_width")
    def _validate_base_width(cls, v: float, values: dict[str, float]) -> float:
        tight = values.get("tight_width")
        if tight is not None and v <= tight:
            raise ValueError("base_width must be greater than tight_width")
        return v

    @validator("wide_width")
    def _validate_wide_width(cls, v: float, values: dict[str, float]) -> float:
        base = values.get("base_width")
        if base is not None and v <= base:
            raise ValueError("wide_width must be greater than base_width")
        return v

    @property
    def total_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)

    @property
    def steps(self) -> int:
        return self.total_minutes // self.rebalance_interval_minutes

    def ensure_directories(self) -> None:
        """Create cache/report folders when missing."""

        for path in (self.cache_dir, self.raw_data_dir, self.processed_data_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def _window_slug(self) -> str:
        start_slug = self.start.strftime("%Y%m%d%H%M")
        end_slug = self.end.strftime("%Y%m%d%H%M")
        return f"{start_slug}_{end_slug}_{self.rebalance_interval_minutes}m"

    def swaps_cache_path(self) -> Path:
        return self.raw_data_dir / f"swaps_{self.pool_address}_{self._window_slug}.json"

    def bars_cache_path(self) -> Path:
        return self.processed_data_dir / f"bars_{self.pool_address}_{self._window_slug}.json"

    def report_path(self) -> Path:
        return self.reports_dir / f"report_{self.pool_address}_{self._window_slug}.json"

    def chunk_descriptor_path(self) -> Path:
        return self.cache_dir / f"chunks_{self.pool_address}_{self._window_slug}.json"
