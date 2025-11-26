"""GraphQL-backed data ingestion helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

from .config import BacktestConfig


@dataclass(slots=True)
class Swap:
    timestamp: datetime
    price: float  # USD per unit of token1 (assumes USDC token0, WETH token1)
    usd_volume: float


@dataclass(slots=True)
class PoolInfo:
    total_value_locked_usd: float
    fee_tier: int | None = None


class GraphClient:
    """Tiny helper around The Graph's POST endpoint with disk caching."""

    def __init__(self, config: BacktestConfig) -> None:
        self._endpoint = config.graph_endpoint
        self._cache_dir = config.cache_dir / "graph"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._headers = {"Content-Type": "application/json"}
        if config.graph_api_key:
            self._headers["Authorization"] = f"Bearer {config.graph_api_key}"

    def query(self, name: str, document: str, variables: dict[str, Any]) -> dict[str, Any]:
        cache_key = self._cache_dir / f"{name}_{hash(json.dumps(variables, sort_keys=True))}.json"
        if cache_key.exists():
            return json.loads(cache_key.read_text())

        try:
            response = httpx.post(
                self._endpoint,
                json={"query": document, "variables": variables},
                headers=self._headers,
                timeout=30,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            location = exc.response.headers.get("location", "")
            msg = (
                "GraphQL request failed with status "
                f"{exc.response.status_code}. Set BACKTEST_GRAPH_ENDPOINT or "
                "GRAPH_API_KEY if the default endpoint requires authentication."
            )
            if "error.thegraph.com" in location:
                msg += " The public The Graph endpoint for Arbitrum now requires a studio API key."
            raise RuntimeError(msg) from exc
        payload = response.json()
        if "errors" in payload:
            details = payload["errors"][0].get("message", "Unknown error") if payload["errors"] else "Unknown"
            raise RuntimeError(
                "GraphQL request returned errors: "
                f"{details}. If this is a gated endpoint, set BACKTEST_GRAPH_ENDPOINT to a Studio URL."
            )
        cache_key.write_text(json.dumps(payload))
        return payload


SWAPS_QUERY = """
query PoolSwaps($pool: String!, $start: Int!, $end: Int!, $first: Int!, $skip: Int!) {
  swaps(
    where: { pool: $pool, timestamp_gte: $start, timestamp_lt: $end }
    orderBy: timestamp
    orderDirection: asc
    first: $first
    skip: $skip
  ) {
    timestamp
        amountIn
        amountInUSD
        amountOut
        amountOutUSD
        tokenIn {
            id
            symbol
            decimals
        }
        tokenOut {
            id
            symbol
            decimals
        }
  }
}
"""

POOL_INFO_QUERY = """
query PoolInfo($id: ID!) {
    liquidityPool(id: $id) {
        totalValueLockedUSD
    }
}
"""


class SwapLoader:
    """Loads and normalizes swap events into OHLCV style rows."""

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._client = GraphClient(config)
        self._cache_path = config.swaps_cache_path()
        self._chunk_meta_path = config.chunk_descriptor_path()

    def fetch(self, force_refresh: bool = False, progress_cb: Callable[[str], None] | None = None) -> list[Swap]:
        if not force_refresh and self._cache_path.exists():
            return self._read_cache()

        chunks = self._build_chunks()
        all_swaps: list[Swap] = []
        for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            if progress_cb:
                progress_cb(
                    f"Chunk {idx}/{len(chunks)}: {chunk_start.isoformat()} → {chunk_end.isoformat()}"
                )
            swaps = self._fetch_chunk(chunk_start, chunk_end)
            all_swaps.extend(swaps)
        self._write_cache(all_swaps)
        self._write_chunk_meta(chunks)
        return all_swaps

    def _build_chunks(self) -> list[tuple[datetime, datetime]]:
        chunks: list[tuple[datetime, datetime]] = []
        cursor = self._config.start
        while cursor < self._config.end:
            next_cursor = min(cursor + timedelta(days=self._config.chunk_size_days), self._config.end)
            chunks.append((cursor, next_cursor))
            cursor = next_cursor
        return chunks

    def _fetch_chunk(self, start_dt: datetime, end_dt: datetime) -> list[Swap]:
        start = int(start_dt.timestamp())
        end = int(end_dt.timestamp())
        page = 0
        page_size = 1000
        swaps: list[Swap] = []
        pool_id = self._config.pool_address.lower()

        while True:
            data = self._client.query(
                name="swaps",
                document=SWAPS_QUERY,
                variables={
                    "pool": pool_id,
                    "start": start,
                    "end": end,
                    "first": page_size,
                    "skip": page * page_size,
                },
            )
            batch = data["data"]["swaps"]
            if not batch:
                break
            swaps.extend(self._normalize(batch))
            page += 1
        return swaps

    def _normalize(self, rows: Iterable[dict[str, Any]]) -> list[Swap]:
        result: list[Swap] = []
        for row in rows:
            timestamp = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc)
            price = self._derive_price(row)
            if price is None:
                continue
            usd_volume = float(
                (Decimal(row["amountInUSD"]) + Decimal(row["amountOutUSD"])) / Decimal(2)
            )
            result.append(Swap(timestamp=timestamp, price=price, usd_volume=usd_volume))
        return result

    def _derive_price(self, row: dict[str, Any]) -> float | None:
        base_symbol = self._config.base_token.upper()
        token_in = row["tokenIn"]
        token_out = row["tokenOut"]
        if token_in["symbol"].upper() == base_symbol:
            base_amount = self._to_decimal(row["amountIn"], int(token_in["decimals"]))
            usd_value = Decimal(row["amountOutUSD"])
        elif token_out["symbol"].upper() == base_symbol:
            base_amount = self._to_decimal(row["amountOut"], int(token_out["decimals"]))
            usd_value = Decimal(row["amountInUSD"])
        else:
            return None
        if base_amount <= 0:
            return None
        return float(usd_value / base_amount)

    @staticmethod
    def _to_decimal(raw_amount: str, decimals: int) -> Decimal:
        if decimals <= 0:
            return Decimal(raw_amount)
        scale = Decimal(10) ** decimals
        return Decimal(raw_amount) / scale

    def _read_cache(self) -> list[Swap]:
        payload = json.loads(self._cache_path.read_text())
        return [
            Swap(
                timestamp=datetime.fromisoformat(entry["timestamp"]),
                price=float(entry["price"]),
                usd_volume=float(entry["usd_volume"]),
            )
            for entry in payload
        ]

    def _write_cache(self, swaps: list[Swap]) -> None:
        serializable = [
            {
                "timestamp": swap.timestamp.isoformat(),
                "price": swap.price,
                "usd_volume": swap.usd_volume,
            }
            for swap in swaps
        ]
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(serializable))

    def _write_chunk_meta(self, chunks: list[tuple[datetime, datetime]]) -> None:
        payload = [
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
            for start, end in chunks
        ]
        self._chunk_meta_path.write_text(json.dumps(payload, indent=2))


class PoolInfoLoader:
    """Fetches static pool-level metrics such as TVL and fee tier."""

    def __init__(self, config: BacktestConfig) -> None:
        self._config = config
        self._client = GraphClient(config)

    def fetch(self) -> PoolInfo:
        data = self._client.query(
            name="poolInfo",
            document=POOL_INFO_QUERY,
            variables={"id": self._config.pool_address.lower()},
        )
        pool = data["data"].get("liquidityPool")
        if not pool:
            raise ValueError(f"Pool {self._config.pool_address} not found in subgraph")
        return PoolInfo(
            total_value_locked_usd=float(pool["totalValueLockedUSD"] or 0.0),
            fee_tier=self._config.fee_tier,
        )
