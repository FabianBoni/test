"""Feature building utilities for volatility modeling."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_loader import Swap


@dataclass(slots=True)
class Bar:
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume_usd: float


def build_time_bars(swaps: Iterable[Swap], interval_minutes: int) -> list[Bar]:
    """Convert raw swaps to fixed-width OHLCV bars."""

    if not swaps:
        return []
    df = pd.DataFrame(
        {
            "timestamp": [s.timestamp for s in swaps],
            "price": [s.price for s in swaps],
            "usd_volume": [s.usd_volume for s in swaps],
        }
    ).set_index("timestamp")
    rule = f"{interval_minutes}min"
    grouped = df.resample(rule)
    bars: list[Bar] = []
    for open_time, window in grouped:
        if window.empty:
            continue
        close_time = open_time + timedelta(minutes=interval_minutes)
        bars.append(
            Bar(
                open_time=open_time,
                close_time=close_time,
                open=float(window["price"].iloc[0]),
                high=float(window["price"].max()),
                low=float(window["price"].min()),
                close=float(window["price"].iloc[-1]),
                volume_usd=float(window["usd_volume"].sum()),
            )
        )
    return bars


def load_cached_bars(path: Path) -> list[Bar]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [
        Bar(
            open_time=datetime.fromisoformat(entry["open_time"]),
            close_time=datetime.fromisoformat(entry["close_time"]),
            open=float(entry["open"]),
            high=float(entry["high"]),
            low=float(entry["low"]),
            close=float(entry["close"]),
            volume_usd=float(entry["volume_usd"]),
        )
        for entry in payload
    ]


def save_bars(path: Path, bars: Iterable[Bar]) -> None:
    serializable = [
        {
            "open_time": bar.open_time.isoformat(),
            "close_time": bar.close_time.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume_usd": bar.volume_usd,
        }
        for bar in bars
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable))
