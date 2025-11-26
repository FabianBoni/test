"""Utility functions for evaluating simulation output."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def daily_return(equity_curve: Sequence[float], points_per_day: int) -> float:
    if len(equity_curve) < points_per_day:
        return 0.0
    start = equity_curve[-points_per_day]
    end = equity_curve[-1]
    return (end - start) / start


def cagr(equity_curve: Sequence[float], minutes_per_step: int) -> float:
    if not equity_curve:
        return 0.0
    start = equity_curve[0]
    end = equity_curve[-1]
    periods = len(equity_curve)
    years = periods * minutes_per_step / (60 * 24 * 365)
    if years <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1


def sharpe_ratio(returns: Sequence[float], risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = np.array(returns) - risk_free_rate
    return float(np.mean(excess) / (np.std(excess) + 1e-9) * math.sqrt(365))


def max_drawdown(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(curve)
    drawdowns = (curve - running_max) / (running_max + 1e-9)
    return float(drawdowns.min())
