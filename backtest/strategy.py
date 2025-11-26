"""Range sizing and compounding policy."""
from __future__ import annotations

from dataclasses import dataclass

from .volatility import VolatilityForecast


@dataclass(slots=True)
class RangeDecision:
    lower_price: float
    upper_price: float
    compound: bool
    reason: str


class VolatilityAwareStrategy:
    """Translates volatility forecasts into Uniswap V3 tick ranges."""

    def __init__(self, tight_width: float = 0.005, base_width: float = 0.01, wide_width: float = 0.02) -> None:
        self._tight = tight_width
        self._base = base_width
        self._wide = wide_width

    def decide(self, price: float, forecast: VolatilityForecast, fees_usd: float, gas_cost_usd: float) -> RangeDecision:
        width = self._select_width(forecast.sigma)
        lower = price * (1.0 - width)
        upper = price * (1.0 + width)
        compound = fees_usd >= gas_cost_usd
        reason = f"sigma={forecast.sigma:.4f} width={width:.3%}"
        return RangeDecision(lower_price=lower, upper_price=upper, compound=compound, reason=reason)

    def _select_width(self, sigma: float) -> float:
        if sigma <= 0.005:
            return self._tight
        if sigma <= 0.015:
            return self._base
        return self._wide
