"""Simulation loop for Snowball compounding."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import BacktestConfig
from .feature_engineering import Bar
from .strategy import RangeDecision, VolatilityAwareStrategy
from .volatility import GarchForecaster


@dataclass(slots=True)
class PositionState:
    lower_price: float
    upper_price: float
    notional_usd: float
    fees_accrued_usd: float
    entry_price: float
    base_amount: float
    quote_amount: float
    invariant_k: float
    max_base: float
    max_quote: float

    @classmethod
    def initialize(cls, price: float, notional_usd: float, lower: float, upper: float) -> "PositionState":
        base_amount = cls._half_allocation_base(notional_usd, price)
        quote_amount = notional_usd / 2
        invariant_k = base_amount * quote_amount
        max_base = cls._max_base(invariant_k, lower)
        max_quote = cls._max_quote(invariant_k, upper)
        return cls(
            lower_price=lower,
            upper_price=upper,
            notional_usd=notional_usd,
            fees_accrued_usd=0.0,
            entry_price=price,
            base_amount=base_amount,
            quote_amount=quote_amount,
            invariant_k=invariant_k,
            max_base=max_base,
            max_quote=max_quote,
        )

    @staticmethod
    def _half_allocation_base(notional_usd: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return (notional_usd / 2) / price

    @staticmethod
    def _max_base(invariant_k: float, lower_price: float) -> float:
        if invariant_k <= 0 or lower_price <= 0:
            return 0.0
        return math.sqrt(invariant_k / lower_price)

    @staticmethod
    def _max_quote(invariant_k: float, upper_price: float) -> float:
        if invariant_k <= 0:
            return 0.0
        return math.sqrt(invariant_k * upper_price)

    def redeploy(self, price: float, lower: float, upper: float) -> None:
        self.lower_price = lower
        self.upper_price = upper
        self.entry_price = price
        self._rebalance_amounts(price)

    def _rebalance_amounts(self, price: float) -> None:
        self.base_amount = self._half_allocation_base(self.notional_usd, price)
        self.quote_amount = self.notional_usd / 2
        self.invariant_k = self.base_amount * self.quote_amount
        self.max_base = self._max_base(self.invariant_k, self.lower_price)
        self.max_quote = self._max_quote(self.invariant_k, self.upper_price)


@dataclass(slots=True)
class BacktestResult:
    equity_curve: list[float]
    fee_curve: list[float]
    il_curve: list[float]
    decisions: list[RangeDecision]


class Simulator:
    """Runs a discrete-time compounding simulation given OHLCV bars."""

    def __init__(self, config: BacktestConfig, pool_tvl_usd: float | None = None) -> None:
        self._config = config
        self._strategy = VolatilityAwareStrategy()
        self._forecaster = GarchForecaster()
        self._pool_tvl_usd = max(pool_tvl_usd or 0.0, 1e3)  # avoid divide-by-zero

    def run(self, bars: Sequence[Bar]) -> BacktestResult:
        if not bars:
            raise ValueError("no bars supplied")
        price_series = np.array([bar.close for bar in bars])
        returns = np.diff(np.log(price_series))

        initial_price = bars[0].close
        initial_width = 0.01
        state = PositionState.initialize(
            price=initial_price,
            notional_usd=350.0,
            lower=initial_price * (1 - initial_width),
            upper=initial_price * (1 + initial_width),
        )
        hodl_base = state.base_amount
        hodl_quote = state.quote_amount

        equity_curve: list[float] = []
        fee_curve: list[float] = []
        il_curve: list[float] = []
        decisions: list[RangeDecision] = []
        cumulative_fees = 0.0

        for idx in range(1, len(bars)):
            bar = bars[idx]
            recent_returns = returns[max(0, idx - 50) : idx]
            forecast = self._forecaster.predict(recent_returns, horizon=4)

            pool_value = self._update_pool_distribution(state, bar.close)
            in_range = state.lower_price <= bar.close <= state.upper_price

            if in_range:
                fee_rate = self._config.fee_tier / 1_000_000
                share = pool_value / self._pool_tvl_usd
                fees_increment = bar.volume_usd * fee_rate * share
                state.fees_accrued_usd += fees_increment
                cumulative_fees += fees_increment
            else:
                fees_increment = 0.0

            compound = state.fees_accrued_usd >= self._config.gas_cost_usd * self._config.fee_threshold_multiple
            decision = self._strategy.decide(
                price=bar.close,
                forecast=forecast,
                fees_usd=state.fees_accrued_usd,
                gas_cost_usd=self._config.gas_cost_usd,
            )

            if not in_range:
                state.redeploy(bar.close, decision.lower_price, decision.upper_price)
            elif decision.lower_price != state.lower_price or decision.upper_price != state.upper_price:
                # proactively adjust range when forecast changes significantly
                state.redeploy(bar.close, decision.lower_price, decision.upper_price)

            if compound and pool_value > 0:
                state.notional_usd += state.fees_accrued_usd
                half_fee = state.fees_accrued_usd / 2
                hodl_quote += half_fee
                if bar.close > 0:
                    hodl_base += half_fee / bar.close
                state.fees_accrued_usd = 0.0
                state.redeploy(bar.close, state.lower_price, state.upper_price)

            pool_value = self._update_pool_distribution(state, bar.close)
            hodl_value = hodl_quote + hodl_base * bar.close
            equity = pool_value + state.fees_accrued_usd
            il = (equity / hodl_value - 1.0) if hodl_value > 0 else 0.0

            equity_curve.append(equity)
            fee_curve.append(cumulative_fees)
            il_curve.append(il)
            decisions.append(decision)

        return BacktestResult(
            equity_curve=equity_curve,
            fee_curve=fee_curve,
            il_curve=il_curve,
            decisions=decisions,
        )

    def _update_pool_distribution(self, state: PositionState, price: float) -> float:
        if price <= 0 or state.invariant_k <= 0:
            state.base_amount = 0.0
            state.quote_amount = 0.0
            return 0.0

        if price <= state.lower_price:
            state.base_amount = state.max_base
            state.quote_amount = 0.0
            return state.base_amount * price

        if price >= state.upper_price:
            state.base_amount = 0.0
            state.quote_amount = state.max_quote
            return state.quote_amount

        base_amount = math.sqrt(state.invariant_k / price)
        quote_amount = state.invariant_k / base_amount
        state.base_amount = base_amount
        state.quote_amount = quote_amount
        return quote_amount + base_amount * price
