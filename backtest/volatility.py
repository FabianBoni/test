"""Volatility forecasting helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy.linalg as npl

import numpy as np
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning

try:  # statsmodels provides its own warning class
    from statsmodels.tools.sm_exceptions import ConvergenceWarning as StatsConvergenceWarning
except ImportError:  # pragma: no cover - optional dependency behind arch
    StatsConvergenceWarning = None


@dataclass(slots=True)
class VolatilityForecast:
    sigma: float
    model_name: str


class GarchForecaster:
    """Simple GARCH(1,1) wrapper with sane defaults."""

    def __init__(self, p: int = 1, q: int = 1) -> None:
        self._p = p
        self._q = q

    def predict(self, returns: Sequence[float], horizon: int = 4) -> VolatilityForecast:
        if len(returns) < 30:
            sigma = float(np.std(returns)) if len(returns) else 0.0
            return VolatilityForecast(sigma=sigma, model_name="naive_std")

        def _fallback() -> VolatilityForecast:
            sigma = float(np.std(returns)) if len(returns) else 0.0
            return VolatilityForecast(sigma=sigma, model_name="naive_std")

        warning_types = [ConvergenceWarning]
        if StatsConvergenceWarning is not None:
            warning_types.append(StatsConvergenceWarning)

        fallback_exceptions = tuple(warning_types) + (ValueError, npl.LinAlgError)

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = arch_model(returns, vol="Garch", p=self._p, q=self._q, rescale=False)
                fitted = model.fit(disp="off")
        except fallback_exceptions:
            return _fallback()

        if caught and any(issubclass(w.category, tuple(warning_types)) for w in caught):
            return _fallback()

        if getattr(fitted, "convergence_flag", 0) != 0:
            return _fallback()

        forecast = fitted.forecast(horizon=horizon)
        sigma = float(np.sqrt(forecast.variance.values[-1, -1]))
        return VolatilityForecast(sigma=sigma, model_name="garch")
