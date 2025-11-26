"""Typer CLI entry for running the Snowball backtest."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Optional

import optuna
import typer
from dotenv import load_dotenv
from optuna.trial import TrialState

from backtest import BacktestConfig, BacktestResult, Simulator
from backtest.data_loader import PoolInfoLoader, SwapLoader
from backtest.feature_engineering import Bar, build_time_bars, load_cached_bars, save_bars
from backtest.metrics import cagr, daily_return, max_drawdown, sharpe_ratio

load_dotenv(dotenv_path=Path(".env"), override=False)

app = typer.Typer(help="Backtest the Snowball bot over historical Uniswap V3 data")


def _config_overrides(
    tight_width: float | None,
    base_width: float | None,
    wide_width: float | None,
    fee_threshold_multiple: float | None,
) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if tight_width is not None:
        overrides["tight_width"] = tight_width
    if base_width is not None:
        overrides["base_width"] = base_width
    if wide_width is not None:
        overrides["wide_width"] = wide_width
    if fee_threshold_multiple is not None:
        overrides["fee_threshold_multiple"] = fee_threshold_multiple
    return overrides


def _load_or_build_bars(config: BacktestConfig, interval: int, refresh_data: bool) -> list[Bar]:
    bars_cache = config.bars_cache_path()
    bars = [] if refresh_data else load_cached_bars(bars_cache)

    if not bars:
        typer.echo("↻ Fetching swaps from The Graph …")
        loader = SwapLoader(config)

        def progress_update(message: str) -> None:
            typer.echo(f"   → {message}")

        swaps = loader.fetch(force_refresh=refresh_data, progress_cb=progress_update)
        typer.echo(f"✔ Retrieved {len(swaps):,} swaps")
        typer.echo("↻ Building OHLCV bars …")
        bars = build_time_bars(swaps, interval_minutes=interval)
        save_bars(bars_cache, bars)
        typer.echo(f"✔ Built {len(bars):,} bars")
    else:
        typer.echo("✔ Loaded cached bars")
    return bars


def _compute_metrics(result: BacktestResult, interval: int) -> dict[str, float]:
    equity_curve = result.equity_curve
    if not equity_curve:
        raise ValueError("simulation produced empty equity curve")
    period_returns = []
    for prev, curr in zip(equity_curve[:-1], equity_curve[1:]):
        if prev <= 0:
            period_returns.append(0.0)
        else:
            period_returns.append((curr - prev) / prev)
    points_per_day = max(1, (24 * 60) // interval)
    return {
        "final_equity": equity_curve[-1],
        "total_fees": result.fee_curve[-1] if result.fee_curve else 0.0,
        "daily_return": daily_return(equity_curve, points_per_day),
        "cagr": cagr(equity_curve, interval),
        "sharpe": sharpe_ratio(period_returns),
        "max_drawdown": max_drawdown(equity_curve),
        "avg_period_return": fmean(period_returns) if period_returns else 0.0,
        "final_il": result.il_curve[-1] if result.il_curve else 0.0,
    }


@app.command()
def run(
    start: datetime = typer.Option(..., help="UTC start timestamp"),
    end: datetime = typer.Option(..., help="UTC end timestamp"),
    pool: str = typer.Option(..., help="Pool address"),
    interval: int = typer.Option(15, help="Bar interval in minutes"),
    refresh_data: bool = typer.Option(False, help="Force refetch graph data and rebuild bars"),
    tight_width: Optional[float] = typer.Option(None, help="Override tight width (fractional)"),
    base_width: Optional[float] = typer.Option(None, help="Override base width (fractional)"),
    wide_width: Optional[float] = typer.Option(None, help="Override wide width (fractional)"),
    fee_threshold_multiple: Optional[float] = typer.Option(
        None, help="Override compounding threshold multiplier"
    ),
) -> None:
    overrides = _config_overrides(tight_width, base_width, wide_width, fee_threshold_multiple)
    config = BacktestConfig(start=start, end=end, pool_address=pool, rebalance_interval_minutes=interval, **overrides)
    config.ensure_directories()

    bars = _load_or_build_bars(config, interval, refresh_data)

    typer.echo("↻ Fetching pool metadata …")
    pool_info = PoolInfoLoader(config).fetch()
    typer.echo(f"✔ Pool TVL: ${pool_info.total_value_locked_usd:,.0f}")

    typer.echo("↻ Running simulator …")
    simulator = Simulator(config, pool_tvl_usd=pool_info.total_value_locked_usd)
    result = simulator.run(bars)
    typer.echo("✔ Simulation complete")

    metrics = _compute_metrics(result, interval)

    report = {
        "pool": pool,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "interval_minutes": interval,
        "pool_tvl_usd": pool_info.total_value_locked_usd,
        "rebalance_steps": len(result.equity_curve),
        "metrics": metrics,
    }

    report_path = config.report_path()
    report_path.write_text(json.dumps(report, indent=2))

    typer.echo(f"Final equity: ${metrics['final_equity']:.2f}")
    typer.echo(f"Total fees: ${metrics['total_fees']:.2f}")
    typer.echo(
        f"CAGR: {metrics['cagr']:.2%}, Max DD: {metrics['max_drawdown']:.2%}, IL: {metrics['final_il']:.2%}"
    )
    typer.echo(f"Report saved to {report_path}")


@app.command()
def optimize(
    start: datetime = typer.Option(..., help="UTC start timestamp"),
    end: datetime = typer.Option(..., help="UTC end timestamp"),
    pool: str = typer.Option(..., help="Pool address"),
    interval: int = typer.Option(15, help="Bar interval in minutes"),
    trials: int = typer.Option(25, min=1, help="Number of Optuna trials to run"),
    metric: str = typer.Option("total_fees", help="Metric to maximize"),
    refresh_data: bool = typer.Option(False, help="Force refetch graph data and rebuild bars"),
    seed: Optional[int] = typer.Option(None, help="Random seed for sampler reproducibility"),
) -> None:
    metric = metric.strip()
    valid_metrics = {
        "final_equity",
        "total_fees",
        "daily_return",
        "cagr",
        "sharpe",
        "avg_period_return",
    }
    if metric not in valid_metrics:
        raise typer.BadParameter(f"metric must be one of: {', '.join(sorted(valid_metrics))}")

    config = BacktestConfig(start=start, end=end, pool_address=pool, rebalance_interval_minutes=interval)
    config.ensure_directories()

    bars = _load_or_build_bars(config, interval, refresh_data)
    typer.echo("↻ Fetching pool metadata …")
    pool_info = PoolInfoLoader(config).fetch()
    typer.echo(f"✔ Pool TVL: ${pool_info.total_value_locked_usd:,.0f}")

    typer.echo(f"↻ Launching Bayesian optimization targeting '{metric}' ({trials} trials)")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        tight = trial.suggest_float("tight_width", 0.0005, 0.02)
        base_low = tight + 0.0005
        base = trial.suggest_float("base_width", base_low, 0.05)
        wide_low = base + 0.0005
        wide = trial.suggest_float("wide_width", wide_low, 0.08)
        fee_threshold = trial.suggest_float("fee_threshold_multiple", 1.05, 2.0)
        trial_config = config.model_copy(
            update={
                "tight_width": tight,
                "base_width": base,
                "wide_width": wide,
                "fee_threshold_multiple": fee_threshold,
            }
        )
        simulator = Simulator(trial_config, pool_tvl_usd=pool_info.total_value_locked_usd)
        result = simulator.run(bars)
        metrics = _compute_metrics(result, interval)
        trial.set_user_attr("metrics", metrics)
        return metrics[metric]

    study.optimize(objective, n_trials=trials)
    best_trial = study.best_trial
    best_metrics = best_trial.user_attrs.get("metrics", {})

    typer.echo(f"✔ Best trial #{best_trial.number} → {metric}={best_trial.value:.4f}")
    typer.echo(f"   tight={best_trial.params['tight_width']:.4%}, base={best_trial.params['base_width']:.4%}, "
               f"wide={best_trial.params['wide_width']:.4%}, fee_threshold={best_trial.params['fee_threshold_multiple']:.2f}")

    best_payload = {
        "objective_metric": metric,
        "best_value": best_trial.value,
        "params": best_trial.params,
        "metrics": best_metrics,
    }
    best_path = config.reports_dir / f"optimization_best_{config.pool_address}_{config._window_slug}.json"
    best_path.write_text(json.dumps(best_payload, indent=2))

    rows = []
    for trial in study.trials:
        if trial.state != TrialState.COMPLETE:
            continue
        row = {
            "number": trial.number,
            "value": trial.value,
            **trial.params,
        }
        metrics_payload = trial.user_attrs.get("metrics")
        if metrics_payload:
            for key, value in metrics_payload.items():
                row[f"metric_{key}"] = value
        rows.append(row)

    trials_path = config.reports_dir / f"optimization_trials_{config.pool_address}_{config._window_slug}.csv"
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with trials_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    typer.echo(f"Best configuration saved to {best_path}")
    if rows:
        typer.echo(f"All completed trials stored in {trials_path}")


if __name__ == "__main__":
    app()
