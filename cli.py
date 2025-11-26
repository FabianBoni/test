"""Typer CLI entry for running the Snowball backtest."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import fmean

import typer
from dotenv import load_dotenv

from backtest import BacktestConfig, Simulator
from backtest.data_loader import PoolInfoLoader, SwapLoader
from backtest.feature_engineering import build_time_bars, load_cached_bars, save_bars
from backtest.metrics import cagr, daily_return, max_drawdown, sharpe_ratio

load_dotenv(dotenv_path=Path(".env"), override=False)

app = typer.Typer(help="Backtest the Snowball bot over historical Uniswap V3 data")


@app.command()
def run(
    start: datetime = typer.Option(..., help="UTC start timestamp"),
    end: datetime = typer.Option(..., help="UTC end timestamp"),
    pool: str = typer.Option(..., help="Pool address"),
    interval: int = typer.Option(15, help="Bar interval in minutes"),
    refresh_data: bool = typer.Option(False, help="Force refetch graph data and rebuild bars"),
) -> None:
    config = BacktestConfig(start=start, end=end, pool_address=pool, rebalance_interval_minutes=interval)
    config.ensure_directories()

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

    typer.echo("↻ Fetching pool metadata …")
    pool_info = PoolInfoLoader(config).fetch()
    typer.echo(f"✔ Pool TVL: ${pool_info.total_value_locked_usd:,.0f}")

    typer.echo("↻ Running simulator …")
    simulator = Simulator(config, pool_tvl_usd=pool_info.total_value_locked_usd)
    result = simulator.run(bars)
    typer.echo("✔ Simulation complete")

    period_returns = []
    for prev, curr in zip(result.equity_curve[:-1], result.equity_curve[1:]):
        if prev <= 0:
            period_returns.append(0.0)
        else:
            period_returns.append((curr - prev) / prev)
    points_per_day = max(1, (24 * 60) // interval)
    metrics = {
        "final_equity": result.equity_curve[-1],
        "total_fees": result.fee_curve[-1],
        "daily_return": daily_return(result.equity_curve, points_per_day),
        "cagr": cagr(result.equity_curve, interval),
        "sharpe": sharpe_ratio(period_returns),
        "max_drawdown": max_drawdown(result.equity_curve),
        "avg_period_return": fmean(period_returns) if period_returns else 0.0,
        "final_il": result.il_curve[-1] if result.il_curve else 0.0,
    }

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


if __name__ == "__main__":
    app()
