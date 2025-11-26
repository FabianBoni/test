# Uniswap V3 90-Day Backtest Plan

## Objectives
- Quantify whether a 1% daily compounding target is realistic for a concentrated WETH/USDC LP strategy on Arbitrum.
- Simulate a "Snowball" auto-compounding loop that collects fees, redeploys liquidity, and adapts ranges based on predicted short-term volatility.
- Produce metrics (equity curve, cumulative fees, IL, drawdowns) and reusable components for further optimization.

## Data Strategy
1. **Source**: The Graph Uniswap V3 Arbitrum subgraph (`https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-arbitrum-one`).
2. **Endpoints**:
   - `swaps` for realized prices/volume (15-min buckets built client-side).
   - `poolDayData` / `poolHourData` for liquidity, fees, TVL sanity checks.
   - `ticks`/`positions` snapshots to estimate fee growth outside events.
3. **Caching**: Store raw JSON responses in `data/raw/` and normalized parquet/CSV tables in `data/processed/` for replay without re-querying.
4. **Fallback**: Allow CSV import so we can backtest offline if the subgraph rate-limits us.

## Module Layout
```
backtest/
  __init__.py
  config.py          # Network, pool, fee tier, time window, gas assumptions.
  data_loader.py     # GraphQL client, pagination, local caching helpers.
  feature_engineering.py  # Build price series, realized volatility inputs.
  volatility.py      # ARIMA/GARCH wrappers using statsmodels/arch.
  strategy.py        # Range sizing policy based on predicted volatility + thresholds.
  simulator.py       # Event loop: track position, fees, IL, compounding, rebalancing.
  metrics.py         # KPI calculations + plotting hooks.
cli.py               # Click/Typer entry point to run a backtest from the shell.
notebooks/
  exploration.ipynb  # Optional EDA/visualization.
```

## Core Algorithms
1. **Volatility Forecast**
   - Compute rolling log-returns from 5-min or 15-min prices.
   - Train/update ARIMA or GARCH(1,1) on the fly; predict next-hour sigma.
2. **Range Selection Policy**
   - When predicted sigma < low threshold: tighten to e.g. ±0.5%.
   - When sigma medium: default ±1%.
   - When sigma high: widen to ±2% and reduce notional exposure.
3. **Compounding Loop (15-min cadence)**
   - Verify current price within tick range; if not, trigger withdraw + redeploy after adjusting ticks.
   - Estimate fees accrued since last action using feeGrowthGlobal data; add to balance when `fees > gas_cost * buffer`.
   - Reinvest fees by increasing liquidity in same or new range.
   - Track inventory (ETH vs USDC), impermanent loss, and PnL vs buy-and-hold benchmark.

## Dependencies
- `python>=3.10`
- `pandas`, `numpy`, `httpx`, `gql`, `pydantic`
- `statsmodels`, `arch`, `scipy`
- `typer` for CLI, `rich` for logging, `plotly`/`matplotlib` for visuals (optional)

## Deliverables (v0)
1. Configurable CLI: `python -m backtest --start 2025-08-28 --end 2025-11-26 --pool 0x...`
2. JSON/CSV report with KPIs + equity curve.
3. Notebook showing last 90 days performance vs target 1% daily compounding.
4. Clear separation between data ingestion, forecasting, and execution logic for future on-chain integration.
