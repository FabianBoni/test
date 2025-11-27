# Snowball Bot Backtesting Suite

Python toolkit to evaluate a concentrated liquidity auto-compounding strategy ("Snowball Bot") on Uniswap V3 / Arbitrum over the last 90 days. See `docs/backtest_plan.md` for architecture details.

## CLI usage

Set global defaults in `.env` (loaded automatically):

```
BACKTEST_POOL_ADDRESS=0xc31e54c7a869b9fcbecc14363cf510d1c41fa443
```

Then `--pool` becomes optional on all commands.

- Run a single simulation (cached data is reused unless `--refresh-data` is provided):
	```bash
	python cli.py run --start 2025-08-28T00:00:00 --end 2025-11-26T00:00:00 --pool 0xc31e54c7a869b9fcbecc14363cf510d1c41fa443
	```
- Override strategy knobs directly from CLI, e.g. concentrate liquidity and redeploy aggressively:
	```bash
	python cli.py run --start ... --end ... --pool ... --tight-width 0.002 --base-width 0.004 --wide-width 0.006 --initial-width 0.005 --fee-threshold-multiple 1.05 --gas-cost-usd 0.01
	```
- Launch Bayesian optimization of the range widths and compounding threshold (set `--trials` as needed):
	```bash
	python cli.py optimize --start 2025-08-28T00:00:00 --end 2025-11-26T00:00:00 --pool 0xc31e54c7a869b9fcbecc14363cf510d1c41fa443 --metric daily_return --target-daily-return-min 0.01 --target-daily-return-max 0.02 --trials 60
	```
	The optimizer now searches across multiple rebalance intervals, initial widths, gas cost assumptions, and compounding thresholds while nudging solutions toward 1–2% daily return.
