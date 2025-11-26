# Snowball Bot Backtesting Suite

Python toolkit to evaluate a concentrated liquidity auto-compounding strategy ("Snowball Bot") on Uniswap V3 / Arbitrum over the last 90 days. See `docs/backtest_plan.md` for architecture details.

## CLI usage

- Run a single simulation (cached data is reused unless `--refresh-data` is provided):
	```bash
	python cli.py run --start 2025-08-28T00:00:00 --end 2025-11-26T00:00:00 --pool 0xc31e54c7a869b9fcbecc14363cf510d1c41fa443
	```
- Launch Bayesian optimization of the range widths and compounding threshold (set `--trials` as needed):
	```bash
	python cli.py optimize --start 2025-08-28T00:00:00 --end 2025-11-26T00:00:00 --pool 0xc31e54c7a869b9fcbecc14363cf510d1c41fa443 --trials 40
	```
