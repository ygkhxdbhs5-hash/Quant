# Quant Trade Backtester

Python backtesting framework for quantitative trading strategies. Plug in your own strategy, load OHLCV data, simulate fills, and get performance metrics plus an equity chart.

## Features

- Strategy interface (`BUY` / `SELL` / `HOLD`) with no-lookahead bar windows
- Next-bar open fill model (realistic default)
- Portfolio tracking with commission
- Metrics: total return, CAGR, Sharpe, max drawdown, win rate, profit factor
- Built-in examples: SMA crossover, RSI mean reversion, z-score template for your code
- CLI runner + unit tests

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run SMA crossover on synthetic data
python main.py --strategy sma

# Run your strategy template
python main.py --strategy yours --lookback 20 --entry-z -1.0 --exit-z 0.0

# Run against your CSV
python main.py --strategy rsi --data data/sample_ohlcv.csv --plot output/equity.png
```

Generate a sample CSV:

```bash
python main.py --write-sample-data data/sample_ohlcv.csv
```

## Plug in your strategy

1. Open `strategies/your_strategy.py` (or add a new file under `strategies/`).
2. Subclass `Strategy` and implement `on_bar(self, window) -> Signal`.
3. Register it in `main.py` (`STRATEGY_MAP`) if you want CLI access.

Minimal example:

```python
from quant.strategy import Signal, Strategy
import pandas as pd

class MyStrategy(Strategy):
    name = "MyStrategy"

    def on_bar(self, window: pd.DataFrame) -> Signal:
        # window ends at the current bar — do not look ahead
        if len(window) < 2:
            return Signal.HOLD
        if window["close"].iloc[-1] > window["close"].iloc[-2]:
            return Signal.BUY
        return Signal.SELL
```

Run programmatically:

```python
from quant.data import load_csv
from quant.engine import BacktestEngine
from strategies.your_strategy import YourStrategy

data = load_csv("data/sample_ohlcv.csv")
result = BacktestEngine(YourStrategy()).run(data)
print(result.metrics.summary())
```

## CSV format

Required columns (case-insensitive): `date` (or datetime index), `open`, `high`, `low`, `close`. Optional: `volume`.

```csv
date,open,high,low,close,volume
2022-01-03,100.0,101.2,99.5,100.8,500000
```

## Project layout

```
quant/           # engine, portfolio, metrics, data, report
strategies/      # SMA, RSI, and your strategy template
tests/           # pytest suite
main.py          # CLI
```

## Tests

```bash
pytest -q
```
