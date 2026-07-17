# Q_Alpha v5 — Pure Python Standalone Backtest

LEAN-free institutional engine. Downloaders write into `data/`; the engine only
reads those local files. Entry point: `run_backtest.py`.

## Layout

```
downloader/              # FMP download pipeline (stable API)
data/
  metadata/universe.pkl
  prices/panels.pkl
  fundamentals/pit_history.pkl
engine/strategy.py       # StandaloneEngine (v5 investment logic)
config/config.yaml
cache/                   # HTTP cache + equity outputs
run_backtest.py
```

## Setup

```bash
pip install -r requirements.txt
export FMP_API_KEY=...   # optional; overrides config
```

## 1) Download data

```bash
python -m downloader.update_data --config config/config.yaml
```

Or:

```bash
python -m downloader.download_universe
python -m downloader.download_prices
python -m downloader.download_fundamentals
```

Tips for limited FMP plans:
- Set `universe_limit` (e.g. `50`) and maintain `data/metadata/symbols.txt`
- `benchmark: SPY` if `QQQ` is plan-restricted
- `statement_limit: 5` on free plans (raise if your plan allows more history)

## 2) Run backtest

```bash
python run_backtest.py --config config/config.yaml
```

Outputs: `cache/equity_curve.csv`, `cache/equity_curve_v5.png`

## Notes

- Investment pipeline matches the provided Q_Alpha v5 engine (regime, factors,
  rank, portfolio, risk, next-open execution, delisting handlers).
- Value (FCF/Sales yield) omitted in v5 (no share count / EV scale).
- ROIC uses a flat 21% tax approximation.
- Downloaders use FMP **stable** endpoints (legacy `/api/v3/...` is shut down).
