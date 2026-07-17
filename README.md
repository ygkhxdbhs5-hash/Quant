# Quant Backtest Project

Clean project layout around the existing Nasdaq institutional production engine.

Investment logic is preserved as-is in `engine/strategy.py`.  
Data download is separate. The engine only reads local files under `data/`.

## Layout

```
project/
├── downloader/
│   ├── download_prices.py
│   ├── download_fundamentals.py
│   └── update_data.py
├── data/
│   ├── prices/
│   ├── fundamentals/
│   └── metadata/
├── engine/
│   └── strategy.py          # existing engine (logic unchanged)
├── config/
│   └── config.yaml
├── cache/
└── run_backtest.py          # single backtest entry point
```

## Setup

```bash
pip install -r requirements.txt
```

## 1) Download / update data

Prices (Yahoo via yfinance):

```bash
python -m downloader.download_prices --config config/config.yaml
```

Fundamentals — import a vendor/export CSV that matches the engine Fine schema:

```bash
python -m downloader.download_fundamentals --source path/to/fundamentals.csv
```

Required fundamental columns:

`as_of_date, symbol, op_margin, roic, gross_profit, total_revenue, total_assets, ocf, capex, market_cap, industry`

Optional: `total_debt, cash, dollar_volume`

Or update both:

```bash
python -m downloader.update_data --fundamentals-source path/to/fundamentals.csv
```

Smoke-test placeholders (not for production results):

```bash
python -m downloader.update_data --placeholder-fundamentals
```

## 2) Run the backtest

```bash
python run_backtest.py --config config/config.yaml
```

Outputs:

- `cache/equity_curve.csv`
- `cache/orders.csv`

## Important

- Do not change factor calculations, ranking, portfolio construction, or risk management in `engine/strategy.py`.
- Config defaults in `config/config.yaml` mirror the original `Initialize()` values.
- Identical results require the same local price/fundamental history the strategy was validated on (export QC/LEAN Fine+price history into `data/` for exact parity).
