# Q_Alpha v5 — Pure Python Standalone Backtest

LEAN-free institutional engine. Downloaders use **Massive.com** (ex-Polygon)
and write into `data/`. The engine only reads those local files.
Entry point: `run_backtest.py`.

## Layout

```
downloader/              # Massive.com download pipeline
data/
  metadata/universe.pkl
  prices/panels.pkl
  fundamentals/pit_history.pkl
engine/strategy.py       # StandaloneEngine (v5 investment logic)
engine/breakout_screener.py  # Local-data breakout candidate filter
config/config.yaml
cache/                   # HTTP cache + equity outputs
run_backtest.py
run_breakout_screener.py
app.py                   # Streamlit UI (includes Breakout Screener tab)
```

## Setup

```bash
pip install -r requirements.txt
export MASSIVE_API_KEY=your_key_here
```

Or set `massive_api_key` in `config/config.yaml`.

## 1) Download data

```bash
python -m downloader.update_data --config config/config.yaml
```

Or step by step:

```bash
python -m downloader.download_universe
python -m downloader.download_prices
python -m downloader.download_fundamentals
```

Tips:
- Set `universe_limit` (e.g. `50`) for a smaller first download
- Keep a curated list in `data/metadata/symbols.txt` if desired
- Financials require a Massive plan that includes Financials & Ratios

## 2) Run backtest

```bash
python run_backtest.py --config config/config.yaml
```

Outputs: `cache/equity_curve.csv`, `cache/equity_curve_v5.png`

## 3) Breakout candidate screener

Scan the **already-downloaded** price panels for fresh 30-EMA + horizontal
resistance breakouts (no re-download, no trading).

### Streamlit UI

```bash
streamlit run app.py
```

Open the **Breakout Screener** tab. Filter knobs are in the sidebar.

### Console

```bash
python run_breakout_screener.py --config config/config.yaml
```

Optional: `--as-of YYYY-MM-DD`, `--out cache/breakout_candidates.csv`.

Tunable knobs also live under `breakout_screener:` in `config/config.yaml`
(EMA period, resistance lookback, volume threshold, max breakout age, etc.).

## Notes

- Investment pipeline matches Q_Alpha v5 (regime / factors / rank / portfolio / risk / next-open execution).
- Value (FCF/Sales yield) omitted in v5 (documented in strategy).
- ROIC uses a flat 21% tax approximation.
- PIT fundamentals use Massive `filing_date` as the accepted-date proxy.
