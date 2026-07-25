# Baseline v1 — 12-1 Momentum + ATR Trail

Minimal strategy baseline for future factor experiments.

## Spec

| Piece | Rule |
|---|---|
| Entry | `mom_12_1 = price[t-21]/price[t-252]-1` among Top 250 dollar-volume → Top 30 |
| Exit | ATR(14) Wilder trail, mult=2.5; Low < stop → next Open |
| Sizing | Equal weight `1/N` |
| Leverage | Always 1.0x |
| Rebalance | Monthly candidate refresh; hold until ATR; refill empties only |

## Modules

- `engine/strategy_baseline_v1.py` — strategy decisions only
- `engine/baseline_engine.py` — loads existing panels + execution model + reporting
- `run_baseline_v1.py` — runs 2018–20 / 2021–22 / 2023–25 and appends `experiment_history.json`
- `run_backtest.py` — single-window entry point (override with `--start` / `--end`)

## Run

```bash
python -m downloader.update_data --config config/config.yaml --skip-fundamentals
python run_baseline_v1.py --config config/config.yaml
```

Downloader / PIT / panel builders / research report modules are not modified.
