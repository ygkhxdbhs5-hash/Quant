#!/usr/bin/env python3
"""Single entry point for running the local backtest.

The engine reads only from data/ (prices, fundamentals). Use the downloader
package first to populate those directories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from engine.backtest_runner import BacktestRunner
from engine.data_store import DataStore


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Nasdaq institutional backtest")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--equity-out", default="cache/equity_curve.csv")
    parser.add_argument("--orders-out", default="cache/orders.csv")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    paths = config.get("paths", {})
    store = DataStore(
        prices_dir=paths.get("prices", "data/prices"),
        fundamentals_dir=paths.get("fundamentals", "data/fundamentals"),
        metadata_dir=paths.get("metadata", "data/metadata"),
    )

    if not store.list_price_symbols():
        raise SystemExit(
            "No price files found in data/prices/. "
            "Run: python -m downloader.update_data --symbols QQQ AAPL MSFT ..."
        )

    runner = BacktestRunner(config=config, store=store)
    result = runner.run()

    print("\n=== Backtest Complete ===")
    print(f"Final portfolio value: ${result.final_value:,.2f}")
    if len(result.equity_curve):
        start_val = float(result.equity_curve.iloc[0])
        total_return = result.final_value / start_val - 1.0 if start_val else 0.0
        print(f"Total return: {total_return * 100:.2f}%")
        print(f"Bars: {len(result.equity_curve)} | Orders: {len(result.orders)}")

    equity_out = Path(args.equity_out)
    equity_out.parent.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(equity_out, header=True)
    print(f"Wrote equity curve -> {equity_out}")

    orders_out = Path(args.orders_out)
    if result.orders:
        pd.DataFrame(result.orders).to_csv(orders_out, index=False)
        print(f"Wrote orders -> {orders_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
