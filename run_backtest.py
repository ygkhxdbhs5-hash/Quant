#!/usr/bin/env python3
"""Single entry point for the Baseline v1 backtest.

Requires local datasets under data/ (populate via downloader.update_data).
Strategy logic lives in engine/strategy_baseline_v1.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from engine.baseline_engine import BaselineEngineV1
from engine.strategy import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Baseline v1 (mom 12-1 + ATR trail)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", default=None, help="Override START_DATE (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Override END_DATE (YYYY-MM-DD)")
    parser.add_argument("--equity-out", default="cache/baseline_v1/equity_curve.csv")
    parser.add_argument("--chart-out", default="cache/baseline_v1/equity_curve.png")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.start:
        config["start_date"] = args.start
    if args.end:
        config["end_date"] = args.end

    engine = BaselineEngineV1(config=config, config_path=args.config)
    result = engine.run()

    if result.empty:
        print("No equity curve produced (check data coverage / warmup).")
        return 1

    equity_out = Path(args.equity_out)
    equity_out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(equity_out)
    print(f"Wrote equity curve -> {equity_out}")

    chart_out = Path(args.chart_out)
    chart_out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))
    plt.plot(result["Total_Equity"], color="darkgreen", lw=2.5, label="Baseline v1")
    qqq_path = Path("cache/baseline_v1/qqq_equity_curve.csv")
    if qqq_path.exists():
        import pandas as pd

        qqq = pd.read_csv(qqq_path, parse_dates=["Date"]).set_index("Date")
        if "Total_Equity" in qqq.columns:
            plt.plot(qqq["Total_Equity"], color="steelblue", lw=1.8, label="QQQ B&H")
    plt.title("Baseline v1 — 12-1 Momentum + ATR Trail", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.savefig(chart_out, dpi=150)
    plt.close()
    print(f"Chart saved: {chart_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
