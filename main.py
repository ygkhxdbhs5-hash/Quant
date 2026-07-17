#!/usr/bin/env python3
"""CLI entry point for running quant backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant.data import generate_sample_ohlcv, load_csv, save_csv
from quant.engine import BacktestEngine
from quant.report import plot_equity, print_report
from strategies import RSIMeanReversion, SMACrossover, YourStrategy


STRATEGY_MAP = {
    "sma": SMACrossover,
    "rsi": RSIMeanReversion,
    "yours": YourStrategy,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant trade backtester")
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGY_MAP),
        default="sma",
        help="Strategy to run (default: sma)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to OHLCV CSV. If omitted, synthetic data is used.",
    )
    parser.add_argument("--cash", type=float, default=100_000.0, help="Initial cash")
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Commission rate per fill (e.g. 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="output/equity.png",
        help="Path to save equity chart (empty string to skip)",
    )
    parser.add_argument(
        "--write-sample-data",
        type=str,
        default=None,
        help="Optional path to write generated sample CSV and exit",
    )
    # SMA params
    parser.add_argument("--fast", type=int, default=10)
    parser.add_argument("--slow", type=int, default=30)
    # RSI params
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--oversold", type=float, default=30.0)
    parser.add_argument("--overbought", type=float, default=70.0)
    # YourStrategy params
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--entry-z", type=float, default=-1.0)
    parser.add_argument("--exit-z", type=float, default=0.0)
    return parser


def make_strategy(args: argparse.Namespace):
    if args.strategy == "sma":
        return SMACrossover(fast=args.fast, slow=args.slow)
    if args.strategy == "rsi":
        return RSIMeanReversion(
            period=args.rsi_period,
            oversold=args.oversold,
            overbought=args.overbought,
        )
    return YourStrategy(
        lookback=args.lookback,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.write_sample_data:
        df = generate_sample_ohlcv()
        path = save_csv(df, args.write_sample_data)
        print(f"Wrote sample data to {path}")
        return 0

    if args.data:
        data = load_csv(args.data)
    else:
        data = generate_sample_ohlcv()
        sample_path = Path("data/sample_ohlcv.csv")
        if not sample_path.exists():
            save_csv(data, sample_path)

    strategy = make_strategy(args)
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=args.cash,
        commission_rate=args.commission,
    )
    result = engine.run(data)
    print_report(result, strategy_name=strategy.name)

    if args.plot:
        saved = plot_equity(result, price=data["close"], output_path=args.plot)
        if saved:
            print(f"\nSaved chart to {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
