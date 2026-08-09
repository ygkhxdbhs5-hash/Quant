#!/usr/bin/env python3
"""Console entry point: scan local panels for fresh EMA + resistance breakouts.

Uses data already downloaded under data/ (same panels as the backtest engine).
Does not download market data or place trades.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.breakout_screener import (
    breakout_config_from_dict,
    config_summary,
    format_candidates_table,
    run_screener,
)
from engine.strategy import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Breakout candidate screener over local price panels"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--as-of",
        default=None,
        help="Scan date YYYY-MM-DD (default: latest session in panels)",
    )
    parser.add_argument(
        "--out",
        default="cache/breakout_candidates.csv",
        help="CSV path for filtered candidates",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    screener_cfg = breakout_config_from_dict(config)

    print("=" * 64)
    print("  BREAKOUT CANDIDATE SCREENER  (local dataset only)")
    print("=" * 64)
    print(f"Config: {config_summary(screener_cfg)}")
    print(f"Source: {Path(config.get('paths', {}).get('prices', 'data/prices')) / 'panels.pkl'}")
    if args.as_of:
        print(f"As-of : {args.as_of}")
    print("-" * 64)

    try:
        candidates = run_screener(config=config, as_of=args.as_of)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(format_candidates_table(candidates))
    print("-" * 64)
    print(f"Candidates: {len(candidates)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out, index=False)
    print(f"Wrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
