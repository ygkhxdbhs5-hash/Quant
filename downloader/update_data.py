"""Orchestrate price + fundamentals updates into the local data/ directory."""

from __future__ import annotations

import argparse
from typing import List

from downloader.download_fundamentals import main as fundamentals_main
from downloader.download_prices import main as prices_main


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update local price and fundamental datasets")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--fundamentals-source", type=str, default=None)
    parser.add_argument("--placeholder-fundamentals", action="store_true")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-fundamentals", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_prices:
        price_argv = ["--config", args.config]
        if args.symbols:
            price_argv += ["--symbols", *args.symbols]
        prices_main(price_argv)

    if not args.skip_fundamentals:
        fund_argv = ["--config", args.config]
        if args.fundamentals_source:
            fund_argv += ["--source", args.fundamentals_source]
        elif args.placeholder_fundamentals:
            fund_argv += ["--placeholder"]
        else:
            print("[update] skipping fundamentals (pass --fundamentals-source or --placeholder-fundamentals)")
            return 0
        fundamentals_main(fund_argv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
