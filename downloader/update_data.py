"""Orchestrate full local dataset refresh via Massive.com."""

from __future__ import annotations

import argparse

from downloader.download_fundamentals import main as fundamentals_main
from downloader.download_prices import main as prices_main
from downloader.download_universe import main as universe_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update all local datasets (Massive.com)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-universe", action="store_true")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-fundamentals", action="store_true")
    args = parser.parse_args(argv)

    cfg = ["--config", args.config]
    if not args.skip_universe:
        universe_main(cfg)
    if not args.skip_prices:
        prices_main(cfg)
    if not args.skip_fundamentals:
        fundamentals_main(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
