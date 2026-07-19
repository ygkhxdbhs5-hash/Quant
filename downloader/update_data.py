"""Orchestrate full local dataset refresh via Massive.com."""

from __future__ import annotations

import argparse

from downloader.download_fundamentals import main as fundamentals_main
from downloader.download_prices import main as prices_main
from downloader.download_universe import main as universe_main
from downloader.download_universe_utils import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update all local datasets (Massive.com)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-universe", action="store_true")
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument(
        "--skip-fundamentals",
        action="store_true",
        help="Skip PIT fundamentals download (default when config.download_fundamentals is false)",
    )
    parser.add_argument(
        "--with-fundamentals",
        action="store_true",
        help="Force fundamentals download even if config.download_fundamentals is false",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    # CMVS v3 does not need fundamentals; default off unless explicitly enabled.
    want_fundamentals = bool(config.get("download_fundamentals", False))
    if args.with_fundamentals:
        want_fundamentals = True
    if args.skip_fundamentals:
        want_fundamentals = False

    cfg = ["--config", args.config]
    if not args.skip_universe:
        universe_main(cfg)
    if not args.skip_prices:
        prices_main(cfg)
    if want_fundamentals:
        fundamentals_main(cfg)
    else:
        print(
            "[update_data] Skipping fundamentals "
            "(set download_fundamentals: true or pass --with-fundamentals)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
