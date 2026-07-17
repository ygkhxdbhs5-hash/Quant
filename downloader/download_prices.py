"""Download daily OHLCV prices into data/prices/."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yaml


def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _read_symbol_list(metadata_dir: Path, config: dict) -> List[str]:
    symbols_file = metadata_dir / "symbols.txt"
    symbols: List[str] = []
    if symbols_file.exists():
        symbols = [
            line.strip().upper()
            for line in symbols_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    benchmark = str(config.get("benchmark", "QQQ")).upper()
    if benchmark not in symbols:
        symbols.insert(0, benchmark)
    return symbols


def download_prices(
    symbols: Iterable[str],
    output_dir: Path,
    start: str,
    end: str,
) -> List[Path]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit(
            "yfinance is required for price download. Install with: pip install yfinance"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=False)
        if df.empty:
            print(f"[prices] no data for {symbol}")
            continue
        out = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )[["open", "high", "low", "close", "volume"]].copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out.index.name = "date"
        path = output_dir / f"{symbol.upper()}.csv"
        out.to_csv(path)
        written.append(path)
        print(f"[prices] wrote {path} ({len(out)} rows)")
    return written


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download OHLCV prices into data/prices/")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--symbols", nargs="*", default=None, help="Optional symbol override list")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    paths = config.get("paths", {})
    prices_dir = Path(paths.get("prices", "data/prices"))
    metadata_dir = Path(paths.get("metadata", "data/metadata"))

    start = "{:04d}-{:02d}-{:02d}".format(*config.get("start_date", [2007, 1, 1]))
    end = "{:04d}-{:02d}-{:02d}".format(*config.get("end_date", [2026, 6, 30]))

    symbols = [s.upper() for s in args.symbols] if args.symbols else _read_symbol_list(metadata_dir, config)
    download_prices(symbols, prices_dir, start=start, end=end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
