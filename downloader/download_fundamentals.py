"""Download / import fundamental snapshots into data/fundamentals/.

Local fundamentals must supply the raw fields that FineSelectionFunction reads
so investment logic stays untouched:

  as_of_date, op_margin, roic, gross_profit, total_revenue, total_assets,
  ocf, capex, market_cap, total_debt, cash, industry, dollar_volume

This module can:
  1) Import a vendor/export CSV into per-symbol files, or
  2) Build placeholder fundamentals for smoke tests (NOT for production results).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = [
    "as_of_date",
    "symbol",
    "op_margin",
    "roic",
    "gross_profit",
    "total_revenue",
    "total_assets",
    "ocf",
    "capex",
    "market_cap",
    "industry",
]


def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def import_fundamentals_csv(source: Path, output_dir: Path) -> List[Path]:
    df = pd.read_csv(source)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Fundamentals CSV missing columns: {missing}")

    if "total_debt" not in df.columns:
        df["total_debt"] = np.nan
    if "cash" not in df.columns:
        df["cash"] = np.nan
    if "dollar_volume" not in df.columns:
        df["dollar_volume"] = 0.0

    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["symbol"] = df["symbol"].astype(str).str.upper()

    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for symbol, group in df.groupby("symbol"):
        path = output_dir / f"{symbol}.csv"
        cols = [
            "as_of_date",
            "op_margin",
            "roic",
            "gross_profit",
            "total_revenue",
            "total_assets",
            "ocf",
            "capex",
            "market_cap",
            "total_debt",
            "cash",
            "industry",
            "dollar_volume",
        ]
        group[cols].sort_values("as_of_date").to_csv(path, index=False)
        written.append(path)
        print(f"[fundamentals] wrote {path} ({len(group)} rows)")
    return written


def generate_placeholder_fundamentals(
    symbols: List[str],
    prices_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> List[Path]:
    """Create synthetic fundamentals aligned to month-ends for pipeline smoke tests."""
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    industries = ["Software", "Semiconductors", "Biotech", "Retail", "Banks", "Telecom", "Energy", "Industrials", "Unknown"]
    written: List[Path] = []

    for i, symbol in enumerate(symbols):
        price_path = prices_dir / f"{symbol}.csv"
        if not price_path.exists():
            print(f"[fundamentals] skip {symbol}: no price file")
            continue
        prices = pd.read_csv(price_path, parse_dates=["date"]).set_index("date").sort_index()
        month_ends = prices.groupby([prices.index.year, prices.index.month]).tail(1).index
        rows = []
        for ts in month_ends:
            close = float(prices.loc[ts, "close"])
            volume = float(prices.loc[ts, "volume"]) if "volume" in prices.columns else 0.0
            total_assets = abs(rng.normal(5e9, 1e9))
            gross_profit = abs(rng.normal(0.35, 0.05)) * total_assets
            sales = abs(rng.normal(0.8, 0.1)) * total_assets
            ocf = abs(rng.normal(0.12, 0.03)) * total_assets
            capex = abs(rng.normal(0.04, 0.01)) * total_assets
            market_cap = close * abs(rng.normal(1e9, 2e8))
            rows.append(
                {
                    "as_of_date": ts.strftime("%Y-%m-%d"),
                    "op_margin": float(rng.uniform(0.05, 0.35)),
                    "roic": float(rng.uniform(0.05, 0.30)),
                    "gross_profit": gross_profit,
                    "total_revenue": sales,
                    "total_assets": total_assets,
                    "ocf": ocf,
                    "capex": capex,
                    "market_cap": market_cap,
                    "total_debt": float(rng.uniform(0, 1e9)),
                    "cash": float(rng.uniform(0, 5e8)),
                    "industry": industries[i % len(industries)],
                    "dollar_volume": close * volume,
                }
            )
        out = pd.DataFrame(rows)
        path = output_dir / f"{symbol}.csv"
        out.to_csv(path, index=False)
        written.append(path)
        print(f"[fundamentals] wrote placeholder {path} ({len(out)} rows)")
    return written


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import or generate fundamentals into data/fundamentals/")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", type=str, default=None, help="Vendor/export CSV to import")
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Generate synthetic fundamentals for smoke tests (not production)",
    )
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    paths = config.get("paths", {})
    output_dir = Path(paths.get("fundamentals", "data/fundamentals"))
    prices_dir = Path(paths.get("prices", "data/prices"))
    metadata_dir = Path(paths.get("metadata", "data/metadata"))

    if args.source:
        import_fundamentals_csv(Path(args.source), output_dir)
        return 0

    if args.placeholder:
        symbols_file = metadata_dir / "symbols.txt"
        if symbols_file.exists():
            symbols = [
                line.strip().upper()
                for line in symbols_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        else:
            symbols = sorted(p.stem.upper() for p in prices_dir.glob("*.csv") if p.stem.upper() != "QQQ")
        generate_placeholder_fundamentals(symbols, prices_dir, output_dir)
        return 0

    raise SystemExit("Provide --source CSV or --placeholder for smoke-test data.")


if __name__ == "__main__":
    raise SystemExit(main())
