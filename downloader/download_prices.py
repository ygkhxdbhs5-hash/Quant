"""Download OHLCV panels into data/prices/panels.pkl (FMP stable EOD)."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from downloader.download_universe_utils import load_config, make_client
from downloader.fmp_client import FmpClient


def _historical_to_df(res) -> pd.DataFrame:
    """Normalize stable list response and legacy {historical:[...]} shapes."""
    if res is None:
        return pd.DataFrame()
    if isinstance(res, dict) and res.get("historical"):
        rows = res["historical"]
    elif isinstance(res, list):
        rows = res
    else:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Stable EOD has close (no adjClose). Prefer adjClose when present.
    if "adjClose" not in df.columns and "close" in df.columns:
        df["adjClose"] = df["close"]
    return df


def build_price_panel(
    client: FmpClient,
    tickers: List[str],
    delisted_meta: dict,
    profile_meta: dict,
    start_date: str,
    end_date: str,
    benchmark: str,
    silent_delist_gap_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.Timestamp]]:
    prices_open, prices_close, highs, lows, vols = {}, {}, {}, {}, {}
    silent_delist_flags: Dict[str, pd.Timestamp] = {}
    end_ts = pd.to_datetime(end_date)

    print(f">> {len(tickers)}개 종목 OHLCV 수집...")
    for i, t in enumerate(tickers):
        res = client.cached_get(
            f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={t}&from={start_date}&to={end_date}&apikey={client.api_key}",
            f"prices_stable_{t}_{start_date}_{end_date}",
            ttl_days=1,
        )
        df = _historical_to_df(res)
        if df.empty:
            continue

        first_price_date, last_price_date = df.index.min(), df.index.max()

        if t in delisted_meta and delisted_meta[t].get("delistingDate"):
            df = df.loc[: pd.to_datetime(delisted_meta[t]["delistingDate"])]
        elif (end_ts - last_price_date).days > silent_delist_gap_days:
            silent_delist_flags[t] = last_price_date

        ipo_date = profile_meta.get(t, {}).get("ipoDate", pd.NaT)
        effective_start = (
            max([d for d in [ipo_date, first_price_date] if pd.notna(d)])
            if pd.notna(ipo_date)
            else first_price_date
        )
        df = df.loc[df.index >= effective_start]

        if not df.empty and {"open", "high", "low", "adjClose", "volume"}.issubset(df.columns):
            prices_open[t] = df["open"]
            prices_close[t] = df["adjClose"]
            highs[t] = df["high"]
            lows[t] = df["low"]
            vols[t] = df["adjClose"] * df["volume"]

        if (i + 1) % 50 == 0:
            print(f"    ...{i+1}/{len(tickers)}")

    # 벤치마크
    bm_res = client.cached_get(
        f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={benchmark}&from={start_date}&to={end_date}&apikey={client.api_key}",
        f"prices_stable_{benchmark}_{start_date}_{end_date}",
        ttl_days=1,
    )
    bm_df = _historical_to_df(bm_res)
    if bm_df.empty:
        raise RuntimeError(
            f"Benchmark prices unavailable for {benchmark}. "
            "Your FMP plan may restrict this symbol — set config.benchmark to an available ticker (e.g. SPY)."
        )
    prices_close[benchmark] = bm_df["adjClose"]
    prices_open[benchmark] = bm_df["open"]
    highs[benchmark] = bm_df["high"]
    lows[benchmark] = bm_df["low"]
    vols[benchmark] = bm_df["adjClose"] * bm_df["volume"]

    close_matrix = pd.DataFrame(prices_close)
    idx = close_matrix.index
    open_matrix = pd.DataFrame(prices_open).reindex(idx)
    high_matrix = pd.DataFrame(highs).reindex(idx)
    low_matrix = pd.DataFrame(lows).reindex(idx)
    dvol_matrix = pd.DataFrame(vols).reindex(idx)

    for t in close_matrix.columns:
        mask = close_matrix[t].notna()
        open_matrix[t] = open_matrix[t].where(mask)
        high_matrix[t] = high_matrix[t].where(mask)
        low_matrix[t] = low_matrix[t].where(mask)
        dvol_matrix[t] = dvol_matrix[t].where(mask)

    print(f" -> 조용한 상장폐지 감지: {len(silent_delist_flags)}건")
    return open_matrix, close_matrix, high_matrix, low_matrix, dvol_matrix, silent_delist_flags


def save_panels(prices_dir: Path, panels: dict) -> Path:
    prices_dir.mkdir(parents=True, exist_ok=True)
    path = prices_dir / "panels.pkl"
    with open(path, "wb") as f:
        pickle.dump(panels, f)
    print(f"[prices] wrote {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download OHLCV price panels")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    client = make_client(config)
    paths = config.get("paths", {})

    universe_path = Path(paths.get("metadata", "data/metadata")) / "universe.pkl"
    if not universe_path.exists():
        raise SystemExit(f"Missing {universe_path}. Run download_universe first.")
    with open(universe_path, "rb") as f:
        universe = pickle.load(f)

    open_m, close_m, high_m, low_m, dvol_m, silent = build_price_panel(
        client=client,
        tickers=universe["tickers"],
        delisted_meta=universe["delisted_meta"],
        profile_meta=universe["profile_meta"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        benchmark=config.get("benchmark", "QQQ"),
        silent_delist_gap_days=int(config.get("silent_delist_gap_days", 10)),
    )
    save_panels(
        Path(paths.get("prices", "data/prices")),
        {
            "open_m": open_m,
            "close_m": close_m,
            "high_m": high_m,
            "low_m": low_m,
            "dvol_m": dvol_m,
            "silent_delist_flags": silent,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
