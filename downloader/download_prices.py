"""Download OHLCV panels via Massive aggregates into data/prices/panels.pkl."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from downloader.download_universe_utils import load_config, make_client
from downloader.massive_client import MassiveClient


def _aggs_to_df(payload) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()
    rows = payload.get("results") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # t = unix ms, o/h/l/c/v
    if "t" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.normalize()
    df["date"] = df["date"].dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    out = pd.DataFrame(
        {
            "open": df["o"],
            "high": df["h"],
            "low": df["l"],
            "close": df["c"],
            "adjClose": df["c"],  # aggregates are split-adjusted by default
            "volume": df["v"],
        }
    )
    return out[~out.index.duplicated(keep="last")]


def _fetch_daily_bars(
    client: MassiveClient,
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    # Paginate via next_url when history exceeds limit.
    path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    frames: List[pd.DataFrame] = []
    page = 0
    next_path: str | None = path
    next_params: dict | None = params
    while next_path and page < 20:
        payload = client.cached_get(
            next_path,
            cache_key=f"aggs_{ticker}_{start_date}_{end_date}_p{page}",
            params=next_params,
            ttl_days=1,
        )
        part = _aggs_to_df(payload)
        if not part.empty:
            frames.append(part)
        if not isinstance(payload, dict) or not payload.get("next_url"):
            break
        next_path = payload["next_url"]
        next_params = None
        page += 1
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def build_price_panel(
    client: MassiveClient,
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

    print(f">> {len(tickers)}개 종목 OHLCV 수집 (Massive aggregates)...")
    for i, t in enumerate(tickers):
        df = _fetch_daily_bars(client, t, start_date, end_date)
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
        if df.empty:
            continue

        prices_open[t] = df["open"]
        prices_close[t] = df["adjClose"]
        highs[t] = df["high"]
        lows[t] = df["low"]
        vols[t] = df["adjClose"] * df["volume"]

        if (i + 1) % 10 == 0 or (i + 1) == len(tickers):
            print(f"    ... prices {i+1}/{len(tickers)}", flush=True)

    bm_df = _fetch_daily_bars(client, benchmark, start_date, end_date)
    if bm_df.empty:
        raise RuntimeError(f"Benchmark prices unavailable for {benchmark} from Massive.")
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
    parser = argparse.ArgumentParser(description="Download OHLCV price panels (Massive)")
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
