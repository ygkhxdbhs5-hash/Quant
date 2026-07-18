"""Download OHLCV panels via Massive aggregates into data/prices/panels.pkl."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from downloader.download_universe_utils import load_config, make_client
from downloader.massive_client import MassiveClient
from downloader.parallel import map_parallel


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
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch daily OHLCV; retry a few times on empty/rate-limited responses."""
    path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}

    for attempt in range(max(1, retries)):
        frames: List[pd.DataFrame] = []
        page = 0
        next_path: str | None = path
        next_params: dict | None = params
        while next_path and page < 20:
            payload = client.cached_get(
                next_path,
                cache_key=f"aggs_{ticker}_{start_date}_{end_date}_p{page}_v2",
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
        if frames:
            out = pd.concat(frames).sort_index()
            return out[~out.index.duplicated(keep="last")]
        print(f"    [prices] empty bars for {ticker} attempt={attempt+1}/{retries}", flush=True)
        time.sleep(1.0 + attempt)
    return pd.DataFrame()


def _store_bars(
    df: pd.DataFrame,
    ticker: str,
    prices_open: dict,
    prices_close: dict,
    highs: dict,
    lows: dict,
    vols: dict,
) -> None:
    prices_open[ticker] = df["open"]
    prices_close[ticker] = df["adjClose"]
    highs[ticker] = df["high"]
    lows[ticker] = df["low"]
    vols[ticker] = df["adjClose"] * df["volume"]


def build_price_panel(
    client: MassiveClient,
    tickers: List[str],
    delisted_meta: dict,
    profile_meta: dict,
    start_date: str,
    end_date: str,
    benchmark: str,
    silent_delist_gap_days: int,
    workers: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.Timestamp]]:
    prices_open, prices_close, highs, lows, vols = {}, {}, {}, {}, {}
    silent_delist_flags: Dict[str, pd.Timestamp] = {}
    end_ts = pd.to_datetime(end_date)
    benchmark = (benchmark or "QQQ").upper()
    fallback_benchmarks = [benchmark] + [b for b in ("QQQ", "SPY", "IVV") if b != benchmark]

    # Fetch benchmark FIRST so regime data exists even if later calls are rate-limited.
    print(f">> Benchmark OHLCV first (try {fallback_benchmarks})...")
    bm_used = None
    for bm in fallback_benchmarks:
        bm_df = _fetch_daily_bars(client, bm, start_date, end_date, retries=4)
        if bm_df.empty:
            print(f"    [prices] benchmark candidate {bm} unavailable", flush=True)
            continue
        _store_bars(bm_df, bm, prices_open, prices_close, highs, lows, vols)
        bm_used = bm
        print(
            f"    [prices] benchmark={bm} bars={len(bm_df)} "
            f"{bm_df.index.min().date()} → {bm_df.index.max().date()}",
            flush=True,
        )
        break
    if bm_used is None:
        raise RuntimeError(
            f"Benchmark prices unavailable for any of {fallback_benchmarks} from Massive. "
            "Check API key entitlements / rate limits, then retry."
        )
    if bm_used != benchmark:
        print(
            f"    [prices] WARN: requested benchmark={benchmark} unavailable; "
            f"using {bm_used} instead (update config.benchmark if needed)",
            flush=True,
        )
        benchmark = bm_used

    # Avoid double-fetching the benchmark inside the main loop.
    tickers_to_pull = [t for t in tickers if str(t).upper() != benchmark]

    print(
        f">> {len(tickers_to_pull)}개 종목 OHLCV 수집 "
        f"(Massive aggregates, workers={workers})..."
    )
    fetched = map_parallel(
        tickers_to_pull,
        lambda t: (t, _fetch_daily_bars(client, t, start_date, end_date)),
        workers=workers,
        progress_every=25,
        label="prices",
    )

    for t, df in fetched:
        if df is None or df.empty:
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

        _store_bars(df, t, prices_open, prices_close, highs, lows, vols)

    if benchmark not in prices_close:
        raise RuntimeError(f"Benchmark prices missing after download for {benchmark}.")

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
        workers=max(1, int(config.get("download_workers", 8))),
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
