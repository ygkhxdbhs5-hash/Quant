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


def _fetch_daily_bars_fresh(
    client: MassiveClient,
    ticker: str,
    start_date: str,
    end_date: str,
    retries: int = 2,
) -> pd.DataFrame:
    """Like ``_fetch_daily_bars`` but bypasses disk cache (for incremental/live catch-up)."""
    path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}

    for attempt in range(max(1, retries)):
        frames: List[pd.DataFrame] = []
        page = 0
        next_path: str | None = path
        next_params: dict | None = params
        while next_path and page < 20:
            payload = client.get_json(next_path, params=next_params)
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
        time.sleep(0.5 + attempt)
    return pd.DataFrame()


def _upsert_series(frame: pd.DataFrame, ticker: str, series: pd.Series) -> pd.DataFrame:
    if ticker not in frame.columns:
        frame[ticker] = pd.NA
    for ts, val in series.items():
        frame.loc[pd.Timestamp(ts).normalize(), ticker] = val
    return frame


def incremental_update_panels(
    client: MassiveClient,
    panels: dict,
    tickers: List[str],
    *,
    end_date: str,
    lookback_calendar_days: int = 7,
    workers: int = 8,
    benchmark: str = "QQQ",
) -> dict:
    """Fetch recent daily bars and upsert into existing panels (no full rebuild)."""
    close_m = panels["close_m"].copy()
    open_m = panels["open_m"].copy()
    high_m = panels["high_m"].copy()
    low_m = panels["low_m"].copy()
    dvol_m = panels["dvol_m"].copy()

    if close_m.empty:
        raise RuntimeError("Existing panels are empty; run a full price download first.")

    last_dt = pd.Timestamp(close_m.dropna(how="all").index.max()).normalize()
    start_dt = last_dt - pd.Timedelta(days=max(1, int(lookback_calendar_days)))
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = pd.Timestamp(end_date).strftime("%Y-%m-%d")

    symbols = list(dict.fromkeys([*(tickers or []), *(close_m.columns.astype(str)), benchmark]))
    symbols = [s for s in symbols if s]
    print(
        f">> Incremental price refresh {start_date} → {end_date} "
        f"for {len(symbols)} symbols (workers={workers})..."
    )

    fetched = map_parallel(
        symbols,
        lambda t: (t, _fetch_daily_bars_fresh(client, t, start_date, end_date)),
        workers=workers,
        progress_every=25,
        label="prices-incr",
    )

    updated = 0
    for t, df in fetched:
        if df is None or df.empty:
            continue
        open_m = _upsert_series(open_m, t, df["open"])
        close_m = _upsert_series(close_m, t, df["adjClose"])
        high_m = _upsert_series(high_m, t, df["high"])
        low_m = _upsert_series(low_m, t, df["low"])
        dvol_m = _upsert_series(dvol_m, t, df["adjClose"] * df["volume"])
        updated += 1

    # Align matrices on a shared sorted calendar
    idx = close_m.index.union(open_m.index).union(high_m.index).union(low_m.index).union(dvol_m.index)
    idx = idx.sort_values()
    open_m = open_m.reindex(idx).sort_index()
    close_m = close_m.reindex(idx).sort_index()
    high_m = high_m.reindex(idx).sort_index()
    low_m = low_m.reindex(idx).sort_index()
    dvol_m = dvol_m.reindex(idx).sort_index()

    print(f" -> incremental upsert complete: {updated}/{len(symbols)} symbols had bars")
    out = dict(panels)
    out.update(
        {
            "open_m": open_m,
            "close_m": close_m,
            "high_m": high_m,
            "low_m": low_m,
            "dvol_m": dvol_m,
            "incremental_updated_at": pd.Timestamp.now(tz="UTC"),
        }
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download OHLCV price panels (Massive)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--mode",
        choices=("full", "incremental"),
        default="full",
        help="full = rebuild panels from start_date; incremental = upsert recent bars only",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    client = make_client(config)
    paths = config.get("paths", {})
    prices_dir = Path(paths.get("prices", "data/prices"))
    panels_path = prices_dir / "panels.pkl"

    universe_path = Path(paths.get("metadata", "data/metadata")) / "universe.pkl"
    if not universe_path.exists():
        raise SystemExit(f"Missing {universe_path}. Run download_universe first.")
    with open(universe_path, "rb") as f:
        universe = pickle.load(f)

    workers = max(1, int(config.get("download_workers", 8)))
    rt_cfg = config.get("realtime") or {}
    lookback = int(rt_cfg.get("incremental_lookback_days", 7))

    if args.mode == "incremental":
        if not panels_path.exists():
            raise SystemExit(
                f"Missing {panels_path}. Run a full download first "
                "(python -m downloader.download_prices --mode full)."
            )
        with open(panels_path, "rb") as f:
            panels = pickle.load(f)
        panels = incremental_update_panels(
            client=client,
            panels=panels,
            tickers=universe["tickers"],
            end_date=config["end_date"],
            lookback_calendar_days=lookback,
            workers=workers,
            benchmark=config.get("benchmark", "QQQ"),
        )
        save_panels(prices_dir, panels)
        return 0

    open_m, close_m, high_m, low_m, dvol_m, silent = build_price_panel(
        client=client,
        tickers=universe["tickers"],
        delisted_meta=universe["delisted_meta"],
        profile_meta=universe["profile_meta"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        benchmark=config.get("benchmark", "QQQ"),
        silent_delist_gap_days=int(config.get("silent_delist_gap_days", 10)),
        workers=workers,
    )
    save_panels(
        prices_dir,
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
