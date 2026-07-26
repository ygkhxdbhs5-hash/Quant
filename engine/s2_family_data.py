"""Download/cache dividends + short-interest panels for S2 family search."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from downloader.download_universe_utils import load_config, make_client
from downloader.parallel import map_parallel

ROOT = Path(__file__).resolve().parents[1]
# Under cache/ (gitignored) so downloads do not dirty the worktree mid-run.
CACHE = ROOT / "cache" / "s2_family"
SI_LAG_DAYS = 14  # FINRA publication lag proxy after settlement_date


def ensure_cache_dir() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE


def load_or_build_dividends(
    tickers: Iterable[str],
    *,
    force: bool = False,
    workers: int = 8,
) -> pd.DataFrame:
    """Return DataFrame columns: ticker, ex_dividend_date, declaration_date."""
    ensure_cache_dir()
    path = CACHE / "dividends.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    client = make_client(cfg)
    tickers = [str(t) for t in tickers]

    def _one(t: str):
        rows = client.paginate(
            "/v3/reference/dividends",
            cache_key_prefix=f"div_{t}",
            params={"ticker": t, "limit": 100, "order": "asc"},
            ttl_days=7,
            max_pages=5,
            verbose=False,
        )
        out = []
        for r in rows or []:
            ex = r.get("ex_dividend_date")
            if not ex:
                continue
            out.append(
                {
                    "ticker": t,
                    "ex_dividend_date": pd.Timestamp(ex),
                    "declaration_date": pd.Timestamp(r["declaration_date"])
                    if r.get("declaration_date")
                    else pd.NaT,
                    "cash_amount": r.get("cash_amount"),
                }
            )
        return out

    print(f"[S2 DATA] downloading dividends for {len(tickers)} tickers...")
    chunks = map_parallel(tickers, _one, workers=workers, label="dividends")
    flat: List[dict] = []
    for ch in chunks:
        if ch:
            flat.extend(ch)
    df = pd.DataFrame(flat)
    if not df.empty:
        df = df.dropna(subset=["ex_dividend_date"]).sort_values(
            ["ticker", "ex_dividend_date"]
        )
    df.to_parquet(path, index=False)
    print(f"[S2 DATA] dividends rows={len(df)} -> {path}")
    return df


def load_or_build_short_interest_panel(
    tickers: Iterable[str],
    trading_index: pd.DatetimeIndex,
    *,
    force: bool = False,
    workers: int = 8,
    lag_days: int = SI_LAG_DAYS,
) -> pd.DataFrame:
    """Wide panel of days_to_cover, indexed by trading days, PIT-lagged."""
    ensure_cache_dir()
    path = CACHE / f"short_interest_dtc_lag{lag_days}.pkl"
    if path.exists() and not force:
        with open(path, "rb") as f:
            return pickle.load(f)

    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    client = make_client(cfg)
    tickers = [str(t) for t in tickers]

    def _one(t: str):
        rows = client.paginate(
            "/stocks/v1/short-interest",
            cache_key_prefix=f"si_{t}",
            params={"ticker": t, "limit": 100, "sort": "settlement_date.asc"},
            ttl_days=7,
            max_pages=10,
            verbose=False,
        )
        if not rows:
            return t, None
        df = pd.DataFrame(rows)
        df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
        df["available_date"] = df["settlement_date"] + pd.Timedelta(days=int(lag_days))
        df["days_to_cover"] = pd.to_numeric(df["days_to_cover"], errors="coerce")
        df = df.dropna(subset=["available_date", "days_to_cover"]).sort_values(
            "available_date"
        )
        return t, df[["available_date", "days_to_cover"]]

    print(f"[S2 DATA] downloading short interest for {len(tickers)} tickers...")
    results = map_parallel(tickers, _one, workers=workers, label="short_interest")
    series_map = {}
    for t, df in results:
        if df is None or df.empty:
            continue
        s = df.set_index("available_date")["days_to_cover"]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        # reindex to trading days with ffill
        s2 = s.reindex(trading_index, method="ffill")
        series_map[t] = s2

    panel = pd.DataFrame(series_map, index=trading_index)
    with open(path, "wb") as f:
        pickle.dump(panel, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[S2 DATA] SI panel shape={panel.shape} -> {path}")
    return panel
