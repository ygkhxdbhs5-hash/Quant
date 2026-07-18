"""Download NASDAQ universe + profiles via Massive reference APIs."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from downloader.download_universe_utils import load_config, make_client
from downloader.massive_client import MassiveClient

# ISO 10383 MIC for NASDAQ
NASDAQ_EXCHANGE = "XNAS"


def download_complete_nasdaq_universe(
    client: MassiveClient,
) -> Tuple[List[str], Dict[str, dict]]:
    """Fetch the full NASDAQ active + delisted ticker set from Massive.

    No config limits or external symbol-file restrictions are applied.
    """
    print(">> 나스닥 전체 상장 + 상장폐지 전수 명단 수집 (Massive)...")
    active_params = {
        "market": "stocks",       # verified: stocks
        "exchange": NASDAQ_EXCHANGE,
        "active": "true",         # verified: active listings only
        "limit": 1000,
        "sort": "ticker",
        "order": "asc",
    }
    print(f"    [active] request params={active_params} max_pages=500")
    active_rows = client.paginate(
        "/v3/reference/tickers",
        # bump cache prefix so stale limited pages are not reused
        cache_key_prefix="nasdaq_active_v500",
        params=active_params,
        ttl_days=1,
        max_pages=500,
    )
    print(f"    [active] raw API rows={len(active_rows)}")

    active_tickers = []
    n_skipped_type = 0
    for r in active_rows:
        if not isinstance(r, dict) or not r.get("ticker"):
            continue
        ttype = (r.get("type") or "").upper()
        # Prefer common stock / ADR; skip ETFs and funds at universe stage.
        if ttype in {"ETF", "ETV", "ETS", "FUND", "UNIT"}:
            n_skipped_type += 1
            continue
        active_tickers.append(r["ticker"])
    if not active_tickers:
        active_tickers = [r["ticker"] for r in active_rows if isinstance(r, dict) and r.get("ticker")]
        print("    [active] WARN: type filter left 0 names; using all raw tickers")
    print(
        f"    [active] kept={len(active_tickers)} "
        f"skipped_etf_fund={n_skipped_type}"
    )

    delisted_params = {
        "market": "stocks",       # verified: stocks
        "exchange": NASDAQ_EXCHANGE,
        "active": "false",
        "limit": 1000,
        "sort": "ticker",
        "order": "asc",
    }
    print(f"    [delisted] request params={delisted_params} max_pages=500")
    delisted_rows = client.paginate(
        "/v3/reference/tickers",
        cache_key_prefix="nasdaq_delisted_v500",
        params=delisted_params,
        ttl_days=3,
        max_pages=500,
    )
    print(f"    [delisted] raw API rows={len(delisted_rows)}")

    delisted_meta: Dict[str, dict] = {}
    for item in delisted_rows:
        t = item.get("ticker")
        if not t:
            continue
        delisted_utc = item.get("delisted_utc")
        delisting_date = None
        if delisted_utc:
            delisting_date = pd.to_datetime(delisted_utc).strftime("%Y-%m-%d")
        delisted_meta[t] = {"delistingDate": delisting_date}

    # Full universe: active ∪ delisted (never restricted by symbols.txt / universe_limit)
    all_tickers = sorted(set(active_tickers) | set(delisted_meta.keys()))

    print(
        f"    SUMMARY active={len(active_tickers)} delisted={len(delisted_meta)} "
        f"total={len(all_tickers)}"
    )
    if len(all_tickers) < 1000:
        print(
            "    WARNING: total < 1000 — pagination/API may still be truncated. "
            "Check Page N logs above."
        )
    return all_tickers, delisted_meta


def fetch_profile_meta(client: MassiveClient, tickers: List[str]) -> Dict[str, dict]:
    print(f">> {len(tickers)}개 종목 profile(섹터/산업/IPO일) 조회...")
    meta: Dict[str, dict] = {}
    for i, t in enumerate(tickers):
        payload = client.cached_get(
            f"/v3/reference/tickers/{t}",
            cache_key=f"ticker_overview_{t}",
            ttl_days=30,
        )
        results = payload.get("results") if isinstance(payload, dict) else None
        if isinstance(results, dict):
            list_date = results.get("list_date")
            sic = results.get("sic_description") or results.get("sic_code") or "Unknown"
            # Massive/Polygon overview does not always expose GICS industry;
            # sic_description is the closest stable industry label.
            meta[t] = {
                "sector": results.get("sic_description") or "Unknown",
                "industry": str(sic),
                "ipoDate": pd.to_datetime(list_date) if list_date else pd.NaT,
                "isEtf": (results.get("type") or "").upper() in {"ETF", "ETV", "ETS"},
            }
        else:
            meta[t] = {"sector": "Unknown", "industry": "Unknown", "ipoDate": pd.NaT, "isEtf": False}
        if (i + 1) % 50 == 0:
            print(f"    ...{i+1}/{len(tickers)}")
    return meta


def save_universe(
    metadata_dir: Path,
    all_tickers: List[str],
    delisted_meta: dict,
    profile_meta: dict,
    tickers: List[str],
) -> Path:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / "universe.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "all_tickers": all_tickers,
                "tickers": tickers,
                "delisted_meta": delisted_meta,
                "profile_meta": profile_meta,
            },
            f,
        )
    print(f"[metadata] wrote {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download NASDAQ universe + profiles (Massive)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    client = make_client(config)
    paths = config.get("paths", {})
    metadata_dir = Path(paths.get("metadata", "data/metadata"))

    all_tickers, delisted_meta = download_complete_nasdaq_universe(client)

    # Strictly enforce full universe: ignore universe_limit and symbols.txt entirely.
    # tickers must always equal all_tickers for profile fetch + save.
    tickers = all_tickers

    print(f">> Full NASDAQ universe enforced: tickers={len(tickers)} (== all_tickers)")
    profile_meta = fetch_profile_meta(client, tickers)
    save_universe(metadata_dir, all_tickers, delisted_meta, profile_meta, tickers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
