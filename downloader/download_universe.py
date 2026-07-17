"""Download universe metadata into data/metadata/universe.pkl.

Uses FMP stable endpoints. Falls back to data/metadata/symbols.txt when the
full exchange listing endpoint is unavailable on the current plan.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from downloader.fmp_client import FmpClient
from downloader.download_universe_utils import load_config, make_client, read_symbol_file


def download_complete_nasdaq_universe(client: FmpClient, symbol_file: Path) -> Tuple[List[str], Dict[str, dict]]:
    print(">> 나스닥 유니버스 명단 수집...")
    active_tickers: List[str] = []

    # Prefer stable constituent list when the plan allows it.
    active_res = client.cached_get(
        f"https://financialmodelingprep.com/stable/nasdaq-constituent?apikey={client.api_key}",
        "nasdaq_constituent_stable",
        ttl_days=1,
    )
    if isinstance(active_res, list) and active_res and isinstance(active_res[0], dict):
        active_tickers = [x["symbol"] for x in active_res if x.get("symbol")]

    used_symbol_file = False
    if not active_tickers:
        print("    (nasdaq-constituent unavailable — using symbols.txt fallback)")
        active_tickers = read_symbol_file(symbol_file)
        used_symbol_file = True

    delisted_meta: Dict[str, dict] = {}
    page = 0
    while True:
        d_res = client.cached_get(
            f"https://financialmodelingprep.com/stable/delisted-companies?page={page}&apikey={client.api_key}",
            f"delisted_page_stable_{page}",
            ttl_days=3,
        )
        if not d_res or not isinstance(d_res, list):
            break
        for item in d_res:
            t = item.get("symbol")
            exchange = (item.get("exchange") or "").upper()
            if t and ("NASDAQ" in exchange or exchange in {"", "NASDAQ"}):
                # Map stable field name -> engine expects delistingDate
                delisted_meta[t] = {"delistingDate": item.get("delistedDate") or item.get("delistingDate")}
        if len(d_res) < 100:
            break
        page += 1
        if page > 50:
            break

    if used_symbol_file:
        # Keep download scope to the explicit list; still retain delisted metadata
        # for symbols that appear in that list.
        all_tickers = list(active_tickers)
        delisted_meta = {k: v for k, v in delisted_meta.items() if k in set(active_tickers)}
    else:
        all_tickers = sorted(set(active_tickers) | set(delisted_meta.keys()))
    print(f"    active={len(active_tickers)} delisted={len(delisted_meta)} total={len(all_tickers)}")
    return all_tickers, delisted_meta


def fetch_profile_meta(client: FmpClient, tickers: List[str]) -> Dict[str, dict]:
    print(f">> {len(tickers)}개 종목 profile(섹터/산업/IPO일) 조회...")
    meta: Dict[str, dict] = {}
    for i, t in enumerate(tickers):
        res = client.cached_get(
            f"https://financialmodelingprep.com/stable/profile?symbol={t}&apikey={client.api_key}",
            f"profile_stable_{t}",
            ttl_days=30,
        )
        if res and isinstance(res, list) and res and isinstance(res[0], dict):
            p = res[0]
            meta[t] = {
                "sector": p.get("sector") or "Unknown",
                "industry": p.get("industry") or "Unknown",
                "ipoDate": pd.to_datetime(p.get("ipoDate")) if p.get("ipoDate") else pd.NaT,
                "isEtf": bool(p.get("isEtf", False)),
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
    parser = argparse.ArgumentParser(description="Download NASDAQ universe + profiles")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    client = make_client(config)
    paths = config.get("paths", {})
    metadata_dir = Path(paths.get("metadata", "data/metadata"))

    all_tickers, delisted_meta = download_complete_nasdaq_universe(
        client, metadata_dir / "symbols.txt"
    )
    limit = config.get("universe_limit")
    tickers = all_tickers if limit is None else all_tickers[: int(limit)]
    # Always include explicit symbol-file names first when limiting.
    file_syms = read_symbol_file(metadata_dir / "symbols.txt")
    if limit is not None and file_syms:
        tickers = file_syms[: int(limit)]

    profile_meta = fetch_profile_meta(client, tickers)
    save_universe(metadata_dir, all_tickers, delisted_meta, profile_meta, tickers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
