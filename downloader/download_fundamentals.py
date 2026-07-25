"""Download PIT fundamentals via Massive financials into data/fundamentals/.

Output schema matches StandaloneEngine expectations:
op_margin, roic, gross_profitability, revenue, operating_income, net_income,
op_cf, capex, total_debt, total_equity, total_assets, cash_eq, debt_to_equity,
revenue_growth_yoy, diluted_shares_outstanding, basic_shares_outstanding —
indexed by filing_date (PIT accepted proxy).
"""

from __future__ import annotations

import argparse
import pickle
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from downloader.download_universe_utils import load_config, make_client
from downloader.massive_client import MassiveClient
from downloader.parallel import map_parallel


def _statement_df(
    client: MassiveClient,
    endpoint: str,
    ticker: str,
    limit: int,
    verbose_paginate: bool = False,
) -> Tuple[pd.DataFrame, str, int]:
    """Fetch one statement type.

    Returns
    -------
    (df, skip_reason, raw_row_count)
        skip_reason is "" on success. raw_row_count is API results length before cleaning.
    """
    # Massive allows sort on: cik, period_end, fiscal_year, fiscal_quarter, timeframe
    # (filing_date is NOT a valid sort key — that was returning HTTP 400 and empty pulls).
    rows = client.paginate(
        endpoint,
        cache_key_prefix=f"{endpoint.strip('/').replace('/', '_')}_{ticker}_v3",
        params={
            "tickers.any_of": ticker,
            "timeframe": "quarterly",
            "limit": limit,
            "sort": "period_end.asc",
        },
        ttl_days=3,
        max_pages=20,
        verbose=verbose_paginate,
    )
    raw_n = len(rows) if isinstance(rows, list) else 0
    if not rows:
        return pd.DataFrame(), f"{endpoint} empty/unauthorized", raw_n

    df = pd.DataFrame(rows)
    if "filing_date" not in df.columns or "period_end" not in df.columns:
        return (
            pd.DataFrame(),
            f"{endpoint} missing filing_date/period_end cols={list(df.columns)[:12]}",
            raw_n,
        )

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["filing_date"]).sort_values(["period_end", "filing_date"])
    if df.empty:
        return pd.DataFrame(), f"{endpoint} all filing_date NaT (raw={before})", raw_n

    if {"fiscal_year", "fiscal_quarter"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["fiscal_year", "fiscal_quarter"], keep="first")
    else:
        df = df.drop_duplicates(subset=["period_end"], keep="first")
    return df, "", raw_n


def fetch_pit_fundamentals(
    client: MassiveClient,
    ticker: str,
    flat_tax_rate: float,
    statement_limit: int = 100,
    debug: bool = True,
) -> Tuple[Optional[pd.DataFrame], str, Dict[str, Any]]:
    """Build PIT factor history for one ticker.

    Returns
    -------
    (dataframe_or_none, reason, stats)
        stats always includes raw API row counts for income/cashflow/balance.
    """
    stats: Dict[str, Any] = {
        "ticker": ticker,
        "inc_raw": 0,
        "cfs_raw": 0,
        "bal_raw": 0,
        "pit_rows": 0,
        "stored": False,
    }

    try:
        inc, inc_reason, inc_raw = _statement_df(
            client, "/stocks/financials/v1/income-statements", ticker, statement_limit
        )
        cfs, cfs_reason, cfs_raw = _statement_df(
            client, "/stocks/financials/v1/cash-flow-statements", ticker, statement_limit
        )
        bal, bal_reason, bal_raw = _statement_df(
            client, "/stocks/financials/v1/balance-sheets", ticker, statement_limit
        )
        stats.update({"inc_raw": inc_raw, "cfs_raw": cfs_raw, "bal_raw": bal_raw})

        if inc.empty or cfs.empty or bal.empty:
            parts = [r for r in (inc_reason, cfs_reason, bal_reason) if r]
            reason = "; ".join(parts) if parts else "one or more statements empty after parse"
            if debug:
                print(
                    f"    [SKIP] {ticker}: raw inc={inc_raw} cfs={cfs_raw} bal={bal_raw} "
                    f"| cleaned inc={len(inc)} cfs={len(cfs)} bal={len(bal)} | {reason}"
                )
            return None, reason, stats

        inc_small = pd.DataFrame(
            {
                "period_end": inc["period_end"],
                "filing_date": inc["filing_date"],
                "revenue": inc["revenue"] if "revenue" in inc.columns else np.nan,
                "operating_income": inc["operating_income"] if "operating_income" in inc.columns else np.nan,
                "gross_profit": inc["gross_profit"] if "gross_profit" in inc.columns else np.nan,
                "net_income": (
                    inc["net_income_loss"]
                    if "net_income_loss" in inc.columns
                    else (inc["net_income"] if "net_income" in inc.columns else np.nan)
                ),
                "diluted_shares_outstanding": (
                    inc["diluted_shares_outstanding"]
                    if "diluted_shares_outstanding" in inc.columns
                    else np.nan
                ),
                "basic_shares_outstanding": (
                    inc["basic_shares_outstanding"]
                    if "basic_shares_outstanding" in inc.columns
                    else np.nan
                ),
                "fiscal_year": inc["fiscal_year"] if "fiscal_year" in inc.columns else np.nan,
                "fiscal_quarter": inc["fiscal_quarter"] if "fiscal_quarter" in inc.columns else np.nan,
            }
        )
        cfs_small = pd.DataFrame(
            {
                "period_end": cfs["period_end"],
                "op_cf": cfs["net_cash_from_operating_activities"]
                if "net_cash_from_operating_activities" in cfs.columns
                else np.nan,
                "capex": cfs["purchase_of_property_plant_and_equipment"]
                if "purchase_of_property_plant_and_equipment" in cfs.columns
                else np.nan,
                "fiscal_year": cfs["fiscal_year"] if "fiscal_year" in cfs.columns else np.nan,
                "fiscal_quarter": cfs["fiscal_quarter"] if "fiscal_quarter" in cfs.columns else np.nan,
            }
        )
        cfs_small["capex"] = pd.to_numeric(cfs_small["capex"], errors="coerce").abs()

        debt_current = (
            pd.to_numeric(bal["debt_current"], errors="coerce")
            if "debt_current" in bal.columns
            else pd.Series(0.0, index=bal.index)
        )
        debt_lt = (
            pd.to_numeric(bal["long_term_debt_and_capital_lease_obligations"], errors="coerce")
            if "long_term_debt_and_capital_lease_obligations" in bal.columns
            else pd.Series(0.0, index=bal.index)
        )
        total_debt = debt_current.fillna(0) + debt_lt.fillna(0)

        bal_small = pd.DataFrame(
            {
                "period_end": bal["period_end"],
                "total_assets": bal["total_assets"] if "total_assets" in bal.columns else np.nan,
                "total_debt": total_debt.values,
                "total_equity": bal["total_equity"] if "total_equity" in bal.columns else np.nan,
                "cash_eq": bal["cash_and_equivalents"] if "cash_and_equivalents" in bal.columns else np.nan,
                "fiscal_year": bal["fiscal_year"] if "fiscal_year" in bal.columns else np.nan,
                "fiscal_quarter": bal["fiscal_quarter"] if "fiscal_quarter" in bal.columns else np.nan,
            }
        )

        merge_keys = ["fiscal_year", "fiscal_quarter"]
        if inc_small[merge_keys].isna().any().any():
            merged = inc_small.merge(
                cfs_small.drop(columns=["fiscal_year", "fiscal_quarter"], errors="ignore"),
                on="period_end",
                how="outer",
            ).merge(
                bal_small.drop(columns=["fiscal_year", "fiscal_quarter"], errors="ignore"),
                on="period_end",
                how="outer",
            )
        else:
            merged = inc_small.merge(
                cfs_small.drop(columns=["period_end"], errors="ignore"),
                on=merge_keys,
                how="outer",
            ).merge(
                bal_small.drop(columns=["period_end"], errors="ignore"),
                on=merge_keys,
                how="outer",
            )

        merged = merged.dropna(subset=["filing_date"]).sort_values("filing_date")
        if merged.empty:
            reason = "merge produced no rows with filing_date"
            if debug:
                print(
                    f"    [SKIP] {ticker}: raw inc={inc_raw} cfs={cfs_raw} bal={bal_raw} "
                    f"| merged empty after filing_date dropna"
                )
            return None, reason, stats

        merged["op_margin"] = merged["operating_income"] / merged["revenue"].replace(0, np.nan)
        merged["gross_profitability"] = merged["gross_profit"] / merged["total_assets"].replace(0, np.nan)
        invested_capital = merged["total_debt"] + merged["total_equity"] - merged["cash_eq"]
        invested_capital = invested_capital.where(invested_capital > 0, merged["total_assets"])
        merged["roic"] = (merged["operating_income"] * (1 - flat_tax_rate)) / invested_capital.replace(0, np.nan)
        # Debt-to-Equity for universe debt filter (total_debt / total_equity)
        merged["debt_to_equity"] = merged["total_debt"] / merged["total_equity"].replace(0, np.nan)

        out = merged.set_index("filing_date").sort_index()
        # YoY revenue growth for quarterly statements: compare to 4 periods earlier
        rev = pd.to_numeric(out["revenue"], errors="coerce")
        prior_q = rev.shift(4)
        out["revenue_growth_yoy"] = (rev - prior_q) / prior_q.abs().replace(0, np.nan)
        keep_cols = [
            "op_margin",
            "roic",
            "gross_profitability",
            "revenue",
            "operating_income",
            "net_income",
            "op_cf",
            "capex",
            "total_debt",
            "total_equity",
            "total_assets",
            "cash_eq",
            "debt_to_equity",
            "revenue_growth_yoy",
            "diluted_shares_outstanding",
            "basic_shares_outstanding",
        ]
        out = out[[c for c in keep_cols if c in out.columns]]
        before_drop = len(out)
        out = out.dropna(subset=["op_margin", "roic", "gross_profitability"], how="all")
        out.index.name = "acceptedDate"
        stats["pit_rows"] = int(len(out))

        if out.empty:
            reason = f"all factor rows NaN after dropna (had {before_drop} merged rows)"
            if debug:
                print(
                    f"    [SKIP] {ticker}: raw inc={inc_raw} cfs={cfs_raw} bal={bal_raw} "
                    f"| merged={before_drop} -> pit=0 | {reason}"
                )
            return None, reason, stats

        stats["stored"] = True
        if debug:
            print(
                f"    [STORE] {ticker}: raw inc={inc_raw} cfs={cfs_raw} bal={bal_raw} "
                f"| pit_rows={len(out)} ({out.index.min().date()} → {out.index.max().date()}) "
                f"| APPENDED to fundamental_history"
            )
        return out, "ok", stats

    except Exception as exc:
        reason = f"exception: {type(exc).__name__}: {exc}"
        if debug:
            print(f"    [ERROR] {ticker}: {reason}")
            traceback.print_exc()
        return None, reason, stats


def _resolve_tickers(universe: dict, close_m: pd.DataFrame, benchmark: str) -> list[str]:
    """Prefer universe tickers that also have prices; never silently use an empty list."""
    uni = [str(t).upper() for t in (universe.get("tickers") or []) if t]
    priced = [str(c).upper() for c in close_m.columns if str(c).upper() != benchmark.upper()]
    if uni:
        priced_set = set(priced)
        # Keep universe order; include only names present in the price panel when possible.
        tickers = [t for t in uni if t in priced_set and t != benchmark.upper()]
        if not tickers:
            # Prices missing for universe — still try universe list so logs show API attempts.
            tickers = [t for t in uni if t != benchmark.upper()]
            print(
                f"[fundamentals] WARN: none of universe tickers found in price panel; "
                f"falling back to universe list ({len(tickers)} names)"
            )
        return tickers
    if priced:
        print(f"[fundamentals] WARN: universe tickers empty; using price-panel columns ({len(priced)})")
        return priced
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download PIT fundamentals (Massive)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-ticker logs (default is verbose)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose per-ticker logs (default; kept for compatibility)",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    client = make_client(config)
    paths = config.get("paths", {})
    benchmark = str(config.get("benchmark", "QQQ")).upper()
    flat_tax_rate = float(config.get("flat_tax_rate", 0.21))
    statement_limit = int(config.get("statement_limit", 100))
    debug = not bool(args.quiet)

    universe_path = Path(paths.get("metadata", "data/metadata")) / "universe.pkl"
    panels_path = Path(paths.get("prices", "data/prices")) / "panels.pkl"
    if not universe_path.exists():
        raise SystemExit(f"Missing {universe_path}. Run download_universe first.")
    if not panels_path.exists():
        raise SystemExit(f"Missing {panels_path}. Run download_prices first.")

    with open(universe_path, "rb") as f:
        universe = pickle.load(f)
    with open(panels_path, "rb") as f:
        panels = pickle.load(f)

    close_m = panels["close_m"]
    symbols = _resolve_tickers(universe, close_m, benchmark)
    print(">> PIT 재무 데이터 수집 (Massive financials)...")
    print(f"[fundamentals] benchmark={benchmark} tickers_to_process={len(symbols)}")
    if not symbols:
        print(
            "[fundamentals] ERROR: no tickers to process. "
            "Check data/metadata/universe.pkl and data/prices/panels.pkl."
        )
        out_dir = Path(paths.get("fundamentals", "data/fundamentals"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pit_history.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({}, f)
        print(f"[fundamentals] wrote {out_path} (0 symbols) — empty because ticker list was empty")
        return 1

    if debug:
        print(f"[fundamentals] ticker list: {symbols[:20]}{'...' if len(symbols) > 20 else ''}")

    fundamental_history: Dict[str, pd.DataFrame] = {}
    skip_reasons: Counter[str] = Counter()
    workers = max(1, int(config.get("download_workers", 8)))
    # Per-ticker debug spam is hard to read under many workers; keep summary progress.
    per_ticker_debug = debug and workers == 1
    print(f"[fundamentals] parallel workers={workers}")

    results = map_parallel(
        symbols,
        lambda t: fetch_pit_fundamentals(
            client,
            t,
            flat_tax_rate=flat_tax_rate,
            statement_limit=statement_limit,
            debug=per_ticker_debug,
        ),
        workers=workers,
        progress_every=25,
        label="fundamentals",
    )

    for t, (pit, reason, stats) in zip(symbols, results):
        # Explicit integration: only store real DataFrames with rows.
        if isinstance(pit, pd.DataFrame) and not pit.empty:
            fundamental_history[t] = pit
            skip_reasons["ok"] += 1
            if debug and workers == 1:
                print(
                    f"    [DICT] fundamental_history['{t}'] = {len(pit)} rows "
                    f"(dict size now {len(fundamental_history)})"
                )
        else:
            bucket = reason.split(";")[0].strip() if reason else "unknown"
            if "empty/unauthorized" in bucket:
                bucket = "statement empty/unauthorized"
            skip_reasons[bucket] += 1
            if debug and workers == 1:
                print(
                    f"    [DICT] NOT stored {t}: reason={reason} "
                    f"raw=(inc={stats.get('inc_raw')}, cfs={stats.get('cfs_raw')}, bal={stats.get('bal_raw')})"
                )

    out_dir = Path(paths.get("fundamentals", "data/fundamentals"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pit_history.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(fundamental_history, f)

    # Verify round-trip so "wrote 0" cannot lie about a non-empty dict.
    with open(out_path, "rb") as f:
        verify = pickle.load(f)
    n_saved = len(verify) if isinstance(verify, dict) else -1

    print(
        f"[fundamentals] wrote {out_path} "
        f"({len(fundamental_history)} symbols in memory / {n_saved} symbols on disk / {len(symbols)} tickers)"
    )
    print("[fundamentals] reason counts:")
    for reason, count in skip_reasons.most_common():
        print(f"    {count:5d}  {reason}")

    if n_saved == 0:
        print(
            "[fundamentals] ERROR: saved 0 symbols despite processing tickers. "
            "See per-ticker [SKIP]/[ERROR] lines above for raw API counts and filter reasons."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
