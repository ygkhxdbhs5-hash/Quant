"""Download PIT fundamentals via Massive financials into data/fundamentals/.

Output schema matches StandaloneEngine expectations:
op_margin, roic, gross_profitability, revenue, operating_income, op_cf, capex,
total_debt, cash_eq — indexed by filing_date (PIT accepted proxy).
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from downloader.download_universe_utils import load_config, make_client
from downloader.massive_client import MassiveClient


def _statement_df(
    client: MassiveClient,
    endpoint: str,
    ticker: str,
    limit: int,
) -> Tuple[pd.DataFrame, str]:
    """Fetch one statement type. Returns (df, skip_reason). skip_reason empty on success."""
    # Massive allows sort on: cik, period_end, fiscal_year, fiscal_quarter, timeframe
    # (filing_date is NOT a valid sort key — that was returning HTTP 400 and empty pulls).
    rows = client.paginate(
        endpoint,
        cache_key_prefix=f"{endpoint.strip('/').replace('/', '_')}_{ticker}_v2",
        params={
            "tickers.any_of": ticker,
            "timeframe": "quarterly",
            "limit": limit,
            "sort": "period_end.asc",
        },
        ttl_days=3,
        max_pages=20,
    )
    if not rows:
        return pd.DataFrame(), f"{endpoint} empty/unauthorized"

    df = pd.DataFrame(rows)
    if "filing_date" not in df.columns or "period_end" not in df.columns:
        return pd.DataFrame(), f"{endpoint} missing filing_date/period_end cols={list(df.columns)[:12]}"

    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df = df.dropna(subset=["filing_date"]).sort_values(["period_end", "filing_date"])
    # Keep earliest filing for each fiscal period (PIT-friendly).
    if {"fiscal_year", "fiscal_quarter"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["fiscal_year", "fiscal_quarter"], keep="first")
    else:
        df = df.drop_duplicates(subset=["period_end"], keep="first")
    return df, ""


def fetch_pit_fundamentals(
    client: MassiveClient,
    ticker: str,
    flat_tax_rate: float,
    statement_limit: int = 100,
    debug: bool = False,
) -> Tuple[Optional[pd.DataFrame], str]:
    """Build PIT factor history for one ticker.

    Returns
    -------
    (dataframe_or_none, reason)
        reason is "ok" on success, otherwise a short skip explanation.
    """
    inc, inc_reason = _statement_df(client, "/stocks/financials/v1/income-statements", ticker, statement_limit)
    cfs, cfs_reason = _statement_df(client, "/stocks/financials/v1/cash-flow-statements", ticker, statement_limit)
    bal, bal_reason = _statement_df(client, "/stocks/financials/v1/balance-sheets", ticker, statement_limit)

    if inc.empty or cfs.empty or bal.empty:
        parts = [r for r in (inc_reason, cfs_reason, bal_reason) if r]
        reason = "; ".join(parts) if parts else "one or more statements empty"
        if debug:
            print(f"    [skip] {ticker}: {reason} (inc={len(inc)} cfs={len(cfs)} bal={len(bal)})")
        return None, reason

    inc_small = pd.DataFrame(
        {
            "period_end": inc["period_end"],
            "filing_date": inc["filing_date"],
            "revenue": inc["revenue"] if "revenue" in inc.columns else np.nan,
            "operating_income": inc["operating_income"] if "operating_income" in inc.columns else np.nan,
            "gross_profit": inc["gross_profit"] if "gross_profit" in inc.columns else np.nan,
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
        # Fall back to period_end merge when fiscal keys are missing.
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
            print(f"    [skip] {ticker}: {reason}")
        return None, reason

    merged["op_margin"] = merged["operating_income"] / merged["revenue"].replace(0, np.nan)
    merged["gross_profitability"] = merged["gross_profit"] / merged["total_assets"].replace(0, np.nan)
    invested_capital = merged["total_debt"] + merged["total_equity"] - merged["cash_eq"]
    invested_capital = invested_capital.where(invested_capital > 0, merged["total_assets"])
    merged["roic"] = (merged["operating_income"] * (1 - flat_tax_rate)) / invested_capital.replace(0, np.nan)

    out = merged.set_index("filing_date").sort_index()
    out = out[
        [
            "op_margin",
            "roic",
            "gross_profitability",
            "revenue",
            "operating_income",
            "op_cf",
            "capex",
            "total_debt",
            "cash_eq",
        ]
    ]
    out = out.dropna(subset=["op_margin", "roic", "gross_profitability"], how="all")
    out.index.name = "acceptedDate"
    if out.empty:
        reason = "all factor rows NaN after dropna"
        if debug:
            print(f"    [skip] {ticker}: {reason}")
        return None, reason

    if debug:
        print(f"    [ok] {ticker}: {len(out)} PIT rows ({out.index.min().date()} → {out.index.max().date()})")
    return out, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download PIT fundamentals (Massive)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--debug", action="store_true", help="Print per-ticker skip/ok reasons")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    client = make_client(config)
    paths = config.get("paths", {})
    benchmark = config.get("benchmark", "QQQ")
    flat_tax_rate = float(config.get("flat_tax_rate", 0.21))
    statement_limit = int(config.get("statement_limit", 100))
    debug = bool(args.debug or config.get("debug_fundamentals", False))

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
    print(">> PIT 재무 데이터 수집 (Massive financials)...")
    fundamental_history = {}
    skip_reasons: Counter[str] = Counter()
    symbols = [c for c in close_m.columns if c != benchmark]
    for i, t in enumerate(symbols):
        pit, reason = fetch_pit_fundamentals(
            client,
            t,
            flat_tax_rate=flat_tax_rate,
            statement_limit=statement_limit,
            debug=debug,
        )
        if pit is not None:
            fundamental_history[t] = pit
            skip_reasons["ok"] += 1
        else:
            # Collapse noisy endpoint suffixes into coarser buckets for the summary.
            bucket = reason.split(";")[0].strip() if reason else "unknown"
            if "empty/unauthorized" in bucket:
                bucket = "statement empty/unauthorized"
            skip_reasons[bucket] += 1
        if (i + 1) % 50 == 0:
            print(f"    ...{i+1}/{len(symbols)} 처리 (ok={len(fundamental_history)})")

    out_dir = Path(paths.get("fundamentals", "data/fundamentals"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pit_history.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(fundamental_history, f)

    print(f"[fundamentals] wrote {out_path} ({len(fundamental_history)} symbols / {len(symbols)} tickers)")
    print("[fundamentals] reason counts:")
    for reason, count in skip_reasons.most_common():
        print(f"    {count:5d}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
