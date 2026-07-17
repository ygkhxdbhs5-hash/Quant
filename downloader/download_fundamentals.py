"""Download PIT fundamentals into data/fundamentals/pit_history.pkl.

Computes the same PIT factor schema used by StandaloneEngine.build_factors:
op_margin, roic, gross_profitability, revenue, operating_income, op_cf, capex,
total_debt, cash_eq — indexed by acceptedDate.

Uses FMP stable statement endpoints. Prefers as-reported when available;
falls back to standard quarterly statements (same output schema).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from downloader.download_universe_utils import load_config, make_client
from downloader.fmp_client import FmpClient


def _pick(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(np.nan, index=df.index)


def _first_accepted_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "acceptedDate" not in df.columns:
        return df
    df = df.dropna(subset=["acceptedDate"]).copy()
    key_cols = [c for c in ["fiscalYear", "period"] if c in df.columns]
    df = df.sort_values("acceptedDate")
    if not key_cols:
        return df.drop_duplicates(subset=["date"], keep="first")
    return df.drop_duplicates(subset=key_cols, keep="first")


def _flatten_as_reported(rows: list) -> pd.DataFrame:
    """Stable as-reported nests fields under ``data`` — flatten for _pick()."""
    flat = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = {k: v for k, v in row.items() if k != "data"}
        data = row.get("data") or {}
        if isinstance(data, dict):
            item.update(data)
        flat.append(item)
    return pd.DataFrame(flat)


def _pit_from_frames(inc_df, cfs_df, bal_df, flat_tax_rate: float) -> Optional[pd.DataFrame]:
    for df in (inc_df, cfs_df, bal_df):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    revenue = _pick(inc_df, [
        "revenuefromcontractwithcustomerexcludingassessedtax", "revenues", "salesrevenuenet", "revenue",
    ])
    cogs = _pick(inc_df, ["costofgoodsandservicessold", "costofrevenue", "costOfRevenue"])
    operating_income = _pick(inc_df, ["operatingincomeloss", "operatingIncome"])
    gross_profit = _pick(inc_df, ["grossprofit", "grossProfit"])
    if gross_profit.isna().all():
        gross_profit = revenue - cogs

    op_cf = _pick(cfs_df, [
        "netcashprovidedbyusedinoperatingactivities",
        "netCashProvidedByOperatingActivities",
        "operatingCashFlow",
    ])
    capex = _pick(cfs_df, [
        "paymentstoacquirepropertyplantandequipment",
        "paymentsforcapitalimprovements",
        "capitalExpenditure",
    ]).abs()

    total_assets = _pick(bal_df, ["assets", "totalAssets"])
    total_debt = _pick(bal_df, ["longtermdebtnoncurrent", "longtermdebt", "totalDebt"])
    total_equity = _pick(bal_df, ["stockholdersequity", "totalStockholdersEquity"])
    cash_eq = _pick(bal_df, [
        "cashandcashequivalentsatcarryingvalue",
        "cashandcashequivalents",
        "cashAndCashEquivalents",
    ])

    inc_small = pd.DataFrame(
        {
            "date": inc_df["date"],
            "revenue": revenue.values,
            "operating_income": operating_income.values,
            "gross_profit": gross_profit.values,
        }
    )
    cfs_small = pd.DataFrame({"date": cfs_df["date"], "op_cf": op_cf.values, "capex": capex.values})
    bal_small = pd.DataFrame(
        {
            "date": bal_df["date"],
            "total_assets": total_assets.values,
            "total_debt": total_debt.values,
            "total_equity": total_equity.values,
            "cash_eq": cash_eq.values,
        }
    )

    merged = inc_small.merge(cfs_small, on="date", how="outer").merge(bal_small, on="date", how="outer")
    merged = merged.sort_values("date").set_index("date")

    merged["op_margin"] = merged["operating_income"] / merged["revenue"].replace(0, np.nan)
    merged["gross_profitability"] = merged["gross_profit"] / merged["total_assets"].replace(0, np.nan)

    invested_capital = merged["total_debt"] + merged["total_equity"] - merged["cash_eq"]
    invested_capital = invested_capital.where(invested_capital > 0, merged["total_assets"])
    merged["roic"] = (merged["operating_income"] * (1 - flat_tax_rate)) / invested_capital.replace(0, np.nan)

    if "acceptedDate" not in inc_df.columns:
        return None
    accepted_map = inc_df.set_index("date")["acceptedDate"]
    merged = merged.join(accepted_map, how="left").dropna(subset=["acceptedDate"])
    merged["acceptedDate"] = pd.to_datetime(merged["acceptedDate"])
    merged = merged.set_index("acceptedDate").sort_index()

    out = merged[
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
    return out if not out.empty else None


def fetch_pit_fundamentals(
    client: FmpClient,
    ticker: str,
    flat_tax_rate: float,
    statement_limit: int = 5,
) -> Optional[pd.DataFrame]:
    # Try as-reported quarterly first (matches original v5 design).
    inc = client.cached_get(
        f"https://financialmodelingprep.com/stable/income-statement-as-reported?symbol={ticker}&period=quarter&limit={statement_limit}&apikey={client.api_key}",
        f"inc_as_reported_stable_{ticker}_{statement_limit}",
        ttl_days=3,
    )
    cfs = client.cached_get(
        f"https://financialmodelingprep.com/stable/cash-flow-statement-as-reported?symbol={ticker}&period=quarter&limit={statement_limit}&apikey={client.api_key}",
        f"cfs_as_reported_stable_{ticker}_{statement_limit}",
        ttl_days=3,
    )
    bal = client.cached_get(
        f"https://financialmodelingprep.com/stable/balance-sheet-statement-as-reported?symbol={ticker}&period=quarter&limit={statement_limit}&apikey={client.api_key}",
        f"bal_as_reported_stable_{ticker}_{statement_limit}",
        ttl_days=3,
    )

    use_as_reported = isinstance(inc, list) and isinstance(cfs, list) and isinstance(bal, list) and inc and cfs and bal
    if use_as_reported:
        # Stable as-reported may omit acceptedDate; fall through if so.
        inc_df = _first_accepted_only(_flatten_as_reported(inc))
        cfs_df = _first_accepted_only(_flatten_as_reported(cfs))
        bal_df = _first_accepted_only(_flatten_as_reported(bal))
        if "acceptedDate" in inc_df.columns:
            out = _pit_from_frames(inc_df, cfs_df, bal_df, flat_tax_rate)
            if out is not None:
                return out

    # Fallback: standard quarterly statements (widely available; same PIT schema).
    inc = client.cached_get(
        f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&period=quarter&limit={statement_limit}&apikey={client.api_key}",
        f"inc_std_stable_{ticker}_{statement_limit}",
        ttl_days=3,
    )
    cfs = client.cached_get(
        f"https://financialmodelingprep.com/stable/cash-flow-statement?symbol={ticker}&period=quarter&limit={statement_limit}&apikey={client.api_key}",
        f"cfs_std_stable_{ticker}_{statement_limit}",
        ttl_days=3,
    )
    bal = client.cached_get(
        f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={ticker}&period=quarter&limit={statement_limit}&apikey={client.api_key}",
        f"bal_std_stable_{ticker}_{statement_limit}",
        ttl_days=3,
    )
    if not (isinstance(inc, list) and isinstance(cfs, list) and isinstance(bal, list) and inc and cfs and bal):
        return None

    inc_df = _first_accepted_only(pd.DataFrame(inc))
    cfs_df = _first_accepted_only(pd.DataFrame(cfs))
    bal_df = _first_accepted_only(pd.DataFrame(bal))
    return _pit_from_frames(inc_df, cfs_df, bal_df, flat_tax_rate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download PIT fundamentals")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    client = make_client(config)
    paths = config.get("paths", {})
    benchmark = config.get("benchmark", "QQQ")
    flat_tax_rate = float(config.get("flat_tax_rate", 0.21))
    statement_limit = int(config.get("statement_limit", 5))

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
    print(">> PIT 재무 데이터 수집...")
    fundamental_history = {}
    symbols = [c for c in close_m.columns if c != benchmark]
    for i, t in enumerate(symbols):
        pit = fetch_pit_fundamentals(
            client, t, flat_tax_rate=flat_tax_rate, statement_limit=statement_limit
        )
        if pit is not None:
            fundamental_history[t] = pit
        if (i + 1) % 50 == 0:
            print(f"    ...{i+1}종목 재무데이터 처리")

    out_dir = Path(paths.get("fundamentals", "data/fundamentals"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pit_history.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(fundamental_history, f)
    print(f"[fundamentals] wrote {out_path} ({len(fundamental_history)} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
