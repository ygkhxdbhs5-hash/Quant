"""Download PIT fundamentals via Massive financials into data/fundamentals/.

Output schema matches StandaloneEngine expectations:
op_margin, roic, gross_profitability, revenue, operating_income, op_cf, capex,
total_debt, cash_eq — indexed by filing_date (PIT accepted proxy).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from downloader.download_universe_utils import load_config, make_client
from downloader.massive_client import MassiveClient


def _statement_df(client: MassiveClient, endpoint: str, ticker: str, limit: int) -> pd.DataFrame:
    rows = client.paginate(
        endpoint,
        cache_key_prefix=f"{endpoint.strip('/').replace('/', '_')}_{ticker}",
        params={
            "tickers.any_of": ticker,
            "timeframe": "quarterly",
            "limit": limit,
            "sort": "filing_date.asc",
        },
        ttl_days=3,
        max_pages=20,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "filing_date" not in df.columns or "period_end" not in df.columns:
        return pd.DataFrame()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["period_end"] = pd.to_datetime(df["period_end"])
    df = df.dropna(subset=["filing_date"]).sort_values("filing_date")
    df = df.drop_duplicates(subset=["fiscal_year", "fiscal_quarter"], keep="first")
    return df


def fetch_pit_fundamentals(
    client: MassiveClient,
    ticker: str,
    flat_tax_rate: float,
    statement_limit: int = 100,
) -> Optional[pd.DataFrame]:
    inc = _statement_df(client, "/stocks/financials/v1/income-statements", ticker, statement_limit)
    cfs = _statement_df(client, "/stocks/financials/v1/cash-flow-statements", ticker, statement_limit)
    bal = _statement_df(client, "/stocks/financials/v1/balance-sheets", ticker, statement_limit)
    if inc.empty or cfs.empty or bal.empty:
        return None

    inc_small = pd.DataFrame(
        {
            "period_end": inc["period_end"],
            "filing_date": inc["filing_date"],
            "revenue": inc.get("revenue"),
            "operating_income": inc.get("operating_income"),
            "gross_profit": inc.get("gross_profit"),
            "fiscal_year": inc.get("fiscal_year"),
            "fiscal_quarter": inc.get("fiscal_quarter"),
        }
    )
    cfs_small = pd.DataFrame(
        {
            "period_end": cfs["period_end"],
            "op_cf": cfs.get("net_cash_from_operating_activities"),
            "capex": cfs.get("purchase_of_property_plant_and_equipment"),
            "fiscal_year": cfs.get("fiscal_year"),
            "fiscal_quarter": cfs.get("fiscal_quarter"),
        }
    )
    # CapEx is typically negative in Massive; take absolute value.
    cfs_small["capex"] = pd.to_numeric(cfs_small["capex"], errors="coerce").abs()

    debt_current = bal.get("debt_current")
    debt_lt = bal.get("long_term_debt_and_capital_lease_obligations")
    total_debt = None
    if debt_current is not None or debt_lt is not None:
        total_debt = pd.to_numeric(debt_current, errors="coerce").fillna(0) + pd.to_numeric(
            debt_lt, errors="coerce"
        ).fillna(0)

    bal_small = pd.DataFrame(
        {
            "period_end": bal["period_end"],
            "total_assets": bal.get("total_assets"),
            "total_debt": total_debt,
            "total_equity": bal.get("total_equity"),
            "cash_eq": bal.get("cash_and_equivalents"),
            "fiscal_year": bal.get("fiscal_year"),
            "fiscal_quarter": bal.get("fiscal_quarter"),
        }
    )

    merged = inc_small.merge(
        cfs_small.drop(columns=["period_end"], errors="ignore"),
        on=["fiscal_year", "fiscal_quarter"],
        how="outer",
    ).merge(
        bal_small.drop(columns=["period_end"], errors="ignore"),
        on=["fiscal_year", "fiscal_quarter"],
        how="outer",
    )
    merged = merged.dropna(subset=["filing_date"]).sort_values("filing_date")
    if merged.empty:
        return None

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
    out.index.name = "acceptedDate"  # engine indexes PIT by accepted/filing date
    return out if not out.empty else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download PIT fundamentals (Massive)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    client = make_client(config)
    paths = config.get("paths", {})
    benchmark = config.get("benchmark", "QQQ")
    flat_tax_rate = float(config.get("flat_tax_rate", 0.21))
    statement_limit = int(config.get("statement_limit", 100))

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
