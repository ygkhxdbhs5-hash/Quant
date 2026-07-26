"""Thin PIT fundamentals helpers for Baseline variants.

Reuses the same on-disk artifact and as-of lookup pattern as StandaloneEngine
(``data/fundamentals/pit_history.pkl``, filing_date ≤ as_of). Does not modify
downloader or StandaloneEngine internals.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


QUALITY_FIELDS = ("gross_profitability", "roic", "op_margin")


def load_pit_history(fundamentals_dir: str | Path = "data/fundamentals") -> Dict[str, pd.DataFrame]:
    path = Path(fundamentals_dir) / "pit_history.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        hist = pickle.load(f) or {}
    if not isinstance(hist, dict):
        return {}
    return hist


def build_fund_ts_index(fundamental_history: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    """Precompute filing-date arrays for searchsorted (same as StandaloneEngine)."""
    out: Dict[str, np.ndarray] = {}
    for sym, hist in (fundamental_history or {}).items():
        if hist is None or getattr(hist, "empty", True):
            continue
        try:
            idx = pd.to_datetime(hist.index)
            out[str(sym)] = idx.to_numpy(dtype="datetime64[ns]")
        except Exception:
            continue
    return out


def get_latest_available_fundamentals(
    fundamental_history: Dict[str, pd.DataFrame],
    fund_ts: Dict[str, np.ndarray],
    symbol: str,
    as_of_date,
) -> Optional[pd.Series]:
    """Return latest as-reported fundamentals with filing_date ≤ as_of_date."""
    hist = fundamental_history.get(symbol)
    if hist is None or hist.empty:
        return None
    ts = fund_ts.get(symbol)
    if ts is not None and len(ts):
        try:
            asof = np.datetime64(pd.Timestamp(as_of_date).to_datetime64())
            i = int(np.searchsorted(ts, asof, side="right") - 1)
            if i < 0:
                return None
            return hist.iloc[i]
        except Exception:
            pass
    past = hist.loc[:as_of_date]
    if past.empty:
        return None
    return past.iloc[-1]


def extract_quality_raw(row: Optional[pd.Series]) -> Optional[Dict[str, float]]:
    """Return GP / ROIC / op_margin only if all three are finite. Else None (exclude)."""
    if row is None:
        return None
    vals: Dict[str, float] = {}
    for field in QUALITY_FIELDS:
        if field not in row.index:
            return None
        v = row[field]
        try:
            fv = float(v)
        except Exception:
            return None
        if not np.isfinite(fv):
            return None
        vals[field] = fv
    return vals


def _finite_field(row: pd.Series, field: str) -> Optional[float]:
    if field not in row.index:
        return None
    try:
        fv = float(row[field])
    except Exception:
        return None
    if not np.isfinite(fv):
        return None
    return fv


def extract_value_components(row: Optional[pd.Series]) -> Optional[Dict[str, float]]:
    """PIT ingredients for FCF/EV: op_cf, capex, debt, cash, diluted shares.

    Enterprise value itself is not stored; callers combine with that day's
    close price: EV = close * diluted_shares + total_debt - cash_eq.
    Returns None if any required field is missing/non-finite or shares ≤ 0.
    """
    if row is None:
        return None
    op_cf = _finite_field(row, "op_cf")
    capex = _finite_field(row, "capex")
    debt = _finite_field(row, "total_debt")
    cash = _finite_field(row, "cash_eq")
    shares = _finite_field(row, "diluted_shares_outstanding")
    if shares is None or shares <= 0:
        shares = _finite_field(row, "basic_shares_outstanding")
    if (
        op_cf is None
        or capex is None
        or debt is None
        or cash is None
        or shares is None
        or shares <= 0
    ):
        return None
    return {
        "op_cf": float(op_cf),
        "capex": float(capex),
        "total_debt": float(debt),
        "cash_eq": float(cash),
        "shares_outstanding": float(shares),
        "fcf": float(op_cf) - float(capex),
    }


def fcf_yield_to_ev(
    *,
    close_price: float,
    value_components: Dict[str, float],
) -> Optional[float]:
    """(op_cf - capex) / EV with EV = price×PIT_shares + debt - cash.

    PIT integrity: shares/debt/cash from filing_date ≤ as_of; price is the
    rebalance day's close (no future price, no stale/future share count).
    """
    try:
        px = float(close_price)
    except Exception:
        return None
    if not np.isfinite(px) or px <= 0:
        return None
    shares = float(value_components["shares_outstanding"])
    debt = float(value_components["total_debt"])
    cash = float(value_components["cash_eq"])
    fcf = float(value_components["fcf"])
    ev = px * shares + debt - cash
    if not np.isfinite(ev) or ev <= 0:
        return None
    yld = fcf / ev
    if not np.isfinite(yld):
        return None
    return float(yld)
