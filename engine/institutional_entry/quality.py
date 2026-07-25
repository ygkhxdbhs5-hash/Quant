"""Quality factors — Novy-Marx / Fama–French profitability family.

Gross Profitability (Novy-Marx 2013): Gross Profit / Total Assets
Operating Profitability (Fama–French 2015 style): Operating Income / Book Equity
ROE: Net Income / Book Equity
ROA: Net Income / Total Assets

All use Point-in-Time fundamentals as-of the rebalance date.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _f(fundamentals: Optional[Any], key: str) -> float:
    if fundamentals is None:
        return float("nan")
    try:
        val = fundamentals[key]
    except Exception:
        return float("nan")
    try:
        if pd.isna(val):
            return float("nan")
        return float(val)
    except Exception:
        return float("nan")


def gross_profitability(fundamentals: Optional[Any]) -> float:
    """Novy-Marx (2013): Gross Profit / Total Assets (stored PIT ratio)."""
    try:
        return _f(fundamentals, "gross_profitability")
    except Exception:
        return float("nan")


def operating_profitability(fundamentals: Optional[Any]) -> float:
    """Fama–French operating profitability proxy: Operating Income / Book Equity."""
    try:
        op = _f(fundamentals, "operating_income")
        be = _f(fundamentals, "total_equity")
        if not np.isfinite(op) or not np.isfinite(be) or be == 0.0:
            return float("nan")
        return float(op) / float(be)
    except Exception:
        return float("nan")


def return_on_equity(fundamentals: Optional[Any]) -> float:
    """ROE = Net Income / Book Equity."""
    try:
        ni = _f(fundamentals, "net_income")
        be = _f(fundamentals, "total_equity")
        if not np.isfinite(ni) or not np.isfinite(be) or be == 0.0:
            return float("nan")
        return float(ni) / float(be)
    except Exception:
        return float("nan")


def return_on_assets(fundamentals: Optional[Any]) -> float:
    """ROA = Net Income / Total Assets."""
    try:
        ni = _f(fundamentals, "net_income")
        ta = _f(fundamentals, "total_assets")
        if not np.isfinite(ni) or not np.isfinite(ta) or ta == 0.0:
            return float("nan")
        return float(ni) / float(ta)
    except Exception:
        return float("nan")


def quality_raw_row(fundamentals: Optional[Any]) -> Dict[str, float]:
    return {
        "gp_raw": gross_profitability(fundamentals),
        "op_raw": operating_profitability(fundamentals),
        "roe_raw": return_on_equity(fundamentals),
        "roa_raw": return_on_assets(fundamentals),
    }
