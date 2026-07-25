"""Liquidity factors — dollar volume, turnover, Amihud illiquidity.

Dollar Volume: ADV (trailing dollar volume)
Turnover: shares traded / shares outstanding (PIT shares when available)
Amihud (2002): mean(|ret| / dollar_volume) — higher = less liquid

For the Liquidity category score, Amihud is inverted (higher liquidity better)
before Z-scoring via −Amihud.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def dollar_volume(adv: float) -> float:
    try:
        if not np.isfinite(adv) or float(adv) <= 0:
            return float("nan")
        return float(adv)
    except Exception:
        return float("nan")


def turnover_ratio(dollar_vol: float, price: float, shares: float) -> float:
    """Turnover ≈ (dollar_volume / price) / shares_outstanding."""
    try:
        if not np.isfinite(dollar_vol) or not np.isfinite(price) or not np.isfinite(shares):
            return float("nan")
        if price <= 0 or shares <= 0:
            return float("nan")
        share_vol = float(dollar_vol) / float(price)
        return share_vol / float(shares)
    except Exception:
        return float("nan")


def amihud_illiquidity(
    returns: pd.Series,
    dollar_volumes: pd.Series,
    date_idx: int,
    window: int = 20,
) -> float:
    """Amihud (2002) illiquidity over trailing ``window`` days ending at date_idx."""
    try:
        if date_idx < 1:
            return float("nan")
        lo = max(0, date_idx - window + 1)
        r = pd.to_numeric(returns.iloc[lo : date_idx + 1], errors="coerce")
        dv = pd.to_numeric(dollar_volumes.iloc[lo : date_idx + 1], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = r.abs() / dv.replace(0, np.nan)
        val = float(ratio.replace([np.inf, -np.inf], np.nan).mean(skipna=True))
        return val if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def _shares_outstanding(fundamentals: Optional[Any]) -> float:
    if fundamentals is None:
        return float("nan")
    for key in ("diluted_shares_outstanding", "basic_shares_outstanding"):
        try:
            val = fundamentals[key]
            if pd.notna(val) and float(val) > 0:
                return float(val)
        except Exception:
            continue
    return float("nan")


def liquidity_raw_row(
    adv: float,
    price: float,
    fundamentals: Optional[Any],
    returns: pd.Series,
    dollar_volumes: pd.Series,
    date_idx: int,
) -> Dict[str, float]:
    dv = dollar_volume(adv)
    shares = _shares_outstanding(fundamentals)
    ami = amihud_illiquidity(returns, dollar_volumes, date_idx)
    return {
        "dvol_raw": dv,
        "turnover_raw": turnover_ratio(dv, price, shares),
        # Invert Amihud so higher value = more liquid for Z-scoring direction
        "amihud_liq_raw": (-float(ami)) if np.isfinite(ami) else float("nan"),
        "amihud_raw": ami,
    }
