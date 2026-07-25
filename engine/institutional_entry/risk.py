"""Risk factors — beta, idiosyncratic volatility, 12-month max drawdown.

Higher values = higher risk. Category enters the composite with a minus sign.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def market_beta(
    stock_ret: pd.Series,
    mkt_ret: pd.Series,
    date_idx: int,
    window: int = 126,
) -> float:
    """Rolling OLS beta vs market over ``window`` trading days."""
    try:
        if date_idx < window:
            return float("nan")
        lo = date_idx - window + 1
        y = pd.to_numeric(stock_ret.iloc[lo : date_idx + 1], errors="coerce")
        x = pd.to_numeric(mkt_ret.iloc[lo : date_idx + 1], errors="coerce")
        df = pd.concat([y, x], axis=1).dropna()
        if len(df) < max(40, window // 3):
            return float("nan")
        yy = df.iloc[:, 0].to_numpy(dtype=float)
        xx = df.iloc[:, 1].to_numpy(dtype=float)
        var_x = float(np.var(xx))
        if var_x <= 0:
            return float("nan")
        cov = float(np.cov(yy, xx, ddof=0)[0, 1])
        return cov / var_x
    except Exception:
        return float("nan")


def idiosyncratic_volatility(
    stock_ret: pd.Series,
    mkt_ret: pd.Series,
    date_idx: int,
    window: int = 126,
) -> float:
    """Std of residual from single-factor market model (annualized not required)."""
    try:
        if date_idx < window:
            return float("nan")
        lo = date_idx - window + 1
        y = pd.to_numeric(stock_ret.iloc[lo : date_idx + 1], errors="coerce")
        x = pd.to_numeric(mkt_ret.iloc[lo : date_idx + 1], errors="coerce")
        df = pd.concat([y, x], axis=1).dropna()
        if len(df) < max(40, window // 3):
            return float("nan")
        yy = df.iloc[:, 0].to_numpy(dtype=float)
        xx = df.iloc[:, 1].to_numpy(dtype=float)
        var_x = float(np.var(xx))
        if var_x <= 0:
            return float("nan")
        beta = float(np.cov(yy, xx, ddof=0)[0, 1]) / var_x
        resid = yy - beta * xx
        return float(np.std(resid, ddof=0))
    except Exception:
        return float("nan")


def max_drawdown_12m(close: pd.Series, date_idx: int, window: int = 252) -> float:
    """Maximum peak-to-trough drawdown over the past ``window`` days (positive)."""
    try:
        if date_idx < 5:
            return float("nan")
        lo = max(0, date_idx - window + 1)
        px = pd.to_numeric(close.iloc[lo : date_idx + 1], errors="coerce").dropna()
        if len(px) < 20:
            return float("nan")
        peak = px.cummax()
        dd = 1.0 - px / peak.replace(0, np.nan)
        val = float(dd.max(skipna=True))
        return val if np.isfinite(val) else float("nan")
    except Exception:
        return float("nan")


def risk_raw_row(
    stock_ret: pd.Series,
    mkt_ret: Optional[pd.Series],
    close: pd.Series,
    date_idx: int,
) -> Dict[str, float]:
    beta = float("nan")
    idio = float("nan")
    if mkt_ret is not None:
        beta = market_beta(stock_ret, mkt_ret, date_idx)
        idio = idiosyncratic_volatility(stock_ret, mkt_ret, date_idx)
    return {
        "beta_raw": beta,
        "idiovol_raw": idio,
        "maxdd_raw": max_drawdown_12m(close, date_idx),
    }
