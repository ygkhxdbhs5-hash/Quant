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


def beta_idiovol_panel_at(
    stock_ret: pd.DataFrame,
    mkt_ret: pd.Series,
    date_idx: int,
    window: int = 126,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized beta + residual vol for many symbols at one date (same defs as scalar)."""
    n = stock_ret.shape[1]
    betas = np.full(n, np.nan, dtype=float)
    idios = np.full(n, np.nan, dtype=float)
    if date_idx < window or mkt_ret is None or stock_ret.empty:
        return betas, idios
    lo = date_idx - window + 1
    y = stock_ret.iloc[lo : date_idx + 1].to_numpy(dtype=float, copy=False)
    x = pd.to_numeric(mkt_ret.iloc[lo : date_idx + 1], errors="coerce").to_numpy(dtype=float)
    min_obs = max(40, window // 3)
    x_ok = np.isfinite(x)
    for j in range(n):
        yj = y[:, j]
        mask = x_ok & np.isfinite(yj)
        if int(mask.sum()) < min_obs:
            continue
        xx = x[mask]
        yy = yj[mask]
        var_x = float(np.var(xx))
        if var_x <= 0:
            continue
        cov = float(np.cov(yy, xx, ddof=0)[0, 1])
        b = cov / var_x
        betas[j] = b
        resid = yy - b * xx
        idios[j] = float(np.std(resid, ddof=0))
    return betas, idios


def max_drawdown_panel_at(
    close: pd.DataFrame,
    date_idx: int,
    window: int = 252,
) -> np.ndarray:
    """Max drawdown over trailing window for many symbols at one date."""
    n = close.shape[1]
    out = np.full(n, np.nan, dtype=float)
    if date_idx < 5 or close.empty:
        return out
    lo = max(0, date_idx - window + 1)
    px = close.iloc[lo : date_idx + 1].to_numpy(dtype=float, copy=False)
    for j in range(n):
        col = px[:, j]
        valid = col[np.isfinite(col) & (col > 0)]
        if valid.size < 20:
            continue
        peak = np.maximum.accumulate(valid)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = 1.0 - valid / peak
        val = float(np.nanmax(dd))
        if np.isfinite(val):
            out[j] = val
    return out
