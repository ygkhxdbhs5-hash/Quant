"""Momentum factors — Jegadeesh & Titman / relative strength.

12-1 Momentum: return from t-12m to t-1m (skip most recent month)
6-Month Momentum: close / close.shift(126) − 1
3-Month Momentum: close / close.shift(63) − 1
Relative Strength vs SPY (or configured benchmark if SPY absent)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def momentum_12_1(close: pd.Series, date_idx: int) -> float:
    """Jegadeesh–Titman 12-1: price[t-21]/price[t-252] − 1."""
    try:
        if date_idx < 252:
            return float("nan")
        p_lag1m = close.iloc[date_idx - 21]
        p_lag12m = close.iloc[date_idx - 252]
        if pd.isna(p_lag1m) or pd.isna(p_lag12m) or float(p_lag12m) == 0.0:
            return float("nan")
        return float(p_lag1m) / float(p_lag12m) - 1.0
    except Exception:
        return float("nan")


def momentum_6m(close: pd.Series, date_idx: int, window: int = 126) -> float:
    try:
        if date_idx < window:
            return float("nan")
        p0 = close.iloc[date_idx]
        p1 = close.iloc[date_idx - window]
        if pd.isna(p0) or pd.isna(p1) or float(p1) == 0.0:
            return float("nan")
        return float(p0) / float(p1) - 1.0
    except Exception:
        return float("nan")


def momentum_3m(close: pd.Series, date_idx: int, window: int = 63) -> float:
    return momentum_6m(close, date_idx, window=window)


def relative_strength_vs_benchmark(
    stock_close: pd.Series,
    bench_close: pd.Series,
    date_idx: int,
    window: int = 126,
) -> float:
    """Relative strength: stock window return − benchmark window return."""
    try:
        s = momentum_6m(stock_close, date_idx, window=window)
        b = momentum_6m(bench_close, date_idx, window=window)
        if not np.isfinite(s) or not np.isfinite(b):
            return float("nan")
        return float(s) - float(b)
    except Exception:
        return float("nan")


def momentum_raw_row(
    stock_close: pd.Series,
    bench_close: Optional[pd.Series],
    date_idx: int,
) -> Dict[str, float]:
    rs = float("nan")
    if bench_close is not None:
        rs = relative_strength_vs_benchmark(stock_close, bench_close, date_idx)
    return {
        "mom_12_1_raw": momentum_12_1(stock_close, date_idx),
        "mom_6m_raw": momentum_6m(stock_close, date_idx),
        "mom_3m_raw": momentum_3m(stock_close, date_idx),
        "rs_bench_raw": rs,
    }
