"""Trend factors — price vs long EMAs and ADX.

* Price > EMA200 (binary)
* EMA50 > EMA200 (binary)
* ADX(14) — Wilder Average Directional Index (continuous strength)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def price_above_ema200(price: float, ema200: float) -> float:
    try:
        if not np.isfinite(price) or not np.isfinite(ema200) or ema200 == 0.0:
            return float("nan")
        return 1.0 if float(price) > float(ema200) else 0.0
    except Exception:
        return float("nan")


def ema50_above_ema200(ema50: float, ema200: float) -> float:
    try:
        if not np.isfinite(ema50) or not np.isfinite(ema200):
            return float("nan")
        return 1.0 if float(ema50) > float(ema200) else 0.0
    except Exception:
        return float("nan")


def compute_adx_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Wilder ADX(period) for one symbol (PIT time series)."""
    try:
        high = pd.to_numeric(high, errors="coerce")
        low = pd.to_numeric(low, errors="coerce")
        close = pd.to_numeric(close, errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        up = high.diff()
        down = -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        plus_dm = pd.Series(plus_dm, index=high.index, dtype=float)
        minus_dm = pd.Series(minus_dm, index=high.index, dtype=float)

        alpha = 1.0 / float(period)
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        plus_di = 100.0 * (
            plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
            / atr.replace(0, np.nan)
        )
        minus_di = 100.0 * (
            minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
            / atr.replace(0, np.nan)
        )
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        return adx.astype(float)
    except Exception:
        return pd.Series(np.nan, index=close.index, dtype=float)


def trend_raw_row(
    price: float,
    ema50: float,
    ema200: float,
    adx: float,
) -> Dict[str, float]:
    return {
        "px_gt_ema200_raw": price_above_ema200(price, ema200),
        "ema50_gt_ema200_raw": ema50_above_ema200(ema50, ema200),
        "adx_raw": float(adx) if np.isfinite(adx) else float("nan"),
    }
