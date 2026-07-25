"""Faster regime exposure variant for Baseline v1 (isolated; does not replace sma200).

Trend: QQQ vs **50-day SMA** (not 200).
Breadth: % of liquidity-filtered universe above their SMA50, plus a
rate-of-change override — breadth is "improving" if it rose by at least
+10 percentage points over the trailing 20 trading days. Improving breadth
can satisfy the strong-uptrend breadth condition even if level < 0.40.

Tiers (no leverage above 1.0x):
  - Strong:  QQQ >= SMA50 AND (breadth >= 0.40 OR improving) → 1.0x
  - Mixed:   otherwise, except confirmed down → 0.5x
  - Down:    QQQ < SMA50 AND breadth < 0.20 AND not improving → 0.0x

The rejected sma200/level-only rule lives in ``regime_exposure.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from engine.regime_exposure import BREADTH_STRONG, BREADTH_WEAK

BREADTH_ROC_DAYS = 20
BREADTH_ROC_THRESHOLD = 0.10  # +10 percentage points


@dataclass
class FastRegimeState:
    exposure: float
    breadth: float
    breadth_prev_20d: Optional[float]
    breadth_improving: bool
    benchmark_above_ma50: bool


def _breadth_level(
    *,
    close_m: pd.DataFrame,
    sma50_m: pd.DataFrame,
    date_idx: int,
    benchmark: str,
    liquid_symbols: Sequence[str],
) -> float:
    bm = str(benchmark).upper()
    current_date = close_m.index[date_idx]
    syms = [
        s
        for s in liquid_symbols
        if s != bm and s in close_m.columns and s in sma50_m.columns
    ]
    if not syms:
        return 1.0
    px = close_m.loc[current_date, syms]
    sma = sma50_m.loc[current_date, syms]
    valid = px.notna() & sma.notna()
    total = int(valid.sum())
    if total <= 0:
        return 1.0
    above_50 = int((px[valid] > sma[valid]).sum())
    return float(above_50 / total)


def determine_regime_exposure_fast(
    *,
    close_m: pd.DataFrame,
    sma50_m: pd.DataFrame,
    date_idx: int,
    benchmark: str,
    liquid_symbols: Sequence[str],
) -> Optional[FastRegimeState]:
    """Return fast regime state, or None if SMA50 not ready for QQQ."""
    bm = str(benchmark).upper()
    if bm not in close_m.columns or bm not in sma50_m.columns:
        return None
    bm_price = close_m[bm].iloc[date_idx]
    bm_ma50 = sma50_m[bm].iloc[date_idx]
    if pd.isna(bm_price) or pd.isna(bm_ma50):
        return None

    above_ma50 = bool(float(bm_price) >= float(bm_ma50))
    breadth = _breadth_level(
        close_m=close_m,
        sma50_m=sma50_m,
        date_idx=date_idx,
        benchmark=bm,
        liquid_symbols=liquid_symbols,
    )
    breadth_prev = None
    improving = False
    if date_idx >= BREADTH_ROC_DAYS:
        breadth_prev = _breadth_level(
            close_m=close_m,
            sma50_m=sma50_m,
            date_idx=date_idx - BREADTH_ROC_DAYS,
            benchmark=bm,
            liquid_symbols=liquid_symbols,
        )
        improving = bool((breadth - float(breadth_prev)) >= BREADTH_ROC_THRESHOLD)

    breadth_ok_strong = bool(breadth >= BREADTH_STRONG or improving)

    if above_ma50 and breadth_ok_strong:
        exposure = 1.0
    elif (not above_ma50) and breadth < BREADTH_WEAK and not improving:
        exposure = 0.0
    else:
        exposure = 0.5

    return FastRegimeState(
        exposure=float(exposure),
        breadth=float(breadth),
        breadth_prev_20d=float(breadth_prev) if breadth_prev is not None else None,
        breadth_improving=bool(improving),
        benchmark_above_ma50=above_ma50,
    )
