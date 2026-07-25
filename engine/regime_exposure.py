"""Regime-based gross exposure for Baseline v1 (isolated variant).

Reuses the market-regime tiers from StandaloneEngine.determine_market_regime
(QQQ vs 200-day SMA + breadth = % of liquidity-filtered names above SMA50):

  - Strong uptrend: QQQ >= SMA200 AND breadth >= 0.40 → exposure 1.0x
  - Mixed/weak:     QQQ < SMA200 OR breadth < 0.40 (but breadth >= 0.20) → 0.5x
  - Confirmed down: QQQ < SMA200 AND breadth < 0.20 → 0.0x

No leverage above 1.0x in this module (the StandaloneEngine 1.2x boost is
intentionally omitted for this isolated test).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from engine.strategy import RegimeState

BREADTH_STRONG = 0.40
BREADTH_WEAK = 0.20


def compute_sma_panels(close_m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """SMA50 / SMA200 panels (same rolling construction as StandaloneEngine)."""
    sma50 = close_m.rolling(50, min_periods=50).mean()
    sma200 = close_m.rolling(200, min_periods=200).mean()
    return sma50, sma200


def determine_regime_exposure(
    *,
    close_m: pd.DataFrame,
    sma50_m: pd.DataFrame,
    sma200_m: pd.DataFrame,
    date_idx: int,
    benchmark: str,
    liquid_symbols: Sequence[str],
) -> Optional[RegimeState]:
    """Return RegimeState for date_idx, or None if MA200 not ready."""
    bm = str(benchmark).upper()
    if bm not in close_m.columns or bm not in sma200_m.columns:
        return None
    bm_price = close_m[bm].iloc[date_idx]
    bm_ma200 = sma200_m[bm].iloc[date_idx]
    if pd.isna(bm_price) or pd.isna(bm_ma200):
        return None

    above_ma = bool(float(bm_price) >= float(bm_ma200))
    current_date = close_m.index[date_idx]

    syms = [
        s
        for s in liquid_symbols
        if s != bm and s in close_m.columns and s in sma50_m.columns
    ]
    if syms:
        px = close_m.loc[current_date, syms]
        sma = sma50_m.loc[current_date, syms]
        valid = px.notna() & sma.notna()
        total = int(valid.sum())
        above_50 = int((px[valid] > sma[valid]).sum()) if total else 0
    else:
        total = 0
        above_50 = 0
    breadth = (above_50 / total) if total > 0 else 1.0

    if (not above_ma) and breadth < BREADTH_WEAK:
        exposure = 0.0
    elif (not above_ma) or breadth < BREADTH_STRONG:
        exposure = 0.5
    else:
        exposure = 1.0

    return RegimeState(
        exposure=float(exposure),
        breadth=float(breadth),
        benchmark_above_ma200=above_ma,
    )


def exposure_tier_label(exposure: float) -> str:
    if exposure <= 0.0:
        return "0.0x"
    if exposure < 1.0:
        return "0.5x"
    return "1.0x"


def summarize_tier_months(regime_log: Sequence[Dict]) -> Dict[str, float]:
    """Share of logged months at each tier."""
    if not regime_log:
        return {"1.0x": 0.0, "0.5x": 0.0, "0.0x": 0.0, "n_months": 0}
    counts = {"1.0x": 0, "0.5x": 0, "0.0x": 0}
    for row in regime_log:
        counts[exposure_tier_label(float(row.get("exposure", 1.0)))] += 1
    n = len(regime_log)
    return {
        "1.0x": counts["1.0x"] / n,
        "0.5x": counts["0.5x"] / n,
        "0.0x": counts["0.0x"] / n,
        "n_months": n,
        "counts": counts,
    }
