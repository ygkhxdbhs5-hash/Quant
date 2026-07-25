"""Baseline strategy v1 — 12-1 momentum entry + ATR trailing exit only.

This module contains ONLY strategy decision logic. Infrastructure (data load,
execution costs, reporting) lives outside and must not be modified here.

Configurable constants (edit these for future experiments):
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tunable constants (baseline defaults)
# ---------------------------------------------------------------------------
TOP_LIQUID_POOL = 250
TOP_MOMENTUM_COUNT = 30
ATR_MULTIPLIER = 2.5
GROSS_EXPOSURE = 1.0  # always fully invested at 1.0x

# Future experiment candidate: Quality factor overlay on top of mom_12_1
# Future experiment candidate: Rank hysteresis / buffer bands
# Future experiment candidate: Correlation filter / industry caps


def compute_mom_12_1(close: pd.Series, date_idx: int) -> float:
    """Standard 12-1 momentum: price[t-21] / price[t-252] - 1.

    Do not substitute r252−r21 or cumulative-return alternatives.
    """
    if date_idx < 252:
        return float("nan")
    p_lag1m = close.iloc[date_idx - 21]
    p_lag12m = close.iloc[date_idx - 252]
    if pd.isna(p_lag1m) or pd.isna(p_lag12m) or float(p_lag12m) == 0.0:
        return float("nan")
    return float(p_lag1m) / float(p_lag12m) - 1.0


def compute_mom_12_1_panel(close_m: pd.DataFrame) -> pd.DataFrame:
    """Panel form of the same definition: close.shift(21) / close.shift(252) - 1."""
    return close_m.shift(21) / close_m.shift(252) - 1.0


def select_monthly_candidates(
    *,
    date_idx: int,
    close_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    mom_12_1_m: pd.DataFrame,
    eligible_symbols: Sequence[str],
    top_liquid_pool: int = TOP_LIQUID_POOL,
    top_momentum_count: int = TOP_MOMENTUM_COUNT,
) -> List[str]:
    """Rank by dollar volume → top liquid pool → rank by mom_12_1 → top K.

    Returns ordered candidate list (highest momentum first).
    """
    if date_idx < 252 or not eligible_symbols:
        return []

    current_date = close_m.index[date_idx]
    syms = [s for s in eligible_symbols if s in close_m.columns and s in dvol_m.columns]
    if not syms:
        return []

    dvol_row = dvol_m.loc[current_date, syms]
    dvol_row = pd.to_numeric(dvol_row, errors="coerce").dropna()
    if dvol_row.empty:
        return []

    liquid = dvol_row.nlargest(min(int(top_liquid_pool), len(dvol_row))).index.tolist()
    mom_row = mom_12_1_m.loc[current_date, liquid]
    mom_row = pd.to_numeric(mom_row, errors="coerce").dropna()
    if mom_row.empty:
        return []

    ranked = mom_row.sort_values(ascending=False)
    return ranked.head(int(top_momentum_count)).index.tolist()


def atr_stop_line(
    highest_close_since_entry: float,
    atr14: float,
    atr_multiplier: float = ATR_MULTIPLIER,
) -> float:
    """stop_line = highest_close_since_entry - ATR_MULTIPLIER × ATR(14)."""
    if not np.isfinite(highest_close_since_entry) or not np.isfinite(atr14):
        return float("nan")
    return float(highest_close_since_entry) - float(atr_multiplier) * float(atr14)


def atr_stop_triggered(
    day_low: float,
    highest_close_since_entry: float,
    atr14: float,
    atr_multiplier: float = ATR_MULTIPLIER,
) -> bool:
    """True if day's Low falls below the trailing stop line."""
    stop = atr_stop_line(highest_close_since_entry, atr14, atr_multiplier)
    if not np.isfinite(stop) or not np.isfinite(day_low):
        return False
    return float(day_low) < float(stop)


def equal_weight_targets(
    symbols: Sequence[str],
    exposure: float = GROSS_EXPOSURE,
) -> Dict[str, float]:
    """Equal weight 1/N across currently held (or target) names; sum = exposure."""
    syms = [str(s) for s in symbols]
    n = len(syms)
    if n <= 0:
        return {}
    w = float(exposure) / float(n)
    return {s: w for s in syms}


def refill_from_candidates(
    held: Set[str],
    candidates: Sequence[str],
    max_positions: int = TOP_MOMENTUM_COUNT,
) -> List[str]:
    """Keep existing holdings; fill empty slots from current month candidates.

    Does not drop a holding merely because it left the Top-K ranking.
    """
    keep = [s for s in held if s]
    selected = list(keep)
    held_set = set(keep)
    for sym in candidates:
        if len(selected) >= int(max_positions):
            break
        if sym in held_set:
            continue
        selected.append(sym)
        held_set.add(sym)
    return selected


def strategy_id() -> str:
    return "baseline_v1_mom12_1_atr"


def strategy_knobs() -> Dict[str, object]:
    return {
        "strategy_id": strategy_id(),
        "TOP_LIQUID_POOL": int(TOP_LIQUID_POOL),
        "TOP_MOMENTUM_COUNT": int(TOP_MOMENTUM_COUNT),
        "ATR_MULTIPLIER": float(ATR_MULTIPLIER),
        "GROSS_EXPOSURE": float(GROSS_EXPOSURE),
        "entry": "mom_12_1 = price[t-21]/price[t-252]-1",
        "exit": "ATR(14) Wilder trail; Low < stop → next Open",
        "sizing": "equal_weight_1_over_N",
        "leverage": "1.0x_always",
        "rebalance": "monthly_first_session; hold until ATR stop; refill empties",
    }
