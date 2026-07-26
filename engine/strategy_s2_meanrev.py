"""Strategy #2 — short-term mean-reversion (standalone; does not alter Strategy #1).

Academic framing: Jegadeesh (1990) / Lehmann (1990) short-horizon reversal.
Horizon is weekly evaluation + ~10 trading-day hold — distinct from Strategy #1's
12-1 momentum + monthly ATR trail.

Signal (as-of evaluation date t, no look-ahead):
  liquid pool = top TOP_LIQUID_POOL by dollar volume (same construction as S1)
  ret_5d = close[t]/close[t-5] - 1
  reversal_score = percentile rank of (-ret_5d) within the liquid pool
  optional quality guard: require quality composite pctile >= 0.50 using the
  same PIT GP/ROIC/op_margin composite as Strategy #1

Exit (Strategy #2 specific — not a change to Strategy #1's ATR policy):
  primary: time stop after HOLD_TRADING_DAYS (default 10)
  secondary: hard stop if close/entry - 1 <= STOP_LOSS (default -15%)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from engine.pit_fundamentals import (
    extract_quality_raw,
    get_latest_available_fundamentals,
)
from engine.strategy_baseline_v1 import TOP_LIQUID_POOL  # shared universe constant only

# --- Strategy #2 knobs (isolated) ---
TOP_REVERSAL_COUNT = 30
RET_LOOKBACK_DAYS = 5
HOLD_TRADING_DAYS = 10
STOP_LOSS = -0.15
GROSS_EXPOSURE = 1.0
QUALITY_GUARD_MIN_PCTILE = 0.50


def strategy_id(*, quality_guard: bool) -> str:
    if quality_guard:
        return "strategy2_weekly_rev5_quality_hold10_stop15"
    return "strategy2_weekly_rev5_hold10_stop15"


def strategy_knobs(*, quality_guard: bool) -> Dict[str, object]:
    return {
        "strategy_id": strategy_id(quality_guard=quality_guard),
        "TOP_LIQUID_POOL": int(TOP_LIQUID_POOL),
        "TOP_REVERSAL_COUNT": int(TOP_REVERSAL_COUNT),
        "RET_LOOKBACK_DAYS": int(RET_LOOKBACK_DAYS),
        "HOLD_TRADING_DAYS": int(HOLD_TRADING_DAYS),
        "STOP_LOSS": float(STOP_LOSS),
        "GROSS_EXPOSURE": float(GROSS_EXPOSURE),
        "entry": (
            "reversal_score = pctile(-ret_5d) in liquid top-N; "
            + (
                f"require quality_pctile>={QUALITY_GUARD_MIN_PCTILE}"
                if quality_guard
                else "no quality guard"
            )
        ),
        "exit": (
            f"time_stop_{HOLD_TRADING_DAYS}td OR hard_stop_{STOP_LOSS:.0%} "
            "(S2-specific; S1 ATR trail unchanged)"
        ),
        "sizing": "equal_weight_1_over_N",
        "leverage": "1.0x_always",
        "rebalance": "weekly_first_session; refill empties; no BUY top-up chasing",
        "fill_policy": "no_topup_chasing_default",
        "quality_guard": bool(quality_guard),
    }


def compute_ret_n_panel(close_m: pd.DataFrame, n: int = RET_LOOKBACK_DAYS) -> pd.DataFrame:
    """Trailing n-trading-day return: close[t]/close[t-n] - 1."""
    return close_m / close_m.shift(int(n)) - 1.0


def equal_weight_targets(
    symbols: Sequence[str],
    exposure: float = GROSS_EXPOSURE,
) -> Dict[str, float]:
    syms = [str(s) for s in symbols]
    n = len(syms)
    if n <= 0:
        return {}
    w = float(exposure) / float(n)
    return {s: w for s in syms}


def refill_from_candidates(
    held: Set[str],
    candidates: Sequence[str],
    max_positions: int = TOP_REVERSAL_COUNT,
) -> List[str]:
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


def _quality_pctiles_in_pool(
    *,
    symbols: Sequence[str],
    current_date,
    fundamental_history: Dict[str, pd.DataFrame],
    fund_ts: Dict[str, Any],
) -> pd.Series:
    """Equal-weight avg of GP/ROIC/op_margin cross-sectional pctiles; NaN if missing PIT."""
    gp: Dict[str, float] = {}
    roic: Dict[str, float] = {}
    opm: Dict[str, float] = {}
    for sym in symbols:
        row = get_latest_available_fundamentals(
            fundamental_history, fund_ts, sym, current_date
        )
        q = extract_quality_raw(row)
        if q is None:
            continue
        gp[sym] = q["gross_profitability"]
        roic[sym] = q["roic"]
        opm[sym] = q["op_margin"]
    if not gp:
        return pd.Series(dtype=float)
    gp_s = pd.Series(gp).rank(method="average", pct=True)
    roic_s = pd.Series(roic).rank(method="average", pct=True)
    opm_s = pd.Series(opm).rank(method="average", pct=True)
    return (gp_s + roic_s + opm_s) / 3.0


def select_weekly_reversal_candidates(
    *,
    date_idx: int,
    close_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    ret5_m: pd.DataFrame,
    eligible_symbols: Sequence[str],
    top_liquid_pool: int = TOP_LIQUID_POOL,
    top_reversal_count: int = TOP_REVERSAL_COUNT,
    quality_guard: bool = False,
    fundamental_history: Optional[Dict[str, pd.DataFrame]] = None,
    fund_ts: Optional[Dict[str, Any]] = None,
    quality_min_pctile: float = QUALITY_GUARD_MIN_PCTILE,
) -> List[str]:
    """Liquidity → (optional quality half) → rank by -ret_5d → top K."""
    if date_idx < int(RET_LOOKBACK_DAYS) + 1 or not eligible_symbols:
        return []

    current_date = close_m.index[date_idx]
    syms = [s for s in eligible_symbols if s in close_m.columns and s in dvol_m.columns]
    if not syms:
        return []

    dvol_row = pd.to_numeric(dvol_m.loc[current_date, syms], errors="coerce").dropna()
    if dvol_row.empty:
        return []
    liquid = dvol_row.nlargest(min(int(top_liquid_pool), len(dvol_row))).index.tolist()

    ret_row = pd.to_numeric(ret5_m.loc[current_date, liquid], errors="coerce").dropna()
    if ret_row.empty:
        return []

    if quality_guard:
        if not fundamental_history or fund_ts is None:
            return []
        q_pct = _quality_pctiles_in_pool(
            symbols=ret_row.index.tolist(),
            current_date=current_date,
            fundamental_history=fundamental_history,
            fund_ts=fund_ts,
        )
        if q_pct.empty:
            return []
        keep = q_pct[q_pct >= float(quality_min_pctile)].index
        ret_row = ret_row.reindex(keep).dropna()
        if ret_row.empty:
            return []

    # Highest reversal score = most negative trailing return
    ranked = (-ret_row).sort_values(ascending=False)
    return ranked.head(int(top_reversal_count)).index.tolist()


def exit_reason_for_position(
    *,
    entry_date_idx: int,
    current_date_idx: int,
    entry_price: float,
    mark_price: float,
    hold_trading_days: int = HOLD_TRADING_DAYS,
    stop_loss: float = STOP_LOSS,
) -> Optional[str]:
    """Return exit reason if time or hard stop triggers; else None.

    Time stop counts trading sessions between entry idx and current idx.
    Hard stop uses mark/entry - 1 (checked on close for signal; fill next open
    is handled by the engine's pending-order path).
    """
    if entry_date_idx < 0 or current_date_idx < entry_date_idx:
        return None
    held_td = int(current_date_idx - entry_date_idx)
    if (
        entry_price is not None
        and np.isfinite(entry_price)
        and entry_price > 0
        and mark_price is not None
        and np.isfinite(mark_price)
        and mark_price > 0
    ):
        ret = float(mark_price) / float(entry_price) - 1.0
        if ret <= float(stop_loss):
            return "hard_stop"
    if held_td >= int(hold_trading_days):
        return "time_stop"
    return None
