"""Baseline v1 + mom/quality/value/low-vol — isolated entry-ranking variant.

Momentum+quality default remains in ``strategy_baseline_v1_quality.py``.
This module only replaces monthly candidate ranking when
``enable_value_lowvol_factor`` is on:

  liquid pool → require PIT quality + PIT value ingredients + vol60 →
  combined = w_m*mom + w_q*quality + w_v*value + w_l*low_vol → top K.

Value = cross-sectional pctile of FCF/EV where
  FCF = op_cf - capex (PIT)
  EV  = close[as_of] * diluted_shares (PIT) + total_debt (PIT) - cash_eq (PIT)

Low-vol = cross-sectional pctile of (−vol60), vol60 = 60d trailing realized vol.

Missing quality, value, or vol → exclude (never impute).
Exit / sizing / leverage / rebalance cadence unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from engine.pit_fundamentals import (
    extract_quality_raw,
    extract_value_components,
    fcf_yield_to_ev,
    get_latest_available_fundamentals,
)
from engine.strategy_baseline_v1 import (
    ATR_MULTIPLIER,
    GROSS_EXPOSURE,
    TOP_LIQUID_POOL,
    TOP_MOMENTUM_COUNT,
    equal_weight_targets,
    refill_from_candidates,
)

LOWVOL_WINDOW = 60
LOWVOL_MIN_PERIODS = 20

# Illustrative weight presets (mom+quality remain the core).
WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "a": {"mom": 0.35, "quality": 0.35, "value": 0.20, "low_vol": 0.10},
    "b": {"mom": 0.30, "quality": 0.30, "value": 0.20, "low_vol": 0.20},
    "c": {"mom": 0.40, "quality": 0.30, "value": 0.15, "low_vol": 0.15},
}


def resolve_weights(
    variant: Optional[str] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    if weights:
        w = {
            "mom": float(weights["mom"]),
            "quality": float(weights["quality"]),
            "value": float(weights["value"]),
            "low_vol": float(weights["low_vol"]),
        }
    else:
        key = str(variant or "a").strip().lower()
        if key not in WEIGHT_PRESETS:
            raise ValueError(f"Unknown value_lowvol variant {variant!r}; use a|b|c")
        w = dict(WEIGHT_PRESETS[key])
    s = sum(w.values())
    if not np.isfinite(s) or abs(s - 1.0) > 1e-6:
        raise ValueError(f"value/lowvol weights must sum to 1.0, got {w} (sum={s})")
    return w


def strategy_id(variant: str = "a") -> str:
    return f"baseline_v1_mqvl_{str(variant).lower()}_atr_nochase"


def strategy_knobs(variant: str = "a", weights: Optional[Mapping[str, float]] = None) -> Dict[str, object]:
    w = resolve_weights(variant, weights)
    return {
        "strategy_id": strategy_id(variant),
        "TOP_LIQUID_POOL": int(TOP_LIQUID_POOL),
        "TOP_MOMENTUM_COUNT": int(TOP_MOMENTUM_COUNT),
        "ATR_MULTIPLIER": float(ATR_MULTIPLIER),
        "GROSS_EXPOSURE": float(GROSS_EXPOSURE),
        "entry": (
            f"combined = {w['mom']}*mom + {w['quality']}*quality + "
            f"{w['value']}*value(FCF/EV) + {w['low_vol']}*low_vol(-vol60); "
            "quality = mean(pctile(GP), pctile(ROIC), pctile(op_margin)); "
            "EV = close*PIT_shares + debt - cash; exclude missing PIT/vol"
        ),
        "exit": "ATR(14) Wilder trail; Low < stop → next Open",
        "sizing": "equal_weight_1_over_N",
        "leverage": "1.0x_always",
        "fill_policy": "no_topup_chasing_default",
        "value_definition": "fcf_yield_ev_pit_shares_times_close",
        "lowvol_window": int(LOWVOL_WINDOW),
        "value_lowvol_variant": str(variant).lower(),
        **{f"w_{k}": float(v) for k, v in w.items()},
    }


def compute_vol60_panel(
    close_m: pd.DataFrame,
    window: int = LOWVOL_WINDOW,
    min_periods: int = LOWVOL_MIN_PERIODS,
) -> pd.DataFrame:
    """Trailing realized vol of daily returns (std)."""
    ret = close_m.pct_change()
    return ret.rolling(int(window), min_periods=int(min_periods)).std()


def _pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def select_monthly_candidates_mqvl(
    *,
    date_idx: int,
    close_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    mom_12_1_m: pd.DataFrame,
    vol60_m: pd.DataFrame,
    eligible_symbols: Sequence[str],
    fundamental_history: Dict[str, pd.DataFrame],
    fund_ts: Dict[str, Any],
    top_liquid_pool: int = TOP_LIQUID_POOL,
    top_momentum_count: int = TOP_MOMENTUM_COUNT,
    weights: Optional[Mapping[str, float]] = None,
    variant: str = "a",
) -> List[str]:
    """Liquidity → require quality+value+vol → weighted combined rank → top K."""
    if date_idx < 252 or not eligible_symbols:
        return []

    w = resolve_weights(variant, weights)
    current_date = close_m.index[date_idx]
    syms = [s for s in eligible_symbols if s in close_m.columns and s in dvol_m.columns]
    if not syms:
        return []

    dvol_row = pd.to_numeric(dvol_m.loc[current_date, syms], errors="coerce").dropna()
    if dvol_row.empty:
        return []

    liquid = dvol_row.nlargest(min(int(top_liquid_pool), len(dvol_row))).index.tolist()
    mom_row = pd.to_numeric(mom_12_1_m.loc[current_date, liquid], errors="coerce").dropna()
    if mom_row.empty:
        return []

    gp_vals: Dict[str, float] = {}
    roic_vals: Dict[str, float] = {}
    opm_vals: Dict[str, float] = {}
    value_vals: Dict[str, float] = {}
    vol_vals: Dict[str, float] = {}

    for sym in mom_row.index.tolist():
        row = get_latest_available_fundamentals(
            fundamental_history, fund_ts, sym, current_date
        )
        q = extract_quality_raw(row)
        if q is None:
            continue
        vc = extract_value_components(row)
        if vc is None:
            continue
        px = close_m[sym].iloc[date_idx] if sym in close_m.columns else np.nan
        yld = fcf_yield_to_ev(close_price=float(px) if pd.notna(px) else float("nan"), value_components=vc)
        if yld is None:
            continue
        if sym not in vol60_m.columns:
            continue
        vol = vol60_m[sym].iloc[date_idx]
        if pd.isna(vol) or not np.isfinite(float(vol)) or float(vol) <= 0:
            continue

        gp_vals[sym] = q["gross_profitability"]
        roic_vals[sym] = q["roic"]
        opm_vals[sym] = q["op_margin"]
        value_vals[sym] = float(yld)
        vol_vals[sym] = float(vol)

    if not gp_vals:
        return []

    eligible = list(gp_vals.keys())
    mom = mom_row.reindex(eligible).dropna()
    if mom.empty:
        return []

    gp = pd.Series(gp_vals).reindex(mom.index)
    roic = pd.Series(roic_vals).reindex(mom.index)
    opm = pd.Series(opm_vals).reindex(mom.index)
    val = pd.Series(value_vals).reindex(mom.index)
    vol = pd.Series(vol_vals).reindex(mom.index)

    quality = (_pct_rank(gp) + _pct_rank(roic) + _pct_rank(opm)) / 3.0
    mom_pct = _pct_rank(mom)
    value_pct = _pct_rank(val)
    # Higher rank = lower volatility
    low_vol_pct = _pct_rank(-vol)

    combined = (
        float(w["mom"]) * mom_pct
        + float(w["quality"]) * quality
        + float(w["value"]) * value_pct
        + float(w["low_vol"]) * low_vol_pct
    )
    ranked = combined.sort_values(ascending=False)
    return ranked.head(int(top_momentum_count)).index.tolist()


__all__ = [
    "select_monthly_candidates_mqvl",
    "compute_vol60_panel",
    "resolve_weights",
    "WEIGHT_PRESETS",
    "equal_weight_targets",
    "refill_from_candidates",
    "strategy_id",
    "strategy_knobs",
    "LOWVOL_WINDOW",
]
