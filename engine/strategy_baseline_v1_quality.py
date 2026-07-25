"""Baseline v1 + quality overlay — isolated entry-ranking variant.

Momentum-only baseline remains in ``strategy_baseline_v1.py`` and is unchanged.
This module only replaces monthly candidate ranking:

  liquid pool (same dvol Top-N) →
  require PIT quality (GP, ROIC, op_margin) as-of rebalance date →
  quality = equal-weight avg of cross-sectional percentile ranks →
  combined = 0.5 * mom_rank + 0.5 * quality_rank →
  top K by combined score.

Names with momentum but no usable PIT fundamentals are **excluded** (never
imputed as 0 / bad). Exit / sizing / leverage / rebalance cadence are reused
from the momentum-only baseline helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from engine.pit_fundamentals import (
    extract_quality_raw,
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

MOM_WEIGHT = 0.5
QUALITY_WEIGHT = 0.5


def strategy_id() -> str:
    return "baseline_v1_mom12_1_quality_atr_nochase"


def strategy_knobs() -> Dict[str, object]:
    return {
        "strategy_id": strategy_id(),
        "TOP_LIQUID_POOL": int(TOP_LIQUID_POOL),
        "TOP_MOMENTUM_COUNT": int(TOP_MOMENTUM_COUNT),
        "ATR_MULTIPLIER": float(ATR_MULTIPLIER),
        "GROSS_EXPOSURE": float(GROSS_EXPOSURE),
        "entry": (
            "combined = 0.5*mom_12_1_pctile + 0.5*quality_pctile; "
            "quality = mean(pctile(GP), pctile(ROIC), pctile(op_margin)); "
            "exclude missing PIT fundamentals"
        ),
        "exit": "ATR(14) Wilder trail; Low < stop → next Open",
        "sizing": "equal_weight_1_over_N",
        "leverage": "1.0x_always",
        "fill_policy": "no_topup_chasing_default",
        "mom_weight": MOM_WEIGHT,
        "quality_weight": QUALITY_WEIGHT,
    }


def _pct_rank(series: pd.Series) -> pd.Series:
    """Higher raw value → higher percentile in (0, 1]."""
    return series.rank(method="average", pct=True)


def select_monthly_candidates_mom_quality(
    *,
    date_idx: int,
    close_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    mom_12_1_m: pd.DataFrame,
    eligible_symbols: Sequence[str],
    fundamental_history: Dict[str, pd.DataFrame],
    fund_ts: Dict[str, Any],
    top_liquid_pool: int = TOP_LIQUID_POOL,
    top_momentum_count: int = TOP_MOMENTUM_COUNT,
    mom_weight: float = MOM_WEIGHT,
    quality_weight: float = QUALITY_WEIGHT,
) -> List[str]:
    """Liquidity → require PIT quality → 50/50 mom+quality rank → top K."""
    if date_idx < 252 or not eligible_symbols:
        return []

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

    # Require usable PIT quality for every name that enters the ranked set.
    gp_vals: Dict[str, float] = {}
    roic_vals: Dict[str, float] = {}
    opm_vals: Dict[str, float] = {}
    for sym in mom_row.index.tolist():
        row = get_latest_available_fundamentals(
            fundamental_history, fund_ts, sym, current_date
        )
        q = extract_quality_raw(row)
        if q is None:
            continue  # exclude — do not impute
        gp_vals[sym] = q["gross_profitability"]
        roic_vals[sym] = q["roic"]
        opm_vals[sym] = q["op_margin"]

    if not gp_vals:
        return []

    eligible = list(gp_vals.keys())
    mom = mom_row.reindex(eligible).dropna()
    if mom.empty:
        return []

    gp = pd.Series(gp_vals).reindex(mom.index)
    roic = pd.Series(roic_vals).reindex(mom.index)
    opm = pd.Series(opm_vals).reindex(mom.index)

    quality = (_pct_rank(gp) + _pct_rank(roic) + _pct_rank(opm)) / 3.0
    mom_pct = _pct_rank(mom)
    combined = float(mom_weight) * mom_pct + float(quality_weight) * quality
    ranked = combined.sort_values(ascending=False)
    return ranked.head(int(top_momentum_count)).index.tolist()


# Re-export helpers so the engine can import from one place for the variant.
__all__ = [
    "select_monthly_candidates_mom_quality",
    "equal_weight_targets",
    "refill_from_candidates",
    "strategy_id",
    "strategy_knobs",
    "MOM_WEIGHT",
    "QUALITY_WEIGHT",
]
