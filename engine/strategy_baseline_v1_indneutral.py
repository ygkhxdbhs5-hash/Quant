"""Baseline v1 + quality + industry-neutral ranking (isolated variant).

Momentum and quality percentile ranks are computed **within** each industry
group (from ``profile_meta['industry']``). Industry groups with fewer than
``MIN_INDUSTRY_SIZE`` members in that month's ranked universe fall back to
**global** ranks (same threshold as StandaloneEngine / config default = 8).

Combined score remains ``0.5 * mom_rank + 0.5 * quality_rank``. Missing PIT
fundamentals are excluded (never imputed). Global-ranking mom+quality stays
in ``strategy_baseline_v1_quality.py``.
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
from engine.strategy_baseline_v1_quality import MOM_WEIGHT, QUALITY_WEIGHT, _pct_rank

MIN_INDUSTRY_SIZE = 8


def strategy_id() -> str:
    return "baseline_v1_mom_quality_indneutral_atr_nochase"


def strategy_knobs() -> Dict[str, object]:
    return {
        "strategy_id": strategy_id(),
        "TOP_LIQUID_POOL": int(TOP_LIQUID_POOL),
        "TOP_MOMENTUM_COUNT": int(TOP_MOMENTUM_COUNT),
        "ATR_MULTIPLIER": float(ATR_MULTIPLIER),
        "GROSS_EXPOSURE": float(GROSS_EXPOSURE),
        "MIN_INDUSTRY_SIZE": int(MIN_INDUSTRY_SIZE),
        "entry": (
            "industry-neutral: mom_rank & quality_rank = within-industry pctiles "
            f"(fallback to global if industry n < {MIN_INDUSTRY_SIZE}); "
            "combined = 0.5*mom + 0.5*quality; quality = mean(pctile GP/ROIC/op_margin)"
        ),
        "exit": "ATR(14) Wilder trail; Low < stop → next Open",
        "sizing": "equal_weight_1_over_N",
        "leverage": "1.0x_always",
        "fill_policy": "no_topup_chasing_default",
        "mom_weight": MOM_WEIGHT,
        "quality_weight": QUALITY_WEIGHT,
    }


def _industry_of(profile_meta: Dict[str, dict], sym: str) -> str:
    meta = profile_meta.get(sym) or {}
    ind = meta.get("industry") or meta.get("Industry") or "Unknown"
    ind = str(ind).strip()
    return ind if ind else "Unknown"


def _quality_pctile_mean(gp: pd.Series, roic: pd.Series, opm: pd.Series) -> pd.Series:
    return (_pct_rank(gp) + _pct_rank(roic) + _pct_rank(opm)) / 3.0


def select_monthly_candidates_mom_quality_indneutral(
    *,
    date_idx: int,
    close_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    mom_12_1_m: pd.DataFrame,
    eligible_symbols: Sequence[str],
    fundamental_history: Dict[str, pd.DataFrame],
    fund_ts: Dict[str, Any],
    profile_meta: Dict[str, dict],
    top_liquid_pool: int = TOP_LIQUID_POOL,
    top_momentum_count: int = TOP_MOMENTUM_COUNT,
    min_industry_size: int = MIN_INDUSTRY_SIZE,
    mom_weight: float = MOM_WEIGHT,
    quality_weight: float = QUALITY_WEIGHT,
) -> List[str]:
    """Liquidity → PIT quality → industry-neutral 50/50 ranks → top K."""
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

    rows = []
    for sym in mom_row.index.tolist():
        fund = get_latest_available_fundamentals(
            fundamental_history, fund_ts, sym, current_date
        )
        q = extract_quality_raw(fund)
        if q is None:
            continue
        rows.append(
            {
                "symbol": sym,
                "mom": float(mom_row[sym]),
                "gp": q["gross_profitability"],
                "roic": q["roic"],
                "opm": q["op_margin"],
                "industry": _industry_of(profile_meta, sym),
            }
        )
    if not rows:
        return []

    df = pd.DataFrame(rows).set_index("symbol")
    # Global ranks (fallback for small industry groups)
    df["mom_rank_global"] = _pct_rank(df["mom"])
    df["q_rank_global"] = _quality_pctile_mean(df["gp"], df["roic"], df["opm"])

    pieces: List[pd.DataFrame] = []
    for ind, g in df.groupby("industry", sort=False):
        g = g.copy()
        if len(g) < int(min_industry_size):
            g["mom_rank"] = g["mom_rank_global"]
            g["q_rank"] = g["q_rank_global"]
            g["rank_mode"] = "global_fallback"
        else:
            g["mom_rank"] = _pct_rank(g["mom"])
            g["q_rank"] = _quality_pctile_mean(g["gp"], g["roic"], g["opm"])
            g["rank_mode"] = "within_industry"
        pieces.append(g)

    ranked_df = pd.concat(pieces, axis=0)
    ranked_df["combined"] = (
        float(mom_weight) * ranked_df["mom_rank"]
        + float(quality_weight) * ranked_df["q_rank"]
    )
    ordered = ranked_df.sort_values("combined", ascending=False)
    return ordered.head(int(top_momentum_count)).index.tolist()


__all__ = [
    "select_monthly_candidates_mom_quality_indneutral",
    "equal_weight_targets",
    "refill_from_candidates",
    "strategy_id",
    "strategy_knobs",
    "MIN_INDUSTRY_SIZE",
]
