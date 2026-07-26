"""Sector-restricted momentum+quality — Satellite 1 wrapper.

Reuses the identical MQ combined score from ``strategy_baseline_v1_quality``
(0.5 mom + 0.5 quality, PIT exclude-missing). Only change: eligible names are
filtered to one coarse sector bucket before the liquid→rank pipeline.

Does NOT modify ``strategy_baseline_v1.py`` / quality module internals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from engine.pit_fundamentals import (
    extract_quality_raw,
    get_latest_available_fundamentals,
)
from engine.sector_taxonomy import sector_of_meta
from engine.strategy_baseline_v1 import (
    ATR_MULTIPLIER,
    GROSS_EXPOSURE,
    TOP_LIQUID_POOL,
    TOP_MOMENTUM_COUNT,
    equal_weight_targets,
    refill_from_candidates,
)
from engine.strategy_baseline_v1_quality import MOM_WEIGHT, QUALITY_WEIGHT, _pct_rank

# Cap holdings at this fraction of that month's sector-liquid eligible set.
K_FRAC_OF_ELIGIBLE = 0.25
K_MIN = 5
K_MAX = 30
# Soft viability: months with fewer than this many sector-liquid names are thin.
THIN_ELIGIBLE = 20


def strategy_id(sector_bucket: str) -> str:
    slug = sector_bucket.lower().replace(" ", "_")
    return f"baseline_v1_mq_sector_{slug}"


def strategy_knobs(sector_bucket: str) -> Dict[str, object]:
    return {
        "strategy_id": strategy_id(sector_bucket),
        "sector_bucket": sector_bucket,
        "TOP_LIQUID_POOL": int(TOP_LIQUID_POOL),
        "TOP_MOMENTUM_COUNT_CAP": int(K_MAX),
        "K_FRAC_OF_ELIGIBLE": float(K_FRAC_OF_ELIGIBLE),
        "ATR_MULTIPLIER": float(ATR_MULTIPLIER),
        "GROSS_EXPOSURE": float(GROSS_EXPOSURE),
        "entry": (
            f"sector={sector_bucket} only; then same MQ as Strategy #1: "
            "combined=0.5*mom_pctile+0.5*quality_pctile; exclude missing PIT"
        ),
        "exit": "ATR(14) Wilder trail; Low < stop → next Open",
        "sizing": "equal_weight_1_over_N",
        "leverage": "1.0x_always",
        "fill_policy": "no_topup_chasing_default",
        "mom_weight": MOM_WEIGHT,
        "quality_weight": QUALITY_WEIGHT,
    }


def adaptive_k(n_eligible: int) -> int:
    """K = min(30, max(5, floor(0.25 * n_eligible)))."""
    if n_eligible <= 0:
        return 0
    return int(min(K_MAX, max(K_MIN, int(n_eligible * K_FRAC_OF_ELIGIBLE))))


def select_monthly_candidates_mq_sector(
    *,
    date_idx: int,
    close_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    mom_12_1_m: pd.DataFrame,
    eligible_symbols: Sequence[str],
    fundamental_history: Dict[str, pd.DataFrame],
    fund_ts: Dict[str, Any],
    profile_meta: Dict[str, dict],
    sector_bucket: str,
    top_liquid_pool: int = TOP_LIQUID_POOL,
    mom_weight: float = MOM_WEIGHT,
    quality_weight: float = QUALITY_WEIGHT,
) -> Tuple[List[str], Dict[str, Any]]:
    """Same MQ rank as Strategy #1, after restricting to ``sector_bucket``.

    Returns (ordered candidates, diagnostics dict for that rebalance).
    """
    diag: Dict[str, Any] = {
        "sector_bucket": sector_bucket,
        "n_live_in_sector": 0,
        "n_liquid_sector": 0,
        "n_with_quality": 0,
        "adaptive_k": 0,
        "flag_thin": False,
    }
    if date_idx < 252 or not eligible_symbols:
        return [], diag

    current_date = close_m.index[date_idx]
    # Restrict to sector before liquidity cut (sector is the only universe change).
    sector_syms = [
        s
        for s in eligible_symbols
        if s in close_m.columns
        and s in dvol_m.columns
        and sector_of_meta(profile_meta.get(s)) == sector_bucket
    ]
    diag["n_live_in_sector"] = len(sector_syms)
    if not sector_syms:
        return [], diag

    dvol_row = pd.to_numeric(
        dvol_m.loc[current_date, sector_syms], errors="coerce"
    ).dropna()
    if dvol_row.empty:
        return [], diag

    # Within-sector liquid pool (cap at global TOP_LIQUID_POOL for parity).
    liquid = dvol_row.nlargest(min(int(top_liquid_pool), len(dvol_row))).index.tolist()
    diag["n_liquid_sector"] = len(liquid)
    diag["flag_thin"] = bool(len(liquid) < THIN_ELIGIBLE)

    mom_row = pd.to_numeric(mom_12_1_m.loc[current_date, liquid], errors="coerce").dropna()
    if mom_row.empty:
        return [], diag

    gp_vals: Dict[str, float] = {}
    roic_vals: Dict[str, float] = {}
    opm_vals: Dict[str, float] = {}
    for sym in mom_row.index.tolist():
        row = get_latest_available_fundamentals(
            fundamental_history, fund_ts, sym, current_date
        )
        q = extract_quality_raw(row)
        if q is None:
            continue
        gp_vals[sym] = q["gross_profitability"]
        roic_vals[sym] = q["roic"]
        opm_vals[sym] = q["op_margin"]

    diag["n_with_quality"] = len(gp_vals)
    if not gp_vals:
        return [], diag

    # Eligible for ranking = liquid ∩ mom ∩ quality (same as S1).
    ranked_universe = list(gp_vals.keys())
    k = adaptive_k(len(ranked_universe))
    diag["adaptive_k"] = int(k)
    if k <= 0:
        return [], diag

    mom_s = mom_row.reindex(ranked_universe).dropna()
    gp_s = pd.Series(gp_vals).reindex(mom_s.index)
    roic_s = pd.Series(roic_vals).reindex(mom_s.index)
    opm_s = pd.Series(opm_vals).reindex(mom_s.index)
    quality = (_pct_rank(gp_s) + _pct_rank(roic_s) + _pct_rank(opm_s)) / 3.0
    mom_pct = _pct_rank(mom_s)
    combined = float(mom_weight) * mom_pct + float(quality_weight) * quality
    ordered = combined.sort_values(ascending=False).head(int(k)).index.tolist()
    return ordered, diag
