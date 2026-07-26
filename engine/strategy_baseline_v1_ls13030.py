"""Baseline v1 long-short 130/30 — isolated structural variant.

Long-only mom+quality baseline remains untouched in
``strategy_baseline_v1_quality.py`` / ``BaselineEngineV1``.

This module only defines:
  - monthly long top-K + short bottom-K from the SAME mom+quality combined score
  - sleeve exposures (long 130%, short 30%)
  - short-eligibility / borrow-fee knobs (HTB ADV filter)

Execution (signed qty, SHORT/COVER, mirrored ATR, daily borrow) lives in
``engine/baseline_engine_ls13030.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from engine.pit_fundamentals import (
    extract_quality_raw,
    get_latest_available_fundamentals,
)
from engine.strategy_baseline_v1 import (
    ATR_MULTIPLIER,
    TOP_LIQUID_POOL,
    TOP_MOMENTUM_COUNT,
)
from engine.strategy_baseline_v1_quality import MOM_WEIGHT, QUALITY_WEIGHT

LONG_EXPOSURE = 1.30
SHORT_EXPOSURE = 0.30
# Flat annualized borrow on short notional (liquid names after HTB filter).
BORROW_FEE_ANNUAL = 0.03
# Exclude from short book if ADV20 is in the bottom quartile of that day's
# liquid pool — proxy for hard-to-borrow / punitive small-cap borrow.
HTB_ADV_PERCENTILE = 0.25
SHORT_BASKET_COUNT = TOP_MOMENTUM_COUNT  # bottom-K = top-K size


def strategy_id() -> str:
    return "baseline_v1_mq_ls13030_atr_nochase"


def strategy_knobs() -> Dict[str, object]:
    return {
        "strategy_id": strategy_id(),
        "TOP_LIQUID_POOL": int(TOP_LIQUID_POOL),
        "TOP_MOMENTUM_COUNT": int(TOP_MOMENTUM_COUNT),
        "SHORT_BASKET_COUNT": int(SHORT_BASKET_COUNT),
        "ATR_MULTIPLIER": float(ATR_MULTIPLIER),
        "LONG_EXPOSURE": float(LONG_EXPOSURE),
        "SHORT_EXPOSURE": float(SHORT_EXPOSURE),
        "BORROW_FEE_ANNUAL": float(BORROW_FEE_ANNUAL),
        "HTB_ADV_PERCENTILE": float(HTB_ADV_PERCENTILE),
        "entry_long": "top-K by combined=0.5*mom+0.5*quality (same as MQ default)",
        "entry_short": (
            "bottom-K by same combined score among short-eligible "
            "(exclude HTB ADV bottom quartile of liquid pool)"
        ),
        "exit_long": "ATR trail Low < peak - mult*ATR → sell next Open",
        "exit_short": "mirrored ATR High > trough + mult*ATR → cover next Open",
        "sizing": "equal-weight within each sleeve; long 130% / short 30% of equity",
        "borrow": f"flat {BORROW_FEE_ANNUAL:.0%} ann. on short notional; HTB excluded not punitive-rated",
        "fill_policy": "no_topup_chasing_default (per sleeve)",
        "uncertain_flags": (
            "borrow is a flat liquid-name assumption (not name-level locate); "
            "HTB via ADV percentile is a proxy, not a real locate feed; "
            "no explicit margin-call / locate-fail modeling"
        ),
    }


def _pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def _combined_mom_quality_ranks(
    *,
    date_idx: int,
    close_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    mom_12_1_m: pd.DataFrame,
    eligible_symbols: Sequence[str],
    fundamental_history: Dict[str, pd.DataFrame],
    fund_ts: Dict[str, Any],
    top_liquid_pool: int = TOP_LIQUID_POOL,
    mom_weight: float = MOM_WEIGHT,
    quality_weight: float = QUALITY_WEIGHT,
) -> Tuple[pd.Series, pd.Series]:
    """Return (combined_score desc-ranked Series, ADV20-proxy Series for liquid set).

    ADV series is dollar-volume on as_of (same panel used for liquid filter).
    """
    empty = pd.Series(dtype=float)
    if date_idx < 252 or not eligible_symbols:
        return empty, empty

    current_date = close_m.index[date_idx]
    syms = [s for s in eligible_symbols if s in close_m.columns and s in dvol_m.columns]
    if not syms:
        return empty, empty

    dvol_row = pd.to_numeric(dvol_m.loc[current_date, syms], errors="coerce").dropna()
    if dvol_row.empty:
        return empty, empty

    liquid = dvol_row.nlargest(min(int(top_liquid_pool), len(dvol_row))).index.tolist()
    mom_row = pd.to_numeric(mom_12_1_m.loc[current_date, liquid], errors="coerce").dropna()
    if mom_row.empty:
        return empty, empty

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

    if not gp_vals:
        return empty, empty

    eligible = list(gp_vals.keys())
    mom = mom_row.reindex(eligible).dropna()
    if mom.empty:
        return empty, empty

    gp = pd.Series(gp_vals).reindex(mom.index)
    roic = pd.Series(roic_vals).reindex(mom.index)
    opm = pd.Series(opm_vals).reindex(mom.index)
    quality = (_pct_rank(gp) + _pct_rank(roic) + _pct_rank(opm)) / 3.0
    combined = float(mom_weight) * _pct_rank(mom) + float(quality_weight) * quality
    ranked = combined.sort_values(ascending=False)
    adv = dvol_row.reindex(ranked.index).astype(float)
    return ranked, adv


def select_monthly_long_short_mom_quality(
    *,
    date_idx: int,
    close_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    mom_12_1_m: pd.DataFrame,
    eligible_symbols: Sequence[str],
    fundamental_history: Dict[str, pd.DataFrame],
    fund_ts: Dict[str, Any],
    top_liquid_pool: int = TOP_LIQUID_POOL,
    long_count: int = TOP_MOMENTUM_COUNT,
    short_count: int = SHORT_BASKET_COUNT,
    htb_adv_percentile: float = HTB_ADV_PERCENTILE,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Return (long_symbols, short_symbols, diagnostics).

    Shorts = lowest combined score among names with ADV >= HTB percentile
    threshold within the ranked (liquid+quality) set. Longs never overlap shorts.
    """
    ranked, adv = _combined_mom_quality_ranks(
        date_idx=date_idx,
        close_m=close_m,
        dvol_m=dvol_m,
        mom_12_1_m=mom_12_1_m,
        eligible_symbols=eligible_symbols,
        fundamental_history=fundamental_history,
        fund_ts=fund_ts,
        top_liquid_pool=top_liquid_pool,
    )
    diag: Dict[str, Any] = {
        "n_ranked": int(len(ranked)),
        "n_short_eligible": 0,
        "n_htb_excluded": 0,
        "adv_htb_threshold": None,
    }
    if ranked.empty:
        return [], [], diag

    longs = ranked.head(int(long_count)).index.tolist()
    long_set = set(longs)

    adv_valid = adv.dropna()
    if adv_valid.empty:
        return longs, [], diag

    thr = float(adv_valid.quantile(float(htb_adv_percentile)))
    short_eligible = adv_valid[adv_valid >= thr].index.tolist()
    htb_excluded = [s for s in ranked.index.tolist() if s not in short_eligible]
    diag["adv_htb_threshold"] = thr
    diag["n_short_eligible"] = int(len(short_eligible))
    diag["n_htb_excluded"] = int(len(htb_excluded))

    # Lowest combined among short-eligible, excluding long book names.
    short_pool = [s for s in ranked.index.tolist()[::-1] if s in set(short_eligible) and s not in long_set]
    shorts = short_pool[: int(short_count)]
    return longs, shorts, diag


def equal_weight_sleeve_targets(
    symbols: Sequence[str],
    exposure: float,
) -> Dict[str, float]:
    """Positive exposure → positive weights; pass negative exposure for shorts."""
    syms = [str(s) for s in symbols]
    n = len(syms)
    if n <= 0 or abs(float(exposure)) < 1e-12:
        return {}
    w = float(exposure) / float(n)
    return {s: w for s in syms}


def refill_sleeve(
    held: Sequence[str],
    candidates: Sequence[str],
    max_positions: int,
) -> List[str]:
    """Keep existing sleeve holdings; fill empties from candidates (no rank-drop)."""
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


__all__ = [
    "select_monthly_long_short_mom_quality",
    "equal_weight_sleeve_targets",
    "refill_sleeve",
    "strategy_id",
    "strategy_knobs",
    "LONG_EXPOSURE",
    "SHORT_EXPOSURE",
    "BORROW_FEE_ANNUAL",
    "HTB_ADV_PERCENTILE",
    "SHORT_BASKET_COUNT",
]
