"""Fundamental Bonus Score — recovered from commented get_universe screens.

Reuses the PIT field extraction and the old hard-filter thresholds
(MIN_ROIC, MIN_FCF_SALES_YIELD, MAX_DEBT_TO_EQUITY, MIN_REVENUE_GROWTH_YOY)
plus the pre-CMVS quality blend (0.50 ROIC / 0.30 GP / 0.20 OP).

This is NOT a hard filter: missing metrics contribute 0; names are never
dropped for fundamentals. Each former reject/pass check becomes a signed
point contribution, clipped to roughly:

    Excellent  +15 … +20
    Good        +8 … +15
    Neutral     ~0
    Weak        −5 … −10

Applied as a limited ranking tilt only:

    FinalScore = CMVS(+technical EQS) + points * FUND_POINT_SCALE

with ``FUND_POINT_SCALE ≈ 0.01`` so +20 pts ≈ +0.20 on the CMVS unit scale
(~15–20% influence; cannot override strong technical gaps).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# Original rank_quality blend from pre-CMVS rank_universe
_W_ROIC = 0.50
_W_GP = 0.30
_W_OP = 0.20

# Asymmetric point band requested for the soft bonus
POINTS_LO = -10.0
POINTS_HI = 20.0

# Default: 20 pts → +0.20 on CMVS-scale final score (~15–20% influence)
DEFAULT_FUND_POINT_SCALE = 0.01


def _fget(fundamentals, key: str, default=np.nan) -> float:
    """Same accessor pattern as the commented get_universe / old build_factors."""
    if fundamentals is None:
        return default
    try:
        val = fundamentals[key]
        return default if pd.isna(val) else float(val)
    except Exception:
        return default


def extract_fundamental_metrics(
    engine,
    symbol: str,
    as_of_date,
    price: float,
) -> Dict[str, Any]:
    """Recover PIT metrics used by the old fundamental screens + quality ranks.

    Fields mirror the commented get_universe block and former build_factors:
    roic, op_margin, gross_profitability, revenue growth, FCF/Sales (or
    Sales/MCap approx), debt_to_equity.
    """
    fundamentals = engine.get_latest_available_fundamentals(symbol, as_of_date)
    out: Dict[str, Any] = {
        "roic_raw": np.nan,
        "op_margin_raw": np.nan,
        "gross_prof_raw": np.nan,
        "rev_growth_raw": np.nan,
        "fcf_sales_yield": np.nan,
        "debt_to_equity": np.nan,
        "has_fundamentals": False,
        "value_approx": False,
    }
    if fundamentals is None:
        return out

    out["has_fundamentals"] = True
    out["roic_raw"] = _fget(fundamentals, "roic")
    out["op_margin_raw"] = _fget(fundamentals, "op_margin")
    out["gross_prof_raw"] = _fget(fundamentals, "gross_profitability")

    # --- Growth (same as commented get_universe / old build_factors) ---
    rev_growth = _fget(fundamentals, "revenue_growth_yoy")
    if pd.isna(rev_growth):
        rev_growth = engine._revenue_growth_yoy(symbol, as_of_date)
    out["rev_growth_raw"] = rev_growth if pd.notna(rev_growth) else np.nan

    # --- Value: FCF/Sales with Sales/MCap fallback (commented get_universe) ---
    revenue = _fget(fundamentals, "revenue")
    op_cf = _fget(fundamentals, "op_cf")
    capex = _fget(fundamentals, "capex")
    value_yield = np.nan
    if pd.notna(revenue) and revenue != 0 and pd.notna(op_cf) and pd.notna(capex):
        value_yield = (op_cf - capex) / revenue
    else:
        shares = _fget(fundamentals, "diluted_shares_outstanding")
        if pd.isna(shares) or shares <= 0:
            shares = _fget(fundamentals, "basic_shares_outstanding")
        if (
            pd.notna(revenue)
            and revenue > 0
            and pd.notna(shares)
            and shares > 0
            and pd.notna(price)
            and price > 0
        ):
            market_cap = float(price) * float(shares)
            if market_cap > 0:
                value_yield = float(revenue) / market_cap
                out["value_approx"] = True
    out["fcf_sales_yield"] = value_yield if pd.notna(value_yield) else np.nan

    # --- Debt (commented get_universe) ---
    debt_to_equity = _fget(fundamentals, "debt_to_equity")
    if pd.isna(debt_to_equity):
        total_debt = _fget(fundamentals, "total_debt")
        total_equity = _fget(fundamentals, "total_equity")
        if pd.notna(total_debt) and pd.notna(total_equity) and total_equity != 0:
            debt_to_equity = float(total_debt) / float(total_equity)
    out["debt_to_equity"] = debt_to_equity if pd.notna(debt_to_equity) else np.nan
    return out


def _industry_pct_rank(df: pd.DataFrame, col: str, min_industry_size: int) -> pd.Series:
    """Industry-relative percentile rank (same grouping idea as old rank_universe)."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    industry = df["industry"] if "industry" in df.columns else pd.Series("Unknown", index=df.index)
    counts = industry.groupby(industry).transform("count")
    group_key = np.where(counts >= int(min_industry_size), industry, "__GLOBAL_FALLBACK__")
    tmp = df[[col]].copy()
    tmp["_group_key"] = group_key
    return tmp.groupby("_group_key")[col].rank(pct=True)


def _piecewise_higher_better(
    values: pd.Series,
    *,
    fail_thr: float,
    good_thr: float,
    excel_thr: float,
    fail_pts: float,
    neutral_pts: float,
    good_pts: float,
    excel_pts: float,
) -> pd.Series:
    """Map a higher-is-better metric onto signed points; NaN → 0.

    At ``fail_thr`` (old hard-filter border) → ``neutral_pts``.
    Below → penalty toward ``fail_pts``; above → bonus toward ``excel_pts``.
    """
    v = pd.to_numeric(values, errors="coerce")
    out = np.zeros(len(values), dtype=float)
    known = v.notna().to_numpy()
    if not known.any():
        return pd.Series(out, index=values.index, dtype=float)

    vv = v.to_numpy(dtype=float)
    mid = float(good_thr)
    hi = float(max(excel_thr, good_thr + 1e-12))
    fail = float(fail_thr)

    below = known & (vv < fail)
    span = max(abs(fail) * 0.5, 0.02)
    depth = np.clip((fail - vv) / span, 0.0, 1.0)
    out = np.where(below, neutral_pts + depth * (fail_pts - neutral_pts), out)

    band1 = known & (vv >= fail) & (vv < mid)
    t1 = np.clip((vv - fail) / max(mid - fail, 1e-12), 0.0, 1.0)
    out = np.where(band1, neutral_pts + t1 * (good_pts - neutral_pts), out)

    band2 = known & (vv >= mid) & (vv < hi)
    t2 = np.clip((vv - mid) / max(hi - mid, 1e-12), 0.0, 1.0)
    out = np.where(band2, good_pts + t2 * (excel_pts - good_pts), out)

    above = known & (vv >= hi)
    out = np.where(above, excel_pts, out)

    return pd.Series(out, index=values.index, dtype=float)


def _piecewise_lower_better(
    values: pd.Series,
    *,
    fail_thr: float,
    good_thr: float,
    excel_thr: float,
    fail_pts: float,
    neutral_pts: float,
    good_pts: float,
    excel_pts: float,
) -> pd.Series:
    """Lower-is-better (e.g. D/E): invert via negation of thresholds."""
    v = pd.to_numeric(values, errors="coerce")
    # Map with higher-better on -v and negated thresholds
    return _piecewise_higher_better(
        -v,
        fail_thr=-fail_thr,
        good_thr=-good_thr,
        excel_thr=-excel_thr,
        fail_pts=fail_pts,
        neutral_pts=neutral_pts,
        good_pts=good_pts,
        excel_pts=excel_pts,
    )


def compute_fundamental_bonus_points(
    df: pd.DataFrame,
    *,
    min_roic: float = 0.03,
    min_fcf_sales_yield: float = -0.05,
    max_debt_to_equity: float = 3.0,
    min_revenue_growth_yoy: float = -0.15,
    min_industry_size: int = 8,
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """Convert old filter checks + quality ranks into points ∈ [−10, +20].

    Missing metrics contribute 0 (never exclude).
    """
    if df.empty:
        return pd.Series(dtype=float), {}

    # --- Former hard screens → signed points (thresholds reused) ---
    # Quality / ROIC (reject was roic <= MIN_ROIC)
    roic_pts = _piecewise_higher_better(
        df["roic_raw"] if "roic_raw" in df.columns else pd.Series(np.nan, index=df.index),
        fail_thr=float(min_roic),
        good_thr=max(float(min_roic) * 2.5, 0.10),  # old hard threshold ~10%
        excel_thr=max(float(min_roic) * 5.0, 0.18),
        fail_pts=-4.0,
        neutral_pts=0.0,
        good_pts=5.0,
        excel_pts=7.0,
    )

    # Value / FCF-Sales (reject was yield <= MIN_FCF_SALES_YIELD)
    fcf_pts = _piecewise_higher_better(
        df["fcf_sales_yield"] if "fcf_sales_yield" in df.columns else pd.Series(np.nan, index=df.index),
        fail_thr=float(min_fcf_sales_yield),
        good_thr=max(float(min_fcf_sales_yield) + 0.08, 0.05),
        excel_thr=max(float(min_fcf_sales_yield) + 0.18, 0.12),
        fail_pts=-3.0,
        neutral_pts=0.0,
        good_pts=2.5,
        excel_pts=4.0,
    )

    # Debt (reject was D/E >= MAX_DEBT_TO_EQUITY); lower better
    # good ≈ half of max (old 1.5 when max was 3.0 / original max 1.5)
    debt_good = float(max_debt_to_equity) * 0.5
    debt_excel = float(max_debt_to_equity) * 0.2
    debt_pts = _piecewise_lower_better(
        df["debt_to_equity"] if "debt_to_equity" in df.columns else pd.Series(np.nan, index=df.index),
        fail_thr=float(max_debt_to_equity),
        good_thr=debt_good,
        excel_thr=debt_excel,
        fail_pts=-3.0,
        neutral_pts=0.0,
        good_pts=2.0,
        excel_pts=3.0,
    )

    # Growth (reject was rev_growth <= MIN_REVENUE_GROWTH_YOY)
    growth_pts = _piecewise_higher_better(
        df["rev_growth_raw"] if "rev_growth_raw" in df.columns else pd.Series(np.nan, index=df.index),
        fail_thr=float(min_revenue_growth_yoy),
        good_thr=max(float(min_revenue_growth_yoy) + 0.15, 0.05),
        excel_thr=max(float(min_revenue_growth_yoy) + 0.40, 0.25),
        fail_pts=-2.5,
        neutral_pts=0.0,
        good_pts=2.5,
        excel_pts=4.0,
    )

    # --- Pre-CMVS quality rank polish (ROIC/GP/OP) → 0..+5 pts ---
    rank_roic = _industry_pct_rank(df, "roic_raw", min_industry_size)
    rank_gp = _industry_pct_rank(df, "gross_prof_raw", min_industry_size)
    rank_op = _industry_pct_rank(df, "op_margin_raw", min_industry_size)
    rank_quality = (
        (_W_ROIC * rank_roic.fillna(0.0))
        + (_W_GP * rank_gp.fillna(0.0))
        + (_W_OP * rank_op.fillna(0.0))
    )
    has_any_core = (
        df["roic_raw"].notna() | df["gross_prof_raw"].notna() | df["op_margin_raw"].notna()
        if all(c in df.columns for c in ("roic_raw", "gross_prof_raw", "op_margin_raw"))
        else pd.Series(False, index=df.index)
    )
    # Only reward relative quality when data exists; missing → 0 (not a penalty)
    quality_rank_pts = (rank_quality.where(has_any_core, 0.0) * 5.0).fillna(0.0)

    points = (
        roic_pts + fcf_pts + debt_pts + growth_pts + quality_rank_pts
    ).clip(POINTS_LO, POINTS_HI)

    parts = {
        "roic_points": roic_pts,
        "fcf_points": fcf_pts,
        "debt_points": debt_pts,
        "growth_points": growth_pts,
        "quality_rank_points": quality_rank_pts,
        "rank_roic": rank_roic.fillna(0.0),
        "rank_gp": rank_gp.fillna(0.0),
        "rank_op": rank_op.fillna(0.0),
        "rank_quality_core": rank_quality.where(has_any_core, 0.0).fillna(0.0),
    }
    return points.astype(float), parts


def compute_quality_score(
    df: pd.DataFrame,
    *,
    min_industry_size: int = 8,
    min_roic: float = 0.03,
    min_fcf_sales_yield: float = -0.05,
    max_debt_to_equity: float = 3.0,
    min_revenue_growth_yoy: float = -0.15,
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """Backward-compatible wrapper: normalized [0, 1] view of bonus points."""
    points, parts = compute_fundamental_bonus_points(
        df,
        min_roic=min_roic,
        min_fcf_sales_yield=min_fcf_sales_yield,
        max_debt_to_equity=max_debt_to_equity,
        min_revenue_growth_yoy=min_revenue_growth_yoy,
        min_industry_size=min_industry_size,
    )
    # Map [-10, 20] → [0, 1] for diagnostics that expect a unit score
    span = POINTS_HI - POINTS_LO
    quality = ((points - POINTS_LO) / span).clip(0.0, 1.0)
    parts = dict(parts)
    parts["fundamental_bonus_points"] = points
    return quality.astype(float), parts


def quality_bonus(
    points_or_score: pd.Series,
    *,
    use_quality_bonus: bool,
    eqs_weight: float,
    input_is_points: bool = True,
) -> pd.Series:
    """Score-unit addition: points * FUND_POINT_SCALE (eqs_weight).

    ``eqs_weight`` here is FUND_POINT_SCALE (default 0.01). When 0 or disabled,
    returns zeros so rankings match the CMVS(+technical EQS) baseline exactly.
    """
    if (not use_quality_bonus) or float(eqs_weight) == 0.0:
        return pd.Series(0.0, index=points_or_score.index, dtype=float)
    vals = points_or_score.astype(float)
    if not input_is_points:
        # Legacy [0,1] path → map to [0, +20] then scale
        vals = vals * POINTS_HI
    return (float(eqs_weight) * vals).astype(float)


def format_quality_diagnostics(
    points: pd.Series,
    *,
    eqs_weight: float,
    use_quality_bonus: bool,
) -> str:
    """Monthly rebalance printout for the fundamental bonus."""
    ps = pd.to_numeric(points, errors="coerce").dropna()
    w_eff = float(eqs_weight) if use_quality_bonus else 0.0
    if ps.empty:
        avg = top = bottom = 0.0
        bonus_lo = bonus_hi = 0.0
    else:
        avg = float(ps.mean())
        top = float(ps.max())
        bottom = float(ps.min())
        bonus_lo = w_eff * bottom
        bonus_hi = w_eff * top
    return (
        "[FUNDAMENTAL BONUS]\n"
        f"average_points={avg:.2f}\n"
        f"top_points={top:.2f}\n"
        f"bottom_points={bottom:.2f}\n"
        f"point_scale={w_eff:.4f}\n"
        f"bonus_score_range=[{bonus_lo:.4f}, {bonus_hi:.4f}]\n"
        f"use_quality_bonus={bool(use_quality_bonus)}"
    )


def thresholds_from_engine(engine: Any) -> Dict[str, float]:
    """Pull the same threshold knobs the commented get_universe used."""
    return {
        "min_roic": float(getattr(engine, "MIN_ROIC", 0.03)),
        "min_fcf_sales_yield": float(getattr(engine, "MIN_FCF_SALES_YIELD", -0.05)),
        "max_debt_to_equity": float(getattr(engine, "MAX_DEBT_TO_EQUITY", 3.0)),
        "min_revenue_growth_yoy": float(getattr(engine, "MIN_REVENUE_GROWTH_YOY", -0.15)),
        "min_industry_size": float(getattr(engine, "MIN_INDUSTRY_SIZE", 8)),
    }
