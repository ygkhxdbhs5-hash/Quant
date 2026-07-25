"""Fundamental Quality Bonus — recovered from pre-CMVS filter/ranking code.

Reuses the PIT field extraction and quality-rank blend from the commented
``get_universe`` screens and the former ``build_factors`` / ``rank_universe``
quality block (roic / gross_profitability / op_margin + revenue growth,
FCF/Sales, debt-to-equity).

This is NOT a hard filter: missing data → 0 contribution; every name still
gets a CMVS score. The result is a [0, 1] ``quality_score`` used only as a
small ranking bonus:

    final_score = baseline_final + EQS_WEIGHT * quality_score
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# Original rank_quality blend from pre-CMVS rank_universe:
#   rank_quality = 0.50 * rank_roic + 0.30 * rank_gp + 0.20 * rank_op
_W_ROIC = 0.50
_W_GP = 0.30
_W_OP = 0.20

# How those pieces (plus filter-era FCF / debt / growth) combine into quality_score.
_W_CORE_QUALITY = 0.55  # old rank_quality
_W_REV_GROWTH = 0.20
_W_FCF = 0.15
_W_DEBT = 0.10


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
    # pandas rank skips NaNs within group — missing stay NaN → later filled with 0
    return tmp.groupby("_group_key")[col].rank(pct=True)


def compute_quality_score(
    df: pd.DataFrame,
    *,
    min_industry_size: int = 8,
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """Normalize recovered fundamental checks into quality_score ∈ [0, 1].

    Missing metrics contribute 0 (stock is never excluded).
    """
    if df.empty:
        return pd.Series(dtype=float), {}

    rank_roic = _industry_pct_rank(df, "roic_raw", min_industry_size)
    rank_gp = _industry_pct_rank(df, "gross_prof_raw", min_industry_size)
    rank_op = _industry_pct_rank(df, "op_margin_raw", min_industry_size)
    # Original quality blend
    rank_quality = (
        (_W_ROIC * rank_roic.fillna(0.0))
        + (_W_GP * rank_gp.fillna(0.0))
        + (_W_OP * rank_op.fillna(0.0))
    )
    # Only count core when at least one quality input existed (else 0)
    has_any_core = (
        df["roic_raw"].notna() | df["gross_prof_raw"].notna() | df["op_margin_raw"].notna()
        if all(c in df.columns for c in ("roic_raw", "gross_prof_raw", "op_margin_raw"))
        else pd.Series(False, index=df.index)
    )
    rank_quality = rank_quality.where(has_any_core, 0.0)

    rank_rev = _industry_pct_rank(df, "rev_growth_raw", min_industry_size).fillna(0.0)
    rank_fcf = _industry_pct_rank(df, "fcf_sales_yield", min_industry_size).fillna(0.0)

    # Debt quality: lower D/E is better → rank of negative D/E
    if "debt_to_equity" in df.columns:
        tmp = df.copy()
        tmp["_neg_de"] = -pd.to_numeric(tmp["debt_to_equity"], errors="coerce")
        rank_debt = _industry_pct_rank(tmp, "_neg_de", min_industry_size).fillna(0.0)
    else:
        rank_debt = pd.Series(0.0, index=df.index)

    quality = (
        _W_CORE_QUALITY * rank_quality.fillna(0.0)
        + _W_REV_GROWTH * rank_rev
        + _W_FCF * rank_fcf
        + _W_DEBT * rank_debt
    ).clip(0.0, 1.0)

    parts = {
        "rank_roic": rank_roic.fillna(0.0),
        "rank_gp": rank_gp.fillna(0.0),
        "rank_op": rank_op.fillna(0.0),
        "rank_quality_core": rank_quality.fillna(0.0),
        "rank_rev_growth": rank_rev,
        "rank_fcf": rank_fcf,
        "rank_debt_quality": rank_debt,
    }
    return quality.astype(float), parts


def quality_bonus(
    quality_score: pd.Series,
    *,
    use_quality_bonus: bool,
    eqs_weight: float,
) -> pd.Series:
    """Small additive bonus; zero when disabled or EQS_WEIGHT == 0."""
    if (not use_quality_bonus) or float(eqs_weight) == 0.0:
        return pd.Series(0.0, index=quality_score.index, dtype=float)
    return (float(eqs_weight) * quality_score.astype(float)).astype(float)


def format_quality_diagnostics(
    quality_score: pd.Series,
    *,
    eqs_weight: float,
    use_quality_bonus: bool,
) -> str:
    qs = quality_score.dropna()
    w_eff = float(eqs_weight) if use_quality_bonus else 0.0
    if qs.empty:
        avg = top = bottom = 0.0
        bonus_lo = bonus_hi = 0.0
    else:
        avg = float(qs.mean())
        top = float(qs.max())
        bottom = float(qs.min())
        bonus_lo = w_eff * bottom
        bonus_hi = w_eff * top
    return (
        "[QUALITY]\n"
        f"average_quality={avg:.4f}\n"
        f"top_quality={top:.4f}\n"
        f"bottom_quality={bottom:.4f}\n"
        f"eqs_weight={w_eff:.4f}\n"
        f"quality_bonus_range=[{bonus_lo:.4f}, {bonus_hi:.4f}]"
    )
