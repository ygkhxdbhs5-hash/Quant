"""Tests for recovered fundamental Bonus Score (signed points, soft tilt)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.entry_quality import EQSWeights, blend_cmvs_eqs
from engine.fundamental_quality import (
    DEFAULT_FUND_POINT_SCALE,
    compute_fundamental_bonus_points,
    format_quality_diagnostics,
    quality_bonus,
)
from engine.strategy import StandaloneEngine


def _base_row(**overrides):
    row = {
        "symbol": "X",
        "bbs": 0.5,
        "vzs": 0.5,
        "cps": 0.6,
        "rsis": 0.5,
        "rss": 0.6,
        "ret5": 0.05,
        "ret10": 0.08,
        "rsi14": 55.0,
        "trend_score": 0.8,
        "up_frac20": 0.55,
        "atr_pct": 0.05,
        "close_vs_high20": 0.92,
        "price": 100.0,
        "ema20": 98.0,
        "ema50": 95.0,
        "ema20_slope5": 0.01,
        "ema50_slope5": 0.008,
        "pullback_pct": 0.08,
        "pullback_vol_ratio": 0.75,
        "vol_quiet_ratio": 0.80,
        "vol_expand_ratio": 1.40,
        "vol_spike_persist": 0.05,
        "range_pct_5": 0.04,
        "range_pct_10": 0.07,
        "range_pct_20": 0.11,
        "atr_shrink_ratio": 0.85,
        "rs_raw": 0.12,
        "rs_slope5": 0.03,
        "rs_near_high60": 0.95,
        "dist_ema20": 0.02,
        "dist_ema50": 0.05,
        "above_ema50": 1.0,
        "green_streak": 2.0,
        "industry": "Tech",
        "roic_raw": np.nan,
        "op_margin_raw": np.nan,
        "gross_prof_raw": np.nan,
        "rev_growth_raw": np.nan,
        "fcf_sales_yield": np.nan,
        "debt_to_equity": np.nan,
    }
    row.update(overrides)
    return row


def _engine(*, use_quality_bonus: bool = True, eqs_weight: float = DEFAULT_FUND_POINT_SCALE):
    eng = StandaloneEngine.__new__(StandaloneEngine)
    eng.w1, eng.w2, eng.w3, eng.w4, eng.w5 = 0.10, 0.10, 0.15, 0.10, 0.25
    eng.eqs_weights = EQSWeights()
    eng.EQS_BLEND_WEIGHT = 0.55
    eng.USE_QUALITY_BONUS = use_quality_bonus
    eng.EQS_WEIGHT = eqs_weight
    eng.MIN_INDUSTRY_SIZE = 2
    eng.MIN_ROIC = 0.03
    eng.MIN_FCF_SALES_YIELD = -0.05
    eng.MAX_DEBT_TO_EQUITY = 3.0
    eng.MIN_REVENUE_GROWTH_YOY = -0.15
    return eng


def test_missing_fundamentals_contribute_zero_points():
    df = pd.DataFrame([_base_row(symbol="B")])
    pts, _ = compute_fundamental_bonus_points(df, min_industry_size=2)
    assert float(pts.iloc[0]) == 0.0


def test_excellent_vs_weak_point_bands():
    df = pd.DataFrame(
        [
            _base_row(
                symbol="EXCELLENT",
                industry="Tech",
                roic_raw=0.22,
                gross_prof_raw=0.55,
                op_margin_raw=0.30,
                rev_growth_raw=0.40,
                fcf_sales_yield=0.15,
                debt_to_equity=0.3,
            ),
            _base_row(
                symbol="WEAK",
                industry="Tech",
                roic_raw=0.0,
                gross_prof_raw=0.02,
                op_margin_raw=0.0,
                rev_growth_raw=-0.25,
                fcf_sales_yield=-0.10,
                debt_to_equity=4.5,
            ),
            _base_row(symbol="MISSING", industry="Tech"),
        ]
    )
    pts, parts = compute_fundamental_bonus_points(df, min_industry_size=2)
    assert 8.0 <= float(pts.iloc[0]) <= 20.0
    assert -10.0 <= float(pts.iloc[1]) <= -2.0
    assert float(pts.iloc[2]) == 0.0
    assert float(pts.iloc[0]) > float(pts.iloc[1])
    assert "roic_points" in parts


def test_point_scale_zero_matches_baseline_exactly():
    rows = [
        _base_row(
            symbol="Q",
            rss=0.55,
            roic_raw=0.30,
            gross_prof_raw=0.45,
            op_margin_raw=0.28,
            rev_growth_raw=0.35,
            fcf_sales_yield=0.12,
            debt_to_equity=0.3,
        ),
        _base_row(
            symbol="P",
            rss=0.58,
            bbs=0.55,
            roic_raw=0.0,
            gross_prof_raw=0.08,
            op_margin_raw=0.02,
            rev_growth_raw=-0.20,
            fcf_sales_yield=-0.08,
            debt_to_equity=4.0,
        ),
    ]
    df = pd.DataFrame(rows)
    eng0 = _engine(use_quality_bonus=True, eqs_weight=0.0)
    ranked0 = StandaloneEngine.rank_universe(eng0, df.copy())
    baseline = blend_cmvs_eqs(
        ranked0["cmvs_score"], ranked0["eqs"], eng0.EQS_BLEND_WEIGHT
    )
    np.testing.assert_allclose(
        ranked0["final_score"].to_numpy(dtype=float),
        baseline.to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )
    assert (ranked0["quality_bonus"] == 0.0).all()


def test_bonus_breaks_ties_but_does_not_dominate_strong_technicals():
    # Near-tie technicals → fundamentals decide
    a = _base_row(
        symbol="HIQ",
        bbs=0.50,
        vzs=0.50,
        cps=0.60,
        rsis=0.50,
        rss=0.60,
        ret5=0.05,
        rsi14=55.0,
        roic_raw=0.25,
        gross_prof_raw=0.55,
        op_margin_raw=0.32,
        rev_growth_raw=0.45,
        fcf_sales_yield=0.16,
        debt_to_equity=0.2,
    )
    b = _base_row(
        symbol="LOQ",
        bbs=0.50,
        vzs=0.50,
        cps=0.60,
        rsis=0.50,
        rss=0.60,
        ret5=0.05,
        rsi14=55.0,
        roic_raw=0.0,
        gross_prof_raw=0.04,
        op_margin_raw=0.01,
        rev_growth_raw=-0.20,
        fcf_sales_yield=-0.08,
        debt_to_equity=4.0,
    )
    df = pd.DataFrame([b, a])
    eng = _engine(eqs_weight=0.01)
    ranked = StandaloneEngine.rank_universe(eng, df.copy())
    assert ranked.iloc[0]["symbol"] == "HIQ"
    bonus_abs = float(ranked["quality_bonus"].abs().max())
    assert 0.0 < bonus_abs <= 0.20 + 1e-9

    # Strong technical gap must not be overturned by fundamentals
    strong_tech = _base_row(
        symbol="TECH",
        bbs=0.9,
        vzs=0.85,
        cps=0.85,
        rsis=0.7,
        rss=0.90,
        ret5=0.06,
        rsi14=60.0,
        trend_score=0.95,
        up_frac20=0.70,
        close_vs_high20=0.98,
        # weak fundamentals
        roic_raw=0.0,
        gross_prof_raw=0.05,
        op_margin_raw=0.01,
        rev_growth_raw=-0.20,
        fcf_sales_yield=-0.08,
        debt_to_equity=4.0,
    )
    weak_tech = _base_row(
        symbol="FUND",
        bbs=0.35,
        vzs=0.30,
        cps=0.35,
        rsis=0.35,
        rss=0.35,
        ret5=0.02,
        rsi14=48.0,
        trend_score=0.40,
        up_frac20=0.40,
        close_vs_high20=0.80,
        # excellent fundamentals
        roic_raw=0.25,
        gross_prof_raw=0.55,
        op_margin_raw=0.30,
        rev_growth_raw=0.40,
        fcf_sales_yield=0.15,
        debt_to_equity=0.2,
    )
    ranked2 = StandaloneEngine.rank_universe(
        eng, pd.DataFrame([weak_tech, strong_tech])
    )
    assert ranked2.iloc[0]["symbol"] == "TECH"


def test_diagnostics_and_helper():
    pts = pd.Series([-5.0, 0.0, 18.0])
    assert (quality_bonus(pts, use_quality_bonus=True, eqs_weight=0.0) == 0.0).all()
    b = quality_bonus(pts, use_quality_bonus=True, eqs_weight=0.01)
    np.testing.assert_allclose(b.to_numpy(), np.array([-0.05, 0.0, 0.18]))
    text = format_quality_diagnostics(pts, eqs_weight=0.01, use_quality_bonus=True)
    assert "[FUNDAMENTAL BONUS]" in text
    assert "average_points=" in text
    assert "point_scale=0.0100" in text
