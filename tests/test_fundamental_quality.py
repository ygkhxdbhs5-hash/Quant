"""Tests for recovered fundamental Quality Bonus Score (soft EQS_WEIGHT tilt)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.entry_quality import EQSWeights, blend_cmvs_eqs
from engine.fundamental_quality import (
    compute_quality_score,
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


def _engine(*, use_quality_bonus: bool = True, eqs_weight: float = 0.05):
    eng = StandaloneEngine.__new__(StandaloneEngine)
    eng.w1, eng.w2, eng.w3, eng.w4, eng.w5 = 0.10, 0.10, 0.15, 0.10, 0.25
    eng.eqs_weights = EQSWeights()
    eng.EQS_BLEND_WEIGHT = 0.55
    eng.USE_QUALITY_BONUS = use_quality_bonus
    eng.EQS_WEIGHT = eqs_weight
    eng.MIN_INDUSTRY_SIZE = 2
    return eng


def test_quality_score_in_unit_interval_and_missing_is_zero():
    df = pd.DataFrame(
        [
            _base_row(symbol="A", roic_raw=0.20, gross_prof_raw=0.40, op_margin_raw=0.25),
            _base_row(symbol="B"),  # all fundamentals missing
        ]
    )
    q, parts = compute_quality_score(df, min_industry_size=2)
    assert q.between(0.0, 1.0).all()
    assert float(q.iloc[1]) == 0.0
    assert float(q.iloc[0]) > 0.0
    assert "rank_quality_core" in parts


def test_higher_fundamentals_rank_higher_quality():
    df = pd.DataFrame(
        [
            _base_row(
                symbol="STRONG",
                industry="Tech",
                roic_raw=0.25,
                gross_prof_raw=0.50,
                op_margin_raw=0.30,
                rev_growth_raw=0.40,
                fcf_sales_yield=0.15,
                debt_to_equity=0.2,
            ),
            _base_row(
                symbol="WEAK",
                industry="Tech",
                roic_raw=0.01,
                gross_prof_raw=0.05,
                op_margin_raw=0.01,
                rev_growth_raw=-0.10,
                fcf_sales_yield=-0.05,
                debt_to_equity=3.0,
            ),
        ]
    )
    q, _ = compute_quality_score(df, min_industry_size=2)
    assert float(q.iloc[0]) > float(q.iloc[1])


def test_eqs_weight_zero_matches_baseline_final_exactly():
    """Backward compatibility: EQS_WEIGHT=0 → identical final_score to CMVS+tech EQS."""
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
            roic_raw=0.02,
            gross_prof_raw=0.08,
            op_margin_raw=0.02,
            rev_growth_raw=0.0,
            fcf_sales_yield=0.0,
            debt_to_equity=2.5,
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


def test_quality_bonus_can_break_ties_without_dominating():
    """Small EQS_WEIGHT tilts ranking when technical scores are nearly equal."""
    # Nearly identical technicals; large fundamental gap
    a = _base_row(
        symbol="HIQ",
        bbs=0.50,
        vzs=0.50,
        cps=0.60,
        rsis=0.50,
        rss=0.60,
        ret5=0.05,
        rsi14=55.0,
        roic_raw=0.35,
        gross_prof_raw=0.55,
        op_margin_raw=0.32,
        rev_growth_raw=0.50,
        fcf_sales_yield=0.18,
        debt_to_equity=0.1,
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
        roic_raw=0.01,
        gross_prof_raw=0.04,
        op_margin_raw=0.01,
        rev_growth_raw=-0.05,
        fcf_sales_yield=-0.02,
        debt_to_equity=4.0,
    )
    df = pd.DataFrame([b, a])

    eng_off = _engine(eqs_weight=0.0)
    ranked_off = StandaloneEngine.rank_universe(eng_off, df.copy())
    # With weight 0, order is stable but not required to prefer HIQ
    scores_off = {
        r["symbol"]: float(r["final_score"]) for _, r in ranked_off.iterrows()
    }
    assert abs(scores_off["HIQ"] - scores_off["LOQ"]) < 1e-12

    eng_on = _engine(eqs_weight=0.05)
    ranked_on = StandaloneEngine.rank_universe(eng_on, df.copy())
    assert ranked_on.iloc[0]["symbol"] == "HIQ"
    bonus_max = float(ranked_on["quality_bonus"].max())
    assert 0.0 < bonus_max <= 0.05 + 1e-9
    # Bonus never dominates: delta from quality << typical CMVS scale (~0.5+)
    assert bonus_max < 0.10


def test_quality_bonus_helper_and_diagnostics():
    q = pd.Series([0.0, 0.5, 1.0])
    assert (quality_bonus(q, use_quality_bonus=True, eqs_weight=0.0) == 0.0).all()
    b = quality_bonus(q, use_quality_bonus=True, eqs_weight=0.05)
    np.testing.assert_allclose(b.to_numpy(), np.array([0.0, 0.025, 0.05]))
    text = format_quality_diagnostics(q, eqs_weight=0.05, use_quality_bonus=True)
    assert "[QUALITY]" in text
    assert "average_quality=" in text
    assert "eqs_weight=0.0500" in text
    assert "quality_bonus_range=" in text
