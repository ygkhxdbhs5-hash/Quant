"""Tests for Entry Quality Score (EQS) live ranking integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.entry_quality import (
    EQSWeights,
    blend_cmvs_eqs,
    combine_eqs,
    compute_eqs_components,
    score_overextension_penalty,
    score_trend_structure,
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
    }
    row.update(overrides)
    return row


def test_trend_structure_rewards_aligned_emas():
    good = pd.DataFrame([_base_row(symbol="GOOD")])
    bad = pd.DataFrame(
        [_base_row(symbol="BAD", price=90.0, ema20=95.0, ema50=100.0, ema20_slope5=-0.02)]
    )
    assert float(score_trend_structure(good).iloc[0]) > float(score_trend_structure(bad).iloc[0])


def test_overextension_penalty_hits_fomo():
    calm = pd.DataFrame([_base_row()])
    fomo = pd.DataFrame(
        [
            _base_row(
                rsi14=85.0,
                dist_ema20=0.25,
                dist_ema50=0.40,
                green_streak=8.0,
                ret10=0.40,
            )
        ]
    )
    assert float(score_overextension_penalty(fomo).iloc[0]) > float(
        score_overextension_penalty(calm).iloc[0]
    )


def test_eqs_prefers_clean_setup_over_spike():
    clean = _base_row(symbol="CLEAN", rss=0.55, ret5=0.04, rsi14=58.0)
    spike = _base_row(
        symbol="SPIKE",
        rss=0.75,
        ret5=0.35,
        ret10=0.45,
        rsi14=82.0,
        pullback_pct=0.0,
        dist_ema20=0.22,
        dist_ema50=0.35,
        green_streak=7.0,
        vol_spike_persist=0.40,
        range_pct_5=0.15,
        range_pct_10=0.12,
        range_pct_20=0.10,
        atr_shrink_ratio=1.30,
        ema20_slope5=0.02,
        price=120.0,
        ema20=100.0,
        ema50=90.0,
    )
    df = pd.DataFrame([clean, spike])
    comps = compute_eqs_components(df)
    eqs = combine_eqs(comps, EQSWeights())
    assert float(eqs.iloc[0]) > float(eqs.iloc[1])


def test_rank_universe_blends_eqs_into_final_score():
    eng = StandaloneEngine.__new__(StandaloneEngine)
    eng.w1, eng.w2, eng.w3, eng.w4, eng.w5 = 0.10, 0.10, 0.15, 0.10, 0.25
    eng.eqs_weights = EQSWeights()
    eng.EQS_BLEND_WEIGHT = 0.55
    eng.USE_INSTITUTIONAL_ENTRY = False

    # Similar CMVS momentum, but CLEAN has far better setup health
    clean = _base_row(symbol="CLEAN", bbs=0.5, vzs=0.45, rss=0.55, ret5=0.05, rsi14=57.0)
    spike = _base_row(
        symbol="SPIKE",
        bbs=0.9,
        vzs=0.9,
        rss=0.70,
        ret5=0.40,
        rsi14=84.0,
        pullback_pct=0.0,
        dist_ema20=0.28,
        dist_ema50=0.40,
        green_streak=8.0,
        vol_spike_persist=0.45,
        atr_shrink_ratio=1.4,
        range_pct_5=0.16,
        range_pct_10=0.12,
        range_pct_20=0.09,
        price=130.0,
        ema20=100.0,
        ema50=85.0,
        close_vs_high20=0.99,
        trend_score=0.65,
    )
    ranked = StandaloneEngine.rank_universe(eng, pd.DataFrame([spike, clean]))
    assert "eqs" in ranked.columns and "cmvs_score" in ranked.columns
    assert ranked.iloc[0]["symbol"] == "CLEAN"
    assert ranked.iloc[0]["factor_mode"] == "cmvs_v3_eqs"
    assert float(ranked.iloc[0]["final_score"]) == float(
        blend_cmvs_eqs(
            ranked.iloc[0]["cmvs_score"],
            ranked.iloc[0]["eqs"],
            0.55,
        )
    )


def test_blend_weight_is_meaningful():
    cmvs = pd.Series([0.50, 0.60])
    eqs = pd.Series([0.90, 0.20])
    final = blend_cmvs_eqs(cmvs, eqs, 0.55)
    # Lower CMVS + high EQS should beat higher CMVS + poor EQS
    assert float(final.iloc[0]) > float(final.iloc[1])
