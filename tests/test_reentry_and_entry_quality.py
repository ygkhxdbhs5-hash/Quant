"""Tests for adaptive re-entry cooldown and entry-quality ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.reentry_cooldown import AdaptiveReentryCooldown
from engine.strategy import StandaloneEngine


def test_cooldown_arms_only_on_losses():
    cd = AdaptiveReentryCooldown(enabled=True)
    assert cd.record_exit(
        symbol="AAA",
        date_idx=100,
        exit_price=10.0,
        exit_reason="ema9_break",
        final_return=0.05,
    ) is None
    assert "AAA" not in cd._records

    rec = cd.record_exit(
        symbol="CURX",
        date_idx=100,
        exit_price=5.0,
        exit_reason="atr_trail(stop=5.2)",
        final_return=-0.18,
        atr_pct=0.12,
        vol20=0.06,
        peak_return=0.10,
    )
    assert rec is not None
    assert rec.consecutive_losses == 1
    assert rec.hard_expiry_idx > 100 + 10
    assert cd.is_blocked("CURX", 105, snap=None) is True


def test_cooldown_stacks_on_repeated_losses():
    cd = AdaptiveReentryCooldown(enabled=True)
    r1 = cd.record_exit(
        symbol="HKIT",
        date_idx=50,
        exit_price=4.0,
        exit_reason="ema9_break",
        final_return=-0.10,
        atr_pct=0.10,
    )
    r2 = cd.record_exit(
        symbol="HKIT",
        date_idx=70,
        exit_price=3.5,
        exit_reason="atr_trail(stop=3.6)",
        final_return=-0.12,
        atr_pct=0.11,
    )
    assert r2.consecutive_losses == 2
    assert (r2.hard_expiry_idx - 70) > (r1.hard_expiry_idx - 50)


def test_cooldown_clears_only_with_renewed_strength():
    cd = AdaptiveReentryCooldown(enabled=True)
    rec = cd.record_exit(
        symbol="JEM",
        date_idx=100,
        exit_price=8.0,
        exit_reason="ema9_break",
        final_return=-0.15,
        atr_pct=0.09,
    )
    # Still inside min holdout — blocked even with strong snap
    strong = {
        "close": 9.0,
        "exit_price": 8.0,
        "sma20": 8.5,
        "sma50": 8.2,
        "rsi14": 55.0,
        "ret5": 0.04,
        "close_vs_high20": 0.92,
        "atr_pct": 0.06,
    }
    assert cd.is_blocked("JEM", rec.min_holdout_idx - 1, snap=strong) is True

    # After min holdout + strength → early clear
    assert cd.is_blocked("JEM", rec.min_holdout_idx + 1, snap=strong) is False
    assert "JEM" not in cd._records


def test_cooldown_stays_blocked_without_strength():
    cd = AdaptiveReentryCooldown(enabled=True)
    rec = cd.record_exit(
        symbol="CURX",
        date_idx=100,
        exit_price=5.0,
        exit_reason="atr_trail(stop=5.1)",
        final_return=-0.20,
        atr_pct=0.14,
    )
    weak = {
        "close": 4.5,
        "exit_price": 5.0,
        "sma20": 5.2,
        "sma50": 5.5,
        "rsi14": 35.0,
        "ret5": -0.08,
        "close_vs_high20": 0.70,
        "atr_pct": 0.15,
    }
    assert cd.is_blocked("CURX", rec.min_holdout_idx + 2, snap=weak) is True


def test_rank_universe_penalizes_spike_and_prefers_structure():
    """Pump-like row should rank below sustainable RS/trend row."""
    eng = StandaloneEngine.__new__(StandaloneEngine)
    eng.w1, eng.w2, eng.w3, eng.w4, eng.w5 = 0.10, 0.10, 0.15, 0.10, 0.25

    df = pd.DataFrame(
        [
            {
                "symbol": "PUMP",
                "bbs": 0.95,
                "vzs": 0.95,
                "cps": 0.40,
                "rsis": 0.95,
                "rss": 0.40,
                "ret5": 0.45,
                "rsi14": 82.0,
                "trend_score": 0.05,
                "up_frac20": 0.35,
                "atr_pct": 0.14,
                "close_vs_high20": 0.70,
                "industry": "X",
            },
            {
                "symbol": "QUALITY",
                "bbs": 0.55,
                "vzs": 0.50,
                "cps": 0.70,
                "rsis": 0.55,
                "rss": 0.85,
                "ret5": 0.06,
                "rsi14": 58.0,
                "trend_score": 1.0,
                "up_frac20": 0.65,
                "atr_pct": 0.05,
                "close_vs_high20": 0.95,
                "industry": "X",
            },
        ]
    )
    ranked = StandaloneEngine.rank_universe(eng, df)
    assert ranked.iloc[0]["symbol"] == "QUALITY"
    assert ranked.iloc[0]["final_score"] > ranked.iloc[1]["final_score"]
    assert ranked.iloc[0]["factor_mode"] == "cmvs_v3_quality"
