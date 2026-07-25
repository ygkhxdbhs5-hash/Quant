"""Tests for Institutional Entry Engine (academic Z-score multi-factor)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.institutional_entry.composite import CATEGORY_WEIGHTS, build_institutional_scores
from engine.institutional_entry.momentum import momentum_12_1
from engine.institutional_entry.quality import (
    gross_profitability,
    operating_profitability,
    return_on_assets,
    return_on_equity,
)
from engine.institutional_entry.zscore import cross_sectional_zscore
from engine.strategy import StandaloneEngine


def test_cross_sectional_zscore_unit_variance():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = cross_sectional_zscore(s)
    assert abs(float(z.mean())) < 1e-9
    assert abs(float(z.std(ddof=0)) - 1.0) < 1e-9


def test_quality_definitions():
    fund = pd.Series(
        {
            "gross_profitability": 0.25,
            "operating_income": 50.0,
            "total_equity": 200.0,
            "net_income": 40.0,
            "total_assets": 400.0,
        }
    )
    assert gross_profitability(fund) == 0.25
    assert operating_profitability(fund) == 0.25
    assert return_on_equity(fund) == 0.20
    assert return_on_assets(fund) == 0.10
    assert np.isnan(return_on_equity(None))


def test_momentum_12_1_skips_recent_month():
    idx = pd.bdate_range("2020-01-01", periods=300)
    # Linear price path so 12-1 is well-defined
    close = pd.Series(np.linspace(100, 200, 300), index=idx)
    # At end: price[t-21]/price[t-252]-1
    i = 280
    expected = float(close.iloc[i - 21]) / float(close.iloc[i - 252]) - 1.0
    assert abs(momentum_12_1(close, i) - expected) < 1e-12
    assert np.isnan(momentum_12_1(close, 10))


def test_institutional_scores_rank_high_quality_momentum():
    """End-to-end scoring on a tiny synthetic engine."""
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)

    def _path(start, drift):
        rets = drift + rng.normal(0, 0.01, size=n)
        return start * np.cumprod(1.0 + rets)

    close = pd.DataFrame(
        {
            "GOOD": _path(100, 0.002),
            "BAD": _path(100, -0.001),
            "SPY": _path(100, 0.0005),
        },
        index=idx,
    )
    high = close * 1.01
    low = close * 0.99
    dvol = pd.DataFrame(
        {"GOOD": np.full(n, 5e7), "BAD": np.full(n, 1e6), "SPY": np.full(n, 1e9)},
        index=idx,
    )

    eng = StandaloneEngine.__new__(StandaloneEngine)
    eng.close_m = close
    eng.high_m = high
    eng.low_m = low
    eng.dvol_m = dvol
    eng.adv20_m = dvol.rolling(20, min_periods=5).mean()
    eng.ret1_m = close.pct_change()
    eng.ema50_m = close.ewm(span=50, adjust=False).mean()
    eng.ema200_m = close.ewm(span=200, adjust=False).mean()
    eng.BENCHMARK_TICKER = "SPY"
    eng.profile_meta = {
        "GOOD": {"industry": "Tech"},
        "BAD": {"industry": "Tech"},
    }
    eng._adx_cache = {}

    good_fund = pd.Series(
        {
            "gross_profitability": 0.40,
            "operating_income": 80.0,
            "total_equity": 100.0,
            "net_income": 50.0,
            "total_assets": 200.0,
            "diluted_shares_outstanding": 1e6,
        }
    )
    bad_fund = pd.Series(
        {
            "gross_profitability": 0.05,
            "operating_income": 5.0,
            "total_equity": 100.0,
            "net_income": 1.0,
            "total_assets": 200.0,
            "diluted_shares_outstanding": 1e6,
        }
    )

    def _fund(sym, as_of):
        return good_fund if sym == "GOOD" else bad_fund if sym == "BAD" else None

    eng.get_latest_available_fundamentals = _fund

    scored = build_institutional_scores(eng, n - 1, ["GOOD", "BAD"])
    assert len(scored) == 2
    assert scored.iloc[0]["symbol"] == "GOOD"
    assert float(scored.iloc[0]["final_score"]) > float(scored.iloc[1]["final_score"])
    assert scored.iloc[0]["factor_mode"] == "institutional_v1"
    # Weights present
    assert abs(CATEGORY_WEIGHTS["quality"] - 0.35) < 1e-12
    assert abs(CATEGORY_WEIGHTS["risk"] + 0.05) < 1e-12


def test_rank_universe_institutional_passthrough():
    eng = StandaloneEngine.__new__(StandaloneEngine)
    eng.USE_INSTITUTIONAL_ENTRY = True
    df = pd.DataFrame(
        [
            {"symbol": "A", "final_score": 0.5, "factor_mode": "institutional_v1", "industry": "X"},
            {"symbol": "B", "final_score": 1.2, "factor_mode": "institutional_v1", "industry": "X"},
        ]
    )
    ranked = StandaloneEngine.rank_universe(eng, df)
    assert ranked.iloc[0]["symbol"] == "B"
