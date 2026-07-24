"""Unit tests for v5 ranking / risk sizing (investment formulas)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.strategy import RebalanceContext, StandaloneEngine


class _TinyEngine(StandaloneEngine):
    """Bypass data loading; only exercise pure pipeline methods."""

    def __init__(self):
        # Skip StandaloneEngine.__init__ data load
        self.MAX_PORTFOLIO_SIZE = 5
        self.SELECTION_BUFFER_SIZE = 8
        self.ENTRY_RANK = 5
        self.EXIT_RANK = 8
        self.MAX_INDUSTRY_WEIGHT = 0.20
        self.MIN_INDUSTRY_SIZE = 8
        self.VOL_FLOOR = 1e-4
        self.WEIGHT_SUM_TOLERANCE = 0.02
        self.CORR_WINDOW = 60
        self.CORR_THRESHOLD = 0.80
        self.MIN_ROIC = 0.03
        self.MIN_FCF_SALES_YIELD = -0.05
        self.MAX_DEBT_TO_EQUITY = 3.0
        self.MIN_REVENUE_GROWTH_YOY = -0.15
        self.VOL_BREAKOUT_ATR_MULT = 1.5
        self.MIN_ENTRY_PRICE = 3.0
        self.MAX_ATR_PCT_ENTRY = 0.15
        self.MAX_SPIKE_5D = 0.35
        self.MIN_CLOSE_VS_HIGH20 = 0.60
        self.w1 = 0.10
        self.w2 = 0.10
        self.w3 = 0.15
        self.w4 = 0.10
        self.w5 = 0.25
        from engine.entry_quality import EQSWeights

        self.eqs_weights = EQSWeights()
        self.EQS_BLEND_WEIGHT = 0.55
        self.previous_target_symbols = set()
        self.profile_meta = {}
        self.close_m = pd.DataFrame()
        self.open_m = pd.DataFrame()
        self.atr20_m = pd.DataFrame()
        self.mom_12_1_m = pd.DataFrame()
        self.vol60_m = pd.DataFrame()
        self.fundamental_history = {}


def _pit_rows(rows, start="2022-01-01"):
    """Build a small PIT frame; each row is a dict of fields."""
    idx = pd.date_range(start, periods=len(rows), freq="90D")
    return pd.DataFrame(rows, index=idx)


def test_get_universe_is_price_volume_passthrough():
    """CMVS v3: get_universe no longer hard-rejects on fundamentals."""
    eng = _TinyEngine()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    eng.close_m = pd.DataFrame(
        {
            "GOOD": [100.0, 101.0],
            "LOW_ROIC": [50.0, 51.0],
            "NEG_FCF": [80.0, 81.0],
        },
        index=dates,
    )
    eng.fundamental_history = {}
    out = eng.get_universe(["GOOD", "LOW_ROIC", "NEG_FCF"], date_idx=1)
    assert set(out) == {"GOOD", "LOW_ROIC", "NEG_FCF"}


def _cmvs_row(sym, **overrides):
    base = {
        "symbol": sym,
        "bbs": 0.5,
        "vzs": 0.5,
        "cps": 0.5,
        "rsis": 0.5,
        "rss": 0.5,
        "ret5": 0.05,
        "ret10": 0.08,
        "rsi14": 55.0,
        "trend_score": 0.65,
        "up_frac20": 0.55,
        "atr_pct": 0.05,
        "close_vs_high20": 0.92,
        "price": 100.0,
        "ema20": 98.0,
        "ema50": 95.0,
        "ema20_slope5": 0.01,
        "ema50_slope5": 0.008,
        "pullback_pct": 0.08,
        "pullback_vol_ratio": 0.8,
        "vol_quiet_ratio": 0.85,
        "vol_expand_ratio": 1.2,
        "vol_spike_persist": 0.05,
        "range_pct_5": 0.05,
        "range_pct_10": 0.07,
        "range_pct_20": 0.10,
        "atr_shrink_ratio": 0.9,
        "rs_raw": 0.05,
        "rs_slope5": 0.01,
        "rs_near_high60": 0.8,
        "dist_ema20": 0.02,
        "dist_ema50": 0.05,
        "above_ema50": 1.0,
        "green_streak": 2.0,
        "industry": "X",
    }
    base.update(overrides)
    return base


def test_rank_universe_quality_orders_by_final_score():
    eng = _TinyEngine()
    df = pd.DataFrame(
        [
            _cmvs_row("WEAK", rss=0.2, trend_score=0.05, ret5=0.40, rsi14=80.0, atr_pct=0.12),
            _cmvs_row("STRONG", rss=0.9, trend_score=1.0, ret5=0.04, rsi14=58.0, atr_pct=0.04),
            _cmvs_row("MID", rss=0.5, trend_score=0.65, ret5=0.08, rsi14=60.0, atr_pct=0.06),
        ]
    )
    ranked = eng.rank_universe(df)
    assert "final_score" in ranked.columns
    assert ranked["final_score"].iloc[0] >= ranked["final_score"].iloc[-1]
    assert ranked.iloc[0]["symbol"] == "STRONG"
    assert ranked.iloc[0]["factor_mode"] == "cmvs_v3_eqs"


def test_rank_universe_penalizes_pump_signature():
    eng = _TinyEngine()
    df = pd.DataFrame(
        [
            _cmvs_row(
                "PUMP",
                bbs=0.95,
                vzs=0.95,
                rsis=0.95,
                rss=0.45,
                ret5=0.40,
                rsi14=85.0,
                trend_score=0.05,
                up_frac20=0.30,
                atr_pct=0.13,
                close_vs_high20=0.68,
                pullback_pct=0.0,
                dist_ema20=0.25,
                dist_ema50=0.40,
                green_streak=7.0,
                atr_shrink_ratio=1.3,
                price=50.0,
                ema20=40.0,
                ema50=30.0,
            ),
            _cmvs_row(
                "QUALITY",
                bbs=0.55,
                vzs=0.45,
                rsis=0.55,
                rss=0.80,
                ret5=0.05,
                rsi14=57.0,
                trend_score=1.0,
                up_frac20=0.60,
                atr_pct=0.04,
                close_vs_high20=0.94,
                pullback_pct=0.08,
                dist_ema20=0.02,
                dist_ema50=0.05,
                green_streak=2.0,
                atr_shrink_ratio=0.85,
                price=100.0,
                ema20=98.0,
                ema50=95.0,
                ema20_slope5=0.01,
                ema50_slope5=0.008,
            ),
        ]
    )
    ranked = eng.rank_universe(df)
    assert ranked.iloc[0]["symbol"] == "QUALITY"


def test_rank_universe_rss_dominates_when_structure_tied():
    eng = _TinyEngine()
    df = pd.DataFrame(
        [
            _cmvs_row("LOW_RS", rss=0.2, trend_score=1.0, rs_raw=-0.05),
            _cmvs_row("HIGH_RS", rss=0.95, trend_score=1.0, rs_raw=0.20, rs_slope5=0.04, rs_near_high60=0.98),
        ]
    )
    ranked = eng.rank_universe(df)
    assert ranked.iloc[0]["symbol"] == "HIGH_RS"
    assert ranked.iloc[0]["factor_mode"] == "cmvs_v3_eqs"


def test_apply_risk_adjustments_sums_to_exposure():
    eng = _TinyEngine()
    targets = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "vol_raw": [0.02, 0.03, 0.04],
            "final_score": [1.0, 0.9, 0.8],
            "industry": ["X", "X", "Y"],
        }
    )
    sized = eng.apply_risk_adjustments(targets, exposure=0.5)
    assert sized["final_weight"].sum() == pytest.approx(0.5, abs=1e-10)


@pytest.mark.skip(reason="Legacy % trailing stop replaced by CMVS ATR/EMA/exhaustion exits")
def test_trailing_stop_exits_and_resets_on_repurchase():
    eng = _TinyEngine()
    eng.TRAILING_STOP_PCT = 0.20
    eng.highest_prices = {}
    eng.cash = 0.0
    eng.portfolio = {"AAA": 10}
    eng.previous_target_symbols = {"AAA"}
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    eng.close_m = pd.DataFrame({"AAA": [100.0, 96.0, 79.0]}, index=dates)
    eng.high_m = pd.DataFrame({"AAA": [100.0, 100.0, 95.0]}, index=dates)

    # Day 0: establish peak at 100
    eng.check_trailing_stops(0)
    assert eng.portfolio.get("AAA") == 10
    assert eng.highest_prices["AAA"] == 100.0

    # Day 2: close 79 is >20% below peak 100 -> immediate exit
    eng.check_trailing_stops(2)
    assert "AAA" not in eng.portfolio
    assert "AAA" not in eng.highest_prices
    assert eng.cash == pytest.approx(790.0)

    # Re-purchase resets peak to new entry
    eng.portfolio["AAA"] = 5
    eng.highest_prices["AAA"] = 50.0  # simulate execute_pending_orders reset
    eng.check_trailing_stops(2)  # high=95 raises peak; close=79 within 20% of 95
    assert eng.portfolio.get("AAA") == 5
    assert eng.highest_prices["AAA"] == 95.0


@pytest.mark.skip(reason="build_factors is CMVS v3 (BBS/VZS/CPS/RSIS/RSS); vol_expansion_raw removed")
def test_build_factors_vol_expansion_score():
    """All names kept; vol_expansion_raw = (close - open) / ATR20."""
    eng = _TinyEngine()
    dates = pd.to_datetime(["2024-06-03", "2024-06-04"])
    eng.close_m = pd.DataFrame(
        {"BREAK": [10.0, 14.0], "QUIET": [10.0, 10.5]},
        index=dates,
    )
    eng.open_m = pd.DataFrame(
        {"BREAK": [10.0, 10.0], "QUIET": [10.0, 10.0]},
        index=dates,
    )
    eng.atr20_m = pd.DataFrame(
        {"BREAK": [2.0, 2.0], "QUIET": [2.0, 2.0]},
        index=dates,
    )
    eng.mom_12_1_m = pd.DataFrame(
        {"BREAK": [0.2, 0.3], "QUIET": [0.4, 0.5]},
        index=dates,
    )
    eng.vol60_m = pd.DataFrame(
        {"BREAK": [0.02, 0.02], "QUIET": [0.02, 0.02]},
        index=dates,
    )
    eng.fundamental_history = {}
    eng.profile_meta = {s: {"industry": "X"} for s in ["BREAK", "QUIET"]}

    df = eng.build_factors(1, ["BREAK", "QUIET"]).set_index("symbol")
    assert set(df.index) == {"BREAK", "QUIET"}
    assert df.loc["BREAK", "vol_expansion_raw"] == pytest.approx(2.0)  # (14-10)/2
    assert df.loc["QUIET", "vol_expansion_raw"] == pytest.approx(0.25)  # (10.5-10)/2


def test_allocate_weights_equal_weight():
    """Aggressive config: equal weights (inverse-ATR path commented out in engine)."""
    eng = _TinyEngine()
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    eng.close_m = pd.DataFrame(
        {"A": [100.0, 100.0, 100.0], "B": [100.0, 100.0, 100.0], "C": [100.0, 100.0, 100.0]},
        index=dates,
    )
    eng.atr20_m = pd.DataFrame(
        {"A": [1.0, 1.0, 1.0], "B": [2.0, 2.0, 2.0], "C": [4.0, 4.0, 4.0]},
        index=dates,
    )
    targets = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "vol_raw": [0.02, 0.03, 0.04],
            "final_score": [1.0, 0.9, 0.8],
            "industry": ["X", "X", "Y"],
        }
    )
    sized = eng.allocate_weights(targets, date_idx=2, exposure=1.0)
    assert sized["final_weight"].sum() == pytest.approx(1.0, abs=1e-10)
    assert sized["final_weight"].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-10)
