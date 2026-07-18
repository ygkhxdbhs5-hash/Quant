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
        self.MAX_INDUSTRY_WEIGHT = 0.20
        self.MIN_INDUSTRY_SIZE = 8
        self.VOL_FLOOR = 1e-4
        self.WEIGHT_SUM_TOLERANCE = 0.02
        self.CORR_WINDOW = 60
        self.CORR_THRESHOLD = 0.80
        self.MIN_ROIC = 0.10
        self.MIN_FCF_SALES_YIELD = 0.0
        self.MAX_DEBT_TO_EQUITY = 1.50
        self.MIN_REVENUE_GROWTH_YOY = 0.0
        self.VOL_BREAKOUT_ATR_MULT = 1.5
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


def test_get_universe_quality_and_value_filters():
    eng = _TinyEngine()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    eng.close_m = pd.DataFrame(
        {
            "GOOD": [100.0, 101.0],
            "LOW_ROIC": [50.0, 51.0],
            "NEG_FCF": [80.0, 81.0],
            "APPROX": [20.0, 21.0],
            "HIGH_DEBT": [40.0, 41.0],
            "NEG_GROWTH": [60.0, 61.0],
        },
        index=dates,
    )

    def _series(roic, revenues, op_cf, capex, debt, equity, shares=np.nan):
        rows = []
        for i, rev in enumerate(revenues):
            rows.append(
                {
                    "roic": roic,
                    "revenue": rev,
                    "op_cf": op_cf if i == len(revenues) - 1 else op_cf,
                    "capex": capex,
                    "total_debt": debt,
                    "total_equity": equity,
                    "debt_to_equity": debt / equity if equity else np.nan,
                    "diluted_shares_outstanding": shares,
                    "basic_shares_outstanding": shares,
                }
            )
        return _pit_rows(rows)

    # 5 quarterly points so YoY (shift 4 / ~1y) is defined on the last row
    eng.fundamental_history = {
        "GOOD": _series(0.15, [80, 85, 90, 95, 100], 40.0, 10.0, 50.0, 100.0),
        "LOW_ROIC": _series(0.05, [80, 85, 90, 95, 100], 40.0, 10.0, 50.0, 100.0),
        "NEG_FCF": _series(0.20, [80, 85, 90, 95, 100], 5.0, 20.0, 50.0, 100.0),
        "APPROX": _series(0.12, [80, 85, 90, 95, 100], np.nan, np.nan, 40.0, 100.0, shares=10.0),
        "HIGH_DEBT": _series(0.15, [80, 85, 90, 95, 100], 40.0, 10.0, 200.0, 100.0),  # D/E=2.0
        "NEG_GROWTH": _series(0.15, [120, 110, 105, 100, 90], 40.0, 10.0, 50.0, 100.0),
    }

    out = eng.get_universe(
        ["GOOD", "LOW_ROIC", "NEG_FCF", "APPROX", "HIGH_DEBT", "NEG_GROWTH"],
        date_idx=1,
    )
    assert out == ["GOOD", "APPROX"]


def test_rank_universe_weights():
    eng = _TinyEngine()
    df = pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(12)],
            "mom_raw": np.linspace(0.1, 0.5, 12),
            "vol_raw": np.linspace(0.01, 0.05, 12),
            "vol_expansion_raw": np.linspace(-0.5, 2.0, 12),
            "op_margin_raw": np.linspace(0.1, 0.3, 12),
            "roic_raw": np.linspace(0.1, 0.3, 12),
            "gross_prof_raw": np.linspace(0.1, 0.3, 12),
            "industry": ["A"] * 6 + ["B"] * 6,
        }
    )
    ranked = eng.rank_universe(df)
    assert "final_score" in ranked.columns
    assert "rank_vol_expansion" in ranked.columns
    assert ranked["final_score"].iloc[0] >= ranked["final_score"].iloc[-1]


def test_rank_universe_price_volume_fallback():
    eng = _TinyEngine()
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "mom_raw": [0.4, 0.3, 0.2, 0.1],
            "vol_raw": [0.02, 0.03, 0.04, 0.05],
            "vol_expansion_raw": [1.5, 0.5, 2.0, 0.0],
            "op_margin_raw": [0.2, np.nan, 0.15, np.nan],
            "roic_raw": [0.2, np.nan, 0.15, np.nan],
            "gross_prof_raw": [0.2, np.nan, 0.15, np.nan],
            "industry": ["X", "X", "Y", "Y"],
        }
    )
    ranked = eng.rank_universe(df)
    assert set(ranked["factor_mode"]) == {"full", "price_volume_fallback"}
    assert ranked["final_score"].notna().all()
    assert len(ranked) == 4


def test_rank_universe_vol_expansion_boosts_score():
    """Higher vol expansion should lift final_score when mom/quality are tied."""
    eng = _TinyEngine()
    df = pd.DataFrame(
        {
            "symbol": ["LOW_EXP", "HIGH_EXP"],
            "mom_raw": [0.3, 0.3],
            "vol_raw": [0.02, 0.02],
            "vol_expansion_raw": [0.1, 2.0],
            "op_margin_raw": [0.2, 0.2],
            "roic_raw": [0.2, 0.2],
            "gross_prof_raw": [0.2, 0.2],
            "industry": ["X", "X"],
        }
    )
    ranked = eng.rank_universe(df)
    assert ranked.iloc[0]["symbol"] == "HIGH_EXP"


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


def test_trailing_stop_exits_and_resets_on_repurchase():
    eng = _TinyEngine()
    eng.TRAILING_STOP_PCT = 0.10
    eng.highest_prices = {}
    eng.cash = 0.0
    eng.portfolio = {"AAA": 10}
    eng.previous_target_symbols = {"AAA"}
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    eng.close_m = pd.DataFrame({"AAA": [100.0, 96.0, 89.0]}, index=dates)
    eng.high_m = pd.DataFrame({"AAA": [100.0, 100.0, 95.0]}, index=dates)

    # Day 0: establish peak at 100
    eng.check_trailing_stops(0)
    assert eng.portfolio.get("AAA") == 10
    assert eng.highest_prices["AAA"] == 100.0

    # Day 2: close 89 is >10% below peak 100 -> immediate exit
    eng.check_trailing_stops(2)
    assert "AAA" not in eng.portfolio
    assert "AAA" not in eng.highest_prices
    assert eng.cash == pytest.approx(890.0)

    # Re-purchase resets peak to new entry
    eng.portfolio["AAA"] = 5
    eng.highest_prices["AAA"] = 50.0  # simulate execute_pending_orders reset
    eng.check_trailing_stops(2)  # high=95 raises peak; close=89 within 10% of 95
    assert eng.portfolio.get("AAA") == 5
    assert eng.highest_prices["AAA"] == 95.0


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
