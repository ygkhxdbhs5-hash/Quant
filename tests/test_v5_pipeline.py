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
        self.previous_target_symbols = set()
        self.profile_meta = {}
        self.close_m = pd.DataFrame()


def test_rank_universe_weights():
    eng = _TinyEngine()
    df = pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(12)],
            "mom_raw": np.linspace(0.1, 0.5, 12),
            "vol_raw": np.linspace(0.01, 0.05, 12),
            "op_margin_raw": np.linspace(0.1, 0.3, 12),
            "roic_raw": np.linspace(0.1, 0.3, 12),
            "gross_prof_raw": np.linspace(0.1, 0.3, 12),
            "industry": ["A"] * 6 + ["B"] * 6,
        }
    )
    ranked = eng.rank_universe(df)
    assert "final_score" in ranked.columns
    assert ranked["final_score"].iloc[0] >= ranked["final_score"].iloc[-1]


def test_rank_universe_price_volume_fallback():
    eng = _TinyEngine()
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "mom_raw": [0.4, 0.3, 0.2, 0.1],
            "vol_raw": [0.02, 0.03, 0.04, 0.05],
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
