"""Unit tests for Baseline v1 strategy module (no full data required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy_baseline_v1 import (
    ATR_MULTIPLIER,
    TOP_LIQUID_POOL,
    TOP_MOMENTUM_COUNT,
    atr_stop_line,
    atr_stop_triggered,
    compute_mom_12_1,
    compute_mom_12_1_panel,
    equal_weight_targets,
    refill_from_candidates,
    select_monthly_candidates,
)


def test_mom_12_1_exact_definition():
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(np.linspace(100, 200, 300), index=idx)
    i = 280
    expected = float(close.iloc[i - 21]) / float(close.iloc[i - 252]) - 1.0
    assert abs(compute_mom_12_1(close, i) - expected) < 1e-12
    panel = compute_mom_12_1_panel(close.to_frame("A"))
    assert abs(float(panel["A"].iloc[i]) - expected) < 1e-12


def test_atr_stop_and_trigger():
    assert abs(atr_stop_line(100.0, 2.0, 2.5) - (100.0 - 5.0)) < 1e-12
    assert atr_stop_triggered(94.9, 100.0, 2.0, 2.5) is True
    assert atr_stop_triggered(95.1, 100.0, 2.0, 2.5) is False
    assert ATR_MULTIPLIER == 2.5


def test_equal_weight_and_refill_keeps_holdings():
    w = equal_weight_targets(["A", "B", "C"], exposure=1.0)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert all(abs(v - 1.0 / 3) < 1e-12 for v in w.values())

    # Holding outside top candidates is kept; empties filled from candidates
    held = {"OLD"}
    candidates = ["N1", "N2", "N3"]
    selected = refill_from_candidates(held, candidates, max_positions=3)
    assert selected[0] == "OLD"
    assert "N1" in selected and "N2" in selected
    assert len(selected) == 3


def test_select_monthly_candidates_liquidity_then_momentum():
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    # Three names; HIGH_LIQ_HIGH_MOM should win
    close = pd.DataFrame(
        {
            "A": np.linspace(100, 200, n),  # strong mom
            "B": np.linspace(100, 110, n),  # weak mom
            "C": np.linspace(100, 180, n),  # mid mom
        },
        index=idx,
    )
    dvol = pd.DataFrame(
        {"A": np.full(n, 1e8), "B": np.full(n, 1e8), "C": np.full(n, 1e3)},  # C illiquid
        index=idx,
    )
    mom = compute_mom_12_1_panel(close)
    out = select_monthly_candidates(
        date_idx=n - 1,
        close_m=close,
        dvol_m=dvol,
        mom_12_1_m=mom,
        eligible_symbols=["A", "B", "C"],
        top_liquid_pool=2,
        top_momentum_count=1,
    )
    assert out == ["A"]
    assert TOP_LIQUID_POOL == 250
    assert TOP_MOMENTUM_COUNT == 30
