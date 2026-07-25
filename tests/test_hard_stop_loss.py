"""Tests for optional hard % stop loss from entry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.research.config_toggles import ResearchToggles, load_research_toggles
from engine.strategy import StandaloneEngine


def _exit_engine(n: int = 5) -> StandaloneEngine:
    eng = StandaloneEngine.__new__(StandaloneEngine)
    eng.USE_EMA9_EXIT = False
    eng.USE_ATR_EXIT = False
    eng.USE_EXHAUSTION_EXIT = False
    eng.USE_TIME_STOP = False
    eng.USE_STOP_LOSS = True
    eng.STOP_LOSS_PCT = 0.20
    eng.MIN_HOLD_DAYS = 0
    eng.EMA_EXIT_LENGTH = 9
    eng.atr_multiplier = 2.0
    eng.highest_prices = {}
    eng.portfolio = {"AAA": 10}
    eng.previous_target_symbols = {"AAA"}
    eng.pending_orders = []
    eng._cmvs_forced_exits = set()

    class _OT:
        entry_price = 100.0

    eng.trade_journal = type(
        "J",
        (),
        {
            "open": {"AAA": _OT()},
            "mark_peak": lambda *a, **k: None,
            "holding_days": lambda *a, **k: 0,
        },
    )()

    idx = pd.bdate_range("2024-01-02", periods=n)
    close = np.array([100.0, 95.0, 90.0, 85.0, 79.0], dtype=float)[:n]
    eng.close_m = pd.DataFrame({"AAA": close}, index=idx)
    eng.atr14_m = pd.DataFrame({"AAA": np.full(n, 2.0)}, index=idx)
    eng.rsi14_m = pd.DataFrame({"AAA": np.full(n, 55.0)}, index=idx)
    eng.daily_cp_m = pd.DataFrame({"AAA": np.full(n, 0.5)}, index=idx)
    eng.ema_exit_m = pd.DataFrame({"AAA": close * 0.99}, index=idx)
    eng.ema20_m = pd.DataFrame({"AAA": close * 0.98}, index=idx)
    eng.sma50_m = pd.DataFrame({"AAA": close * 0.97}, index=idx)
    eng.sma50_slope5_m = pd.DataFrame({"AAA": np.full(n, 0.01)}, index=idx)
    eng.atr_pct_m = pd.DataFrame({"AAA": np.full(n, 0.04)}, index=idx)
    eng.ret1_m = pd.DataFrame({"AAA": eng.close_m["AAA"].pct_change()}, index=idx)
    eng.ret5_m = pd.DataFrame({"AAA": np.full(n, 0.0)}, index=idx)
    eng.rel_vol_m = pd.DataFrame({"AAA": np.full(n, 1.0)}, index=idx)
    return eng


def test_stop_loss_off_never_fires():
    eng = _exit_engine()
    eng.USE_STOP_LOSS = False
    eng.check_cmvs_exits(len(eng.close_m) - 1)  # close 79 = -21% from 100
    assert eng.pending_orders == []


def test_stop_loss_fires_at_threshold():
    eng = _exit_engine()
    # Day with close 79 <= 100 * 0.80 = 80
    eng.check_cmvs_exits(len(eng.close_m) - 1)
    assert len(eng.pending_orders) == 1
    assert "stop_loss" in eng.pending_orders[0]["reason"]


def test_stop_loss_does_not_fire_above_threshold():
    eng = _exit_engine()
    # close 85 > 80
    eng.check_cmvs_exits(3)
    assert eng.pending_orders == []


def test_stop_loss_bypasses_min_hold():
    eng = _exit_engine()
    eng.MIN_HOLD_DAYS = 30
    eng.trade_journal.holding_days = lambda *a, **k: 1
    eng.check_cmvs_exits(len(eng.close_m) - 1)
    assert len(eng.pending_orders) == 1
    assert "stop_loss" in eng.pending_orders[0]["reason"]


def test_stop_loss_pct_clamped_in_loader():
    t = load_research_toggles({"research": {"USE_STOP_LOSS": True, "STOP_LOSS_PCT": 0.99}})
    assert t.USE_STOP_LOSS is True
    assert t.STOP_LOSS_PCT == 0.50
    panel = ResearchToggles(USE_STOP_LOSS=True, STOP_LOSS_PCT=0.15).format_panel()
    assert "USE_STOP_LOSS = True" in panel
    assert "STOP_LOSS_PCT = 15.00%" in panel
