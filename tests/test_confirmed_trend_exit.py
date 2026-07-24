"""Tests for confirmation-based trend exit (EMA9 alone never sells)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import StandaloneEngine


def _exit_engine(n: int = 8) -> StandaloneEngine:
    eng = StandaloneEngine.__new__(StandaloneEngine)
    eng.USE_EMA9_EXIT = True
    eng.USE_ATR_EXIT = True
    eng.USE_EXHAUSTION_EXIT = False
    eng.USE_TIME_STOP = False
    eng.MIN_HOLD_DAYS = 0
    eng.EMA_EXIT_LENGTH = 9
    eng.atr_multiplier = 2.0
    eng.HIGH_ATR_PCT_EXIT = 0.08
    eng.TREND_EXIT_MIN_CONFIRM = 2
    eng.TREND_EXIT_MIN_CONFIRM_HIGH_VOL = 3
    eng.PROTECT_HEALTHY_TREND_PULLBACK = True
    eng.highest_prices = {}
    eng.portfolio = {"AAA": 10}
    eng.previous_target_symbols = {"AAA"}
    eng.pending_orders = []
    eng._cmvs_forced_exits = set()
    eng.trade_journal = type("J", (), {"mark_peak": lambda *a, **k: None, "holding_days": lambda *a, **k: 10})()

    idx = pd.bdate_range("2024-01-02", periods=n)
    # Default: mild uptrend then shallow dip below ema9 only on last day
    close = np.array([100, 101, 102, 103, 104, 103, 102, 101], dtype=float)[:n]
    eng.close_m = pd.DataFrame({"AAA": close}, index=idx)
    eng.ema_exit_m = pd.DataFrame({"AAA": close * 0.995}, index=idx)  # close usually above
    eng.ema9_m = eng.ema_exit_m
    eng.ema20_m = pd.DataFrame({"AAA": close * 0.98}, index=idx)
    eng.sma50_m = pd.DataFrame({"AAA": close * 0.97}, index=idx)
    eng.sma50_slope5_m = pd.DataFrame({"AAA": np.full(n, 0.01)}, index=idx)
    eng.rsi14_m = pd.DataFrame({"AAA": np.full(n, 55.0)}, index=idx)
    eng.daily_cp_m = pd.DataFrame({"AAA": np.full(n, 0.5)}, index=idx)
    eng.atr14_m = pd.DataFrame({"AAA": np.full(n, 2.0)}, index=idx)
    eng.atr_pct_m = pd.DataFrame({"AAA": np.full(n, 0.04)}, index=idx)
    eng.ret1_m = pd.DataFrame({"AAA": eng.close_m["AAA"].pct_change()}, index=idx)
    eng.ret5_m = pd.DataFrame({"AAA": np.full(n, 0.02)}, index=idx)
    eng.rel_vol_m = pd.DataFrame({"AAA": np.full(n, 1.0)}, index=idx)
    return eng


def test_single_ema9_pierce_does_not_sell():
    eng = _exit_engine()
    # Last two days: only TODAY below short EMA; yesterday still above → no two_closes
    eng.ema_exit_m.iloc[-1, 0] = 102.0  # close 101 < 102
    eng.ema_exit_m.iloc[-2, 0] = 100.0  # close 102 > 100
    eng.ema20_m.iloc[-1, 0] = 95.0      # still above ema20
    eng.highest_prices["AAA"] = 104.0
    # Disable pullback protection so we only test confirmation count
    eng.PROTECT_HEALTHY_TREND_PULLBACK = False
    eng.rsi14_m.iloc[-1, 0] = 52.0
    eng.ret5_m.iloc[-1, 0] = 0.01
    eng.check_cmvs_exits(len(eng.close_m) - 1)
    assert eng.pending_orders == []


def test_two_closes_below_ema_plus_rsi_sells():
    eng = _exit_engine()
    eng.PROTECT_HEALTHY_TREND_PULLBACK = False
    # Both days below short EMA
    eng.ema_exit_m.iloc[-1, 0] = 110.0
    eng.ema_exit_m.iloc[-2, 0] = 110.0
    eng.ema20_m.iloc[-1, 0] = 95.0  # still above ema20 — not required for normal vol
    eng.rsi14_m.iloc[-1, 0] = 40.0  # confirmation
    eng.ret5_m.iloc[-1, 0] = 0.02
    eng.highest_prices["AAA"] = 104.0
    eng.check_cmvs_exits(len(eng.close_m) - 1)
    assert len(eng.pending_orders) == 1
    assert "trend_confirm" in eng.pending_orders[0]["reason"]
    assert "ema9_break" not in eng.pending_orders[0]["reason"]


def test_healthy_trend_pullback_skips_trend_exit():
    eng = _exit_engine()
    eng.PROTECT_HEALTHY_TREND_PULLBACK = True
    # Would otherwise confirm (2 closes below + RSI)
    eng.ema_exit_m.iloc[-1, 0] = 110.0
    eng.ema_exit_m.iloc[-2, 0] = 110.0
    eng.rsi14_m.iloc[-1, 0] = 42.0
    eng.sma50_m.iloc[-1, 0] = 95.0  # close 101 > sma50
    eng.sma50_slope5_m.iloc[-1, 0] = 0.01
    eng.highest_prices["AAA"] = 104.0  # shallow DD
    eng.check_cmvs_exits(len(eng.close_m) - 1)
    assert eng.pending_orders == []


def test_atr_trail_still_sells_alone():
    eng = _exit_engine()
    eng.highest_prices["AAA"] = 120.0  # stop = 120 - 2*2 = 116; close 101 << stop
    eng.PROTECT_HEALTHY_TREND_PULLBACK = True
    # No trend confirmations
    eng.ema_exit_m.iloc[-1, 0] = 90.0
    eng.ema20_m.iloc[-1, 0] = 90.0
    eng.rsi14_m.iloc[-1, 0] = 60.0
    eng.check_cmvs_exits(len(eng.close_m) - 1)
    assert len(eng.pending_orders) == 1
    assert "atr_trail" in eng.pending_orders[0]["reason"]


def test_high_vol_requires_ema20_and_three_confirms():
    eng = _exit_engine()
    eng.PROTECT_HEALTHY_TREND_PULLBACK = False
    eng.atr_pct_m.iloc[:] = 0.12  # high vol
    # Two closes below ema20 + RSI but NOT neg mom / vol → only 2 tags if ema20_break
    # two_closes_below_ema20 + ema20_break + rsi = 3
    eng.ema20_m.iloc[-1, 0] = 110.0
    eng.ema20_m.iloc[-2, 0] = 110.0
    eng.ema_exit_m.iloc[-1, 0] = 110.0
    eng.ema_exit_m.iloc[-2, 0] = 110.0
    eng.rsi14_m.iloc[-1, 0] = 40.0
    eng.ret5_m.iloc[-1, 0] = 0.05  # not negative
    eng.rel_vol_m.iloc[-1, 0] = 0.8
    eng.highest_prices["AAA"] = 104.0
    eng.check_cmvs_exits(len(eng.close_m) - 1)
    assert len(eng.pending_orders) == 1
    reason = eng.pending_orders[0]["reason"]
    assert "trend_confirm" in reason
    assert "ema20" in reason
