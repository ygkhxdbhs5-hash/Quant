"""MIN_HOLD_DAYS must not block ATR trailing stop; only discretionary exits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import StandaloneEngine


def _engine(*, n: int = 6) -> StandaloneEngine:
    eng = StandaloneEngine.__new__(StandaloneEngine)
    eng.USE_EMA9_EXIT = True
    eng.USE_ATR_EXIT = True
    eng.USE_EXHAUSTION_EXIT = True
    eng.USE_TIME_STOP = False
    eng.USE_STOP_LOSS = False
    eng.STOP_LOSS_PCT = 0.15
    eng.MIN_HOLD_DAYS = 10
    eng._hard_stopped_today = set()
    eng.EMA_EXIT_LENGTH = 9
    eng.atr_multiplier = 2.0
    eng.HIGH_ATR_PCT_EXIT = 0.08
    eng.TREND_EXIT_MIN_CONFIRM = 2
    eng.TREND_EXIT_MIN_CONFIRM_HIGH_VOL = 3
    eng.PROTECT_HEALTHY_TREND_PULLBACK = False
    eng.highest_prices = {"AAA": 120.0}
    eng.portfolio = {"AAA": 10}
    eng.previous_target_symbols = {"AAA"}
    eng.pending_orders = []
    eng._cmvs_forced_exits = set()
    eng.trade_journal = type(
        "J",
        (),
        {
            "open": {},
            "mark_peak": lambda *a, **k: None,
            "holding_days": lambda *a, **k: 1,  # inside min-hold
        },
    )()

    idx = pd.bdate_range("2024-01-02", periods=n)
    # Last close well below ATR stop (peak 120 - 2*2 = 116)
    close = np.array([100, 101, 102, 103, 102, 100], dtype=float)[:n]
    eng.close_m = pd.DataFrame({"AAA": close}, index=idx)
    eng.ema_exit_m = pd.DataFrame({"AAA": np.full(n, 110.0)}, index=idx)
    eng.ema9_m = eng.ema_exit_m
    eng.ema20_m = pd.DataFrame({"AAA": np.full(n, 95.0)}, index=idx)
    eng.sma50_m = pd.DataFrame({"AAA": np.full(n, 90.0)}, index=idx)
    eng.sma50_slope5_m = pd.DataFrame({"AAA": np.full(n, 0.01)}, index=idx)
    eng.rsi14_m = pd.DataFrame({"AAA": np.full(n, 40.0)}, index=idx)
    eng.daily_cp_m = pd.DataFrame({"AAA": np.full(n, 0.5)}, index=idx)
    eng.atr14_m = pd.DataFrame({"AAA": np.full(n, 2.0)}, index=idx)
    eng.atr_pct_m = pd.DataFrame({"AAA": np.full(n, 0.04)}, index=idx)
    eng.ret1_m = pd.DataFrame({"AAA": eng.close_m["AAA"].pct_change()}, index=idx)
    eng.ret5_m = pd.DataFrame({"AAA": np.full(n, -0.05)}, index=idx)
    eng.rel_vol_m = pd.DataFrame({"AAA": np.full(n, 1.2)}, index=idx)
    return eng


def test_atr_trail_fires_during_min_hold():
    eng = _engine()
    eng.check_cmvs_exits(len(eng.close_m) - 1)
    assert len(eng.pending_orders) == 1
    assert "atr_trail" in eng.pending_orders[0]["reason"]


def test_discretionary_trend_suppressed_during_min_hold():
    eng = _engine()
    eng.USE_ATR_EXIT = False  # isolate trend exit
    # Two closes below EMA + RSI confirm would sell after min-hold
    eng.highest_prices["AAA"] = 104.0
    eng.check_cmvs_exits(len(eng.close_m) - 1)
    assert eng.pending_orders == []


def test_discretionary_exhaustion_suppressed_during_min_hold():
    eng = _engine()
    eng.USE_ATR_EXIT = False
    eng.USE_EMA9_EXIT = False
    eng.rsi14_m.iloc[:] = 85.0
    eng.daily_cp_m.iloc[:] = 0.1
    eng.highest_prices["AAA"] = 104.0
    eng.check_cmvs_exits(len(eng.close_m) - 1)
    assert eng.pending_orders == []


def test_discretionary_resumes_after_min_hold():
    eng = _engine()
    eng.USE_ATR_EXIT = False
    eng.trade_journal.holding_days = lambda *a, **k: 15  # past min-hold
    eng.highest_prices["AAA"] = 104.0
    eng.check_cmvs_exits(len(eng.close_m) - 1)
    assert len(eng.pending_orders) == 1
    assert "trend_confirm" in eng.pending_orders[0]["reason"]
