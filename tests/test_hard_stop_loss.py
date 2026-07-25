"""Tests for intraday hard % stop loss (Low trigger, min(Open, stop) fill)."""

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
    eng.STOP_LOSS_PCT = 0.15
    eng.MIN_HOLD_DAYS = 0
    eng.EMA_EXIT_LENGTH = 9
    eng.atr_multiplier = 2.0
    eng.PARTICIPATION_CAP_SELL = 1.0  # allow full exit in unit tests
    eng.COMMISSION_RATE = 0.0
    eng.SLIPPAGE_RATE = 0.0
    eng.highest_prices = {"AAA": 100.0}
    eng.portfolio = {"AAA": 10}
    eng.previous_target_symbols = {"AAA"}
    eng.pending_orders = []
    eng._cmvs_forced_exits = set()
    eng._hard_stopped_today = set()
    eng.cash = 0.0
    eng.adv20_m = pd.DataFrame()  # unused when PARTICIPATION allows via override
    eng.vol20_m = pd.DataFrame()
    eng.cs_spread_m = pd.DataFrame()
    eng.atr_pct_m = pd.DataFrame()

    class _OT:
        entry_price = 100.0

    eng.trade_journal = type(
        "J",
        (),
        {
            "open": {"AAA": _OT()},
            "mark_peak": lambda *a, **k: None,
            "holding_days": lambda *a, **k: 0,
            "on_exit": lambda *a, **k: None,
        },
    )()
    eng.reentry_cooldown = type(
        "C", (), {"record_exit": lambda *a, **k: None}
    )()
    eng._symbol_snapshot = lambda *a, **k: {
        "rank": None,
        "cmvs": None,
        "rsi": None,
        "atr": None,
    }

    idx = pd.bdate_range("2024-01-02", periods=n)
    close = np.full(n, 95.0)
    open_ = np.full(n, 96.0)
    low = np.full(n, 94.0)
    eng.close_m = pd.DataFrame({"AAA": close}, index=idx)
    eng.open_m = pd.DataFrame({"AAA": open_}, index=idx)
    eng.low_m = pd.DataFrame({"AAA": low}, index=idx)
    eng.high_m = pd.DataFrame({"AAA": np.full(n, 97.0)}, index=idx)
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
    # Large ADV so participation cap does not block
    eng.adv20_m = pd.DataFrame({"AAA": np.full(n, 1e12)}, index=idx)
    eng.vol20_m = pd.DataFrame({"AAA": np.full(n, 0.02)}, index=idx)
    eng.cs_spread_m = pd.DataFrame({"AAA": np.full(n, 0.001)}, index=idx)
    eng._cost_ratio = lambda *a, **k: 0.0
    return eng


def test_stop_loss_off_never_fires():
    eng = _exit_engine()
    eng.USE_STOP_LOSS = False
    eng.low_m.iloc[-1, 0] = 70.0
    stopped = eng.check_and_execute_hard_stop_loss(len(eng.close_m) - 1)
    assert stopped == set()
    assert "AAA" in eng.portfolio


def test_low_triggers_fill_at_stop_when_open_above():
    """Intraday pierce: Low <= stop, Open > stop → fill at stop (15%)."""
    eng = _exit_engine()
    # entry 100, stop 85; open 90, low 84 → fill min(90, 85) = 85
    eng.open_m.iloc[-1, 0] = 90.0
    eng.low_m.iloc[-1, 0] = 84.0
    eng.close_m.iloc[-1, 0] = 88.0  # close above stop — old rule would miss
    stopped = eng.check_and_execute_hard_stop_loss(len(eng.close_m) - 1)
    assert "AAA" in stopped
    assert "AAA" not in eng.portfolio
    assert eng.cash == 85.0 * 10  # fill at stop


def test_gap_down_fills_at_open():
    """Overnight gap: Open < stop → fill at Open (loss can exceed STOP_LOSS_PCT)."""
    eng = _exit_engine()
    eng.open_m.iloc[-1, 0] = 70.0
    eng.low_m.iloc[-1, 0] = 68.0
    eng.close_m.iloc[-1, 0] = 72.0
    stopped = eng.check_and_execute_hard_stop_loss(len(eng.close_m) - 1)
    assert "AAA" in stopped
    assert eng.cash == 70.0 * 10  # fill at open, not stop


def test_no_trigger_when_low_above_stop():
    eng = _exit_engine()
    eng.open_m.iloc[2, 0] = 92.0
    eng.low_m.iloc[2, 0] = 86.0  # above 85
    eng.close_m.iloc[2, 0] = 80.0  # close below stop — must NOT trigger on close
    stopped = eng.check_and_execute_hard_stop_loss(2)
    assert stopped == set()
    assert "AAA" in eng.portfolio


def test_atr_runs_only_when_hard_stop_not_hit():
    eng = _exit_engine()
    eng.USE_ATR_EXIT = True
    eng.highest_prices["AAA"] = 120.0
    eng.open_m.iloc[2, 0] = 95.0
    eng.low_m.iloc[2, 0] = 90.0  # above stop 85
    eng.close_m.iloc[2, 0] = 90.0  # ATR: 120 - 4 = 116 → close 90 triggers ATR
    eng._hard_stopped_today = eng.check_and_execute_hard_stop_loss(2)
    assert eng._hard_stopped_today == set()
    eng.check_cmvs_exits(2)
    assert len(eng.pending_orders) == 1
    assert "atr_trail" in eng.pending_orders[0]["reason"]


def test_atr_skipped_after_hard_stop():
    eng = _exit_engine()
    eng.USE_ATR_EXIT = True
    eng.highest_prices["AAA"] = 120.0
    eng.open_m.iloc[2, 0] = 90.0
    eng.low_m.iloc[2, 0] = 80.0  # triggers hard stop
    eng.close_m.iloc[2, 0] = 80.0
    eng._hard_stopped_today = eng.check_and_execute_hard_stop_loss(2)
    assert "AAA" in eng._hard_stopped_today
    eng.check_cmvs_exits(2)
    assert eng.pending_orders == []


def test_ema_still_works_when_hard_stop_not_hit():
    eng = _exit_engine(n=8)
    eng.USE_ATR_EXIT = False
    eng.USE_EMA9_EXIT = True
    eng.PROTECT_HEALTHY_TREND_PULLBACK = False
    eng.HIGH_ATR_PCT_EXIT = 0.08
    eng.TREND_EXIT_MIN_CONFIRM = 2
    eng.TREND_EXIT_MIN_CONFIRM_HIGH_VOL = 3
    i = len(eng.close_m) - 1
    eng.open_m.iloc[i, 0] = 95.0
    eng.low_m.iloc[i, 0] = 90.0  # above stop
    eng.close_m.iloc[i, 0] = 92.0
    eng.close_m.iloc[i - 1, 0] = 93.0
    eng.highest_prices["AAA"] = 104.0
    eng.ema_exit_m.iloc[i, 0] = 110.0
    eng.ema_exit_m.iloc[i - 1, 0] = 110.0
    eng.ema20_m.iloc[i, 0] = 90.0
    eng.rsi14_m.iloc[i, 0] = 40.0
    eng.ret5_m.iloc[i, 0] = -0.06
    eng.rel_vol_m.iloc[i, 0] = 1.3
    eng.sma50_m.iloc[i, 0] = 80.0
    eng._hard_stopped_today = eng.check_and_execute_hard_stop_loss(i)
    eng.check_cmvs_exits(i)
    assert len(eng.pending_orders) == 1
    assert "trend_confirm" in eng.pending_orders[0]["reason"]


def test_hard_stop_active_during_min_hold():
    eng = _exit_engine()
    eng.MIN_HOLD_DAYS = 30
    eng.trade_journal.holding_days = lambda *a, **k: 1
    eng.open_m.iloc[-1, 0] = 90.0
    eng.low_m.iloc[-1, 0] = 80.0
    stopped = eng.check_and_execute_hard_stop_loss(len(eng.close_m) - 1)
    assert "AAA" in stopped


def test_stop_loss_pct_clamped_in_loader():
    t = load_research_toggles({"research": {"USE_STOP_LOSS": True, "STOP_LOSS_PCT": 0.99}})
    assert t.USE_STOP_LOSS is True
    assert t.STOP_LOSS_PCT == 0.50
    panel = ResearchToggles(USE_STOP_LOSS=True, STOP_LOSS_PCT=0.15).format_panel()
    assert "USE_STOP_LOSS = True" in panel
    assert "STOP_LOSS_PCT = 15.00%" in panel
