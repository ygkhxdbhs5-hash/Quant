"""Unit tests for the local-data breakout candidate screener."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.breakout_screener import (
    BreakoutScreenerConfig,
    breakout_config_from_dict,
    screen_breakouts,
    _ema,
)


def _fresh_dual_breakout_panels(n: int = 120):
    """Synthetic panels where BREAKER freshly crosses EMA30 + resistance with volume."""
    idx = pd.bdate_range("2024-01-02", periods=n)

    close = np.full(n, 88.0)
    close[:40] = np.linspace(70, 88, 40)
    close[-1] = 102.0
    high = np.full(n, 89.0)
    high[:40] = close[:40] + 1.0
    high[40:100] = 100.0  # horizontal resistance shelf
    high[-1] = 103.0

    close_m = pd.DataFrame(
        {
            "BREAKER": close,
            "LATE": np.linspace(80, 130, n),  # already far extended
            "QUIET": np.full(n, 50.0),
        },
        index=idx,
    )
    high_m = pd.DataFrame(
        {
            "BREAKER": high,
            "LATE": close_m["LATE"].to_numpy() + 1.0,
            "QUIET": close_m["QUIET"].to_numpy() + 0.5,
        },
        index=idx,
    )

    # Keep BREAKER closes strictly below EMA until the final breakout bar
    ema = _ema(close_m[["BREAKER"]], 30)
    for i in range(40, n - 1):
        if close_m.iloc[i, close_m.columns.get_loc("BREAKER")] > ema.iloc[i, 0]:
            close_m.iloc[i, close_m.columns.get_loc("BREAKER")] = float(ema.iloc[i, 0]) - 0.5
            ema = _ema(close_m[["BREAKER"]], 30)
    close_m.iloc[-1, close_m.columns.get_loc("BREAKER")] = 102.0

    shares = np.concatenate([np.full(n - 1, 1e6), [3e6]])
    dvol_m = pd.DataFrame(
        {
            "BREAKER": close_m["BREAKER"].to_numpy() * shares,
            "LATE": close_m["LATE"].to_numpy() * 1e6,
            "QUIET": close_m["QUIET"].to_numpy() * 1e6,
        },
        index=idx,
    )
    return close_m, high_m, dvol_m


def test_breakout_config_from_dict_reads_nested_section():
    cfg = breakout_config_from_dict(
        {
            "breakout_screener": {
                "ema_period": 21,
                "volume_threshold": 2.0,
                "resistance_lookback": 40,
                "max_breakout_age": 7,
            }
        }
    )
    assert cfg.ema_period == 21
    assert cfg.volume_threshold == 2.0
    assert cfg.resistance_lookback == 40
    assert cfg.max_breakout_age == 7
    assert cfg.max_extension_pct == 0.05  # default retained


def test_fresh_ema_and_resistance_breakout_selected():
    close_m, high_m, dvol_m = _fresh_dual_breakout_panels()
    cfg = BreakoutScreenerConfig(
        ema_period=30,
        resistance_lookback=60,
        volume_avg_window=20,
        volume_threshold=1.5,
        max_breakout_age=5,
        max_breakout_separation=3,
        max_extension_pct=0.05,
        min_history_bars=80,
    )
    out = screen_breakouts(close_m, high_m, dvol_m, cfg=cfg, exclude={"QQQ"})
    assert "BREAKER" in set(out["symbol"])
    assert "LATE" not in set(out["symbol"])
    assert "QUIET" not in set(out["symbol"])
    row = out.loc[out["symbol"] == "BREAKER"].iloc[0]
    assert row["ema_break_age"] <= cfg.max_breakout_age
    assert row["resistance_break_age"] <= cfg.max_breakout_age
    assert row["volume_ratio"] >= cfg.volume_threshold
    assert row["extension_pct"] <= cfg.max_extension_pct


def test_already_above_ema_without_fresh_cross_is_excluded():
    """State 'above EMA' alone must not count — need a fresh cross event."""
    n = 100
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = np.linspace(100, 150, n)
    high = close + 1.0
    close_m = pd.DataFrame({"STALE": close}, index=idx)
    high_m = pd.DataFrame({"STALE": high}, index=idx)
    dvol_m = close_m * 1e6

    cfg = BreakoutScreenerConfig(max_breakout_age=3, min_history_bars=50)
    out = screen_breakouts(close_m, high_m, dvol_m, cfg=cfg)
    assert out.empty


def test_extension_filter_excludes_chase():
    n = 120
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = np.full(n, 95.0)
    high = np.full(n, 100.0)
    # Break resistance several bars ago and run far above
    close[-6] = 99.0
    close[-5] = 101.0  # breakout bar
    close[-4:] = [105, 110, 115, 120]
    high[-5:] = close[-5:] + 1.0
    close_m = pd.DataFrame({"CHASE": close}, index=idx)
    high_m = pd.DataFrame({"CHASE": high}, index=idx)
    vol = np.full(n, 1e6)
    vol[-5] = 3e6
    dvol_m = close_m.mul(vol, axis=0)

    cfg = BreakoutScreenerConfig(
        max_breakout_age=10,
        max_extension_pct=0.05,
        volume_threshold=1.2,
        min_history_bars=80,
    )
    out = screen_breakouts(close_m, high_m, dvol_m, cfg=cfg)
    assert out.empty
