"""Technical breakout candidate screener over local Massive price panels.

Reads the same `data/prices/panels.pkl` / `data/metadata/universe.pkl` files
produced by `downloader/` — does not download or mutate market data.

Workflow: Existing Dataset → Breakout Filter → Filtered Candidates
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.strategy import load_config


@dataclass
class BreakoutScreenerConfig:
    """Tunable breakout-filter knobs (also mirrored under config.yaml)."""

    ema_period: int = 30
    resistance_lookback: int = 60
    volume_avg_window: int = 20
    volume_threshold: float = 1.5
    max_breakout_age: int = 5
    max_breakout_separation: int = 3
    max_extension_pct: float = 0.05
    min_history_bars: int = 80


def breakout_config_from_dict(cfg: dict) -> BreakoutScreenerConfig:
    """Build screener config from top-level or nested `breakout_screener` YAML."""
    section = cfg.get("breakout_screener") or {}
    known = {f.name for f in fields(BreakoutScreenerConfig)}
    kwargs = {k: section[k] for k in known if k in section}
    return BreakoutScreenerConfig(**kwargs)


def load_local_market_data(config: dict) -> tuple[dict, dict]:
    """Load universe + OHLCV panels already written by the downloader."""
    paths = config.get("paths", {})
    universe_path = Path(paths.get("metadata", "data/metadata")) / "universe.pkl"
    panels_path = Path(paths.get("prices", "data/prices")) / "panels.pkl"
    for required in (universe_path, panels_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Run: python -m downloader.update_data "
                f"--config config/config.yaml"
            )
    with open(universe_path, "rb") as f:
        universe = pickle.load(f)
    with open(panels_path, "rb") as f:
        panels = pickle.load(f)
    return universe, panels


def _ema(close: pd.DataFrame, period: int) -> pd.DataFrame:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def _share_volume(close_m: pd.DataFrame, dvol_m: pd.DataFrame) -> pd.DataFrame:
    """Recover share volume from dollar-volume panels (adjClose × volume)."""
    return dvol_m / close_m.replace(0.0, np.nan)


def _event_ages(event: pd.DataFrame, as_of: pd.Timestamp, max_age: int) -> pd.Series:
    """Days since the most recent True event on/before as_of, capped by max_age.

    Returns NaN when no event falls inside the lookback window.
    """
    if as_of not in event.index:
        raise KeyError(f"as_of {as_of.date()} not in price index")
    loc = event.index.get_loc(as_of)
    if isinstance(loc, slice):
        loc = loc.start
    start = max(0, int(loc) - max_age + 1)
    window = event.iloc[start : int(loc) + 1]
    ages = pd.Series(np.nan, index=event.columns, dtype=float)
    if window.empty:
        return ages
    # idxmax on bool finds first True; reverse so the most recent True wins
    rev = window.iloc[::-1]
    last_true_ts = rev.apply(lambda col: col.idxmax() if col.any() else pd.NaT)
    for sym, ts in last_true_ts.items():
        if pd.isna(ts):
            continue
        # Bar-distance (trading days in window), not calendar days
        ages[sym] = float(len(window) - 1 - window.index.get_loc(ts))
    return ages


def _fresh_cross_above(close: pd.DataFrame, level: pd.DataFrame) -> pd.DataFrame:
    """True on bars where price freshly crosses above `level` (event, not state)."""
    prev_close = close.shift(1)
    prev_level = level.shift(1)
    return (prev_close <= prev_level) & (close > level)


def screen_breakouts(
    close_m: pd.DataFrame,
    high_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    tickers: Optional[list[str]] = None,
    *,
    cfg: Optional[BreakoutScreenerConfig] = None,
    as_of: Optional[pd.Timestamp] = None,
    exclude: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Apply the breakout filter and return ranked candidate rows.

    Conditions (all required):
      1. Fresh EMA upward breakout within ``max_breakout_age`` bars
      2. Fresh horizontal-resistance breakout within ``max_breakout_age`` bars
      3. The two breakout ages differ by at most ``max_breakout_separation``
      4. Current volume ≥ ``volume_threshold`` × recent average volume
      5. Close is not more than ``max_extension_pct`` above the resistance level
    """
    cfg = cfg or BreakoutScreenerConfig()
    exclude = exclude or set()

    if tickers is None:
        cols = [c for c in close_m.columns if c not in exclude]
    else:
        cols = [c for c in tickers if c in close_m.columns and c not in exclude]
    if not cols:
        return pd.DataFrame()

    close = close_m[cols]
    high = high_m.reindex(columns=cols)
    dvol = dvol_m.reindex(columns=cols)

    if as_of is None:
        as_of = close.dropna(how="all").index.max()
    else:
        as_of = pd.Timestamp(as_of)
        if as_of not in close.index:
            # snap to last available session on/before as_of
            eligible = close.index[close.index <= as_of]
            if len(eligible) == 0:
                return pd.DataFrame()
            as_of = eligible[-1]

    ema = _ema(close, cfg.ema_period)
    # Horizontal resistance = prior-session rolling high (no lookahead)
    resistance = high.rolling(cfg.resistance_lookback, min_periods=cfg.resistance_lookback).max().shift(1)

    ema_event = _fresh_cross_above(close, ema)
    res_event = _fresh_cross_above(close, resistance)

    vol = _share_volume(close, dvol)
    vol_avg = vol.rolling(cfg.volume_avg_window, min_periods=max(5, cfg.volume_avg_window // 2)).mean().shift(1)
    vol_ratio = vol / vol_avg.replace(0.0, np.nan)

    ema_age = _event_ages(ema_event, as_of, cfg.max_breakout_age)
    res_age = _event_ages(res_event, as_of, cfg.max_breakout_age)

    close_now = close.loc[as_of]
    ema_now = ema.loc[as_of]
    res_now = resistance.loc[as_of]
    vol_ratio_now = vol_ratio.loc[as_of]
    history_ok = close.loc[:as_of].count() >= cfg.min_history_bars

    sep = (ema_age - res_age).abs()
    extension = (close_now - res_now) / res_now.replace(0.0, np.nan)

    mask = (
        history_ok
        & ema_age.notna()
        & res_age.notna()
        & (sep <= cfg.max_breakout_separation)
        & (vol_ratio_now >= cfg.volume_threshold)
        & (extension <= cfg.max_extension_pct)
        & (extension >= -0.02)  # must be at/near resistance, not far below
        & close_now.notna()
        & ema_now.notna()
        & res_now.notna()
    )

    hits = mask[mask].index.tolist()
    if not hits:
        return pd.DataFrame(
            columns=[
                "symbol",
                "as_of",
                "close",
                "ema",
                "resistance",
                "ema_break_age",
                "resistance_break_age",
                "breakout_separation",
                "volume_ratio",
                "extension_pct",
            ]
        )

    rows = []
    for sym in hits:
        rows.append(
            {
                "symbol": sym,
                "as_of": as_of,
                "close": float(close_now[sym]),
                "ema": float(ema_now[sym]),
                "resistance": float(res_now[sym]),
                "ema_break_age": int(ema_age[sym]),
                "resistance_break_age": int(res_age[sym]),
                "breakout_separation": int(sep[sym]),
                "volume_ratio": float(vol_ratio_now[sym]),
                "extension_pct": float(extension[sym]),
            }
        )

    out = pd.DataFrame(rows)
    # Freshest dual breakout first, then strongest volume confirmation
    out = out.sort_values(
        by=["breakout_separation", "ema_break_age", "resistance_break_age", "volume_ratio"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    return out


def run_screener(
    config: Optional[dict] = None,
    config_path: str = "config/config.yaml",
    *,
    as_of: Optional[str] = None,
) -> pd.DataFrame:
    """End-to-end: load local dataset → filter → candidate table."""
    config = config or load_config(config_path)
    screener_cfg = breakout_config_from_dict(config)
    universe, panels = load_local_market_data(config)

    tickers = list(universe.get("tickers") or universe.get("all_tickers") or [])
    benchmark = config.get("benchmark", "QQQ")
    exclude = {benchmark} if benchmark else set()

    as_of_ts = pd.Timestamp(as_of) if as_of else None
    return screen_breakouts(
        close_m=panels["close_m"],
        high_m=panels["high_m"],
        dvol_m=panels["dvol_m"],
        tickers=tickers or None,
        cfg=screener_cfg,
        as_of=as_of_ts,
        exclude=exclude,
    )


def format_candidates_table(candidates: pd.DataFrame) -> str:
    """Human-readable console table for filtered candidates."""
    if candidates is None or candidates.empty:
        return "No breakout candidates matched the filter."

    display = candidates.copy()
    display["as_of"] = pd.to_datetime(display["as_of"]).dt.strftime("%Y-%m-%d")
    display["close"] = display["close"].map(lambda x: f"{x:.2f}")
    display["ema"] = display["ema"].map(lambda x: f"{x:.2f}")
    display["resistance"] = display["resistance"].map(lambda x: f"{x:.2f}")
    display["volume_ratio"] = display["volume_ratio"].map(lambda x: f"{x:.2f}x")
    display["extension_pct"] = display["extension_pct"].map(lambda x: f"{x * 100:.2f}%")
    display = display.rename(
        columns={
            "symbol": "Symbol",
            "as_of": "AsOf",
            "close": "Close",
            "ema": "EMA",
            "resistance": "Resistance",
            "ema_break_age": "EMA Age",
            "resistance_break_age": "Res Age",
            "breakout_separation": "Sep",
            "volume_ratio": "Vol Ratio",
            "extension_pct": "Extension",
        }
    )
    return display.to_string(index=False)


def config_summary(cfg: BreakoutScreenerConfig) -> str:
    parts = [f"{k}={v}" for k, v in asdict(cfg).items()]
    return ", ".join(parts)
