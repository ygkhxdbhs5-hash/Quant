"""Trade journal + shadow (counterfactual) exit analysis.

Observation-only: does not generate orders. Produces Facts about each closed trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class OpenTrade:
    symbol: str
    entry_date: pd.Timestamp
    entry_idx: int
    entry_price: float
    entry_rank: Optional[int] = None
    entry_cmvs: Optional[float] = None
    entry_rsi: Optional[float] = None
    entry_atr: Optional[float] = None
    peak_price: float = 0.0
    qty: int = 0


@dataclass
class ClosedTrade:
    symbol: str
    entry_date: object
    exit_date: object
    holding_days: int
    exit_reason: str
    entry_rank: Optional[int]
    exit_rank: Optional[int]
    entry_cmvs: Optional[float]
    exit_cmvs: Optional[float]
    entry_rsi: Optional[float]
    exit_rsi: Optional[float]
    entry_atr: Optional[float]
    exit_atr: Optional[float]
    entry_price: float
    exit_price: float
    peak_return: float
    final_return: float
    forward_return_5d: Optional[float]
    forward_return_10d: Optional[float]
    forward_return_20d: Optional[float]
    # Shadow / counterfactual (20d post-exit)
    post_exit_max_return: Optional[float]
    post_exit_max_drawdown: Optional[float]
    days_to_peak: Optional[int]
    days_to_trough: Optional[int]
    missed_upside: Optional[float]
    saved_drawdown: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TradeJournal:
    """Structured list of closed trades + open-position book."""

    def __init__(self, shadow_horizon_days: int = 20):
        self.shadow_horizon_days = int(shadow_horizon_days)
        self.open: Dict[str, OpenTrade] = {}
        self.closed: List[ClosedTrade] = []
        self.rebalance_events: List[Dict[str, Any]] = []

    def record_rebalance_turnover(
        self,
        date,
        n_entries: int,
        n_exits: int,
        n_held: int,
        *,
        n_forced_rank_exits: int = 0,
        n_new_entries: Optional[int] = None,
        avg_holding_days: Optional[float] = None,
        avg_rank_holdings: Optional[float] = None,
        decisions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        denom = max(n_held, 1)
        entries = int(n_new_entries) if n_new_entries is not None else int(n_entries)
        self.rebalance_events.append(
            {
                "date": date,
                "entries": entries,
                "exits": n_exits,
                "turnover": (entries + n_exits) / denom,
                "n_forced_rank_exits": int(n_forced_rank_exits),
                "n_new_entries": entries,
                "avg_holding_days": avg_holding_days,
                "avg_rank_holdings": avg_rank_holdings,
                "decisions": list(decisions or []),
            }
        )

    def on_entry(
        self,
        symbol: str,
        date,
        date_idx: int,
        price: float,
        qty: int,
        rank: Optional[int],
        cmvs: Optional[float],
        rsi: Optional[float],
        atr: Optional[float],
    ) -> None:
        if symbol in self.open:
            # Add-on buy: keep original entry; bump qty / peak
            ot = self.open[symbol]
            ot.qty += int(qty)
            ot.peak_price = max(ot.peak_price, float(price))
            return
        self.open[symbol] = OpenTrade(
            symbol=symbol,
            entry_date=pd.Timestamp(date),
            entry_idx=int(date_idx),
            entry_price=float(price),
            entry_rank=rank,
            entry_cmvs=cmvs,
            entry_rsi=rsi,
            entry_atr=atr,
            peak_price=float(price),
            qty=int(qty),
        )

    def mark_peak(self, symbol: str, price: float) -> None:
        if symbol in self.open and pd.notna(price) and price > 0:
            self.open[symbol].peak_price = max(self.open[symbol].peak_price, float(price))

    def holding_days(self, symbol: str, as_of_date) -> int:
        ot = self.open.get(symbol)
        if ot is None:
            return 0
        return int((pd.Timestamp(as_of_date) - ot.entry_date).days)

    def on_exit(
        self,
        symbol: str,
        date,
        date_idx: int,
        price: float,
        exit_reason: str,
        close_m: pd.DataFrame,
        exit_rank: Optional[int] = None,
        exit_cmvs: Optional[float] = None,
        exit_rsi: Optional[float] = None,
        exit_atr: Optional[float] = None,
    ) -> Optional[ClosedTrade]:
        ot = self.open.pop(symbol, None)
        if ot is None:
            return None
        exit_px = float(price)
        entry_px = float(ot.entry_price)
        if entry_px <= 0 or exit_px <= 0:
            return None

        peak_ret = (float(ot.peak_price) / entry_px) - 1.0
        final_ret = (exit_px / entry_px) - 1.0
        hold_days = int((pd.Timestamp(date) - ot.entry_date).days)

        fwd5, fwd10, fwd20 = self._forward_returns(close_m, symbol, date_idx, exit_px)
        shadow = self._shadow_exit(close_m, symbol, date_idx, exit_px)

        rec = ClosedTrade(
            symbol=symbol,
            entry_date=ot.entry_date,
            exit_date=pd.Timestamp(date),
            holding_days=hold_days,
            exit_reason=exit_reason or "unknown",
            entry_rank=ot.entry_rank,
            exit_rank=exit_rank,
            entry_cmvs=ot.entry_cmvs,
            exit_cmvs=exit_cmvs,
            entry_rsi=ot.entry_rsi,
            exit_rsi=exit_rsi,
            entry_atr=ot.entry_atr,
            exit_atr=exit_atr,
            entry_price=entry_px,
            exit_price=exit_px,
            peak_return=float(peak_ret),
            final_return=float(final_ret),
            forward_return_5d=fwd5,
            forward_return_10d=fwd10,
            forward_return_20d=fwd20,
            post_exit_max_return=shadow["post_exit_max_return"],
            post_exit_max_drawdown=shadow["post_exit_max_drawdown"],
            days_to_peak=shadow["days_to_peak"],
            days_to_trough=shadow["days_to_trough"],
            missed_upside=shadow["missed_upside"],
            saved_drawdown=shadow["saved_drawdown"],
        )
        self.closed.append(rec)
        return rec

    def _forward_returns(
        self, close_m: pd.DataFrame, symbol: str, exit_idx: int, exit_px: float
    ):
        out = []
        for h in (5, 10, 20):
            j = exit_idx + h
            if symbol not in close_m.columns or j >= len(close_m.index):
                out.append(None)
                continue
            px = close_m[symbol].iloc[j]
            if pd.isna(px) or px <= 0:
                out.append(None)
            else:
                out.append(float(px) / exit_px - 1.0)
        return tuple(out)

    def _shadow_exit(
        self, close_m: pd.DataFrame, symbol: str, exit_idx: int, exit_px: float
    ) -> Dict[str, Optional[float]]:
        empty = {
            "post_exit_max_return": None,
            "post_exit_max_drawdown": None,
            "days_to_peak": None,
            "days_to_trough": None,
            "missed_upside": None,
            "saved_drawdown": None,
        }
        if symbol not in close_m.columns or exit_px <= 0:
            return empty
        end = min(exit_idx + self.shadow_horizon_days, len(close_m.index) - 1)
        if end <= exit_idx:
            return empty
        window = close_m[symbol].iloc[exit_idx + 1 : end + 1].dropna()
        if window.empty:
            return empty
        rel = window.astype(float) / exit_px
        max_rel = float(rel.max())
        min_rel = float(rel.min())
        # days after exit (1-based within window)
        days_to_peak = int(rel.values.argmax()) + 1
        days_to_trough = int(rel.values.argmin()) + 1
        missed_upside = max_rel - 1.0  # Post-Exit Max / ExitPrice - 1
        saved_drawdown = 1.0 - min_rel  # 1 - Post-Exit Min / ExitPrice
        post_dd = min_rel - 1.0  # negative when price falls
        return {
            "post_exit_max_return": missed_upside,
            "post_exit_max_drawdown": post_dd,
            "days_to_peak": days_to_peak,
            "days_to_trough": days_to_trough,
            "missed_upside": missed_upside,
            "saved_drawdown": saved_drawdown,
        }

    def to_frame(self) -> pd.DataFrame:
        if not self.closed:
            return pd.DataFrame()
        return pd.DataFrame([c.as_dict() for c in self.closed])

    def summary_counts(self) -> Dict[str, Any]:
        return {
            "n_closed_trades": len(self.closed),
            "n_open_trades": len(self.open),
            "n_rebalance_events": len(self.rebalance_events),
        }
