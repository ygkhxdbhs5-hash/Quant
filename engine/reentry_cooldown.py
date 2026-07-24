"""Adaptive re-entry cooldown after losing exits.

Prevents immediate re-buys into the same weak name after a stop-out.
Cooldown length scales with loss severity, volatility, exit reason, and
loss streak. Early clearance requires renewed-strength evidence — not
just calendar time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CooldownRecord:
    symbol: str
    exit_idx: int
    exit_price: float
    exit_reason: str
    loss_pct: float  # negative for losses
    atr_pct: float
    vol20: float
    consecutive_losses: int
    min_holdout_idx: int  # cannot clear before this index
    hard_expiry_idx: int  # absolute max holdout


def _reason_family(exit_reason: str) -> str:
    r = (exit_reason or "").lower()
    if "atr_trail" in r:
        return "atr_trail"
    if "trend_confirm" in r or "ema9_break" in r or "ema" in r:
        return "ema_break"
    if "exhaustion" in r:
        return "exhaustion"
    if "bear_flatten" in r:
        return "bear_flatten"
    if "rebalance_exit" in r:
        return "rebalance_exit"
    if "delist" in r:
        return "delist"
    return "other"


# Base trading-day holdouts by exit family (weak structure → longer).
_BASE_DAYS = {
    "atr_trail": 18,
    "ema_break": 14,
    "exhaustion": 8,
    "bear_flatten": 10,
    "rebalance_exit": 10,
    "delist": 60,
    "other": 12,
}


class AdaptiveReentryCooldown:
    """Tracks per-symbol re-entry bans after losing trades."""

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self._records: Dict[str, CooldownRecord] = {}
        self._loss_streak: Dict[str, int] = {}
        self.blocks = 0
        self.arms = 0
        self.early_clears = 0

    def record_exit(
        self,
        *,
        symbol: str,
        date_idx: int,
        exit_price: float,
        exit_reason: str,
        final_return: float,
        atr_pct: float = 0.0,
        vol20: float = 0.0,
        peak_return: float = 0.0,
    ) -> Optional[CooldownRecord]:
        """Arm cooldown only after a losing exit. Winning exits clear streaks."""
        if not self.enabled:
            return None

        sym = str(symbol)
        if float(final_return) >= 0.0:
            self._loss_streak[sym] = 0
            self._records.pop(sym, None)
            return None

        streak = int(self._loss_streak.get(sym, 0)) + 1
        self._loss_streak[sym] = streak

        family = _reason_family(exit_reason)
        base = int(_BASE_DAYS.get(family, 12))

        # Larger losses → longer ban (e.g. -5% → +0d, -20% → +9d, -40% → +18d)
        loss_mag = abs(float(final_return))
        loss_extra = int(round(min(loss_mag, 0.50) / 0.05 * 3))

        # High ATR% / vol → microstructure still unstable
        atr_extra = int(round(max(0.0, float(atr_pct) - 0.06) / 0.02 * 3))
        vol_extra = int(round(max(0.0, float(vol20) - 0.04) / 0.02 * 2))

        # Give-back from peak (failed runner) → extra caution
        giveback = max(0.0, float(peak_return) - float(final_return))
        giveback_extra = int(round(min(giveback, 0.40) / 0.10 * 2))

        # Stacking losses on the same name (CURX/HKIT/JEM pattern)
        streak_extra = max(0, (streak - 1) * 7)

        holdout = base + loss_extra + atr_extra + vol_extra + giveback_extra + streak_extra
        holdout = int(min(max(holdout, 5), 60))

        # Minimum days before adaptive early-clear is allowed
        min_hold = int(min(max(5, holdout // 3), holdout))

        rec = CooldownRecord(
            symbol=sym,
            exit_idx=int(date_idx),
            exit_price=float(exit_price),
            exit_reason=str(exit_reason or ""),
            loss_pct=float(final_return),
            atr_pct=float(atr_pct or 0.0),
            vol20=float(vol20 or 0.0),
            consecutive_losses=streak,
            min_holdout_idx=int(date_idx) + min_hold,
            hard_expiry_idx=int(date_idx) + holdout,
        )
        self._records[sym] = rec
        self.arms += 1
        return rec

    def renewed_strength(self, snap: Dict[str, Any]) -> bool:
        """Evidence the name has healed enough to allow re-entry."""
        close = snap.get("close")
        exit_px = snap.get("exit_price")
        sma20 = snap.get("sma20")
        sma50 = snap.get("sma50")
        rsi = snap.get("rsi14")
        ret5 = snap.get("ret5")
        close_vs_high20 = snap.get("close_vs_high20")
        atr_pct = snap.get("atr_pct")

        if close is None or exit_px is None or exit_px <= 0:
            return False

        checks = []
        # Reclaimed exit with buffer — not still grinding lower
        checks.append(float(close) >= float(exit_px) * 1.02)
        # Trend structure recovering
        if sma20 is not None and sma20 == sma20:  # not NaN
            checks.append(float(close) > float(sma20))
        else:
            checks.append(False)
        if sma50 is not None and sma50 == sma50:
            checks.append(float(close) > float(sma50) * 0.98)
        # Momentum / RSI not still broken
        if rsi is not None and rsi == rsi:
            checks.append(float(rsi) >= 50.0)
        else:
            checks.append(False)
        if ret5 is not None and ret5 == ret5:
            checks.append(float(ret5) > 0.0)
        else:
            checks.append(False)
        # Not still collapsing from local highs
        if close_vs_high20 is not None and close_vs_high20 == close_vs_high20:
            checks.append(float(close_vs_high20) >= 0.85)
        # Volatility cooled off vs exit ATR regime
        if atr_pct is not None and atr_pct == atr_pct:
            checks.append(float(atr_pct) <= 0.12)

        # Require strong majority — adaptive clear is strict by design
        return sum(1 for c in checks if c) >= max(4, len(checks) - 1)

    def is_blocked(self, symbol: str, date_idx: int, snap: Optional[Dict[str, Any]] = None) -> bool:
        if not self.enabled:
            return False
        sym = str(symbol)
        rec = self._records.get(sym)
        if rec is None:
            return False

        idx = int(date_idx)
        if idx >= rec.hard_expiry_idx:
            self._records.pop(sym, None)
            return False

        if idx < rec.min_holdout_idx:
            self.blocks += 1
            return True

        # Adaptive window: clear early only with renewed strength
        if snap is not None:
            snap = dict(snap)
            snap.setdefault("exit_price", rec.exit_price)
            if self.renewed_strength(snap):
                self._records.pop(sym, None)
                self.early_clears += 1
                return False

        self.blocks += 1
        return True

    def blocked_symbols(self, date_idx: int, snap_fn) -> set:
        """Return symbols still on cooldown; snap_fn(sym) -> strength dict."""
        out = set()
        for sym in list(self._records.keys()):
            snap = snap_fn(sym) if snap_fn is not None else None
            if self.is_blocked(sym, date_idx, snap):
                out.add(sym)
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "active": len(self._records),
            "arms": self.arms,
            "blocks": self.blocks,
            "early_clears": self.early_clears,
            "symbols": sorted(self._records.keys()),
        }
