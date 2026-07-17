"""Strategy interface for the backtester.

Drop your own logic into a subclass of ``Strategy`` and implement
``on_bar``. The engine calls that method once per bar with a lookback
window ending at the current bar (no lookahead).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class Signal(str, Enum):
    """Trade intent emitted by a strategy."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Strategy(ABC):
    """Base class for trading strategies.

    Subclass this and implement ``on_bar``. Optionally override
    ``on_start`` / ``on_finish`` for setup and teardown.
    """

    name: str = "BaseStrategy"

    def on_start(self, data: pd.DataFrame) -> None:
        """Called once before the first bar is processed."""

    def on_finish(self) -> None:
        """Called once after the last bar is processed."""

    @abstractmethod
    def on_bar(self, window: pd.DataFrame) -> Signal:
        """Decide action for the bar at ``window.index[-1]``.

        Parameters
        ----------
        window:
            Historical OHLCV rows up to and including the current bar.
            Columns expected: ``open``, ``high``, ``low``, ``close``,
            ``volume`` (volume optional depending on strategy).

        Returns
        -------
        Signal
            BUY, SELL, or HOLD.
        """
