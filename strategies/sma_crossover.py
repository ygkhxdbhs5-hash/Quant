"""Classic dual moving-average crossover strategy."""

from __future__ import annotations

import pandas as pd

from quant.strategy import Signal, Strategy


class SMACrossover(Strategy):
    """Go long when fast SMA crosses above slow SMA; exit on cross below."""

    name = "SMACrossover"

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        self.fast = fast
        self.slow = slow
        self._position_on = False

    def on_bar(self, window: pd.DataFrame) -> Signal:
        if len(window) < self.slow + 1:
            return Signal.HOLD

        close = window["close"]
        fast_now = close.iloc[-self.fast :].mean()
        slow_now = close.iloc[-self.slow :].mean()
        fast_prev = close.iloc[-self.fast - 1 : -1].mean()
        slow_prev = close.iloc[-self.slow - 1 : -1].mean()

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_down = fast_prev >= slow_prev and fast_now < slow_now

        if crossed_up and not self._position_on:
            self._position_on = True
            return Signal.BUY
        if crossed_down and self._position_on:
            self._position_on = False
            return Signal.SELL
        return Signal.HOLD
