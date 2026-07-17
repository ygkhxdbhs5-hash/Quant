"""Template for plugging in your own strategy code.

Replace the body of ``on_bar`` with your signal logic. Keep the return
type as ``Signal.BUY``, ``Signal.SELL``, or ``Signal.HOLD``.

The engine passes a lookback ``window`` that ends at the current bar —
never peek at future rows.
"""

from __future__ import annotations

import pandas as pd

from quant.strategy import Signal, Strategy


class YourStrategy(Strategy):
    """Placeholder — swap this for your existing strategy logic."""

    name = "YourStrategy"

    def __init__(self, lookback: int = 20, entry_z: float = -1.0, exit_z: float = 0.0) -> None:
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self._position_on = False

    def on_bar(self, window: pd.DataFrame) -> Signal:
        # --- replace from here with your strategy ---------------------
        if len(window) < self.lookback:
            return Signal.HOLD

        closes = window["close"].iloc[-self.lookback :]
        mean = closes.mean()
        std = closes.std(ddof=0)
        if std == 0:
            return Signal.HOLD

        z = (closes.iloc[-1] - mean) / std

        if z <= self.entry_z and not self._position_on:
            self._position_on = True
            return Signal.BUY
        if z >= self.exit_z and self._position_on:
            self._position_on = False
            return Signal.SELL
        return Signal.HOLD
        # --- replace until here ---------------------------------------
