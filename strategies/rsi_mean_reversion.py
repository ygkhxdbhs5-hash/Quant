"""RSI mean-reversion strategy."""

from __future__ import annotations

import pandas as pd

from quant.strategy import Signal, Strategy


class RSIMeanReversion(Strategy):
    """Buy when RSI is oversold; sell when RSI is overbought (while long)."""

    name = "RSIMeanReversion"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self._position_on = False

    def on_bar(self, window: pd.DataFrame) -> Signal:
        if len(window) < self.period + 2:
            return Signal.HOLD

        rsi = self._rsi(window["close"], self.period)
        if pd.isna(rsi):
            return Signal.HOLD

        if rsi < self.oversold and not self._position_on:
            self._position_on = True
            return Signal.BUY
        if rsi > self.overbought and self._position_on:
            self._position_on = False
            return Signal.SELL
        return Signal.HOLD

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.iloc[-period:].mean()
        avg_loss = loss.iloc[-period:].mean()
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))
