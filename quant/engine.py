"""Core event-driven backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from quant.metrics import PerformanceReport, compute_metrics
from quant.portfolio import Portfolio
from quant.strategy import Signal, Strategy


REQUIRED_COLUMNS = ("open", "high", "low", "close")


@dataclass
class BacktestResult:
    """Outputs from a finished backtest run."""

    equity: pd.Series
    trades: list
    metrics: PerformanceReport
    signals: pd.Series


class BacktestEngine:
    """Run a ``Strategy`` over OHLCV bars with simple long-only execution.

    Execution model
    ---------------
    - Signal is generated on bar ``t`` using data through ``t`` (no lookahead).
    - Orders fill at the **next** bar's open (if available), otherwise at
      the current close. This is a common realistic default.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_cash: float = 100_000.0,
        commission_rate: float = 0.001,
        fill_on_next_open: bool = True,
    ) -> None:
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.fill_on_next_open = fill_on_next_open

    def run(self, data: pd.DataFrame) -> BacktestResult:
        bars = self._normalize(data)
        portfolio = Portfolio(
            initial_cash=self.initial_cash,
            commission_rate=self.commission_rate,
        )

        self.strategy.on_start(bars)
        pending: Optional[Signal] = None
        signal_records: list[tuple[pd.Timestamp, str]] = []

        for i in range(len(bars)):
            row = bars.iloc[i]
            ts = bars.index[i]
            price_mark = float(row["close"])

            # Execute any signal decided on the previous bar.
            if pending is not None:
                fill_price = (
                    float(row["open"])
                    if self.fill_on_next_open
                    else price_mark
                )
                if pending == Signal.BUY and portfolio.position == 0:
                    portfolio.buy(ts.to_pydatetime(), fill_price)
                elif pending == Signal.SELL and portfolio.position > 0:
                    portfolio.sell(ts.to_pydatetime(), fill_price)
                pending = None

            window = bars.iloc[: i + 1]
            signal = self.strategy.on_bar(window)
            if not isinstance(signal, Signal):
                signal = Signal(signal)
            signal_records.append((ts, signal.value))

            # Queue for next-bar fill, except on the final bar where we
            # fill immediately at close so the run can flatten cleanly.
            if signal in (Signal.BUY, Signal.SELL):
                if i == len(bars) - 1:
                    fill_price = price_mark
                    if signal == Signal.BUY and portfolio.position == 0:
                        portfolio.buy(ts.to_pydatetime(), fill_price)
                    elif signal == Signal.SELL and portfolio.position > 0:
                        portfolio.sell(ts.to_pydatetime(), fill_price)
                else:
                    pending = signal

            portfolio.mark_to_market(ts.to_pydatetime(), price_mark)

        self.strategy.on_finish()

        equity = portfolio.equity_series()
        signals = pd.Series(
            [s for _, s in signal_records],
            index=pd.DatetimeIndex([t for t, _ in signal_records]),
            name="signal",
        )
        metrics = compute_metrics(equity, portfolio.trades, self.initial_cash)
        return BacktestResult(
            equity=equity,
            trades=portfolio.trades,
            metrics=metrics,
            signals=signals,
        )

    @staticmethod
    def _normalize(data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            raise ValueError("Backtest data is empty.")
        df = data.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            else:
                df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df
