"""Portfolio and trade bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class Trade:
    """A completed round-trip or a single fill record."""

    timestamp: datetime
    side: str
    price: float
    quantity: float
    commission: float
    cash_after: float
    position_after: float


@dataclass
class Portfolio:
    """Tracks cash, position, equity, and fills."""

    initial_cash: float = 100_000.0
    commission_rate: float = 0.001
    cash: float = field(init=False)
    position: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)

    def mark_to_market(self, timestamp: datetime, price: float) -> float:
        equity = self.cash + self.position * price
        self.equity_curve.append((timestamp, equity))
        return equity

    def buy(self, timestamp: datetime, price: float, quantity: Optional[float] = None) -> Optional[Trade]:
        """Buy ``quantity`` shares, or use all available cash if quantity is None."""
        if quantity is None:
            if price <= 0:
                return None
            quantity = self.cash / (price * (1.0 + self.commission_rate))
        if quantity <= 0 or price <= 0:
            return None

        cost = quantity * price
        commission = cost * self.commission_rate
        total = cost + commission
        if total > self.cash:
            quantity = self.cash / (price * (1.0 + self.commission_rate))
            cost = quantity * price
            commission = cost * self.commission_rate
            total = cost + commission
        if quantity <= 0:
            return None

        self.cash -= total
        self.position += quantity
        trade = Trade(
            timestamp=timestamp,
            side="BUY",
            price=price,
            quantity=quantity,
            commission=commission,
            cash_after=self.cash,
            position_after=self.position,
        )
        self.trades.append(trade)
        return trade

    def sell(self, timestamp: datetime, price: float, quantity: Optional[float] = None) -> Optional[Trade]:
        """Sell ``quantity`` shares, or flatten the whole position if quantity is None."""
        if self.position <= 0 or price <= 0:
            return None
        if quantity is None:
            quantity = self.position
        quantity = min(quantity, self.position)
        if quantity <= 0:
            return None

        proceeds = quantity * price
        commission = proceeds * self.commission_rate
        self.cash += proceeds - commission
        self.position -= quantity
        trade = Trade(
            timestamp=timestamp,
            side="SELL",
            price=price,
            quantity=quantity,
            commission=commission,
            cash_after=self.cash,
            position_after=self.position,
        )
        self.trades.append(trade)
        return trade

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        index, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(index), name="equity")
