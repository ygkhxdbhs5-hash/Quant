"""Local runtime helpers that mirror the LEAN APIs the strategy already uses.

These are infrastructure only. They do not implement investment logic.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


class RollingWindow:
    """Fixed-length window; index 0 is the most recent value (LEAN order)."""

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self._data: List[Any] = []

    def Add(self, value: Any) -> None:
        self._data.insert(0, value)
        if len(self._data) > self.size:
            self._data.pop()

    def __getitem__(self, index: int) -> Any:
        return self._data[index]

    @property
    def Count(self) -> int:
        return len(self._data)

    @property
    def IsReady(self) -> bool:
        return len(self._data) >= self.size


@dataclass
class IndicatorDataPoint:
    Value: float
    Time: Optional[datetime] = None


class SMA:
    """Simple moving average over daily closes."""

    def __init__(self, period: int) -> None:
        self.period = int(period)
        self._window = RollingWindow(self.period)
        self.Current = IndicatorDataPoint(Value=0.0)
        self.Updated: Optional[Callable[[Any, IndicatorDataPoint], None]] = None
        self.IsReady = False

    def Update(self, time: datetime, value: float) -> None:
        self._window.Add(float(value))
        if self._window.Count >= self.period:
            self.IsReady = True
            avg = sum(self._window[i] for i in range(self.period)) / self.period
            point = IndicatorDataPoint(Value=avg, Time=time)
            self.Current = point
            if self.Updated is not None:
                self.Updated(self, point)


@dataclass
class SecurityHolding:
    quantity: float = 0.0
    average_price: float = 0.0
    market_value: float = 0.0

    @property
    def Invested(self) -> bool:
        return abs(self.quantity) > 1e-12


@dataclass
class Security:
    symbol: str
    price: float = 0.0
    volume: float = 0.0

    @property
    def Symbol(self) -> str:
        return self.symbol

    @property
    def Price(self) -> float:
        return self.price


@dataclass
class Delisting:
    Type: str


class DelistingType:
    Warning = "Warning"


@dataclass
class TradeBar:
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float


class BarDict:
    def __init__(self) -> None:
        self._bars: Dict[str, TradeBar] = {}

    def ContainsKey(self, symbol: str) -> bool:
        return symbol in self._bars

    def __getitem__(self, symbol: str) -> TradeBar:
        return self._bars[symbol]

    def set(self, symbol: str, bar: TradeBar) -> None:
        self._bars[symbol] = bar


@dataclass
class Slice:
    Bars: BarDict = field(default_factory=BarDict)
    Delistings: Dict[str, Delisting] = field(default_factory=dict)


class Portfolio:
    def __init__(self, cash: float) -> None:
        self.Cash = float(cash)
        self.TotalPortfolioValue = float(cash)
        self._holdings: Dict[str, SecurityHolding] = {}

    def __getitem__(self, symbol: str) -> SecurityHolding:
        if symbol not in self._holdings:
            self._holdings[symbol] = SecurityHolding()
        return self._holdings[symbol]

    @property
    def Keys(self) -> Iterable[str]:
        return list(self._holdings.keys())

    def mark_to_market(self, securities: Dict[str, Security]) -> None:
        equity = self.Cash
        for symbol, holding in self._holdings.items():
            px = securities.get(symbol).price if symbol in securities else holding.average_price
            holding.market_value = holding.quantity * px
            equity += holding.market_value
        self.TotalPortfolioValue = equity


class LocalAlgorithm:
    """Minimal QCAlgorithm-compatible host for local backtests."""

    def __init__(self) -> None:
        self.Time: datetime = datetime(2007, 1, 1)
        self.StartDate: datetime = datetime(2007, 1, 1)
        self.EndDate: datetime = datetime(2026, 6, 30)
        self.Portfolio = Portfolio(0.0)
        self.Securities: Dict[str, Security] = {}
        self._cash_set = False
        self._logs: List[str] = []
        self._errors: List[str] = []
        self._equity_curve: List[tuple] = []
        self._orders: List[dict] = []
        self._sma_registry: Dict[str, SMA] = {}
        self.benchmark: Optional[str] = None
        self._scheduled: List[Callable[[], None]] = []

    # --- LEAN-like setup API -------------------------------------------------
    def SetStartDate(self, year: int, month: int, day: int) -> None:
        self.StartDate = datetime(year, month, day)
        self.Time = self.StartDate

    def SetEndDate(self, year: int, month: int, day: int) -> None:
        self.EndDate = datetime(year, month, day)

    def SetCash(self, cash: float) -> None:
        self.Portfolio = Portfolio(float(cash))
        self._cash_set = True

    def AddEquity(self, ticker: str, resolution: Any = None) -> Security:
        sec = Security(symbol=ticker)
        self.Securities[ticker] = sec
        return sec

    def SetBenchmark(self, symbol: str) -> None:
        self.benchmark = symbol

    def SMA(self, symbol: str, period: int, resolution: Any = None) -> SMA:
        key = f"{symbol}:{period}"
        if key not in self._sma_registry:
            self._sma_registry[key] = SMA(period)
        return self._sma_registry[key]

    def Log(self, msg: str) -> None:
        line = f"{self.Time.date()} {msg}"
        self._logs.append(line)
        print(line)

    def Error(self, msg: str) -> None:
        line = f"{self.Time.date()} ERROR {msg}"
        self._errors.append(line)
        print(line)

    def History(self, symbols: List[str], periods: int, resolution: Any = None) -> pd.DataFrame:
        """Filled by the runner via ``_history_provider`` if attached."""
        provider = getattr(self, "_history_provider", None)
        if provider is None:
            return pd.DataFrame()
        return provider(symbols, periods, self.Time)

    def SetHoldings(self, symbol: str, weight: float) -> None:
        self.Portfolio.mark_to_market(self.Securities)
        target_value = self.Portfolio.TotalPortfolioValue * float(weight)
        price = self.Securities[symbol].price if symbol in self.Securities else 0.0
        if price <= 0:
            return
        target_qty = target_value / price
        current_qty = self.Portfolio[symbol].quantity
        delta = target_qty - current_qty
        if abs(delta) < 1e-12:
            return
        self._apply_fill(symbol, delta, price, tag="SetHoldings")

    def Liquidate(self, symbol: Optional[str] = None, tag: str = "Liquidate") -> None:
        if symbol is None:
            for sym in list(self.Portfolio.Keys):
                if self.Portfolio[sym].Invested:
                    self.Liquidate(sym, tag)
            return
        qty = self.Portfolio[symbol].quantity
        if abs(qty) < 1e-12:
            return
        price = self.Securities[symbol].price if symbol in self.Securities else 0.0
        if price <= 0:
            return
        self._apply_fill(symbol, -qty, price, tag=tag)

    def _apply_fill(self, symbol: str, quantity_delta: float, price: float, tag: str) -> None:
        # Simple commission approximation (Interactive Brokers-like floor).
        notional = abs(quantity_delta) * price
        commission = max(1.0, min(0.005 * abs(quantity_delta), 0.01 * notional))
        holding = self.Portfolio[symbol]
        if quantity_delta > 0:
            new_qty = holding.quantity + quantity_delta
            if new_qty != 0:
                holding.average_price = (
                    (holding.average_price * holding.quantity) + (price * quantity_delta)
                ) / new_qty
            holding.quantity = new_qty
            self.Portfolio.Cash -= notional + commission
        else:
            holding.quantity += quantity_delta
            self.Portfolio.Cash += notional - commission
            if abs(holding.quantity) < 1e-8:
                holding.quantity = 0.0
                holding.average_price = 0.0
        self._orders.append(
            {
                "time": self.Time,
                "symbol": symbol,
                "quantity": quantity_delta,
                "price": price,
                "commission": commission,
                "tag": tag,
            }
        )
        self.Portfolio.mark_to_market(self.Securities)

    def update_smas(self, symbol: str, time: datetime, close: float) -> None:
        for key, sma in self._sma_registry.items():
            sym, _ = key.split(":", 1)
            if sym == symbol:
                sma.Update(time, close)


# Namespaces used by the original LEAN imports style
class Resolution:
    Daily = "Daily"


class Universe:
    Unchanged = "UNCHANGED"


class VolumeShareSlippageModel:
    def __init__(self, volume_limit: float = 0.10) -> None:
        self.volume_limit = volume_limit


class InteractiveBrokersFeeModel:
    pass


@dataclass
class UniverseSettings:
    Resolution: Any = Resolution.Daily
    SlippageModel: Any = None
    FeeModel: Any = None


# Symbol is just a string in the local runtime.
Symbol = str
