"""Performance metrics for backtest results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PerformanceReport:
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    final_equity: float
    initial_cash: float

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "=== Backtest Performance ===",
            f"Initial cash   : ${self.initial_cash:,.2f}",
            f"Final equity   : ${self.final_equity:,.2f}",
            f"Total return   : {self.total_return * 100:.2f}%",
            f"CAGR           : {self.cagr * 100:.2f}%",
            f"Sharpe ratio   : {self.sharpe:.2f}",
            f"Max drawdown   : {self.max_drawdown * 100:.2f}%",
            f"Win rate       : {self.win_rate * 100:.2f}%",
            f"Profit factor  : {self.profit_factor:.2f}",
            f"Trades         : {self.num_trades}",
        ]
        return "\n".join(lines)


def compute_metrics(
    equity: pd.Series,
    trades: Iterable,
    initial_cash: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceReport:
    if equity.empty:
        return PerformanceReport(
            total_return=0.0,
            cagr=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            num_trades=0,
            final_equity=float(initial_cash),
            initial_cash=float(initial_cash),
        )

    final_equity = float(equity.iloc[-1])
    total_return = (final_equity / initial_cash) - 1.0

    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    cagr = (final_equity / initial_cash) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    returns = equity.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        excess = returns - (risk_free_rate / periods_per_year)
        sharpe = float(np.sqrt(periods_per_year) * excess.mean() / excess.std())
    else:
        sharpe = 0.0

    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    round_trips = _round_trip_pnls(trades)
    wins = [p for p in round_trips if p > 0]
    losses = [p for p in round_trips if p < 0]
    win_rate = len(wins) / len(round_trips) if round_trips else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    return PerformanceReport(
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor) if np.isfinite(profit_factor) else 0.0,
        num_trades=len(round_trips),
        final_equity=final_equity,
        initial_cash=float(initial_cash),
    )


def _round_trip_pnls(trades: Iterable) -> list[float]:
    """Pair BUY then SELL fills into round-trip PnLs (commission-aware)."""
    pnls: list[float] = []
    entry_cost = 0.0
    entry_qty = 0.0

    for trade in trades:
        if trade.side == "BUY":
            entry_cost += trade.price * trade.quantity + trade.commission
            entry_qty += trade.quantity
        elif trade.side == "SELL" and entry_qty > 0:
            exit_proceeds = trade.price * trade.quantity - trade.commission
            # Allocate entry cost pro-rata if partial exits ever occur.
            allocated = entry_cost * (trade.quantity / entry_qty)
            pnls.append(exit_proceeds - allocated)
            entry_cost -= allocated
            entry_qty -= trade.quantity
            if entry_qty < 1e-12:
                entry_cost = 0.0
                entry_qty = 0.0
    return pnls
