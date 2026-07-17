"""Local backtest loop: feeds on-disk data into the unchanged strategy pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set

import pandas as pd

from engine.data_store import DataStore
from engine.fine_adapter import CoarseFundamental, SecurityChanges, fine_from_row
from engine.runtime import Slice, TradeBar
from engine.strategy import NasdaqInstitutionalProductionEngine


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    orders: list
    logs: list
    final_value: float


class BacktestRunner:
    """Runs NasdaqInstitutionalProductionEngine against local data only."""

    def __init__(self, config: dict, store: DataStore) -> None:
        self.config = config
        self.store = store
        self.engine = NasdaqInstitutionalProductionEngine()
        self.engine.Initialize(config)
        self.engine._history_provider = self._history_provider
        self._current_universe: Set[str] = set()

    def _history_provider(self, symbols: List[str], periods: int, as_of: datetime) -> pd.DataFrame:
        return self.store.history_panel(symbols, periods, pd.Timestamp(as_of))

    def run(self) -> BacktestResult:
        start = pd.Timestamp(self.engine.StartDate)
        end = pd.Timestamp(self.engine.EndDate)
        calendar = self.store.trading_calendar(self.engine.benchmark, start, end)

        # Warm-up benchmark SMA history before the official start when possible.
        self._warmup_benchmark(calendar)

        last_month = None
        for ts in calendar:
            self.engine.Time = ts.to_pydatetime()
            self._update_prices(ts)

            month_key = (ts.month, ts.year)
            if month_key != last_month:
                self._monthly_universe_update(ts)
                last_month = month_key
                # Month-start rebalance (same cadence as original Schedule.On MonthStart)
                self.engine.RebalanceStrategy()

            slice_ = self._build_slice(ts)
            self.engine.OnData(slice_)
            self.engine.Portfolio.mark_to_market(self.engine.Securities)
            self.engine._equity_curve.append((ts, self.engine.Portfolio.TotalPortfolioValue))

        equity = pd.Series(
            [v for _, v in self.engine._equity_curve],
            index=pd.DatetimeIndex([t for t, _ in self.engine._equity_curve]),
            name="equity",
        )
        return BacktestResult(
            equity_curve=equity,
            orders=list(self.engine._orders),
            logs=list(self.engine._logs),
            final_value=float(equity.iloc[-1]) if len(equity) else float(self.engine.Portfolio.TotalPortfolioValue),
        )

    def _warmup_benchmark(self, calendar: pd.DatetimeIndex) -> None:
        if calendar.empty:
            return
        first = calendar[0]
        bench = self.store.load_prices(self.engine.benchmark)
        hist = bench.loc[:first].tail(250)
        for ts, row in hist.iterrows():
            self.engine.Time = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            px = float(row["close"])
            self.engine.Securities[self.engine.benchmark].price = px
            self.engine.update_smas(self.engine.benchmark, self.engine.Time, px)

    def _update_prices(self, ts: pd.Timestamp) -> None:
        symbols = set(self._current_universe) | {self.engine.benchmark}
        for symbol in symbols:
            row = self.store.price_on(symbol, ts)
            if row is None:
                continue
            if symbol not in self.engine.Securities:
                self.engine.AddEquity(symbol)
            sec = self.engine.Securities[symbol]
            sec.price = float(row["close"])
            sec.volume = float(row.get("volume", 0.0))
            self.engine.update_smas(symbol, ts.to_pydatetime(), sec.price)

    def _build_slice(self, ts: pd.Timestamp) -> Slice:
        slice_ = Slice()
        for symbol in list(self._current_universe) + [self.engine.benchmark]:
            row = self.store.price_on(symbol, ts)
            if row is None:
                continue
            slice_.Bars.set(
                symbol,
                TradeBar(
                    Open=float(row["open"]),
                    High=float(row["high"]),
                    Low=float(row["low"]),
                    Close=float(row["close"]),
                    Volume=float(row.get("volume", 0.0)),
                ),
            )
        return slice_

    def _monthly_universe_update(self, ts: pd.Timestamp) -> None:
        coarse = self._build_coarse(ts)
        selected = self.engine.CoarseSelectionFunction(coarse)
        if selected == "UNCHANGED":
            return

        fine = self._build_fine(selected, ts)
        fine_selected = self.engine.FineSelectionFunction(fine)
        if fine_selected == "UNCHANGED":
            return

        new_universe = set(fine_selected)
        added = new_universe - self._current_universe
        removed = self._current_universe - new_universe

        for symbol in added:
            if symbol not in self.engine.Securities:
                self.engine.AddEquity(symbol)
            row = self.store.price_on(symbol, ts)
            if row is not None:
                self.engine.Securities[symbol].price = float(row["close"])
                self.engine.Securities[symbol].volume = float(row.get("volume", 0.0))

        changes = SecurityChanges(
            AddedSecurities=[self.engine.Securities[s] for s in added if s in self.engine.Securities],
            RemovedSecurities=[self.engine.Securities[s] for s in removed if s in self.engine.Securities],
        )
        self.engine.OnSecuritiesChanged(changes)
        # LEAN warms indicator consolidators from history; mirror that here.
        self._warm_added_indicators(added, ts)
        self._current_universe = set(self.engine.active_universe)

    def _warm_added_indicators(self, symbols: Set[str], ts: pd.Timestamp) -> None:
        for symbol in symbols:
            hist = self.store.load_prices(symbol)
            if hist.empty:
                continue
            window = hist.loc[:ts].tail(self.engine.MOM_WINDOW + 5)
            for bar_ts, row in window.iterrows():
                t = bar_ts.to_pydatetime() if hasattr(bar_ts, "to_pydatetime") else bar_ts
                self.engine.update_smas(symbol, t, float(row["close"]))

    def _build_coarse(self, ts: pd.Timestamp) -> List[CoarseFundamental]:
        out: List[CoarseFundamental] = []
        for symbol in self.store.list_price_symbols():
            if symbol == self.engine.benchmark:
                continue
            px = self.store.price_on(symbol, ts)
            if px is None:
                continue
            has_fund = not self.store.load_fundamentals(symbol).empty
            price = float(px["close"])
            volume = float(px.get("volume", 0.0))
            out.append(
                CoarseFundamental(
                    Symbol=symbol,
                    Price=price,
                    Volume=volume,
                    HasFundamentalData=has_fund,
                    DollarVolume=price * volume,
                )
            )
        return out

    def _build_fine(self, symbols: List[str], ts: pd.Timestamp) -> list:
        fine = []
        for symbol in symbols:
            fund = self.store.load_fundamentals(symbol)
            if fund.empty:
                continue
            hist = fund.loc[:ts]
            if hist.empty:
                continue
            row = hist.iloc[-1].to_dict()
            # Prefer latest available PIT row at or before ts
            px = self.store.price_on(symbol, ts)
            if px is not None and "dollar_volume" not in row:
                row["dollar_volume"] = float(px["close"]) * float(px.get("volume", 0.0))
            elif "dollar_volume" not in row:
                row["dollar_volume"] = 0.0
            fine.append(fine_from_row(symbol, row))
        return fine
