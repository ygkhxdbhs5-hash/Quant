"""Unit tests for the backtest engine and metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from quant.data import generate_sample_ohlcv
from quant.engine import BacktestEngine
from quant.metrics import compute_metrics
from quant.portfolio import Portfolio
from quant.strategy import Signal, Strategy
from strategies.sma_crossover import SMACrossover
from strategies.your_strategy import YourStrategy


class AlwaysBuy(Strategy):
    name = "AlwaysBuy"

    def on_bar(self, window: pd.DataFrame) -> Signal:
        return Signal.BUY if len(window) == 5 else Signal.HOLD


class BuyThenSell(Strategy):
    name = "BuyThenSell"

    def on_bar(self, window: pd.DataFrame) -> Signal:
        n = len(window)
        if n == 5:
            return Signal.BUY
        if n == 10:
            return Signal.SELL
        return Signal.HOLD


def test_portfolio_buy_sell_round_trip():
    p = Portfolio(initial_cash=10_000, commission_rate=0.0)
    ts = pd.Timestamp("2024-01-02").to_pydatetime()
    assert p.buy(ts, price=100.0, quantity=10) is not None
    assert p.position == 10
    assert p.cash == pytest.approx(9_000)
    assert p.sell(ts, price=110.0) is not None
    assert p.position == 0
    assert p.cash == pytest.approx(10_100)


def test_engine_runs_on_sample_data():
    data = generate_sample_ohlcv(n_bars=120, seed=7)
    engine = BacktestEngine(SMACrossover(fast=5, slow=20), initial_cash=50_000)
    result = engine.run(data)
    assert len(result.equity) == len(data)
    assert result.metrics.final_equity > 0
    assert result.metrics.initial_cash == 50_000


def test_engine_executes_pending_on_next_open():
    data = generate_sample_ohlcv(n_bars=30, seed=1)
    engine = BacktestEngine(BuyThenSell(), commission_rate=0.0)
    result = engine.run(data)
    # One buy + one sell
    sides = [t.side for t in result.trades]
    assert sides.count("BUY") == 1
    assert sides.count("SELL") == 1
    # Fill happens on the bar after the signal
    buy = next(t for t in result.trades if t.side == "BUY")
    assert buy.timestamp == data.index[5].to_pydatetime()
    assert buy.price == pytest.approx(float(data.iloc[5]["open"]))


def test_metrics_total_return():
    idx = pd.bdate_range("2024-01-01", periods=5)
    equity = pd.Series([100.0, 110.0, 105.0, 120.0, 130.0], index=idx)
    report = compute_metrics(equity, trades=[], initial_cash=100.0)
    assert report.total_return == pytest.approx(0.30)
    assert report.final_equity == 130.0


def test_your_strategy_template_emits_valid_signals():
    data = generate_sample_ohlcv(n_bars=80, seed=3)
    strat = YourStrategy(lookback=10)
    for i in range(len(data)):
        signal = strat.on_bar(data.iloc[: i + 1])
        assert signal in (Signal.BUY, Signal.SELL, Signal.HOLD)


def test_missing_columns_raise():
    bad = pd.DataFrame({"close": [1, 2, 3]}, index=pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(ValueError, match="Missing required"):
        BacktestEngine(AlwaysBuy()).run(bad)
