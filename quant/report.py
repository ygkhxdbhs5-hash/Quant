"""Text and chart reporting for backtest results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from quant.engine import BacktestResult


def print_report(result: BacktestResult, strategy_name: str = "") -> None:
    header = f"Strategy: {strategy_name}" if strategy_name else "Strategy"
    print(header)
    print(result.metrics.summary())
    if result.trades:
        print("\n=== Recent Trades ===")
        for trade in result.trades[-10:]:
            print(
                f"{trade.timestamp:%Y-%m-%d} {trade.side:4} "
                f"qty={trade.quantity:.4f} @ {trade.price:.2f} "
                f"comm={trade.commission:.2f}"
            )


def plot_equity(
    result: BacktestResult,
    price: Optional[pd.Series] = None,
    output_path: Optional[str | Path] = None,
    show: bool = False,
) -> Path | None:
    """Plot equity curve (and optional price) and optionally save to disk."""
    fig, axes = plt.subplots(
        2 if price is not None else 1,
        1,
        figsize=(10, 6 if price is not None else 4),
        sharex=True,
    )
    if price is not None:
        ax_price, ax_eq = axes
        ax_price.plot(price.index, price.values, color="#1f4e79", linewidth=1.2)
        ax_price.set_ylabel("Price")
        ax_price.set_title("Price")
        buys = [t for t in result.trades if t.side == "BUY"]
        sells = [t for t in result.trades if t.side == "SELL"]
        if buys:
            ax_price.scatter(
                [t.timestamp for t in buys],
                [t.price for t in buys],
                marker="^",
                color="#2e7d32",
                s=40,
                label="Buy",
                zorder=3,
            )
        if sells:
            ax_price.scatter(
                [t.timestamp for t in sells],
                [t.price for t in sells],
                marker="v",
                color="#c62828",
                s=40,
                label="Sell",
                zorder=3,
            )
        if buys or sells:
            ax_price.legend(loc="upper left")
    else:
        ax_eq = axes

    ax_eq.plot(result.equity.index, result.equity.values, color="#0d47a1", linewidth=1.4)
    ax_eq.set_ylabel("Equity")
    ax_eq.set_title("Equity Curve")
    ax_eq.grid(True, alpha=0.25)
    fig.tight_layout()

    saved: Path | None = None
    if output_path is not None:
        saved = Path(output_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return saved
