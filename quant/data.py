"""Market data helpers: CSV load and synthetic OHLCV generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from CSV.

    Accepts a ``date``/``datetime`` column or a datetime index.
    Column names are normalized to lowercase.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    for candidate in ("date", "datetime", "timestamp"):
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate])
            df = df.set_index(candidate)
            break
    else:
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


def generate_sample_ohlcv(
    n_bars: int = 504,
    start: str = "2022-01-03",
    seed: int = 42,
    start_price: float = 100.0,
    annual_vol: float = 0.20,
    drift: float = 0.08,
) -> pd.DataFrame:
    """Generate reproducible geometric-Brownian-motion daily OHLCV."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start=start, periods=n_bars)
    dt = 1.0 / 252.0
    shocks = rng.normal((drift - 0.5 * annual_vol**2) * dt, annual_vol * np.sqrt(dt), size=n_bars)
    close = start_price * np.exp(np.cumsum(shocks))

    # Build OHLC around close with small intraday ranges.
    intraday = np.abs(rng.normal(0.0, 0.005, size=n_bars))
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + intraday)
    low = np.minimum(open_, close) * (1.0 - intraday)
    volume = rng.integers(100_000, 1_000_000, size=n_bars).astype(float)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "date"
    out.to_csv(path)
    return path
