"""Load local price and fundamental datasets from the data/ directory."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


class DataStore:
    """Read-only access to prices/ and fundamentals/ on disk."""

    def __init__(self, prices_dir: str | Path, fundamentals_dir: str | Path, metadata_dir: str | Path) -> None:
        self.prices_dir = Path(prices_dir)
        self.fundamentals_dir = Path(fundamentals_dir)
        self.metadata_dir = Path(metadata_dir)
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._fund_cache: Dict[str, pd.DataFrame] = {}

    def list_price_symbols(self) -> List[str]:
        symbols = set()
        for path in self.prices_dir.glob("*"):
            if path.suffix.lower() in {".csv", ".parquet"}:
                symbols.add(path.stem.upper())
        return sorted(symbols)

    def load_prices(self, symbol: str) -> pd.DataFrame:
        symbol = symbol.upper()
        if symbol in self._price_cache:
            return self._price_cache[symbol]

        parquet = self.prices_dir / f"{symbol}.parquet"
        csv = self.prices_dir / f"{symbol}.csv"
        if parquet.exists():
            df = pd.read_parquet(parquet)
        elif csv.exists():
            df = pd.read_csv(csv)
        else:
            return pd.DataFrame()

        df.columns = [str(c).strip().lower() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        else:
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"{symbol} prices missing column: {col}")
        if "volume" not in df.columns:
            df["volume"] = 0.0
        self._price_cache[symbol] = df
        return df

    def load_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Return PIT fundamental snapshots indexed by as_of_date."""
        symbol = symbol.upper()
        if symbol in self._fund_cache:
            return self._fund_cache[symbol]

        parquet = self.fundamentals_dir / f"{symbol}.parquet"
        csv = self.fundamentals_dir / f"{symbol}.csv"
        if parquet.exists():
            df = pd.read_parquet(parquet)
        elif csv.exists():
            df = pd.read_csv(csv)
        else:
            return pd.DataFrame()

        df.columns = [str(c).strip().lower() for c in df.columns]
        date_col = "as_of_date" if "as_of_date" in df.columns else ("date" if "date" in df.columns else None)
        if date_col is None:
            raise ValueError(f"{symbol} fundamentals missing as_of_date/date column")
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)
        self._fund_cache[symbol] = df
        return df

    def trading_calendar(self, benchmark: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        prices = self.load_prices(benchmark)
        if prices.empty:
            raise FileNotFoundError(
                f"Benchmark prices not found for {benchmark} under {self.prices_dir}. "
                "Run the downloader first."
            )
        idx = prices.index[(prices.index >= start) & (prices.index <= end)]
        return pd.DatetimeIndex(idx)

    def price_on(self, symbol: str, as_of: pd.Timestamp) -> Optional[pd.Series]:
        df = self.load_prices(symbol)
        if df.empty:
            return None
        hist = df.loc[:as_of]
        if hist.empty:
            return None
        return hist.iloc[-1]

    def history_panel(self, symbols: Iterable[str], periods: int, as_of: pd.Timestamp) -> pd.DataFrame:
        frames = []
        for symbol in symbols:
            df = self.load_prices(symbol)
            if df.empty:
                continue
            hist = df.loc[:as_of].tail(periods)
            if hist.empty:
                continue
            part = hist.copy()
            part["symbol"] = symbol
            frames.append(part)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames)
        out = out.set_index("symbol", append=True).swaplevel(0, 1).sort_index()
        # index: symbol, date
        out.index.names = ["symbol", "time"]
        return out
