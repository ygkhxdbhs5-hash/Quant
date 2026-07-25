"""Thin PIT fundamentals helpers for Baseline variants.

Reuses the same on-disk artifact and as-of lookup pattern as StandaloneEngine
(``data/fundamentals/pit_history.pkl``, filing_date ≤ as_of). Does not modify
downloader or StandaloneEngine internals.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


QUALITY_FIELDS = ("gross_profitability", "roic", "op_margin")


def load_pit_history(fundamentals_dir: str | Path = "data/fundamentals") -> Dict[str, pd.DataFrame]:
    path = Path(fundamentals_dir) / "pit_history.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        hist = pickle.load(f) or {}
    if not isinstance(hist, dict):
        return {}
    return hist


def build_fund_ts_index(fundamental_history: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
    """Precompute filing-date arrays for searchsorted (same as StandaloneEngine)."""
    out: Dict[str, np.ndarray] = {}
    for sym, hist in (fundamental_history or {}).items():
        if hist is None or getattr(hist, "empty", True):
            continue
        try:
            idx = pd.to_datetime(hist.index)
            out[str(sym)] = idx.to_numpy(dtype="datetime64[ns]")
        except Exception:
            continue
    return out


def get_latest_available_fundamentals(
    fundamental_history: Dict[str, pd.DataFrame],
    fund_ts: Dict[str, np.ndarray],
    symbol: str,
    as_of_date,
) -> Optional[pd.Series]:
    """Return latest as-reported fundamentals with filing_date ≤ as_of_date."""
    hist = fundamental_history.get(symbol)
    if hist is None or hist.empty:
        return None
    ts = fund_ts.get(symbol)
    if ts is not None and len(ts):
        try:
            asof = np.datetime64(pd.Timestamp(as_of_date).to_datetime64())
            i = int(np.searchsorted(ts, asof, side="right") - 1)
            if i < 0:
                return None
            return hist.iloc[i]
        except Exception:
            pass
    past = hist.loc[:as_of_date]
    if past.empty:
        return None
    return past.iloc[-1]


def extract_quality_raw(row: Optional[pd.Series]) -> Optional[Dict[str, float]]:
    """Return GP / ROIC / op_margin only if all three are finite. Else None (exclude)."""
    if row is None:
        return None
    vals: Dict[str, float] = {}
    for field in QUALITY_FIELDS:
        if field not in row.index:
            return None
        v = row[field]
        try:
            fv = float(v)
        except Exception:
            return None
        if not np.isfinite(fv):
            return None
        vals[field] = fv
    return vals
