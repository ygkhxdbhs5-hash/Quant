"""Cross-sectional Z-score helpers (PIT; no look-ahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(values: pd.Series, *, clip: float = 5.0) -> pd.Series:
    """Standardize a cross-section: (x − μ) / σ using only non-NaN peers.

    Missing inputs stay NaN. Degenerate σ → all zeros for observed names.
    """
    try:
        x = pd.to_numeric(values, errors="coerce")
        mu = x.mean(skipna=True)
        sigma = x.std(skipna=True, ddof=0)
        if pd.isna(mu):
            return pd.Series(np.nan, index=values.index, dtype=float)
        if pd.isna(sigma) or float(sigma) == 0.0:
            out = pd.Series(np.nan, index=values.index, dtype=float)
            out = out.where(x.isna(), 0.0)
            return out.astype(float)
        z = (x - float(mu)) / float(sigma)
        if clip is not None and clip > 0:
            z = z.clip(lower=-float(clip), upper=float(clip))
        return z.astype(float)
    except Exception:
        return pd.Series(np.nan, index=values.index, dtype=float)


def nanmean_row(df: pd.DataFrame) -> pd.Series:
    """Row-wise nanmean; all-NaN rows → 0.0 (cross-sectional neutral)."""
    if df.empty:
        return pd.Series(dtype=float)
    try:
        m = df.mean(axis=1, skipna=True)
        return m.fillna(0.0).astype(float)
    except Exception:
        return pd.Series(0.0, index=df.index, dtype=float)
