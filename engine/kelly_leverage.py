"""Kelly-criterion leverage helpers for the confirmed momentum+quality baseline.

Primary estimator uses *daily* excess returns:
  f* = mean(r - r_f) / var(r)

Daily is preferred over monthly because (1) it matches the continuous-time
Kelly formula and the project's Sharpe frequency, and (2) ~1100 observations
vs ~50 months materially reduces estimation noise. Monthly f* is reported as
a sensitivity check only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_RF_PATH = "data/rates/dtb3.csv"


def load_tbill_annual(path: str | Path = DEFAULT_RF_PATH) -> pd.Series:
    """FRED DTB3 CSV → decimal annualized yield series."""
    p = Path(path)
    df = pd.read_csv(p)
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    rate_col = "DTB3" if "DTB3" in df.columns else df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[rate_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
        dtype=float,
    ).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if float(s.abs().median()) > 1.0:
        s = s / 100.0
    return s


def equity_returns(equity: pd.DataFrame) -> pd.Series:
    if equity is None or equity.empty or "Total_Equity" not in equity.columns:
        return pd.Series(dtype=float)
    eq = equity["Total_Equity"].astype(float).dropna()
    return eq.pct_change().dropna()


def align_rf_daily(returns: pd.Series, rf_annual: pd.Series) -> pd.Series:
    """Map annualized T-bill to daily rate on each return date (ffill)."""
    if returns.empty:
        return pd.Series(dtype=float)
    ann = rf_annual.reindex(returns.index).ffill().bfill()
    return ann / 252.0


def kelly_from_returns(
    returns: pd.Series,
    rf_daily: pd.Series,
    *,
    label: str = "",
) -> Dict[str, Any]:
    """Compute f* = mean_excess / variance on the provided return frequency."""
    aligned = pd.concat(
        [returns.rename("r"), rf_daily.rename("rf")], axis=1, join="inner"
    ).dropna()
    n = int(len(aligned))
    if n < 5:
        return {
            "label": label,
            "n": n,
            "mean_excess": None,
            "variance": None,
            "f_star": None,
            "sharpe_excess": None,
        }
    r = aligned["r"]
    ex = r - aligned["rf"]
    mean_ex = float(ex.mean())
    var = float(r.var(ddof=1))
    std = float(r.std(ddof=1))
    f_star = float(mean_ex / var) if var > 0 else None
    sharpe = float(mean_ex / std * np.sqrt(252)) if std > 0 else None
    return {
        "label": label,
        "n": n,
        "mean_return": float(r.mean()),
        "mean_excess": mean_ex,
        "mean_rf": float(aligned["rf"].mean()),
        "variance": var,
        "std": std,
        "f_star": f_star,
        "sharpe_excess": sharpe,
        # Annualized companions (same f* — units cancel)
        "mean_excess_ann": mean_ex * 252.0,
        "mean_return_ann": float(r.mean()) * 252.0,
        "mean_rf_ann": float(aligned["rf"].mean()) * 252.0,
        "variance_ann": var * 252.0,
        "vol_ann": std * np.sqrt(252.0),
    }


def kelly_daily_from_equity(
    equity: pd.DataFrame,
    rf_annual: pd.Series,
    *,
    label: str = "full",
) -> Dict[str, Any]:
    r = equity_returns(equity)
    rf_d = align_rf_daily(r, rf_annual)
    out = kelly_from_returns(r, rf_d, label=label)
    out["frequency"] = "daily"
    return out


def kelly_monthly_from_equity(
    equity: pd.DataFrame,
    rf_annual: pd.Series,
    *,
    label: str = "full",
) -> Dict[str, Any]:
    if equity is None or equity.empty:
        return {"label": label, "frequency": "monthly", "n": 0, "f_star": None}
    eq = equity["Total_Equity"].astype(float).dropna()
    m = eq.resample("ME").last().pct_change().dropna()
    rf_m_ann = rf_annual.reindex(m.index).ffill().bfill()
    rf_m = rf_m_ann / 12.0
    aligned = pd.concat([m.rename("r"), rf_m.rename("rf")], axis=1).dropna()
    n = int(len(aligned))
    if n < 3:
        return {"label": label, "frequency": "monthly", "n": n, "f_star": None}
    r = aligned["r"]
    ex = r - aligned["rf"]
    mean_ex = float(ex.mean())
    var = float(r.var(ddof=1))
    f_star = float(mean_ex / var) if var > 0 else None
    return {
        "label": label,
        "frequency": "monthly",
        "n": n,
        "mean_excess": mean_ex,
        "mean_excess_ann": mean_ex * 12.0,
        "variance": var,
        "variance_ann": var * 12.0,
        "f_star": f_star,
        "sharpe_excess": float(ex.mean() / r.std(ddof=1) * np.sqrt(12))
        if float(r.std(ddof=1)) > 0
        else None,
    }


def fractional_levels(f_star: float, fractions: Sequence[float] = (0.25, 0.50, 0.75)) -> Dict[str, float]:
    """Map fraction labels → fixed leverage multiples (can be < 1)."""
    return {f"{int(100 * frac)}pct_kelly": float(f_star) * float(frac) for frac in fractions}


def subperiod_kelly(
    equity: pd.DataFrame,
    rf_annual: pd.Series,
    windows: Sequence[Tuple[str, str, str]],
) -> List[Dict[str, Any]]:
    """windows: list of (label, start, end) inclusive date strings."""
    rows = []
    for label, start, end in windows:
        sl = equity.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        rows.append(kelly_daily_from_equity(sl, rf_annual, label=label))
    return rows


def worst_month_and_mdd(
    equity: pd.DataFrame,
    initial_capital: float = 50_000_000.0,
) -> Dict[str, Any]:
    """Worst calendar-month return + peak-to-trough MDD in % and dollars."""
    if equity is None or equity.empty:
        return {}
    eq = equity["Total_Equity"].astype(float).dropna()
    month_end = eq.resample("ME").last()
    # Include first partial month from first equity point
    month_rets = month_end.pct_change()
    # First month: from first day in that month to month-end
    first_month = eq.index[0].to_period("M")
    first_slice = eq[eq.index.to_period("M") == first_month]
    if len(first_slice) >= 2:
        month_rets.iloc[0] = float(first_slice.iloc[-1] / first_slice.iloc[0] - 1.0)
    month_rets = month_rets.dropna()
    worst_i = month_rets.idxmin()
    worst_ret = float(month_rets.loc[worst_i])
    peak = eq.cummax()
    dd = (eq - peak) / peak
    mdd = float(dd.min())
    mdd_date = dd.idxmin()
    # Dollar impact relative to starting capital (simple: apply % to initial)
    return {
        "worst_month": str(pd.Timestamp(worst_i).date()),
        "worst_month_return": worst_ret,
        "worst_month_dollar_on_initial": worst_ret * float(initial_capital),
        "mdd": mdd,
        "mdd_date": str(pd.Timestamp(mdd_date).date()),
        "mdd_dollar_on_initial": mdd * float(initial_capital),
        # Also report dollars from the actual peak equity at MDD trough
        "mdd_dollar_from_peak": float((eq.loc[mdd_date] - peak.loc[mdd_date])),
        "peak_equity_at_mdd": float(peak.loc[mdd_date]),
        "trough_equity_at_mdd": float(eq.loc[mdd_date]),
        "final_equity": float(eq.iloc[-1]),
        "initial_equity": float(eq.iloc[0]),
    }


def long_only_leverage(f_star: Optional[float]) -> Optional[float]:
    """Clamp negative Kelly to 0 for long-only interpretation."""
    if f_star is None:
        return None
    return float(max(0.0, f_star))
