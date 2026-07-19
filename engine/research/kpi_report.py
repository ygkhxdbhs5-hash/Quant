"""Hierarchical KPI summary (research → risk → return)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _safe_mean(s: pd.Series) -> Optional[float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def build_hierarchical_kpi_report(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    rf_daily: float = 0.0,
) -> Dict[str, Any]:
    """Return prioritized KPI dict. Equity index = Date, col Total_Equity."""
    report: Dict[str, Any] = {"research": {}, "risk": {}, "return": {}}

    # --- Research metrics ---
    if trades is not None and not trades.empty:
        hold = pd.to_numeric(trades.get("holding_days"), errors="coerce")
        missed = pd.to_numeric(trades.get("missed_upside"), errors="coerce")
        saved = pd.to_numeric(trades.get("saved_drawdown"), errors="coerce")
        # Efficiency: saved_drawdown / (missed_upside + eps) — higher means exits protect more vs upside given up
        eff = saved / (missed.abs() + 1e-8)
        n_entries = len(trades)
        avg_hold = _safe_mean(hold)
        # Turnover proxy: closed trades / years
        report["research"] = {
            "n_closed_trades": int(n_entries),
            "avg_holding_period_days": avg_hold,
            "avg_missed_upside": _safe_mean(missed),
            "avg_saved_drawdown": _safe_mean(saved),
            "efficiency_ratio": _safe_mean(eff),
            "turnover_trades_per_year": None,
        }
    else:
        report["research"] = {
            "n_closed_trades": 0,
            "avg_holding_period_days": None,
            "avg_missed_upside": None,
            "avg_saved_drawdown": None,
            "efficiency_ratio": None,
            "turnover_trades_per_year": None,
        }

    if equity is None or equity.empty or "Total_Equity" not in equity.columns:
        report["risk"] = {}
        report["return"] = {}
        return report

    eq = equity["Total_Equity"].astype(float).dropna()
    if len(eq) < 2:
        return report

    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if eq.iloc[0] > 0 else None
    peak = eq.cummax()
    dd = (eq - peak) / peak
    mdd = float(dd.min())
    vol = float(rets.std() * np.sqrt(252)) if len(rets) else None
    mean_r = float(rets.mean() * 252) if len(rets) else None
    sharpe = (mean_r - rf_daily * 252) / vol if vol and vol > 0 and mean_r is not None else None
    downside = rets[rets < 0]
    dvol = float(downside.std() * np.sqrt(252)) if len(downside) else None
    sortino = (mean_r - rf_daily * 252) / dvol if dvol and dvol > 0 and mean_r is not None else None
    calmar = (cagr / abs(mdd)) if cagr is not None and mdd < 0 else None

    if report["research"].get("n_closed_trades"):
        report["research"]["turnover_trades_per_year"] = float(
            report["research"]["n_closed_trades"] / years
        )

    report["risk"] = {
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
    }

    # Alpha vs benchmark if provided
    alpha = None
    if benchmark_returns is not None and len(benchmark_returns):
        aligned = pd.concat([rets, benchmark_returns.rename("bm")], axis=1, join="inner").dropna()
        if len(aligned) > 5:
            excess = aligned.iloc[:, 0] - aligned["bm"]
            alpha = float(excess.mean() * 252)

    win_rate = None
    profit_factor = None
    if trades is not None and not trades.empty and "final_return" in trades.columns:
        fr = pd.to_numeric(trades["final_return"], errors="coerce").dropna()
        if len(fr):
            win_rate = float((fr > 0).mean())
            gains = fr[fr > 0].sum()
            losses = (-fr[fr < 0]).sum()
            profit_factor = float(gains / losses) if losses > 0 else None

    report["return"] = {
        "total_return": total_return,
        "alpha": alpha,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }
    return report


def format_kpi_report(report: Dict[str, Any]) -> str:
    lines = ["=" * 64, " HIERARCHICAL KPI SUMMARY", "=" * 64]
    lines.append("1) Research Metrics")
    for k, v in (report.get("research") or {}).items():
        lines.append(f"   - {k}: {_fmt(v)}")
    lines.append("2) Risk Metrics")
    for k, v in (report.get("risk") or {}).items():
        lines.append(f"   - {k}: {_fmt(v)}")
    lines.append("3) Return Metrics")
    for k, v in (report.get("return") or {}).items():
        lines.append(f"   - {k}: {_fmt(v)}")
    lines.append("=" * 64)
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "n/a"
    if isinstance(v, float):
        if abs(v) < 2:
            return f"{v:.4f} ({v*100:.2f}%)" if abs(v) <= 1.5 else f"{v:.4f}"
        return f"{v:.4f}"
    return str(v)
