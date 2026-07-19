"""Delta report Fact: Exp vs Baseline absolute changes."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _num(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _delta(a, b) -> Optional[float]:
    a, b = _num(a), _num(b)
    if a is None or b is None:
        return None
    return b - a


def build_delta_report(
    baseline_kpi: Dict[str, Any],
    experiment_kpi: Dict[str, Any],
    baseline_trades: pd.DataFrame,
    experiment_trades: pd.DataFrame,
) -> Dict[str, Any]:
    b_r = baseline_kpi.get("research") or {}
    e_r = experiment_kpi.get("research") or {}
    b_k = baseline_kpi.get("risk") or {}
    e_k = experiment_kpi.get("risk") or {}
    b_ret = baseline_kpi.get("return") or {}
    e_ret = experiment_kpi.get("return") or {}

    rows = {
        "n_closed_trades": {
            "baseline": b_r.get("n_closed_trades"),
            "experiment": e_r.get("n_closed_trades"),
            "delta": _delta(b_r.get("n_closed_trades"), e_r.get("n_closed_trades")),
        },
        "turnover_trades_per_year": {
            "baseline": b_r.get("turnover_trades_per_year"),
            "experiment": e_r.get("turnover_trades_per_year"),
            "delta": _delta(b_r.get("turnover_trades_per_year"), e_r.get("turnover_trades_per_year")),
        },
        "avg_holding_period_days": {
            "baseline": b_r.get("avg_holding_period_days"),
            "experiment": e_r.get("avg_holding_period_days"),
            "delta": _delta(b_r.get("avg_holding_period_days"), e_r.get("avg_holding_period_days")),
        },
        "cagr": {
            "baseline": b_k.get("cagr"),
            "experiment": e_k.get("cagr"),
            "delta": _delta(b_k.get("cagr"), e_k.get("cagr")),
        },
        "mdd": {
            "baseline": b_k.get("mdd"),
            "experiment": e_k.get("mdd"),
            "delta": _delta(b_k.get("mdd"), e_k.get("mdd")),
        },
        "total_return": {
            "baseline": b_ret.get("total_return"),
            "experiment": e_ret.get("total_return"),
            "delta": _delta(b_ret.get("total_return"), e_ret.get("total_return")),
        },
        "avg_missed_upside": {
            "baseline": b_r.get("avg_missed_upside"),
            "experiment": e_r.get("avg_missed_upside"),
            "delta": _delta(b_r.get("avg_missed_upside"), e_r.get("avg_missed_upside")),
        },
        "avg_saved_drawdown": {
            "baseline": b_r.get("avg_saved_drawdown"),
            "experiment": e_r.get("avg_saved_drawdown"),
            "delta": _delta(b_r.get("avg_saved_drawdown"), e_r.get("avg_saved_drawdown")),
        },
        "efficiency_ratio": {
            "baseline": b_r.get("efficiency_ratio"),
            "experiment": e_r.get("efficiency_ratio"),
            "delta": _delta(b_r.get("efficiency_ratio"), e_r.get("efficiency_ratio")),
        },
    }
    return {
        "metrics": rows,
        "baseline_n_trades": int(len(baseline_trades)) if baseline_trades is not None else 0,
        "experiment_n_trades": int(len(experiment_trades)) if experiment_trades is not None else 0,
    }


def format_delta_report(delta: Dict[str, Any]) -> str:
    lines = [
        "=" * 64,
        " DELTA REPORT (Experiment − Baseline)",
        "=" * 64,
        f"baseline_n_trades={delta.get('baseline_n_trades')}  experiment_n_trades={delta.get('experiment_n_trades')}",
        "",
        f"{'metric':28s} {'baseline':>14s} {'experiment':>14s} {'delta':>14s}",
        "-" * 64,
    ]
    for name, row in (delta.get("metrics") or {}).items():
        def f(v):
            if v is None:
                return "n/a"
            if isinstance(v, float):
                return f"{v:.6f}"
            return str(v)

        lines.append(
            f"{name:28s} {f(row.get('baseline')):>14s} {f(row.get('experiment')):>14s} {f(row.get('delta')):>14s}"
        )
    lines.append("=" * 64)
    return "\n".join(lines)
