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
        "win_rate": {
            "baseline": b_ret.get("win_rate"),
            "experiment": e_ret.get("win_rate"),
            "delta": _delta(b_ret.get("win_rate"), e_ret.get("win_rate")),
        },
        "efficiency_ratio": {
            "baseline": b_r.get("efficiency_ratio"),
            "experiment": e_r.get("efficiency_ratio"),
            "delta": _delta(b_r.get("efficiency_ratio"), e_r.get("efficiency_ratio")),
        },
    }
    exit_breakdown = _exit_breakdown_delta(baseline_trades, experiment_trades)
    return {
        "metrics": rows,
        "exit_breakdown": exit_breakdown,
        "baseline_n_trades": int(len(baseline_trades)) if baseline_trades is not None else 0,
        "experiment_n_trades": int(len(experiment_trades)) if experiment_trades is not None else 0,
    }


def _normalize_exit_family(reason: str) -> str:
    r = str(reason or "unknown").lower()
    if "ema9" in r:
        return "ema9_break"
    if "atr_trail" in r:
        return "atr_trail"
    if "exhaustion" in r:
        return "exhaustion"
    if "rebalance" in r:
        return "rebalance"
    if "bear" in r:
        return "bear_flatten"
    if "delist" in r:
        return "delist"
    if "time_stop" in r:
        return "time_stop"
    return r.split(",")[0][:40] or "unknown"


def _exit_breakdown_delta(
    baseline_trades: pd.DataFrame, experiment_trades: pd.DataFrame
) -> Dict[str, Dict[str, Optional[float]]]:
    def _counts(df: pd.DataFrame) -> Dict[str, float]:
        if df is None or df.empty or "exit_reason" not in df.columns:
            return {}
        fam = df["exit_reason"].map(_normalize_exit_family)
        vc = fam.value_counts(normalize=False)
        n = float(len(df))
        out = {}
        for k, c in vc.items():
            out[str(k)] = {"count": float(c), "share": float(c) / n if n else 0.0}
        return out

    b = _counts(baseline_trades)
    e = _counts(experiment_trades)
    keys = sorted(set(b) | set(e))
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k in keys:
        bc = (b.get(k) or {}).get("count", 0.0)
        ec = (e.get(k) or {}).get("count", 0.0)
        bs = (b.get(k) or {}).get("share", 0.0)
        es = (e.get(k) or {}).get("share", 0.0)
        out[k] = {
            "baseline_count": bc,
            "experiment_count": ec,
            "delta_count": ec - bc,
            "baseline_share": bs,
            "experiment_share": es,
            "delta_share": es - bs,
        }
    return out


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

    def f(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    # Preferred order for the Exp1 brief
    preferred = [
        "turnover_trades_per_year",
        "avg_holding_period_days",
        "cagr",
        "mdd",
        "win_rate",
        "avg_missed_upside",
        "avg_saved_drawdown",
        "n_closed_trades",
        "total_return",
        "efficiency_ratio",
    ]
    metrics = delta.get("metrics") or {}
    ordered = [k for k in preferred if k in metrics] + [k for k in metrics if k not in preferred]
    for name in ordered:
        row = metrics[name]
        lines.append(
            f"{name:28s} {f(row.get('baseline')):>14s} {f(row.get('experiment')):>14s} {f(row.get('delta')):>14s}"
        )

    lines += ["", "Exit Breakdown (count / share)", "-" * 64]
    for fam, row in (delta.get("exit_breakdown") or {}).items():
        lines.append(
            f"{fam:20s} count {f(row.get('baseline_count')):>8s} → {f(row.get('experiment_count')):>8s} "
            f"(Δ {f(row.get('delta_count')):>8s}) | "
            f"share {f(row.get('baseline_share')):>8s} → {f(row.get('experiment_share')):>8s} "
            f"(Δ {f(row.get('delta_share')):>8s})"
        )
    lines.append("=" * 64)
    return "\n".join(lines)
