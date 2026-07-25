#!/usr/bin/env python3
"""No-chase baseline vs QQQ buy & hold — fingerprinted clean-state report.

Uses existing BaselineEngineV1 QQQ B&H infrastructure:
  - Buy full notional once at the first eligible trading day's **Open**
    on/after START (same cash base as strategy).
  - Mark equity daily at **Close** (consistent with strategy equity curve).
  - No rebalancing, no leverage, no trading costs on the QQQ leg.

Must run with a clean git working tree (git_dirty=False).

Usage:
  python3 run_nochase_qqq_alpha_report.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from engine.baseline_diagnostics import _cagr_sharpe_mdd
from engine.baseline_engine import BaselineEngineV1
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
from engine.topup_chase_diagnostics import kpi_with_trades

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent


def _git_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
    )
    return bool(out.strip())


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):+.2f}%"


def _num(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.3f}"


def _year_slice_kpis(
    strat_eq: pd.DataFrame, qqq_eq: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Per-calendar-year Strategy vs QQQ vs Alpha (CAGR/Sharpe/MDD)."""
    if strat_eq.empty or qqq_eq.empty:
        return []
    years = sorted(
        set(strat_eq.index.year.astype(int)).intersection(
            set(qqq_eq.index.year.astype(int))
        )
    )
    rows: List[Dict[str, Any]] = []
    for y in years:
        s = strat_eq[strat_eq.index.year == y]
        q = qqq_eq[qqq_eq.index.year == y]
        if len(s) < 2 or len(q) < 2:
            continue
        sk = _cagr_sharpe_mdd(s)
        qk = _cagr_sharpe_mdd(q)
        alpha = None
        if sk.get("CAGR") is not None and qk.get("CAGR") is not None:
            alpha = float(sk["CAGR"]) - float(qk["CAGR"])
        # Also report simple period total return (useful for partial years).
        s_tot = float(s["Total_Equity"].iloc[-1] / s["Total_Equity"].iloc[0] - 1.0)
        q_tot = float(q["Total_Equity"].iloc[-1] / q["Total_Equity"].iloc[0] - 1.0)
        rows.append(
            {
                "year": int(y),
                "n_days_strategy": int(len(s)),
                "n_days_qqq": int(len(q)),
                "partial_year": int(y) == 2026 or len(s) < 240,
                "strategy": sk,
                "QQQ": qk,
                "Alpha_CAGR": alpha,
                "strategy_total_return": s_tot,
                "qqq_total_return": q_tot,
                "alpha_total_return": s_tot - q_tot,
            }
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_NOCHASE_DEFAULT",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: working tree dirty — commit first (git_dirty=False required).")
        return 1

    cfg = load_config(args.config)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = "corwin_schultz_v2"
    cfg["winsorize_adv"] = True
    cfg["enable_topup_chasing"] = False
    cfg.pop("disable_topup_chasing", None)
    cfg["benchmark"] = cfg.get("benchmark", "QQQ")

    repro = build_repro_fingerprint(
        start=START,
        end=END,
        config=cfg,
        extra={
            "report": "nochase_vs_qqq_alpha",
            "fill_policy": "no_chase_default",
            "qqq_bh": "buy_first_open_mark_close_no_costs",
            "invalidates": "topup_chase_report_2.txt",
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')}")
    if repro.get("git_dirty"):
        print("ERROR: fingerprint git_dirty=True — abort.")
        return 1

    print("\n=== No-chase baseline (Fixed CS v2) vs QQQ B&H ===")
    engine = BaselineEngineV1(config=cfg, config_path=args.config)
    engine.run()

    metrics = kpi_with_trades(engine)
    comparison = (engine.research_artifacts or {}).get("comparison") or {}
    strat = comparison.get("strategy") or {}
    qqq = comparison.get("QQQ") or {}
    alpha = comparison.get("Alpha_CAGR")

    strat_eq = (
        pd.DataFrame(engine.equity_curve).set_index("Date")
        if engine.equity_curve
        else pd.DataFrame()
    )
    qqq_eq = (
        pd.DataFrame(engine.qqq_equity_curve).set_index("Date")
        if engine.qqq_equity_curve
        else pd.DataFrame()
    )
    by_year = _year_slice_kpis(strat_eq, qqq_eq)

    payload = {
        "repro": repro,
        "window": {"start": START, "end": END},
        "qqq_methodology": {
            "ticker": getattr(engine, "BENCHMARK_TICKER", "QQQ"),
            "entry": "first eligible Open on/after START",
            "mark": "daily Close",
            "rebalance": False,
            "leverage": 1.0,
            "trading_costs": 0.0,
            "note": (
                "Matches BaselineEngineV1._maybe_buy_qqq / _mark_qqq; "
                "strategy equity also marked at Close."
            ),
        },
        "strategy_metrics": {
            "label": "Fixed CS v2 + NO-CHASE (default)",
            "CAGR": strat.get("CAGR"),
            "Sharpe": strat.get("Sharpe"),
            "Maximum_Drawdown": strat.get("Maximum_Drawdown"),
            "total_cost_dollars": metrics.get("total_cost_dollars"),
            "n_closed_trades": metrics.get("n_closed_trades"),
            "enable_topup_chasing": False,
        },
        "qqq_metrics": {
            "label": "QQQ buy & hold",
            "CAGR": qqq.get("CAGR"),
            "Sharpe": qqq.get("Sharpe"),
            "Maximum_Drawdown": qqq.get("Maximum_Drawdown"),
        },
        "Alpha_CAGR": alpha,
        "by_calendar_year": by_year,
        "audit": {
            "topup_chase_report_2": "INVALID_DO_NOT_CITE",
            "baseline_tag": "baseline-v1-nochase",
            "prior_clean_nochase_repro_id": "97c21a974bf8",
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha16 = hashlib.sha256(blob).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha16

    lines = [
        "# No-chase baseline vs QQQ buy & hold (VALID — clean committed state)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha16}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        f"panels={(repro.get('data_cache') or {}).get('panels.pkl')}",
        "",
        "## QQQ B&H methodology",
        "- Buy once at first eligible **Open** on/after 2022-01-01 (full INITIAL_CASH).",
        "- Mark daily at **Close** (same mark convention as strategy equity).",
        "- No rebalancing, no leverage, no trading costs.",
        "",
        "## TASK 2 — Side-by-side (full window)",
        f"{'Metric':<22} {'Strategy':>14} {'QQQ B&H':>14}",
        "-" * 52,
        f"{'CAGR':<22} {_pct(strat.get('CAGR')):>14} {_pct(qqq.get('CAGR')):>14}",
        f"{'Sharpe':<22} {_num(strat.get('Sharpe')):>14} {_num(qqq.get('Sharpe')):>14}",
        f"{'Maximum Drawdown':<22} {_pct(strat.get('Maximum_Drawdown')):>14} {_pct(qqq.get('Maximum_Drawdown')):>14}",
        "-" * 52,
        f"{'Alpha (CAGR−QQQ)':<22} {_pct(alpha):>14}",
        f"{'Strategy cost $':<22} {float(metrics.get('total_cost_dollars') or 0):>14,.0f}",
        f"{'n_trades':<22} {int(metrics.get('n_closed_trades') or 0):>14,}",
        "",
        "## TASK 3 — Per-calendar-year (Strategy vs QQQ vs Alpha)",
        f"{'Year':<8} {'Strat CAGR':>12} {'QQQ CAGR':>12} {'Alpha':>12} "
        f"{'Strat MDD':>12} {'QQQ MDD':>12} {'note':>10}",
        "-" * 82,
    ]
    for row in by_year:
        note = "partial" if row.get("partial_year") else ""
        lines.append(
            f"{row['year']:<8} {_pct(row['strategy'].get('CAGR')):>12} "
            f"{_pct(row['QQQ'].get('CAGR')):>12} {_pct(row.get('Alpha_CAGR')):>12} "
            f"{_pct(row['strategy'].get('Maximum_Drawdown')):>12} "
            f"{_pct(row['QQQ'].get('Maximum_Drawdown')):>12} {note:>10}"
        )
    lines += [
        "-" * 82,
        "",
        "Notes:",
        "- Per-year CAGR is computed on that year's equity slice (same KPI helper).",
        "- 2026 is partial (ends 2026-06-30).",
        "- topup_chase_report_2.txt numbers are INVALID and are not used here.",
        "",
        "# END",
    ]

    txt = out_dir / "nochase_vs_qqq_alpha_report.txt"
    js = out_dir / "nochase_vs_qqq_alpha_report.json"
    txt.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    print(f"Wrote -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
