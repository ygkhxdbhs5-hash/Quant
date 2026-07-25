#!/usr/bin/env python3
"""A/B: mom+quality vs mom+quality+regime exposure (clean git required).

Regime tiers (no >1.0x): QQQ vs SMA200 + liquid-universe breadth vs SMA50.
Single in-memory payload + fingerprint.

Usage:
  python3 run_regime_exposure_ab_report.py --config config/config.yaml
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

from engine.baseline_diagnostics import _cagr_sharpe_mdd, trade_stats
from engine.baseline_engine import BaselineEngineV1
from engine.regime_exposure import summarize_tier_months
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
from engine.topup_chase_diagnostics import kpi_with_trades

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent

# Confirmed mom+quality baseline (repro_id=42799a276531).
BASE_MQ = {
    "label": "mom+quality no-chase (checkpoint)",
    "repro_id": "42799a276531",
    "CAGR": 0.0302,
    "Sharpe": 0.268,
    "Maximum_Drawdown": -0.2560,
    "total_cost_dollars": 13_404_546.0,
    "n_closed_trades": 1580,
    "Alpha_CAGR": -0.1144,
    "qqq_CAGR": 0.1446,
    "avg_holding_period_days": 28.9,
    "portfolio_replacement_rate_per_year": 11.74,
    "by_year_alpha": {
        2022: 0.1576,
        2023: -0.5100,
        2024: -0.1776,
        2025: -0.2383,
        2026: 0.1666,
    },
}


def _git_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
    )
    return bool(out.strip())


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):+.2f}%"


def _num(x: Optional[float], d: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.{d}f}"


def _year_rows(strat_eq: pd.DataFrame, qqq_eq: pd.DataFrame) -> List[Dict[str, Any]]:
    if strat_eq.empty or qqq_eq.empty:
        return []
    years = sorted(
        set(strat_eq.index.year.astype(int)).intersection(set(qqq_eq.index.year.astype(int)))
    )
    rows = []
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
        rows.append(
            {
                "year": int(y),
                "partial_year": int(y) == 2026 or len(s) < 240,
                "strategy": sk,
                "QQQ": qk,
                "Alpha_CAGR": alpha,
            }
        )
    return rows


def _run(cfg_base: dict, *, enable_regime: bool, label: str) -> Dict[str, Any]:
    cfg = dict(cfg_base)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = "corwin_schultz_v2"
    cfg["winsorize_adv"] = True
    cfg["enable_topup_chasing"] = False
    cfg["enable_quality_factor"] = True
    cfg["enable_regime_exposure"] = bool(enable_regime)
    cfg.pop("disable_topup_chasing", None)
    print("\n" + "#" * 72)
    print(f"# {label}")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path="config/config.yaml")
    engine.run()
    metrics = kpi_with_trades(engine)
    ts = trade_stats(engine)
    comparison = (engine.research_artifacts or {}).get("comparison") or {}
    strat_eq = (
        pd.DataFrame(engine.equity_curve).set_index("Date") if engine.equity_curve else pd.DataFrame()
    )
    qqq_eq = (
        pd.DataFrame(engine.qqq_equity_curve).set_index("Date")
        if engine.qqq_equity_curve
        else pd.DataFrame()
    )
    regime_log = list(getattr(engine, "diag_regime_months", []) or [])
    # Q1 2023 months
    q1_2023 = []
    for row in regime_log:
        d = pd.Timestamp(row["date"])
        if d.year == 2023 and d.month in (1, 2, 3):
            q1_2023.append(
                {
                    "date": str(d.date()),
                    "exposure": row.get("exposure"),
                    "tier": row.get("tier"),
                    "breadth": row.get("breadth"),
                    "benchmark_above_ma200": row.get("benchmark_above_ma200"),
                }
            )
    return {
        "label": label,
        "enable_regime_exposure": bool(enable_regime),
        "metrics": metrics,
        "trade_stats": ts,
        "comparison": comparison,
        "by_year": _year_rows(strat_eq, qqq_eq),
        "regime_tier_share": summarize_tier_months(regime_log),
        "regime_log": [
            {
                "date": str(pd.Timestamp(r["date"]).date()),
                "exposure": r.get("exposure"),
                "tier": r.get("tier"),
                "breadth": r.get("breadth"),
                "benchmark_above_ma200": r.get("benchmark_above_ma200"),
            }
            for r in regime_log
        ],
        "q1_2023_regime": q1_2023,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_REGIME_EXPOSURE",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: working tree dirty — commit first.")
        return 1
    if not Path("data/fundamentals/pit_history.pkl").exists():
        print("ERROR: missing pit_history.pkl")
        return 1

    base_cfg = load_config(args.config)
    repro = build_repro_fingerprint(
        start=START,
        end=END,
        config={
            **base_cfg,
            "cost_model": "corwin_schultz_v2",
            "winsorize_adv": True,
            "enable_topup_chasing": False,
            "enable_quality_factor": True,
            "enable_regime_exposure": True,
        },
        extra={
            "report": "regime_exposure_ab",
            "baseline_mq_repro_id": BASE_MQ["repro_id"],
            "regime_tiers": "1.0/0.5/0.0 no >1.0x",
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')}")
    if repro.get("git_dirty"):
        return 1

    # Baseline leg from checkpoint KPIs (avoid redundant full re-run of adopted default).
    base = {
        "label": "Mom+Quality (checkpoint default)",
        "enable_regime_exposure": False,
        "metrics": {
            "CAGR": BASE_MQ["CAGR"],
            "Sharpe": BASE_MQ["Sharpe"],
            "Maximum_Drawdown": BASE_MQ["Maximum_Drawdown"],
            "total_cost_dollars": BASE_MQ["total_cost_dollars"],
            "n_closed_trades": BASE_MQ["n_closed_trades"],
        },
        "trade_stats": {
            "avg_holding_period_days": BASE_MQ["avg_holding_period_days"],
            "portfolio_replacement_rate_per_year": BASE_MQ[
                "portfolio_replacement_rate_per_year"
            ],
        },
        "comparison": {
            "strategy": {
                "CAGR": BASE_MQ["CAGR"],
                "Sharpe": BASE_MQ["Sharpe"],
                "Maximum_Drawdown": BASE_MQ["Maximum_Drawdown"],
            },
            "QQQ": {"CAGR": BASE_MQ["qqq_CAGR"]},
            "Alpha_CAGR": BASE_MQ["Alpha_CAGR"],
        },
        "by_year": [
            {"year": y, "partial_year": y == 2026, "Alpha_CAGR": a, "strategy": {}, "QQQ": {}}
            for y, a in BASE_MQ["by_year_alpha"].items()
        ],
        "regime_tier_share": {"1.0x": 1.0, "0.5x": 0.0, "0.0x": 0.0, "n_months": None},
        "from_checkpoint": True,
    }

    regime = _run(
        base_cfg,
        enable_regime=True,
        label="Mom+Quality+Regime (1.0/0.5/0.0, no >1.0x)",
    )

    r_s = (regime["comparison"] or {}).get("strategy") or {}
    r_alpha = (regime["comparison"] or {}).get("Alpha_CAGR")
    r_m = regime["metrics"]
    r_ts = regime["trade_stats"]
    b_s = (base["comparison"] or {}).get("strategy") or {}
    b_alpha = (base["comparison"] or {}).get("Alpha_CAGR")
    b_m = base["metrics"]
    b_ts = base["trade_stats"]

    sharpe_improved = (
        r_s.get("Sharpe") is not None
        and b_s.get("Sharpe") is not None
        and float(r_s["Sharpe"]) > float(b_s["Sharpe"])
    )
    # Material CAGR worsen: more than 1pp lower
    cagr_ok = (
        r_s.get("CAGR") is not None
        and b_s.get("CAGR") is not None
        and float(r_s["CAGR"]) >= float(b_s["CAGR"]) - 0.01
    )
    alpha_2023_base = BASE_MQ["by_year_alpha"][2023]
    alpha_2023_reg = None
    for row in regime["by_year"]:
        if row["year"] == 2023:
            alpha_2023_reg = row.get("Alpha_CAGR")
            break
    if alpha_2023_reg is None:
        a2023_note = "2023 alpha unavailable"
        a2023_improved = False
        a2023_worse = False
    else:
        delta = float(alpha_2023_reg) - float(alpha_2023_base)
        a2023_improved = delta > 0.02
        a2023_worse = delta < -0.02
        if abs(delta) < 0.02:
            shape = "materially unchanged"
        elif delta > 0:
            shape = "IMPROVED"
        else:
            shape = "WORSENED"
        a2023_note = (
            f"2023 Alpha: MQ {_pct(alpha_2023_base)} → MQ+Regime {_pct(alpha_2023_reg)} "
            f"(Δ={_pct(delta)}; {shape})"
        )

    decision_keep = bool(sharpe_improved and a2023_improved and cagr_ok and not a2023_worse)

    payload = {
        "repro": repro,
        "baseline_mq": BASE_MQ,
        "mom_quality": base,
        "mom_quality_regime": regime,
        "decision": {
            "sharpe_improved": sharpe_improved,
            "alpha_2023_improved": a2023_improved,
            "alpha_2023_worsened": a2023_worse,
            "cagr_not_materially_worse": cagr_ok,
            "keep_as_new_default": decision_keep,
            "rule": (
                "keep if Sharpe improves AND 2023 alpha improves "
                "AND overall CAGR not materially worse"
            ),
        },
        "year_2023_callout": a2023_note,
        "q1_2023_regime": regime.get("q1_2023_regime"),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha16 = hashlib.sha256(blob).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha16

    tiers = regime.get("regime_tier_share") or {}
    lines = [
        "# Mom+Quality vs Mom+Quality+Regime A/B (VALID — clean committed state)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha16}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        "",
        "Regime: QQQ vs SMA200 + breadth(% liquid names > SMA50); tiers 1.0/0.5/0.0; no >1.0x.",
        "Unchanged: mom+quality entry, ATR 2.5, equal-weight×exposure, monthly, no-chase, CS v2.",
        "",
        "## TASK 3 — Side-by-side",
        f"{'Metric':<28} {'Mom+Qual':>14} {'+Regime':>14}",
        "-" * 58,
        f"{'CAGR':<28} {_pct(b_s.get('CAGR')):>14} {_pct(r_s.get('CAGR')):>14}",
        f"{'Sharpe':<28} {_num(b_s.get('Sharpe')):>14} {_num(r_s.get('Sharpe')):>14}",
        f"{'MDD':<28} {_pct(b_s.get('Maximum_Drawdown')):>14} {_pct(r_s.get('Maximum_Drawdown')):>14}",
        f"{'Cost $':<28} {float(b_m.get('total_cost_dollars') or 0):>14,.0f} {float(r_m.get('total_cost_dollars') or 0):>14,.0f}",
        f"{'n_trades':<28} {int(b_m.get('n_closed_trades') or 0):>14,} {int(r_m.get('n_closed_trades') or 0):>14,}",
        f"{'Avg hold days':<28} {_num(b_ts.get('avg_holding_period_days'), 1):>14} {_num(r_ts.get('avg_holding_period_days'), 1):>14}",
        f"{'Turnover (repl/yr)':<28} {_num(b_ts.get('portfolio_replacement_rate_per_year'), 2):>14} {_num(r_ts.get('portfolio_replacement_rate_per_year'), 2):>14}",
        f"{'Alpha vs QQQ':<28} {_pct(b_alpha):>14} {_pct(r_alpha):>14}",
        "-" * 58,
        f"% months @ 1.0x                 {'100% (fixed)':>14} {100*float(tiers.get('1.0x') or 0):>13.1f}%",
        f"% months @ 0.5x                 {'0%':>14} {100*float(tiers.get('0.5x') or 0):>13.1f}%",
        f"% months @ 0.0x                 {'0%':>14} {100*float(tiers.get('0.0x') or 0):>13.1f}%",
        "",
        "## TASK 4 — +Regime per-year vs QQQ",
        f"{'Year':<8} {'Strat CAGR':>12} {'QQQ CAGR':>12} {'Alpha':>12} {'Strat MDD':>12} {'note':>10}",
        "-" * 70,
    ]
    for row in regime["by_year"]:
        note = "partial" if row.get("partial_year") else ""
        lines.append(
            f"{row['year']:<8} {_pct((row.get('strategy') or {}).get('CAGR')):>12} "
            f"{_pct((row.get('QQQ') or {}).get('CAGR')):>12} {_pct(row.get('Alpha_CAGR')):>12} "
            f"{_pct((row.get('strategy') or {}).get('Maximum_Drawdown')):>12} {note:>10}"
        )
    lines += [
        "-" * 70,
        f"CALLOUT: {a2023_note}",
        "",
        "## Q1 2023 regime tiers (rally onset)",
    ]
    if regime.get("q1_2023_regime"):
        for row in regime["q1_2023_regime"]:
            lines.append(
                f"  {row['date']}: tier={row['tier']} exposure={row['exposure']} "
                f"above_ma200={row['benchmark_above_ma200']} breadth={row['breadth']}"
            )
    else:
        lines.append("  (no Q1 2023 regime log rows)")
    lines += [
        "",
        "## Decision rule",
        f"  sharpe_improved             : {sharpe_improved}",
        f"  alpha_2023_improved         : {a2023_improved}",
        f"  alpha_2023_worsened         : {a2023_worse}",
        f"  cagr_not_materially_worse   : {cagr_ok}",
        f"  KEEP as new default         : {decision_keep}",
        "",
        "# END",
    ]

    txt = out_dir / "regime_exposure_ab_report.txt"
    js = out_dir / "regime_exposure_ab_report.json"
    txt.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    print(f"Wrote -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
