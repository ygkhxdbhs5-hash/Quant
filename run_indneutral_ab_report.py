#!/usr/bin/env python3
"""A/B: mom+quality (global ranks) vs mom+quality (industry-neutral ranks).

Clean git required. Single in-memory payload + fingerprint.
Also reports average max industry weight and mean # industries in selections.

Usage:
  python3 run_indneutral_ab_report.py --config config/config.yaml
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
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
from engine.topup_chase_diagnostics import kpi_with_trades

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent

MQ = {
    "label": "Mom+Quality global ranks (default)",
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


def _concentration(engine) -> Dict[str, Any]:
    rows = list(getattr(engine, "diag_industry_selection", []) or [])
    if not rows:
        return {
            "mean_max_industry_weight": None,
            "mean_n_industries": None,
            "n_rebalance_months": 0,
        }
    max_w = [float(r["max_industry_weight"]) for r in rows if r.get("max_industry_weight") is not None]
    n_ind = [float(r["n_industries"]) for r in rows if r.get("n_industries") is not None]
    return {
        "mean_max_industry_weight": float(sum(max_w) / len(max_w)) if max_w else None,
        "mean_n_industries": float(sum(n_ind) / len(n_ind)) if n_ind else None,
        "n_rebalance_months": len(rows),
    }


def _run(cfg_base: dict, *, industry_neutral: bool, label: str) -> Dict[str, Any]:
    cfg = dict(cfg_base)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = "corwin_schultz_v2"
    cfg["winsorize_adv"] = True
    cfg["enable_topup_chasing"] = False
    cfg["enable_quality_factor"] = True
    cfg["enable_regime_exposure"] = False
    cfg["enable_regime_exposure_fast"] = False
    cfg["enable_industry_neutral_ranking"] = bool(industry_neutral)
    cfg["min_industry_size"] = 8
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
    return {
        "label": label,
        "enable_industry_neutral_ranking": bool(industry_neutral),
        "metrics": metrics,
        "trade_stats": ts,
        "comparison": comparison,
        "by_year": _year_rows(strat_eq, qqq_eq),
        "concentration": _concentration(engine),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_INDNEUTRAL",
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
            "enable_industry_neutral_ranking": True,
            "min_industry_size": 8,
        },
        extra={
            "report": "indneutral_ab",
            "baseline_mq_repro_id": MQ["repro_id"],
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')}")
    if repro.get("git_dirty"):
        return 1

    # Re-run global MQ so concentration metrics are comparable from same code path.
    global_mq = _run(
        base_cfg,
        industry_neutral=False,
        label="Mom+Quality GLOBAL ranks",
    )
    ind = _run(
        base_cfg,
        industry_neutral=True,
        label="Mom+Quality INDUSTRY-NEUTRAL ranks",
    )

    g_s = (global_mq["comparison"] or {}).get("strategy") or {}
    g_alpha = (global_mq["comparison"] or {}).get("Alpha_CAGR")
    g_m = global_mq["metrics"]
    g_ts = global_mq["trade_stats"]
    g_c = global_mq["concentration"]

    i_s = (ind["comparison"] or {}).get("strategy") or {}
    i_alpha = (ind["comparison"] or {}).get("Alpha_CAGR")
    i_m = ind["metrics"]
    i_ts = ind["trade_stats"]
    i_c = ind["concentration"]

    sharpe_improved = (
        i_s.get("Sharpe") is not None
        and g_s.get("Sharpe") is not None
        and float(i_s["Sharpe"]) > float(g_s["Sharpe"])
    )
    mdd_i = float(i_s.get("Maximum_Drawdown") or 0.0)
    mdd_g = float(g_s.get("Maximum_Drawdown") or 0.0)
    mdd_worsened = mdd_i < (mdd_g - 0.03)
    decision_keep = bool(sharpe_improved and not mdd_worsened)

    payload = {
        "repro": repro,
        "mq_checkpoint": MQ,
        "global_ranking": global_mq,
        "industry_neutral": ind,
        "decision": {
            "sharpe_improved": sharpe_improved,
            "mdd_worsened_gt_3pp": mdd_worsened,
            "keep_as_new_default": decision_keep,
            "rule": "keep if Sharpe improves AND MDD not materially worse (>3pp deeper)",
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha16 = hashlib.sha256(blob).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha16

    lines = [
        "# Mom+Quality global vs industry-neutral A/B (VALID — clean committed state)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha16}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        "",
        "Industry-neutral: mom/quality pctiles within industry; global fallback if n < 8.",
        "Unchanged: ATR 2.5, equal-weight, 1.0x, monthly, no-chase, Fixed CS v2, no regime.",
        "",
        "## TASK 3 — Side-by-side",
        f"{'Metric':<32} {'Global MQ':>14} {'Ind-neutral':>14}",
        "-" * 62,
        f"{'CAGR':<32} {_pct(g_s.get('CAGR')):>14} {_pct(i_s.get('CAGR')):>14}",
        f"{'Sharpe':<32} {_num(g_s.get('Sharpe')):>14} {_num(i_s.get('Sharpe')):>14}",
        f"{'MDD':<32} {_pct(g_s.get('Maximum_Drawdown')):>14} {_pct(i_s.get('Maximum_Drawdown')):>14}",
        f"{'Cost $':<32} {float(g_m.get('total_cost_dollars') or 0):>14,.0f} {float(i_m.get('total_cost_dollars') or 0):>14,.0f}",
        f"{'n_trades':<32} {int(g_m.get('n_closed_trades') or 0):>14,} {int(i_m.get('n_closed_trades') or 0):>14,}",
        f"{'Avg hold days':<32} {_num(g_ts.get('avg_holding_period_days'), 1):>14} {_num(i_ts.get('avg_holding_period_days'), 1):>14}",
        f"{'Turnover (repl/yr)':<32} {_num(g_ts.get('portfolio_replacement_rate_per_year'), 2):>14} {_num(i_ts.get('portfolio_replacement_rate_per_year'), 2):>14}",
        f"{'Alpha vs QQQ':<32} {_pct(g_alpha):>14} {_pct(i_alpha):>14}",
        f"{'Mean max industry weight':<32} {_pct(g_c.get('mean_max_industry_weight')):>14} {_pct(i_c.get('mean_max_industry_weight')):>14}",
        f"{'Mean # industries / month':<32} {_num(g_c.get('mean_n_industries'), 2):>14} {_num(i_c.get('mean_n_industries'), 2):>14}",
        "-" * 62,
        f"(checkpoint MQ Sharpe/CAGR for reference: {_num(MQ['Sharpe'])} / {_pct(MQ['CAGR'])})",
        "",
        "## TASK 4 — Industry-neutral per-year vs QQQ",
        f"{'Year':<8} {'Strat CAGR':>12} {'QQQ CAGR':>12} {'Alpha':>12} {'Strat MDD':>12} {'note':>10}",
        "-" * 70,
    ]
    for row in ind["by_year"]:
        note = "partial" if row.get("partial_year") else ""
        lines.append(
            f"{row['year']:<8} {_pct((row.get('strategy') or {}).get('CAGR')):>12} "
            f"{_pct((row.get('QQQ') or {}).get('CAGR')):>12} {_pct(row.get('Alpha_CAGR')):>12} "
            f"{_pct((row.get('strategy') or {}).get('Maximum_Drawdown')):>12} {note:>10}"
        )
    lines += [
        "-" * 70,
        "",
        "## Decision rule",
        f"  sharpe_improved             : {sharpe_improved}",
        f"  mdd_worsened (>3pp deeper)  : {mdd_worsened}",
        f"  KEEP as new default         : {decision_keep}",
        "",
        "# END",
    ]

    txt = out_dir / "indneutral_ab_report.txt"
    js = out_dir / "indneutral_ab_report.json"
    txt.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    print(f"Wrote -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
