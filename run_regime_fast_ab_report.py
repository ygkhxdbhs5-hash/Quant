#!/usr/bin/env python3
"""Three-way A/B: MQ default vs rejected sma200-regime vs fast sma50+RoC regime.

Clean git required. Single in-memory payload + fingerprint.

Usage:
  python3 run_regime_fast_ab_report.py --config config/config.yaml
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

MQ = {
    "label": "Mom+Quality (default)",
    "repro_id": "42799a276531",
    "CAGR": 0.0302,
    "Sharpe": 0.268,
    "Maximum_Drawdown": -0.2560,
    "total_cost_dollars": 13_404_546.0,
    "n_closed_trades": 1580,
    "Alpha_CAGR": -0.1144,
    "avg_holding_period_days": 28.9,
    "portfolio_replacement_rate_per_year": 11.74,
    "alpha_2023": -0.5100,
}
REGIME_200 = {
    "label": "MQ+Regime sma200 (rejected)",
    "repro_id": "a2f5d870e697",
    "CAGR": 0.0240,
    "Sharpe": 0.255,
    "Maximum_Drawdown": -0.1913,
    "total_cost_dollars": 11_810_933.0,
    "n_closed_trades": 1550,
    "Alpha_CAGR": -0.1206,
    "avg_holding_period_days": 27.9,
    "portfolio_replacement_rate_per_year": 11.51,
    "alpha_2023": -0.5377,
    "tier_share": {"1.0x": 0.537, "0.5x": 0.407, "0.0x": 0.056},
    "q1_2023": [
        {"date": "2023-01-03", "tier": "0.5x"},
        {"date": "2023-02-01", "tier": "1.0x"},
        {"date": "2023-03-01", "tier": "1.0x"},
    ],
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


def _run_fast(cfg_base: dict) -> Dict[str, Any]:
    cfg = dict(cfg_base)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = "corwin_schultz_v2"
    cfg["winsorize_adv"] = True
    cfg["enable_topup_chasing"] = False
    cfg["enable_quality_factor"] = True
    cfg["enable_regime_exposure"] = False
    cfg["enable_regime_exposure_fast"] = True
    cfg.pop("disable_topup_chasing", None)
    print("\n" + "#" * 72)
    print("# Mom+Quality + Regime FAST (SMA50 + breadth RoC)")
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
    q1 = []
    for row in regime_log:
        d = pd.Timestamp(row["date"])
        if d.year == 2023 and d.month in (1, 2, 3):
            q1.append(
                {
                    "date": str(d.date()),
                    "exposure": row.get("exposure"),
                    "tier": row.get("tier"),
                    "breadth": row.get("breadth"),
                    "breadth_prev_20d": row.get("breadth_prev_20d"),
                    "breadth_improving": row.get("breadth_improving"),
                    "benchmark_above_ma50": row.get("benchmark_above_ma50"),
                }
            )
    return {
        "label": "MQ+Regime FAST (sma50+RoC)",
        "metrics": metrics,
        "trade_stats": ts,
        "comparison": comparison,
        "by_year": _year_rows(strat_eq, qqq_eq),
        "regime_tier_share": summarize_tier_months(regime_log),
        "q1_2023_regime": q1,
        "regime_log": [
            {
                "date": str(pd.Timestamp(r["date"]).date()),
                "tier": r.get("tier"),
                "exposure": r.get("exposure"),
                "breadth": r.get("breadth"),
                "breadth_improving": r.get("breadth_improving"),
                "benchmark_above_ma50": r.get("benchmark_above_ma50"),
            }
            for r in regime_log
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_REGIME_FAST",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: working tree dirty — commit first.")
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
            "enable_regime_exposure": False,
            "enable_regime_exposure_fast": True,
        },
        extra={
            "report": "regime_fast_ab",
            "mq_repro": MQ["repro_id"],
            "regime200_repro": REGIME_200["repro_id"],
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')}")
    if repro.get("git_dirty"):
        return 1

    fast = _run_fast(base_cfg)
    f_s = (fast["comparison"] or {}).get("strategy") or {}
    f_alpha = (fast["comparison"] or {}).get("Alpha_CAGR")
    f_m = fast["metrics"]
    f_ts = fast["trade_stats"]
    tiers = fast.get("regime_tier_share") or {}

    a2023 = None
    for row in fast["by_year"]:
        if row["year"] == 2023:
            a2023 = row.get("Alpha_CAGR")
            break

    improves_vs_mq = a2023 is not None and float(a2023) > float(MQ["alpha_2023"]) + 0.005
    improves_vs_200 = a2023 is not None and float(a2023) > float(REGIME_200["alpha_2023"]) + 0.005
    # Material Sharpe drop vs MQ: more than 0.02
    sharpe_ok = (
        f_s.get("Sharpe") is not None
        and float(f_s["Sharpe"]) >= float(MQ["Sharpe"]) - 0.02
    )
    decision_keep = bool(improves_vs_mq and improves_vs_200 and sharpe_ok)

    jan_tier = None
    for row in fast.get("q1_2023_regime") or []:
        if str(row["date"]).startswith("2023-01"):
            jan_tier = row.get("tier")
            break
    caught_jan = jan_tier == "1.0x"

    if a2023 is None:
        a2023_note = "2023 alpha unavailable"
    else:
        a2023_note = (
            f"2023 Alpha: MQ {_pct(MQ['alpha_2023'])} | "
            f"sma200 {_pct(REGIME_200['alpha_2023'])} | "
            f"FAST {_pct(a2023)} | Jan tier={jan_tier} "
            f"({'caught Jan 1.0x' if caught_jan else 'did NOT reach 1.0x in January'})"
        )

    payload = {
        "repro": repro,
        "mq_checkpoint": MQ,
        "regime200_checkpoint": REGIME_200,
        "regime_fast": fast,
        "decision": {
            "alpha_2023_improves_vs_mq": improves_vs_mq,
            "alpha_2023_improves_vs_sma200": improves_vs_200,
            "sharpe_not_materially_worse_vs_mq": sharpe_ok,
            "keep_as_new_default": decision_keep,
            "caught_january_1x": caught_jan,
            "rule": (
                "keep if 2023 alpha improves vs BOTH MQ and sma200-regime "
                "AND Sharpe not materially worse vs MQ"
            ),
            "stop_regime_tuning_if_rejected": True,
        },
        "year_2023_callout": a2023_note,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha16 = hashlib.sha256(blob).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha16

    lines = [
        "# Three-way regime A/B: MQ | sma200-regime | FAST sma50+RoC (VALID)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha16}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        "",
        "FAST rule: QQQ vs SMA50; breadth level OR +10pp/20d RoC override; tiers 1.0/0.5/0.0; no >1.0x.",
        "",
        "## TASK 3 — Three-way comparison",
        f"{'Metric':<22} {'MQ default':>14} {'sma200 rej.':>14} {'FAST':>14}",
        "-" * 66,
        f"{'CAGR':<22} {_pct(MQ['CAGR']):>14} {_pct(REGIME_200['CAGR']):>14} {_pct(f_s.get('CAGR')):>14}",
        f"{'Sharpe':<22} {_num(MQ['Sharpe']):>14} {_num(REGIME_200['Sharpe']):>14} {_num(f_s.get('Sharpe')):>14}",
        f"{'MDD':<22} {_pct(MQ['Maximum_Drawdown']):>14} {_pct(REGIME_200['Maximum_Drawdown']):>14} {_pct(f_s.get('Maximum_Drawdown')):>14}",
        f"{'Cost $':<22} {MQ['total_cost_dollars']:>14,.0f} {REGIME_200['total_cost_dollars']:>14,.0f} {float(f_m.get('total_cost_dollars') or 0):>14,.0f}",
        f"{'n_trades':<22} {MQ['n_closed_trades']:>14,} {REGIME_200['n_closed_trades']:>14,} {int(f_m.get('n_closed_trades') or 0):>14,}",
        f"{'Avg hold days':<22} {_num(MQ['avg_holding_period_days'], 1):>14} {_num(REGIME_200['avg_holding_period_days'], 1):>14} {_num(f_ts.get('avg_holding_period_days'), 1):>14}",
        f"{'Turnover (repl/yr)':<22} {_num(MQ['portfolio_replacement_rate_per_year'], 2):>14} {_num(REGIME_200['portfolio_replacement_rate_per_year'], 2):>14} {_num(f_ts.get('portfolio_replacement_rate_per_year'), 2):>14}",
        f"{'Alpha vs QQQ':<22} {_pct(MQ['Alpha_CAGR']):>14} {_pct(REGIME_200['Alpha_CAGR']):>14} {_pct(f_alpha):>14}",
        f"{'2023 Alpha':<22} {_pct(MQ['alpha_2023']):>14} {_pct(REGIME_200['alpha_2023']):>14} {_pct(a2023):>14}",
        "-" * 66,
        f"% mo @ 1.0x            {'100%':>14} {100*REGIME_200['tier_share']['1.0x']:>13.1f}% {100*float(tiers.get('1.0x') or 0):>13.1f}%",
        f"% mo @ 0.5x            {'0%':>14} {100*REGIME_200['tier_share']['0.5x']:>13.1f}% {100*float(tiers.get('0.5x') or 0):>13.1f}%",
        f"% mo @ 0.0x            {'0%':>14} {100*REGIME_200['tier_share']['0.0x']:>13.1f}% {100*float(tiers.get('0.0x') or 0):>13.1f}%",
        "",
        "## TASK 4 — FAST per-year vs QQQ",
        f"{'Year':<8} {'Strat CAGR':>12} {'QQQ CAGR':>12} {'Alpha':>12} {'Strat MDD':>12} {'note':>10}",
        "-" * 70,
    ]
    for row in fast["by_year"]:
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
        "## Q1 2023 exposure (FAST vs rejected sma200)",
        "  sma200 (rejected): Jan 0.5x → Feb 1.0x → Mar 1.0x",
        "  FAST:",
    ]
    for row in fast.get("q1_2023_regime") or []:
        lines.append(
            f"    {row['date']}: tier={row['tier']} above_ma50={row['benchmark_above_ma50']} "
            f"breadth={row['breadth']} prev20={row['breadth_prev_20d']} "
            f"improving={row['breadth_improving']}"
        )
    lines += [
        "",
        "## Decision rule (LAST regime variant this round)",
        f"  alpha_2023_improves_vs_mq      : {improves_vs_mq}",
        f"  alpha_2023_improves_vs_sma200  : {improves_vs_200}",
        f"  sharpe_not_materially_worse    : {sharpe_ok}",
        f"  caught_january_1.0x            : {caught_jan}",
        f"  KEEP as new default            : {decision_keep}",
        (
            "  If rejected: stop regime tuning; keep MQ default; "
            "next = industry-neutral ranking."
        ),
        "",
        "# END",
    ]

    txt = out_dir / "regime_fast_ab_report.txt"
    js = out_dir / "regime_fast_ab_report.json"
    txt.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    print(f"Wrote -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
