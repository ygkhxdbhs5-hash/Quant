#!/usr/bin/env python3
"""A/B: momentum-only no-chase baseline vs momentum+quality (clean git required).

Single in-memory payload + fingerprint. Exit/sizing/leverage/no-chase/CS v2
unchanged. Quality is an isolated entry-ranking variant
(``enable_quality_factor=True``).

Usage:
  python3 run_mom_quality_ab_report.py --config config/config.yaml
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

# Confirmed clean momentum-only no-chase checkpoint (do not re-cite report_2).
BASELINE_MOM = {
    "label": "momentum-only no-chase (checkpoint)",
    "repro_id": "50978c877d89",
    "CAGR": 0.0189,
    "Sharpe": 0.194,
    "Maximum_Drawdown": -0.3197,
    "total_cost_dollars": 12_536_785.0,
    "n_closed_trades": 1589,
    "Alpha_CAGR": -0.1257,
    "qqq_CAGR": 0.1446,
    "by_year_alpha": {
        2022: 0.1935,
        2023: -0.6277,
        2024: -0.2086,
        2025: -0.1553,
        2026: -0.0260,
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


def _run_variant(cfg_base: dict, *, enable_quality: bool, label: str) -> Dict[str, Any]:
    cfg = dict(cfg_base)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = "corwin_schultz_v2"
    cfg["winsorize_adv"] = True
    cfg["enable_topup_chasing"] = False
    cfg.pop("disable_topup_chasing", None)
    cfg["enable_quality_factor"] = bool(enable_quality)
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
        "enable_quality_factor": bool(enable_quality),
        "metrics": metrics,
        "trade_stats": ts,
        "comparison": comparison,
        "by_year": _year_rows(strat_eq, qqq_eq),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_MOM_QUALITY",
    )
    parser.add_argument(
        "--skip-mom-rerun",
        action="store_true",
        help="Use checkpoint KPIs for momentum-only leg (still runs quality).",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: working tree dirty — commit first (git_dirty=False required).")
        return 1

    pit = Path("data/fundamentals/pit_history.pkl")
    if not pit.exists() or pit.stat().st_size < 1000:
        print(f"ERROR: missing/empty {pit}. Download fundamentals first.")
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
        },
        extra={
            "report": "mom_quality_ab",
            "variants": ["mom_only_nochase", "mom_quality_nochase"],
            "baseline_mom_repro_id": BASELINE_MOM["repro_id"],
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')}")
    if repro.get("git_dirty"):
        print("ERROR: fingerprint git_dirty=True — abort.")
        return 1

    if args.skip_mom_rerun:
        mom = {
            "label": "Fixed CS v2 + NO-CHASE momentum-only (checkpoint)",
            "enable_quality_factor": False,
            "metrics": {
                "CAGR": BASELINE_MOM["CAGR"],
                "Sharpe": BASELINE_MOM["Sharpe"],
                "Maximum_Drawdown": BASELINE_MOM["Maximum_Drawdown"],
                "total_cost_dollars": BASELINE_MOM["total_cost_dollars"],
                "n_closed_trades": BASELINE_MOM["n_closed_trades"],
            },
            "trade_stats": {
                "avg_holding_period_days": None,
                "portfolio_replacement_rate_per_year": None,
                "n_closed_trades": BASELINE_MOM["n_closed_trades"],
            },
            "comparison": {
                "strategy": {
                    "CAGR": BASELINE_MOM["CAGR"],
                    "Sharpe": BASELINE_MOM["Sharpe"],
                    "Maximum_Drawdown": BASELINE_MOM["Maximum_Drawdown"],
                },
                "QQQ": {"CAGR": BASELINE_MOM["qqq_CAGR"]},
                "Alpha_CAGR": BASELINE_MOM["Alpha_CAGR"],
            },
            "by_year": [
                {
                    "year": y,
                    "partial_year": y == 2026,
                    "strategy": {},
                    "QQQ": {},
                    "Alpha_CAGR": a,
                }
                for y, a in BASELINE_MOM["by_year_alpha"].items()
            ],
            "from_checkpoint": True,
        }
    else:
        mom = _run_variant(
            base_cfg,
            enable_quality=False,
            label="Fixed CS v2 + NO-CHASE momentum-only",
        )
        mom["from_checkpoint"] = False

    qual = _run_variant(
        base_cfg,
        enable_quality=True,
        label="Fixed CS v2 + NO-CHASE momentum+quality",
    )

    q_comp = qual["comparison"]
    q_s = q_comp.get("strategy") or {}
    q_alpha = q_comp.get("Alpha_CAGR")
    q_m = qual["metrics"]
    q_ts = qual["trade_stats"]

    m_s = (mom["comparison"] or {}).get("strategy") or {}
    m_alpha = (mom["comparison"] or {}).get("Alpha_CAGR")
    m_m = mom["metrics"]
    m_ts = mom["trade_stats"]

    sharpe_improved = (q_s.get("Sharpe") is not None and m_s.get("Sharpe") is not None) and (
        float(q_s["Sharpe"]) > float(m_s["Sharpe"])
    )
    # Material MDD worsen: more than 3pp deeper drawdown
    mdd_q = float(q_s.get("Maximum_Drawdown") or 0.0)
    mdd_m = float(m_s.get("Maximum_Drawdown") or 0.0)
    mdd_worsened = mdd_q < (mdd_m - 0.03)
    decision_keep = bool(sharpe_improved and not mdd_worsened)

    # 2023 alpha callout
    mom_2023 = BASELINE_MOM["by_year_alpha"][2023]
    qual_2023 = None
    for row in qual["by_year"]:
        if row["year"] == 2023:
            qual_2023 = row.get("Alpha_CAGR")
            break
    if qual_2023 is None:
        y2023_note = "2023 alpha unavailable for quality variant"
    else:
        delta = float(qual_2023) - float(mom_2023)
        if abs(delta) < 0.02:
            shape = "materially unchanged"
        elif delta > 0:
            shape = "IMPROVED (less negative / higher)"
        else:
            shape = "WORSENED (more negative)"
        y2023_note = (
            f"2023 Alpha: mom-only {_pct(mom_2023)} → mom+quality {_pct(qual_2023)} "
            f"(Δ={_pct(delta)}; {shape})"
        )

    payload = {
        "repro": repro,
        "baseline_mom_checkpoint": BASELINE_MOM,
        "momentum_only": mom,
        "momentum_quality": qual,
        "decision": {
            "sharpe_improved": sharpe_improved,
            "mdd_worsened_gt_3pp": mdd_worsened,
            "keep_as_new_default": decision_keep,
            "rule": "keep if Sharpe improves AND MDD not materially worse (>3pp deeper)",
        },
        "year_2023_callout": y2023_note,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha16 = hashlib.sha256(blob).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha16

    lines = [
        "# Momentum-only vs Momentum+Quality A/B (VALID — clean committed state)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha16}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        f"panels={(repro.get('data_cache') or {}).get('panels.pkl')}",
        "",
        "Quality: equal-weight avg pctile(GP, ROIC, op_margin) within liquid pool;",
        "combined = 0.5*mom_pctile + 0.5*quality_pctile; exclude missing PIT.",
        "Unchanged: ATR 2.5 exit, equal-weight, 1.0x, monthly, no-chase, Fixed CS v2.",
        "",
        "## TASK 3 — Side-by-side",
        f"{'Metric':<28} {'Mom-only':>14} {'Mom+Quality':>14}",
        "-" * 58,
        f"{'CAGR':<28} {_pct(m_s.get('CAGR')):>14} {_pct(q_s.get('CAGR')):>14}",
        f"{'Sharpe':<28} {_num(m_s.get('Sharpe')):>14} {_num(q_s.get('Sharpe')):>14}",
        f"{'MDD':<28} {_pct(m_s.get('Maximum_Drawdown')):>14} {_pct(q_s.get('Maximum_Drawdown')):>14}",
        f"{'Cost $':<28} {float(m_m.get('total_cost_dollars') or 0):>14,.0f} {float(q_m.get('total_cost_dollars') or 0):>14,.0f}",
        f"{'n_trades':<28} {int(m_m.get('n_closed_trades') or 0):>14,} {int(q_m.get('n_closed_trades') or 0):>14,}",
        f"{'Avg hold days':<28} {_num(m_ts.get('avg_holding_period_days'), 1):>14} {_num(q_ts.get('avg_holding_period_days'), 1):>14}",
        f"{'Turnover (repl/yr)':<28} {_num(m_ts.get('portfolio_replacement_rate_per_year'), 2):>14} {_num(q_ts.get('portfolio_replacement_rate_per_year'), 2):>14}",
        f"{'Alpha vs QQQ':<28} {_pct(m_alpha):>14} {_pct(q_alpha):>14}",
        "-" * 58,
        "",
        "## TASK 4 — Mom+Quality per-year vs QQQ",
        f"{'Year':<8} {'Strat CAGR':>12} {'QQQ CAGR':>12} {'Alpha':>12} {'Strat MDD':>12} {'note':>10}",
        "-" * 70,
    ]
    for row in qual["by_year"]:
        note = "partial" if row.get("partial_year") else ""
        lines.append(
            f"{row['year']:<8} {_pct((row.get('strategy') or {}).get('CAGR')):>12} "
            f"{_pct((row.get('QQQ') or {}).get('CAGR')):>12} {_pct(row.get('Alpha_CAGR')):>12} "
            f"{_pct((row.get('strategy') or {}).get('Maximum_Drawdown')):>12} {note:>10}"
        )
    lines += [
        "-" * 70,
        f"CALLOUT: {y2023_note}",
        "",
        "## Decision rule",
        f"  sharpe_improved             : {sharpe_improved}",
        f"  mdd_worsened (>3pp deeper)  : {mdd_worsened}",
        f"  KEEP as new default         : {decision_keep}",
        "",
        "# END",
    ]

    txt = out_dir / "mom_quality_ab_report.txt"
    js = out_dir / "mom_quality_ab_report.json"
    txt.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    print(f"Wrote -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
