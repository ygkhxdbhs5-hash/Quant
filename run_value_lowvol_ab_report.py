#!/usr/bin/env python3
"""A/B: mom+quality baseline vs three mom+quality+value+lowvol weight presets.

Isolated entry-ranking only. Exit/sizing/leverage/no-chase/CS v2 unchanged.
Requires clean committed git state.

Usage:
  python3 run_value_lowvol_ab_report.py --config config/config.yaml
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
from engine.strategy_baseline_v1_mqvl import WEIGHT_PRESETS
from engine.topup_chase_diagnostics import kpi_with_trades

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent
MDD_MATERIAL_PP = 0.03  # >3pp deeper = material worsen

# Confirmed mom+quality default checkpoint (repro_id=42799a276531 family).
MQ_BASELINE = {
    "label": "mom+quality global (checkpoint)",
    "repro_id": "42799a276531",
    "CAGR": 0.0302,
    "Sharpe": 0.268,
    "Maximum_Drawdown": -0.2560,
    "total_cost_dollars": 13_404_546.0,
    "n_closed_trades": 1580,
    "Alpha_CAGR": -0.1144,
    "qqq_CAGR": 0.1446,
    "turnover_repl_per_year": 11.7,
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


def _money(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"${float(x):,.0f}"


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


def _base_flags(cfg_base: dict) -> dict:
    cfg = dict(cfg_base)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = "corwin_schultz_v2"
    cfg["winsorize_adv"] = True
    cfg["enable_topup_chasing"] = False
    cfg["enable_quality_factor"] = True
    cfg["enable_regime_exposure"] = False
    cfg["enable_regime_exposure_fast"] = False
    cfg["enable_industry_neutral_ranking"] = False
    cfg.pop("disable_topup_chasing", None)
    return cfg


def _run_mqvl(cfg_base: dict, *, variant: str, label: str) -> Dict[str, Any]:
    cfg = _base_flags(cfg_base)
    cfg["enable_value_lowvol_factor"] = True
    cfg["value_lowvol_variant"] = str(variant).lower()
    print("\n" + "#" * 72)
    print(f"# {label}")
    print(f"# weights={WEIGHT_PRESETS[variant]}")
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
        "variant": variant,
        "weights": dict(WEIGHT_PRESETS[variant]),
        "enable_value_lowvol_factor": True,
        "metrics": metrics,
        "trade_stats": ts,
        "comparison": comparison,
        "by_year": _year_rows(strat_eq, qqq_eq),
    }


def _run_mq_baseline(cfg_base: dict, *, skip_rerun: bool) -> Dict[str, Any]:
    if skip_rerun:
        return {
            "label": MQ_BASELINE["label"],
            "variant": "mq",
            "weights": {"mom": 0.5, "quality": 0.5, "value": 0.0, "low_vol": 0.0},
            "enable_value_lowvol_factor": False,
            "from_checkpoint": True,
            "metrics": {
                "CAGR": MQ_BASELINE["CAGR"],
                "Sharpe": MQ_BASELINE["Sharpe"],
                "Maximum_Drawdown": MQ_BASELINE["Maximum_Drawdown"],
                "total_cost_dollars": MQ_BASELINE["total_cost_dollars"],
                "n_closed_trades": MQ_BASELINE["n_closed_trades"],
            },
            "trade_stats": {
                "n_closed_trades": MQ_BASELINE["n_closed_trades"],
                "portfolio_replacement_rate_per_year": MQ_BASELINE["turnover_repl_per_year"],
                "avg_holding_period_days": None,
            },
            "comparison": {
                "strategy": {
                    "CAGR": MQ_BASELINE["CAGR"],
                    "Sharpe": MQ_BASELINE["Sharpe"],
                    "Maximum_Drawdown": MQ_BASELINE["Maximum_Drawdown"],
                },
                "QQQ": {"CAGR": MQ_BASELINE["qqq_CAGR"]},
                "Alpha_CAGR": MQ_BASELINE["Alpha_CAGR"],
            },
            "by_year": [
                {
                    "year": y,
                    "partial_year": y == 2026,
                    "strategy": {},
                    "QQQ": {},
                    "Alpha_CAGR": a,
                }
                for y, a in MQ_BASELINE["by_year_alpha"].items()
            ],
        }

    cfg = _base_flags(cfg_base)
    cfg["enable_value_lowvol_factor"] = False
    print("\n" + "#" * 72)
    print("# Mom+Quality baseline (rerun)")
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
        "label": "mom+quality global (rerun)",
        "variant": "mq",
        "weights": {"mom": 0.5, "quality": 0.5, "value": 0.0, "low_vol": 0.0},
        "enable_value_lowvol_factor": False,
        "from_checkpoint": False,
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
        default="docs/experiments/BASELINE_V1_VALUE_LOWVOL",
    )
    parser.add_argument(
        "--skip-mq-rerun",
        action="store_true",
        help="Use mom+quality checkpoint KPIs for the baseline column.",
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
            "enable_value_lowvol_factor": True,
            "value_lowvol_variant": "a|b|c",
        },
        extra={
            "report": "value_lowvol_ab",
            "variants": ["mq", "a", "b", "c"],
            "weights": WEIGHT_PRESETS,
            "value_def": "fcf_yield_ev = (op_cf-capex)/(close*PIT_shares+debt-cash)",
            "baseline_mq_repro_id": MQ_BASELINE["repro_id"],
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')}")
    if repro.get("git_dirty"):
        return 1

    mq = _run_mq_baseline(base_cfg, skip_rerun=bool(args.skip_mq_rerun))
    variants: Dict[str, Dict[str, Any]] = {}
    for key in ("a", "b", "c"):
        w = WEIGHT_PRESETS[key]
        variants[key] = _run_mqvl(
            base_cfg,
            variant=key,
            label=(
                f"MQVL-{key.upper()} "
                f"(m{w['mom']:.0%}/q{w['quality']:.0%}/v{w['value']:.0%}/l{w['low_vol']:.0%})"
            ),
        )

    def _pack(row: Dict[str, Any]) -> Dict[str, Any]:
        s = (row.get("comparison") or {}).get("strategy") or {}
        m = row.get("metrics") or {}
        ts = row.get("trade_stats") or {}
        return {
            "CAGR": s.get("CAGR", m.get("CAGR")),
            "Sharpe": s.get("Sharpe", m.get("Sharpe")),
            "MDD": s.get("Maximum_Drawdown", m.get("Maximum_Drawdown")),
            "cost": m.get("total_cost_dollars"),
            "n_trades": m.get("n_closed_trades") or ts.get("n_closed_trades"),
            "turnover": ts.get("portfolio_replacement_rate_per_year"),
            "Alpha": (row.get("comparison") or {}).get("Alpha_CAGR"),
            "weights": row.get("weights"),
            "label": row.get("label"),
        }

    mq_p = _pack(mq)
    packs = {k: _pack(variants[k]) for k in ("a", "b", "c")}

    # Best by Sharpe among a/b/c
    best_key = max(
        ("a", "b", "c"),
        key=lambda k: float(packs[k]["Sharpe"] if packs[k]["Sharpe"] is not None else -1e9),
    )
    best = packs[best_key]
    mq_sharpe = float(mq_p["Sharpe"] or 0.0)
    best_sharpe = float(best["Sharpe"] or 0.0)
    mq_mdd = float(mq_p["MDD"] or 0.0)
    best_mdd = float(best["MDD"] or 0.0)
    sharpe_improved = best_sharpe > mq_sharpe
    mdd_worsened = best_mdd < (mq_mdd - MDD_MATERIAL_PP)
    adopt = bool(sharpe_improved and not mdd_worsened)

    if not sharpe_improved:
        decision_side = (
            "NON-ADDITIVE: none of (a)/(b)/(c) improved Sharpe over mom+quality "
            f"({mq_sharpe:.3f}); keep mom+quality as default. Further long-only "
            "factor tweaks are unlikely to close the alpha gap — long-short is next."
        )
    elif mdd_worsened:
        decision_side = (
            f"REJECT: best variant {best_key} improved Sharpe ({best_sharpe:.3f} > "
            f"{mq_sharpe:.3f}) but materially worsened MDD "
            f"({_pct(best_mdd)} vs {_pct(mq_mdd)}); keep mom+quality."
        )
    else:
        decision_side = (
            f"ADOPT: variant {best_key} improves Sharpe ({best_sharpe:.3f} > "
            f"{mq_sharpe:.3f}) without material MDD worsen "
            f"({_pct(best_mdd)} vs {_pct(mq_mdd)})."
        )

    # 2023 callout for best variant
    mq_2023 = MQ_BASELINE["by_year_alpha"][2023]
    best_2023 = None
    for row in variants[best_key]["by_year"]:
        if row["year"] == 2023:
            best_2023 = row.get("Alpha_CAGR")
            break
    if best_2023 is None:
        y2023_note = "2023 alpha unavailable for winning variant"
    else:
        delta = float(best_2023) - float(mq_2023)
        if abs(delta) < 0.02:
            shape = "materially unchanged"
        elif delta > 0:
            shape = "IMPROVED (less negative / higher)"
        else:
            shape = "WORSENED (more negative)"
        y2023_note = (
            f"2023 Alpha: MQ {_pct(mq_2023)} → MQVL-{best_key.upper()} {_pct(best_2023)} "
            f"(Δ={_pct(delta)}; {shape})"
        )

    payload = {
        "repro": repro,
        "methodology": {
            "value": (
                "FCF/EV with FCF=op_cf-capex (PIT filing≤as_of); "
                "EV=close[as_of]*diluted_shares(PIT)+total_debt(PIT)-cash_eq(PIT). "
                "No FMP market-cap endpoint; constructed with PIT shares × that day's close "
                "(no look-ahead). Exclude if EV≤0 or missing ingredients."
            ),
            "low_vol": "pctile(-vol60); vol60=60d trailing std of daily returns",
            "exclusion": "missing quality OR value OR vol → exclude (no impute)",
            "unchanged": "ATR 2.5, equal-weight, 1.0x, monthly, no-chase, Fixed CS v2",
            "presets": WEIGHT_PRESETS,
        },
        "mq_baseline": mq,
        "variants": variants,
        "best_variant": best_key,
        "decision": {
            "sharpe_improved": sharpe_improved,
            "mdd_worsened_gt_3pp": mdd_worsened,
            "adopt": adopt,
            "best_variant": best_key,
            "side": decision_side,
            "rule": (
                "adopt highest-Sharpe among a/b/c IF Sharpe > MQ 0.268 AND "
                "MDD not materially worse (>3pp deeper than -25.60%)"
            ),
        },
        "year_2023_callout": y2023_note,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha16 = hashlib.sha256(blob).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha16

    # Also emit per-variant payload hashes for Task 5 "one per variant"
    variant_payload_hashes = {}
    for k, v in variants.items():
        vb = json.dumps(v, sort_keys=True, default=str).encode("utf-8")
        variant_payload_hashes[k] = hashlib.sha256(vb).hexdigest()[:16]
    payload["variant_payload_sha256_16"] = variant_payload_hashes

    hdr = f"{'Metric':<22} {'MQ base':>12} {'(a)':>12} {'(b)':>12} {'(c)':>12}"
    lines = [
        "# Mom+Quality vs Value+LowVol weight presets A/B (VALID)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha16}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        f"variant_payload_sha256_16={variant_payload_hashes}",
        f"mq_baseline_from_checkpoint={mq.get('from_checkpoint')}",
        "",
        "Value (PIT-safe FCF/EV):",
        "  FCF = op_cf - capex (filing_date ≤ as_of)",
        "  EV  = close[as_of] × diluted_shares(PIT) + total_debt(PIT) - cash_eq(PIT)",
        "  No FMP mcap/EV endpoint used; shares from existing Massive PIT pickle.",
        "Low-vol: pctile(-vol60), vol60 = 60d realized vol of daily returns.",
        "Exclude missing quality / value / vol (no impute).",
        "Unchanged: ATR 2.5, EW, 1.0x, monthly, no-chase, Fixed CS v2,",
        "  industry-neutral / corr filter / hysteresis OFF.",
        "",
        "Weights:",
        f"  (a) {WEIGHT_PRESETS['a']}",
        f"  (b) {WEIGHT_PRESETS['b']}",
        f"  (c) {WEIGHT_PRESETS['c']}",
        "",
        "## TASK 5 — Four-way comparison",
        hdr,
        "-" * len(hdr),
        f"{'CAGR':<22} {_pct(mq_p['CAGR']):>12} {_pct(packs['a']['CAGR']):>12} {_pct(packs['b']['CAGR']):>12} {_pct(packs['c']['CAGR']):>12}",
        f"{'Sharpe':<22} {_num(mq_p['Sharpe']):>12} {_num(packs['a']['Sharpe']):>12} {_num(packs['b']['Sharpe']):>12} {_num(packs['c']['Sharpe']):>12}",
        f"{'MDD':<22} {_pct(mq_p['MDD']):>12} {_pct(packs['a']['MDD']):>12} {_pct(packs['b']['MDD']):>12} {_pct(packs['c']['MDD']):>12}",
        f"{'Cost $':<22} {_money(mq_p['cost']):>12} {_money(packs['a']['cost']):>12} {_money(packs['b']['cost']):>12} {_money(packs['c']['cost']):>12}",
        f"{'n_trades':<22} {int(mq_p['n_trades'] or 0):>12,} {int(packs['a']['n_trades'] or 0):>12,} {int(packs['b']['n_trades'] or 0):>12,} {int(packs['c']['n_trades'] or 0):>12,}",
        f"{'Turnover (repl/yr)':<22} {_num(mq_p['turnover'], 2):>12} {_num(packs['a']['turnover'], 2):>12} {_num(packs['b']['turnover'], 2):>12} {_num(packs['c']['turnover'], 2):>12}",
        f"{'Alpha vs QQQ':<22} {_pct(mq_p['Alpha']):>12} {_pct(packs['a']['Alpha']):>12} {_pct(packs['b']['Alpha']):>12} {_pct(packs['c']['Alpha']):>12}",
        "-" * len(hdr),
        f"Highest Sharpe among a/b/c : {best_key} ({_num(best_sharpe)})",
        "",
        f"## TASK 6 — Per-year breakdown (best = {best_key})",
        f"{'Year':<8} {'Strat CAGR':>12} {'QQQ CAGR':>12} {'Alpha':>12} {'Strat MDD':>12} {'note':>10}",
        "-" * 70,
    ]
    for row in variants[best_key]["by_year"]:
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
        f"  best_variant                : {best_key}",
        f"  sharpe_improved             : {sharpe_improved}  ({_num(best_sharpe)} vs MQ {_num(mq_sharpe)})",
        f"  mdd_worsened (>3pp deeper)  : {mdd_worsened}",
        f"  ADOPT                       : {adopt}",
        f"  SIDE                        : {decision_side}",
        "",
        "# END",
        "",
    ]

    txt = out_dir / "value_lowvol_ab_report.txt"
    js = out_dir / "value_lowvol_ab_report.json"
    txt.write_text("\n".join(lines), encoding="utf-8")
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    print(f"Wrote -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
