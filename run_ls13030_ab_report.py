#!/usr/bin/env python3
"""A/B: mom+quality long-only vs 130/30 long-short structural variant.

Clean git required. Reports beta regression, sleeve costs, and 2023 short-book
contribution.

Usage:
  python3 run_ls13030_ab_report.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.baseline_diagnostics import _cagr_sharpe_mdd, trade_stats
from engine.baseline_engine import BaselineEngineV1
from engine.baseline_engine_ls13030 import BaselineEngineLS13030
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
from engine.topup_chase_diagnostics import kpi_with_trades

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent
MDD_MATERIAL_PP = 0.03
BETA_REDUCE_MIN = 0.15  # "meaningful" beta drop vs long-only

MQ = {
    "label": "mom+quality long-only (checkpoint)",
    "repro_id": "42799a276531",
    "CAGR": 0.0302,
    "Sharpe": 0.268,
    "Maximum_Drawdown": -0.2560,
    "total_cost_dollars": 13_404_546.0,
    "n_closed_trades": 1580,
    "Alpha_CAGR": -0.1144,
    "qqq_CAGR": 0.1446,
    "turnover_repl_per_year": 11.7,
    "beta_vs_qqq": None,  # filled if rerun; else estimated ~1
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


def _beta(strat_eq: pd.DataFrame, qqq_eq: pd.DataFrame) -> Optional[float]:
    if strat_eq.empty or qqq_eq.empty:
        return None
    s = strat_eq["Total_Equity"].pct_change().dropna()
    q = qqq_eq["Total_Equity"].pct_change().dropna()
    df = pd.concat([s.rename("s"), q.rename("q")], axis=1).dropna()
    if len(df) < 30:
        return None
    var_q = float(df["q"].var())
    if var_q <= 0:
        return None
    return float(df["s"].cov(df["q"]) / var_q)


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


def _short_year_pnl(engine: BaselineEngineLS13030, year: int) -> Dict[str, Any]:
    lots = list(getattr(engine, "diag_closed_lots_ls", []) or [])
    short = [
        r
        for r in lots
        if str(r.get("side")) == "SHORT"
        and r.get("exit_date") is not None
        and pd.Timestamp(r["exit_date"]).year == int(year)
    ]
    pnl = float(sum(float(r.get("dollar_pnl") or 0.0) for r in short))
    n = len(short)
    wins = sum(1 for r in short if float(r.get("dollar_pnl") or 0.0) > 0)
    return {
        "year": year,
        "n_short_closes": n,
        "short_dollar_pnl": pnl,
        "short_win_rate": (wins / n) if n else None,
        "mean_short_pct_return": (
            float(np.mean([float(r.get("pct_return")) for r in short if r.get("pct_return") is not None]))
            if short
            else None
        ),
    }


def _run_ls(cfg_base: dict) -> Dict[str, Any]:
    cfg = dict(cfg_base)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = "corwin_schultz_v2"
    cfg["winsorize_adv"] = True
    cfg["enable_topup_chasing"] = False
    cfg["enable_quality_factor"] = True
    print("\n" + "#" * 72)
    print("# 130/30 mom+quality long-short")
    print("#" * 72)
    engine = BaselineEngineLS13030(config=cfg, config_path="config/config.yaml")
    engine.run()
    metrics = kpi_with_trades(engine)
    # n_trades from LS lots (long+short)
    lots = pd.DataFrame(getattr(engine, "diag_closed_lots_ls", []) or [])
    n_long = int((lots["side"] == "LONG").sum()) if not lots.empty and "side" in lots.columns else 0
    n_short = int((lots["side"] == "SHORT").sum()) if not lots.empty and "side" in lots.columns else 0
    # Also count long journal closes if LS lots under-captured longs
    tj = engine.trade_journal.to_frame()
    n_long_journal = int(len(tj)) if tj is not None and not tj.empty else 0
    if n_long == 0 and n_long_journal:
        n_long = n_long_journal

    strat_eq = (
        pd.DataFrame(engine.equity_curve).set_index("Date") if engine.equity_curve else pd.DataFrame()
    )
    qqq_eq = (
        pd.DataFrame(engine.qqq_equity_curve).set_index("Date")
        if engine.qqq_equity_curve
        else pd.DataFrame()
    )
    years = _cagr_sharpe_mdd(strat_eq).get("CAGR")
    # turnover approx: (long+short closes) / years / (30+30 capacity)
    n_years = None
    if not strat_eq.empty:
        n_years = (strat_eq.index[-1] - strat_eq.index[0]).days / 365.25
    capacity = float(engine.TOP_MOMENTUM_COUNT + engine.SHORT_BASKET_COUNT)
    turnover = None
    if n_years and n_years > 0:
        turnover = (n_long + n_short) / n_years / capacity

    ts = trade_stats(engine)
    comparison = (engine.research_artifacts or {}).get("comparison") or {}
    beta = _beta(strat_eq, qqq_eq)
    sleeve = pd.DataFrame(getattr(engine, "diag_sleeve_daily", []) or [])
    mean_net = float(sleeve["net_exposure"].mean()) if not sleeve.empty else None
    mean_gross = float(sleeve["gross_exposure"].mean()) if not sleeve.empty else None

    return {
        "label": "130/30 MQ long-short",
        "engine": engine,
        "metrics": metrics,
        "trade_stats": ts,
        "comparison": comparison,
        "by_year": _year_rows(strat_eq, qqq_eq),
        "beta_vs_qqq": beta,
        "n_long_trades": n_long,
        "n_short_trades": n_short,
        "turnover_repl_per_year": turnover,
        "costs": {
            "total": float(getattr(engine, "diag_total_cost_dollars", 0.0)),
            "long_exec": float(getattr(engine, "diag_long_exec_cost_dollars", 0.0)),
            "short_exec": float(getattr(engine, "diag_short_exec_cost_dollars", 0.0)),
            "borrow": float(getattr(engine, "diag_borrow_cost_dollars", 0.0)),
        },
        "mean_net_exposure": mean_net,
        "mean_gross_exposure": mean_gross,
        "short_2023": _short_year_pnl(engine, 2023),
        "short_by_year": {y: _short_year_pnl(engine, y) for y in range(2022, 2027)},
        "strat_eq": strat_eq,
        "qqq_eq": qqq_eq,
    }


def _run_mq_for_beta(cfg_base: dict) -> Dict[str, Any]:
    """Rerun long-only MQ to get beta on the same window/code path."""
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
    print("\n" + "#" * 72)
    print("# Mom+Quality long-only (for beta + side-by-side)")
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
        "label": "mom+quality long-only",
        "metrics": metrics,
        "trade_stats": ts,
        "comparison": comparison,
        "by_year": _year_rows(strat_eq, qqq_eq),
        "beta_vs_qqq": _beta(strat_eq, qqq_eq),
        "costs": {
            "total": float(getattr(engine, "diag_total_cost_dollars", 0.0)),
            "long_exec": float(getattr(engine, "diag_total_cost_dollars", 0.0)),
            "short_exec": 0.0,
            "borrow": 0.0,
        },
        "n_long_trades": int(metrics.get("n_closed_trades") or 0),
        "n_short_trades": 0,
        "turnover_repl_per_year": ts.get("portfolio_replacement_rate_per_year"),
        "from_checkpoint": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_LS13030",
    )
    parser.add_argument(
        "--skip-mq-rerun",
        action="store_true",
        help="Use MQ checkpoint KPIs (still runs LS; beta for MQ marked n/a unless rerun).",
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
            "structure": "ls_13030",
            "ls_long_exposure": 1.3,
            "ls_short_exposure": 0.3,
            "ls_borrow_fee_annual": 0.03,
        },
        extra={
            "report": "ls13030_ab",
            "baseline_mq_repro_id": MQ["repro_id"],
            "borrow": "flat_3pct_ann_HTB_ADV_p25_excluded",
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')}")
    if repro.get("git_dirty"):
        return 1

    if args.skip_mq_rerun:
        mq = {
            "label": MQ["label"],
            "from_checkpoint": True,
            "metrics": {
                "CAGR": MQ["CAGR"],
                "Sharpe": MQ["Sharpe"],
                "Maximum_Drawdown": MQ["Maximum_Drawdown"],
                "total_cost_dollars": MQ["total_cost_dollars"],
                "n_closed_trades": MQ["n_closed_trades"],
            },
            "trade_stats": {
                "portfolio_replacement_rate_per_year": MQ["turnover_repl_per_year"],
                "n_closed_trades": MQ["n_closed_trades"],
            },
            "comparison": {
                "strategy": {
                    "CAGR": MQ["CAGR"],
                    "Sharpe": MQ["Sharpe"],
                    "Maximum_Drawdown": MQ["Maximum_Drawdown"],
                },
                "QQQ": {"CAGR": MQ["qqq_CAGR"]},
                "Alpha_CAGR": MQ["Alpha_CAGR"],
            },
            "by_year": [
                {
                    "year": y,
                    "partial_year": y == 2026,
                    "strategy": {},
                    "QQQ": {},
                    "Alpha_CAGR": a,
                }
                for y, a in MQ["by_year_alpha"].items()
            ],
            "beta_vs_qqq": None,
            "costs": {
                "total": MQ["total_cost_dollars"],
                "long_exec": MQ["total_cost_dollars"],
                "short_exec": 0.0,
                "borrow": 0.0,
            },
            "n_long_trades": MQ["n_closed_trades"],
            "n_short_trades": 0,
            "turnover_repl_per_year": MQ["turnover_repl_per_year"],
        }
    else:
        mq = _run_mq_for_beta(base_cfg)

    ls = _run_ls(base_cfg)
    # Drop heavy frames from payload
    engine = ls.pop("engine", None)
    ls.pop("strat_eq", None)
    ls.pop("qqq_eq", None)

    mq_s = (mq.get("comparison") or {}).get("strategy") or {}
    ls_s = (ls.get("comparison") or {}).get("strategy") or {}
    mq_alpha = (mq.get("comparison") or {}).get("Alpha_CAGR")
    ls_alpha = (ls.get("comparison") or {}).get("Alpha_CAGR")
    mq_beta = mq.get("beta_vs_qqq")
    ls_beta = ls.get("beta_vs_qqq")

    mq_sharpe = float(mq_s.get("Sharpe") or mq["metrics"].get("Sharpe") or 0.0)
    ls_sharpe = float(ls_s.get("Sharpe") or 0.0)
    mq_mdd = float(mq_s.get("Maximum_Drawdown") or mq["metrics"].get("Maximum_Drawdown") or 0.0)
    ls_mdd = float(ls_s.get("Maximum_Drawdown") or 0.0)

    sharpe_improved = ls_sharpe > mq_sharpe
    mdd_worsened = ls_mdd < (mq_mdd - MDD_MATERIAL_PP)
    beta_reduced = False
    beta_note = "n/a"
    if mq_beta is not None and ls_beta is not None:
        beta_reduced = float(ls_beta) <= float(mq_beta) - BETA_REDUCE_MIN
        beta_note = (
            f"MQ beta={mq_beta:.3f} → LS beta={ls_beta:.3f} "
            f"(Δ={ls_beta - mq_beta:+.3f}; "
            f"{'MEANINGFUL reduce' if beta_reduced else 'NOT meaningful reduce'} "
            f"vs threshold {BETA_REDUCE_MIN:.2f})"
        )
    elif ls_beta is not None:
        # Classic 130/30 net≈100%; treat beta<0.85 as meaningful vs ~1.0 assumption
        beta_reduced = float(ls_beta) <= (1.0 - BETA_REDUCE_MIN)
        beta_note = (
            f"LS beta={ls_beta:.3f} vs assumed MQ~1.0 "
            f"({'MEANINGFUL reduce' if beta_reduced else 'NOT meaningful reduce'})"
        )

    adopt = bool(sharpe_improved and beta_reduced and not mdd_worsened)
    if not sharpe_improved:
        side = (
            "DO NOT ADOPT: Sharpe did not improve vs long-only MQ. "
            "Consider 100/100 market-neutral before abandoning long-short."
        )
    elif not beta_reduced:
        side = (
            "DO NOT ADOPT: Sharpe improved but beta reduction did not materialize "
            "(130/30 still ~100% net). Try 100/100 market-neutral next."
        )
    elif mdd_worsened:
        side = (
            "DO NOT ADOPT: Sharpe improved and beta fell, but MDD worsened materially."
        )
    else:
        side = "ADOPT: Sharpe up, beta meaningfully down, MDD not materially worse."

    # 2023 callout
    mq_2023 = MQ["by_year_alpha"][2023]
    ls_2023 = None
    for row in ls["by_year"]:
        if row["year"] == 2023:
            ls_2023 = row.get("Alpha_CAGR")
            break
    s2023 = ls.get("short_2023") or {}
    if ls_2023 is None:
        y2023 = "2023 alpha unavailable for LS"
    else:
        delta = float(ls_2023) - float(mq_2023)
        y2023 = (
            f"2023 Alpha: MQ {_pct(mq_2023)} → LS {_pct(ls_2023)} (Δ={_pct(delta)}); "
            f"short-book closed-lot $PnL in 2023: {_money(s2023.get('short_dollar_pnl'))} "
            f"on {int(s2023.get('n_short_closes') or 0)} covers "
            f"(win%={_num(s2023.get('short_win_rate'), 2)})"
        )
        if s2023.get("short_dollar_pnl") is not None:
            if float(s2023["short_dollar_pnl"]) > 0:
                y2023 += " — short book NET HELPED in 2023"
            else:
                y2023 += " — short book NET HURT in 2023 (rallies in weak names)"

    # strip engine ref if any
    ls_payload = {k: v for k, v in ls.items() if k != "engine"}
    payload = {
        "repro": repro,
        "methodology": {
            "structure": "130% long / 30% short, net ~100%, gross ~160%",
            "long": "top-30 mom+quality; hold until ATR; refill; no-chase",
            "short": (
                "bottom-30 same score; exclude ADV bottom quartile of liquid+quality "
                "set (HTB proxy); hold until mirrored ATR cover; refill; no-chase"
            ),
            "borrow": "flat 3% annualized / 252 on short notional; HTB excluded not punitive-rated",
            "costs": "Fixed CS v2 on BUY/SELL/SHORT/COVER; borrow separate",
            "uncertain": (
                "no real locate feed; flat borrow; no margin-call model; "
                "trade_journal long-oriented — short stats from diag_closed_lots_ls"
            ),
        },
        "mq_long_only": {k: v for k, v in mq.items() if k not in ("strat_eq", "qqq_eq", "engine")},
        "ls_13030": ls_payload,
        "decision": {
            "sharpe_improved": sharpe_improved,
            "beta_reduced": beta_reduced,
            "mdd_worsened_gt_3pp": mdd_worsened,
            "adopt": adopt,
            "side": side,
            "beta_note": beta_note,
            "rule": (
                "adopt if Sharpe↑ AND beta meaningfully↓ "
                f"(≥{BETA_REDUCE_MIN} abs) AND MDD not >3pp worse"
            ),
        },
        "year_2023_callout": y2023,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha16 = hashlib.sha256(blob).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha16

    hdr = f"{'Metric':<28} {'MQ long-only':>14} {'LS 130/30':>14}"
    lines = [
        "# Mom+Quality long-only vs 130/30 long-short A/B (VALID)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha16}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        f"mq_from_checkpoint={mq.get('from_checkpoint')}",
        "",
        "Structure: long 130% EW top-30 MQ / short 30% EW bottom-30 MQ.",
        "Short HTB filter: exclude ADV < p25 of liquid+quality set.",
        "Borrow: flat 3% ann. on short notional (HTB excluded, not punitive-rated).",
        "Short risk: mirrored ATR cover (High > trough + 2.5*ATR).",
        "Exec: Fixed CS v2 on BUY/SELL/SHORT/COVER. No-chase per sleeve.",
        "UNCERTAIN: no real locate feed; flat borrow; no margin-call model.",
        "",
        "## TASK 3 — Side-by-side (+ beta)",
        hdr,
        "-" * len(hdr),
        f"{'CAGR':<28} {_pct(mq_s.get('CAGR', mq['metrics'].get('CAGR'))):>14} {_pct(ls_s.get('CAGR')):>14}",
        f"{'Sharpe':<28} {_num(mq_sharpe):>14} {_num(ls_sharpe):>14}",
        f"{'MDD':<28} {_pct(mq_mdd):>14} {_pct(ls_mdd):>14}",
        f"{'Total cost $':<28} {_money(mq['costs']['total']):>14} {_money(ls['costs']['total']):>14}",
        f"{'  long exec $':<28} {_money(mq['costs']['long_exec']):>14} {_money(ls['costs']['long_exec']):>14}",
        f"{'  short exec $':<28} {_money(mq['costs']['short_exec']):>14} {_money(ls['costs']['short_exec']):>14}",
        f"{'  borrow $':<28} {_money(mq['costs']['borrow']):>14} {_money(ls['costs']['borrow']):>14}",
        f"{'n_trades long':<28} {int(mq['n_long_trades']):>14,} {int(ls['n_long_trades']):>14,}",
        f"{'n_trades short':<28} {int(mq['n_short_trades']):>14,} {int(ls['n_short_trades']):>14,}",
        f"{'Turnover (repl/yr)':<28} {_num(mq.get('turnover_repl_per_year'), 2):>14} {_num(ls.get('turnover_repl_per_year'), 2):>14}",
        f"{'Alpha vs QQQ':<28} {_pct(mq_alpha):>14} {_pct(ls_alpha):>14}",
        f"{'Beta vs QQQ':<28} {_num(mq_beta):>14} {_num(ls_beta):>14}",
        f"{'Mean net exposure':<28} {'~1.00':>14} {_num(ls.get('mean_net_exposure'), 2):>14}",
        f"{'Mean gross exposure':<28} {'~1.00':>14} {_num(ls.get('mean_gross_exposure'), 2):>14}",
        "-" * len(hdr),
        f"BETA: {beta_note}",
        "",
        "## TASK 4 — LS per-year + 2023 short detail",
        f"{'Year':<8} {'Strat CAGR':>12} {'QQQ CAGR':>12} {'Alpha':>12} {'Short $PnL':>14} {'n_covers':>10}",
        "-" * 72,
    ]
    short_by = ls.get("short_by_year") or {}
    for row in ls["by_year"]:
        y = row["year"]
        sp = short_by.get(y) or short_by.get(str(y)) or {}
        note = "partial" if row.get("partial_year") else ""
        lines.append(
            f"{y:<8} {_pct((row.get('strategy') or {}).get('CAGR')):>12} "
            f"{_pct((row.get('QQQ') or {}).get('CAGR')):>12} {_pct(row.get('Alpha_CAGR')):>12} "
            f"{_money(sp.get('short_dollar_pnl')):>14} {int(sp.get('n_short_closes') or 0):>10} {note}"
        )
    lines += [
        "-" * 72,
        f"CALLOUT: {y2023}",
        "",
        "## Decision rule",
        f"  sharpe_improved             : {sharpe_improved}",
        f"  beta_reduced (meaningful)   : {beta_reduced}",
        f"  mdd_worsened (>3pp deeper)  : {mdd_worsened}",
        f"  ADOPT                       : {adopt}",
        f"  SIDE                        : {side}",
        "",
        "# END",
        "",
    ]

    txt = out_dir / "ls13030_ab_report.txt"
    js = out_dir / "ls13030_ab_report.json"
    txt.write_text("\n".join(lines), encoding="utf-8")
    # JSON without non-serializable leftovers
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    print(f"Wrote -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
