#!/usr/bin/env python3
"""Strategy #2 short-term mean-reversion validation report.

Tasks: cost diagnostic → standalone A/B (w/ & w/o quality) → correlation vs
Strategy #1 → 50/50 blend → adopt/reject decision.

Requires clean git. Does not modify Strategy #1 modules.

Usage:
  python3 run_strategy2_meanrev_report.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.baseline_diagnostics import _cagr_sharpe_mdd, trade_stats
from engine.baseline_engine import BaselineEngineV1
from engine.cost_model_audit import entry_notional_over_adv_distribution
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
from engine.strategy2_engine import Strategy2MeanRevEngine
from engine.topup_chase_diagnostics import analyze_cost_components, kpi_with_trades

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent
INITIAL_CASH = 50_000_000.0

# Confirmed Strategy #1 (momentum+quality no-chase) — reference only.
S1_CONFIRMED = {
    "label": "Strategy #1 MQ (confirmed)",
    "CAGR": 0.0302,
    "Sharpe": 0.268,
    "Maximum_Drawdown": -0.2560,
    "total_cost_dollars": 13_404_546.0,
    "n_closed_trades": 1580,
    "turnover_repl_per_year": 11.74,
    "Alpha_CAGR": -0.1144,
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


def _usd(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"${float(x):,.0f}"


def _years(equity: pd.DataFrame) -> float:
    if equity is None or equity.empty:
        return 1e-9
    return max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)


def _turnover_repl_per_year(engine, equity: pd.DataFrame) -> Optional[float]:
    """Closed trades / years / typical book size — comparable to S1 ~11.7."""
    trades = engine.trade_journal.to_frame()
    if trades is None or trades.empty:
        return 0.0
    n = len(trades)
    yrs = _years(equity)
    book = float(getattr(engine, "TOP_REVERSAL_COUNT", 30))
    return float(n / yrs / book) if yrs > 0 and book > 0 else None


def _win_rate(engine) -> Optional[float]:
    trades = engine.trade_journal.to_frame()
    if trades is None or trades.empty or "final_return" not in trades.columns:
        return None
    r = pd.to_numeric(trades["final_return"], errors="coerce").dropna()
    if r.empty:
        return None
    return float((r > 0).mean())


def _cost_pct_avg_equity(engine, equity: pd.DataFrame) -> Optional[float]:
    cost = float(getattr(engine, "diag_total_cost_dollars", 0.0) or 0.0)
    if equity is None or equity.empty:
        return None
    avg_eq = float(equity["Total_Equity"].mean())
    if avg_eq <= 0:
        return None
    return cost / avg_eq


def _run_s2(cfg_base: dict, *, quality_guard: bool, label: str) -> Dict[str, Any]:
    cfg = dict(cfg_base)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = "corwin_schultz_v2"
    cfg["winsorize_adv"] = True
    cfg["enable_topup_chasing"] = False
    print("\n" + "#" * 72)
    print(f"# {label}")
    print("#" * 72)
    engine = Strategy2MeanRevEngine(
        config=cfg, config_path="config/config.yaml", enable_quality_guard=quality_guard
    )
    engine.run()
    equity = (
        pd.DataFrame(engine.equity_curve).set_index("Date")
        if engine.equity_curve
        else pd.DataFrame()
    )
    qqq = (
        pd.DataFrame(engine.qqq_equity_curve).set_index("Date")
        if engine.qqq_equity_curve
        else pd.DataFrame()
    )
    metrics = kpi_with_trades(engine)
    ts = trade_stats(engine)
    comps = analyze_cost_components(engine)
    part = entry_notional_over_adv_distribution(engine)
    return {
        "label": label,
        "quality_guard": bool(quality_guard),
        "metrics": metrics,
        "trade_stats": ts,
        "cost_components": comps,
        "participation": part,
        "turnover_repl_per_year": _turnover_repl_per_year(engine, equity),
        "win_rate": _win_rate(engine),
        "cost_pct_avg_equity": _cost_pct_avg_equity(engine, equity),
        "n_closed_trades": metrics.get("n_closed_trades"),
        "equity": equity,
        "qqq_equity": qqq,
        "engine_knobs": engine._active_strategy_knobs(),
    }


def _run_s1(cfg_base: dict) -> Dict[str, Any]:
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
    print("# Strategy #1 MQ (for correlation + blend; logic unchanged)")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path="config/config.yaml")
    engine.run()
    equity = (
        pd.DataFrame(engine.equity_curve).set_index("Date")
        if engine.equity_curve
        else pd.DataFrame()
    )
    metrics = kpi_with_trades(engine)
    return {
        "label": "Strategy #1 MQ (this session)",
        "metrics": metrics,
        "equity": equity,
        "cost_pct_avg_equity": _cost_pct_avg_equity(engine, equity),
        "turnover_repl_per_year": _turnover_repl_per_year(engine, equity),
    }


def _corr_pair(
    eq_a: pd.DataFrame, eq_b: pd.DataFrame
) -> Dict[str, Optional[float]]:
    if eq_a.empty or eq_b.empty:
        return {"daily": None, "monthly": None, "n_daily": 0, "n_monthly": 0}
    a = eq_a["Total_Equity"].astype(float).pct_change()
    b = eq_b["Total_Equity"].astype(float).pct_change()
    daily = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    d_corr = float(daily["a"].corr(daily["b"])) if len(daily) > 5 else None
    am = eq_a["Total_Equity"].astype(float).resample("ME").last().pct_change()
    bm = eq_b["Total_Equity"].astype(float).resample("ME").last().pct_change()
    monthly = pd.concat([am.rename("a"), bm.rename("b")], axis=1, join="inner").dropna()
    m_corr = float(monthly["a"].corr(monthly["b"])) if len(monthly) > 3 else None
    return {
        "daily": d_corr,
        "monthly": m_corr,
        "n_daily": int(len(daily)),
        "n_monthly": int(len(monthly)),
    }


def _blend_50_50(
    eq1: pd.DataFrame, eq2: pd.DataFrame, initial: float = INITIAL_CASH
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fixed 50/50 capital split, rebalanced to 50/50 on month-start trading days."""
    s1 = eq1["Total_Equity"].astype(float)
    s2 = eq2["Total_Equity"].astype(float)
    aligned = pd.concat([s1.rename("s1"), s2.rename("s2")], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return pd.DataFrame(), {}

    # Unitized NAVs
    nav1 = aligned["s1"] / float(aligned["s1"].iloc[0])
    nav2 = aligned["s2"] / float(aligned["s2"].iloc[0])

    months = aligned.index.to_period("M")
    month_starts = ~months.duplicated(keep="first")

    w1 = 0.5
    w2 = 0.5
    # wealth in each sleeve (dollars)
    v1 = initial * w1
    v2 = initial * w2
    # shares of each unit NAV
    u1 = v1 / float(nav1.iloc[0])
    u2 = v2 / float(nav2.iloc[0])

    curve = []
    for i, dt in enumerate(aligned.index):
        if i > 0 and bool(month_starts[i]):
            total = u1 * float(nav1.iloc[i]) + u2 * float(nav2.iloc[i])
            u1 = (0.5 * total) / float(nav1.iloc[i])
            u2 = (0.5 * total) / float(nav2.iloc[i])
        total = u1 * float(nav1.iloc[i]) + u2 * float(nav2.iloc[i])
        curve.append({"Date": dt, "Total_Equity": total})

    blend_eq = pd.DataFrame(curve).set_index("Date")
    kpi = _cagr_sharpe_mdd(blend_eq)
    # Alpha vs QQQ: use S1's QQQ path if available via eq1 window — caller adds
    return blend_eq, kpi


def _payload_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def format_report(
    *,
    repro: dict,
    s1: dict,
    s2_plain: dict,
    s2_q: dict,
    chosen: dict,
    corr: dict,
    blend_kpi: dict,
    blend_alpha: Optional[float],
    decision: dict,
) -> str:
    lines: List[str] = []
    lines.append("# Strategy #2 — weekly short-term mean-reversion (VALID)")
    lines.append("")
    lines.append(f"Window: {START} → {END}")
    lines.append(f"repro_id={repro.get('fingerprint_id') or repro.get('repro_id')}")
    lines.append(f"payload_sha256_16={repro.get('payload_sha256_16')}")
    lines.append(f"git_head={repro.get('git_head')}")
    lines.append(f"git_dirty={repro.get('git_dirty')}")
    lines.append("")
    lines.append(
        "Isolated module: engine/strategy_s2_meanrev.py + engine/strategy2_engine.py."
    )
    lines.append(
        "Strategy #1 code paths untouched. Shared infra only: panels, Fixed CS v2,"
    )
    lines.append("participation caps, no-chase fills, PIT quality helpers.")
    lines.append("")
    lines.append("Signal: weekly first-session eval; reversal_score = pctile(-ret_5d)")
    lines.append(
        f"in liquid top-{s2_plain['engine_knobs'].get('TOP_LIQUID_POOL', 250)}; "
        f"top-K={s2_plain['engine_knobs'].get('TOP_REVERSAL_COUNT', 30)}; "
        "equal-weight 1.0x."
    )
    lines.append(
        "Exit (S2-specific): time stop 10 trading days OR hard stop -15% "
        "(does NOT reverse S1's ATR-only policy)."
    )
    lines.append("")

    # ---- TASK 3 ----
    lines.append("## TASK 3 — Turnover / cost diagnostic (BEFORE return judgment)")
    lines.append("")
    lines.append(
        f"{'Metric':<28} {'S1 confirmed':>14} {'S2 rev-only':>14} {'S2+quality':>14}"
    )
    lines.append("-" * 74)
    lines.append(
        f"{'n_closed_trades':<28} {S1_CONFIRMED['n_closed_trades']:>14} "
        f"{s2_plain.get('n_closed_trades'):>14} {s2_q.get('n_closed_trades'):>14}"
    )
    lines.append(
        f"{'turnover (repl/yr)':<28} {_num(S1_CONFIRMED['turnover_repl_per_year'], 2):>14} "
        f"{_num(s2_plain.get('turnover_repl_per_year'), 2):>14} "
        f"{_num(s2_q.get('turnover_repl_per_year'), 2):>14}"
    )
    lines.append(
        f"{'total cost $':<28} {_usd(S1_CONFIRMED['total_cost_dollars']):>14} "
        f"{_usd(s2_plain['metrics'].get('total_cost_dollars')):>14} "
        f"{_usd(s2_q['metrics'].get('total_cost_dollars')):>14}"
    )
    lines.append(
        f"{'cost % of avg equity':<28} "
        f"{_pct(s1.get('cost_pct_avg_equity')):>14} "
        f"{_pct(s2_plain.get('cost_pct_avg_equity')):>14} "
        f"{_pct(s2_q.get('cost_pct_avg_equity')):>14}"
    )
    lines.append("-" * 74)

    def _comp_line(tag: str, block: dict) -> None:
        a = (block.get("cost_components") or {}).get("all_fills") or {}
        lines.append(
            f"  {tag}: spread={_usd(a.get('spread_cost'))} ({_pct(a.get('pct_spread'))})  "
            f"impact={_usd(a.get('impact_cost'))} ({_pct(a.get('pct_impact'))})  "
            f"comm={_usd(a.get('commission_cost'))}  slip={_usd(a.get('slippage_cost'))}"
        )

    lines.append("Cost component breakdown (S2):")
    _comp_line("rev-only", s2_plain)
    _comp_line("rev+quality", s2_q)
    lines.append("")
    lines.append("Entry participation (notional/ADV) distribution:")
    for tag, blk in (("rev-only", s2_plain), ("rev+quality", s2_q)):
        p = blk.get("participation") or {}
        lines.append(
            f"  {tag}: n={p.get('n')}  p50={_num(p.get('p50'), 4)}  "
            f"p90={_num(p.get('p90'), 4)}  max={_num(p.get('max'), 4)}"
        )
    lines.append("")

    s1_cost_pct = s1.get("cost_pct_avg_equity")
    # Guardrail uses the better/chosen variant later; flag both here
    for tag, blk in (("rev-only", s2_plain), ("rev+quality", s2_q)):
        c2 = blk.get("cost_pct_avg_equity")
        ratio = None
        if s1_cost_pct and c2 is not None and s1_cost_pct > 0:
            ratio = float(c2) / float(s1_cost_pct)
        flag = "OK" if (ratio is not None and ratio <= 2.0) else (
            "FLAG_>2x" if ratio is not None else "n/a"
        )
        lines.append(
            f"  Cost guardrail vs S1 ({tag}): ratio={_num(ratio, 2)} → {flag} "
            f"(limit ~2x cost-as-%-equity)"
        )
    lines.append("")

    # ---- TASK 4 ----
    lines.append("## TASK 4 — Standalone performance (context vs S1, not a bake-off)")
    lines.append("")
    lines.append(
        f"{'Metric':<16} {'S1 confirmed':>14} {'S2 rev-only':>14} {'S2+quality':>14}"
    )
    lines.append("-" * 62)
    rows = [
        ("CAGR", "CAGR", True),
        ("Sharpe", "Sharpe", False),
        ("MDD", "Maximum_Drawdown", True),
        ("Cost $", "total_cost_dollars", "usd"),
        ("n_trades", "n_closed_trades", "int"),
        ("Alpha vs QQQ", "Alpha_CAGR", True),
    ]
    for label, key, kind in rows:
        if kind is True:
            lines.append(
                f"{label:<16} {_pct(S1_CONFIRMED.get(key)):>14} "
                f"{_pct(s2_plain['metrics'].get(key)):>14} "
                f"{_pct(s2_q['metrics'].get(key)):>14}"
            )
        elif kind is False:
            lines.append(
                f"{label:<16} {_num(S1_CONFIRMED.get(key)):>14} "
                f"{_num(s2_plain['metrics'].get(key)):>14} "
                f"{_num(s2_q['metrics'].get(key)):>14}"
            )
        elif kind == "usd":
            lines.append(
                f"{label:<16} {_usd(S1_CONFIRMED.get(key)):>14} "
                f"{_usd(s2_plain['metrics'].get(key)):>14} "
                f"{_usd(s2_q['metrics'].get(key)):>14}"
            )
        else:
            lines.append(
                f"{label:<16} {S1_CONFIRMED.get(key):>14} "
                f"{s2_plain['metrics'].get(key):>14} "
                f"{s2_q['metrics'].get(key):>14}"
            )
    lines.append(
        f"{'turnover/yr':<16} {_num(S1_CONFIRMED['turnover_repl_per_year'], 2):>14} "
        f"{_num(s2_plain.get('turnover_repl_per_year'), 2):>14} "
        f"{_num(s2_q.get('turnover_repl_per_year'), 2):>14}"
    )
    lines.append(
        f"{'win_rate':<16} {'n/a':>14} "
        f"{_pct(s2_plain.get('win_rate')):>14} {_pct(s2_q.get('win_rate')):>14}"
    )
    lines.append("-" * 62)
    lines.append(f"Chosen for Tasks 5–6: {chosen['label']}")
    lines.append("")

    # ---- TASK 5 ----
    lines.append("## TASK 5 — Correlation with Strategy #1 (DECISION-CRITICAL)")
    lines.append("")
    lines.append(f"  Pair: {chosen['label']}  vs  Strategy #1 MQ (this session)")
    lines.append(
        f"  Daily correlation   : {_num(corr.get('daily'), 3)}  (n={corr.get('n_daily')})"
    )
    lines.append(
        f"  Monthly correlation : {_num(corr.get('monthly'), 3)}  (n={corr.get('n_monthly')})"
    )
    lines.append(
        "  Target for diversification value: monthly corr below ~0.30."
    )
    lines.append("")

    # ---- TASK 6 ----
    lines.append("## TASK 6 — 50/50 blended portfolio (monthly rebalance to 50/50)")
    lines.append("")
    lines.append(
        f"{'Book':<28} {'CAGR':>10} {'Sharpe':>10} {'MDD':>10} {'Alpha':>10}"
    )
    lines.append("-" * 72)
    lines.append(
        f"{'Strategy #1 alone':<28} {_pct(s1['metrics'].get('CAGR')):>10} "
        f"{_num(s1['metrics'].get('Sharpe')):>10} "
        f"{_pct(s1['metrics'].get('Maximum_Drawdown')):>10} "
        f"{_pct(s1['metrics'].get('Alpha_CAGR')):>10}"
    )
    lines.append(
        f"{chosen['label']:<28} {_pct(chosen['metrics'].get('CAGR')):>10} "
        f"{_num(chosen['metrics'].get('Sharpe')):>10} "
        f"{_pct(chosen['metrics'].get('Maximum_Drawdown')):>10} "
        f"{_pct(chosen['metrics'].get('Alpha_CAGR')):>10}"
    )
    lines.append(
        f"{'50/50 blend':<28} {_pct(blend_kpi.get('CAGR')):>10} "
        f"{_num(blend_kpi.get('Sharpe')):>10} "
        f"{_pct(blend_kpi.get('Maximum_Drawdown')):>10} "
        f"{_pct(blend_alpha):>10}"
    )
    lines.append("-" * 72)
    lines.append("")

    # ---- DECISION ----
    lines.append("## DECISION RULE")
    lines.append("")
    lines.append(f"  (a) cost ≤ ~2x S1 cost%%-equity : {decision['a_pass']}  "
                 f"(ratio={_num(decision.get('cost_ratio'), 2)})")
    lines.append(f"  (b) monthly corr < ~0.30         : {decision['b_pass']}  "
                 f"(corr={_num(decision.get('monthly_corr'), 3)})")
    lines.append(f"  (c) blend Sharpe > S1 Sharpe     : {decision['c_pass']}  "
                 f"(blend={_num(decision.get('blend_sharpe'), 3)} vs "
                 f"S1={_num(decision.get('s1_sharpe'), 3)})")
    lines.append(f"  ADOPT Strategy #2 as portfolio component: {decision['adopt']}")
    lines.append(f"  Rationale: {decision['rationale']}")
    lines.append("")
    lines.append("# END")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/STRATEGY2_MEANREV",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: working tree dirty — commit first (git_dirty=False required).")
        return 1

    pit = Path("data/fundamentals/pit_history.pkl")
    if not pit.exists() or pit.stat().st_size < 1000:
        print(f"ERROR: missing/empty {pit}")
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
            "report": "strategy2_meanrev",
        },
        extra={
            "report": "strategy2_meanrev",
            "hold_trading_days": 10,
            "stop_loss": -0.15,
            "ret_lookback": 5,
            "top_k": 30,
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")

    s2_plain = _run_s2(base_cfg, quality_guard=False, label="S2 reversal-only")
    s2_q = _run_s2(base_cfg, quality_guard=True, label="S2 reversal+quality guard")
    s1 = _run_s1(base_cfg)

    # Choose better standalone risk-adjusted (Sharpe); tie → quality (lesson-aligned)
    sh_p = s2_plain["metrics"].get("Sharpe")
    sh_q = s2_q["metrics"].get("Sharpe")
    if sh_q is None and sh_p is None:
        chosen = s2_plain
    elif sh_q is None:
        chosen = s2_plain
    elif sh_p is None:
        chosen = s2_q
    elif float(sh_q) >= float(sh_p) - 1e-6:
        chosen = s2_q
    else:
        chosen = s2_plain

    corr = _corr_pair(chosen["equity"], s1["equity"])
    blend_eq, blend_kpi = _blend_50_50(s1["equity"], chosen["equity"])
    # Alpha vs QQQ for blend: need QQQ equity — reuse from s2 run
    qqq = chosen.get("qqq_equity")
    blend_alpha = None
    if qqq is not None and not qqq.empty and blend_kpi.get("CAGR") is not None:
        qk = _cagr_sharpe_mdd(qqq)
        if qk.get("CAGR") is not None:
            blend_alpha = float(blend_kpi["CAGR"]) - float(qk["CAGR"])

    s1_cost = s1.get("cost_pct_avg_equity")
    c2_cost = chosen.get("cost_pct_avg_equity")
    cost_ratio = (
        float(c2_cost) / float(s1_cost)
        if s1_cost and c2_cost is not None and s1_cost > 0
        else None
    )
    monthly_corr = corr.get("monthly")
    blend_sharpe = blend_kpi.get("Sharpe")
    s1_sharpe = s1["metrics"].get("Sharpe")

    a_pass = cost_ratio is not None and cost_ratio <= 2.0
    b_pass = monthly_corr is not None and float(monthly_corr) < 0.30
    c_pass = (
        blend_sharpe is not None
        and s1_sharpe is not None
        and float(blend_sharpe) > float(s1_sharpe)
    )
    adopt = bool(a_pass and b_pass and c_pass)

    if not a_pass:
        rationale = (
            "FAIL (a): cost-as-%-equity too severe vs S1 — consider lower turnover "
            "(e.g. monthly eval) or tighter liquidity / smaller K before judging signal."
        )
    elif not b_pass:
        rationale = (
            "FAIL (b): correlation with S1 too high — limited diversification value; "
            "consider a different Strategy #2 family (e.g. pairs/stat-arb)."
        )
    elif not c_pass:
        rationale = (
            "FAIL (c): blend Sharpe did not beat S1 alone — signal may be marginal "
            "even with tolerable cost/corr; refine or reject before portfolio use."
        )
    else:
        rationale = (
            "PASS (a)(b)(c): cost guardrail OK, monthly corr low, blend Sharpe improves "
            "on S1 alone — adopt as validated portfolio component candidate."
        )

    decision = {
        "a_pass": a_pass,
        "b_pass": b_pass,
        "c_pass": c_pass,
        "adopt": adopt,
        "cost_ratio": cost_ratio,
        "monthly_corr": monthly_corr,
        "blend_sharpe": blend_sharpe,
        "s1_sharpe": s1_sharpe,
        "rationale": rationale,
        "chosen_label": chosen["label"],
    }

    def _slim(v: dict) -> dict:
        return {k: val for k, val in v.items() if k not in ("equity", "qqq_equity")}

    payload = {
        "repro": repro,
        "s1_confirmed": S1_CONFIRMED,
        "s1_session": _slim(s1),
        "s2_reversal_only": _slim(s2_plain),
        "s2_reversal_quality": _slim(s2_q),
        "chosen": chosen["label"],
        "correlation": corr,
        "blend_kpi": blend_kpi,
        "blend_alpha": blend_alpha,
        "decision": decision,
    }
    payload_sha = _payload_hash(payload)
    repro = dict(repro)
    repro["payload_sha256_16"] = payload_sha

    report = format_report(
        repro=repro,
        s1=s1,
        s2_plain=s2_plain,
        s2_q=s2_q,
        chosen=chosen,
        corr=corr,
        blend_kpi=blend_kpi,
        blend_alpha=blend_alpha,
        decision=decision,
    )

    (out_dir / "strategy2_meanrev_report.txt").write_text(report + "\n", encoding="utf-8")
    (out_dir / "strategy2_meanrev_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    # Save blend + chosen equities for audit
    if not blend_eq.empty:
        blend_eq.to_csv(out_dir / "blend_50_50_equity.csv")
    chosen["equity"].to_csv(out_dir / "s2_chosen_equity.csv")
    s1["equity"].to_csv(out_dir / "s1_equity.csv")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# STRATEGY2_MEANREV",
                "",
                "Standalone Strategy #2: weekly short-term reversal (Jegadeesh/Lehmann-style)",
                "with optional quality guard, 10-trading-day / -15% exits.",
                "",
                "Run (clean git): `python3 run_strategy2_meanrev_report.py`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(report)
    print(f"\nWrote {out_dir / 'strategy2_meanrev_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
