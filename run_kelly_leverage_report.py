#!/usr/bin/env python3
"""Kelly leverage analysis for the finalized momentum+quality baseline.

Tasks:
  1) Compute full-Kelly f* from daily excess returns (T-bill proxy)
  2) Sub-period sensitivity (why fractional Kelly, not full)
  3) Fixed-leverage backtests at 25/50/75% Kelly vs 1.0x baseline
  4) Dollar drawdown disclosure alongside CAGR

Usage (requires clean git):
  python3 run_kelly_leverage_report.py --config config/config.yaml
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
from engine.kelly_leverage import (
    DEFAULT_RF_PATH,
    fractional_levels,
    kelly_daily_from_equity,
    kelly_monthly_from_equity,
    load_tbill_annual,
    long_only_leverage,
    subperiod_kelly,
    worst_month_and_mdd,
)
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
from engine.topup_chase_diagnostics import kpi_with_trades

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent
INITIAL_CASH = 50_000_000.0

# Confirmed finalized MQ baseline (do not alter signal).
CONFIRMED = {
    "CAGR": 0.0302,
    "Sharpe": 0.268,
    "Maximum_Drawdown": -0.2560,
    "total_cost_dollars": 13_404_546.0,
    "n_closed_trades": 1580,
    "Alpha_CAGR": -0.1144,
}

SUBPERIODS = [
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2025", "2024-01-01", "2025-12-31"),
    ("2026_partial", "2026-01-01", "2026-06-30"),
]


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


def _run_engine(
    cfg_base: dict,
    *,
    leverage: float,
    enable_cash_interest: bool,
    label: str,
) -> Dict[str, Any]:
    cfg = dict(cfg_base)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = "corwin_schultz_v2"
    cfg["winsorize_adv"] = True
    cfg["enable_topup_chasing"] = False
    cfg.pop("disable_topup_chasing", None)
    cfg["enable_quality_factor"] = True
    cfg["enable_regime_exposure"] = False
    cfg["enable_regime_exposure_fast"] = False
    cfg["enable_industry_neutral_ranking"] = False
    cfg["fixed_leverage"] = float(leverage)
    cfg["allow_margin"] = bool(leverage > 1.0 + 1e-12)
    cfg["enable_cash_interest"] = bool(enable_cash_interest)
    cfg["rf_rate_path"] = DEFAULT_RF_PATH
    print("\n" + "#" * 72)
    print(f"# {label}  leverage={leverage:.4f}x  cash_interest={enable_cash_interest}")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path="config/config.yaml")
    engine.run()
    metrics = kpi_with_trades(engine)
    comparison = (engine.research_artifacts or {}).get("comparison") or {}
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
    risk = worst_month_and_mdd(strat_eq, initial_capital=INITIAL_CASH)
    # Realized average gross exposure proxy
    avg_invested = None
    if not strat_eq.empty:
        # approximate from cash interest path: not stored; use leverage target
        avg_invested = float(leverage)
    return {
        "label": label,
        "leverage": float(leverage),
        "enable_cash_interest": bool(enable_cash_interest),
        "allow_margin": bool(leverage > 1.0 + 1e-12),
        "metrics": metrics,
        "comparison": comparison,
        "risk": risk,
        "equity": strat_eq,
        "qqq_equity": qqq_eq,
        "financing_dollars_net_charge": float(
            getattr(engine, "diag_financing_dollars", 0.0)
        ),
        "avg_target_gross_exposure": avg_invested,
        "kpi_eq": _cagr_sharpe_mdd(strat_eq) if not strat_eq.empty else {},
        "qqq_kpi": _cagr_sharpe_mdd(qqq_eq) if not qqq_eq.empty else {},
    }


def _payload_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def format_report(
    *,
    repro: dict,
    rf_mean: float,
    daily_kelly: dict,
    monthly_kelly: dict,
    frac: dict,
    subrows: List[dict],
    variants: List[dict],
    note_confirmed: dict,
) -> str:
    lines: List[str] = []
    lines.append("# Kelly leverage — finalized momentum+quality baseline (VALID)")
    lines.append("")
    lines.append(f"Window: {START} → {END}")
    lines.append(f"repro_id={repro.get('repro_id')}")
    lines.append(f"payload_sha256_16={repro.get('payload_sha256_16')}")
    lines.append(f"git_head={repro.get('git_head')}")
    lines.append(f"git_dirty={repro.get('git_dirty')}")
    lines.append("")
    lines.append(
        "Signal (FINAL for this round): momentum+quality, no-chase, ATR trail 2.5x,"
    )
    lines.append(
        "equal-weight, monthly rebalance, Fixed CS v2; NO regime / industry-neutral /"
    )
    lines.append("corr-filter / hysteresis / value / low-vol / 130-30.")
    lines.append(
        f"Confirmed unlevered KPIs: CAGR {_pct(CONFIRMED['CAGR'])}, Sharpe {_num(CONFIRMED['Sharpe'])}, "
        f"MDD {_pct(CONFIRMED['Maximum_Drawdown'])}, cost {_usd(CONFIRMED['total_cost_dollars'])}, "
        f"n_trades {CONFIRMED['n_closed_trades']}, Alpha vs QQQ {_pct(CONFIRMED['Alpha_CAGR'])}."
    )
    lines.append("")
    lines.append(
        "HONESTY NOTE: Sharpe 0.268 (rf=0 project convention) is modest. After subtracting"
    )
    lines.append(
        "a T-bill proxy, excess return is near zero — Kelly leverage is expected to be"
    )
    lines.append(
        "well under 2x and often under 1x. This is NOT a path to beat QQQ; it is a"
    )
    lines.append("principled cap on how hard to press a thin validated edge.")
    lines.append("")

    # ---- TASK 1 ----
    lines.append("## TASK 1 — Kelly-optimal leverage from realized stats")
    lines.append("")
    lines.append(
        "Frequency: DAILY (primary). Why: continuous-time Kelly f*=μ/σ² matches daily"
    )
    lines.append(
        "Sharpe construction and uses ~1,100 observations vs ~50 months (less noise)."
    )
    lines.append(
        f"Risk-free proxy: FRED 3-month T-bill (DTB3), path={DEFAULT_RF_PATH};"
    )
    lines.append(
        f"period average annualized yield ≈ {100.0 * rf_mean:.2f}%."
    )
    lines.append("Monthly Kelly reported below as sensitivity only.")
    lines.append("")
    lines.append(f"  n_daily_returns          : {daily_kelly.get('n')}")
    lines.append(f"  mean return (ann)        : {_pct(daily_kelly.get('mean_return_ann'))}")
    lines.append(f"  mean rf (ann)            : {_pct(daily_kelly.get('mean_rf_ann'))}")
    lines.append(
        f"  mean EXCESS return (ann) : {_pct(daily_kelly.get('mean_excess_ann'))}"
    )
    lines.append(f"  return variance (daily)  : {_num(daily_kelly.get('variance'), 8)}")
    lines.append(f"  return variance (ann)    : {_num(daily_kelly.get('variance_ann'), 6)}")
    lines.append(f"  vol (ann)                : {_pct(daily_kelly.get('vol_ann'))}")
    lines.append(
        f"  excess Sharpe (rf-aware) : {_num(daily_kelly.get('sharpe_excess'))}"
    )
    f_star = daily_kelly.get("f_star")
    lines.append(f"  full Kelly f*            : {_num(f_star, 4)}x")
    lines.append(
        "  plain language           : "
        f"full Kelly implies running at approximately {_num(f_star, 2)}x gross exposure "
        f"(i.e. {'under' if (f_star or 0) < 1 else 'above'} fully invested)."
    )
    lines.append("")
    lines.append("  Fractional Kelly (fixed leverage = fraction × f*):")
    for k, v in frac.items():
        lines.append(
            f"    {k:<16} : {_num(v, 4)}x   "
            f"(≈ {100.0 * v:.2f}% of equity in the signal; rest in T-bills)"
        )
    lines.append("")
    lines.append(
        f"  Monthly Kelly (sensitivity): f*={_num(monthly_kelly.get('f_star'), 4)}x "
        f"(n={monthly_kelly.get('n')})"
    )
    lines.append("")

    # ---- TASK 2 ----
    lines.append("## TASK 2 — Estimation uncertainty (sub-period f*)")
    lines.append("")
    lines.append(
        f"{'Sub-period':<16} {'n':>5} {'ex_ann':>10} {'f*':>10} {'long-only':>10}"
    )
    lines.append("-" * 56)
    for row in subrows:
        lo = long_only_leverage(row.get("f_star"))
        lines.append(
            f"{str(row.get('label')):<16} {row.get('n') or 0:>5} "
            f"{_pct(row.get('mean_excess_ann')):>10} {_num(row.get('f_star'), 2):>10} "
            f"{_num(lo, 2):>10}"
        )
    lines.append("-" * 56)
    f_vals = [r.get("f_star") for r in subrows if r.get("f_star") is not None]
    if f_vals:
        best = max(f_vals)
        worst = min(f_vals)
        lines.append(
            f"  BEST sub-period full-Kelly  : {_num(best, 2)}x "
            f"(long-only clamp {_num(long_only_leverage(best), 2)}x)"
        )
        lines.append(
            f"  WORST sub-period full-Kelly : {_num(worst, 2)}x "
            f"(long-only clamp {_num(long_only_leverage(worst), 2)}x)"
        )
        lines.append(
            "  TAKEAWAY: f* flips from largely negative (no bet) in 2022–25 to very large"
        )
        lines.append(
            "  in the short 2026 partial sample. Full-Kelly on the best window would be"
        )
        lines.append(
            "  reckless look-ahead; fractional Kelly on the *full-period* estimate is the"
        )
        lines.append("  responsible choice.")
    lines.append("")

    # ---- TASK 3 + 4 interleaved for equal prominence ----
    lines.append("## TASK 3 — Fixed-leverage backtests (no look-ahead)")
    lines.append("")
    lines.append(
        "Each level uses FIXED leverage from Task 1 full-period f* (not re-estimated"
    )
    lines.append(
        "inside the backtest). Cash interest ON for all levered/fractional legs so"
    )
    lines.append(
        "uninvested cash earns the T-bill proxy and margin (if any) is charged."
    )
    lines.append(
        "1.0x baseline row also run with cash interest for apples-to-apples; confirmed"
    )
    lines.append(
        "checkpoint without cash interest is restated below for reference."
    )
    lines.append("")
    lines.append(
        f"{'Level':<22} {'Lev':>7} {'CAGR':>9} {'Sharpe':>8} {'MDD':>9} "
        f"{'Cost$':>14} {'Alpha':>9}"
    )
    lines.append("-" * 84)
    # confirmed reference first
    lines.append(
        f"{'1.0x confirmed (no rf)':<22} {'1.0000':>7} {_pct(CONFIRMED['CAGR']):>9} "
        f"{_num(CONFIRMED['Sharpe']):>8} {_pct(CONFIRMED['Maximum_Drawdown']):>9} "
        f"{_usd(CONFIRMED['total_cost_dollars']):>14} {_pct(CONFIRMED['Alpha_CAGR']):>9}"
    )
    for v in variants:
        m = v["metrics"]
        lines.append(
            f"{v['label']:<22} {_num(v['leverage'], 4):>7} {_pct(m.get('CAGR')):>9} "
            f"{_num(m.get('Sharpe')):>8} {_pct(m.get('Maximum_Drawdown')):>9} "
            f"{_usd(m.get('total_cost_dollars')):>14} {_pct(m.get('Alpha_CAGR')):>9}"
        )
    lines.append("-" * 84)
    lines.append("")
    lines.append("Cost-model check: costs should scale roughly with notional traded.")
    base_cost = None
    for v in variants:
        if abs(v["leverage"] - 1.0) < 1e-9:
            base_cost = v["metrics"].get("total_cost_dollars")
    if base_cost:
        for v in variants:
            c = v["metrics"].get("total_cost_dollars")
            if c is None:
                continue
            ratio = float(c) / float(base_cost) if base_cost else None
            lines.append(
                f"  {v['label']}: cost/1.0x_cost={_num(ratio, 3)} "
                f"(target lev={_num(v['leverage'], 3)}x)"
            )
    lines.append("")

    lines.append("## TASK 4 — Risk disclosure (equal prominence with CAGR)")
    lines.append("")
    lines.append(f"Starting capital reference: {_usd(INITIAL_CASH)}")
    lines.append("")
    lines.append(
        f"{'Level':<22} {'CAGR':>9} {'Worst month':>12} {'Worst mo $':>14} "
        f"{'MDD':>9} {'MDD $ on $50M':>14}"
    )
    lines.append("-" * 90)
    for v in variants:
        r = v["risk"]
        m = v["metrics"]
        lines.append(
            f"{v['label']:<22} {_pct(m.get('CAGR')):>9} "
            f"{_pct(r.get('worst_month_return')):>12} "
            f"{_usd(r.get('worst_month_dollar_on_initial')):>14} "
            f"{_pct(r.get('mdd')):>9} "
            f"{_usd(r.get('mdd_dollar_on_initial')):>14}"
        )
        lines.append(
            f"  plain: at {v['label']}, the worst historical drawdown would have "
            f"reduced {_usd(INITIAL_CASH)} starting capital by "
            f"{_usd(abs(r.get('mdd_dollar_on_initial') or 0))} "
            f"(peak-to-trough {_pct(r.get('mdd'))}; trough date {r.get('mdd_date')})."
        )
        lines.append(
            f"         Worst single month ({r.get('worst_month')}): "
            f"{_pct(r.get('worst_month_return'))} → "
            f"{_usd(r.get('worst_month_dollar_on_initial'))} on {_usd(INITIAL_CASH)}."
        )
    lines.append("-" * 90)
    lines.append("")

    # QQQ comparison summary
    qqq = None
    for v in variants:
        if v.get("qqq_kpi"):
            qqq = v["qqq_kpi"]
            break
    lines.append("## Plain-language close — recommended range vs QQQ")
    lines.append("")
    # pick 25% and 50% kelly variants
    v25 = next((v for v in variants if "25%" in v["label"]), None)
    v50 = next((v for v in variants if "50%" in v["label"]), None)
    v100 = next((v for v in variants if v["label"].startswith("1.0x")), None)
    if qqq and v25 and v50:
        lines.append(
            f"QQQ buy & hold over the same window: CAGR {_pct(qqq.get('CAGR'))}, "
            f"Sharpe {_num(qqq.get('Sharpe'))}, MDD {_pct(qqq.get('Maximum_Drawdown'))}."
        )
        lines.append(
            f"Recommended fractional band (25–50% Kelly ≈ {_num(v25['leverage'], 3)}x–"
            f"{_num(v50['leverage'], 3)}x): CAGR {_pct(v25['metrics'].get('CAGR'))} to "
            f"{_pct(v50['metrics'].get('CAGR'))}, vs QQQ {_pct(qqq.get('CAGR'))} — "
            f"still short by roughly "
            f"{_pct((v50['metrics'].get('CAGR') or 0) - (qqq.get('CAGR') or 0))} to "
            f"{_pct((v25['metrics'].get('CAGR') or 0) - (qqq.get('CAGR') or 0))} of CAGR."
        )
        lines.append(
            f"Drawdown cost in that band: MDD {_pct(v25['risk'].get('mdd'))} to "
            f"{_pct(v50['risk'].get('mdd'))} "
            f"({_usd(abs(v25['risk'].get('mdd_dollar_on_initial') or 0))} to "
            f"{_usd(abs(v50['risk'].get('mdd_dollar_on_initial') or 0))} on "
            f"{_usd(INITIAL_CASH)})."
        )
        if v100:
            lines.append(
                f"Running the signal at fully invested 1.0x "
                f"(≈ {_num(1.0 / (f_star or 1), 1)}× full Kelly) raised CAGR to "
                f"{_pct(v100['metrics'].get('CAGR'))} but also deepened MDD to "
                f"{_pct(v100['risk'].get('mdd'))} "
                f"({_usd(abs(v100['risk'].get('mdd_dollar_on_initial') or 0))} on "
                f"{_usd(INITIAL_CASH)}) — and still trailed QQQ by "
                f"{_pct(v100['metrics'].get('Alpha_CAGR'))}."
            )
    lines.append("")
    lines.append(
        "BOTTOM LINE: Kelly does not unlock a QQQ-beating version of this strategy."
    )
    lines.append(
        "It says the validated edge after T-bills is thin, so responsible sizing is"
    )
    lines.append(
        "fractional / under-invested — accepting a larger performance gap vs QQQ in"
    )
    lines.append("exchange for not over-betting estimation error.")
    lines.append("")
    lines.append(
        f"Confirmed 1.0x re-run check (this session, cash interest OFF): "
        f"CAGR {_pct(note_confirmed.get('CAGR'))}, Sharpe {_num(note_confirmed.get('Sharpe'))}, "
        f"MDD {_pct(note_confirmed.get('Maximum_Drawdown'))}, "
        f"cost {_usd(note_confirmed.get('total_cost_dollars'))}, "
        f"n_trades {note_confirmed.get('n_closed_trades')}."
    )
    lines.append("")
    lines.append("# END")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_KELLY_LEVERAGE",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: working tree dirty — commit first (git_dirty=False required).")
        return 1

    rf_path = Path(DEFAULT_RF_PATH)
    if not rf_path.exists():
        print(f"ERROR: missing {rf_path}")
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
            "report": "kelly_leverage",
        },
        extra={
            "report": "kelly_leverage",
            "rf_path": DEFAULT_RF_PATH,
            "fractions": [0.25, 0.5, 0.75],
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")

    # --- Baseline equity for Kelly stats (cash interest OFF = confirmed path) ---
    confirmed_run = _run_engine(
        base_cfg,
        leverage=1.0,
        enable_cash_interest=False,
        label="1.0x confirmed (cash interest OFF)",
    )
    eq = confirmed_run["equity"]
    rf = load_tbill_annual(DEFAULT_RF_PATH)
    rf_mean = float(
        rf.loc[pd.Timestamp(START) : pd.Timestamp(END)].mean()
    )

    daily_kelly = kelly_daily_from_equity(eq, rf, label="full")
    monthly_kelly = kelly_monthly_from_equity(eq, rf, label="full")
    f_star = float(daily_kelly["f_star"])
    frac = fractional_levels(f_star)
    subrows = subperiod_kelly(eq, rf, SUBPERIODS)

    # --- Fixed leverage variants (cash interest ON) ---
    variants: List[Dict[str, Any]] = []
    # 1.0x with cash interest for fair table
    variants.append(
        _run_engine(
            base_cfg,
            leverage=1.0,
            enable_cash_interest=True,
            label="1.0x (cash interest ON)",
        )
    )
    for key, lev in frac.items():
        label = {
            "25pct_kelly": "25% Kelly",
            "50pct_kelly": "50% Kelly",
            "75pct_kelly": "75% Kelly",
        }[key]
        variants.append(
            _run_engine(
                base_cfg,
                leverage=float(lev),
                enable_cash_interest=True,
                label=label,
            )
        )

    # Drop heavy frames from JSON
    def _slim(v: dict) -> dict:
        return {
            k: val
            for k, val in v.items()
            if k not in ("equity", "qqq_equity")
        }

    payload = {
        "repro": repro,
        "confirmed_checkpoint": CONFIRMED,
        "confirmed_rerun": {
            k: confirmed_run["metrics"].get(k)
            for k in (
                "CAGR",
                "Sharpe",
                "Maximum_Drawdown",
                "total_cost_dollars",
                "n_closed_trades",
                "Alpha_CAGR",
            )
        },
        "rf_mean_annual": rf_mean,
        "daily_kelly": daily_kelly,
        "monthly_kelly": monthly_kelly,
        "fractional_levels": frac,
        "subperiods": subrows,
        "variants": [_slim(v) for v in variants],
    }
    payload_sha = _payload_hash(payload)
    repro = dict(repro)
    repro["payload_sha256_16"] = payload_sha

    report = format_report(
        repro=repro,
        rf_mean=rf_mean,
        daily_kelly=daily_kelly,
        monthly_kelly=monthly_kelly,
        frac=frac,
        subrows=subrows,
        variants=variants,
        note_confirmed=confirmed_run["metrics"],
    )

    txt_path = out_dir / "kelly_leverage_report.txt"
    json_path = out_dir / "kelly_leverage_report.json"
    readme_path = out_dir / "README.md"
    txt_path.write_text(report + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# BASELINE_V1_KELLY_LEVERAGE",
                "",
                "Final-round Kelly leverage sizing for the confirmed momentum+quality",
                "no-chase baseline (Fixed CS v2).",
                "",
                "Artifacts:",
                "- `kelly_leverage_report.txt` — paste-ready Tasks 1–4",
                "- `kelly_leverage_report.json` — machine-readable payload",
                "",
                "Run: `python3 run_kelly_leverage_report.py` (clean git required).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(report)
    print(f"\nWrote {txt_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
