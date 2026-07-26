#!/usr/bin/env python3
"""Satellite 1: sector-restricted momentum+quality vs full-universe Strategy #1.

Tasks 1–5: S1 sector composition → viability → corr (adjusted threshold) → blend.

Clean git required. Does not modify Strategy #1 core modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.baseline_diagnostics import _cagr_sharpe_mdd
from engine.baseline_engine import BaselineEngineV1
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.sector_satellite_engine import SectorSatelliteEngine
from engine.sector_taxonomy import (
    SATELLITE_SECTORS,
    build_symbol_sector_map,
    sector_of_meta,
)
from engine.strategy import load_config
from engine.strategy_baseline_v1_sector import THIN_ELIGIBLE
from engine.topup_chase_diagnostics import kpi_with_trades

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "experiments" / "SATELLITE1_SECTOR"
START, END = "2022-01-01", "2026-06-30"
BASE_CORR = 0.30
S1_SHARPE = 0.268
INITIAL = 50_000_000.0
# Thin months share above this → NOT VIABLE
THIN_MONTH_FRAC_MAX = 0.25
# Cost ratio vs S1 above this → NOT VIABLE
COST_RATIO_MAX = 2.0


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
        ).strip()
    )


def _pct(x):
    return "n/a" if x is None else f"{100 * float(x):+.2f}%"


def _num(x, d=3):
    return "n/a" if x is None else f"{float(x):.{d}f}"


def _usd(x):
    return "n/a" if x is None else f"${float(x):,.0f}"


def _corr_monthly(eq_a: pd.DataFrame, eq_b: pd.DataFrame) -> Optional[float]:
    if eq_a is None or eq_b is None or eq_a.empty or eq_b.empty:
        return None
    a = eq_a["Total_Equity"].astype(float).pct_change()
    b = eq_b["Total_Equity"].astype(float).pct_change()
    ma = a.resample("ME").apply(lambda s: (1 + s).prod() - 1)
    mb = b.resample("ME").apply(lambda s: (1 + s).prod() - 1)
    j = pd.concat([ma.rename("a"), mb.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 4:
        return None
    return float(j["a"].corr(j["b"]))


def _cost_pct(engine, equity) -> Optional[float]:
    cost = float(getattr(engine, "diag_total_cost_dollars", 0.0) or 0.0)
    if equity is None or equity.empty:
        return None
    avg = float(equity["Total_Equity"].mean())
    return cost / avg if avg > 0 else None


def _turnover(engine, equity) -> Optional[float]:
    trades = engine.trade_journal.to_frame()
    if trades is None or trades.empty or equity is None or equity.empty:
        return 0.0
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    book = float(getattr(engine, "TOP_MOMENTUM_COUNT", 30) or 30)
    # use mean adaptive k if available
    months = getattr(engine, "diag_sector_months", None) or []
    if months:
        ks = [m.get("adaptive_k") for m in months if m.get("adaptive_k")]
        if ks:
            book = float(np.mean(ks))
    return float(len(trades) / years / max(book, 1.0))


def _blend_50_50(eq1, eq2, initial=INITIAL):
    aligned = pd.concat(
        [
            eq1["Total_Equity"].astype(float).rename("s1"),
            eq2["Total_Equity"].astype(float).rename("s2"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 5:
        return pd.DataFrame(), {}
    nav1 = aligned["s1"] / float(aligned["s1"].iloc[0])
    nav2 = aligned["s2"] / float(aligned["s2"].iloc[0])
    months = aligned.index.to_period("M")
    month_starts = ~months.duplicated(keep="first")
    u1 = (0.5 * initial) / float(nav1.iloc[0])
    u2 = (0.5 * initial) / float(nav2.iloc[0])
    curve = []
    for i, dt in enumerate(aligned.index):
        if i > 0 and bool(month_starts[i]):
            total = u1 * float(nav1.iloc[i]) + u2 * float(nav2.iloc[i])
            u1 = (0.5 * total) / float(nav1.iloc[i])
            u2 = (0.5 * total) / float(nav2.iloc[i])
        total = u1 * float(nav1.iloc[i]) + u2 * float(nav2.iloc[i])
        curve.append({"Date": dt, "Total_Equity": total})
    be = pd.DataFrame(curve).set_index("Date")
    return be, _cagr_sharpe_mdd(be)


def _base_cfg(cfg):
    c = dict(cfg)
    c.update(
        {
            "start_date": START,
            "end_date": END,
            "cost_model": "corwin_schultz_v2",
            "winsorize_adv": True,
            "enable_topup_chasing": False,
            "enable_quality_factor": True,
            "enable_regime_exposure": False,
            "enable_regime_exposure_fast": False,
            "enable_industry_neutral_ranking": False,
            "fixed_leverage": 1.0,
            "enable_cash_interest": False,
        }
    )
    return c


def _run_s1_with_composition(cfg) -> Dict[str, Any]:
    """Full MQ Strategy #1 + monthly holdings sector composition."""
    eng = BaselineEngineV1(config=_base_cfg(cfg), config_path="config/config.yaml")
    orig = eng._mark_qqq
    snaps: List[dict] = []

    def _mark(date_idx):
        orig(date_idx)
        current_date = eng.close_m.index[date_idx]
        if current_date < eng._run_start or not eng.portfolio:
            return
        months = eng.close_m.index.to_period("M")
        if date_idx > 0 and months[date_idx] == months[date_idx - 1]:
            return
        closes = eng.close_m.loc[current_date]
        rows = []
        total = 0.0
        for sym, qty in eng.portfolio.items():
            px = closes.get(sym)
            if px is None or pd.isna(px):
                continue
            val = float(qty) * float(px)
            total += val
            rows.append((val, sector_of_meta(eng.profile_meta.get(sym))))
        if total <= 0 or not rows:
            return
        by_c: Dict[str, float] = defaultdict(float)
        by_n: Dict[str, int] = defaultdict(int)
        for val, sec in rows:
            by_c[sec] += val
            by_n[sec] += 1
        snaps.append(
            {
                "date": current_date,
                "n_holdings": len(rows),
                "pct_capital": {k: v / total for k, v in by_c.items()},
                "pct_names": {k: n / len(rows) for k, n in by_n.items()},
            }
        )

    eng._mark_qqq = _mark  # type: ignore
    print("\n# Strategy #1 MQ (composition + reference)")
    eng.run()
    eng._mark_qqq = orig  # type: ignore
    eq = pd.DataFrame(eng.equity_curve).set_index("Date")
    qqq = pd.DataFrame(eng.qqq_equity_curve).set_index("Date")

    # Average composition across months
    cap_acc: Dict[str, List[float]] = defaultdict(list)
    nam_acc: Dict[str, List[float]] = defaultdict(list)
    all_secs = set()
    for s in snaps:
        all_secs.update(s["pct_capital"].keys())
    for s in snaps:
        for sec in all_secs:
            cap_acc[sec].append(float(s["pct_capital"].get(sec, 0.0)))
            nam_acc[sec].append(float(s["pct_names"].get(sec, 0.0)))
    avg_cap = {k: float(np.mean(v)) for k, v in cap_acc.items()}
    avg_nam = {k: float(np.mean(v)) for k, v in nam_acc.items()}

    return {
        "engine": eng,
        "equity": eq,
        "qqq": qqq,
        "metrics": kpi_with_trades(eng),
        "cost_pct": _cost_pct(eng, eq),
        "turnover": _turnover(eng, eq),
        "composition_months": snaps,
        "avg_pct_capital": avg_cap,
        "avg_pct_names": avg_nam,
        "corr_qqq": _corr_monthly(eq, qqq),
    }


def _run_sector(cfg, sector: str) -> Dict[str, Any]:
    print(f"\n# Satellite sector = {sector}")
    eng = SectorSatelliteEngine(
        config=_base_cfg(cfg), config_path="config/config.yaml", sector_bucket=sector
    )
    eng.run()
    eq = pd.DataFrame(eng.equity_curve).set_index("Date")
    qqq = pd.DataFrame(eng.qqq_equity_curve).set_index("Date")
    months = eng.diag_sector_months or []
    n_liq = [m.get("n_liquid_sector") or 0 for m in months]
    n_qual = [m.get("n_with_quality") or 0 for m in months]
    ks = [m.get("adaptive_k") or 0 for m in months]
    thin = [bool(m.get("flag_thin")) for m in months]
    return {
        "sector": sector,
        "engine": eng,
        "equity": eq,
        "qqq": qqq,
        "metrics": kpi_with_trades(eng),
        "cost_pct": _cost_pct(eng, eq),
        "turnover": _turnover(eng, eq),
        "avg_n_liquid": float(np.mean(n_liq)) if n_liq else 0.0,
        "avg_n_quality_eligible": float(np.mean(n_qual)) if n_qual else 0.0,
        "avg_adaptive_k": float(np.mean(ks)) if ks else 0.0,
        "min_n_liquid": int(min(n_liq)) if n_liq else 0,
        "thin_month_frac": float(np.mean(thin)) if thin else 1.0,
        "n_rebalance_months": len(months),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: dirty git — commit first")
        return 1

    cfg = load_config(args.config)
    n_variants = len(SATELLITE_SECTORS)
    adj_thr = BASE_CORR / math.sqrt(n_variants / 10.0)
    print(f"[GUARDRAIL] N={n_variants} adjusted_monthly_corr_threshold={adj_thr:.4f}")

    repro = build_repro_fingerprint(
        start=START,
        end=END,
        config={**cfg, "report": "satellite1_sector"},
        extra={
            "report": "satellite1_sector",
            "sectors": list(SATELLITE_SECTORS),
            "N": n_variants,
            "adj_threshold": adj_thr,
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")

    s1 = _run_s1_with_composition(cfg)

    # Universe-wide sector counts (for context)
    import pickle

    uni = pickle.load(open("data/metadata/universe.pkl", "rb"))
    smap = build_symbol_sector_map(uni.get("profile_meta") or {})
    from collections import Counter

    uni_counts = Counter(smap.values())

    sector_results = []
    for sec in SATELLITE_SECTORS:
        sector_results.append(_run_sector(cfg, sec))

    # Task 3 viability
    for r in sector_results:
        reasons = []
        viable = True
        if r["thin_month_frac"] > THIN_MONTH_FRAC_MAX:
            viable = False
            reasons.append(
                f"thin_month_frac={r['thin_month_frac']:.2f}>{THIN_MONTH_FRAC_MAX} "
                f"(eligible liquid <{THIN_ELIGIBLE})"
            )
        if r["avg_n_quality_eligible"] < THIN_ELIGIBLE:
            viable = False
            reasons.append(
                f"avg quality-eligible={r['avg_n_quality_eligible']:.1f}<{THIN_ELIGIBLE}"
            )
        cost_ratio = None
        if r["cost_pct"] is not None and s1["cost_pct"]:
            cost_ratio = float(r["cost_pct"]) / float(s1["cost_pct"])
            if cost_ratio > COST_RATIO_MAX:
                viable = False
                reasons.append(f"cost_ratio={cost_ratio:.2f}>{COST_RATIO_MAX}")
        r["cost_ratio_vs_s1"] = cost_ratio
        r["viable"] = viable
        r["viability_reasons"] = reasons

        # Task 4 corrs (compute for all; decision uses viable only)
        r["corr_s1"] = _corr_monthly(r["equity"], s1["equity"])
        r["corr_qqq"] = _corr_monthly(r["equity"], r["qqq"])
        r["corr_pass"] = (
            r["viable"]
            and r["corr_s1"] is not None
            and float(r["corr_s1"]) < adj_thr
        )

    # Task 5 blends
    validated = []
    for r in sector_results:
        if not r["corr_pass"]:
            r["blend_kpi"] = None
            r["blend_ok"] = False
            r["adopt"] = False
            continue
        blend_eq, blend_kpi = _blend_50_50(s1["equity"], r["equity"])
        blend_sharpe = blend_kpi.get("Sharpe")
        blend_ok = blend_sharpe is not None and float(blend_sharpe) > S1_SHARPE
        qk = _cagr_sharpe_mdd(r["qqq"]) if not r["qqq"].empty else {}
        blend_alpha = None
        if blend_kpi.get("CAGR") is not None and qk.get("CAGR") is not None:
            blend_alpha = float(blend_kpi["CAGR"]) - float(qk["CAGR"])
        adopt = bool(r["viable"] and r["corr_pass"] and blend_ok)
        r["blend_kpi"] = blend_kpi
        r["blend_alpha"] = blend_alpha
        r["blend_ok"] = blend_ok
        r["adopt"] = adopt
        r["sharpe_improvement"] = (
            float(blend_sharpe) - float(s1["metrics"].get("Sharpe") or S1_SHARPE)
            if blend_sharpe is not None
            else None
        )
        if adopt:
            validated.append(r)

    validated_sorted = sorted(
        validated, key=lambda x: -(x.get("sharpe_improvement") or -999)
    )

    def _slim_sector(r):
        return {
            k: v
            for k, v in r.items()
            if k not in ("engine", "equity", "qqq")
        }

    payload = {
        "repro": repro,
        "N": n_variants,
        "adjusted_threshold": adj_thr,
        "universe_sector_counts": dict(uni_counts),
        "s1": {
            "metrics": s1["metrics"],
            "cost_pct": s1["cost_pct"],
            "turnover": s1["turnover"],
            "corr_qqq": s1["corr_qqq"],
            "avg_pct_capital": s1["avg_pct_capital"],
            "avg_pct_names": s1["avg_pct_names"],
        },
        "sectors": [_slim_sector(r) for r in sector_results],
        "validated_ranked": [
            {"sector": v["sector"], "sharpe_improvement": v.get("sharpe_improvement")}
            for v in validated_sorted
        ],
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    payload_sha = hashlib.sha256(raw).hexdigest()[:16]
    repro = dict(repro)
    repro["payload_sha256_16"] = payload_sha
    payload["repro"] = repro

    # ----- report text -----
    lines: List[str] = []
    lines.append("# Satellite 1 — sector-restricted momentum+quality (VALID)")
    lines.append("")
    lines.append(f"Window: {START} → {END}")
    lines.append(f"repro_id={repro.get('fingerprint_id')}")
    lines.append(f"payload_sha256_16={payload_sha}")
    lines.append(f"git_head={repro.get('git_head')}")
    lines.append(f"git_dirty={repro.get('git_dirty')}")
    lines.append("")
    lines.append(
        "Signal: identical MQ (0.5 mom + 0.5 quality), ATR 2.5, equal-weight,"
    )
    lines.append(
        "monthly, no-chase, Fixed CS v2. ONLY change: eligible universe = one sector."
    )
    lines.append(
        f"N={n_variants} sector variants; adjusted_threshold="
        f"0.30/sqrt(N/10)={adj_thr:.4f}"
    )
    lines.append(
        "Note: profile_meta.sector holds SIC descriptions; mapped via "
        "engine/sector_taxonomy.py keyword rules to coarse buckets."
    )
    lines.append("")

    # Task 1
    lines.append("## TASK 1 — Strategy #1 holdings sector composition (monthly avg)")
    lines.append("")
    lines.append(
        f"S1 reference: CAGR {_pct(s1['metrics'].get('CAGR'))}  "
        f"Sharpe {_num(s1['metrics'].get('Sharpe'))}  "
        f"MDD {_pct(s1['metrics'].get('Maximum_Drawdown'))}  "
        f"cost {_usd(s1['metrics'].get('total_cost_dollars'))}  "
        f"monthly corr vs QQQ {_num(s1.get('corr_qqq'))}"
    )
    lines.append("")
    lines.append(f"{'Sector':<22} {'% capital':>10} {'% names':>10}")
    lines.append("-" * 46)
    caps = s1["avg_pct_capital"]
    nams = s1["avg_pct_names"]
    for sec in sorted(caps.keys(), key=lambda k: -caps[k]):
        lines.append(
            f"{sec:<22} {_pct(caps[sec]):>10} {_pct(nams.get(sec)):>10}"
        )
    lines.append("-" * 46)
    tech_c = caps.get("Technology", 0.0)
    lines.append(
        f"Technology capital share: {_pct(tech_c)} — "
        + (
            "PREMISE SUPPORTED (tech-heavy book)."
            if tech_c >= 0.30
            else "PREMISE WEAK (Technology <30% of capital)."
        )
    )
    lines.append(
        f"Universe label counts (non-exhaustive): "
        + ", ".join(f"{k}={v}" for k, v in uni_counts.most_common(8))
    )
    lines.append("")

    # Task 2/3
    lines.append("## TASK 2/3 — Per-sector universe size, K, cost viability")
    lines.append("")
    lines.append(
        f"{'Sector':<18} {'avgLiq':>7} {'avgQual':>8} {'avgK':>6} {'thin%':>7} "
        f"{'cost$':>12} {'cost%eq':>8} {'vsS1':>6} {'viable':>7}"
    )
    lines.append("-" * 96)
    lines.append(
        f"{'Strategy #1 (ref)':<18} {'250':>7} {'n/a':>8} {'30':>6} {'n/a':>7} "
        f"{_usd(s1['metrics'].get('total_cost_dollars')):>12} "
        f"{_pct(s1.get('cost_pct')):>8} {'1.00':>6} {'ref':>7}"
    )
    for r in sector_results:
        lines.append(
            f"{r['sector']:<18} {_num(r['avg_n_liquid'], 1):>7} "
            f"{_num(r['avg_n_quality_eligible'], 1):>8} "
            f"{_num(r['avg_adaptive_k'], 1):>6} "
            f"{_pct(r['thin_month_frac']):>7} "
            f"{_usd(r['metrics'].get('total_cost_dollars')):>12} "
            f"{_pct(r.get('cost_pct')):>8} "
            f"{_num(r.get('cost_ratio_vs_s1'), 2):>6} "
            f"{'YES' if r['viable'] else 'NO':>7}"
        )
        if not r["viable"]:
            for reason in r["viability_reasons"]:
                lines.append(f"    NOT VIABLE: {reason}")
    lines.append("-" * 96)
    lines.append(
        f"K rule: min(30, max(5, floor(0.25 * n_quality_eligible))); "
        f"thin = liquid_sector < {THIN_ELIGIBLE}."
    )
    lines.append("")

    # Task 4
    lines.append("## TASK 4 — Standalone performance + correlations (key)")
    lines.append("")
    lines.append(
        f"{'Sector':<18} {'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'ρ_S1':>7} "
        f"{'ρ_QQQ':>7} {'passρ':>6}"
    )
    lines.append("-" * 72)
    lines.append(
        f"{'Strategy #1':<18} {_pct(s1['metrics'].get('CAGR')):>8} "
        f"{_num(s1['metrics'].get('Sharpe')):>7} "
        f"{_pct(s1['metrics'].get('Maximum_Drawdown')):>8} "
        f"{'1.000':>7} {_num(s1.get('corr_qqq')):>7} {'ref':>6}"
    )
    for r in sector_results:
        if not r["viable"]:
            lines.append(
                f"{r['sector']:<18} {'n/a':>8} {'n/a':>7} {'n/a':>8} "
                f"{_num(r.get('corr_s1')):>7} {_num(r.get('corr_qqq')):>7} {'SKIP':>6}"
            )
            continue
        lines.append(
            f"{r['sector']:<18} {_pct(r['metrics'].get('CAGR')):>8} "
            f"{_num(r['metrics'].get('Sharpe')):>7} "
            f"{_pct(r['metrics'].get('Maximum_Drawdown')):>8} "
            f"{_num(r.get('corr_s1')):>7} {_num(r.get('corr_qqq')):>7} "
            f"{'PASS' if r['corr_pass'] else 'fail':>6}"
        )
        lines.append(
            f"    Alpha {_pct(r['metrics'].get('Alpha_CAGR'))}  "
            f"cost {_usd(r['metrics'].get('total_cost_dollars'))}  "
            f"n_trades {r['metrics'].get('n_closed_trades')}  "
            f"turnover/yr {_num(r.get('turnover'), 2)}"
        )
    lines.append("-" * 72)
    lines.append(
        f"Corr pass rule: viable AND monthly ρ vs S1 < {adj_thr:.4f}"
    )
    lines.append("")

    # Task 5
    lines.append("## TASK 5 — 50/50 blends (corr-pass sectors only)")
    lines.append("")
    any_blend = False
    for r in sector_results:
        if not r.get("corr_pass"):
            continue
        any_blend = True
        bk = r.get("blend_kpi") or {}
        lines.append(
            f"{r['sector']}: blend CAGR {_pct(bk.get('CAGR'))}  "
            f"Sharpe {_num(bk.get('Sharpe'))}  MDD {_pct(bk.get('Maximum_Drawdown'))}  "
            f"Alpha {_pct(r.get('blend_alpha'))}  "
            f"ΔSharpe vs S1 {_num(r.get('sharpe_improvement'))}  "
            f"adopt={r.get('adopt')}"
        )
    if not any_blend:
        lines.append("No sector passed the correlation threshold — no blends constructed.")
    lines.append("")

    lines.append("## DECISION — ranked validated satellite candidates")
    lines.append("")
    if not validated_sorted:
        n_viable = sum(1 for r in sector_results if r["viable"])
        n_corr = sum(1 for r in sector_results if r.get("corr_pass"))
        lines.append(
            f"NONE PASSED. viable={n_viable}/{n_variants}, "
            f"corr_pass={n_corr}/{n_variants}, threshold={adj_thr:.4f}."
        )
        lines.append(
            "CONCLUSION: sector composition is NOT the correlation driver either "
            "(or no sector sleeve clears the gate). Diversification likely needs "
            "a structural change (shorts/other asset classes) or accepting "
            "Strategy #1 alone as final — consistent with Strategy #2 mean-reversion "
            "and the 39-family factor search."
        )
    else:
        for i, v in enumerate(validated_sorted, 1):
            lines.append(
                f"{i}. {v['sector']}  ΔSharpe={_num(v.get('sharpe_improvement'))}  "
                f"blend_Sharpe={_num((v.get('blend_kpi') or {}).get('Sharpe'))}  "
                f"ρ_S1={_num(v.get('corr_s1'))}"
            )
    lines.append("")
    lines.append("# END")
    report = "\n".join(lines)

    (OUT / "satellite1_sector_report.txt").write_text(report + "\n", encoding="utf-8")
    (OUT / "satellite1_sector_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# SATELLITE1_SECTOR",
                "",
                "Sector-restricted momentum+quality satellites (same signal as Strategy #1).",
                "",
                "Run (clean git): `python3 run_satellite1_sector_report.py`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(report)
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
