#!/usr/bin/env python3
"""20-family Strategy #2+ search with train/test corr guardrails.

1) Data-availability already documented in docs/experiments/S2_FAMILY_SEARCH_20/
2) Implement VIABLE/DEGRADED variants only
3) adjusted_threshold = 0.30 / sqrt(N/10)
4) Train 2022-01-01..2024-06-30; test 2024-07-01..2026-06-30
5) Passers → cost + full-period standalone + 50/50 blend vs S1

Clean git required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.baseline_diagnostics import _cagr_sharpe_mdd
from engine.baseline_engine import BaselineEngineV1
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.s2_family_data import (
    load_or_build_dividends,
    load_or_build_short_interest_panel,
)
from engine.s2_family_engine import S2FamilyEngine
from engine.s2_family_signals import build_catalog
from engine.strategy import load_config
from engine.topup_chase_diagnostics import kpi_with_trades

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "experiments" / "S2_FAMILY_SEARCH_20"
TRAIN_START, TRAIN_END = "2022-01-01", "2024-06-30"
TEST_START, TEST_END = "2024-07-01", "2026-06-30"
FULL_START, FULL_END = "2022-01-01", "2026-06-30"
BASE_CORR = 0.30
S1_SHARPE = 0.268
INITIAL = 50_000_000.0


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT), text=True).strip()
    )


def _pct(x):
    return "n/a" if x is None else f"{100*float(x):+.2f}%"


def _num(x, d=3):
    return "n/a" if x is None else f"{float(x):.{d}f}"


def _usd(x):
    return "n/a" if x is None else f"${float(x):,.0f}"


def _corr(eq_a: pd.DataFrame, eq_b: pd.DataFrame, start: str, end: str) -> Optional[float]:
    if eq_a is None or eq_b is None or eq_a.empty or eq_b.empty:
        return None
    a = eq_a.loc[start:end, "Total_Equity"].astype(float).pct_change()
    b = eq_b.loc[start:end, "Total_Equity"].astype(float).pct_change()
    m = a.resample("ME").apply(lambda s: (1 + s).prod() - 1)
    n = b.resample("ME").apply(lambda s: (1 + s).prod() - 1)
    j = pd.concat([m.rename("a"), n.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 4:
        return None
    return float(j["a"].corr(j["b"]))


def _cost_pct(engine, equity) -> Optional[float]:
    cost = float(getattr(engine, "diag_total_cost_dollars", 0.0) or 0.0)
    if equity is None or equity.empty:
        return None
    avg = float(equity["Total_Equity"].mean())
    return cost / avg if avg > 0 else None


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


def _run_s1(cfg) -> Dict[str, Any]:
    c = dict(cfg)
    c.update(
        {
            "start_date": FULL_START,
            "end_date": FULL_END,
            "cost_model": "corwin_schultz_v2",
            "winsorize_adv": True,
            "enable_topup_chasing": False,
            "enable_quality_factor": True,
            "enable_regime_exposure": False,
            "enable_regime_exposure_fast": False,
            "enable_industry_neutral_ranking": False,
        }
    )
    print("\n# Strategy #1 MQ reference")
    eng = BaselineEngineV1(config=c, config_path="config/config.yaml")
    eng.run()
    eq = pd.DataFrame(eng.equity_curve).set_index("Date")
    return {
        "equity": eq,
        "metrics": kpi_with_trades(eng),
        "cost_pct": _cost_pct(eng, eq),
        "engine": eng,
    }


def _run_variant(cfg, spec, si_panel, div_events) -> Dict[str, Any]:
    c = dict(cfg)
    c.update(
        {
            "start_date": FULL_START,
            "end_date": FULL_END,
            "cost_model": "corwin_schultz_v2",
            "winsorize_adv": True,
            "enable_topup_chasing": False,
        }
    )
    eng = S2FamilyEngine(
        config=c,
        config_path="config/config.yaml",
        variant_id=spec["variant_id"],
        family_id=spec["family_id"],
        score_fn=spec["score_fn"],
        cadence=spec["cadence"],
        mode=spec["mode"],
        extra=spec["extra"],
        si_panel=si_panel,
        div_events=div_events,
    )
    eng.run()
    eq = pd.DataFrame(eng.equity_curve).set_index("Date")
    return {
        "spec": {k: v for k, v in spec.items() if k != "score_fn"},
        "equity": eq,
        "metrics": kpi_with_trades(eng),
        "cost_pct": _cost_pct(eng, eq),
        "engine": eng,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: dirty git — commit first")
        return 1

    cfg = load_config(args.config)
    catalog = build_catalog()
    N = len(catalog)
    adj_threshold = BASE_CORR / math.sqrt(N / 10.0)
    print(f"[GUARDRAIL] N={N}  adjusted_monthly_corr_threshold={adj_threshold:.4f}")

    repro = build_repro_fingerprint(
        start=FULL_START,
        end=FULL_END,
        config={**cfg, "report": "s2_family_search_20"},
        extra={
            "report": "s2_family_search_20",
            "N": N,
            "adj_threshold": adj_threshold,
            "train": f"{TRAIN_START}:{TRAIN_END}",
            "test": f"{TEST_START}:{TEST_END}",
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")

    # Reference panels for downloads
    import pickle

    panels = pickle.load(open("data/prices/panels.pkl", "rb"))
    tickers = list(panels["close_m"].columns)
    trading_index = panels["close_m"].index

    # Downloads land in cache/s2_family (gitignored) — safe with clean-tree gate.
    div_events = load_or_build_dividends(
        tickers, workers=int(cfg.get("download_workers", 8)), force=False
    )
    si_panel = load_or_build_short_interest_panel(
        tickers,
        trading_index,
        workers=int(cfg.get("download_workers", 8)),
        force=False,
    )

    s1 = _run_s1(cfg)
    s1_eq = s1["equity"]

    attempt_log: List[Dict[str, Any]] = []
    train_passers: List[Dict[str, Any]] = []

    for i, spec in enumerate(catalog, 1):
        vid = spec["variant_id"]
        print("\n" + "=" * 72)
        print(f"[{i}/{N}] family={spec['family_id']} variant={vid}")
        print("=" * 72)
        try:
            res = _run_variant(cfg, spec, si_panel, div_events)
        except Exception as e:
            row = {
                "family_id": spec["family_id"],
                "variant_id": vid,
                "note": spec.get("note"),
                "error": str(e),
                "train_corr": None,
                "train_pass": False,
                "test_corr": None,
                "test_pass": False,
            }
            attempt_log.append(row)
            print("ERROR", e)
            continue

        train_c = _corr(res["equity"], s1_eq, TRAIN_START, TRAIN_END)
        train_pass = train_c is not None and abs(float(train_c)) < adj_threshold
        # Also allow negative corr (diversifying) — threshold is on absolute? 
        # User said "correlation ... below ~0.30" historically meaning low positive;
        # for diversification, negative is better. Use abs for the bar: |ρ| < threshold
        # so high positive OR high negative magnitude both... wait negative is good.
        # Standard: pass if ρ < threshold (including large negatives).
        train_pass = train_c is not None and float(train_c) < adj_threshold

        test_c = None
        test_pass = False
        if train_pass:
            test_c = _corr(res["equity"], s1_eq, TEST_START, TEST_END)
            test_pass = test_c is not None and float(test_c) < adj_threshold

        row = {
            "family_id": spec["family_id"],
            "variant_id": vid,
            "note": spec.get("note"),
            "mode": spec.get("mode"),
            "train_corr": train_c,
            "train_pass": bool(train_pass),
            "test_corr": test_c,
            "test_pass": bool(test_pass),
            "full_CAGR": res["metrics"].get("CAGR"),
            "full_Sharpe": res["metrics"].get("Sharpe"),
            "full_MDD": res["metrics"].get("Maximum_Drawdown"),
            "full_cost": res["metrics"].get("total_cost_dollars"),
            "cost_pct": res["cost_pct"],
            "n_trades": res["metrics"].get("n_closed_trades"),
        }
        attempt_log.append(row)
        print(
            f"  train_corr={_num(train_c)} pass={train_pass} | "
            f"test_corr={_num(test_c)} pass={test_pass}"
        )
        if train_pass and test_pass:
            train_passers.append(res)

    # Deep dive on double-passers
    validated: List[Dict[str, Any]] = []
    for res in train_passers:
        m = res["metrics"]
        cost_ratio = (
            float(res["cost_pct"]) / float(s1["cost_pct"])
            if res["cost_pct"] is not None and s1["cost_pct"]
            else None
        )
        cost_ok = cost_ratio is not None and cost_ratio <= 2.0
        blend_eq, blend_kpi = _blend_50_50(s1_eq, res["equity"])
        blend_sharpe = blend_kpi.get("Sharpe")
        blend_ok = blend_sharpe is not None and float(blend_sharpe) > S1_SHARPE
        # Alpha vs QQQ for blend
        qqq = (
            pd.DataFrame(res["engine"].qqq_equity_curve).set_index("Date")
            if res["engine"].qqq_equity_curve
            else pd.DataFrame()
        )
        blend_alpha = None
        if not qqq.empty and blend_kpi.get("CAGR") is not None:
            qk = _cagr_sharpe_mdd(qqq)
            if qk.get("CAGR") is not None:
                blend_alpha = float(blend_kpi["CAGR"]) - float(qk["CAGR"])
        adopt = bool(cost_ok and blend_ok)
        validated.append(
            {
                "variant_id": res["spec"]["variant_id"],
                "family_id": res["spec"]["family_id"],
                "metrics": m,
                "cost_pct": res["cost_pct"],
                "cost_ratio_vs_s1": cost_ratio,
                "cost_ok": cost_ok,
                "blend_kpi": blend_kpi,
                "blend_alpha": blend_alpha,
                "blend_ok": blend_ok,
                "adopt": adopt,
                "sharpe_improvement": (
                    float(blend_sharpe) - float(s1["metrics"].get("Sharpe") or S1_SHARPE)
                    if blend_sharpe is not None
                    else None
                ),
            }
        )

    validated_sorted = sorted(
        [v for v in validated if v["adopt"]],
        key=lambda x: -(x["sharpe_improvement"] or -999),
    )

    payload = {
        "repro": repro,
        "N": N,
        "adjusted_threshold": adj_threshold,
        "train_window": [TRAIN_START, TRAIN_END],
        "test_window": [TEST_START, TEST_END],
        "s1_metrics": s1["metrics"],
        "s1_cost_pct": s1["cost_pct"],
        "attempt_log": attempt_log,
        "n_train_pass": sum(1 for r in attempt_log if r.get("train_pass")),
        "n_both_pass": sum(1 for r in attempt_log if r.get("test_pass")),
        "validated": validated,
        "validated_adopted_ranked": validated_sorted,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    payload_sha = hashlib.sha256(raw).hexdigest()[:16]
    repro = dict(repro)
    repro["payload_sha256_16"] = payload_sha
    payload["repro"] = repro

    # ----- text report -----
    lines: List[str] = []
    lines.append("# Strategy #2+ 20-family search (VALID)")
    lines.append("")
    lines.append(f"Window full: {FULL_START} → {FULL_END}")
    lines.append(f"Train: {TRAIN_START} → {TRAIN_END} | Test: {TEST_START} → {TEST_END}")
    lines.append(f"repro_id={repro.get('fingerprint_id')}")
    lines.append(f"payload_sha256_16={payload_sha}")
    lines.append(f"git_head={repro.get('git_head')}")
    lines.append(f"git_dirty={repro.get('git_dirty')}")
    lines.append("")
    lines.append("See DATA_AVAILABILITY.md for VIABLE/DEGRADED/NOT VIABLE (all 20).")
    lines.append(
        f"Implemented N={N} variants. "
        f"adjusted_threshold = 0.30/sqrt(N/10) = {adj_threshold:.4f}"
    )
    lines.append("(No post-hoc threshold adjustment.)")
    lines.append("")
    lines.append("## FULL ATTEMPT LOG (training corr vs S1 monthly)")
    lines.append("")
    lines.append(
        f"{'fam':>3} {'variant':<28} {'trainρ':>8} {'tr':>4} {'testρ':>8} {'te':>4} {'note'}"
    )
    lines.append("-" * 100)
    for r in attempt_log:
        lines.append(
            f"{r['family_id']:>3} {r['variant_id']:<28} {_num(r.get('train_corr')):>8} "
            f"{'PASS' if r.get('train_pass') else 'fail':>4} "
            f"{_num(r.get('test_corr')):>8} "
            f"{'PASS' if r.get('test_pass') else ('n/a' if not r.get('train_pass') else 'fail'):>4} "
            f"{(r.get('note') or '')[:40]}"
        )
    lines.append("-" * 100)
    lines.append(
        f"Train passers: {payload['n_train_pass']} / {N} | "
        f"Both-window passers: {payload['n_both_pass']} / {N}"
    )
    lines.append("")

    lines.append("## Held-out + cost + blend (both-window passers only)")
    lines.append("")
    if not validated:
        lines.append("None passed both train and test correlation gates.")
    else:
        for v in validated:
            lines.append(
                f"### fam {v['family_id']} / {v['variant_id']}"
            )
            lines.append(
                f"  standalone: CAGR {_pct(v['metrics'].get('CAGR'))}  "
                f"Sharpe {_num(v['metrics'].get('Sharpe'))}  "
                f"MDD {_pct(v['metrics'].get('Maximum_Drawdown'))}  "
                f"cost {_usd(v['metrics'].get('total_cost_dollars'))}  "
                f"n_trades {v['metrics'].get('n_closed_trades')}"
            )
            lines.append(
                f"  cost%-equity {_pct(v.get('cost_pct'))}  "
                f"vs S1 ratio {_num(v.get('cost_ratio_vs_s1'), 2)}  "
                f"cost_ok={v['cost_ok']}"
            )
            bk = v.get("blend_kpi") or {}
            lines.append(
                f"  50/50 blend: CAGR {_pct(bk.get('CAGR'))}  "
                f"Sharpe {_num(bk.get('Sharpe'))}  "
                f"MDD {_pct(bk.get('Maximum_Drawdown'))}  "
                f"Alpha {_pct(v.get('blend_alpha'))}  "
                f"blend_ok={v['blend_ok']}  adopt={v['adopt']}"
            )
            lines.append("")

    lines.append("## Ranked validated candidates (adopt=True)")
    lines.append("")
    if not validated_sorted:
        lines.append(
            f"NONE PASSED. N={N}, adjusted_threshold={adj_threshold:.4f}. "
            "No variant cleared train+test corr, cost≤2x, and blend Sharpe>0.268."
        )
    else:
        for i, v in enumerate(validated_sorted, 1):
            lines.append(
                f"{i}. fam{v['family_id']}/{v['variant_id']}  "
                f"ΔSharpe(blend−S1)={_num(v.get('sharpe_improvement'))}  "
                f"blend_Sharpe={_num((v.get('blend_kpi') or {}).get('Sharpe'))}"
            )
    lines.append("")
    lines.append("# END")
    report = "\n".join(lines)

    (OUT / "s2_family_search_20_report.txt").write_text(report + "\n", encoding="utf-8")
    (OUT / "s2_family_search_20_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(attempt_log).to_csv(OUT / "attempt_log.csv", index=False)
    print(report)
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
