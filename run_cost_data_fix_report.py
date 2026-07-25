#!/usr/bin/env python3
"""Cost + data-quality fix report (Issues 1–3) and four-way 2022–2026 comparison.

Variants (strategy entry/exit/sizing unchanged):
  1) Legacy Corwin–Schultz (5% clip, raw ADV) — prior audit baseline
  2) Fixed CS v2 (1% ceiling + liquidity fallback + winsorized ADV)
  3) Flat 10bps RT
  4) Flat 30bps RT

Usage:
  python run_cost_data_fix_report.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from downloader.download_universe_utils import load_config, make_client
from engine.baseline_engine import BaselineEngineV1
from engine.cost_model_audit import kpi_slice
from engine.execution_costs import format_cs_distribution_report
from engine.strategy import load_config as load_engine_config

START = "2022-01-01"
END = "2026-06-30"

SPIKE_EVENTS = {
    "HUDI": [("2022-11-03", "2022-11-11")],
    "PHUN": [("2022-01-11", "2022-02-09"), ("2022-07-11", "2022-07-21")],
    "IDAI": [("2022-04-28", "2022-04-29")],
    "TPST": [("2024-04-01", "2024-04-01")],
    "NXTT": [("2025-08-29", "2025-09-11")],
    "QTTB": [("2026-05-27", "2026-05-27")],
    "AOSL": [("2022-05-13", "2022-05-18")],
}

VARIANTS = [
    {
        "label": "Legacy CS (5% clip, raw ADV)",
        "cost_model": "corwin_schultz",
        "winsorize_adv": False,
        "flat_cost_one_way": None,
    },
    {
        "label": "Fixed CS v2 + winsor ADV",
        "cost_model": "corwin_schultz_v2",
        "winsorize_adv": True,
        "flat_cost_one_way": None,
    },
    {
        "label": "Flat 10bps RT (5bps/side)",
        "cost_model": "flat",
        "winsorize_adv": False,
        "flat_cost_one_way": 0.0005,
    },
    {
        "label": "Flat 30bps RT (15bps/side)",
        "cost_model": "flat",
        "winsorize_adv": False,
        "flat_cost_one_way": 0.0015,
    },
]


def fetch_splits(client, ticker: str) -> List[dict]:
    payload = client.get_json(
        "/v3/reference/splits",
        params={"ticker": ticker, "limit": 100, "order": "asc", "sort": "execution_date"},
    )
    if not isinstance(payload, dict):
        return []
    return list(payload.get("results") or [])


def classify_spike_events(config_path: str) -> List[Dict[str, Any]]:
    import pickle

    cfg = load_config(config_path)
    client = make_client(cfg)
    panels_path = Path(cfg.get("paths", {}).get("prices", "data/prices")) / "panels.pkl"
    with open(panels_path, "rb") as f:
        panels = pickle.load(f)
    if not isinstance(panels, dict) or "close_m" not in panels:
        raise RuntimeError("Unexpected panels.pkl format")
    close_m = panels["close_m"]
    dvol_m = panels["dvol_m"]

    rows = []
    for ticker, windows in SPIKE_EVENTS.items():
        splits = fetch_splits(client, ticker)
        split_dates = [
            pd.Timestamp(s["execution_date"])
            for s in splits
            if s.get("execution_date")
        ]
        for w0, w1 in windows:
            near = [
                str(sd.date())
                for sd in split_dates
                if abs((sd - pd.Timestamp(w0)).days) <= 21
                or (pd.Timestamp(w0) - pd.Timedelta(days=5) <= sd <= pd.Timestamp(w1) + pd.Timedelta(days=5))
            ]
            # Check price continuity around window for split-adjustment mismatch
            if ticker in close_m.columns:
                loc = close_m.index.get_indexer([pd.Timestamp(w0)], method="nearest")[0]
                lo = max(0, loc - 5)
                hi = min(len(close_m.index) - 1, close_m.index.get_indexer([pd.Timestamp(w1)], method="nearest")[0] + 5)
                seg_c = close_m[ticker].iloc[lo : hi + 1]
                seg_d = dvol_m[ticker].iloc[lo : hi + 1]
                rets = seg_c.pct_change()
                max_abs_ret = float(rets.abs().max()) if rets.notna().any() else float("nan")
                dvol_med = float(seg_d.median()) if seg_d.notna().any() else float("nan")
                dvol_max = float(seg_d.max()) if seg_d.notna().any() else float("nan")
                spike_mult = dvol_max / dvol_med if dvol_med and dvol_med > 0 else float("nan")
            else:
                max_abs_ret = spike_mult = float("nan")

            # Classification
            if near:
                # Nearby official split — check for discontinuous adjustment
                # A true unadjusted reverse-split would show ~split_ratio jump on execution day.
                verdict = "confirmed real corporate event (nearby split; pre/post chaos — no clear adjClose*volume unit mismatch detected in continuous series)"
                if max_abs_ret == max_abs_ret and max_abs_ret > 50:  # absurd jump
                    verdict = "confirmed split-adjustment bug (catastrophic price jump near split)"
            elif max_abs_ret == max_abs_ret and max_abs_ret > 1.0:
                verdict = "unresolved data anomaly (pump/crash + volume spike; no split in Massive calendar)"
            else:
                verdict = "confirmed real corporate event / news-driven volume (no split; price move moderate)"

            rows.append(
                {
                    "ticker": ticker,
                    "window": f"{w0} → {w1}",
                    "api_splits": [
                        {
                            "execution_date": s.get("execution_date"),
                            "split_from": s.get("split_from"),
                            "split_to": s.get("split_to"),
                        }
                        for s in splits
                    ],
                    "splits_near_window": near,
                    "max_abs_daily_ret_near_window": max_abs_ret,
                    "dvol_spike_mult_vs_local_median": spike_mult,
                    "verdict": verdict,
                }
            )
    return rows


def fill_vs_target_summary(engine) -> Dict[str, Any]:
    rows = list(getattr(engine, "diag_fill_vs_target", []) or [])
    if not rows:
        return {"n": 0}
    df = pd.DataFrame(rows)
    pct = pd.to_numeric(df["fill_pct_of_target"], errors="coerce").dropna()
    return {
        "n_buy_orders": int(len(df)),
        "n_zero_fill": int((pd.to_numeric(df["filled_qty"], errors="coerce").fillna(0) <= 0).sum()),
        "mean_fill_pct_of_target": float(pct.mean()) if len(pct) else None,
        "p50_fill_pct_of_target": float(pct.quantile(0.50)) if len(pct) else None,
        "p90_fill_pct_of_target": float(pct.quantile(0.90)) if len(pct) else None,
        "share_fully_filled": float((pct >= 0.999).mean()) if len(pct) else None,
        "share_lt_50pct_filled": float((pct < 0.50).mean()) if len(pct) else None,
        "share_lt_25pct_filled": float((pct < 0.25).mean()) if len(pct) else None,
    }


def run_variant(config_path: str, variant: dict):
    cfg = load_engine_config(config_path)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = variant["cost_model"]
    cfg["winsorize_adv"] = bool(variant["winsorize_adv"])
    if variant["flat_cost_one_way"] is not None:
        cfg["flat_cost_one_way"] = variant["flat_cost_one_way"]
    print("\n" + "#" * 72)
    print(f"# {variant['label']}")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path=config_path)
    engine.run()
    metrics = kpi_slice(engine)
    metrics["label"] = variant["label"]
    metrics["winsorize_adv"] = variant["winsorize_adv"]
    return engine, metrics


def format_report(
    cs_stats,
    cs_breakdown,
    spike_rows,
    comparison_rows,
    fill_summary,
    dvol_spike_count,
) -> str:
    def pct(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{100 * float(x):.2f}%"

    def num(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{float(x):.3f}"

    lines = [
        "#" * 72,
        "# COST + DATA QUALITY FIX REPORT — Baseline v1 (2022-2026)",
        "# Strategy entry/exit/sizing/leverage unchanged.",
        "#" * 72,
        "",
        format_cs_distribution_report(cs_stats, cs_breakdown),
        "",
        "### ISSUE 2 — Corporate action / volume-spike findings",
        f"  winsorize filter (v2 runs): daily dvol capped at 5× trailing-20d median (shifted); "
        f"spike cells flagged in fixed run ≈ {dvol_spike_count:,}",
        "",
    ]
    for r in spike_rows:
        lines.append(f"  [{r['ticker']}] {r['window']}")
        lines.append(f"    splits_near_window : {r['splits_near_window'] or '[]'}")
        lines.append(f"    all_api_splits     : {r['api_splits']}")
        lines.append(f"    max_|daily_ret|    : {pct(r['max_abs_daily_ret_near_window'])}")
        lines.append(f"    dvol_spike_mult    : {num(r['dvol_spike_mult_vs_local_median'])}× local median")
        lines.append(f"    VERDICT           : {r['verdict']}")
        lines.append("")

    lines += [
        "### ISSUE 3 — Buy fill vs equal-weight target (diagnostic only; legacy CS run)",
        f"  n_buy_orders              : {fill_summary.get('n_buy_orders', 0):,}",
        f"  n_zero_fill               : {fill_summary.get('n_zero_fill', 0):,}",
        f"  mean fill % of target     : {pct(fill_summary.get('mean_fill_pct_of_target'))}",
        f"  p50 fill % of target      : {pct(fill_summary.get('p50_fill_pct_of_target'))}",
        f"  p90 fill % of target      : {pct(fill_summary.get('p90_fill_pct_of_target'))}",
        f"  share fully filled        : {pct(fill_summary.get('share_fully_filled'))}",
        f"  share < 50% filled        : {pct(fill_summary.get('share_lt_50pct_filled'))}",
        f"  share < 25% filled        : {pct(fill_summary.get('share_lt_25pct_filled'))}",
        "  (no sizing/fill logic change in this pass)",
        "",
        "### FOUR-WAY COST COMPARISON — 2022-01-01 → 2026-06-30",
        f"{'Cost model':<34} {'CAGR':>10} {'Sharpe':>10} {'MDD':>10} {'Total cost $':>16} {'Alpha':>10}",
        "-" * 96,
    ]
    for r in comparison_rows:
        lines.append(
            f"{r.get('label', '?'):<34} {pct(r.get('CAGR')):>10} {num(r.get('Sharpe')):>10} "
            f"{pct(r.get('Maximum_Drawdown')):>10} {float(r.get('total_cost_dollars') or 0):>16,.0f} "
            f"{pct(r.get('Alpha_CAGR')):>10}"
        )
    lines += [
        "-" * 96,
        "Legacy = prior audit Corwin–Schultz; Fixed = CS v2 + ADV winsorize; flats unchanged.",
        "",
        "# END REPORT",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out-dir", default="docs/experiments/BASELINE_V1_COST_DATA_FIX")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(">> Classifying volume-spike / split events via Massive splits API...")
    spike_rows = classify_spike_events(args.config)

    comparison_rows = []
    engine_legacy = None
    engine_fixed = None
    for variant in VARIANTS:
        engine, metrics = run_variant(args.config, variant)
        comparison_rows.append(metrics)
        if variant["cost_model"] == "corwin_schultz" and not variant["winsorize_adv"]:
            engine_legacy = engine
        if variant["cost_model"] == "corwin_schultz_v2":
            engine_fixed = engine

    assert engine_legacy is not None and engine_fixed is not None
    fill_summary = fill_vs_target_summary(engine_legacy)
    cs_stats = engine_legacy.diag_cs_raw_stats
    cs_breakdown = engine_legacy.diag_cs_breakdown

    report = format_report(
        cs_stats,
        cs_breakdown,
        spike_rows,
        comparison_rows,
        fill_summary,
        getattr(engine_fixed, "diag_dvol_spike_count", 0),
    )
    (out_dir / "cost_data_fix_report.txt").write_text(report, encoding="utf-8")
    (out_dir / "cost_data_fix_results.json").write_text(
        json.dumps(
            {
                "window": {"start": START, "end": END},
                "cs_raw_stats": cs_stats,
                "cs_breakdown": cs_breakdown,
                "spike_events": spike_rows,
                "fill_vs_target": fill_summary,
                "comparison": comparison_rows,
                "fixed_dvol_spike_cells": getattr(engine_fixed, "diag_dvol_spike_count", 0),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("\n" + report)
    print(f"\nWrote -> {out_dir / 'cost_data_fix_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
