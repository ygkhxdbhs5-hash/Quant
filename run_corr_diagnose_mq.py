#!/usr/bin/env python3
"""Diagnose pairwise correlation inside monthly mom+quality selections.

Task 1 only (no filter A/B): for each month-start 2022-01-01→2026-06-30,
select top-K via the current global mom+quality default, then compute the
Spearman correlation matrix of CORR_WINDOW-day trailing returns among those
names. Aggregate pair-corr distribution + high-corr industry clustering.

Must run on a clean git tree.

Usage:
  python3 run_corr_diagnose_mq.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.pit_fundamentals import build_fund_ts_index, load_pit_history
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
from engine.strategy_baseline_v1 import (
    TOP_LIQUID_POOL,
    TOP_MOMENTUM_COUNT,
    compute_mom_12_1_panel,
)
from engine.strategy_baseline_v1_quality import select_monthly_candidates_mom_quality

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent
# Historical project threshold called out in the prompt (config may be looser).
HIGH_CORR_THRESHOLD = 0.80


def _git_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
    )
    return bool(out.strip())


def _industry(profile_meta: dict, sym: str) -> str:
    meta = profile_meta.get(sym) or {}
    ind = str(meta.get("industry") or "Unknown").strip()
    return ind if ind else "Unknown"


def _sector(profile_meta: dict, sym: str) -> str:
    meta = profile_meta.get(sym) or {}
    sec = str(meta.get("sector") or "Unknown").strip()
    return sec if sec else "Unknown"


def _pairwise_upper(corr: pd.DataFrame) -> np.ndarray:
    if corr is None or corr.empty or corr.shape[0] < 2:
        return np.array([], dtype=float)
    arr = corr.to_numpy(dtype=float)
    iu = np.triu_indices(arr.shape[0], k=1)
    vals = arr[iu]
    return vals[np.isfinite(vals)]


def _month_starts(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    in_win = index[(index >= start) & (index <= end)]
    months = in_win.to_period("M")
    return pd.to_datetime(in_win[~months.duplicated(keep="first")])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_CORR_DIAGNOSE",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: working tree dirty — commit first.")
        return 1

    cfg = load_config(args.config)
    corr_window = int(cfg.get("corr_window", 60))
    paths = cfg.get("paths", {})
    panels_path = Path(paths.get("prices", "data/prices")) / "panels.pkl"
    universe_path = Path(paths.get("metadata", "data/metadata")) / "universe.pkl"
    funds_dir = Path(paths.get("fundamentals", "data/fundamentals"))

    import pickle

    with open(panels_path, "rb") as f:
        panels = pickle.load(f)
    with open(universe_path, "rb") as f:
        universe = pickle.load(f)

    close_m: pd.DataFrame = panels["close_m"]
    dvol_m: pd.DataFrame = panels["dvol_m"]
    profile_meta = universe["profile_meta"]
    benchmark = str(cfg.get("benchmark", "QQQ")).upper()

    hist = load_pit_history(funds_dir)
    fund_ts = build_fund_ts_index(hist)
    if not hist:
        print("ERROR: missing PIT fundamentals")
        return 1

    mom_m = compute_mom_12_1_panel(close_m)
    rets = close_m.pct_change()

    investable = [
        s
        for s in close_m.columns
        if s != benchmark and not profile_meta.get(s, {}).get("isEtf", False)
    ]

    start = pd.Timestamp(START)
    end = pd.Timestamp(END)
    month_starts = _month_starts(close_m.index, start, end)

    repro = build_repro_fingerprint(
        start=START,
        end=END,
        config={
            **cfg,
            "cost_model": "corwin_schultz_v2",
            "winsorize_adv": True,
            "enable_topup_chasing": False,
            "enable_quality_factor": True,
            "enable_industry_neutral_ranking": False,
            "corr_window": corr_window,
        },
        extra={
            "report": "corr_diagnose_mq",
            "high_corr_threshold": HIGH_CORR_THRESHOLD,
            "selection": "mom_quality_global_topK",
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')}")
    if repro.get("git_dirty"):
        return 1

    all_pairs: List[float] = []
    month_rows: List[Dict[str, Any]] = []
    high_pair_industry: Counter = Counter()
    high_pair_same_industry = 0
    high_pair_diff_industry = 0
    high_pair_sector: Counter = Counter()
    high_pair_same_sector = 0
    high_pair_diff_sector = 0
    n_high_total = 0

    for dt in month_starts:
        idx = int(close_m.index.get_loc(dt))
        if idx < 252 or idx < corr_window:
            continue
        live = [
            s
            for s in investable
            if s in close_m.columns and pd.notna(close_m[s].iloc[idx])
        ]
        selected = select_monthly_candidates_mom_quality(
            date_idx=idx,
            close_m=close_m,
            dvol_m=dvol_m,
            mom_12_1_m=mom_m,
            eligible_symbols=live,
            fundamental_history=hist,
            fund_ts=fund_ts,
            top_liquid_pool=TOP_LIQUID_POOL,
            top_momentum_count=TOP_MOMENTUM_COUNT,
        )
        if len(selected) < 2:
            continue

        win = rets[selected].iloc[idx - corr_window + 1 : idx + 1]
        # drop names with too few valid returns
        valid_cols = [c for c in win.columns if win[c].notna().sum() >= max(20, corr_window // 3)]
        if len(valid_cols) < 2:
            continue
        corr = win[valid_cols].corr(method="spearman")
        pairs = _pairwise_upper(corr)
        if pairs.size == 0:
            continue

        all_pairs.extend(float(x) for x in pairs)
        n_high = int((pairs >= HIGH_CORR_THRESHOLD).sum())
        n_high_total += n_high
        pct_high = float(n_high / len(pairs))

        # Industry/sector clustering among high-corr pairs
        syms = list(corr.columns)
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                c = corr.iloc[i, j]
                if not np.isfinite(c) or float(c) < HIGH_CORR_THRESHOLD:
                    continue
                a, b = syms[i], syms[j]
                ia, ib = _industry(profile_meta, a), _industry(profile_meta, b)
                sa, sb = _sector(profile_meta, a), _sector(profile_meta, b)
                if ia == ib:
                    high_pair_same_industry += 1
                    high_pair_industry[ia] += 1
                else:
                    high_pair_diff_industry += 1
                if sa == sb:
                    high_pair_same_sector += 1
                    high_pair_sector[sa] += 1
                else:
                    high_pair_diff_sector += 1

        month_rows.append(
            {
                "date": str(pd.Timestamp(dt).date()),
                "n_selected": len(selected),
                "n_corr_names": len(valid_cols),
                "n_pairs": int(len(pairs)),
                "n_pairs_ge_0_80": n_high,
                "pct_pairs_ge_0_80": pct_high,
                "pair_p50": float(np.quantile(pairs, 0.50)),
                "pair_p75": float(np.quantile(pairs, 0.75)),
                "pair_p90": float(np.quantile(pairs, 0.90)),
                "pair_max": float(np.max(pairs)),
                "selected": selected,
            }
        )

    arr = np.asarray(all_pairs, dtype=float)
    if arr.size == 0:
        print("ERROR: no pairwise correlations computed")
        return 1

    dist = {
        "n_months": len(month_rows),
        "n_pairs_total": int(arr.size),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "pct_pairs_ge_0_80": float(np.mean(arr >= HIGH_CORR_THRESHOLD)),
        "n_pairs_ge_0_80": int(np.sum(arr >= HIGH_CORR_THRESHOLD)),
    }

    n_high_pairs = high_pair_same_industry + high_pair_diff_industry
    cluster = {
        "threshold": HIGH_CORR_THRESHOLD,
        "n_high_pairs": n_high_pairs,
        "pct_same_industry": (
            high_pair_same_industry / n_high_pairs if n_high_pairs else None
        ),
        "pct_diff_industry": (
            high_pair_diff_industry / n_high_pairs if n_high_pairs else None
        ),
        "pct_same_sector": (
            high_pair_same_sector / n_high_pairs if n_high_pairs else None
        ),
        "pct_diff_sector": (
            high_pair_diff_sector / n_high_pairs if n_high_pairs else None
        ),
        "top_industries_among_same_industry_high_pairs": [
            {"industry": k, "n_pairs": v}
            for k, v in high_pair_industry.most_common(15)
        ],
        "top_sectors_among_same_sector_high_pairs": [
            {"sector": k, "n_pairs": v} for k, v in high_pair_sector.most_common(10)
        ],
    }

    # Verdict heuristic for "is this a real problem?"
    # Moderate if p75 < 0.5 and pct>=0.80 under ~5%; material if pct>=0.80 > 10% or p90>=0.7
    if dist["pct_pairs_ge_0_80"] >= 0.10 or dist["p90"] >= 0.70:
        verdict = (
            "MATERIAL: high-correlation pairs are common enough that a "
            "filter could plausibly diversify the book — proceed to design A/B."
        )
        proceed = True
    elif dist["pct_pairs_ge_0_80"] >= 0.03 or dist["p75"] >= 0.50:
        verdict = (
            "MODERATE: some elevated correlations exist; a filter might help "
            "at the margin, but premise is weaker than a clear concentration crisis."
        )
        proceed = True  # still worth discussing, but flag uncertainty
    else:
        verdict = (
            "LOW: pairwise correlations are mostly moderate/low and pairs ≥0.80 "
            "are rare — like industry-neutral, a filter may be solving a "
            "non-problem; do NOT rush to A/B without a stronger rationale."
        )
        proceed = False

    payload = {
        "repro": repro,
        "methodology": {
            "selection": "mom+quality global, TOP_LIQUID_POOL→TOP_MOMENTUM_COUNT",
            "corr_window": corr_window,
            "corr_method": "spearman",
            "returns": f"{corr_window}-day trailing daily pct_change window ending on rebalance date",
            "high_corr_threshold_reported": HIGH_CORR_THRESHOLD,
            "config_corr_threshold_note": cfg.get("corr_threshold"),
            "window": {"start": START, "end": END},
        },
        "distribution": dist,
        "clustering": cluster,
        "monthly": [
            {k: v for k, v in m.items() if k != "selected"} for m in month_rows
        ],
        "verdict": {"summary": verdict, "recommend_filter_ab": proceed},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha16 = hashlib.sha256(blob).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha16

    def pct(x):
        return "n/a" if x is None else f"{100.0 * float(x):.2f}%"

    lines = [
        "# Correlation diagnose — mom+quality global default (VALID)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha16}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        "",
        f"Selection: global mom+quality top-{TOP_MOMENTUM_COUNT} after liquid top-{TOP_LIQUID_POOL}.",
        f"Corr: Spearman on trailing {corr_window}d returns; high-corr threshold = {HIGH_CORR_THRESHOLD:.2f}",
        f"(config corr_threshold currently {cfg.get('corr_threshold')} — report uses 0.80 as requested).",
        "",
        "## Pairwise correlation distribution (all months pooled)",
        f"  n_months              : {dist['n_months']}",
        f"  n_pairs_total         : {dist['n_pairs_total']:,}",
        f"  p50                   : {dist['p50']:+.3f}",
        f"  p75                   : {dist['p75']:+.3f}",
        f"  p90                   : {dist['p90']:+.3f}",
        f"  max                   : {dist['max']:+.3f}",
        f"  mean                  : {dist['mean']:+.3f}",
        f"  % pairs ≥ 0.80        : {pct(dist['pct_pairs_ge_0_80'])}  ({dist['n_pairs_ge_0_80']:,} pairs)",
        "",
        "## High-corr pair clustering (≥ 0.80)",
        f"  n_high_pairs          : {cluster['n_high_pairs']:,}",
        f"  same industry         : {pct(cluster['pct_same_industry'])}",
        f"  different industry    : {pct(cluster['pct_diff_industry'])}",
        f"  same sector           : {pct(cluster['pct_same_sector'])}",
        f"  different sector      : {pct(cluster['pct_diff_sector'])}",
        "",
        "  Top industries among same-industry high-corr pairs:",
    ]
    for row in cluster["top_industries_among_same_industry_high_pairs"][:10]:
        lines.append(f"    {row['n_pairs']:>5}  {row['industry']}")
    lines += ["", "  Top sectors among same-sector high-corr pairs:"]
    for row in cluster["top_sectors_among_same_sector_high_pairs"][:8]:
        lines.append(f"    {row['n_pairs']:>5}  {row['sector']}")
    lines += [
        "",
        "## Verdict (diagnose-first)",
        f"  {verdict}",
        f"  recommend_filter_ab   : {proceed}",
        "",
        "# END",
    ]

    txt = out_dir / "corr_diagnose_mq_report.txt"
    js = out_dir / "corr_diagnose_mq_report.json"
    txt.write_text("\n".join(lines) + "\n")
    # Keep JSON lean: drop selected lists from monthly already done
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    print(f"Wrote -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
