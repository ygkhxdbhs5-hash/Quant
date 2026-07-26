#!/usr/bin/env python3
"""Diagnose rank_too_low big winners: mom vs quality bottleneck + rank trajectory.

Tasks 1–2 only (read-only). Prints whether Task 3 acceleration variant is warranted.

Usage:
  python3 run_rank_too_low_diagnose.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.pit_fundamentals import (
    build_fund_ts_index,
    extract_quality_raw,
    get_latest_available_fundamentals,
    load_pit_history,
)
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
from engine.strategy_baseline_v1 import (
    TOP_LIQUID_POOL,
    TOP_MOMENTUM_COUNT,
    compute_mom_12_1_panel,
)
from engine.strategy_baseline_v1_quality import MOM_WEIGHT, QUALITY_WEIGHT

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent
BIGWINNER_JSON = Path(
    "docs/experiments/BASELINE_V1_BIGWINNER_DIAGNOSE/bigwinner_diagnose_report.json"
)
# Gate thresholds for Task 3
MOM_BOTTLENECK_FRAC = 0.50  # ≥50% of RTL have mom pctile < quality pctile as primary drag
NEVER_TOP30_FRAC = 0.60  # ≥60% never reach top-30 in next 1–3 months


def _git_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
    )
    return bool(out.strip())


def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)


def _month_starts(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    in_win = index[(index >= start) & (index <= end)]
    months = in_win.to_period("M")
    return list(pd.to_datetime(in_win[~months.duplicated(keep="first")]))


def _cross_section(
    *,
    date_idx: int,
    close_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    mom_m: pd.DataFrame,
    hist,
    fund_ts,
    investable: List[str],
) -> Optional[pd.DataFrame]:
    """Return DataFrame indexed by symbol with factor columns for liquid+quality set."""
    current_date = close_m.index[date_idx]
    live = [
        s
        for s in investable
        if s in close_m.columns and pd.notna(close_m[s].iloc[date_idx])
    ]
    dvol_row = pd.to_numeric(
        dvol_m.loc[current_date, [s for s in live if s in dvol_m.columns]],
        errors="coerce",
    ).dropna()
    if dvol_row.empty:
        return None
    liquid = dvol_row.nlargest(min(TOP_LIQUID_POOL, len(dvol_row))).index.tolist()
    mom_row = pd.to_numeric(mom_m.loc[current_date, liquid], errors="coerce").dropna()
    if mom_row.empty:
        return None

    gp_vals, roic_vals, opm_vals = {}, {}, {}
    for sym in mom_row.index.tolist():
        row = get_latest_available_fundamentals(hist, fund_ts, sym, current_date)
        q = extract_quality_raw(row)
        if q is None:
            continue
        gp_vals[sym] = q["gross_profitability"]
        roic_vals[sym] = q["roic"]
        opm_vals[sym] = q["op_margin"]
    if not gp_vals:
        return None

    eligible = list(gp_vals.keys())
    mom = mom_row.reindex(eligible).dropna()
    if mom.empty:
        return None
    gp = pd.Series(gp_vals).reindex(mom.index)
    roic = pd.Series(roic_vals).reindex(mom.index)
    opm = pd.Series(opm_vals).reindex(mom.index)
    mom_pct = _pct_rank(mom)
    q_pct = (_pct_rank(gp) + _pct_rank(roic) + _pct_rank(opm)) / 3.0
    combined = float(MOM_WEIGHT) * mom_pct + float(QUALITY_WEIGHT) * q_pct
    ranked = combined.sort_values(ascending=False)
    ranks = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)

    # Short-term returns for diagnosis
    ret_1m = close_m.pct_change(21).loc[current_date].reindex(mom.index)
    ret_3m = close_m.pct_change(63).loc[current_date].reindex(mom.index)

    return pd.DataFrame(
        {
            "mom_raw": mom,
            "mom_pctile": mom_pct,
            "gp_raw": gp,
            "roic_raw": roic,
            "opm_raw": opm,
            "quality_pctile": q_pct,
            "combined": combined,
            "combined_rank": ranks.reindex(mom.index),
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "n_ranked": len(ranked),
        }
    )


def _classify_bottleneck(row: pd.Series) -> str:
    """Primary drag: lower of mom vs quality pctile (with blend note)."""
    m = float(row["mom_pctile"])
    q = float(row["quality_pctile"])
    # Both moderate (≥40th) but still rank>30 → blend/dilution
    if m >= 0.40 and q >= 0.40:
        return "blend_dilution"
    if m < q - 0.05:
        return "momentum_low"
    if q < m - 0.05:
        return "quality_low"
    return "both_low"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_RANK_TOO_LOW",
    )
    parser.add_argument("--bigwinner-json", default=str(BIGWINNER_JSON))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: working tree dirty — commit first.")
        return 1
    bw_path = Path(args.bigwinner_json)
    if not bw_path.exists():
        print(f"ERROR: missing {bw_path}")
        return 1

    bw = json.loads(bw_path.read_text())
    a_events = [
        e for e in bw["task2"]["events"] if e["category"] == "a_never_entered"
    ]

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    import pickle

    with open(Path(paths.get("prices", "data/prices")) / "panels.pkl", "rb") as f:
        panels = pickle.load(f)
    with open(Path(paths.get("metadata", "data/metadata")) / "universe.pkl", "rb") as f:
        universe = pickle.load(f)
    close_m = panels["close_m"]
    dvol_m = panels["dvol_m"]
    profile_meta = universe["profile_meta"]
    benchmark = str(cfg.get("benchmark", "QQQ")).upper()
    investable = [
        s
        for s in close_m.columns
        if s != benchmark and not profile_meta.get(s, {}).get("isEtf", False)
    ]
    hist = load_pit_history(Path(paths.get("fundamentals", "data/fundamentals")))
    fund_ts = build_fund_ts_index(hist)
    mom_m = compute_mom_12_1_panel(close_m)
    month_starts = _month_starts(
        close_m.index, pd.Timestamp(START), pd.Timestamp(END)
    )
    month_starts_ts = [pd.Timestamp(m) for m in month_starts]

    repro = build_repro_fingerprint(
        start=START,
        end=END,
        config={**cfg, "enable_quality_factor": True},
        extra={
            "report": "rank_too_low_diagnose",
            "bigwinner_payload": bw.get("payload_sha256_16"),
            "n_a_events": len(a_events),
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    if repro.get("git_dirty"):
        return 1

    # Cache cross-sections by date
    cs_cache: Dict[pd.Timestamp, Optional[pd.DataFrame]] = {}

    def cs_at(dt: pd.Timestamp) -> Optional[pd.DataFrame]:
        dt = pd.Timestamp(dt)
        # snap to month-start on/before
        cands = [m for m in month_starts_ts if m <= dt]
        if not cands:
            return None
        key = cands[-1]
        if key not in cs_cache:
            idx = int(close_m.index.get_loc(key))
            cs_cache[key] = _cross_section(
                date_idx=idx,
                close_m=close_m,
                dvol_m=dvol_m,
                mom_m=mom_m,
                hist=hist,
                fund_ts=fund_ts,
                investable=investable,
            )
        return cs_cache[key]

    # Classify all (a) events → find full rank_too_low set
    rtl_rows: List[Dict[str, Any]] = []
    reason_all = Counter()
    for ev in a_events:
        sym = ev["symbol"]
        w0 = pd.Timestamp(ev["window_start"])
        cs = cs_at(w0)
        if cs is None or sym not in cs.index:
            # finer reason
            idx = int(close_m.index.get_loc(w0)) if w0 in close_m.index else None
            if idx is None:
                pos = close_m.index.searchsorted(w0, side="right") - 1
                if pos < 0:
                    reason_all["no_date"] += 1
                    continue
                idx = pos
                w0 = close_m.index[idx]
                cs = cs_at(w0)
            if cs is None or sym not in cs.index:
                # distinguish liquid/mom/pit via lightweight checks
                dvol_row = pd.to_numeric(
                    dvol_m.loc[
                        close_m.index[idx],
                        [s for s in investable if s in dvol_m.columns],
                    ],
                    errors="coerce",
                ).dropna()
                liquid = set(
                    dvol_row.nlargest(min(TOP_LIQUID_POOL, len(dvol_row))).index
                )
                if sym not in liquid:
                    reason_all["liquidity_excluded"] += 1
                elif sym not in mom_m.columns or pd.isna(mom_m[sym].iloc[idx]):
                    reason_all["missing_momentum"] += 1
                else:
                    reason_all["missing_pit_quality"] += 1
                continue
        row = cs.loc[sym]
        rank = int(row["combined_rank"]) if pd.notna(row["combined_rank"]) else None
        if rank is None:
            reason_all["unknown"] += 1
            continue
        if rank <= TOP_MOMENTUM_COUNT:
            reason_all["would_be_selected"] += 1
            continue
        reason_all["rank_too_low"] += 1
        bottleneck = _classify_bottleneck(row)
        rtl_rows.append(
            {
                "symbol": sym,
                "window_start": str(pd.Timestamp(ev["window_start"]).date()),
                "eval_date": str(w0.date()) if hasattr(w0, "date") else str(w0),
                "multiple": float(ev["multiple"]),
                "mom_raw": float(row["mom_raw"]),
                "mom_pctile": float(row["mom_pctile"]),
                "gp_raw": float(row["gp_raw"]),
                "roic_raw": float(row["roic_raw"]),
                "opm_raw": float(row["opm_raw"]),
                "quality_pctile": float(row["quality_pctile"]),
                "combined": float(row["combined"]),
                "combined_rank": rank,
                "n_ranked": int(row["n_ranked"]),
                "ret_1m": float(row["ret_1m"]) if pd.notna(row["ret_1m"]) else None,
                "ret_3m": float(row["ret_3m"]) if pd.notna(row["ret_3m"]) else None,
                "bottleneck": bottleneck,
            }
        )

    print(f"[TASK1] n_a={len(a_events)} reasons={dict(reason_all)} n_rtl={len(rtl_rows)}")

    # Task 2: trajectory
    for r in rtl_rows:
        w0 = pd.Timestamp(r["eval_date"])
        # next 1/2/3 month-starts after eval
        future = [m for m in month_starts_ts if m > w0][:3]
        traj = []
        ever_top30 = False
        for fm in future:
            cs = cs_at(fm)
            if cs is None or r["symbol"] not in cs.index:
                traj.append({"date": str(fm.date()), "rank": None, "in_top30": False})
                continue
            rk = int(cs.loc[r["symbol"], "combined_rank"])
            in30 = rk <= TOP_MOMENTUM_COUNT
            if in30:
                ever_top30 = True
            traj.append({"date": str(fm.date()), "rank": rk, "in_top30": in30})
        r["rank_trajectory"] = traj
        r["ever_top30_next_1_3m"] = ever_top30
        # best (lowest) rank in trajectory
        ranks = [t["rank"] for t in traj if t["rank"] is not None]
        r["best_rank_next_1_3m"] = int(min(ranks)) if ranks else None

    n_rtl = len(rtl_rows) or 1
    bn_counts = Counter(r["bottleneck"] for r in rtl_rows)
    mom_primary = bn_counts.get("momentum_low", 0) + bn_counts.get("both_low", 0) * 0.5
    # Stricter: momentum_low alone
    frac_mom_low = bn_counts.get("momentum_low", 0) / n_rtl
    frac_qual_low = bn_counts.get("quality_low", 0) / n_rtl
    frac_blend = bn_counts.get("blend_dilution", 0) / n_rtl
    frac_both = bn_counts.get("both_low", 0) / n_rtl

    never_top = sum(1 for r in rtl_rows if not r["ever_top30_next_1_3m"])
    frac_never_top = never_top / n_rtl

    mom_pctiles = [r["mom_pctile"] for r in rtl_rows]
    q_pctiles = [r["quality_pctile"] for r in rtl_rows]
    ret1 = [r["ret_1m"] for r in rtl_rows if r["ret_1m"] is not None]
    ret3 = [r["ret_3m"] for r in rtl_rows if r["ret_3m"] is not None]

    # Also: among momentum_low, is 1m return high? (supports accel hypothesis)
    mom_low_rows = [r for r in rtl_rows if r["bottleneck"] == "momentum_low"]
    mom_low_ret1 = [r["ret_1m"] for r in mom_low_rows if r["ret_1m"] is not None]

    # Gate Task 3
    mom_is_primary = frac_mom_low >= MOM_BOTTLENECK_FRAC or (
        frac_mom_low + frac_both >= MOM_BOTTLENECK_FRAC and frac_mom_low >= frac_qual_low
    )
    # Refine: primary reason is momentum if median mom pctile < median quality pctile
    # and mom_low is the largest single bucket
    largest_bn = max(bn_counts, key=bn_counts.get) if bn_counts else None
    mom_is_primary = largest_bn in ("momentum_low", "both_low") and (
        float(np.median(mom_pctiles)) < float(np.median(q_pctiles))
        if mom_pctiles and q_pctiles
        else False
    )
    # Or simply: mom_low is plurality and mom median < quality median
    if bn_counts.get("momentum_low", 0) >= bn_counts.get("quality_low", 0) and mom_pctiles:
        mom_is_primary = float(np.median(mom_pctiles)) + 0.05 < float(np.median(q_pctiles))

    traj_supports = frac_never_top >= NEVER_TOP30_FRAC
    # Also require elevated short-term returns among RTL (hypothesis substance)
    high_1m = (
        float(np.median(ret1)) > 0.10 if ret1 else False
    )  # median 1m > +10%

    task3_proceed = bool(mom_is_primary and traj_supports and high_1m)

    if not mom_is_primary and bn_counts.get("quality_low", 0) > bn_counts.get(
        "momentum_low", 0
    ):
        alt = (
            "STOP Task 3: quality_rank is the primary bottleneck — consider "
            "loosening/reweighting quality rather than adding 1m acceleration."
        )
    elif not traj_supports:
        alt = (
            "STOP Task 3: many RTL names climb into top-30 within 1–3m (timing lag). "
            "Check late captures in journal before changing the momentum definition."
        )
    elif not high_1m:
        alt = (
            "STOP Task 3: RTL names do not show systematically elevated 1m returns at "
            "evaluation — acceleration-skip hypothesis not supported."
        )
    elif not mom_is_primary:
        alt = (
            "STOP Task 3: momentum is not clearly the primary bottleneck "
            f"(buckets={dict(bn_counts)})."
        )
    else:
        alt = (
            "PROCEED Task 3: momentum_rank is primary drag, names stay out of top-30, "
            "and 1m returns are elevated — test 0.45*mom + 0.35*quality + 0.20*accel_1m."
        )

    payload = {
        "repro": repro,
        "methodology": {
            "source_a_events": len(a_events),
            "bigwinner_payload": bw.get("payload_sha256_16"),
            "rank_too_low_def": f"combined_rank > {TOP_MOMENTUM_COUNT} among liquid+PIT set",
            "bottleneck_rule": (
                "momentum_low if mom_pctile < quality_pctile-0.05; "
                "quality_low if reverse; blend_dilution if both ≥0.40; else both_low"
            ),
            "trajectory": "combined_rank at next 1–3 month-starts",
            "task3_gates": {
                "mom_primary": mom_is_primary,
                "never_top30_frac_ge": NEVER_TOP30_FRAC,
                "median_ret_1m_gt": 0.10,
                "proceed": task3_proceed,
            },
        },
        "task1": {
            "reason_counts_all_a": dict(reason_all),
            "n_rank_too_low": len(rtl_rows),
            "bottleneck_counts": dict(bn_counts),
            "frac_momentum_low": frac_mom_low,
            "frac_quality_low": frac_qual_low,
            "frac_blend_dilution": frac_blend,
            "frac_both_low": frac_both,
            "distributions": {
                "mom_pctile": {
                    "p25": float(np.percentile(mom_pctiles, 25)) if mom_pctiles else None,
                    "p50": float(np.median(mom_pctiles)) if mom_pctiles else None,
                    "p75": float(np.percentile(mom_pctiles, 75)) if mom_pctiles else None,
                    "mean": float(np.mean(mom_pctiles)) if mom_pctiles else None,
                },
                "quality_pctile": {
                    "p25": float(np.percentile(q_pctiles, 25)) if q_pctiles else None,
                    "p50": float(np.median(q_pctiles)) if q_pctiles else None,
                    "p75": float(np.percentile(q_pctiles, 75)) if q_pctiles else None,
                    "mean": float(np.mean(q_pctiles)) if q_pctiles else None,
                },
                "ret_1m": {
                    "p50": float(np.median(ret1)) if ret1 else None,
                    "p75": float(np.percentile(ret1, 75)) if ret1 else None,
                    "mean": float(np.mean(ret1)) if ret1 else None,
                },
                "ret_3m": {
                    "p50": float(np.median(ret3)) if ret3 else None,
                    "mean": float(np.mean(ret3)) if ret3 else None,
                },
                "combined_rank": {
                    "p50": float(np.median([r["combined_rank"] for r in rtl_rows]))
                    if rtl_rows
                    else None,
                    "mean": float(np.mean([r["combined_rank"] for r in rtl_rows]))
                    if rtl_rows
                    else None,
                },
            },
            "mom_low_subset_ret_1m_p50": (
                float(np.median(mom_low_ret1)) if mom_low_ret1 else None
            ),
            "rows": rtl_rows,
        },
        "task2": {
            "n_rtl": len(rtl_rows),
            "n_ever_top30_next_1_3m": len(rtl_rows) - never_top,
            "n_never_top30_next_1_3m": never_top,
            "frac_never_top30": frac_never_top,
            "frac_ever_top30": 1.0 - frac_never_top,
        },
        "task3_gate": {
            "mom_is_primary_bottleneck": mom_is_primary,
            "trajectory_never_top30": traj_supports,
            "elevated_1m_returns": high_1m,
            "proceed": task3_proceed,
            "decision_text": alt,
        },
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha = hashlib.sha256(raw).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha

    def pct(x):
        return f"{100 * float(x):.1f}%"

    def num(x, d=3):
        if x is None:
            return "n/a"
        return f"{float(x):.{d}f}"

    lines = [
        "# Rank-too-low big-winner diagnose (VALID)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        f"source_bigwinner_payload={bw.get('payload_sha256_16')}",
        "",
        f"Full (a) never-entered events: {len(a_events)}",
        f"Reclassified reasons: {dict(reason_all)}",
        f"rank_too_low set: {len(rtl_rows)}",
        "",
        "## TASK 1 — Factor values at evaluation (rank_too_low)",
        f"  bottleneck counts: {dict(bn_counts)}",
        f"  frac momentum_low : {pct(frac_mom_low)}",
        f"  frac quality_low  : {pct(frac_qual_low)}",
        f"  frac blend_dilution: {pct(frac_blend)}",
        f"  frac both_low     : {pct(frac_both)}",
        f"  mom_pctile    p25/p50/p75 : {num(payload['task1']['distributions']['mom_pctile']['p25'])} / "
        f"{num(payload['task1']['distributions']['mom_pctile']['p50'])} / "
        f"{num(payload['task1']['distributions']['quality_pctile']['p75'] and payload['task1']['distributions']['mom_pctile']['p75'])}",
        f"  quality_pctile p25/p50/p75: {num(payload['task1']['distributions']['quality_pctile']['p25'])} / "
        f"{num(payload['task1']['distributions']['quality_pctile']['p50'])} / "
        f"{num(payload['task1']['distributions']['quality_pctile']['p75'])}",
        f"  ret_1m p50/mean           : {num(payload['task1']['distributions']['ret_1m']['p50'])} / "
        f"{num(payload['task1']['distributions']['ret_1m']['mean'])}",
        f"  ret_3m p50/mean           : {num(payload['task1']['distributions']['ret_3m']['p50'])} / "
        f"{num(payload['task1']['distributions']['ret_3m']['mean'])}",
        f"  combined_rank p50/mean    : {num(payload['task1']['distributions']['combined_rank']['p50'], 1)} / "
        f"{num(payload['task1']['distributions']['combined_rank']['mean'], 1)}",
        f"  mom_low subset ret_1m p50 : {num(payload['task1']['mom_low_subset_ret_1m_p50'])}",
        "",
        "  Top RTL rows by multiple:",
    ]
    for r in sorted(rtl_rows, key=lambda x: -x["multiple"])[:25]:
        lines.append(
            f"    {r['symbol']:6} {r['window_start']} mult={r['multiple']:.2f}x "
            f"rank={r['combined_rank']} mom%={r['mom_pctile']:.2f} q%={r['quality_pctile']:.2f} "
            f"1m={num(r['ret_1m'])} 3m={num(r['ret_3m'])} bn={r['bottleneck']}"
        )

    lines += [
        "",
        "## TASK 2 — Rank trajectory next 1–3 months",
        f"  ever reached top-{TOP_MOMENTUM_COUNT}  : {len(rtl_rows) - never_top}  ({pct(1 - frac_never_top)})",
        f"  NEVER reached top-{TOP_MOMENTUM_COUNT} : {never_top}  ({pct(frac_never_top)})",
        "",
        "## TASK 3 gate",
        f"  mom_is_primary_bottleneck : {mom_is_primary}",
        f"  never_top30_frac≥{NEVER_TOP30_FRAC:.0%}   : {traj_supports} (actual {pct(frac_never_top)})",
        f"  elevated_1m (median>10%)  : {high_1m} (p50={num(payload['task1']['distributions']['ret_1m']['p50'])})",
        f"  PROCEED to accel variant  : {task3_proceed}",
        f"  {alt}",
        "",
        "# END",
        "",
    ]

    # Fix botched mom p75 line — rewrite cleanly
    md = payload["task1"]["distributions"]
    lines = [
        "# Rank-too-low big-winner diagnose (VALID)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        f"source_bigwinner_payload={bw.get('payload_sha256_16')}",
        "",
        f"Full (a) never-entered events: {len(a_events)}",
        f"Reclassified reasons: {dict(reason_all)}",
        f"rank_too_low set: {len(rtl_rows)}",
        "",
        "## TASK 1 — Factor values at evaluation (rank_too_low)",
        f"  bottleneck counts: {dict(bn_counts)}",
        f"  frac momentum_low  : {pct(frac_mom_low)}",
        f"  frac quality_low   : {pct(frac_qual_low)}",
        f"  frac blend_dilution: {pct(frac_blend)}",
        f"  frac both_low      : {pct(frac_both)}",
        f"  mom_pctile     p25/p50/p75 : {num(md['mom_pctile']['p25'])} / {num(md['mom_pctile']['p50'])} / {num(md['mom_pctile']['p75'])}",
        f"  quality_pctile p25/p50/p75 : {num(md['quality_pctile']['p25'])} / {num(md['quality_pctile']['p50'])} / {num(md['quality_pctile']['p75'])}",
        f"  ret_1m p50/mean            : {num(md['ret_1m']['p50'])} / {num(md['ret_1m']['mean'])}",
        f"  ret_3m p50/mean            : {num(md['ret_3m']['p50'])} / {num(md['ret_3m']['mean'])}",
        f"  combined_rank p50/mean     : {num(md['combined_rank']['p50'], 1)} / {num(md['combined_rank']['mean'], 1)}",
        f"  mom_low subset ret_1m p50  : {num(payload['task1']['mom_low_subset_ret_1m_p50'])}",
        "",
        "  Top RTL rows by multiple:",
    ]
    for r in sorted(rtl_rows, key=lambda x: -x["multiple"])[:25]:
        lines.append(
            f"    {r['symbol']:6} {r['window_start']} mult={r['multiple']:.2f}x "
            f"rank={r['combined_rank']} mom%={r['mom_pctile']:.2f} q%={r['quality_pctile']:.2f} "
            f"1m={num(r['ret_1m'])} 3m={num(r['ret_3m'])} bn={r['bottleneck']}"
        )
    lines += [
        "",
        "## TASK 2 — Rank trajectory next 1–3 months",
        f"  ever reached top-{TOP_MOMENTUM_COUNT}  : {len(rtl_rows) - never_top}  ({pct(1 - frac_never_top)})",
        f"  NEVER reached top-{TOP_MOMENTUM_COUNT} : {never_top}  ({pct(frac_never_top)})",
        "",
        "## TASK 3 gate",
        f"  mom_is_primary_bottleneck : {mom_is_primary}",
        f"  never_top30_frac≥{NEVER_TOP30_FRAC:.0%}   : {traj_supports} (actual {pct(frac_never_top)})",
        f"  elevated_1m (median>10%)  : {high_1m} (p50={num(md['ret_1m']['p50'])})",
        f"  PROCEED to accel variant  : {task3_proceed}",
        f"  {alt}",
        "",
        "# END",
        "",
    ]

    txt = out_dir / "rank_too_low_diagnose_report.txt"
    js = out_dir / "rank_too_low_diagnose_report.json"
    txt.write_text("\n".join(lines), encoding="utf-8")
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    return 0 if not task3_proceed else 0  # always 0; gate in report


if __name__ == "__main__":
    sys.exit(main())
