#!/usr/bin/env python3
"""Diagnose whether big winners (1.5x+ in 12m) are missed at entry or cut by ATR.

Read-only. Uses mom+quality liquidity universe + confirmed trade journal.
Must run on a clean git tree.

Usage:
  python3 run_bigwinner_diagnose.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    ATR_MULTIPLIER,
    TOP_LIQUID_POOL,
    TOP_MOMENTUM_COUNT,
    compute_mom_12_1_panel,
)
from engine.strategy_baseline_v1_quality import MOM_WEIGHT, QUALITY_WEIGHT

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent
WINDOW_BARS = 252  # ~12 months trading days
MIN_MULTIPLE = 1.5
CAPTURE_HELD_THRESHOLD = 0.60  # category (c): capture ≥ 60% of window multiple-1
POSITION_NOTIONAL = 50_000_000 / 30.0  # equal-weight 1/N on $50M book
ATR_PULLBACK_NORMAL_MULT = 2.0  # drawdown < 2×ATR = "normal" pullback
SAMPLE_A_MAX = 20


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
    return pd.to_datetime(in_win[~months.duplicated(keep="first")])


def _multiple_bucket(m: float) -> str:
    if m < 2.0:
        return "1.5-2x"
    if m < 3.0:
        return "2-3x"
    if m < 5.0:
        return "3-5x"
    if m < 10.0:
        return "5-10x"
    return "10x+"


def _atr14_panel(high_m, low_m, close_m) -> pd.DataFrame:
    prev = close_m.shift(1)
    tr = (high_m - low_m).combine((high_m - prev).abs(), np.maximum).combine(
        (low_m - prev).abs(), np.maximum
    )
    return tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()


def _vol60_panel(close_m: pd.DataFrame) -> pd.DataFrame:
    return close_m.pct_change().rolling(60, min_periods=20).std()


def _liquid_set(
    dvol_m: pd.DataFrame,
    investable: Sequence[str],
    date: pd.Timestamp,
    top_n: int,
) -> set:
    syms = [s for s in investable if s in dvol_m.columns]
    row = pd.to_numeric(dvol_m.loc[date, syms], errors="coerce").dropna()
    if row.empty:
        return set()
    return set(row.nlargest(min(int(top_n), len(row))).index.tolist())


def _find_winner_windows(
    close_m: pd.DataFrame,
    investable: Sequence[str],
    dvol_m: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_bars: int = WINDOW_BARS,
    min_multiple: float = MIN_MULTIPLE,
) -> List[Dict[str, Any]]:
    """Month-start entries: forward window peak / start_close >= min_multiple.

    Deduplicate per ticker: greedy keep non-overlapping windows by descending multiple.
    """
    month_starts = list(_month_starts(close_m.index, start, end))
    # Need room for a full forward window; last start must be <= end - roughly
    candidates: List[Dict[str, Any]] = []
    idx = close_m.index

    for dt in month_starts:
        i0 = int(idx.get_loc(dt))
        i1 = i0 + int(window_bars) - 1
        if i1 >= len(idx):
            continue
        win_end_cap = idx[i1]
        if win_end_cap > end + pd.Timedelta(days=5):
            # allow peak slightly past END if window started in-period
            pass
        liquid = _liquid_set(dvol_m, investable, dt, TOP_LIQUID_POOL)
        for sym in liquid:
            if sym not in close_m.columns:
                continue
            px0 = close_m[sym].iloc[i0]
            if pd.isna(px0) or float(px0) <= 0:
                continue
            seg = close_m[sym].iloc[i0 : i1 + 1]
            if seg.notna().sum() < max(60, window_bars // 3):
                continue
            peak = float(seg.max())
            if not np.isfinite(peak) or peak <= 0:
                continue
            multiple = peak / float(px0)
            if multiple < float(min_multiple):
                continue
            peak_i = int(seg.values.argmax())
            peak_date = seg.index[peak_i]
            candidates.append(
                {
                    "symbol": sym,
                    "window_start": pd.Timestamp(dt),
                    "window_end": pd.Timestamp(peak_date),
                    "window_end_cap": pd.Timestamp(win_end_cap),
                    "start_price": float(px0),
                    "peak_price": peak,
                    "multiple": float(multiple),
                    "tradable_at_start": True,  # by construction (liquid that day)
                    "bucket": _multiple_bucket(float(multiple)),
                }
            )

    # Deduplicate overlapping windows per symbol (keep highest multiple first)
    candidates.sort(key=lambda r: (-r["multiple"], r["window_start"]))
    kept: List[Dict[str, Any]] = []
    occupied: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for ev in candidates:
        sym = ev["symbol"]
        a, b = ev["window_start"], ev["window_end"]
        overlap = False
        for x, y in occupied.get(sym, []):
            if a <= y and b >= x:
                overlap = True
                break
        if overlap:
            continue
        occupied.setdefault(sym, []).append((a, b))
        kept.append(ev)
    return kept


def _classify_event(
    ev: Dict[str, Any],
    trades: pd.DataFrame,
) -> Dict[str, Any]:
    sym = ev["symbol"]
    w0, w1 = ev["window_start"], ev["window_end"]
    full_gain = float(ev["multiple"]) - 1.0  # e.g. 1.5x → 0.5
    sub = trades[trades["symbol"] == sym].copy()
    if sub.empty:
        return {
            **ev,
            "category": "a_never_entered",
            "capture_ratio": 0.0,
            "n_round_trips": 0,
            "captured_gain": 0.0,
            "holdings": [],
        }

    # Overlapping holdings: entry < window_end and exit > window_start
    sub["entry_date"] = pd.to_datetime(sub["entry_date"])
    sub["exit_date"] = pd.to_datetime(sub["exit_date"])
    ov = sub[(sub["entry_date"] < w1) & (sub["exit_date"] > w0)].copy()
    if ov.empty:
        return {
            **ev,
            "category": "a_never_entered",
            "capture_ratio": 0.0,
            "n_round_trips": 0,
            "captured_gain": 0.0,
            "holdings": [],
        }

    holdings = []
    total_captured = 0.0
    for _, r in ov.iterrows():
        # Clip holding contribution to window: use actual trade return
        # Capture = strategy final_return (entry→exit), floored at 0 for opp-cost framing
        fr = float(r["final_return"]) if pd.notna(r["final_return"]) else 0.0
        holdings.append(
            {
                "entry_date": str(pd.Timestamp(r["entry_date"]).date()),
                "exit_date": str(pd.Timestamp(r["exit_date"]).date()),
                "entry_price": float(r["entry_price"]) if pd.notna(r["entry_price"]) else None,
                "exit_price": float(r["exit_price"]) if pd.notna(r["exit_price"]) else None,
                "final_return": fr,
                "exit_reason": str(r["exit_reason"]),
            }
        )
        total_captured += max(fr, 0.0)  # only positive capture counts toward ratio

    n_rt = int(len(ov))
    capture_ratio = float(total_captured / full_gain) if full_gain > 1e-12 else 0.0

    if n_rt >= 2:
        cat = "d_churned"
    elif capture_ratio >= CAPTURE_HELD_THRESHOLD:
        cat = "c_held_most"
    else:
        cat = "b_exited_early"

    return {
        **ev,
        "category": cat,
        "capture_ratio": capture_ratio,
        "n_round_trips": n_rt,
        "captured_gain": float(total_captured),
        "full_gain": float(full_gain),
        "holdings": holdings,
    }


def _task3_early_exit(
    events_b: List[Dict[str, Any]],
    close_m: pd.DataFrame,
    atr_m: pd.DataFrame,
    vol60_m: pd.DataFrame,
    events_c: List[Dict[str, Any]],
) -> Dict[str, Any]:
    normal_pullback = 0
    large_giveback = 0
    atr_mults_to_peak: List[float] = []
    vol_b: List[float] = []
    vol_c: List[float] = []
    details = []

    for ev in events_b:
        sym = ev["symbol"]
        if not ev.get("holdings"):
            continue
        # Use first overlapping holding for exit diagnosis
        h = ev["holdings"][0]
        entry_dt = pd.Timestamp(h["entry_date"])
        exit_dt = pd.Timestamp(h["exit_date"])
        entry_px = h.get("entry_price") or ev["start_price"]
        exit_px = h.get("exit_price")
        peak = float(ev["peak_price"])

        # ATR at exit
        atr_exit = np.nan
        if sym in atr_m.columns and exit_dt in atr_m.index:
            atr_exit = float(atr_m.at[exit_dt, sym]) if pd.notna(atr_m.at[exit_dt, sym]) else np.nan
        # Peak during holding or window
        pullback = peak - float(exit_px) if exit_px else np.nan
        if np.isfinite(atr_exit) and atr_exit > 0 and np.isfinite(pullback):
            if pullback < ATR_PULLBACK_NORMAL_MULT * atr_exit:
                normal_pullback += 1
                pull_type = "normal_lt_2xATR"
            else:
                large_giveback += 1
                pull_type = "large_giveback_ge_2xATR"
        else:
            pull_type = "unknown"

        atr_entry = np.nan
        if sym in atr_m.columns and entry_dt in atr_m.index:
            atr_entry = float(atr_m.at[entry_dt, sym]) if pd.notna(atr_m.at[entry_dt, sym]) else np.nan
        if np.isfinite(atr_entry) and atr_entry > 0:
            atr_mults_to_peak.append((peak - float(entry_px)) / atr_entry)

        if sym in vol60_m.columns and entry_dt in vol60_m.index:
            v = vol60_m.at[entry_dt, sym]
            if pd.notna(v):
                vol_b.append(float(v))

        details.append(
            {
                "symbol": sym,
                "multiple": ev["multiple"],
                "capture_ratio": ev["capture_ratio"],
                "pull_type": pull_type,
                "atr_mults_entry_to_peak": (
                    (peak - float(entry_px)) / atr_entry
                    if np.isfinite(atr_entry) and atr_entry > 0
                    else None
                ),
            }
        )

    for ev in events_c:
        for h in ev.get("holdings") or []:
            entry_dt = pd.Timestamp(h["entry_date"])
            sym = ev["symbol"]
            if sym in vol60_m.columns and entry_dt in vol60_m.index:
                v = vol60_m.at[entry_dt, sym]
                if pd.notna(v):
                    vol_c.append(float(v))

    n_diag = normal_pullback + large_giveback
    return {
        "n_b_with_exit_diag": n_diag,
        "pct_normal_pullback_lt_2xATR": (normal_pullback / n_diag) if n_diag else None,
        "pct_large_giveback_ge_2xATR": (large_giveback / n_diag) if n_diag else None,
        "n_normal_pullback": normal_pullback,
        "n_large_giveback": large_giveback,
        "atr_mults_entry_to_peak": {
            "n": len(atr_mults_to_peak),
            "p50": float(np.median(atr_mults_to_peak)) if atr_mults_to_peak else None,
            "p75": float(np.percentile(atr_mults_to_peak, 75)) if atr_mults_to_peak else None,
            "p90": float(np.percentile(atr_mults_to_peak, 90)) if atr_mults_to_peak else None,
            "max": float(np.max(atr_mults_to_peak)) if atr_mults_to_peak else None,
            "mean": float(np.mean(atr_mults_to_peak)) if atr_mults_to_peak else None,
        },
        "vol60_at_entry": {
            "b_mean": float(np.mean(vol_b)) if vol_b else None,
            "b_p50": float(np.median(vol_b)) if vol_b else None,
            "c_mean": float(np.mean(vol_c)) if vol_c else None,
            "c_p50": float(np.median(vol_c)) if vol_c else None,
            "b_higher_than_c": (
                bool(np.mean(vol_b) > np.mean(vol_c))
                if vol_b and vol_c
                else None
            ),
        },
        "details_sample": details[:30],
    }


def _rank_at_date(
    *,
    date_idx: int,
    close_m,
    dvol_m,
    mom_m,
    hist,
    fund_ts,
    investable,
    symbol: str,
) -> Dict[str, Any]:
    """Return eligibility/rank diagnostics for one symbol on a month-start."""
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
    liquid = dvol_row.nlargest(min(TOP_LIQUID_POOL, len(dvol_row))).index.tolist()
    in_liquid = symbol in liquid
    if not in_liquid:
        # Rank by ADV among all live
        adv_rank = None
        if symbol in dvol_row.index:
            adv_rank = int(dvol_row.rank(ascending=False).loc[symbol])
        return {
            "in_liquid_topN": False,
            "missing_pit_quality": None,
            "combined_rank": None,
            "adv_rank_among_live": adv_rank,
            "reason": "liquidity_excluded",
        }

    mom_row = pd.to_numeric(mom_m.loc[current_date, liquid], errors="coerce").dropna()
    if symbol not in mom_row.index:
        return {
            "in_liquid_topN": True,
            "missing_pit_quality": None,
            "combined_rank": None,
            "adv_rank_among_live": int(dvol_row.rank(ascending=False).loc[symbol])
            if symbol in dvol_row.index
            else None,
            "reason": "missing_momentum",
        }

    gp_vals, roic_vals, opm_vals = {}, {}, {}
    for sym in mom_row.index.tolist():
        row = get_latest_available_fundamentals(hist, fund_ts, sym, current_date)
        q = extract_quality_raw(row)
        if q is None:
            continue
        gp_vals[sym] = q["gross_profitability"]
        roic_vals[sym] = q["roic"]
        opm_vals[sym] = q["op_margin"]

    if symbol not in gp_vals:
        return {
            "in_liquid_topN": True,
            "missing_pit_quality": True,
            "combined_rank": None,
            "adv_rank_among_live": int(dvol_row.rank(ascending=False).loc[symbol])
            if symbol in dvol_row.index
            else None,
            "reason": "missing_pit_quality",
        }

    eligible = list(gp_vals.keys())
    mom = mom_row.reindex(eligible).dropna()
    gp = pd.Series(gp_vals).reindex(mom.index)
    roic = pd.Series(roic_vals).reindex(mom.index)
    opm = pd.Series(opm_vals).reindex(mom.index)
    quality = (_pct_rank(gp) + _pct_rank(roic) + _pct_rank(opm)) / 3.0
    combined = float(MOM_WEIGHT) * _pct_rank(mom) + float(QUALITY_WEIGHT) * quality
    ranked = combined.sort_values(ascending=False)
    rank = int(ranked.index.get_loc(symbol)) + 1 if symbol in ranked.index else None
    return {
        "in_liquid_topN": True,
        "missing_pit_quality": False,
        "combined_rank": rank,
        "n_ranked": int(len(ranked)),
        "in_top_k": bool(rank is not None and rank <= TOP_MOMENTUM_COUNT),
        "adv_rank_among_live": int(dvol_row.rank(ascending=False).loc[symbol])
        if symbol in dvol_row.index
        else None,
        "reason": (
            "rank_too_low"
            if rank is not None and rank > TOP_MOMENTUM_COUNT
            else "would_be_selected"
            if rank is not None
            else "unknown"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_BIGWINNER_DIAGNOSE",
    )
    parser.add_argument(
        "--journal",
        default="cache/baseline_v1/trade_journal.csv",
        help="Confirmed MQ default trade journal",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: working tree dirty — commit first.")
        return 1

    journal_path = Path(args.journal)
    if not journal_path.exists():
        print(f"ERROR: missing trade journal {journal_path}")
        return 1

    cfg = load_config(args.config)
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
    high_m: pd.DataFrame = panels["high_m"]
    low_m: pd.DataFrame = panels["low_m"]
    dvol_m: pd.DataFrame = panels["dvol_m"]
    profile_meta = universe["profile_meta"]
    benchmark = str(cfg.get("benchmark", "QQQ")).upper()

    investable = [
        s
        for s in close_m.columns
        if s != benchmark and not profile_meta.get(s, {}).get("isEtf", False)
    ]

    trades = pd.read_csv(journal_path)
    trades["symbol"] = trades["symbol"].astype(str)
    journal_sha = hashlib.sha256(journal_path.read_bytes()).hexdigest()[:16]

    repro = build_repro_fingerprint(
        start=START,
        end=END,
        config={
            **cfg,
            "enable_quality_factor": True,
            "enable_topup_chasing": False,
            "cost_model": "corwin_schultz_v2",
        },
        extra={
            "report": "bigwinner_diagnose",
            "journal": str(journal_path),
            "journal_sha256_16": journal_sha,
            "min_multiple": MIN_MULTIPLE,
            "window_bars": WINDOW_BARS,
            "capture_held_threshold": CAPTURE_HELD_THRESHOLD,
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')}")
    if repro.get("git_dirty"):
        return 1

    start, end = pd.Timestamp(START), pd.Timestamp(END)
    print("\n[TASK1] scanning rolling 12m winner windows...")
    raw_events = _find_winner_windows(
        close_m, investable, dvol_m, start, end, WINDOW_BARS, MIN_MULTIPLE
    )
    print(f"  n_events (deduped, tradable at start) = {len(raw_events)}")

    bucket_counts = Counter(e["bucket"] for e in raw_events)
    n_tickers = len({e["symbol"] for e in raw_events})

    print("[TASK2] classifying vs trade journal...")
    classified = [_classify_event(e, trades) for e in raw_events]
    cat_counts = Counter(e["category"] for e in classified)
    n = len(classified) or 1

    # Opportunity cost: missed full_gain (or residual) × position notional
    opp_a = 0.0
    opp_b = 0.0
    for e in classified:
        full = float(e.get("full_gain") or (e["multiple"] - 1.0))
        if e["category"] == "a_never_entered":
            opp_a += POSITION_NOTIONAL * full
        elif e["category"] == "b_exited_early":
            residual = max(full - float(e.get("captured_gain") or 0.0), 0.0)
            opp_b += POSITION_NOTIONAL * residual
        elif e["category"] == "d_churned":
            residual = max(full - float(e.get("captured_gain") or 0.0), 0.0)
            # count residual under (a)+(b) style opportunity — attribute to early/partial
            opp_b += POSITION_NOTIONAL * residual * 0.0  # keep d separate; not in a+b
    # User asked a+b combined; churn residual noted separately
    opp_d_residual = 0.0
    for e in classified:
        if e["category"] == "d_churned":
            full = float(e.get("full_gain") or (e["multiple"] - 1.0))
            opp_d_residual += POSITION_NOTIONAL * max(
                full - float(e.get("captured_gain") or 0.0), 0.0
            )

    atr_m = _atr14_panel(high_m, low_m, close_m)
    vol60_m = _vol60_panel(close_m)
    events_b = [e for e in classified if e["category"] == "b_exited_early"]
    events_c = [e for e in classified if e["category"] == "c_held_most"]
    events_a = [e for e in classified if e["category"] == "a_never_entered"]

    print("[TASK3] early-exit diagnosis...")
    task3 = _task3_early_exit(events_b, close_m, atr_m, vol60_m, events_c)

    print("[TASK4] never-entered diagnosis...")
    hist = load_pit_history(funds_dir)
    fund_ts = build_fund_ts_index(hist)
    mom_m = compute_mom_12_1_panel(close_m)

    # Sample: highest multiples first
    events_a_sorted = sorted(events_a, key=lambda e: -e["multiple"])
    sample_a = events_a_sorted[:SAMPLE_A_MAX] if len(events_a_sorted) > SAMPLE_A_MAX else events_a_sorted
    task4_rows = []
    reason_counts = Counter()
    for e in sample_a:
        dt = e["window_start"]
        if dt not in close_m.index:
            # nearest prior
            pos = close_m.index.searchsorted(dt, side="right") - 1
            if pos < 0:
                continue
            dt = close_m.index[pos]
        idx = int(close_m.index.get_loc(dt))
        info = _rank_at_date(
            date_idx=idx,
            close_m=close_m,
            dvol_m=dvol_m,
            mom_m=mom_m,
            hist=hist,
            fund_ts=fund_ts,
            investable=investable,
            symbol=e["symbol"],
        )
        reason_counts[info["reason"]] += 1
        task4_rows.append(
            {
                "symbol": e["symbol"],
                "window_start": str(e["window_start"].date()),
                "multiple": e["multiple"],
                **info,
            }
        )

    # Recommendation
    pct_a = cat_counts.get("a_never_entered", 0) / n
    pct_b = cat_counts.get("b_exited_early", 0) / n
    pct_c = cat_counts.get("c_held_most", 0) / n
    pct_d = cat_counts.get("d_churned", 0) / n
    if pct_c >= 0.50 and pct_a < 0.25 and pct_b < 0.25:
        reco = (
            "NON-PROBLEM: most big-winner windows were already captured well (c). "
            "Neither wider ATR stops nor entry redesign is strongly motivated by this scan."
        )
        reco_side = "non_problem"
    elif pct_b >= pct_a and pct_b >= 0.25:
        reco = (
            "NEXT: asymmetric/wider trailing stop for winning positions — "
            "category (b) early ATR exits are a meaningful share of big-winner windows."
        )
        reco_side = "wider_stop"
    elif pct_a >= 0.25:
        reco = (
            "NEXT: entry-signal / eligibility focus — category (a) never-entered "
            "dominates big-winner windows (score timing or PIT/liquidity gates)."
        )
        reco_side = "entry_signal"
    else:
        reco = (
            "MIXED: neither (a) nor (b) clearly dominates; inspect Task 3/4 before choosing "
            "wider stops vs entry changes."
        )
        reco_side = "mixed"

    payload = {
        "repro": repro,
        "methodology": {
            "window": {"start": START, "end": END},
            "winner_def": (
                f"month-start in liquid top-{TOP_LIQUID_POOL}; "
                f"forward {WINDOW_BARS}d peak/start_close >= {MIN_MULTIPLE}; "
                "dedupe overlapping windows per ticker (keep highest multiple)"
            ),
            "capture_ratio": (
                "sum(max(trade final_return,0) over overlapping holdings) / (multiple-1)"
            ),
            "category_c_threshold": CAPTURE_HELD_THRESHOLD,
            "position_notional_for_opp_cost": POSITION_NOTIONAL,
            "journal": str(journal_path),
            "journal_sha256_16": journal_sha,
            "atr_multiplier_baseline": ATR_MULTIPLIER,
        },
        "task1": {
            "n_events": len(raw_events),
            "n_unique_tickers": n_tickers,
            "multiple_buckets": dict(bucket_counts),
            "all_tradable_at_start": True,
        },
        "task2": {
            "n_events": len(classified),
            "counts": dict(cat_counts),
            "pct": {
                "a_never_entered": pct_a,
                "b_exited_early": pct_b,
                "c_held_most": pct_c,
                "d_churned": pct_d,
            },
            "opportunity_cost_dollars": {
                "a_never_entered": opp_a,
                "b_exited_early_residual": opp_b,
                "a_plus_b": opp_a + opp_b,
                "d_churn_residual_not_in_ab": opp_d_residual,
                "assumed_notional_per_name": POSITION_NOTIONAL,
            },
            "events": [
                {
                    "symbol": e["symbol"],
                    "window_start": str(e["window_start"].date()),
                    "window_end": str(e["window_end"].date()),
                    "multiple": e["multiple"],
                    "bucket": e["bucket"],
                    "category": e["category"],
                    "capture_ratio": e["capture_ratio"],
                    "n_round_trips": e["n_round_trips"],
                }
                for e in classified
            ],
        },
        "task3": task3,
        "task4": {
            "n_a_total": len(events_a),
            "n_sampled": len(task4_rows),
            "reason_counts": dict(reason_counts),
            "sample": task4_rows,
        },
        "recommendation": {"side": reco_side, "text": reco},
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha = hashlib.sha256(raw).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha

    def pct(x):
        return f"{100.0 * float(x):.1f}%"

    def num(x, d=2):
        if x is None:
            return "n/a"
        return f"{float(x):.{d}f}"

    lines = [
        "# Big-winner capture diagnose — mom+quality default (VALID)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        f"journal={journal_path} sha16={journal_sha}",
        "",
        f"Winner def: liquid top-{TOP_LIQUID_POOL} at month-start; "
        f"forward {WINDOW_BARS}d peak/start ≥ {MIN_MULTIPLE}x; "
        "overlap-deduped per ticker.",
        f"Capture ratio = sum(max(trade ret,0)) / (multiple−1); "
        f"(c) if capture ≥ {CAPTURE_HELD_THRESHOLD:.0%}.",
        f"Opp-cost notional ≈ ${POSITION_NOTIONAL:,.0f}/name (50M/30).",
        "",
        "## TASK 1 — Big winners in tradable universe",
        f"  n_events (deduped)     : {len(raw_events):,}",
        f"  n_unique_tickers       : {n_tickers:,}",
        f"  tradable_at_start      : 100% (liquid-filtered by construction)",
        "  Multiple distribution:",
    ]
    for b in ("1.5-2x", "2-3x", "3-5x", "5-10x", "10x+"):
        lines.append(f"    {b:<8} {bucket_counts.get(b, 0):>6}")

    lines += [
        "",
        "## TASK 2 — Cross-ref vs MQ trade journal (KEY)",
        f"  (a) NEVER ENTERED              : {cat_counts.get('a_never_entered', 0):>5}  ({pct(pct_a)})",
        f"  (b) ENTERED BUT EXITED EARLY   : {cat_counts.get('b_exited_early', 0):>5}  ({pct(pct_b)})",
        f"  (c) HELD THROUGH MOST (≥{CAPTURE_HELD_THRESHOLD:.0%}) : {cat_counts.get('c_held_most', 0):>5}  ({pct(pct_c)})",
        f"  (d) CHURNED (re-entered)       : {cat_counts.get('d_churned', 0):>5}  ({pct(pct_d)})",
        "",
        "  Opportunity cost estimate (a)+(b residual vs full window gain):",
        f"    (a) missed entries $         : ${opp_a:,.0f}",
        f"    (b) early-exit residual $    : ${opp_b:,.0f}",
        f"    (a)+(b) TOTAL $           : ${opp_a + opp_b:,.0f}",
        f"    (d) churn residual $ (excl.) : ${opp_d_residual:,.0f}",
        "",
        "## TASK 3 — Why early exit (b)",
    ]
    if events_b:
        lines += [
            f"  n_b                        : {len(events_b)}",
            f"  normal pullback (<2×ATR)   : {task3['n_normal_pullback']}  "
            f"({pct(task3['pct_normal_pullback_lt_2xATR'] or 0)})",
            f"  large giveback (≥2×ATR)    : {task3['n_large_giveback']}  "
            f"({pct(task3['pct_large_giveback_ge_2xATR'] or 0)})",
            "  ATR multiples entry→peak (b): "
            f"p50={num(task3['atr_mults_entry_to_peak']['p50'])} "
            f"p75={num(task3['atr_mults_entry_to_peak']['p75'])} "
            f"p90={num(task3['atr_mults_entry_to_peak']['p90'])} "
            f"max={num(task3['atr_mults_entry_to_peak']['max'])}",
            "  vol60 at entry: "
            f"b_mean={num(task3['vol60_at_entry']['b_mean'], 4)} "
            f"b_p50={num(task3['vol60_at_entry']['b_p50'], 4)} | "
            f"c_mean={num(task3['vol60_at_entry']['c_mean'], 4)} "
            f"c_p50={num(task3['vol60_at_entry']['c_p50'], 4)} | "
            f"b>c? {task3['vol60_at_entry']['b_higher_than_c']}",
        ]
    else:
        lines.append("  (b) empty — skip.")

    lines += ["", "## TASK 4 — Why never entered (a)"]
    if events_a:
        lines.append(
            f"  n_a={len(events_a)}; sampled={len(task4_rows)}; reasons={dict(reason_counts)}"
        )
        for r in task4_rows[:SAMPLE_A_MAX]:
            lines.append(
                f"    {r['symbol']:6} start={r['window_start']} mult={r['multiple']:.2f}x "
                f"reason={r['reason']} rank={r.get('combined_rank')} "
                f"liquid={r.get('in_liquid_topN')} pit_miss={r.get('missing_pit_quality')}"
            )
    else:
        lines.append("  (a) empty — skip.")

    lines += [
        "",
        "## Recommendation",
        f"  {reco}",
        f"  side={reco_side}",
        "",
        "# END",
        "",
    ]

    txt = out_dir / "bigwinner_diagnose_report.txt"
    js = out_dir / "bigwinner_diagnose_report.json"
    txt.write_text("\n".join(lines), encoding="utf-8")
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    print(f"Wrote -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
