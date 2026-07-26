#!/usr/bin/env python3
"""Diagnose whether rank-dropout churn is costly in mom+quality default.

Task 1 only (no hysteresis A/B). Runs the current momentum+quality global
default (no-chase, Fixed CS v2, no regime / industry-neutral / corr filter),
classifies closed exits into ATR-trail vs rank-dropout (rebalance_dropout),
and reports prior ranks, attributable round-trip costs, and 1–3 month flicker.

Must run on a clean git tree.

Usage:
  python3 run_hysteresis_churn_diagnose.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from engine.baseline_engine import BaselineEngineV1
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
from engine.topup_chase_diagnostics import kpi_with_trades

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent
TOP_K = int(TOP_MOMENTUM_COUNT)
MARGINAL_LO, MARGINAL_HI = 31, 35
CLEAR_WEAK_LO = 50


def _git_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
    )
    return bool(out.strip())


def _pct_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def _month_starts(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp):
    in_win = index[(index >= start) & (index <= end)]
    months = in_win.to_period("M")
    return pd.to_datetime(in_win[~months.duplicated(keep="first")])


def _exit_family(reason: str) -> str:
    r = (reason or "").lower()
    if "atr_trail" in r:
        return "atr_trail"
    if "rebalance_dropout" in r:
        return "rank_dropout"
    if "delist" in r:
        return "delist"
    if "rebalance_trim" in r:
        return "rebalance_trim"
    return "other"


def _rank_bucket(rank: Optional[float]) -> str:
    if rank is None or not np.isfinite(rank):
        return "unranked"
    r = int(rank)
    if r <= TOP_K:
        return f"in_top_{TOP_K}"
    if MARGINAL_LO <= r <= MARGINAL_HI:
        return f"marginal_{MARGINAL_LO}_{MARGINAL_HI}"
    if TOP_K < r < CLEAR_WEAK_LO:
        return f"moderate_{TOP_K + 1}_{CLEAR_WEAK_LO - 1}"
    return f"clear_weak_{CLEAR_WEAK_LO}_plus"


def _full_mom_quality_ranks(
    *,
    date_idx: int,
    close_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    mom_12_1_m: pd.DataFrame,
    eligible_symbols: Sequence[str],
    fundamental_history: Dict[str, pd.DataFrame],
    fund_ts: Dict[str, Any],
    top_liquid_pool: int = TOP_LIQUID_POOL,
) -> pd.Series:
    """1=best combined mom+quality rank over liquid+PIT-eligible names."""
    if date_idx < 252 or not eligible_symbols:
        return pd.Series(dtype=float)

    current_date = close_m.index[date_idx]
    syms = [s for s in eligible_symbols if s in close_m.columns and s in dvol_m.columns]
    if not syms:
        return pd.Series(dtype=float)

    dvol_row = pd.to_numeric(dvol_m.loc[current_date, syms], errors="coerce").dropna()
    if dvol_row.empty:
        return pd.Series(dtype=float)

    liquid = dvol_row.nlargest(min(int(top_liquid_pool), len(dvol_row))).index.tolist()
    mom_row = pd.to_numeric(mom_12_1_m.loc[current_date, liquid], errors="coerce").dropna()
    if mom_row.empty:
        return pd.Series(dtype=float)

    gp_vals: Dict[str, float] = {}
    roic_vals: Dict[str, float] = {}
    opm_vals: Dict[str, float] = {}
    for sym in mom_row.index.tolist():
        row = get_latest_available_fundamentals(
            fundamental_history, fund_ts, sym, current_date
        )
        q = extract_quality_raw(row)
        if q is None:
            continue
        gp_vals[sym] = q["gross_profitability"]
        roic_vals[sym] = q["roic"]
        opm_vals[sym] = q["op_margin"]

    if not gp_vals:
        return pd.Series(dtype=float)

    eligible = list(gp_vals.keys())
    mom = mom_row.reindex(eligible).dropna()
    if mom.empty:
        return pd.Series(dtype=float)

    gp = pd.Series(gp_vals).reindex(mom.index)
    roic = pd.Series(roic_vals).reindex(mom.index)
    opm = pd.Series(opm_vals).reindex(mom.index)
    quality = (_pct_rank(gp) + _pct_rank(roic) + _pct_rank(opm)) / 3.0
    combined = float(MOM_WEIGHT) * _pct_rank(mom) + float(QUALITY_WEIGHT) * quality
    ranked = combined.sort_values(ascending=False)
    # rank 1 = best
    return pd.Series(
        np.arange(1, len(ranked) + 1, dtype=float), index=ranked.index, dtype=float
    )


def _cost_lookup(events: List[dict]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(
            columns=["date", "symbol", "side", "cost_dollars", "order_role", "qty"]
        )
    df = pd.DataFrame(events)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str).str.upper()
    df["cost_dollars"] = pd.to_numeric(df["cost_dollars"], errors="coerce").fillna(0.0)
    df["order_role"] = df.get("order_role")
    return df


def _sum_cost(
    costs: pd.DataFrame,
    *,
    symbol: str,
    side: str,
    on_date: pd.Timestamp,
    order_role: Optional[str] = None,
    window_days: int = 0,
) -> float:
    if costs.empty:
        return 0.0
    d0 = pd.Timestamp(on_date).normalize()
    d1 = d0 + pd.Timedelta(days=int(window_days))
    m = (
        (costs["symbol"] == str(symbol))
        & (costs["side"] == side.upper())
        & (costs["date"] >= d0)
        & (costs["date"] <= d1)
    )
    if order_role is not None:
        m = m & (costs["order_role"].astype(str) == str(order_role))
    return float(costs.loc[m, "cost_dollars"].sum())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_HYSTERESIS_DIAGNOSE",
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
    cfg = dict(base_cfg)
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

    repro = build_repro_fingerprint(
        start=START,
        end=END,
        config=cfg,
        extra={
            "report": "hysteresis_churn_diagnose",
            "selection": "mom_quality_global_topK",
            "policy_note": "hold_until_ATR_refill_empties",
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')}")
    if repro.get("git_dirty"):
        return 1

    print("\n" + "#" * 72)
    print("# Mom+Quality default — churn diagnose backtest")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path=args.config)
    engine.run()
    metrics = kpi_with_trades(engine)

    # Closed trades from journal
    tj = engine.trade_journal.to_frame()
    if tj is None or tj.empty:
        closed = list(getattr(engine.trade_journal, "closed", []) or [])
        tj = pd.DataFrame([t.as_dict() for t in closed]) if closed else pd.DataFrame()
    if tj is None or tj.empty:
        cache_p = Path("cache/baseline_v1/trade_journal.csv")
        if cache_p.exists():
            tj = pd.read_csv(cache_p)
        else:
            print("ERROR: empty trade journal")
            return 1

    tj = tj.copy()
    tj["symbol"] = tj["symbol"].astype(str)
    tj["entry_date"] = pd.to_datetime(tj["entry_date"])
    tj["exit_date"] = pd.to_datetime(tj["exit_date"])
    tj["exit_family"] = tj["exit_reason"].astype(str).map(_exit_family)

    n_exits = int(len(tj))
    fam_counts = tj["exit_family"].value_counts().to_dict()
    n_atr = int(fam_counts.get("atr_trail", 0))
    n_drop = int(fam_counts.get("rank_dropout", 0))
    pct_drop = float(n_drop / n_exits) if n_exits else 0.0

    costs = _cost_lookup(list(getattr(engine, "diag_cost_events", []) or []))

    # Monthly full ranks + top-K sets (same scoring as engine selection)
    import pickle

    panels_path = Path(cfg.get("paths", {}).get("prices", "data/prices")) / "panels.pkl"
    universe_path = (
        Path(cfg.get("paths", {}).get("metadata", "data/metadata")) / "universe.pkl"
    )
    funds_dir = Path(cfg.get("paths", {}).get("fundamentals", "data/fundamentals"))
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
    mom_m = compute_mom_12_1_panel(close_m)
    investable = [
        s
        for s in close_m.columns
        if s != benchmark and not profile_meta.get(s, {}).get("isEtf", False)
    ]
    start_ts, end_ts = pd.Timestamp(START), pd.Timestamp(END)
    month_starts = list(_month_starts(close_m.index, start_ts, end_ts))

    month_rank: Dict[pd.Timestamp, pd.Series] = {}
    month_topk: Dict[pd.Timestamp, List[str]] = {}
    for dt in month_starts:
        idx = int(close_m.index.get_loc(dt))
        if idx < 252:
            continue
        live = [
            s
            for s in investable
            if s in close_m.columns and pd.notna(close_m[s].iloc[idx])
        ]
        ranks = _full_mom_quality_ranks(
            date_idx=idx,
            close_m=close_m,
            dvol_m=dvol_m,
            mom_12_1_m=mom_m,
            eligible_symbols=live,
            fundamental_history=hist,
            fund_ts=fund_ts,
        )
        month_rank[pd.Timestamp(dt)] = ranks
        month_topk[pd.Timestamp(dt)] = ranks.head(TOP_K).index.tolist() if len(ranks) else []

    month_starts_sorted = sorted(month_rank.keys())

    def _month_on_or_before(dt: pd.Timestamp) -> Optional[pd.Timestamp]:
        cands = [m for m in month_starts_sorted if m <= pd.Timestamp(dt).normalize()]
        return cands[-1] if cands else None

    def _prev_month(m: pd.Timestamp) -> Optional[pd.Timestamp]:
        cands = [x for x in month_starts_sorted if x < m]
        return cands[-1] if cands else None

    def _next_months(m: pd.Timestamp, n: int = 3) -> List[pd.Timestamp]:
        cands = [x for x in month_starts_sorted if x > m]
        return cands[:n]

    # Per-day new_entry buy costs (for replacement allocation)
    new_entry_by_date: Dict[pd.Timestamp, float] = {}
    if not costs.empty:
        ne = costs[(costs["side"] == "BUY") & (costs["order_role"].astype(str) == "new_entry")]
        for d, g in ne.groupby(ne["date"].dt.normalize()):
            new_entry_by_date[pd.Timestamp(d)] = float(g["cost_dollars"].sum())

    drop_rows = tj[tj["exit_family"] == "rank_dropout"].copy()
    atr_rows = tj[tj["exit_family"] == "atr_trail"].copy()

    # Count dropouts per exit date for pro-rata replacement allocation
    drop_per_day = (
        drop_rows["exit_date"].dt.normalize().value_counts().to_dict() if len(drop_rows) else {}
    )

    dropout_details: List[Dict[str, Any]] = []
    prior_ranks: List[float] = []
    bucket_counts: Dict[str, int] = {}
    drop_sell_cost = 0.0
    drop_repl_buy_cost = 0.0
    flicker_details: List[Dict[str, Any]] = []
    flicker_cost = 0.0

    for _, row in drop_rows.iterrows():
        sym = str(row["symbol"])
        exit_dt = pd.Timestamp(row["exit_date"])
        entry_dt = pd.Timestamp(row["entry_date"])
        m_exit = _month_on_or_before(exit_dt)
        m_prev = _prev_month(m_exit) if m_exit is not None else None
        rank_exit = None
        rank_prev = None
        if m_exit is not None and m_exit in month_rank and sym in month_rank[m_exit].index:
            rank_exit = float(month_rank[m_exit].loc[sym])
        if m_prev is not None and m_prev in month_rank and sym in month_rank[m_prev].index:
            rank_prev = float(month_rank[m_prev].loc[sym])
            prior_ranks.append(rank_prev)
        bucket = _rank_bucket(rank_prev)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        sell_c = _sum_cost(costs, symbol=sym, side="SELL", on_date=exit_dt, window_days=1)
        day_key = exit_dt.normalize()
        n_same_day = int(drop_per_day.get(day_key, 1)) or 1
        repl_pool = float(new_entry_by_date.get(day_key, 0.0))
        repl_c = repl_pool / float(n_same_day)
        # also try +1 session if sell fills next open after month-start signal
        if repl_c == 0.0:
            for off in (1, 2, 3):
                alt = day_key + pd.Timedelta(days=off)
                if alt in new_entry_by_date:
                    repl_c = float(new_entry_by_date[alt]) / float(n_same_day)
                    break

        rt = float(sell_c + repl_c)
        drop_sell_cost += float(sell_c)
        drop_repl_buy_cost += float(repl_c)

        # Flicker: reappears in top-K within next 1–3 month-starts after exit month
        flicker_months: List[str] = []
        if m_exit is not None:
            for nm in _next_months(m_exit, 3):
                if sym in set(month_topk.get(nm, [])):
                    flicker_months.append(str(nm.date()))

        detail = {
            "symbol": sym,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "holding_days": int(row["holding_days"]) if pd.notna(row["holding_days"]) else None,
            "final_return": float(row["final_return"]) if pd.notna(row["final_return"]) else None,
            "rank_month_before_exit": rank_prev,
            "rank_at_exit_month": rank_exit,
            "prior_rank_bucket": bucket,
            "sell_cost_dollars": float(sell_c),
            "replacement_buy_cost_dollars": float(repl_c),
            "round_trip_churn_cost_dollars": rt,
            "flicker_reselected_months": flicker_months,
            "is_flicker_1_3m": bool(flicker_months),
        }
        dropout_details.append(detail)
        if flicker_months:
            flicker_details.append(detail)
            flicker_cost += rt

    # ATR category costs (same definition: sell + same-day/next new_entry pro-rata)
    atr_per_day = (
        atr_rows["exit_date"].dt.normalize().value_counts().to_dict() if len(atr_rows) else {}
    )
    atr_sell_cost = 0.0
    atr_repl_buy_cost = 0.0
    for _, row in atr_rows.iterrows():
        sym = str(row["symbol"])
        exit_dt = pd.Timestamp(row["exit_date"])
        sell_c = _sum_cost(costs, symbol=sym, side="SELL", on_date=exit_dt, window_days=1)
        day_key = exit_dt.normalize()
        n_same = int(atr_per_day.get(day_key, 1)) or 1
        # ATR exits free slots; refill new_entry may land same or next sessions.
        # Attribute only new_entry costs on exit day pro-rata among ATR exits that day
        # (conservative; mid-month refill is the ATR→replacement path).
        repl_c = float(new_entry_by_date.get(day_key, 0.0)) / float(n_same)
        if repl_c == 0.0:
            for off in (1, 2, 3):
                alt = day_key + pd.Timedelta(days=off)
                if alt in new_entry_by_date:
                    # if multiple ATR exits share nearby refill, still split by exit-day count
                    repl_c = float(new_entry_by_date[alt]) / float(n_same)
                    break
        atr_sell_cost += float(sell_c)
        atr_repl_buy_cost += float(repl_c)

    # Prior-rank distribution summary for dropouts
    prior_dist: Dict[str, Any] = {
        "n_with_prior_rank": len(prior_ranks),
        "p50": float(np.median(prior_ranks)) if prior_ranks else None,
        "p75": float(np.percentile(prior_ranks, 75)) if prior_ranks else None,
        "min": float(np.min(prior_ranks)) if prior_ranks else None,
        "max": float(np.max(prior_ranks)) if prior_ranks else None,
        "mean": float(np.mean(prior_ranks)) if prior_ranks else None,
        "n_marginal_31_35": int(bucket_counts.get(f"marginal_{MARGINAL_LO}_{MARGINAL_HI}", 0)),
        "n_clear_weak_50_plus": int(bucket_counts.get(f"clear_weak_{CLEAR_WEAK_LO}_plus", 0)),
        "bucket_counts": bucket_counts,
    }

    total_cost = float(getattr(engine, "diag_total_cost_dollars", 0.0) or 0.0)
    drop_rt = float(drop_sell_cost + drop_repl_buy_cost)
    atr_rt = float(atr_sell_cost + atr_repl_buy_cost)

    # Verdict: hysteresis targets rank-boundary flicker. Under hold-until-ATR,
    # rank-dropout exits are rare by design — if pct_drop is tiny, skip A/B.
    if pct_drop < 0.05 and n_drop < 20:
        verdict_summary = (
            "LOW: current default holds until ATR and almost never sells on "
            "rank-dropout; rank-buffer hysteresis would address a negligible "
            "exit path (like industry-neutral / corr filter — non-problem)."
        )
        recommend_ab = False
        verdict_level = "LOW"
    elif (
        prior_dist.get("n_marginal_31_35", 0) >= max(3, n_drop // 3)
        and len(flicker_details) >= max(2, n_drop // 4)
    ):
        verdict_summary = (
            "MODERATE/HIGH: meaningful share of exits are rank-dropouts near the "
            "Top-K boundary with flicker re-selection — hysteresis A/B may help."
        )
        recommend_ab = True
        verdict_level = "HIGH"
    else:
        verdict_summary = (
            "LOW/UNCERTAIN: rank-dropout exits exist but look more like genuine "
            "signal deterioration than boundary flicker — do not rush hysteresis A/B."
        )
        recommend_ab = False
        verdict_level = "LOW"

    payload = {
        "repro": repro,
        "methodology": {
            "window": {"start": START, "end": END},
            "selection": "mom_quality_global_topK",
            "flags": {
                "enable_quality_factor": True,
                "enable_topup_chasing": False,
                "enable_regime_exposure": False,
                "enable_industry_neutral_ranking": False,
            },
            "exit_policy": (
                "hold until ATR trail; refill empties from monthly candidates; "
                "does NOT drop holdings merely for leaving Top-K "
                "(refill_from_candidates). rank_dropout == exit_reason "
                "rebalance_dropout (SELL while not in that day's targets)."
            ),
            "rank_definition": (
                "combined = 0.5*mom_pctile + 0.5*quality_pctile over liquid "
                f"top-{TOP_LIQUID_POOL} with usable PIT; rank 1 = best; "
                f"Top-K = {TOP_K}."
            ),
            "cost_attribution": (
                "category round-trip ≈ sell cost_dollars of exit + pro-rata "
                "new_entry buy cost_dollars on/near exit date among same-family "
                "exits that day (replacement leg)."
            ),
            "flicker": "rank-dropout name reappears in monthly Top-K within next 1–3 month-starts",
            "marginal_band": [MARGINAL_LO, MARGINAL_HI],
            "clear_weak_from": CLEAR_WEAK_LO,
        },
        "run_kpis": {
            "CAGR": metrics.get("CAGR"),
            "Sharpe": metrics.get("Sharpe"),
            "Maximum_Drawdown": metrics.get("Maximum_Drawdown"),
            "total_cost_dollars": metrics.get("total_cost_dollars", total_cost),
            "n_closed_trades": metrics.get("n_closed_trades", n_exits),
        },
        "exit_mix": {
            "n_exits": n_exits,
            "by_family": fam_counts,
            "n_atr_trail": n_atr,
            "n_rank_dropout": n_drop,
            "pct_rank_dropout": pct_drop,
            "pct_atr_trail": float(n_atr / n_exits) if n_exits else 0.0,
        },
        "rank_dropout": {
            "prior_rank_distribution": prior_dist,
            "details": dropout_details,
            "costs": {
                "sell_cost_dollars": drop_sell_cost,
                "replacement_buy_cost_dollars": drop_repl_buy_cost,
                "round_trip_churn_cost_dollars": drop_rt,
                "share_of_total_cost": (drop_rt / total_cost) if total_cost > 0 else None,
            },
            "flicker_1_3m": {
                "n": len(flicker_details),
                "pct_of_rank_dropouts": (len(flicker_details) / n_drop) if n_drop else 0.0,
                "round_trip_churn_cost_dollars": flicker_cost,
                "details": flicker_details,
            },
        },
        "atr_trail_costs": {
            "sell_cost_dollars": atr_sell_cost,
            "replacement_buy_cost_dollars": atr_repl_buy_cost,
            "round_trip_churn_cost_dollars": atr_rt,
            "share_of_total_cost": (atr_rt / total_cost) if total_cost > 0 else None,
        },
        "total_diag_cost_dollars": total_cost,
        "verdict": {
            "level": verdict_level,
            "summary": verdict_summary,
            "recommend_hysteresis_ab": recommend_ab,
        },
    }

    raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
    payload_sha = hashlib.sha256(raw).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha

    # Text report
    lines = [
        "# Turnover hysteresis diagnose — mom+quality default (VALID)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        "",
        "Policy: hold until ATR; refill empties; no rank-forced exit by design",
        f"(refill_from_candidates). Top-K={TOP_K}. Costs: Fixed CS v2.",
        "",
        "## Exit mix",
        f"  n_exits                : {n_exits:,}",
        f"  ATR-trail (a)          : {n_atr:,}  ({100.0 * n_atr / n_exits:.2f}%)" if n_exits else "  ATR-trail (a)          : 0",
        f"  rank-dropout (b)       : {n_drop:,}  ({100.0 * pct_drop:.2f}%)",
        f"  other families         : { {k: v for k, v in fam_counts.items() if k not in ('atr_trail', 'rank_dropout')} }",
        "",
        "## Rank-dropout (b) — prior-month combined rank",
        f"  n_with_prior_rank      : {prior_dist['n_with_prior_rank']}",
        f"  prior rank p50/p75/max : "
        + (
            f"{prior_dist['p50']:.1f} / {prior_dist['p75']:.1f} / {prior_dist['max']:.1f}"
            if prior_dist["p50"] is not None
            else "n/a"
        ),
        f"  marginal ranks {MARGINAL_LO}-{MARGINAL_HI}: {prior_dist['n_marginal_31_35']}",
        f"  clear-weak rank ≥{CLEAR_WEAK_LO}  : {prior_dist['n_clear_weak_50_plus']}",
        f"  bucket_counts          : {bucket_counts}",
        "",
        "## Cost attribution (sell + replacement new_entry buy)",
        f"  total diag cost $      : ${total_cost:,.0f}",
        f"  (b) rank-dropout RT $  : ${drop_rt:,.0f}"
        + (
            f"  ({100.0 * drop_rt / total_cost:.2f}% of total)"
            if total_cost > 0
            else ""
        ),
        f"      sell $             : ${drop_sell_cost:,.0f}",
        f"      replacement buy $  : ${drop_repl_buy_cost:,.0f}",
        f"  (a) ATR-trail RT $     : ${atr_rt:,.0f}"
        + (
            f"  ({100.0 * atr_rt / total_cost:.2f}% of total)"
            if total_cost > 0
            else ""
        ),
        f"      sell $             : ${atr_sell_cost:,.0f}",
        f"      replacement buy $  : ${atr_repl_buy_cost:,.0f}",
        "",
        "## Flicker (rank-dropout re-selected in Top-K within 1–3 months)",
        f"  n_flicker              : {len(flicker_details)}"
        + (f"  ({100.0 * len(flicker_details) / n_drop:.1f}% of dropouts)" if n_drop else ""),
        f"  flicker RT cost $      : ${flicker_cost:,.0f}",
        "",
    ]
    if dropout_details:
        lines.append("  Dropout detail rows:")
        for d in dropout_details:
            lines.append(
                f"    {d['symbol']:6} exit={d['exit_date']} "
                f"prior_rank={d['rank_month_before_exit']} "
                f"exit_rank={d['rank_at_exit_month']} "
                f"bucket={d['prior_rank_bucket']} "
                f"RT=${d['round_trip_churn_cost_dollars']:,.0f} "
                f"flicker={d['flicker_reselected_months']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Verdict (diagnose-first)",
            f"  {verdict_level}: {verdict_summary}",
            f"  recommend_hysteresis_ab : {recommend_ab}",
            "",
            "# END",
            "",
        ]
    )

    txt_path = out_dir / "hysteresis_churn_diagnose_report.txt"
    json_path = out_dir / "hysteresis_churn_diagnose_report.json"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"[WRITE] {txt_path}")
    print(f"[WRITE] {json_path}")
    print(f"[KPI] CAGR={metrics.get('CAGR')} Sharpe={metrics.get('Sharpe')} "
          f"cost=${metrics.get('total_cost_dollars', total_cost):,.0f} "
          f"n_trades={metrics.get('n_closed_trades', n_exits)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
