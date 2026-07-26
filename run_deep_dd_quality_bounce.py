#!/usr/bin/env python3
"""Diagnose: deep 3m drawdown + decent quality → 12m bounce vs further crash.

Cohort: liquid top-N + usable PIT quality, ret_3m ≤ -50%, quality_pctile ≥ threshold.
Outcomes over next ~252 trading days from evaluation close:
  bounce:  forward peak/start − 1 ≥ +50%   (OR end return ≥ +50% — use peak for
            consistency with big-winner scan; also report end-to-end)
  crash:   forward trough/start − 1 ≤ -50%
  neither / both (path can touch both)

Primary metrics: win rate (bounce among resolved), payoff (mean bounce gain /
mean |crash loss|), and bounce-vs-crash counts.

Must run on clean git.

Usage:
  python3 run_deep_dd_quality_bounce.py --config config/config.yaml
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

from engine.pit_fundamentals import (
    build_fund_ts_index,
    extract_quality_raw,
    get_latest_available_fundamentals,
    load_pit_history,
)
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
from engine.strategy_baseline_v1 import TOP_LIQUID_POOL

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent
LOOKBACK_3M = 63
FORWARD_12M = 252
DD_3M_MAX = -0.50  # ret_3m ≤ -50%
BOUNCE_MIN = 0.50  # +50%
CRASH_MAX = -0.50  # -50% further
QUALITY_PCTILE_MIN = 0.50  # above-median quality in liquid+PIT cross-section


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


def _quality_universe(
    *,
    date_idx: int,
    close_m: pd.DataFrame,
    dvol_m: pd.DataFrame,
    hist,
    fund_ts,
    investable: List[str],
) -> Optional[pd.DataFrame]:
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

    gp_vals, roic_vals, opm_vals = {}, {}, {}
    for sym in liquid:
        row = get_latest_available_fundamentals(hist, fund_ts, sym, current_date)
        q = extract_quality_raw(row)
        if q is None:
            continue
        gp_vals[sym] = q["gross_profitability"]
        roic_vals[sym] = q["roic"]
        opm_vals[sym] = q["op_margin"]
    if not gp_vals:
        return None

    idx = list(gp_vals.keys())
    gp = pd.Series(gp_vals)
    roic = pd.Series(roic_vals)
    opm = pd.Series(opm_vals)
    q_pct = (_pct_rank(gp) + _pct_rank(roic) + _pct_rank(opm)) / 3.0
    ret_3m = close_m.pct_change(LOOKBACK_3M).loc[current_date].reindex(idx)
    px = close_m.loc[current_date].reindex(idx)
    return pd.DataFrame(
        {
            "gp_raw": gp,
            "roic_raw": roic,
            "opm_raw": opm,
            "quality_pctile": q_pct,
            "ret_3m": ret_3m,
            "price": px,
        }
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_DEEP_DD_QUALITY_BOUNCE",
    )
    parser.add_argument("--quality-pctile-min", type=float, default=QUALITY_PCTILE_MIN)
    parser.add_argument("--dd-3m-max", type=float, default=DD_3M_MAX)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: working tree dirty — commit first.")
        return 1

    q_min = float(args.quality_pctile_min)
    dd_max = float(args.dd_3m_max)

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    import pickle

    with open(Path(paths.get("prices", "data/prices")) / "panels.pkl", "rb") as f:
        panels = pickle.load(f)
    with open(Path(paths.get("metadata", "data/metadata")) / "universe.pkl", "rb") as f:
        universe = pickle.load(f)

    close_m: pd.DataFrame = panels["close_m"]
    dvol_m: pd.DataFrame = panels["dvol_m"]
    profile_meta = universe["profile_meta"]
    benchmark = str(cfg.get("benchmark", "QQQ")).upper()
    investable = [
        s
        for s in close_m.columns
        if s != benchmark and not profile_meta.get(s, {}).get("isEtf", False)
    ]
    hist = load_pit_history(Path(paths.get("fundamentals", "data/fundamentals")))
    fund_ts = build_fund_ts_index(hist)

    start, end = pd.Timestamp(START), pd.Timestamp(END)
    month_starts = _month_starts(close_m.index, start, end)

    repro = build_repro_fingerprint(
        start=START,
        end=END,
        config={**cfg, "enable_quality_factor": True},
        extra={
            "report": "deep_dd_quality_bounce",
            "dd_3m_max": dd_max,
            "quality_pctile_min": q_min,
            "bounce_min": BOUNCE_MIN,
            "crash_max": CRASH_MAX,
            "forward_bars": FORWARD_12M,
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    if repro.get("git_dirty"):
        return 1

    rows: List[Dict[str, Any]] = []
    n_screened = 0

    for dt in month_starts:
        idx = int(close_m.index.get_loc(dt))
        if idx < LOOKBACK_3M:
            continue
        i1 = idx + FORWARD_12M
        if i1 >= len(close_m.index):
            continue  # need full forward window for fair rates

        uni = _quality_universe(
            date_idx=idx,
            close_m=close_m,
            dvol_m=dvol_m,
            hist=hist,
            fund_ts=fund_ts,
            investable=investable,
        )
        if uni is None or uni.empty:
            continue

        cohort = uni[(uni["ret_3m"] <= dd_max) & (uni["quality_pctile"] >= q_min)].copy()
        n_screened += int(len(uni))
        if cohort.empty:
            continue

        for sym, r in cohort.iterrows():
            px0 = float(r["price"])
            if not np.isfinite(px0) or px0 <= 0:
                continue
            seg = close_m[sym].iloc[idx : i1 + 1]
            if seg.notna().sum() < max(60, FORWARD_12M // 3):
                continue
            peak = float(seg.max())
            trough = float(seg.min())
            end_px = float(seg.iloc[-1]) if pd.notna(seg.iloc[-1]) else np.nan
            fwd_peak_ret = peak / px0 - 1.0
            fwd_trough_ret = trough / px0 - 1.0
            fwd_end_ret = end_px / px0 - 1.0 if np.isfinite(end_px) and end_px > 0 else np.nan

            bounced = bool(fwd_peak_ret >= BOUNCE_MIN)
            crashed = bool(fwd_trough_ret <= CRASH_MAX)
            # Path labels
            if bounced and crashed:
                path = "both_bounce_and_crash"
            elif bounced:
                path = "bounce_only"
            elif crashed:
                path = "crash_only"
            else:
                path = "neither"

            rows.append(
                {
                    "symbol": sym,
                    "eval_date": str(pd.Timestamp(dt).date()),
                    "ret_3m": float(r["ret_3m"]),
                    "quality_pctile": float(r["quality_pctile"]),
                    "gp_raw": float(r["gp_raw"]),
                    "roic_raw": float(r["roic_raw"]),
                    "opm_raw": float(r["opm_raw"]),
                    "fwd_peak_ret": float(fwd_peak_ret),
                    "fwd_trough_ret": float(fwd_trough_ret),
                    "fwd_end_ret": float(fwd_end_ret) if np.isfinite(fwd_end_ret) else None,
                    "bounced_50": bounced,
                    "crashed_50": crashed,
                    "path": path,
                }
            )

    df = pd.DataFrame(rows)
    n = len(df)
    if n == 0:
        print("ERROR: empty cohort")
        return 1

    n_bounce = int(df["bounced_50"].sum())
    n_crash = int(df["crashed_50"].sum())
    n_bounce_only = int((df["path"] == "bounce_only").sum())
    n_crash_only = int((df["path"] == "crash_only").sum())
    n_both = int((df["path"] == "both_bounce_and_crash").sum())
    n_neither = int((df["path"] == "neither").sum())

    # Win rate: among names that hit bounce OR crash (exclusive preference:
    # bounce_only vs crash_only). Also report bounce rate and crash rate
    # against full cohort.
    resolved = n_bounce_only + n_crash_only
    win_rate_exclusive = (n_bounce_only / resolved) if resolved else None
    bounce_rate = n_bounce / n
    crash_rate = n_crash / n

    # Payoff: mean peak gain among bounce_only (or all bounced) vs
    # mean |trough loss| among crash_only (or all crashed)
    bounce_gains = df.loc[df["path"] == "bounce_only", "fwd_peak_ret"]
    crash_losses = df.loc[df["path"] == "crash_only", "fwd_trough_ret"].abs()
    # Fallback if exclusive sets thin: use all bounced / all crashed
    if len(bounce_gains) < 5:
        bounce_gains = df.loc[df["bounced_50"], "fwd_peak_ret"]
    if len(crash_losses) < 5:
        crash_losses = df.loc[df["crashed_50"], "fwd_trough_ret"].abs()

    mean_win = float(bounce_gains.mean()) if len(bounce_gains) else None
    mean_loss = float(crash_losses.mean()) if len(crash_losses) else None
    payoff = (
        float(mean_win / mean_loss)
        if mean_win is not None and mean_loss is not None and mean_loss > 1e-12
        else None
    )

    # End-to-end (close at +12m) as secondary
    end_rets = df["fwd_end_ret"].dropna()
    end_bounce = int((end_rets >= BOUNCE_MIN).sum())
    end_crash = int((end_rets <= CRASH_MAX).sum())

    # Deduped unique tickers
    n_tickers = int(df["symbol"].nunique())

    payload = {
        "repro": repro,
        "methodology": {
            "universe": f"liquid top-{TOP_LIQUID_POOL} + PIT quality (GP/ROIC/op_margin)",
            "entry_screen": (
                f"month-start; ret_3m(63d) ≤ {dd_max}; quality_pctile ≥ {q_min}"
            ),
            "forward": f"{FORWARD_12M} trading days; need full window",
            "bounce": f"max(close[t:t+12m])/close[t] − 1 ≥ {BOUNCE_MIN}",
            "crash": f"min(close[t:t+12m])/close[t] − 1 ≤ {CRASH_MAX}",
            "win_rate_exclusive": "bounce_only / (bounce_only + crash_only)",
            "payoff": "mean(fwd_peak_ret | bounce_only) / mean(|fwd_trough_ret| | crash_only)",
            "note": (
                "path can hit both peak+50% and trough-50% in same 12m; "
                "exclusive sets used for win/payoff; rates vs full cohort also reported"
            ),
        },
        "cohort": {
            "n_events": n,
            "n_unique_tickers": n_tickers,
            "n_universe_screens_sum": n_screened,
            "quality_pctile_min": q_min,
            "dd_3m_max": dd_max,
            "ret_3m": {
                "p50": float(df["ret_3m"].median()),
                "mean": float(df["ret_3m"].mean()),
                "min": float(df["ret_3m"].min()),
            },
            "quality_pctile": {
                "p50": float(df["quality_pctile"].median()),
                "mean": float(df["quality_pctile"].mean()),
            },
        },
        "outcomes": {
            "n_bounced_peak50": n_bounce,
            "n_crashed_trough50": n_crash,
            "bounce_rate": bounce_rate,
            "crash_rate": crash_rate,
            "path_counts": {
                "bounce_only": n_bounce_only,
                "crash_only": n_crash_only,
                "both": n_both,
                "neither": n_neither,
            },
            "win_rate_exclusive": win_rate_exclusive,
            "mean_bounce_gain": mean_win,
            "mean_crash_loss_abs": mean_loss,
            "payoff_ratio": payoff,
            "end_to_end_12m": {
                "n_end_ge_plus50": end_bounce,
                "n_end_le_minus50": end_crash,
                "end_bounce_rate": float(end_bounce / len(end_rets)) if len(end_rets) else None,
                "end_crash_rate": float(end_crash / len(end_rets)) if len(end_rets) else None,
                "end_ret_p50": float(end_rets.median()) if len(end_rets) else None,
            },
        },
        "sample_rows": df.sort_values("fwd_peak_ret", ascending=False)
        .head(30)
        .to_dict(orient="records"),
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha = hashlib.sha256(raw).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha

    def pct(x):
        if x is None:
            return "n/a"
        return f"{100 * float(x):.1f}%"

    def num(x, d=3):
        if x is None:
            return "n/a"
        return f"{float(x):.{d}f}"

    lines = [
        "# Deep 3m drawdown + quality → 12m bounce vs crash (VALID)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        "",
        f"Screen: liquid top-{TOP_LIQUID_POOL} + PIT quality; "
        f"ret_3m≤{dd_max:.0%}; quality_pctile≥{q_min:.0%}.",
        f"Forward {FORWARD_12M}d: bounce = peak≥+{BOUNCE_MIN:.0%}; "
        f"crash = trough≤{CRASH_MAX:.0%}.",
        "",
        "## Cohort",
        f"  n_events              : {n:,}",
        f"  n_unique_tickers      : {n_tickers:,}",
        f"  ret_3m p50/mean/min   : {num(df['ret_3m'].median())} / {num(df['ret_3m'].mean())} / {num(df['ret_3m'].min())}",
        f"  quality_pctile p50    : {num(df['quality_pctile'].median())}",
        "",
        "## KEY — win rate & payoff",
        f"  bounce_only            : {n_bounce_only:,}",
        f"  crash_only             : {n_crash_only:,}",
        f"  both (path hit both)   : {n_both:,}",
        f"  neither                : {n_neither:,}",
        f"  WIN RATE (exclusive)   : {pct(win_rate_exclusive)}  "
        f"[= bounce_only / (bounce_only+crash_only)]",
        f"  mean bounce gain       : {pct(mean_win)}",
        f"  mean |crash| loss      : {pct(mean_loss)}",
        f"  PAYOFF RATIO           : {num(payoff, 2)}  [= mean_win / mean_|loss|]",
        "",
        "## Rates vs full cohort (peak/trough path)",
        f"  any bounce (≥+50% peak): {n_bounce:,}  ({pct(bounce_rate)})",
        f"  any crash (≤−50% trough): {n_crash:,}  ({pct(crash_rate)})",
        "",
        "## Secondary — end-to-end +12m close",
        f"  end ≥ +50%             : {end_bounce:,}  ({pct(payload['outcomes']['end_to_end_12m']['end_bounce_rate'])})",
        f"  end ≤ −50%             : {end_crash:,}  ({pct(payload['outcomes']['end_to_end_12m']['end_crash_rate'])})",
        f"  end ret p50            : {pct(payload['outcomes']['end_to_end_12m']['end_ret_p50'])}",
        "",
        "## Verdict (diagnose-first)",
    ]
    if win_rate_exclusive is not None and payoff is not None:
        if win_rate_exclusive >= 0.55 and payoff >= 1.0:
            verdict = (
                "SUPPORTIVE: exclusive bounce win-rate and payoff look favorable — "
                "a deep-DD + quality mean-reversion sleeve may be worth an isolated A/B."
            )
        elif win_rate_exclusive < 0.45 or payoff < 0.8:
            verdict = (
                "UNFAVORABLE: bounce does not dominate crash on win-rate/payoff — "
                "do NOT rush a catch-the-knife entry on this cohort."
            )
        else:
            verdict = (
                "MIXED: neither clearly supportive nor dismissive — inspect path=both "
                "rate and end-to-end stats before any strategy change."
            )
    else:
        verdict = "INSUFFICIENT DATA"
    lines += [f"  {verdict}", "", "# END", ""]

    payload["verdict"] = verdict
    txt = out_dir / "deep_dd_quality_bounce_report.txt"
    js = out_dir / "deep_dd_quality_bounce_report.json"
    txt.write_text("\n".join(lines), encoding="utf-8")
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
