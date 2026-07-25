"""Execution cost-model audit (read-only reporting).

Separable from strategy_baseline_v1. Uses engine.diag_cost_events / panels to
sanity-check Corwin–Schultz + sqrt-impact calibration vs flat-cost benchmarks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PARTICIPATION_FLAG = 0.20
COST_RATIO_RT_EQUIV_FLAG = 0.05  # flag if 2 * one-way cost_ratio > 5%
COST_RATIO_ONE_WAY_EXTREME = 0.05
ADV_SPIKE_RATIO = 10.0
SIZING_ADV_WARN = 0.10


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def cost_events_frame(engine) -> pd.DataFrame:
    rows = list(getattr(engine, "diag_cost_events", []) or [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for col in (
        "qty",
        "price",
        "notional",
        "cost_ratio",
        "cost_dollars",
        "adv",
        "adv_raw",
        "sigma",
        "cs_spread",
        "half_spread",
        "participation",
        "impact",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def worst_cost_fills(engine, n: int = 20) -> pd.DataFrame:
    """Task 1 — highest single-trade cost fills by $ cost."""
    df = cost_events_frame(engine)
    if df.empty:
        return df
    out = df.sort_values("cost_dollars", ascending=False).head(n).copy()
    out["rt_equiv_cost_ratio"] = 2.0 * out["cost_ratio"]
    out["flag_participation_gt_20pct"] = out["participation"] > PARTICIPATION_FLAG
    out["flag_rt_equiv_gt_5pct"] = out["rt_equiv_cost_ratio"] > COST_RATIO_RT_EQUIV_FLAG
    out["flag_one_way_gt_5pct"] = out["cost_ratio"] > COST_RATIO_ONE_WAY_EXTREME
    return out.reset_index(drop=True)


def adv_window_series(
    engine,
    symbol: str,
    trade_date,
    days_before: int = 10,
    days_after: int = 10,
) -> pd.DataFrame:
    """Task 3 — raw dollar-volume around trade date (calendar ±N trading days)."""
    dvol = getattr(engine, "dvol_m", None)
    if dvol is None or symbol not in dvol.columns:
        return pd.DataFrame()
    td = pd.Timestamp(trade_date)
    if td not in dvol.index:
        # nearest prior session
        prior = dvol.index[dvol.index <= td]
        if len(prior) == 0:
            return pd.DataFrame()
        td = prior[-1]
    loc = dvol.index.get_loc(td)
    if isinstance(loc, slice):
        loc = loc.start
    lo = max(0, int(loc) - days_before)
    hi = min(len(dvol.index) - 1, int(loc) + days_after)
    window = dvol[symbol].iloc[lo : hi + 1]
    df = pd.DataFrame(
        {
            "date": window.index,
            "symbol": symbol,
            "dollar_volume": pd.to_numeric(window.values, errors="coerce"),
            "is_trade_date": window.index == td,
        }
    )
    # Spike vs surrounding (exclude self)
    vals = df["dollar_volume"].to_numpy(dtype=float)
    flags = []
    for i, v in enumerate(vals):
        if not np.isfinite(v) or v <= 0:
            flags.append(False)
            continue
        others = [vals[j] for j in range(len(vals)) if j != i and np.isfinite(vals[j]) and vals[j] > 0]
        if not others:
            flags.append(False)
            continue
        med = float(np.median(others))
        flags.append(bool(med > 0 and v >= ADV_SPIKE_RATIO * med))
    df["flag_volume_spike_10x"] = flags
    return df


def entry_notional_over_adv_distribution(engine) -> Dict[str, Any]:
    """Task 4 — (entry notional) / ADV20 at entry across BUY fills."""
    df = cost_events_frame(engine)
    if df.empty:
        return {"n": 0}
    buys = df[df["side"].astype(str).str.upper() == "BUY"].copy()
    if buys.empty:
        return {"n": 0}
    # Prefer recorded participation (= notional/ADV used by cost model)
    ratio = buys["participation"].astype(float)
    ratio = ratio[np.isfinite(ratio) & (ratio >= 0)]
    if ratio.empty:
        return {"n": 0}
    pcts = {
        "p50": float(ratio.quantile(0.50)),
        "p75": float(ratio.quantile(0.75)),
        "p90": float(ratio.quantile(0.90)),
        "p95": float(ratio.quantile(0.95)),
        "max": float(ratio.max()),
        "mean": float(ratio.mean()),
    }
    n = int(len(ratio))
    return {
        "n": n,
        "pct_gt_10pct_adv": float((ratio > 0.10).mean()),
        "pct_gt_15pct_adv": float((ratio > 0.15).mean()),
        "pct_gt_participation_cap_buy": float(
            (ratio > float(getattr(engine, "PARTICIPATION_CAP_BUY", 0.08))).mean()
        ),
        **pcts,
    }


def kpi_slice(engine) -> Dict[str, Any]:
    arts = getattr(engine, "research_artifacts", {}) or {}
    comp = arts.get("comparison") or {}
    strat = comp.get("strategy") or {}
    return {
        "CAGR": strat.get("CAGR"),
        "Sharpe": strat.get("Sharpe"),
        "Maximum_Drawdown": strat.get("Maximum_Drawdown"),
        "Alpha_CAGR": comp.get("Alpha_CAGR"),
        "total_cost_dollars": float(getattr(engine, "diag_total_cost_dollars", 0.0) or 0.0),
        "n_cost_events": len(getattr(engine, "diag_cost_events", []) or []),
        "cost_model": getattr(engine, "COST_MODEL", "corwin_schultz"),
        "flat_cost_one_way": getattr(engine, "FLAT_COST_ONE_WAY", None),
    }


def format_worst_cost_table(worst: pd.DataFrame) -> str:
    lines = [
        "### TASK 1 — Worst 20 fills by $ execution cost (2022-2026, Corwin–Schultz model)",
        "",
        f"{'Sym':<8} {'Date':<12} {'Side':<5} {'Qty':>10} {'Px':>10} {'Notional':>14} "
        f"{'ADV20':>14} {'Part%':>7} {'CS%':>7} {'Imp%':>7} {'Cost%':>7} {'Cost$':>14} Flags",
        "-" * 140,
    ]
    if worst is None or worst.empty:
        lines.append("(no cost events)")
        return "\n".join(lines)

    n_part = int(worst["flag_participation_gt_20pct"].sum()) if "flag_participation_gt_20pct" in worst else 0
    n_rt = int(worst["flag_rt_equiv_gt_5pct"].sum()) if "flag_rt_equiv_gt_5pct" in worst else 0
    n_ow = int(worst["flag_one_way_gt_5pct"].sum()) if "flag_one_way_gt_5pct" in worst else 0

    for _, r in worst.iterrows():
        flags = []
        if bool(r.get("flag_participation_gt_20pct")):
            flags.append("PART>20%")
        if bool(r.get("flag_rt_equiv_gt_5pct")):
            flags.append("RT>5%")
        if bool(r.get("flag_one_way_gt_5pct")):
            flags.append("1W>5%")
        flag_s = ",".join(flags) if flags else "-"
        cs = _safe_float(r.get("cs_spread"))
        imp = _safe_float(r.get("impact"))
        part = _safe_float(r.get("participation"))
        cr = _safe_float(r.get("cost_ratio"))
        lines.append(
            f"{str(r['symbol']):<8} {pd.Timestamp(r['date']).date()!s:<12} "
            f"{str(r['side']):<5} {int(r['qty']):>10,} {_safe_float(r['price']):>10.2f} "
            f"{_safe_float(r['notional']):>14,.0f} {_safe_float(r.get('adv')):>14,.0f} "
            f"{100 * part:>6.2f}% {100 * cs:>6.2f}% {100 * imp:>6.2f}% {100 * cr:>6.2f}% "
            f"{_safe_float(r['cost_dollars']):>14,.0f} {flag_s}"
        )
    lines.append("-" * 140)
    lines.append(
        f"Flags in top-20: participation>20%={n_part}  "
        f"round-trip-equiv>5% (2×one-way)={n_rt}  one-way>5%={n_ow}"
    )
    lines.append(
        "Note: Part%=notional/ADV20 used in cost model; CS%=Corwin–Schultz full spread; "
        "Imp%=sqrt-impact component; Cost%=one-way total cost ratio applied."
    )
    return "\n".join(lines)


def format_cost_model_comparison(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "### TASK 2 — Cost-model sensitivity (identical strategy; 2022-01-01 → 2026-06-30)",
        "",
        f"{'Cost model':<28} {'CAGR':>10} {'Sharpe':>10} {'MDD':>10} {'Total cost $':>16} {'Alpha':>10}",
        "-" * 90,
    ]

    def pct(x):
        return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * float(x):.2f}%"

    def num(x):
        return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{float(x):.3f}"

    for r in rows:
        label = r.get("label", r.get("cost_model", "?"))
        lines.append(
            f"{label:<28} {pct(r.get('CAGR')):>10} {num(r.get('Sharpe')):>10} "
            f"{pct(r.get('Maximum_Drawdown')):>10} {float(r.get('total_cost_dollars') or 0):>16,.0f} "
            f"{pct(r.get('Alpha_CAGR')):>10}"
        )
    lines.append("-" * 90)
    lines.append(
        "Variants change ONLY cost_ratio applied at fill; participation caps, "
        "entry/exit/sizing/universe unchanged."
    )
    return "\n".join(lines)


def format_adv_windows(engine, worst: pd.DataFrame) -> str:
    lines = [
        "### TASK 3 — Raw dollar-volume ±10 trading days around top-20 highest-cost fills",
        "",
    ]
    if worst is None or worst.empty:
        lines.append("(no fills)")
        return "\n".join(lines)

    any_spike = False
    for i, r in worst.iterrows():
        sym = str(r["symbol"])
        td = r["date"]
        w = adv_window_series(engine, sym, td)
        lines.append(f"---- #{i+1} {sym} trade_date={pd.Timestamp(td).date()} side={r['side']} ----")
        if w.empty:
            lines.append("  (no dvol series)")
            continue
        spike_days = w[w["flag_volume_spike_10x"]]
        if len(spike_days):
            any_spike = True
            lines.append(
                f"  *** FLAG: {len(spike_days)} day(s) with dollar volume ≥10× median of other days in window ***"
            )
        for _, row in w.iterrows():
            mark = " <<<" if bool(row["is_trade_date"]) else ""
            spike = " SPIKE" if bool(row["flag_volume_spike_10x"]) else ""
            dv = row["dollar_volume"]
            dv_s = f"{dv:,.0f}" if pd.notna(dv) else "nan"
            lines.append(f"  {pd.Timestamp(row['date']).date()}  dvol={dv_s:>16}{mark}{spike}")
        lines.append("")
    if not any_spike:
        lines.append("No 10× volume spikes detected in these windows.")
    return "\n".join(lines)


def format_sizing_distribution(dist: Dict[str, Any]) -> str:
    lines = [
        "### TASK 4 — Entry notional / ADV20 distribution (BUY fills, 2022-2026)",
        "",
    ]
    if not dist or dist.get("n", 0) == 0:
        lines.append("(no BUY fills)")
        return "\n".join(lines)

    def pct(x):
        return "n/a" if x is None else f"{100 * float(x):.2f}%"

    lines.append(f"  n_buy_fills                 : {dist['n']:,}")
    lines.append(f"  p50 (notional/ADV)          : {pct(dist.get('p50'))}")
    lines.append(f"  p75                         : {pct(dist.get('p75'))}")
    lines.append(f"  p90                         : {pct(dist.get('p90'))}")
    lines.append(f"  p95                         : {pct(dist.get('p95'))}")
    lines.append(f"  max                         : {pct(dist.get('max'))}")
    lines.append(f"  mean                        : {pct(dist.get('mean'))}")
    lines.append(f"  share > 10% ADV             : {pct(dist.get('pct_gt_10pct_adv'))}")
    lines.append(f"  share > 15% ADV             : {pct(dist.get('pct_gt_15pct_adv'))}")
    lines.append(
        f"  share > participation_cap_buy : {pct(dist.get('pct_gt_participation_cap_buy'))}"
    )
    if float(dist.get("pct_gt_10pct_adv") or 0) >= 0.05:
        lines.append(
            "  *** FLAG: meaningful share of entries exceed 10% of ADV — sizing aggressive "
            "vs liquidity (or ADV mis-measured) ***"
        )
    return "\n".join(lines)


def format_full_cost_audit_report(
    *,
    worst: pd.DataFrame,
    comparison_rows: Sequence[Dict[str, Any]],
    engine_cs,
    sizing_dist: Dict[str, Any],
) -> str:
    parts = [
        "#" * 72,
        "# COST MODEL AUDIT — Baseline v1 (2022-01-01 → 2026-06-30)",
        "# Strategy entry/exit/sizing/leverage unchanged; cost model variants only in Task 2.",
        "#" * 72,
        "",
        format_worst_cost_table(worst),
        "",
        format_cost_model_comparison(comparison_rows),
        "",
        format_adv_windows(engine_cs, worst),
        "",
        format_sizing_distribution(sizing_dist),
        "",
        "# END COST MODEL AUDIT",
    ]
    return "\n".join(parts)
