"""Diagnostics: new-entry vs top-up chasing + cost component split.

Read-only aggregation over BaselineEngineV1.diag_cost_events /
diag_fill_vs_target. No strategy signal changes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

FLAT_10BPS_ONE_WAY = 0.0005  # 5bps each way → 10bps RT


def _events_df(engine) -> pd.DataFrame:
    rows = list(getattr(engine, "diag_cost_events", []) or [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in (
        "notional",
        "cost_dollars",
        "spread_cost_dollars",
        "impact_cost_dollars",
        "commission_cost_dollars",
        "slippage_cost_dollars",
        "half_spread",
        "impact",
        "cost_ratio",
    ):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fills_df(engine) -> pd.DataFrame:
    rows = list(getattr(engine, "diag_fill_vs_target", []) or [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("fill_pct_of_target", "filled_notional", "target_notional", "filled_qty"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def analyze_new_entry_vs_topup(engine) -> Dict[str, Any]:
    """Task 1 — classify BUY orders / costs as new_entry vs top_up."""
    fills = _fills_df(engine)
    costs = _events_df(engine)
    out: Dict[str, Any] = {"n_buy_orders": 0}

    if not fills.empty:
        buys = fills[fills["side"].astype(str).str.upper() == "BUY"].copy()
        # Prefer tagged role; fallback
        if "order_role" not in buys.columns:
            buys["order_role"] = "unknown"
        buys["order_role"] = buys["order_role"].fillna("unknown").astype(str)
        n = len(buys)
        out["n_buy_orders"] = int(n)
        for role in ("new_entry", "top_up", "unknown"):
            sub = buys[buys["order_role"] == role]
            out[f"n_{role}"] = int(len(sub))
            out[f"pct_{role}"] = float(len(sub) / n) if n else None
            out[f"mean_fill_pct_{role}"] = (
                float(sub["fill_pct_of_target"].mean()) if len(sub) else None
            )
    else:
        for role in ("new_entry", "top_up", "unknown"):
            out[f"n_{role}"] = 0
            out[f"pct_{role}"] = None
            out[f"mean_fill_pct_{role}"] = None

    # Cost $ by role (BUY fills only)
    if not costs.empty:
        buys_c = costs[costs["side"].astype(str).str.upper() == "BUY"].copy()
        if "order_role" not in buys_c.columns:
            buys_c["order_role"] = "unknown"
        buys_c["order_role"] = buys_c["order_role"].fillna("unknown").astype(str)
        out["buy_cost_total"] = float(buys_c["cost_dollars"].sum())
        for role in ("new_entry", "top_up", "unknown"):
            sub = buys_c[buys_c["order_role"] == role]
            out[f"buy_cost_{role}"] = float(sub["cost_dollars"].sum()) if len(sub) else 0.0
            out[f"buy_notional_{role}"] = float(sub["notional"].sum()) if len(sub) else 0.0
        out["sell_cost_total"] = float(
            costs.loc[costs["side"].astype(str).str.upper() == "SELL", "cost_dollars"].sum()
        )
        out["all_cost_total"] = float(costs["cost_dollars"].sum())
    else:
        out["buy_cost_total"] = 0.0
        out["buy_cost_new_entry"] = 0.0
        out["buy_cost_top_up"] = 0.0
        out["sell_cost_total"] = 0.0
        out["all_cost_total"] = 0.0

    # Symbols with 3+ top-up attempts
    chronic: List[Dict[str, Any]] = []
    if not fills.empty:
        buys = fills[fills["side"].astype(str).str.upper() == "BUY"].copy()
        buys["order_role"] = buys.get("order_role", pd.Series("unknown", index=buys.index))
        buys["order_role"] = buys["order_role"].fillna("unknown").astype(str)
        topups = buys[buys["order_role"] == "top_up"]
        if not topups.empty:
            grp = topups.groupby("symbol")
            costs_buy = costs[costs["side"].astype(str).str.upper() == "BUY"].copy() if not costs.empty else pd.DataFrame()
            if not costs_buy.empty:
                costs_buy["order_role"] = costs_buy.get(
                    "order_role", pd.Series("unknown", index=costs_buy.index)
                )
                costs_buy["order_role"] = costs_buy["order_role"].fillna("unknown").astype(str)

            for sym, g in grp:
                if len(g) < 3:
                    continue
                tot_cost = 0.0
                tot_notional = 0.0
                if not costs_buy.empty:
                    cs = costs_buy[
                        (costs_buy["symbol"] == sym) & (costs_buy["order_role"] == "top_up")
                    ]
                    tot_cost = float(cs["cost_dollars"].sum()) if len(cs) else 0.0
                    tot_notional = float(cs["notional"].sum()) if len(cs) else 0.0
                flat_cost = tot_notional * FLAT_10BPS_ONE_WAY
                chronic.append(
                    {
                        "symbol": sym,
                        "n_topup_attempts": int(len(g)),
                        "avg_fill_pct_per_attempt": float(g["fill_pct_of_target"].mean()),
                        "total_topup_cost_dollars": tot_cost,
                        "total_topup_notional": tot_notional,
                        "flat_10bps_one_way_cost_on_same_notional": flat_cost,
                        "cost_multiple_vs_flat_10bps": (
                            tot_cost / flat_cost if flat_cost > 0 else None
                        ),
                    }
                )
            chronic.sort(key=lambda r: r["total_topup_cost_dollars"], reverse=True)

    out["chronic_topup_symbols_3plus"] = chronic
    out["n_chronic_topup_symbols"] = len(chronic)
    if chronic:
        out["chronic_topup_total_cost"] = float(
            sum(r["total_topup_cost_dollars"] for r in chronic)
        )
        out["chronic_topup_flat10_equiv_cost"] = float(
            sum(r["flat_10bps_one_way_cost_on_same_notional"] for r in chronic)
        )
    else:
        out["chronic_topup_total_cost"] = 0.0
        out["chronic_topup_flat10_equiv_cost"] = 0.0
    return out


def analyze_cost_components(engine) -> Dict[str, Any]:
    """Task 2 — spread vs impact $ decomposition, overall and by order_role."""
    costs = _events_df(engine)
    if costs.empty:
        return {"total_cost": 0.0}

    def _block(df: pd.DataFrame) -> Dict[str, Any]:
        tot = float(df["cost_dollars"].sum())
        spread = float(df["spread_cost_dollars"].fillna(0).sum()) if "spread_cost_dollars" in df else 0.0
        impact = float(df["impact_cost_dollars"].fillna(0).sum()) if "impact_cost_dollars" in df else 0.0
        comm = float(df["commission_cost_dollars"].fillna(0).sum()) if "commission_cost_dollars" in df else 0.0
        slip = float(df["slippage_cost_dollars"].fillna(0).sum()) if "slippage_cost_dollars" in df else 0.0
        return {
            "total_cost": tot,
            "spread_cost": spread,
            "impact_cost": impact,
            "commission_cost": comm,
            "slippage_cost": slip,
            "pct_spread": spread / tot if tot else None,
            "pct_impact": impact / tot if tot else None,
            "pct_commission": comm / tot if tot else None,
            "pct_slippage": slip / tot if tot else None,
            "n_fills": int(len(df)),
        }

    out = {"all_fills": _block(costs)}
    buys = costs[costs["side"].astype(str).str.upper() == "BUY"].copy()
    if "order_role" not in buys.columns:
        buys["order_role"] = "unknown"
    buys["order_role"] = buys["order_role"].fillna("unknown").astype(str)
    out["buy_all"] = _block(buys)
    out["buy_new_entry"] = _block(buys[buys["order_role"] == "new_entry"])
    out["buy_top_up"] = _block(buys[buys["order_role"] == "top_up"])
    sells = costs[costs["side"].astype(str).str.upper() == "SELL"]
    out["sell_all"] = _block(sells)
    return out


def kpi_with_trades(engine) -> Dict[str, Any]:
    from engine.cost_model_audit import kpi_slice

    m = kpi_slice(engine)
    trades = engine.trade_journal.to_frame()
    m["n_closed_trades"] = int(len(trades)) if trades is not None and not trades.empty else 0
    m["disable_topup_chasing"] = bool(getattr(engine, "DISABLE_TOPUP_CHASING", False))
    return m


def format_chase_report(
    *,
    task1: Dict[str, Any],
    task2: Dict[str, Any],
    comparison_rows: List[Dict[str, Any]],
    repro: Optional[Dict[str, Any]] = None,
) -> str:
    def pct(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{100 * float(x):.2f}%"

    def money(x):
        if x is None:
            return "n/a"
        return f"{float(x):,.0f}"

    def num(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{float(x):.3f}"

    lines = [
        "#" * 72,
        "# TOP-UP CHASE + COST DECOMPOSITION — Baseline v1 (2022-2026)",
        "# Fixed CS v2 cost model; entry/exit signals unchanged.",
        "#" * 72,
        "",
    ]
    if repro:
        from engine.repro_fingerprint import format_fingerprint_banner

        lines += [
            "### REPRO FINGERPRINT (must match chart footer)",
            f"  {format_fingerprint_banner(repro)}",
            f"  fingerprint_sha256 : {repro.get('fingerprint_sha256')}",
            f"  git_head           : {repro.get('git_head')}"
            f"{' (dirty tree)' if repro.get('git_dirty') else ''}",
            "",
        ]
    lines += [
        "### TASK 1 — New entry vs top-up chasing (BUY orders, Fixed CS v2 + chase ON)",
        f"  n_buy_orders           : {task1.get('n_buy_orders', 0):,}",
        f"  new_entry   count / %  : {task1.get('n_new_entry', 0):,} / {pct(task1.get('pct_new_entry'))}",
        f"  top_up      count / %  : {task1.get('n_top_up', 0):,} / {pct(task1.get('pct_top_up'))}",
        f"  mean fill%  new_entry  : {pct(task1.get('mean_fill_pct_new_entry'))}",
        f"  mean fill%  top_up     : {pct(task1.get('mean_fill_pct_top_up'))}",
        f"  BUY cost $  new_entry  : {money(task1.get('buy_cost_new_entry'))}",
        f"  BUY cost $  top_up     : {money(task1.get('buy_cost_top_up'))}",
        f"  BUY cost $  total      : {money(task1.get('buy_cost_total'))}",
        f"  SELL cost $ total      : {money(task1.get('sell_cost_total'))}",
        f"  ALL cost $  total      : {money(task1.get('all_cost_total'))}",
        f"  top_up share of BUY $  : {pct((task1.get('buy_cost_top_up') or 0) / (task1.get('buy_cost_total') or 1))}",
        "",
        f"  chronic symbols (3+ top-ups): {task1.get('n_chronic_topup_symbols', 0)}",
        f"  chronic top-up cost $       : {money(task1.get('chronic_topup_total_cost'))}",
        f"  same notional @ flat 10bps  : {money(task1.get('chronic_topup_flat10_equiv_cost'))}",
        "  top chronic symbols:",
    ]
    for r in (task1.get("chronic_topup_symbols_3plus") or [])[:15]:
        lines.append(
            f"    {r['symbol']:<8} n={r['n_topup_attempts']:>3}  "
            f"avg_fill={pct(r['avg_fill_pct_per_attempt'])}  "
            f"topup_cost$={money(r['total_topup_cost_dollars'])}  "
            f"flat10$={money(r['flat_10bps_one_way_cost_on_same_notional'])}  "
            f"×flat={num(r['cost_multiple_vs_flat_10bps'])}"
        )

    a = task2.get("all_fills") or {}
    be = task2.get("buy_new_entry") or {}
    bt = task2.get("buy_top_up") or {}
    lines += [
        "",
        "### TASK 2 — Cost component decomposition (Fixed CS v2 + chase ON)",
        f"  ALL fills: total$={money(a.get('total_cost'))}  "
        f"spread$={money(a.get('spread_cost'))} ({pct(a.get('pct_spread'))})  "
        f"impact$={money(a.get('impact_cost'))} ({pct(a.get('pct_impact'))})  "
        f"comm$={money(a.get('commission_cost'))}  slip$={money(a.get('slippage_cost'))}",
        f"  BUY new_entry: total$={money(be.get('total_cost'))}  "
        f"spread$={money(be.get('spread_cost'))} ({pct(be.get('pct_spread'))})  "
        f"impact$={money(be.get('impact_cost'))} ({pct(be.get('pct_impact'))})",
        f"  BUY top_up   : total$={money(bt.get('total_cost'))}  "
        f"spread$={money(bt.get('spread_cost'))} ({pct(bt.get('pct_spread'))})  "
        f"impact$={money(bt.get('impact_cost'))} ({pct(bt.get('pct_impact'))})",
        "",
        "### TASK 3 — No-chase variant vs cost benchmarks (2022-01-01 → 2026-06-30)",
        f"{'Variant':<40} {'CAGR':>10} {'Sharpe':>10} {'MDD':>10} {'Cost $':>14} {'n_trades':>10}",
        "-" * 100,
    ]
    for r in comparison_rows:
        lines.append(
            f"{r.get('label', '?'):<40} {pct(r.get('CAGR')):>10} {num(r.get('Sharpe')):>10} "
            f"{pct(r.get('Maximum_Drawdown')):>10} {money(r.get('total_cost_dollars')):>14} "
            f"{int(r.get('n_closed_trades') or 0):>10,}"
        )
    lines += [
        "-" * 100,
        "No-chase = Fixed CS v2 + DISABLE_TOPUP_CHASING (skip BUY top-ups / size-trims for held names).",
        "Entry momentum factor, ATR trail exit, and leverage unchanged.",
        "",
        "# END REPORT",
    ]
    return "\n".join(lines)
