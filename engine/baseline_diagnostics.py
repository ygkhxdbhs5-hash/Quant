"""Baseline v1 diagnostics — Hypothesis A vs B instrumentation.

Read-only analysis of BaselineEngineV1 runs. Does not modify trading logic.
Removable module: all diagnostic code lives here (plus light hooks in baseline_engine
that only aggregate costs / liquidity counts without changing fills).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from engine.research.kpi_report import build_hierarchical_kpi_report


def _years(equity: pd.DataFrame) -> float:
    if equity is None or equity.empty or len(equity) < 2:
        return 1e-9
    return max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)


def _cagr_sharpe_mdd(equity: pd.DataFrame) -> Dict[str, Optional[float]]:
    kpi = build_hierarchical_kpi_report(equity, pd.DataFrame())
    risk = kpi.get("risk") or {}
    return {
        "CAGR": risk.get("cagr"),
        "Sharpe": risk.get("sharpe"),
        "Maximum_Drawdown": risk.get("mdd"),
    }


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100.0 * float(x):>8.2f}%"


def _fmt_num(x: Optional[float], digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{float(x):>{8 + digits}.{digits}f}"


def trade_stats(engine) -> Dict[str, Any]:
    """Task 2 — trade count / turnover / win-loss / cost drag."""
    trades = engine.trade_journal.to_frame()
    equity = pd.DataFrame(engine.equity_curve).set_index("Date") if engine.equity_curve else pd.DataFrame()
    years = _years(equity) if not equity.empty else 1e-9
    avg_equity = (
        float(equity["Total_Equity"].mean())
        if not equity.empty and "Total_Equity" in equity.columns
        else float(engine.INITIAL_CASH)
    )

    n_closed = 0 if trades is None or trades.empty else len(trades)
    avg_hold = None
    win_rate = None
    avg_win = None
    avg_loss = None
    payoff = None
    if n_closed:
        hold = pd.to_numeric(trades["holding_days"], errors="coerce")
        rets = pd.to_numeric(trades["final_return"], errors="coerce").dropna()
        avg_hold = float(hold.mean()) if hold.notna().any() else None
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        win_rate = float((rets > 0).mean()) if len(rets) else None
        avg_win = float(wins.mean()) if len(wins) else None
        avg_loss = float(losses.mean()) if len(losses) else None
        if avg_win is not None and avg_loss is not None and abs(avg_loss) > 1e-12:
            payoff = float(avg_win) / abs(float(avg_loss))

    total_cost = float(getattr(engine, "diag_total_cost_dollars", 0.0) or 0.0)
    cost_drag_pct_avg_equity = total_cost / max(avg_equity, 1.0)

    # Portfolio replacement rate from closed trades / capacity / years
    capacity = max(int(getattr(engine, "TOP_MOMENTUM_COUNT", 30)), 1)
    replacement_rate = (n_closed / years) / capacity if years > 0 else None

    return {
        "n_closed_trades": int(n_closed),
        "trades_per_year": float(n_closed / years) if years > 0 else None,
        "portfolio_replacement_rate_per_year": replacement_rate,
        "avg_holding_period_days": avg_hold,
        "win_rate": win_rate,
        "avg_return_winning_trade": avg_win,
        "avg_return_losing_trade": avg_loss,
        "win_loss_payoff_ratio": payoff,
        "total_execution_cost_dollars": total_cost,
        "avg_equity": avg_equity,
        "execution_cost_drag_pct_of_avg_equity": cost_drag_pct_avg_equity,
        "years": float(years),
    }


def calendar_year_returns(equity: pd.DataFrame) -> pd.DataFrame:
    """Task 3a — per-calendar-year total return of strategy equity."""
    if equity is None or equity.empty or "Total_Equity" not in equity.columns:
        return pd.DataFrame(columns=["year", "total_return", "n_days"])
    eq = equity["Total_Equity"].astype(float).dropna()
    rows = []
    for year, ser in eq.groupby(eq.index.year):
        if len(ser) < 2:
            continue
        rows.append(
            {
                "year": int(year),
                "total_return": float(ser.iloc[-1] / ser.iloc[0] - 1.0),
                "n_days": int(len(ser)),
            }
        )
    return pd.DataFrame(rows)


def rolling_win_rate(trades: pd.DataFrame, window_days: int = 183) -> pd.DataFrame:
    """Task 3b — rolling ~6-month win rate by exit date."""
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["date", "rolling_win_rate", "n_trades_in_window"])
    df = trades.copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["final_return"] = pd.to_numeric(df["final_return"], errors="coerce")
    df = df.dropna(subset=["exit_date", "final_return"]).sort_values("exit_date")
    if df.empty:
        return pd.DataFrame(columns=["date", "rolling_win_rate", "n_trades_in_window"])

    # Evaluate at month-ends covering the trade span
    start = df["exit_date"].min()
    end = df["exit_date"].max()
    month_ends = pd.date_range(start, end, freq="ME")
    rows = []
    for t in month_ends:
        lo = t - pd.Timedelta(days=int(window_days))
        w = df[(df["exit_date"] > lo) & (df["exit_date"] <= t)]
        if w.empty:
            continue
        rows.append(
            {
                "date": t,
                "rolling_win_rate": float((w["final_return"] > 0).mean()),
                "n_trades_in_window": int(len(w)),
            }
        )
    return pd.DataFrame(rows)


def liquidity_concentration(engine, thin_threshold: int = 100) -> Dict[str, Any]:
    """Task 3c — monthly liquid-pool depth after TOP_LIQUID_POOL filter."""
    rows = list(getattr(engine, "diag_monthly_liquidity", []) or [])
    thin = [r for r in rows if r.get("flag_thin") or int(r.get("n_liquid_after_filter", 0)) < thin_threshold]
    return {
        "n_months": len(rows),
        "thin_threshold": int(thin_threshold),
        "n_thin_months": len(thin),
        "min_liquid": int(min((r["n_liquid_after_filter"] for r in rows), default=0)),
        "median_liquid": float(np.median([r["n_liquid_after_filter"] for r in rows])) if rows else None,
        "thin_months": thin[:24],  # cap print size
        "all_months": rows,
    }


def worst_trades_dominance(engine, top_n: int = 20, flag_share: float = 0.30) -> Dict[str, Any]:
    """Task 3d — worst N closed lots by dollar loss + concentration flag."""
    lots = list(getattr(engine, "diag_closed_lots", []) or [])
    if not lots:
        return {
            "worst": [],
            "total_dollar_pnl": 0.0,
            "top10_loss_share_of_abs_total": None,
            "flag_concentrated_losses": False,
            "note": "no closed lots recorded",
        }
    df = pd.DataFrame(lots)
    df["dollar_pnl"] = pd.to_numeric(df["dollar_pnl"], errors="coerce")
    total = float(df["dollar_pnl"].sum())
    worst = df.sort_values("dollar_pnl", ascending=True).head(int(top_n))
    losses = df[df["dollar_pnl"] < 0]["dollar_pnl"]
    top10_loss = float(losses.nsmallest(10).sum()) if len(losses) else 0.0
    abs_total = float(df["dollar_pnl"].abs().sum())
    share = abs(top10_loss) / abs_total if abs_total > 1e-9 else None
    # Also vs sum of all negative PnL
    sum_losses = float(losses.sum()) if len(losses) else 0.0
    share_of_losses = abs(top10_loss) / abs(sum_losses) if sum_losses < -1e-9 else None
    flag = bool(share is not None and share >= float(flag_share))
    return {
        "worst": worst.to_dict(orient="records"),
        "total_dollar_pnl": total,
        "sum_losing_trade_dollars": sum_losses,
        "top10_loss_dollars": top10_loss,
        "top10_loss_share_of_abs_total_pnl": share,
        "top10_loss_share_of_all_losses": share_of_losses,
        "flag_concentrated_losses": flag,
        "flag_threshold": float(flag_share),
    }


def gross_vs_net(engine) -> Dict[str, Any]:
    """Task 3e — gross (before costs) vs net (after costs) period returns."""
    diag = getattr(engine, "diag_equity_curve", None) or []
    if not diag:
        return {"net_total_return": None, "gross_total_return": None, "cost_drag_return": None}
    df = pd.DataFrame(diag).set_index("Date")
    net0, net1 = float(df["Net_Equity"].iloc[0]), float(df["Net_Equity"].iloc[-1])
    g0, g1 = float(df["Gross_Equity"].iloc[0]), float(df["Gross_Equity"].iloc[-1])
    net_ret = net1 / net0 - 1.0 if net0 > 0 else None
    gross_ret = g1 / g0 - 1.0 if g0 > 0 else None
    drag = None
    if net_ret is not None and gross_ret is not None:
        drag = float(gross_ret) - float(net_ret)
    net_eq = df[["Net_Equity"]].rename(columns={"Net_Equity": "Total_Equity"})
    gross_eq = df[["Gross_Equity"]].rename(columns={"Gross_Equity": "Total_Equity"})
    return {
        "net_total_return": net_ret,
        "gross_total_return": gross_ret,
        "cost_drag_return": drag,
        "net_kpi": _cagr_sharpe_mdd(net_eq),
        "gross_kpi": _cagr_sharpe_mdd(gross_eq),
        "total_cost_dollars": float(getattr(engine, "diag_total_cost_dollars", 0.0) or 0.0),
    }


def period_comparison(engine) -> Dict[str, Any]:
    equity = pd.DataFrame(engine.equity_curve).set_index("Date") if engine.equity_curve else pd.DataFrame()
    qqq = (
        pd.DataFrame(engine.qqq_equity_curve).set_index("Date")
        if engine.qqq_equity_curve
        else pd.DataFrame()
    )
    s = _cagr_sharpe_mdd(equity.rename(columns={"Total_Equity": "Total_Equity"}) if not equity.empty else equity)
    if not equity.empty and "Total_Equity" not in equity.columns and "Net_Equity" in equity.columns:
        s = _cagr_sharpe_mdd(equity.rename(columns={"Net_Equity": "Total_Equity"}))
    q = _cagr_sharpe_mdd(qqq) if not qqq.empty else {"CAGR": None, "Sharpe": None, "Maximum_Drawdown": None}
    alpha = None
    if s.get("CAGR") is not None and q.get("CAGR") is not None:
        alpha = float(s["CAGR"]) - float(q["CAGR"])
    return {"strategy": s, "QQQ": q, "Alpha_CAGR": alpha}


def analyze_run(engine, label: str, start: str, end: str) -> Dict[str, Any]:
    trades = engine.trade_journal.to_frame()
    equity = pd.DataFrame(engine.equity_curve).set_index("Date") if engine.equity_curve else pd.DataFrame()
    return {
        "label": label,
        "start": start,
        "end": end,
        "comparison": period_comparison(engine),
        "trade_stats": trade_stats(engine),
        "calendar_year_returns": calendar_year_returns(equity).to_dict(orient="records"),
        "rolling_6m_win_rate": rolling_win_rate(trades).to_dict(orient="records"),
        "liquidity": liquidity_concentration(engine),
        "worst_trades": worst_trades_dominance(engine),
        "gross_vs_net": gross_vs_net(engine),
        "n_equity_days": 0 if equity.empty else len(equity),
    }


def format_comparison_table(label: str, comparison: Dict[str, Any]) -> str:
    s = comparison.get("strategy") or {}
    q = comparison.get("QQQ") or {}
    lines = [
        "=" * 64,
        f"  KPI SUMMARY — {label}",
        "=" * 64,
        f"{'Metric':<22} {'Strategy':>14} {'QQQ':>14}",
        "-" * 64,
        f"{'CAGR':<22} {_fmt_pct(s.get('CAGR')):>14} {_fmt_pct(q.get('CAGR')):>14}",
        f"{'Sharpe':<22} {_fmt_num(s.get('Sharpe')):>14} {_fmt_num(q.get('Sharpe')):>14}",
        f"{'Maximum Drawdown':<22} {_fmt_pct(s.get('Maximum_Drawdown')):>14} {_fmt_pct(q.get('Maximum_Drawdown')):>14}",
        "-" * 64,
        f"{'Alpha (CAGR−QQQ)':<22} {_fmt_pct(comparison.get('Alpha_CAGR')):>14}",
        "=" * 64,
    ]
    return "\n".join(lines)


def format_full_report(results: List[Dict[str, Any]], full_label: str = "2022-2026") -> str:
    """Structured paste-ready diagnostic report."""
    out: List[str] = []
    out.append("#" * 72)
    out.append("# BASELINE v1 DIAGNOSTIC REPORT — Hypothesis A vs B")
    out.append("# Hypothesis A: momentum-crash / regime (losses concentrated)")
    out.append("# Hypothesis B: pipeline/data/execution bug (losses broad / weird)")
    out.append("#" * 72)

    # ----- Task 1 -----
    out.append("")
    out.append("## TASK 1 — Multi-period Strategy vs QQQ")
    out.append("")
    for r in results:
        out.append(format_comparison_table(r["label"], r["comparison"]))
        out.append("")

    # Summary matrix
    out.append("-" * 72)
    out.append(f"{'Period':<22} {'Strat CAGR':>12} {'QQQ CAGR':>12} {'Alpha':>12} {'Strat MDD':>12}")
    out.append("-" * 72)
    for r in results:
        c = r["comparison"]
        s, q = c.get("strategy") or {}, c.get("QQQ") or {}
        out.append(
            f"{r['label']:<22} {_fmt_pct(s.get('CAGR')):>12} {_fmt_pct(q.get('CAGR')):>12} "
            f"{_fmt_pct(c.get('Alpha_CAGR')):>12} {_fmt_pct(s.get('Maximum_Drawdown')):>12}"
        )
    out.append("-" * 72)

    # ----- Task 2 -----
    out.append("")
    out.append("## TASK 2 — Trade count / turnover / win-loss / cost drag")
    out.append("")
    for r in results:
        ts = r["trade_stats"]
        out.append(f"### {r['label']} ({r['start']} → {r['end']})")
        out.append(f"  closed_trades                 : {ts['n_closed_trades']}")
        out.append(f"  trades_per_year               : {_fmt_num(ts.get('trades_per_year'), 1)}")
        out.append(
            f"  portfolio_replacement_rate/yr : {_fmt_num(ts.get('portfolio_replacement_rate_per_year'), 2)}"
        )
        out.append(f"  avg_holding_period_days       : {_fmt_num(ts.get('avg_holding_period_days'), 1)}")
        out.append(f"  win_rate                      : {_fmt_pct(ts.get('win_rate'))}")
        out.append(f"  avg_return_winning_trade      : {_fmt_pct(ts.get('avg_return_winning_trade'))}")
        out.append(f"  avg_return_losing_trade       : {_fmt_pct(ts.get('avg_return_losing_trade'))}")
        out.append(f"  win/loss payoff ratio         : {_fmt_num(ts.get('win_loss_payoff_ratio'), 2)}")
        out.append(
            f"  execution_cost_$               : {ts.get('total_execution_cost_dollars', 0):,.0f}"
        )
        out.append(
            f"  cost_drag_%_of_avg_equity      : {_fmt_pct(ts.get('execution_cost_drag_pct_of_avg_equity'))}"
        )
        out.append("")

    # ----- Task 3 -----
    out.append("## TASK 3 — Diagnostic checks (A vs B)")
    out.append("")

    # Prefer the full-period run for 3a/3b if present
    full = next((r for r in results if r["label"] == full_label), results[-1] if results else None)

    out.append("### 3a) Per-calendar-year strategy returns")
    if full:
        out.append(f"(from run: {full['label']})")
        out.append(f"{'Year':<8} {'Total Return':>14} {'n_days':>8}")
        for row in full.get("calendar_year_returns") or []:
            out.append(
                f"{row['year']:<8} {_fmt_pct(row['total_return']):>14} {int(row['n_days']):>8}"
            )
        # Also print year returns for every multi-year period
        for r in results:
            if r is full:
                continue
            yrs = r.get("calendar_year_returns") or []
            if len(yrs) >= 2:
                out.append(f"  -- also {r['label']}: " + ", ".join(
                    f"{y['year']}={_fmt_pct(y['total_return']).strip()}" for y in yrs
                ))
    out.append("")

    out.append("### 3b) Rolling 6-month win rate (exit-date window)")
    if full:
        rr = full.get("rolling_6m_win_rate") or []
        out.append(f"(from run: {full['label']}; window ≈ 183 days)")
        if not rr:
            out.append("  (no trades)")
        else:
            out.append(f"{'Month-end':<12} {'WinRate':>10} {'n_trades':>10}")
            for row in rr:
                dt = pd.Timestamp(row["date"]).strftime("%Y-%m")
                out.append(
                    f"{dt:<12} {_fmt_pct(row['rolling_win_rate']):>10} {int(row['n_trades_in_window']):>10}"
                )
            rates = [row["rolling_win_rate"] for row in rr]
            out.append(
                f"  summary: min={_fmt_pct(min(rates)).strip()}  "
                f"median={_fmt_pct(float(np.median(rates))).strip()}  "
                f"max={_fmt_pct(max(rates)).strip()}"
            )
    out.append("")

    out.append("### 3c) Monthly liquidity concentration (after TOP_LIQUID_POOL)")
    for r in results:
        liq = r.get("liquidity") or {}
        out.append(
            f"  {r['label']}: months={liq.get('n_months')}  "
            f"min_liquid={liq.get('min_liquid')}  "
            f"median_liquid={_fmt_num(liq.get('median_liquid'), 0).strip()}  "
            f"thin_months(<{liq.get('thin_threshold')})={liq.get('n_thin_months')}"
        )
        if liq.get("n_thin_months", 0) > 0:
            out.append("  FLAG: thin liquidity months detected (possible universe/data issue):")
            for m in (liq.get("thin_months") or [])[:12]:
                out.append(
                    f"    {pd.Timestamp(m['date']).date()}  "
                    f"n_liquid={m['n_liquid_after_filter']}  n_live={m['n_live']}"
                )
    out.append("")

    out.append("### 3d) Worst 20 closed trades by dollar loss + concentration")
    for r in results:
        wt = r.get("worst_trades") or {}
        out.append(f"#### {r['label']}")
        out.append(
            f"  total_dollar_pnl={wt.get('total_dollar_pnl', 0):,.0f}  "
            f"sum_losses={wt.get('sum_losing_trade_dollars', 0):,.0f}  "
            f"top10_loss={wt.get('top10_loss_dollars', 0):,.0f}"
        )
        out.append(
            f"  top10_share_of_|total_PnL|={_fmt_pct(wt.get('top10_loss_share_of_abs_total_pnl')).strip()}  "
            f"top10_share_of_losses={_fmt_pct(wt.get('top10_loss_share_of_all_losses')).strip()}"
        )
        if wt.get("flag_concentrated_losses"):
            out.append(
                f"  *** FLAG: top-10 losing trades ≥ {100 * float(wt.get('flag_threshold', 0.3)):.0f}% "
                f"of |total PnL| — investigate data/delist/cost bugs (Hypothesis B signal) ***"
            )
        else:
            out.append("  concentration flag: not triggered")
        out.append(
            f"  {'Symbol':<8} {'Entry':<12} {'Exit':<12} {'EntryPx':>10} {'ExitPx':>10} {'Pct':>9} {'$PnL':>14}"
        )
        for row in (wt.get("worst") or [])[:20]:
            out.append(
                f"  {str(row.get('symbol')):<8} "
                f"{str(pd.Timestamp(row.get('entry_date')).date()) if row.get('entry_date') is not None else 'n/a':<12} "
                f"{str(pd.Timestamp(row.get('exit_date')).date()) if row.get('exit_date') is not None else 'n/a':<12} "
                f"{float(row.get('entry_price', np.nan)):10.2f} "
                f"{float(row.get('exit_price', np.nan)):10.2f} "
                f"{_fmt_pct(row.get('pct_return')):>9} "
                f"{float(row.get('dollar_pnl', 0)):14,.0f}"
            )
        out.append("")

    out.append("### 3e) Gross (before costs) vs Net (after costs)")
    for r in results:
        g = r.get("gross_vs_net") or {}
        out.append(f"  {r['label']}:")
        out.append(f"    net_total_return   : {_fmt_pct(g.get('net_total_return')).strip()}")
        out.append(f"    gross_total_return : {_fmt_pct(g.get('gross_total_return')).strip()}")
        out.append(f"    cost_drag_return   : {_fmt_pct(g.get('cost_drag_return')).strip()}")
        nk = g.get("net_kpi") or {}
        gk = g.get("gross_kpi") or {}
        out.append(
            f"    net CAGR/Sharpe/MDD   : {_fmt_pct(nk.get('CAGR')).strip()} / "
            f"{_fmt_num(nk.get('Sharpe')).strip()} / {_fmt_pct(nk.get('Maximum_Drawdown')).strip()}"
        )
        out.append(
            f"    gross CAGR/Sharpe/MDD : {_fmt_pct(gk.get('CAGR')).strip()} / "
            f"{_fmt_num(gk.get('Sharpe')).strip()} / {_fmt_pct(gk.get('Maximum_Drawdown')).strip()}"
        )
        out.append(f"    total_cost_$        : {g.get('total_cost_dollars', 0):,.0f}")
    out.append("")

    out.append("## INTERPRETATION HINTS (auto)")
    out.append(_auto_hints(results, full_label=full_label))
    out.append("")
    out.append("# END DIAGNOSTIC REPORT")
    return "\n".join(out)


def _auto_hints(results: List[Dict[str, Any]], full_label: str) -> str:
    lines = []
    full = next((r for r in results if r["label"] == full_label), None)
    crash = next((r for r in results if "2022 only" in r["label"] or r["label"] == "2022"), None)

    # Year clustering
    if full:
        years = full.get("calendar_year_returns") or []
        bad = [y for y in years if y.get("total_return") is not None and y["total_return"] < -0.15]
        if bad and len(bad) <= max(1, len(years) // 2):
            lines.append(
                f"- Year returns show concentrated weakness in {[y['year'] for y in bad]} "
                f"(supports Hypothesis A if these align with known momentum-crash years)."
            )
        elif years and all((y.get("total_return") or 0) < 0 for y in years):
            lines.append(
                "- Every calendar year in the full window is negative "
                "(leans toward Hypothesis B or a chronically unsuitable universe/signal)."
            )

    if crash:
        c = crash["comparison"]
        s = (c.get("strategy") or {}).get("CAGR")
        q = (c.get("QQQ") or {}).get("CAGR")
        if s is not None and q is not None and float(s) < float(q) - 0.10:
            lines.append(
                f"- Isolated 2022 run: strategy CAGR={_fmt_pct(s).strip()} vs QQQ={_fmt_pct(q).strip()} "
                f"(large relative underperformance in the classic crash year → Hypothesis A signal)."
            )

    # Liquidity
    for r in results:
        if (r.get("liquidity") or {}).get("n_thin_months", 0) > 0:
            lines.append(
                f"- Thin liquidity months in {r['label']} "
                f"(Hypothesis B: universe/data depth issue)."
            )
            break
    else:
        lines.append("- No thin-liquidity months flagged (universe depth OK for TOP_LIQUID_POOL).")

    # Concentration
    for r in results:
        wt = r.get("worst_trades") or {}
        if wt.get("flag_concentrated_losses"):
            lines.append(
                f"- Concentrated dollar losses in {r['label']} "
                f"(Hypothesis B: check delistings / bad prints / cost outliers)."
            )
            break
    else:
        lines.append("- Top-10 loss concentration flag not triggered across periods.")

    # Cost drag
    for r in results:
        g = r.get("gross_vs_net") or {}
        drag = g.get("cost_drag_return")
        net = g.get("net_total_return")
        if drag is not None and net is not None and abs(net) > 1e-9:
            if abs(drag) > 0.5 * abs(net) and net < 0:
                lines.append(
                    f"- In {r['label']}, cost drag ({_fmt_pct(drag).strip()}) is large vs net return "
                    f"({_fmt_pct(net).strip()}) — costs matter, but check if drag alone explains the gap."
                )
                break
    else:
        lines.append(
            "- Execution cost drag is modest relative to net losses "
            "(underperformance is mostly signal/regime, not fee model alone)."
        )

    if not lines:
        lines.append("- Insufficient signal for automatic classification; review tables manually.")
    return "\n".join(lines)
