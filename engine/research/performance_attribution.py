"""Performance Attribution System (observation-only).

Computes an end-of-backtest attribution report explaining where return /
performance drag came from, using only passive diagnostics already produced
by the engine (daily cash/exposure tracking + realized PnL/cost tracking).

This module must never modify trading decisions or engine state.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _percentile(values: List[float], q: float) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q / 100.0))


def _mean(values: List[float]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _median(values: List[float]) -> Optional[float]:
    return _percentile(values, 50.0)


def _format_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{float(x) * 100.0:.{digits}f}%"


def _format_money(x: Optional[float], digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{float(x):,.{digits}f}"


def _sha_equity(equity: Any) -> str:
    try:
        arr = np.asarray(equity["Total_Equity"], dtype=float)
        h = sha256(arr.tobytes()).hexdigest()
        return h
    except Exception:
        return "unavailable"


@dataclass(frozen=True)
class AttributionInputs:
    kpi: Dict[str, Any]
    trades_df: Any
    equity_df: Any


class PerformanceAttributionSystem:
    def __init__(self) -> None:
        pass

    def _collect_daily_series(self, engine) -> Dict[str, Any]:
        # These are set by passive logging inside StandaloneEngine.run().
        cash = getattr(engine, "_daily_cash", None)
        inv = getattr(engine, "_daily_invested_capital", None)
        total = getattr(engine, "_daily_total_equity", None)
        exposure = getattr(engine, "_daily_exposure_frac", None)
        num_holdings = getattr(engine, "_daily_num_holdings", None)
        pos_size = getattr(engine, "_daily_avg_position_size", None)

        return {
            "cash": cash or [],
            "invested_capital": inv or [],
            "total_equity": total or [],
            "exposure_frac": exposure or [],
            "num_holdings": num_holdings or [],
            "avg_position_size": pos_size or [],
        }

    def _collect_realized_pnl(self, engine) -> Dict[str, Any]:
        pnls = getattr(engine, "_closed_trade_realized_pnls", None) or []
        commission = float(getattr(engine, "_total_commission_paid", 0.0))
        slippage = float(getattr(engine, "_total_slippage_paid", 0.0))
        return {
            "closed_trade_realized_pnls": list(pnls),
            "total_commission_paid": commission,
            "total_slippage_paid": slippage,
        }

    def build(self, engine, *, rank_diagnostics_summary: Dict[str, Any], attribution_inputs: AttributionInputs) -> Dict[str, Any]:
        daily = self._collect_daily_series(engine)
        pnl = self._collect_realized_pnl(engine)
        holding_days = []
        try:
            trades_df = attribution_inputs.trades_df
            if trades_df is not None and not trades_df.empty and "holding_days" in trades_df.columns:
                holding_days = trades_df["holding_days"].astype(float).tolist()
        except Exception:
            holding_days = []

        # Rank distribution while holding (holdings-days observations)
        rank_holdings = (getattr(engine, "rank_diagnostics", None) and getattr(engine.rank_diagnostics, "holding_ranks", None)) or []
        if not rank_holdings and rank_diagnostics_summary:
            # Fallback: only counters exist in summary, so leave rank stats N/A.
            rank_holdings = []

        gross_win = float(sum([x for x in pnl["closed_trade_realized_pnls"] if x is not None and np.isfinite(x) and x > 0.0]))
        gross_loss = float(sum([x for x in pnl["closed_trade_realized_pnls"] if x is not None and np.isfinite(x) and x < 0.0]))
        net_pnl = float(gross_win + gross_loss)

        positive_pnls = sorted([x for x in pnl["closed_trade_realized_pnls"] if x is not None and np.isfinite(x) and x > 0.0], reverse=True)
        negative_pnls = sorted([x for x in pnl["closed_trade_realized_pnls"] if x is not None and np.isfinite(x) and x < 0.0])

        largest_win = positive_pnls[0] if positive_pnls else None
        largest_loss = negative_pnls[0] if negative_pnls else None  # most negative

        top10_winners_contrib = None
        if gross_win > 0 and positive_pnls:
            top10 = positive_pnls[:10]
            top10_winners_contrib = float(sum(top10) / gross_win) if gross_win else None

        top10_losers_contrib = None
        if gross_loss < 0 and negative_pnls:
            # losers contribution by absolute loss
            abs_gross_loss = abs(gross_loss)
            top10 = negative_pnls[:10]
            top10_losers_contrib = float(abs(sum(top10)) / abs_gross_loss) if abs_gross_loss else None

        commission_pct_gross_profit = None
        if gross_win > 0:
            commission_pct_gross_profit = pnl["total_commission_paid"] / gross_win
        slippage_pct_gross_profit = None
        if gross_win > 0:
            slippage_pct_gross_profit = pnl["total_slippage_paid"] / gross_win

        cash = daily["cash"]
        inv = daily["invested_capital"]
        total = daily["total_equity"]
        exposure = daily["exposure_frac"]
        num_holdings = daily["num_holdings"]
        avg_position_size = daily["avg_position_size"]

        avg_exposure = _mean(exposure)  # fraction
        avg_cash_frac = None
        if total:
            # compute cash fraction as cash/total
            cash_frac = []
            for c, t in zip(cash, total):
                if t and np.isfinite(t) and t != 0:
                    cash_frac.append(float(c) / float(t))
            avg_cash_frac = _mean(cash_frac)

        max_cash_frac = None
        min_cash_frac = None
        if total:
            cash_frac = []
            for c, t in zip(cash, total):
                if t and np.isfinite(t) and t != 0:
                    cash_frac.append(float(c) / float(t))
            if cash_frac:
                max_cash_frac = float(np.max(cash_frac))
                min_cash_frac = float(np.min(cash_frac))

        avg_num_holdings = _mean([float(x) for x in num_holdings]) if num_holdings else None
        avg_pos_size = _mean(avg_position_size) if avg_position_size else None

        avg_daily_cash = _mean(cash)
        avg_daily_invested = _mean(inv)
        idle_days = int(sum(1 for x in inv if x is not None and np.isfinite(x) and float(x) <= 1e-12))
        # Capital utilization rate: avg exposure fraction
        cap_util_rate = avg_exposure

        # Portfolio distribution
        avg_port_size = _mean([float(x) for x in num_holdings]) if num_holdings else None
        med_port_size = _median([float(x) for x in num_holdings]) if num_holdings else None
        max_port_size = float(np.max(num_holdings)) if num_holdings else None
        min_port_size = float(np.min(num_holdings)) if num_holdings else None

        # Holding distribution from closed trades
        holding_avg = _mean(holding_days)
        holding_med = _median(holding_days)
        holding_p75 = _percentile(holding_days, 75.0)
        holding_p90 = _percentile(holding_days, 90.0)
        holding_max = float(np.max(holding_days)) if holding_days else None

        # Rank diagnostics distributions from holding-day observations
        rank_avg = _mean(rank_holdings) if rank_holdings else None
        rank_med = _median(rank_holdings) if rank_holdings else None
        rank_p75 = _percentile(rank_holdings, 75.0) if rank_holdings else None
        rank_p90 = _percentile(rank_holdings, 90.0) if rank_holdings else None
        rank_p95 = _percentile(rank_holdings, 95.0) if rank_holdings else None
        rank_max = float(np.max(rank_holdings)) if rank_holdings else None

        attribution_summary = {
            "capital_attribution": {
                "avg_exposure_frac": avg_exposure,
                "avg_cash_frac": avg_cash_frac,
                "max_cash_frac": max_cash_frac,
                "min_cash_frac": min_cash_frac,
                "avg_invested_capital_frac": avg_exposure,
                "avg_num_holdings": avg_num_holdings,
                "avg_position_size": avg_pos_size,
            },
            "profit_attribution": {
                "gross_winning_pnl": gross_win if gross_win != 0 or positive_pnls else None,
                "gross_losing_pnl": gross_loss if negative_pnls else None,
                "net_strategy_pnl": net_pnl if pnl["closed_trade_realized_pnls"] else None,
                "largest_winning_position": largest_win,
                "largest_losing_position": largest_loss,
                "top10_winners_contribution_frac": top10_winners_contrib,
                "top10_losers_contribution_frac": top10_losers_contrib,
            },
            "trading_cost_attribution": {
                "total_commission_paid": pnl["total_commission_paid"],
                "total_slippage_paid": pnl["total_slippage_paid"],
                "commission_as_frac_of_gross_profit": commission_pct_gross_profit,
                "slippage_as_frac_of_gross_profit": slippage_pct_gross_profit,
            },
            "capital_efficiency": {
                "avg_daily_cash": avg_daily_cash,
                "avg_daily_invested_capital": avg_daily_invested,
                "idle_capital_days": idle_days,
                "capital_utilization_rate": cap_util_rate,
            },
            "holding_distribution": {
                "avg_holding_days": holding_avg,
                "median_holding_days": holding_med,
                "p75_holding_days": holding_p75,
                "p90_holding_days": holding_p90,
                "max_holding_days": holding_max,
            },
            "portfolio_distribution": {
                "avg_portfolio_size": avg_port_size,
                "median_portfolio_size": med_port_size,
                "max_portfolio_size": max_port_size,
                "min_portfolio_size": min_port_size,
            },
            "rank_attribution": {
                "avg_holding_rank": rank_avg,
                "median_holding_rank": rank_med,
                "p75_holding_rank": rank_p75,
                "p90_holding_rank": rank_p90,
                "p95_holding_rank": rank_p95,
                "max_holding_rank": rank_max,
                "rank_exit_candidates": rank_diagnostics_summary.get("rank_exit_candidates"),
                "ema_preempted_rank_exits": rank_diagnostics_summary.get("ema_preempted_rank_exit"),
            },
            # internal invariants used for validation printing
            "_internal": {"n_trading_days_recorded": len(total), "n_closed_trades": len(pnl["closed_trade_realized_pnls"])},
        }
        return attribution_summary

    def format_report(self, engine, attribution_summary: Dict[str, Any], *, validation: Dict[str, str]) -> str:
        c = attribution_summary["capital_attribution"]
        p = attribution_summary["profit_attribution"]
        tc = attribution_summary["trading_cost_attribution"]
        ce = attribution_summary["capital_efficiency"]
        hd = attribution_summary["holding_distribution"]
        pdist = attribution_summary["portfolio_distribution"]
        r = attribution_summary["rank_attribution"]

        def pct(x: Optional[float]) -> str:
            return _format_pct(x, digits=2) if x is not None else "n/a"

        def pct_from_frac(frac: Optional[float]) -> str:
            # frac already as 0..1
            return "n/a" if frac is None else f"{frac*100.0:.2f}%"

        lines: List[str] = []
        lines += ["=" * 58, "PERFORMANCE ATTRIBUTION REPORT", "=" * 58, ""]

        # 1. Capital Attribution
        lines += ["---", "## 1. Capital Attribution", "---"]
        lines.append(f"Average Exposure (%): {pct(c.get('avg_exposure_frac'))}")
        lines.append(f"Average Cash (%): {pct(c.get('avg_cash_frac'))}")
        lines.append(f"Average Invested Capital (%): {pct(c.get('avg_invested_capital_frac'))}")
        lines.append(f"Maximum Cash Allocation (%): {pct(c.get('max_cash_frac'))}")
        lines.append(f"Minimum Cash Allocation (%): {pct(c.get('min_cash_frac'))}")
        lines.append(f"Average Number of Holdings: {c.get('avg_num_holdings') if c.get('avg_num_holdings') is not None else 'n/a'}")
        lines.append(f"Average Position Size: {c.get('avg_position_size') if c.get('avg_position_size') is not None else 'n/a'}")
        lines.append("")

        # 2. Profit Attribution
        lines += ["---", "## 2. Profit Attribution", "---"]
        lines.append(f"Gross Winning PnL: {_format_money(p.get('gross_winning_pnl'))}")
        lines.append(f"Gross Losing PnL: {_format_money(p.get('gross_losing_pnl'))}")
        lines.append(f"Net Strategy PnL: {_format_money(p.get('net_strategy_pnl'))}")
        lines.append(f"Largest Winning Position: {_format_money(p.get('largest_winning_position'))}")
        lines.append(f"Largest Losing Position: {_format_money(p.get('largest_losing_position'))}")
        if p.get("top10_winners_contribution_frac") is not None:
            lines.append(
                f"Top 10 Winners Contribution (%): {pct_from_frac(p.get('top10_winners_contribution_frac'))}"
            )
        else:
            lines.append("Top 10 Winners Contribution (%): n/a")
        if p.get("top10_losers_contribution_frac") is not None:
            lines.append(
                f"Top 10 Losers Contribution (%): {pct_from_frac(p.get('top10_losers_contribution_frac'))}"
            )
        else:
            lines.append("Top 10 Losers Contribution (%): n/a")
        lines.append("")

        # 3. Trading Cost Attribution
        lines += ["---", "## 3. Trading Cost Attribution", "---"]
        lines.append(f"Total Commission Paid: {_format_money(tc.get('total_commission_paid'))}")
        lines.append(f"Total Slippage Paid: {_format_money(tc.get('total_slippage_paid'))}")
        lines.append(
            f"Commission as % of Gross Profit: {pct_from_frac(tc.get('commission_as_frac_of_gross_profit')) if tc.get('commission_as_frac_of_gross_profit') is not None else 'n/a'}"
        )
        lines.append(
            f"Slippage as % of Gross Profit: {pct_from_frac(tc.get('slippage_as_frac_of_gross_profit')) if tc.get('slippage_as_frac_of_gross_profit') is not None else 'n/a'}"
        )
        lines.append("")

        # 4. Capital Efficiency
        lines += ["---", "## 4. Capital Efficiency", "---"]
        lines.append(f"Average Daily Cash: {_format_money(ce.get('avg_daily_cash'))}")
        lines.append(f"Average Daily Invested Capital: {_format_money(ce.get('avg_daily_invested_capital'))}")
        lines.append(f"Idle Capital Days: {ce.get('idle_capital_days') if ce.get('idle_capital_days') is not None else 'n/a'}")
        lines.append(f"Capital Utilization Rate (%): {pct_from_frac(ce.get('capital_utilization_rate')) if ce.get('capital_utilization_rate') is not None else 'n/a'}")
        lines.append("")

        # 5. Holding Distribution
        lines += ["---", "## 5. Holding Distribution", "---"]
        lines.append(f"Average Holding Days: {_format_money(hd.get('avg_holding_days'), digits=2)}")
        lines.append(f"Median Holding Days: {_format_money(hd.get('median_holding_days'), digits=2)}")
        lines.append(f"P75 Holding Days: {_format_money(hd.get('p75_holding_days'), digits=2)}")
        lines.append(f"P90 Holding Days: {_format_money(hd.get('p90_holding_days'), digits=2)}")
        lines.append(f"Maximum Holding Days: {_format_money(hd.get('max_holding_days'), digits=2)}")
        lines.append("")

        # 6. Portfolio Distribution
        lines += ["---", "## 6. Portfolio Distribution", "---"]
        lines.append(f"Average Portfolio Size: {_format_money(pdist.get('avg_portfolio_size'), digits=2)}")
        lines.append(f"Median Portfolio Size: {_format_money(pdist.get('median_portfolio_size'), digits=2)}")
        lines.append(f"Maximum Portfolio Size: {_format_money(pdist.get('max_portfolio_size'), digits=2)}")
        lines.append(f"Minimum Portfolio Size: {_format_money(pdist.get('min_portfolio_size'), digits=2)}")
        lines.append("")

        # 7. Rank Attribution
        lines += ["---", "## 7. Rank Attribution", "---"]
        lines.append(f"Average Holding Rank: {_format_money(r.get('avg_holding_rank'), digits=2)}")
        lines.append(f"Median Holding Rank: {_format_money(r.get('median_holding_rank'), digits=2)}")
        lines.append(f"P75 Holding Rank: {_format_money(r.get('p75_holding_rank'), digits=2)}")
        lines.append(f"P90 Holding Rank: {_format_money(r.get('p90_holding_rank'), digits=2)}")
        lines.append(f"P95 Holding Rank: {_format_money(r.get('p95_holding_rank'), digits=2)}")
        lines.append(f"Maximum Holding Rank: {_format_money(r.get('max_holding_rank'), digits=2)}")
        lines.append(f"Rank Exit Candidates: {r.get('rank_exit_candidates') if r.get('rank_exit_candidates') is not None else 'n/a'}")
        lines.append(f"EMA Preempted Rank Exits: {r.get('ema_preempted_rank_exits') if r.get('ema_preempted_rank_exits') is not None else 'n/a'}")
        lines.append("")

        # 8. Performance Diagnosis (facts only)
        lines += ["", "==================================================", "PERFORMANCE DIAGNOSIS", "=================================================="]

        # Objective facts, no recommendations.
        avg_cash = c.get("avg_cash_frac")
        avg_invest = c.get("avg_invested_capital_frac")
        if avg_cash is not None:
            lines.append(f"Average cash allocation was {avg_cash*100.0:.2f}%.")
        if avg_invest is not None:
            lines.append(f"Average invested capital allocation was {avg_invest*100.0:.2f}%.")

        if p.get("top10_winners_contribution_frac") is not None:
            lines.append(
                f"Top 10 winners generated {p['top10_winners_contribution_frac']*100.0:.2f}% of total gross profit."
            )
        if p.get("top10_losers_contribution_frac") is not None:
            lines.append(
                f"Top 10 losers generated {p['top10_losers_contribution_frac']*100.0:.2f}% of total gross loss magnitude."
            )

        gross_profit = p.get("gross_winning_pnl")
        if gross_profit is not None and tc.get("total_commission_paid") is not None and gross_profit != 0:
            lines.append(
                f"Commission consumed {tc['total_commission_paid']/gross_profit*100.0:.4f}% of gross profit."
            )
        if gross_profit is not None and tc.get("total_slippage_paid") is not None and gross_profit != 0:
            lines.append(
                f"Slippage consumed {tc['total_slippage_paid']/gross_profit*100.0:.4f}% of gross profit."
            )

        if hd.get("avg_holding_days") is not None:
            lines.append(f"Average holding period was {hd['avg_holding_days']:.2f} trading days.")

        if ce.get("capital_utilization_rate") is not None:
            lines.append(
                f"Capital utilization averaged {ce['capital_utilization_rate']*100.0:.2f}%."
            )

        if r.get("median_holding_rank") is not None:
            lines.append(f"Median holding rank was {r['median_holding_rank']:.2f}.")
        if r.get("p95_holding_rank") is not None:
            lines.append(f"95% of held positions ranked within {r['p95_holding_rank']:.2f}.")
        if r.get("ema_preempted_rank_exits") is not None:
            lines.append(f"EMA exits preempted {r['ema_preempted_rank_exits']} potential rank exits.")

        lines.append("")

        # Validation block
        lines += [
            "",
            "==================================================",
            "PERFORMANCE ATTRIBUTION VALIDATION",
            "==================================================",
            f"Trading logic modified: {validation.get('trading_logic_modified')}",
            f"Trade count unchanged: {validation.get('trade_count_unchanged')}",
            f"Orders unchanged: {validation.get('orders_unchanged')}",
            f"Portfolio construction unchanged: {validation.get('portfolio_unchanged')}",
            f"Performance metrics unchanged: {validation.get('performance_metrics_unchanged')}",
            f"Observation only: {validation.get('observation_only')}",
            "",
        ]

        report_text = "\n".join(lines).rstrip() + "\n"
        return report_text

    def build_and_validate(
        self,
        engine,
        *,
        equity_df,
        trades_df,
        kpi: Dict[str, Any],
        rank_diagnostics_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Capture invariants pre-build
        equity_hash_before = _sha_equity(equity_df)
        trades_count_before = int(len(trades_df)) if trades_df is not None else 0
        closed_trades_len_before = int(len(getattr(engine.trade_journal, "closed", []) or []))
        rebalance_events_len_before = int(len(getattr(engine.trade_journal, "rebalance_events", []) or []))
        pending_orders_len_before = int(len(getattr(engine, "pending_orders", []) or []))
        portfolio_keys_before = set(getattr(engine, "portfolio", {}).keys())
        cash_before = float(getattr(engine, "cash", 0.0))

        inputs = AttributionInputs(kpi=kpi, trades_df=trades_df, equity_df=equity_df)
        summary = self.build(
            engine,
            rank_diagnostics_summary=rank_diagnostics_summary,
            attribution_inputs=inputs,
        )

        equity_hash_after = _sha_equity(equity_df)
        trades_count_after = int(len(trades_df)) if trades_df is not None else 0
        closed_trades_len_after = int(len(getattr(engine.trade_journal, "closed", []) or []))
        rebalance_events_len_after = int(len(getattr(engine.trade_journal, "rebalance_events", []) or []))
        pending_orders_len_after = int(len(getattr(engine, "pending_orders", []) or []))
        portfolio_keys_after = set(getattr(engine, "portfolio", {}).keys())
        cash_after = float(getattr(engine, "cash", 0.0))

        # Performance metrics unchanged: compare recomputed KPI values to original kpi
        # (kpi is deterministic from equity and trades).
        perf_metrics_ok = True
        try:
            from engine.research.kpi_report import build_hierarchical_kpi_report

            # recompute
            bm_rets = None
            if engine.BENCHMARK_TICKER in getattr(engine, "close_m", pd.DataFrame()).columns:
                bm = engine.close_m[engine.BENCHMARK_TICKER].reindex(equity_df.index)
                bm_rets = bm.pct_change()
            kpi_re = build_hierarchical_kpi_report(equity_df, trades_df, benchmark_returns=bm_rets)

            def _get(d, path):
                cur = d
                for p in path:
                    cur = cur.get(p, {})
                return cur if not isinstance(cur, dict) else None

            # required metrics
            checks = [
                ("return", "total_return"),
                ("risk", "cagr"),
                ("risk", "mdd"),
                ("risk", "sharpe"),
                ("risk", "sortino"),
                ("risk", "calmar"),
                ("return", "win_rate"),
                ("return", "profit_factor"),
            ]
            for family, key in checks:
                v0 = kpi.get(family, {}).get(key)
                v1 = kpi_re.get(family, {}).get(key)
                if v0 is None and v1 is None:
                    continue
                if v0 is None or v1 is None:
                    perf_metrics_ok = False
                    break
                if isinstance(v0, float) and isinstance(v1, float):
                    if abs(v0 - v1) > 1e-10:
                        perf_metrics_ok = False
                        break
        except Exception:
            # If recomputation fails, mark as FAIL because strict requirement.
            perf_metrics_ok = False

        trading_logic_modified = equity_hash_before != equity_hash_after or cash_before != cash_after
        trade_count_unchanged = trades_count_before == trades_count_after and closed_trades_len_before == closed_trades_len_after
        orders_unchanged = pending_orders_len_before == pending_orders_len_after and rebalance_events_len_before == rebalance_events_len_after
        portfolio_unchanged = portfolio_keys_before == portfolio_keys_after
        observation_only = (
            not trading_logic_modified
            and trade_count_unchanged
            and orders_unchanged
            and portfolio_unchanged
        )

        validation = {
            "trading_logic_modified": "PASS" if not trading_logic_modified else "FAIL",
            "trade_count_unchanged": "PASS" if trade_count_unchanged else "FAIL",
            "orders_unchanged": "PASS" if orders_unchanged else "FAIL",
            "portfolio_unchanged": "PASS" if portfolio_unchanged else "FAIL",
            "performance_metrics_unchanged": "PASS" if perf_metrics_ok else "FAIL",
            "observation_only": "PASS" if observation_only else "FAIL",
        }

        report_txt = self.format_report(
            engine,
            summary,
            validation=validation,
        )

        return {
            "summary": summary,
            "report_txt": report_txt,
            "validation": validation,
        }

