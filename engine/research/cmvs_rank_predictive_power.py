"""Observation-only CMVS rank predictive power report.

For each monthly rebalance date:
  1. Build the ranked CMVS universe using existing engine methods.
  2. Split the ranked universe into equal-sized groups (deciles by default).
  3. Compute forward-return / risk statistics for each group.

Aggregates across the backtest and prints an end-of-run report.
Never changes trading decisions, entry, exit, sizing, or execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_mean(values: List[float]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _safe_median(values: List[float]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _safe_ratio(num: float, den: float) -> Optional[float]:
    if den <= 0:
        return None
    return float(num / den)


def _safe_spearman(x: pd.Series, y: pd.Series) -> Optional[float]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return None
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    if float(xr.std(ddof=0) or 0.0) == 0.0 or float(yr.std(ddof=0) or 0.0) == 0.0:
        return None
    v = xr.corr(yr, method="pearson")
    if pd.isna(v):
        return None
    return float(v)


def _group_label(group_idx: int, n_groups: int) -> str:
    if n_groups == 10:
        if group_idx == 0:
            return "Top10%"
        if group_idx == n_groups - 1:
            return "Bottom10%"
        lo = group_idx * 10
        hi = (group_idx + 1) * 10
        return f"{lo}-{hi}%"
    # generic fallback
    return f"Group {group_idx + 1}"


def _assign_equal_groups(n_items: int, n_groups: int) -> np.ndarray:
    """Assign positions 0..n-1 into near-equal ordered groups."""
    if n_items <= 0:
        return np.asarray([], dtype=int)
    group_ids = np.floor(np.arange(n_items) * n_groups / n_items).astype(int)
    return np.clip(group_ids, 0, n_groups - 1)


def _forward_path_metrics(prices: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute path metrics over forward window from inclusive price path.

    prices[0] = start price, prices[-1] = end of horizon.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size < 2 or not np.isfinite(prices[0]) or prices[0] <= 0:
        return {
            "forward_return": None,
            "max_drawdown": None,
            "max_upside": None,
            "volatility": None,
        }
    start = float(prices[0])
    rel = prices / start - 1.0
    forward_return = float(rel[-1]) if np.isfinite(rel[-1]) else None
    max_drawdown = float(np.nanmin(rel)) if np.isfinite(rel).any() else None
    max_upside = float(np.nanmax(rel)) if np.isfinite(rel).any() else None
    rets = pd.Series(prices).pct_change().dropna()
    volatility = float(rets.std(ddof=0)) if len(rets) > 0 else None
    return {
        "forward_return": forward_return,
        "max_drawdown": max_drawdown,
        "max_upside": max_upside,
        "volatility": volatility,
    }


@dataclass
class CMVSRankPredictivePowerReport:
    n_groups: int = 10
    horizons: Dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.horizons is None:
            self.horizons = {"5D": 5, "10D": 10, "20D": 20, "60D": 60}

    def build(self, engine: Any) -> Dict[str, Any]:
        trading_days = engine.close_m.index
        warmup = max(engine.MOM_WINDOW, engine.LOWVOL_WINDOW) + 5
        rebalance_days = set(
            pd.to_datetime(
                engine.close_m.groupby(engine.close_m.index.to_period("M")).apply(lambda x: x.index[0]).values
            )
        )

        groups: Dict[int, Dict[str, Any]] = {
            i: {
                "label": _group_label(i, self.n_groups),
                "n_stocks": 0,
                "forward_returns": {k: [] for k in self.horizons},
                "median_return_20d": [],
                "win_rate_20d": {"wins": 0, "count": 0},
                "max_drawdown_20d": [],
                "max_upside_20d": [],
                "volatility_20d": [],
            }
            for i in range(self.n_groups)
        }
        monthly_rank_ic_20d: List[float] = []
        processed_rebalance_dates = 0

        for idx, current_date in enumerate(trading_days):
            if idx < warmup or current_date not in rebalance_days:
                continue

            current_closes = engine.close_m.loc[current_date]
            active_symbols = [
                s
                for s in engine.close_m.columns
                if s != engine.BENCHMARK_TICKER
                and pd.notna(current_closes.get(s))
                and not engine.profile_meta.get(s, {}).get("isEtf", False)
            ]
            if not active_symbols:
                continue

            regime = engine.determine_market_regime(idx, active_symbols)
            if regime is None or float(regime.exposure) == 0.0:
                continue

            top_adv_symbols = (
                engine.dvol_m.loc[current_date, active_symbols]
                .nlargest(min(engine.TOP_ADV_POOL, len(active_symbols)))
                .index.tolist()
            )
            universe_symbols = engine.get_universe(top_adv_symbols, idx, quiet=True)
            factor_df = engine.build_factors(idx, universe_symbols, quiet=True)
            if factor_df.empty:
                continue
            ranked_df = engine.rank_universe(factor_df)
            if ranked_df.empty:
                continue

            processed_rebalance_dates += 1
            symbols = ranked_df["symbol"].astype(str).tolist()
            scores = ranked_df["final_score"].astype(float).reset_index(drop=True)
            group_ids = _assign_equal_groups(len(ranked_df), self.n_groups)

            # Monthly Spearman Rank IC on 20D future returns.
            future_20_idx = idx + int(self.horizons["20D"])
            if future_20_idx < len(trading_days):
                now_px = engine.close_m.loc[current_date, symbols].astype(float)
                fut_px = engine.close_m.iloc[future_20_idx].reindex(symbols).astype(float)
                fut20 = fut_px / now_px - 1.0
                ic20 = _safe_spearman(scores, fut20.reset_index(drop=True))
                if ic20 is not None:
                    monthly_rank_ic_20d.append(ic20)

            for row_pos, sym in enumerate(symbols):
                g = int(group_ids[row_pos])
                groups[g]["n_stocks"] += 1

                start_px = engine.close_m.loc[current_date].get(sym, np.nan)
                if pd.isna(start_px) or float(start_px) <= 0:
                    continue

                # Forward return stats for each requested horizon.
                for horizon_name, horizon_days in self.horizons.items():
                    future_idx = idx + int(horizon_days)
                    if future_idx >= len(trading_days):
                        continue
                    end_px = engine.close_m.iloc[future_idx].get(sym, np.nan)
                    if pd.isna(end_px) or float(end_px) <= 0:
                        continue
                    ret = float(end_px / start_px - 1.0)
                    groups[g]["forward_returns"][horizon_name].append(ret)

                # 20D path metrics
                end20_idx = idx + int(self.horizons["20D"])
                if end20_idx < len(trading_days):
                    path = engine.close_m[sym].iloc[idx : end20_idx + 1].to_numpy(dtype=float)
                    pm = _forward_path_metrics(path)
                    r20 = pm["forward_return"]
                    if r20 is not None:
                        groups[g]["median_return_20d"].append(r20)
                        groups[g]["win_rate_20d"]["count"] += 1
                        if r20 > 0:
                            groups[g]["win_rate_20d"]["wins"] += 1
                    if pm["max_drawdown"] is not None:
                        groups[g]["max_drawdown_20d"].append(pm["max_drawdown"])
                    if pm["max_upside"] is not None:
                        groups[g]["max_upside_20d"].append(pm["max_upside"])
                    if pm["volatility"] is not None:
                        groups[g]["volatility_20d"].append(pm["volatility"])

        group_summary: List[Dict[str, Any]] = []
        avg20_order: List[Optional[float]] = []
        for i in range(self.n_groups):
            g = groups[i]
            row = {
                "group": g["label"],
                "n_stocks": int(g["n_stocks"]),
                "avg_forward_5d_return": _safe_mean(g["forward_returns"]["5D"]),
                "avg_forward_10d_return": _safe_mean(g["forward_returns"]["10D"]),
                "avg_forward_20d_return": _safe_mean(g["forward_returns"]["20D"]),
                "avg_forward_60d_return": _safe_mean(g["forward_returns"]["60D"]),
                "median_return_20d": _safe_median(g["median_return_20d"]),
                "win_rate_20d": _safe_ratio(
                    float(g["win_rate_20d"]["wins"]),
                    float(g["win_rate_20d"]["count"]),
                ),
                "avg_max_drawdown_20d": _safe_mean(g["max_drawdown_20d"]),
                "avg_max_upside_20d": _safe_mean(g["max_upside_20d"]),
                "avg_volatility_20d": _safe_mean(g["volatility_20d"]),
            }
            avg20_order.append(row["avg_forward_20d_return"])
            group_summary.append(row)

        monotonic_pass = True
        previous = None
        comparable_steps = 0
        for v in avg20_order:
            if v is None:
                continue
            if previous is not None:
                comparable_steps += 1
                if previous < v:
                    monotonic_pass = False
            previous = v
        if comparable_steps == 0:
            monotonic_pass = False

        top20 = group_summary[0]["avg_forward_20d_return"] if group_summary else None
        bottom20 = group_summary[-1]["avg_forward_20d_return"] if group_summary else None
        spread_20d = None
        if top20 is not None and bottom20 is not None:
            spread_20d = float(top20 - bottom20)

        avg_rank_ic = _safe_mean(monthly_rank_ic_20d)
        med_rank_ic = _safe_median(monthly_rank_ic_20d)
        positive_ic_ratio = None
        if monthly_rank_ic_20d:
            positive_ic_ratio = float(np.mean(np.asarray(monthly_rank_ic_20d, dtype=float) > 0))

        # Observation-only conclusion.
        if (
            avg_rank_ic is not None
            and avg_rank_ic > 0
            and positive_ic_ratio is not None
            and positive_ic_ratio >= 0.5
            and spread_20d is not None
            and spread_20d > 0
            and monotonic_pass
        ):
            conclusion = (
                "CMVS demonstrates statistically meaningful predictive power because "
                "higher-ranked stocks consistently outperform lower-ranked stocks on this sample."
            )
        else:
            conclusion = (
                "CMVS shows weak ranking ability on this sample because higher-ranked stocks "
                "do not consistently outperform lower-ranked stocks."
            )

        return {
            "n_groups": self.n_groups,
            "horizons": self.horizons,
            "n_rebalance_dates": processed_rebalance_dates,
            "group_summary": group_summary,
            "monotonicity": {
                "status": "PASS" if monotonic_pass else "FAIL",
                "checked_metric": "avg_forward_20d_return",
            },
            "spread_top_bottom_20d": spread_20d,
            "rank_ic_20d": {
                "monthly_values": monthly_rank_ic_20d,
                "average": avg_rank_ic,
                "median": med_rank_ic,
                "positive_ratio": positive_ic_ratio,
                "positive_months": int(sum(1 for x in monthly_rank_ic_20d if x > 0)),
                "total_months": int(len(monthly_rank_ic_20d)),
            },
            "conclusion": conclusion,
        }

    def format_report(self, summary: Dict[str, Any]) -> str:
        def pct(v: Optional[float]) -> str:
            if v is None:
                return "n/a"
            return f"{100.0 * float(v):.2f}%"

        lines = [
            "=" * 52,
            "CMVS RANK PREDICTIVE POWER",
            "=" * 52,
            f"Monthly rebalance dates processed: {summary.get('n_rebalance_dates', 0)}",
            "",
            f"{'Group':12s} {'Count':>8s} {'Avg20DReturn':>14s} {'WinRate':>12s} {'AvgDrawdown':>14s}",
        ]
        for row in summary.get("group_summary") or []:
            lines.append(
                f"{row['group']:12s} "
                f"{int(row['n_stocks']):8d} "
                f"{pct(row['avg_forward_20d_return']):>14s} "
                f"{pct(row['win_rate_20d']):>12s} "
                f"{pct(row['avg_max_drawdown_20d']):>14s}"
            )

        lines += [
            "",
            "=" * 52,
            "Monotonicity Check",
            "=" * 52,
            f"Monotonic = {summary['monotonicity']['status']}",
            "",
            f"Spread (Top10% - Bottom10%): {pct(summary.get('spread_top_bottom_20d'))}",
            "",
            f"Average Rank IC: {summary['rank_ic_20d']['average'] if summary['rank_ic_20d']['average'] is not None else 'n/a'}",
            f"Median Rank IC: {summary['rank_ic_20d']['median'] if summary['rank_ic_20d']['median'] is not None else 'n/a'}",
            f"Positive Months: {summary['rank_ic_20d']['positive_months']}/{summary['rank_ic_20d']['total_months']}",
            f"Positive IC ratio: {pct(summary['rank_ic_20d']['positive_ratio'])}",
            "",
            "Conclusion:",
            "",
            summary["conclusion"],
        ]

        lines += [
            "",
            "Detailed aggregated group statistics:",
        ]
        for row in summary.get("group_summary") or []:
            lines += [
                f"- {row['group']}:",
                f"    n_stocks={row['n_stocks']}",
                f"    avg_forward_5d_return={pct(row['avg_forward_5d_return'])}",
                f"    avg_forward_10d_return={pct(row['avg_forward_10d_return'])}",
                f"    avg_forward_20d_return={pct(row['avg_forward_20d_return'])}",
                f"    avg_forward_60d_return={pct(row['avg_forward_60d_return'])}",
                f"    median_return_20d={pct(row['median_return_20d'])}",
                f"    win_rate_20d={pct(row['win_rate_20d'])}",
                f"    avg_max_drawdown_20d={pct(row['avg_max_drawdown_20d'])}",
                f"    avg_max_upside_20d={pct(row['avg_max_upside_20d'])}",
                f"    avg_volatility_20d={pct(row['avg_volatility_20d'])}",
            ]
        return "\n".join(lines)

    def write_report(self, summary: Dict[str, Any], path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_report(summary), encoding="utf-8")
        return path

