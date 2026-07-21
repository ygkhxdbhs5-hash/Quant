"""Observation-only CMVS Information Coefficient report.

Measures monthly Spearman Rank IC between CMVS final_score and forward returns
for 5D / 10D / 20D / 60D horizons on each rebalance date.

Never changes ranking, portfolio construction, entries, exits, sizing, or
execution. Generates diagnostics only.
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


def _safe_std(values: List[float]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(arr.std(ddof=0))


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


def _rolling_average(values: List[float], window: int) -> List[Optional[float]]:
    if not values:
        return []
    out: List[Optional[float]] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(_safe_mean(chunk))
    return out


@dataclass
class CMVSInformationCoefficientReport:
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

        monthly_dates: List[str] = []
        ic_by_horizon: Dict[str, List[float]] = {k: [] for k in self.horizons}

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

            symbols = ranked_df["symbol"].astype(str).tolist()
            scores = ranked_df["final_score"].astype(float).reset_index(drop=True)
            month_has_any = False

            for horizon_name, horizon_days in self.horizons.items():
                future_idx = idx + int(horizon_days)
                if future_idx >= len(trading_days):
                    continue
                now_px = engine.close_m.loc[current_date, symbols].astype(float)
                fut_px = engine.close_m.iloc[future_idx].reindex(symbols).astype(float)
                future_return = (fut_px / now_px - 1.0).reset_index(drop=True)
                ic = _safe_spearman(scores, future_return)
                if ic is not None:
                    ic_by_horizon[horizon_name].append(ic)
                    month_has_any = True
                else:
                    # Keep alignment when insufficient data for a horizon in an otherwise valid month.
                    ic_by_horizon[horizon_name].append(np.nan)
                    month_has_any = True

            if month_has_any:
                monthly_dates.append(str(current_date.date()))

        horizon_summary: Dict[str, Any] = {}
        highest_monthly_ic = {"horizon": None, "date": None, "value": None}
        lowest_monthly_ic = {"horizon": None, "date": None, "value": None}

        for horizon_name in self.horizons:
            raw_vals = ic_by_horizon[horizon_name]
            finite_vals = [float(v) for v in raw_vals if v is not None and np.isfinite(v)]
            avg = _safe_mean(finite_vals)
            med = _safe_median(finite_vals)
            std = _safe_std(finite_vals)
            pos = int(sum(1 for v in finite_vals if v > 0))
            neg = int(sum(1 for v in finite_vals if v < 0))
            rolling = _rolling_average(finite_vals, 12)
            ir = None if avg is None or std in (None, 0.0) else float(avg / std)

            # Match months to finite values only for extrema / rolling display.
            finite_pairs = [
                (monthly_dates[i], float(v))
                for i, v in enumerate(raw_vals)
                if v is not None and np.isfinite(v)
            ]
            if finite_pairs:
                hi_date, hi_val = max(finite_pairs, key=lambda x: x[1])
                lo_date, lo_val = min(finite_pairs, key=lambda x: x[1])
                if highest_monthly_ic["value"] is None or hi_val > highest_monthly_ic["value"]:
                    highest_monthly_ic = {"horizon": horizon_name, "date": hi_date, "value": hi_val}
                if lowest_monthly_ic["value"] is None or lo_val < lowest_monthly_ic["value"]:
                    lowest_monthly_ic = {"horizon": horizon_name, "date": lo_date, "value": lo_val}

            horizon_summary[horizon_name] = {
                "monthly_ic_values": finite_vals,
                "average_ic": avg,
                "median_ic": med,
                "std_ic": std,
                "positive_ic_months": pos,
                "negative_ic_months": neg,
                "rolling_12m_average_ic": rolling,
                "ic_information_ratio": ir,
            }

        # Observation-only automatic conclusion.
        stable_horizons = 0
        unstable_horizons = 0
        for horizon_name, row in horizon_summary.items():
            avg = row["average_ic"]
            ir = row["ic_information_ratio"]
            if avg is not None and avg > 0 and ir is not None and ir > 0.25:
                stable_horizons += 1
            else:
                unstable_horizons += 1
        if stable_horizons >= 3:
            conclusion = "CMVS demonstrates stable predictive power."
        else:
            conclusion = "Predictive power is unstable across market environments."

        return {
            "horizons": self.horizons,
            "n_rebalance_dates": len(monthly_dates),
            "monthly_dates": monthly_dates,
            "summary_by_horizon": horizon_summary,
            "highest_monthly_ic": highest_monthly_ic,
            "lowest_monthly_ic": lowest_monthly_ic,
            "conclusion": conclusion,
        }

    def format_report(self, summary: Dict[str, Any]) -> str:
        def num(v: Optional[float]) -> str:
            if v is None:
                return "n/a"
            return f"{float(v):.6f}"

        lines = [
            "=" * 50,
            "CMVS INFORMATION COEFFICIENT REPORT",
            "=" * 50,
            f"Monthly rebalance dates processed: {summary.get('n_rebalance_dates', 0)}",
            "",
        ]

        for horizon_name in ("5D", "10D", "20D", "60D"):
            row = (summary.get("summary_by_horizon") or {}).get(horizon_name) or {}
            lines += [
                f"{horizon_name} Horizon",
                f"Average IC: {num(row.get('average_ic'))}",
                f"Median IC: {num(row.get('median_ic'))}",
                f"Std IC: {num(row.get('std_ic'))}",
                f"Positive IC Months: {row.get('positive_ic_months', 0)}",
                f"Negative IC Months: {row.get('negative_ic_months', 0)}",
                "",
            ]

        hi = summary.get("highest_monthly_ic") or {}
        lo = summary.get("lowest_monthly_ic") or {}
        lines += [
            "=" * 50,
            f"Highest Monthly IC: {num(hi.get('value'))} ({hi.get('horizon')}, {hi.get('date')})",
            f"Lowest Monthly IC: {num(lo.get('value'))} ({lo.get('horizon')}, {lo.get('date')})",
            "",
            "Rolling 12-month Average IC",
        ]
        for horizon_name in ("5D", "10D", "20D", "60D"):
            row = (summary.get("summary_by_horizon") or {}).get(horizon_name) or {}
            rolling = row.get("rolling_12m_average_ic") or []
            tail = rolling[-1] if rolling else None
            lines.append(f"{horizon_name}: latest={num(tail)} full_series={rolling}")
        lines += [
            "",
            "IC Information Ratio",
        ]
        for horizon_name in ("5D", "10D", "20D", "60D"):
            row = (summary.get("summary_by_horizon") or {}).get(horizon_name) or {}
            lines.append(f"{horizon_name}: {num(row.get('ic_information_ratio'))}")

        lines += [
            "",
            "Conclusion:",
            "",
            str(summary.get("conclusion") or ""),
        ]
        return "\n".join(lines)

    def write_report(self, summary: Dict[str, Any], path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_report(summary), encoding="utf-8")
        return path

