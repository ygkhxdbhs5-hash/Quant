"""Generic Strategy #2+ family engine (isolated from Strategy #1 signal code)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from engine.baseline_engine import BaselineEngineV1
from engine.research.kpi_report import build_hierarchical_kpi_report
from engine.s2_family_signals import (
    TOP_K,
    TOP_LIQUID,
    liquid_pool,
    top_k_from_scores,
    with_quality_half,
)
from engine.strategy import load_config
from engine.strategy_baseline_v1 import equal_weight_targets


class S2FamilyEngine(BaselineEngineV1):
    """Monthly/daily long-only book driven by an external score function."""

    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: str = "config/config.yaml",
        *,
        variant_id: str,
        family_id: int,
        score_fn: Optional[Callable] = None,
        cadence: str = "monthly",
        mode: str = "rank",
        extra: Optional[dict] = None,
        si_panel: Optional[pd.DataFrame] = None,
        div_events: Optional[pd.DataFrame] = None,
    ):
        cfg = dict(config or load_config(config_path))
        cfg["enable_regime_exposure"] = False
        cfg["enable_regime_exposure_fast"] = False
        cfg["enable_industry_neutral_ranking"] = False
        cfg["fixed_leverage"] = 1.0
        cfg["enable_cash_interest"] = False
        cfg["enable_quality_factor"] = True  # load PIT
        cfg["enable_topup_chasing"] = False
        cfg.pop("disable_topup_chasing", None)
        cfg.setdefault("cost_model", "corwin_schultz_v2")
        cfg.setdefault("winsorize_adv", True)

        self.variant_id = str(variant_id)
        self.family_id = int(family_id)
        self.score_fn = score_fn
        self.cadence = cadence
        self.mode = mode
        self.extra = dict(extra or {})
        self.si_panel = si_panel
        self.div_events = div_events  # columns: ticker, ex_dividend_date

        super().__init__(config=cfg, config_path=config_path)
        self.GROSS_EXPOSURE = 1.0
        self.TOP_K = int(TOP_K)
        self.current_regime_exposure = 1.0
        self._prev_target_set: Set[str] = set()

    def _active_strategy_id(self) -> str:
        return f"s2_fam{self.family_id}_{self.variant_id}"

    def _active_strategy_knobs(self) -> Dict[str, object]:
        return {
            "strategy_id": self._active_strategy_id(),
            "family_id": self.family_id,
            "variant_id": self.variant_id,
            "cadence": self.cadence,
            "mode": self.mode,
            "extra": self.extra,
            "TOP_K": self.TOP_K,
            "GROSS_EXPOSURE": 1.0,
            "leverage": "1.0x_always",
            "fill_policy": "no_topup_chasing_default",
            "cost_model": self.COST_MODEL,
        }

    def _month_starts(self) -> Set[pd.Timestamp]:
        idx = self.close_m.index
        months = idx.to_period("M")
        return set(pd.to_datetime(idx[~months.duplicated(keep="first")]))

    def _eligible(self, idx: int) -> List[str]:
        return [
            s
            for s in self._investable
            if s in self.close_m.columns and pd.notna(self.close_m[s].iloc[idx])
        ]

    def _select_rank(self, idx: int, live: List[str]) -> List[str]:
        as_of = self.close_m.index[idx]
        liquid = liquid_pool(self.dvol_m, as_of, live, n=TOP_LIQUID)
        if not liquid or self.score_fn is None:
            return []
        scores = self.score_fn(
            self.close_m,
            self.dvol_m,
            self.fundamental_history,
            self._fund_ts,
            as_of,
            liquid,
            si_panel=self.si_panel,
        )
        if self.extra.get("quality_half"):
            scores = with_quality_half(
                scores, self.fundamental_history, self._fund_ts, as_of
            )
        return top_k_from_scores(
            scores,
            k=self.TOP_K,
            extreme_frac=self.extra.get("extreme_frac"),
        )

    def _select_tom(self, idx: int, live: List[str]) -> List[str]:
        """Invested only on turn-of-month sessions; else empty (cash)."""
        as_of = self.close_m.index[idx]
        # trading-day position within calendar month
        month = as_of.to_period("M")
        month_days = self.close_m.index[self.close_m.index.to_period("M") == month]
        pos = int(month_days.get_loc(as_of)) if as_of in month_days else -1
        n = len(month_days)
        last_n = int(self.extra.get("last", 1))
        first_n = int(self.extra.get("first", 3))
        in_window = (pos >= 0 and pos < first_n) or (pos >= n - last_n and last_n > 0)
        if not in_window:
            return []
        # Hold liquid top-K by dollar volume (market proxy sleeve when invested)
        as_of = self.close_m.index[idx]
        liquid = liquid_pool(self.dvol_m, as_of, live, n=TOP_LIQUID)
        dvol = pd.to_numeric(self.dvol_m.loc[as_of, liquid], errors="coerce").dropna()
        return dvol.nlargest(min(self.TOP_K, len(dvol))).index.tolist()

    def _select_exdiv(self, idx: int, live: List[str]) -> List[str]:
        if self.div_events is None or self.div_events.empty:
            return []
        as_of = pd.Timestamp(self.close_m.index[idx])
        pre = int(self.extra.get("pre", 5))
        post = int(self.extra.get("post", 5))
        # map trading-day offsets via index
        loc = self.close_m.index.get_loc(as_of)
        i = int(loc)
        # For each ticker, check if some ex-div date is within [i-pre, i+post] trading days
        # Build set of tickers whose ex-div trading index is near i
        ev = self.div_events
        # precompute: filter events with ex dates in panel
        hits = []
        # narrow candidate events by calendar window first
        cal_lo = as_of - pd.Timedelta(days=pre * 2 + 5)
        cal_hi = as_of + pd.Timedelta(days=post * 2 + 5)
        sub = ev[(ev["ex_dividend_date"] >= cal_lo) & (ev["ex_dividend_date"] <= cal_hi)]
        if sub.empty:
            return []
        for _, row in sub.iterrows():
            sym = str(row["ticker"])
            if sym not in live or sym not in self.close_m.columns:
                continue
            ex = pd.Timestamp(row["ex_dividend_date"])
            # find nearest trading day on/after ex
            j = int(self.close_m.index.searchsorted(ex))
            if j >= len(self.close_m.index):
                continue
            if abs(i - j) <= max(pre, post):
                # more precise: require -pre <= i-j <= post
                delta = i - j
                if -pre <= delta <= post:
                    hits.append(sym)
        if not hits:
            return []
        # equal-weight among hit names, cap at TOP_K by dvol
        as_of = self.close_m.index[idx]
        dvol = pd.to_numeric(
            self.dvol_m.loc[as_of, [s for s in hits if s in self.dvol_m.columns]],
            errors="coerce",
        ).dropna()
        return dvol.nlargest(min(self.TOP_K, len(dvol))).index.tolist()

    def _select(self, idx: int, live: List[str]) -> List[str]:
        if self.mode == "calendar_tom":
            return self._select_tom(idx, live)
        if self.mode == "event_div":
            return self._select_exdiv(idx, live)
        return self._select_rank(idx, live)

    def _emit_reports(self, equity: pd.DataFrame, qqq_eq: pd.DataFrame) -> dict:
        out = Path("cache") / "s2_families" / self.variant_id
        out.mkdir(parents=True, exist_ok=True)
        trades = self.trade_journal.to_frame()
        trades_path = out / "trade_journal.csv"
        if not trades.empty:
            trades.to_csv(trades_path, index=False)
        else:
            trades_path.write_text("", encoding="utf-8")
        bm_rets = (
            qqq_eq["Total_Equity"].pct_change()
            if not qqq_eq.empty and "Total_Equity" in qqq_eq.columns
            else None
        )
        kpi = build_hierarchical_kpi_report(equity, trades, benchmark_returns=bm_rets)
        qqq_kpi = build_hierarchical_kpi_report(qqq_eq, pd.DataFrame(), benchmark_returns=None)
        sc = (kpi.get("risk") or {}).get("cagr")
        ss = (kpi.get("risk") or {}).get("sharpe")
        sm = (kpi.get("risk") or {}).get("mdd")
        qc = (qqq_kpi.get("risk") or {}).get("cagr")
        qs = (qqq_kpi.get("risk") or {}).get("sharpe")
        qm = (qqq_kpi.get("risk") or {}).get("mdd")
        alpha = float(sc) - float(qc) if sc is not None and qc is not None else None
        comparison = {
            "strategy": {"CAGR": sc, "Sharpe": ss, "Maximum_Drawdown": sm},
            "QQQ": {"CAGR": qc, "Sharpe": qs, "Maximum_Drawdown": qm},
            "Alpha_CAGR": alpha,
            "knobs": self._active_strategy_knobs(),
            "window": {"start": str(self.START_DATE), "end": str(self.END_DATE)},
        }
        if not equity.empty:
            equity.to_csv(out / "equity_curve.csv")
        (out / "benchmark_comparison.json").write_text(
            json.dumps(comparison, indent=2, default=str), encoding="utf-8"
        )
        return {
            "kpi": kpi,
            "qqq_kpi": qqq_kpi,
            "comparison": comparison,
            "trades_path": str(trades_path),
            "out_dir": str(out),
        }

    def run(self) -> pd.DataFrame:
        print(f"[S2 FAMILY] {self._active_strategy_id()} mode={self.mode}")
        month_starts = self._month_starts()
        for idx, current_date in enumerate(self.close_m.index):
            in_window = current_date >= self._run_start
            self._handle_delistings(idx)
            self.execute_pending_orders(idx)
            self._maybe_buy_qqq(idx)

            if not in_window:
                continue

            live = self._eligible(idx)
            if self.mode in ("calendar_tom", "event_div"):
                selected = self._select(idx, live)
                target_set = set(selected)
                # Rebalance only when membership changes (avoid daily churn/costs).
                do_reb = target_set != self._prev_target_set
            else:
                do_reb = current_date in month_starts
                selected = self._select(idx, live) if do_reb else None
                target_set = set(selected) if selected is not None else self._prev_target_set

            if do_reb:
                targets = (
                    equal_weight_targets(list(target_set), exposure=1.0)
                    if target_set
                    else {}
                )
                # One-shot resize on membership/month change; not mid-period chase.
                self._queue_rebalance_to_targets(
                    targets, idx, allow_exposure_resize=True
                )
                self._prev_target_set = set(target_set)

            closes = self.close_m.loc[current_date]
            if self.portfolio:
                held = list(self.portfolio.keys())
                px = closes.reindex(held)
                qty = pd.Series({s: self.portfolio[s] for s in held}, dtype=float)
                pv = float((qty * px).sum(skipna=True))
            else:
                pv = 0.0
            total_equity = self.cash + pv
            self.equity_curve.append({"Date": current_date, "Total_Equity": total_equity})
            self.diag_equity_curve.append(
                {
                    "Date": current_date,
                    "Net_Equity": total_equity,
                    "Gross_Equity": total_equity + float(self.diag_total_cost_dollars),
                    "Cum_Cost_Dollars": float(self.diag_total_cost_dollars),
                }
            )
            self._mark_qqq(idx)

        equity = (
            pd.DataFrame(self.equity_curve).set_index("Date")
            if self.equity_curve
            else pd.DataFrame()
        )
        qqq_eq = (
            pd.DataFrame(self.qqq_equity_curve).set_index("Date")
            if self.qqq_equity_curve
            else pd.DataFrame()
        )
        self.research_artifacts = self._emit_reports(equity, qqq_eq)
        return equity
