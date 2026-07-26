"""Strategy #2 engine — weekly short-term reversal (isolated from Strategy #1).

Reuses BaselineEngineV1 infrastructure (panels, Fixed CS v2, participation caps,
no-chase fills, QQQ B&H, reporting helpers) via subclassing, but overrides the
daily loop for weekly evaluation + time/hard-stop exits. Does not modify
``strategy_baseline_v1.py`` or Strategy #1's confirmed entry/exit logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from engine.baseline_engine import BaselineEngineV1
from engine.research.kpi_report import build_hierarchical_kpi_report
from engine.strategy import load_config
from engine.strategy_s2_meanrev import (
    GROSS_EXPOSURE,
    HOLD_TRADING_DAYS,
    STOP_LOSS,
    TOP_REVERSAL_COUNT,
    compute_ret_n_panel,
    equal_weight_targets,
    exit_reason_for_position,
    refill_from_candidates,
    select_weekly_reversal_candidates,
    strategy_id,
    strategy_knobs,
)


class Strategy2MeanRevEngine(BaselineEngineV1):
    """Weekly reversal book with 10-trading-day / -15% exits."""

    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: str = "config/config.yaml",
        *,
        enable_quality_guard: bool = False,
    ):
        cfg = dict(config or load_config(config_path))
        # Force S1 overlays OFF so this engine never silently runs MQ/regime paths.
        cfg["enable_regime_exposure"] = False
        cfg["enable_regime_exposure_fast"] = False
        cfg["enable_industry_neutral_ranking"] = False
        cfg["fixed_leverage"] = 1.0
        cfg["enable_cash_interest"] = False
        # Load PIT when quality guard is requested (reuses parent loader).
        cfg["enable_quality_factor"] = bool(enable_quality_guard)
        cfg["enable_topup_chasing"] = False
        cfg.pop("disable_topup_chasing", None)
        cfg.setdefault("cost_model", "corwin_schultz_v2")
        cfg.setdefault("winsorize_adv", True)

        self.ENABLE_S2_QUALITY_GUARD = bool(enable_quality_guard)
        super().__init__(config=cfg, config_path=config_path)

        # S2 sizing / exit knobs (do not touch S1 TOP_MOMENTUM_COUNT semantics beyond refill max)
        self.TOP_REVERSAL_COUNT = int(TOP_REVERSAL_COUNT)
        self.HOLD_TRADING_DAYS = int(HOLD_TRADING_DAYS)
        self.STOP_LOSS = float(STOP_LOSS)
        self.GROSS_EXPOSURE = float(GROSS_EXPOSURE)
        self.FIXED_LEVERAGE = 1.0
        self.current_regime_exposure = 1.0
        self._prev_regime_exposure = 1.0

        self.ret5_m = compute_ret_n_panel(self.close_m, n=5)
        # entry_date_idx per symbol for trading-day hold count
        self.entry_date_idx: Dict[str, int] = {}
        self.entry_price: Dict[str, float] = {}
        self.diag_s2_exits: List[dict] = []

    def _active_strategy_id(self) -> str:
        return strategy_id(quality_guard=self.ENABLE_S2_QUALITY_GUARD)

    def _active_strategy_knobs(self) -> Dict[str, object]:
        knobs = dict(strategy_knobs(quality_guard=self.ENABLE_S2_QUALITY_GUARD))
        knobs["enable_s2_quality_guard"] = bool(self.ENABLE_S2_QUALITY_GUARD)
        knobs["cost_model"] = self.COST_MODEL
        knobs["enable_topup_chasing"] = bool(self.ENABLE_TOPUP_CHASING)
        return knobs

    def _weekly_rebalance_dates(self) -> Set[pd.Timestamp]:
        """First trading session of each ISO calendar week in the panel."""
        idx = self.close_m.index
        weeks = idx.to_series().dt.isocalendar()
        key = weeks["year"].astype(str) + "-" + weeks["week"].astype(str)
        first = idx[~key.duplicated(keep="first")]
        return set(pd.to_datetime(first))

    def _check_s2_exits(self, date_idx: int) -> bool:
        """Queue next-open sells for time stop / hard stop. Returns True if any queued."""
        current_date = self.close_m.index[date_idx]
        closes = self.close_m.loc[current_date]
        pending_sell = {o["symbol"] for o in self.pending_orders if o.get("type") == "SELL"}
        any_exit = False
        for sym in list(self.portfolio.keys()):
            if sym in pending_sell:
                continue
            eidx = self.entry_date_idx.get(sym)
            epx = self.entry_price.get(sym)
            if eidx is None or epx is None:
                # Fallback: use trade journal if meta missing
                ot = self.trade_journal.open.get(sym)
                if ot is None:
                    continue
                epx = float(ot.entry_price)
                # approximate trading-day index via searchsorted
                eidx = int(self.close_m.index.searchsorted(pd.Timestamp(ot.entry_date)))
            mark = closes.get(sym, np.nan)
            reason = exit_reason_for_position(
                entry_date_idx=int(eidx),
                current_date_idx=int(date_idx),
                entry_price=float(epx),
                mark_price=float(mark) if pd.notna(mark) else float("nan"),
                hold_trading_days=self.HOLD_TRADING_DAYS,
                stop_loss=self.STOP_LOSS,
            )
            if reason is None:
                continue
            qty = int(self.portfolio[sym])
            self.pending_orders.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "type": "SELL",
                    "reason": reason,
                    "order_role": None,
                }
            )
            self.diag_s2_exits.append(
                {
                    "signal_date": current_date,
                    "symbol": sym,
                    "reason": reason,
                    "entry_date_idx": int(eidx),
                    "hold_trading_days": int(date_idx - int(eidx)),
                    "entry_price": float(epx),
                    "mark_price": float(mark) if pd.notna(mark) else None,
                }
            )
            any_exit = True
        return any_exit

    def execute_pending_orders(self, date_idx):
        """Parent fills + track S2 entry meta; clear meta on full exit."""
        before = set(self.portfolio.keys())
        super().execute_pending_orders(date_idx)
        after = set(self.portfolio.keys())
        # New entries
        for sym in after - before:
            self.entry_date_idx[sym] = int(date_idx)
            # entry price from journal
            ot = self.trade_journal.open.get(sym)
            if ot is not None:
                self.entry_price[sym] = float(ot.entry_price)
        # Also update entry meta when adding to existing (should not happen with no-chase)
        for sym in after:
            if sym not in self.entry_date_idx:
                self.entry_date_idx[sym] = int(date_idx)
                ot = self.trade_journal.open.get(sym)
                if ot is not None:
                    self.entry_price[sym] = float(ot.entry_price)
        # Cleared
        for sym in before - after:
            self.entry_date_idx.pop(sym, None)
            self.entry_price.pop(sym, None)

    def _select_s2_candidates(self, idx: int, live: List[str]) -> List[str]:
        return select_weekly_reversal_candidates(
            date_idx=idx,
            close_m=self.close_m,
            dvol_m=self.dvol_m,
            ret5_m=self.ret5_m,
            eligible_symbols=live,
            top_liquid_pool=self.TOP_LIQUID_POOL,
            top_reversal_count=self.TOP_REVERSAL_COUNT,
            quality_guard=bool(self.ENABLE_S2_QUALITY_GUARD),
            fundamental_history=getattr(self, "fundamental_history", None),
            fund_ts=getattr(self, "_fund_ts", None),
        )

    def _emit_reports(self, equity: pd.DataFrame, qqq_eq: pd.DataFrame) -> dict:
        """Same KPI contract as S1, artifacts under cache/strategy_s2/."""
        out = Path("cache") / "strategy_s2"
        out.mkdir(parents=True, exist_ok=True)

        trades = self.trade_journal.to_frame()
        trades_path = out / "trade_journal.csv"
        if not trades.empty:
            trades.to_csv(trades_path, index=False)
        else:
            trades_path.write_text("", encoding="utf-8")

        bm_rets = None
        if not qqq_eq.empty and "Total_Equity" in qqq_eq.columns:
            bm_rets = qqq_eq["Total_Equity"].pct_change()

        kpi = build_hierarchical_kpi_report(equity, trades, benchmark_returns=bm_rets)
        qqq_kpi = build_hierarchical_kpi_report(qqq_eq, pd.DataFrame(), benchmark_returns=None)

        strat_cagr = (kpi.get("risk") or {}).get("cagr")
        strat_sharpe = (kpi.get("risk") or {}).get("sharpe")
        strat_mdd = (kpi.get("risk") or {}).get("mdd")
        qqq_cagr = (qqq_kpi.get("risk") or {}).get("cagr")
        qqq_sharpe = (qqq_kpi.get("risk") or {}).get("sharpe")
        qqq_mdd = (qqq_kpi.get("risk") or {}).get("mdd")
        alpha = None
        if strat_cagr is not None and qqq_cagr is not None:
            alpha = float(strat_cagr) - float(qqq_cagr)

        comparison = {
            "strategy": {
                "CAGR": strat_cagr,
                "Sharpe": strat_sharpe,
                "Maximum_Drawdown": strat_mdd,
            },
            "QQQ": {
                "CAGR": qqq_cagr,
                "Sharpe": qqq_sharpe,
                "Maximum_Drawdown": qqq_mdd,
            },
            "Alpha_CAGR": alpha,
            "knobs": self._active_strategy_knobs(),
            "window": {"start": str(self.START_DATE), "end": str(self.END_DATE)},
            "enable_s2_quality_guard": bool(self.ENABLE_S2_QUALITY_GUARD),
        }
        self._print_comparison(comparison)

        (out / "kpi.json").write_text(json.dumps(kpi, indent=2, default=str), encoding="utf-8")
        (out / "qqq_kpi.json").write_text(
            json.dumps(qqq_kpi, indent=2, default=str), encoding="utf-8"
        )
        (out / "benchmark_comparison.json").write_text(
            json.dumps(comparison, indent=2, default=str), encoding="utf-8"
        )
        if not equity.empty:
            equity.to_csv(out / "equity_curve.csv")
        if not qqq_eq.empty:
            qqq_eq.to_csv(out / "qqq_equity_curve.csv")

        return {
            "kpi": kpi,
            "qqq_kpi": qqq_kpi,
            "comparison": comparison,
            "trades_path": str(trades_path),
            "out_dir": str(out),
        }

    def run(self) -> pd.DataFrame:
        print("=" * 50)
        print(f"STRATEGY #2: {self._active_strategy_id()}")
        print("=" * 50)
        for k, v in self._active_strategy_knobs().items():
            print(f"  {k}: {v}")
        print(f"  cost_model: {self.COST_MODEL}")
        print(f"  winsorize_adv: {self.WINSORIZE_ADV}")
        print(f"  enable_topup_chasing: {self.ENABLE_TOPUP_CHASING}")
        print("=" * 50)

        weekly_dates = self._weekly_rebalance_dates()
        trading_days = self.close_m.index

        for idx, current_date in enumerate(trading_days):
            in_window = current_date >= self._run_start

            self._handle_delistings(idx)
            # Drop entry meta for names removed by delisting (bypasses execute path).
            for sym in list(self.entry_date_idx.keys()):
                if sym not in self.portfolio:
                    self.entry_date_idx.pop(sym, None)
                    self.entry_price.pop(sym, None)

            self.execute_pending_orders(idx)
            self._maybe_buy_qqq(idx)

            # S2 exits (signal on close → sell next open via pending)
            if in_window:
                self._check_s2_exits(idx)

            is_weekly = current_date in weekly_dates
            need_initial = in_window and not self.current_candidates and not self.portfolio

            if (is_weekly or need_initial) and in_window:
                live = [
                    s
                    for s in self._investable
                    if s in self.close_m.columns and pd.notna(self.close_m[s].iloc[idx])
                ]
                dvol_row = pd.to_numeric(
                    self.dvol_m.loc[current_date, [s for s in live if s in self.dvol_m.columns]],
                    errors="coerce",
                ).dropna()
                n_liquid = int(min(self.TOP_LIQUID_POOL, len(dvol_row)))
                self.diag_monthly_liquidity.append(
                    {
                        "date": current_date,
                        "n_live": len(live),
                        "n_with_dvol": int(len(dvol_row)),
                        "n_liquid_after_filter": n_liquid,
                        "top_liquid_pool": int(self.TOP_LIQUID_POOL),
                        "flag_thin": bool(n_liquid < 100),
                        "cadence": "weekly",
                    }
                )
                self.current_candidates = self._select_s2_candidates(idx, live)
                held = set(self.portfolio.keys())
                pending_sell = {
                    o["symbol"] for o in self.pending_orders if o.get("type") == "SELL"
                }
                effective_held = held - pending_sell
                selected = refill_from_candidates(
                    effective_held,
                    self.current_candidates,
                    max_positions=self.TOP_REVERSAL_COUNT,
                )
                targets = equal_weight_targets(selected, exposure=float(self.GROSS_EXPOSURE))
                self._queue_rebalance_to_targets(
                    targets, idx, allow_exposure_resize=False
                )
                print(
                    f"[S2 WEEKLY] {current_date.date()} candidates={len(self.current_candidates)} "
                    f"held={len(effective_held)} target_n={len(selected)} "
                    f"exposure={self.GROSS_EXPOSURE:.2f}x"
                )

            # Mark-to-market
            closes = self.close_m.loc[current_date]
            if self.portfolio:
                held = list(self.portfolio.keys())
                px = closes.reindex(held)
                qty = pd.Series({s: self.portfolio[s] for s in held}, dtype=float)
                pv = float((qty * px).sum(skipna=True))
            else:
                pv = 0.0
            total_equity = self.cash + pv
            if in_window:
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
