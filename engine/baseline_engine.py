"""Baseline v1 engine — infrastructure orchestration + strategy_baseline_v1.

Loads existing local panels (built by downloader.build_price_panel), applies the
shared execution model (Corwin–Schultz half-spread + square-root impact +
participation cap + commission/slippage), and delegates ALL entry/exit/sizing
decisions to ``engine.strategy_baseline_v1``.

Optional quality overlay (``enable_quality_factor``) loads existing PIT
``pit_history.pkl`` and ranks via ``strategy_baseline_v1_quality``; momentum-only
path remains the default. Does not modify downloader internals.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from engine.execution_costs import (
    ROBUST_IMPACT_COEFF,
    ROBUST_IMPACT_RATIO_CAP,
    ROBUST_IMPACT_SIGMA_CAP,
    analyze_raw_corwin_schultz,
    apply_legacy_cs_clip,
    apply_robust_cs,
    corwin_schultz_raw,
    diagnose_cs_breakdown,
    winsorize_dollar_volume,
)
from engine.pit_fundamentals import build_fund_ts_index, load_pit_history
from engine.regime_exposure import (
    compute_sma_panels,
    determine_regime_exposure,
    exposure_tier_label,
)
from engine.research.experiment_history import ExperimentHistory
from engine.research.kpi_report import build_hierarchical_kpi_report
from engine.research.trade_journal import TradeJournal
from engine.strategy import load_config
from engine.strategy_baseline_v1 import (
    ATR_MULTIPLIER,
    GROSS_EXPOSURE,
    TOP_LIQUID_POOL,
    TOP_MOMENTUM_COUNT,
    atr_stop_triggered,
    compute_mom_12_1_panel,
    equal_weight_targets,
    refill_from_candidates,
    select_monthly_candidates,
    strategy_id as strategy_id_mom,
    strategy_knobs as strategy_knobs_mom,
)
from engine.strategy_baseline_v1_quality import (
    select_monthly_candidates_mom_quality,
    strategy_id as strategy_id_quality,
    strategy_knobs as strategy_knobs_quality,
)


class BaselineEngineV1:
    """Minimal baseline runner: monthly 12-1 momentum + ATR trail only."""

    def __init__(self, config: Optional[dict] = None, config_path: str = "config/config.yaml"):
        self.config = config or load_config(config_path)
        cfg = self.config
        paths = cfg.get("paths", {})

        self.START_DATE = cfg.get("start_date", "2010-01-01")
        self.END_DATE = cfg.get("end_date")
        self.INITIAL_CASH = float(cfg.get("initial_cash", 50_000_000))
        self.BENCHMARK_TICKER = str(cfg.get("benchmark", "QQQ")).upper()
        self.SILENT_DELIST_RECOVERY = float(cfg.get("silent_delist_recovery", 0.30))
        self.PARTICIPATION_CAP_BUY = float(cfg.get("participation_cap_buy", 0.08))
        self.PARTICIPATION_CAP_SELL = float(cfg.get("participation_cap_sell", 0.15))
        self.COMMISSION_RATE = float(cfg.get("commission_rate", 0.0005))
        self.SLIPPAGE_RATE = float(cfg.get("slippage_rate", 0.0002))
        # Cost-model variant switch (does not change entry/exit/sizing formulas).
        #   corwin_schultz      — legacy CS clip [2bps, 5%] + raw ADV20 (pre-fix)
        #   corwin_schultz_v2   — robust CS (1% ceil + liquidity fallback) + winsorized ADV
        #   flat                — constant one-way fee (sensitivity benchmark)
        self.COST_MODEL = str(cfg.get("cost_model", "corwin_schultz_v2")).lower()
        self.FLAT_COST_ONE_WAY = float(cfg.get("flat_cost_one_way", 0.0005))
        self.WINSORIZE_ADV = bool(
            cfg.get(
                "winsorize_adv",
                self.COST_MODEL in ("corwin_schultz_v2", "robust"),
            )
        )
        # Fill/order policy (NOT entry/exit signal).
        # Baseline default (post no-chase promotion): once a position is opened,
        # do NOT submit further BUY orders solely to catch up to equal-weight
        # after a partial fill. Chase-on remains available via ENABLE_TOPUP_CHASING.
        if "enable_topup_chasing" in cfg or "ENABLE_TOPUP_CHASING" in cfg:
            self.ENABLE_TOPUP_CHASING = bool(
                cfg.get("enable_topup_chasing", cfg.get("ENABLE_TOPUP_CHASING"))
            )
            self.DISABLE_TOPUP_CHASING = not self.ENABLE_TOPUP_CHASING
        elif "disable_topup_chasing" in cfg or "DISABLE_TOPUP_CHASING" in cfg:
            self.DISABLE_TOPUP_CHASING = bool(
                cfg.get("disable_topup_chasing", cfg.get("DISABLE_TOPUP_CHASING"))
            )
            self.ENABLE_TOPUP_CHASING = not self.DISABLE_TOPUP_CHASING
        else:
            self.ENABLE_TOPUP_CHASING = False  # new baseline default
            self.DISABLE_TOPUP_CHASING = True

        # Strategy knobs — defaults from strategy_baseline_v1; optional config.baseline_v1 overrides
        bcfg = dict(cfg.get("baseline_v1") or {})
        self.TOP_LIQUID_POOL = int(bcfg.get("TOP_LIQUID_POOL", TOP_LIQUID_POOL))
        self.TOP_MOMENTUM_COUNT = int(bcfg.get("TOP_MOMENTUM_COUNT", TOP_MOMENTUM_COUNT))
        self.ATR_MULTIPLIER = float(bcfg.get("ATR_MULTIPLIER", ATR_MULTIPLIER))
        self.GROSS_EXPOSURE = float(bcfg.get("GROSS_EXPOSURE", GROSS_EXPOSURE))
        # Quality overlay (entry ranking only). Adopted as default after mom+quality A/B
        # (repro_id=42799a276531): Sharpe 0.194→0.268, MDD −31.97%→−25.60%.
        # Set enable_quality_factor: false to restore momentum-only baseline.
        if "enable_quality_factor" in cfg or "ENABLE_QUALITY_FACTOR" in cfg:
            self.ENABLE_QUALITY_FACTOR = bool(
                cfg.get("enable_quality_factor", cfg.get("ENABLE_QUALITY_FACTOR"))
            )
        elif "ENABLE_QUALITY_FACTOR" in bcfg:
            self.ENABLE_QUALITY_FACTOR = bool(bcfg.get("ENABLE_QUALITY_FACTOR"))
        else:
            self.ENABLE_QUALITY_FACTOR = True  # new default after quality A/B

        # Regime exposure (isolated variant; default OFF until A/B adoption).
        # Scales equal-weight targets by 1.0 / 0.5 / 0.0; never >1.0x.
        self.ENABLE_REGIME_EXPOSURE = bool(
            cfg.get(
                "enable_regime_exposure",
                cfg.get("ENABLE_REGIME_EXPOSURE", bcfg.get("ENABLE_REGIME_EXPOSURE", False)),
            )
        )

        print(">> Baseline v1: loading local panels (infrastructure artifacts)...")
        universe_path = Path(paths.get("metadata", "data/metadata")) / "universe.pkl"
        panels_path = Path(paths.get("prices", "data/prices")) / "panels.pkl"
        funds_dir = Path(paths.get("fundamentals", "data/fundamentals"))
        for required in (universe_path, panels_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"Missing {required}. Run: python -m downloader.update_data --config {config_path}"
                )

        with open(universe_path, "rb") as f:
            universe = pickle.load(f)
        with open(panels_path, "rb") as f:
            panels = pickle.load(f)

        self.all_tickers = universe["all_tickers"]
        self.tickers = universe["tickers"]
        self.delisted_meta = universe["delisted_meta"]
        self.profile_meta = universe["profile_meta"]
        self.open_m = panels["open_m"]
        self.close_m = panels["close_m"]
        self.high_m = panels["high_m"]
        self.low_m = panels["low_m"]
        self.dvol_m = panels["dvol_m"]
        self.silent_delist_flags = panels["silent_delist_flags"]

        self.fundamental_history: Dict[str, pd.DataFrame] = {}
        self._fund_ts: Dict[str, Any] = {}
        if self.ENABLE_QUALITY_FACTOR:
            self.fundamental_history = load_pit_history(funds_dir)
            self._fund_ts = build_fund_ts_index(self.fundamental_history)
            print(
                f"    quality overlay ON: loaded PIT fundamentals for "
                f"{len(self.fundamental_history)} symbols from {funds_dir / 'pit_history.pkl'}"
            )
            if not self.fundamental_history:
                raise FileNotFoundError(
                    f"enable_quality_factor=True but no PIT data at {funds_dir / 'pit_history.pkl'}. "
                    "Run: python3 -m downloader.download_fundamentals --config config/config.yaml"
                )

        # --- Diagnostic fields (must exist before _precompute fills CS/ADV stats) ---
        self.diag_total_cost_dollars = 0.0
        self.diag_cost_events: List[dict] = []
        self.diag_monthly_liquidity: List[dict] = []
        self.diag_closed_lots: List[dict] = []  # includes qty for $ PnL
        self.diag_equity_curve: List[dict] = []  # net + gross (gross = net + cum costs)
        self.diag_cs_raw_stats: Dict[str, Any] = {}
        self.diag_cs_breakdown: Dict[str, Any] = {}
        self.diag_fill_vs_target: List[dict] = []  # Issue 3 — buy fill vs intended target
        self.diag_dvol_spike_count = 0
        self.diag_regime_months: List[dict] = []

        # Slice to configured window (START_DATE / END_DATE only)
        self._slice_to_window()
        self._precompute_infra_matrices()

        self.cash = self.INITIAL_CASH
        self.portfolio: Dict[str, int] = {}
        self.highest_closes: Dict[str, float] = {}
        self.pending_orders: List[dict] = []
        self.equity_curve: List[dict] = []
        self.current_candidates: List[str] = []
        self.trade_journal = TradeJournal(shadow_horizon_days=20)
        self.research_artifacts: Dict[str, Any] = {}
        self.current_regime_exposure = float(self.GROSS_EXPOSURE)
        self._prev_regime_exposure = float(self.GROSS_EXPOSURE)

        # QQQ buy & hold (buy once, never rebalance)
        self.qqq_shares = 0.0
        self.qqq_cash = self.INITIAL_CASH
        self.qqq_equity_curve: List[dict] = []
        self._qqq_bought = False

        print(self._active_strategy_knobs())

    def _slice_to_window(self) -> None:
        start = pd.Timestamp(self.START_DATE)
        end = pd.Timestamp(self.END_DATE) if self.END_DATE else self.close_m.index.max()
        # Keep lookback before start for mom_12_1 / ATR warmup
        lookback = pd.Timedelta(days=400)
        lo = start - lookback
        mask = (self.close_m.index >= lo) & (self.close_m.index <= end)
        for name in ("open_m", "close_m", "high_m", "low_m", "dvol_m"):
            setattr(self, name, getattr(self, name).loc[mask])
        self._run_start = start
        self._run_end = end

    def _precompute_infra_matrices(self) -> None:
        """Matrices required by strategy + execution / cost model."""
        close_m, high_m, low_m = self.close_m, self.high_m, self.low_m
        # Strategy universe ranking continues to use raw dvol_m (unchanged).
        # ADV for participation/impact may use winsorized dvol (cost/data layer).
        dvol_for_adv = self.dvol_m
        if self.WINSORIZE_ADV:
            dvol_for_adv, spike_flag = winsorize_dollar_volume(self.dvol_m)
            self.diag_dvol_spike_count = int(spike_flag.to_numpy().sum())
            self.dvol_for_adv_m = dvol_for_adv
        else:
            self.dvol_for_adv_m = self.dvol_m
            self.diag_dvol_spike_count = 0

        ret_m = close_m.pct_change()
        self.adv20_m = dvol_for_adv.rolling(20, min_periods=5).mean()
        self.vol20_m = ret_m.rolling(20, min_periods=10).std()

        # ATR(14) Wilder smoothing (same construction as existing engine)
        prev_close = close_m.shift(1)
        tr = (high_m - low_m).combine((high_m - prev_close).abs(), np.maximum).combine(
            (low_m - prev_close).abs(), np.maximum
        )
        self.atr14_m = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

        # 12-1 momentum panel (exact definition)
        self.mom_12_1_m = compute_mom_12_1_panel(close_m)

        # Regime SMAs (only used when enable_regime_exposure)
        if self.ENABLE_REGIME_EXPOSURE:
            self.sma50_m, self.sma200_m = compute_sma_panels(close_m)
        else:
            self.sma50_m = None
            self.sma200_m = None

        # Corwin–Schultz: always compute raw for diagnostics; then apply model variant
        spread_raw = corwin_schultz_raw(high_m, low_m)
        self.cs_spread_raw_m = spread_raw
        self.diag_cs_raw_stats = analyze_raw_corwin_schultz(spread_raw)
        self.diag_cs_breakdown = diagnose_cs_breakdown(high_m, low_m, close_m, spread_raw)

        if self.COST_MODEL in ("corwin_schultz_v2", "robust"):
            self.cs_spread_m, self.cs_spread_source_m = apply_robust_cs(
                spread_raw, high_m, low_m, self.adv20_m
            )
        else:
            # Legacy path (and flat model still keeps a CS panel for diagnostics)
            self.cs_spread_m = apply_legacy_cs_clip(spread_raw)
            self.cs_spread_source_m = None

        self._investable = [
            s
            for s in close_m.columns
            if s != self.BENCHMARK_TICKER
            and not self.profile_meta.get(s, {}).get("isEtf", False)
        ]

    # ----- execution model (mirrors existing StandaloneEngine fills) -----
    def _cost_breakdown(self, symbol, date_idx, qty, price) -> dict:
        """Decompose execution cost (same dollars as applied to cash).

        Default model formulas are unchanged. Optional ``cost_model=flat`` replaces
        the ratio with a constant one-way fee for sensitivity audits only.
        """
        adv_raw = self.adv20_m[symbol].iloc[date_idx] if symbol in self.adv20_m.columns else np.nan
        sigma_raw = self.vol20_m[symbol].iloc[date_idx] if symbol in self.vol20_m.columns else np.nan
        cs_raw = self.cs_spread_m[symbol].iloc[date_idx] if symbol in self.cs_spread_m.columns else np.nan
        adv = float(adv_raw) if pd.notna(adv_raw) and float(adv_raw) > 0 else 1e6
        sigma = float(sigma_raw) if pd.notna(sigma_raw) and float(sigma_raw) > 0 else 0.02
        cs = float(cs_raw) if pd.notna(cs_raw) and float(cs_raw) > 0 else float("nan")
        half_spread = float(cs) / 2 if pd.notna(cs) and cs > 0 else 0.0015
        notional = float(qty) * float(price)
        participation = notional / max(adv, 1.0)

        if self.COST_MODEL == "flat":
            cost_ratio = float(self.FLAT_COST_ONE_WAY)
            return {
                "adv": adv,
                "adv_raw": float(adv_raw) if pd.notna(adv_raw) else float("nan"),
                "sigma": sigma,
                "cs_spread": cs,
                "half_spread": 0.0,
                "participation": float(participation),
                "impact": 0.0,
                "commission": float(self.FLAT_COST_ONE_WAY),
                "slippage": 0.0,
                "cost_ratio": cost_ratio,
                "cost_model": "flat",
            }

        if self.COST_MODEL in ("corwin_schultz_v2", "robust"):
            # Same sqrt-impact shape as legacy, with σ / impact caps so microcap
            # daily vol cannot alone charge multi-percent one-way costs.
            sigma_for_impact = min(sigma, ROBUST_IMPACT_SIGMA_CAP)
            impact = ROBUST_IMPACT_COEFF * sigma_for_impact * np.sqrt(max(participation, 0.0))
            impact = float(min(impact, ROBUST_IMPACT_RATIO_CAP))
        else:
            impact = 0.6 * sigma * np.sqrt(max(participation, 0.0))

        cost_ratio = float(half_spread + impact + self.COMMISSION_RATE + self.SLIPPAGE_RATE)
        return {
            "adv": adv,
            "adv_raw": float(adv_raw) if pd.notna(adv_raw) else float("nan"),
            "sigma": sigma,
            "cs_spread": cs,
            "half_spread": float(half_spread),
            "participation": float(participation),
            "impact": float(impact),
            "commission": float(self.COMMISSION_RATE),
            "slippage": float(self.SLIPPAGE_RATE),
            "cost_ratio": cost_ratio,
            "cost_model": self.COST_MODEL,
        }

    def _record_cost(
        self,
        *,
        date,
        symbol,
        side,
        qty,
        price,
        cost_ratio,
        breakdown=None,
        order_role=None,
    ) -> float:
        """Observation-only: aggregate execution cost dollars (formula unchanged)."""
        notional = float(qty) * float(price)
        dollars = notional * float(cost_ratio)
        half = float((breakdown or {}).get("half_spread") or 0.0)
        imp = float((breakdown or {}).get("impact") or 0.0)
        comm = float((breakdown or {}).get("commission") or 0.0)
        slip = float((breakdown or {}).get("slippage") or 0.0)
        self.diag_total_cost_dollars += dollars
        event = {
            "date": date,
            "symbol": symbol,
            "side": side,
            "qty": int(qty),
            "price": float(price),
            "cost_ratio": float(cost_ratio),
            "cost_dollars": float(dollars),
            "notional": float(notional),
            "order_role": order_role,  # "new_entry" | "top_up" | None (sells)
            "spread_cost_dollars": notional * half,
            "impact_cost_dollars": notional * imp,
            "commission_cost_dollars": notional * comm,
            "slippage_cost_dollars": notional * slip,
        }
        if breakdown:
            event.update(
                {
                    "adv": breakdown.get("adv"),
                    "adv_raw": breakdown.get("adv_raw"),
                    "sigma": breakdown.get("sigma"),
                    "cs_spread": breakdown.get("cs_spread"),
                    "half_spread": breakdown.get("half_spread"),
                    "participation": breakdown.get("participation"),
                    "impact": breakdown.get("impact"),
                    "commission": breakdown.get("commission"),
                    "slippage": breakdown.get("slippage"),
                    "cost_model": breakdown.get("cost_model"),
                }
            )
        self.diag_cost_events.append(event)
        return dollars

    def _cost_ratio(self, symbol, date_idx, qty, price):
        return float(self._cost_breakdown(symbol, date_idx, qty, price)["cost_ratio"])

    def _max_shares_participation(self, symbol, date_idx, price, cap_ratio):
        adv = self.adv20_m[symbol].iloc[date_idx] if symbol in self.adv20_m.columns else np.nan
        if pd.isna(adv) or adv <= 0 or price <= 0:
            return 0
        return int((adv * cap_ratio) / price)

    def _execute_sell_fill(self, sym, date_idx, qty, fill_price, reason) -> int:
        if sym not in self.portfolio or qty <= 0:
            return 0
        if pd.isna(fill_price) or float(fill_price) <= 0:
            return 0
        px = float(fill_price)
        current_date = self.close_m.index[date_idx]
        held = int(self.portfolio[sym])
        cap = self._max_shares_participation(sym, date_idx, px, self.PARTICIPATION_CAP_SELL)
        exec_qty = min(held, int(qty), max(int(cap), 0))
        if exec_qty <= 0:
            return 0
        bd = self._cost_breakdown(sym, date_idx, exec_qty, px)
        cost = float(bd["cost_ratio"])
        self._record_cost(
            date=current_date,
            symbol=sym,
            side="SELL",
            qty=exec_qty,
            price=px,
            cost_ratio=cost,
            breakdown=bd,
        )
        # Capture lot info before journal pop (for $ loss diagnostics)
        ot = self.trade_journal.open.get(sym)
        entry_px = float(ot.entry_price) if ot is not None else float("nan")
        entry_date = ot.entry_date if ot is not None else None
        lot_qty = int(ot.qty) if ot is not None else exec_qty

        proceeds = exec_qty * px * (1.0 - cost)
        self.cash += proceeds
        self.portfolio[sym] = held - exec_qty
        if self.portfolio[sym] <= 0:
            del self.portfolio[sym]
            self.highest_closes.pop(sym, None)
            closed = self.trade_journal.on_exit(
                symbol=sym,
                date=current_date,
                date_idx=date_idx,
                price=px,
                exit_reason=reason,
                close_m=self.close_m,
                exit_rank=None,
                exit_cmvs=None,
                exit_rsi=None,
                exit_atr=(
                    float(self.atr14_m[sym].iloc[date_idx])
                    if sym in self.atr14_m.columns and pd.notna(self.atr14_m[sym].iloc[date_idx])
                    else None
                ),
            )
            if closed is not None and np.isfinite(entry_px) and entry_px > 0:
                dollar_pnl = float(lot_qty) * (px - entry_px)
                self.diag_closed_lots.append(
                    {
                        "symbol": sym,
                        "entry_date": entry_date,
                        "exit_date": current_date,
                        "entry_price": entry_px,
                        "exit_price": px,
                        "qty": int(lot_qty),
                        "pct_return": float(closed.final_return),
                        "dollar_pnl": dollar_pnl,
                        "exit_reason": reason,
                        "holding_days": int(closed.holding_days),
                    }
                )
        return exec_qty

    def execute_pending_orders(self, date_idx):
        current_date = self.close_m.index[date_idx]
        opens = self.open_m.loc[current_date]
        still_pending = []
        for order in self.pending_orders:
            sym = order["symbol"]
            o_price = opens.get(sym, np.nan)
            if pd.isna(o_price) or o_price <= 0:
                still_pending.append(order)
                continue
            qty = int(order["qty"])
            if order["type"] == "BUY":
                order_role = order.get("order_role") or (
                    "top_up" if sym in self.portfolio else "new_entry"
                )
                target_qty = int(order.get("target_qty", qty) or qty)
                target_notional = float(
                    order.get("target_notional", target_qty * float(o_price))
                )
                cap = self._max_shares_participation(sym, date_idx, o_price, self.PARTICIPATION_CAP_BUY)
                exec_qty = min(qty, max(int(cap), 0))
                if exec_qty <= 0:
                    self.diag_fill_vs_target.append(
                        {
                            "date": current_date,
                            "symbol": sym,
                            "side": "BUY",
                            "order_role": order_role,
                            "target_qty": target_qty,
                            "target_notional": target_notional,
                            "filled_qty": 0,
                            "filled_notional": 0.0,
                            "fill_pct_of_target": 0.0,
                            "reason": "participation_cap_zero",
                        }
                    )
                    continue
                bd = self._cost_breakdown(sym, date_idx, exec_qty, float(o_price))
                cost = float(bd["cost_ratio"])
                spend = exec_qty * float(o_price) * (1.0 + cost)
                if spend > self.cash:
                    exec_qty = int(self.cash / (float(o_price) * (1.0 + cost)))
                    if exec_qty <= 0:
                        self.diag_fill_vs_target.append(
                            {
                                "date": current_date,
                                "symbol": sym,
                                "side": "BUY",
                                "order_role": order_role,
                                "target_qty": target_qty,
                                "target_notional": target_notional,
                                "filled_qty": 0,
                                "filled_notional": 0.0,
                                "fill_pct_of_target": 0.0,
                                "reason": "insufficient_cash",
                            }
                        )
                        continue
                    bd = self._cost_breakdown(sym, date_idx, exec_qty, float(o_price))
                    cost = float(bd["cost_ratio"])
                    spend = exec_qty * float(o_price) * (1.0 + cost)
                self._record_cost(
                    date=current_date,
                    symbol=sym,
                    side="BUY",
                    qty=exec_qty,
                    price=float(o_price),
                    cost_ratio=cost,
                    breakdown=bd,
                    order_role=order_role,
                )
                self.cash -= spend
                self.portfolio[sym] = self.portfolio.get(sym, 0) + exec_qty
                px = float(o_price)
                filled_notional = float(exec_qty) * px
                fill_pct = (
                    filled_notional / target_notional
                    if target_notional > 0
                    else float("nan")
                )
                self.diag_fill_vs_target.append(
                    {
                        "date": current_date,
                        "symbol": sym,
                        "side": "BUY",
                        "order_role": order_role,
                        "target_qty": target_qty,
                        "target_notional": target_notional,
                        "filled_qty": int(exec_qty),
                        "filled_notional": filled_notional,
                        "fill_pct_of_target": float(fill_pct),
                        "reason": "ok" if exec_qty >= target_qty else "partial_participation_or_cash",
                    }
                )
                self.highest_closes[sym] = max(self.highest_closes.get(sym, px), px)
                self.trade_journal.on_entry(
                    symbol=sym,
                    date=current_date,
                    date_idx=date_idx,
                    price=px,
                    qty=exec_qty,
                    rank=None,
                    cmvs=None,
                    rsi=None,
                    atr=(
                        float(self.atr14_m[sym].iloc[date_idx])
                        if sym in self.atr14_m.columns and pd.notna(self.atr14_m[sym].iloc[date_idx])
                        else None
                    ),
                )
            elif order["type"] == "SELL":
                self._execute_sell_fill(sym, date_idx, qty, float(o_price), order.get("reason", "atr_trail"))
        self.pending_orders = still_pending

    def _handle_delistings(self, date_idx):
        current_date = self.close_m.index[date_idx]
        for sym in list(self.portfolio.keys()):
            meta = self.delisted_meta.get(sym)
            if meta and meta.get("delistingDate"):
                d_date = pd.to_datetime(meta["delistingDate"])
                if current_date == d_date:
                    last_close = self.close_m[sym].iloc[date_idx]
                    if pd.notna(last_close):
                        ot = self.trade_journal.open.get(sym)
                        qty = int(self.portfolio[sym])
                        entry_px = float(ot.entry_price) if ot else float("nan")
                        entry_date = ot.entry_date if ot else None
                        self.cash += self.portfolio[sym] * float(last_close)
                        closed = self.trade_journal.on_exit(
                            symbol=sym,
                            date=current_date,
                            date_idx=date_idx,
                            price=float(last_close),
                            exit_reason="official_delist",
                            close_m=self.close_m,
                            exit_rank=None,
                            exit_cmvs=None,
                            exit_rsi=None,
                            exit_atr=None,
                        )
                        if closed is not None and np.isfinite(entry_px) and entry_px > 0:
                            self.diag_closed_lots.append(
                                {
                                    "symbol": sym,
                                    "entry_date": entry_date,
                                    "exit_date": current_date,
                                    "entry_price": entry_px,
                                    "exit_price": float(last_close),
                                    "qty": qty,
                                    "pct_return": float(closed.final_return),
                                    "dollar_pnl": qty * (float(last_close) - entry_px),
                                    "exit_reason": "official_delist",
                                    "holding_days": int(closed.holding_days),
                                }
                            )
                    del self.portfolio[sym]
                    self.highest_closes.pop(sym, None)

        for sym, last_date in self.silent_delist_flags.items():
            if current_date == last_date + pd.Timedelta(days=1) and sym in self.portfolio:
                last_valid = self.close_m[sym].iloc[self.close_m.index.get_loc(last_date)]
                if pd.notna(last_valid):
                    ot = self.trade_journal.open.get(sym)
                    qty = int(self.portfolio[sym])
                    entry_px = float(ot.entry_price) if ot else float("nan")
                    entry_date = ot.entry_date if ot else None
                    px = float(last_valid) * self.SILENT_DELIST_RECOVERY
                    self.cash += self.portfolio[sym] * px
                    closed = self.trade_journal.on_exit(
                        symbol=sym,
                        date=current_date,
                        date_idx=date_idx,
                        price=px,
                        exit_reason="silent_delist",
                        close_m=self.close_m,
                        exit_rank=None,
                        exit_cmvs=None,
                        exit_rsi=None,
                        exit_atr=None,
                    )
                    if closed is not None and np.isfinite(entry_px) and entry_px > 0:
                        self.diag_closed_lots.append(
                            {
                                "symbol": sym,
                                "entry_date": entry_date,
                                "exit_date": current_date,
                                "entry_price": entry_px,
                                "exit_price": px,
                                "qty": qty,
                                "pct_return": float(closed.final_return),
                                "dollar_pnl": qty * (px - entry_px),
                                "exit_reason": "silent_delist",
                                "holding_days": int(closed.holding_days),
                            }
                        )
                del self.portfolio[sym]
                self.highest_closes.pop(sym, None)

    def _check_atr_exits(self, date_idx):
        """If Low < ATR stop → queue SELL for next day's Open."""
        current_date = self.close_m.index[date_idx]
        lows = self.low_m.loc[current_date]
        closes = self.close_m.loc[current_date]
        pending_sell = {o["symbol"] for o in self.pending_orders if o.get("type") == "SELL"}

        for sym in list(self.portfolio.keys()):
            if sym in pending_sell:
                continue
            low = lows.get(sym, np.nan)
            close_px = closes.get(sym, np.nan)
            atr = (
                self.atr14_m[sym].iloc[date_idx]
                if sym in self.atr14_m.columns
                else np.nan
            )
            if pd.isna(close_px) or close_px <= 0:
                continue
            peak = self.highest_closes.get(sym, float(close_px))
            peak = max(float(peak), float(close_px))
            self.highest_closes[sym] = peak
            if atr_stop_triggered(float(low) if pd.notna(low) else float("nan"), peak, float(atr) if pd.notna(atr) else float("nan"), self.ATR_MULTIPLIER):
                self.pending_orders.append(
                    {
                        "symbol": sym,
                        "qty": int(self.portfolio[sym]),
                        "type": "SELL",
                        "reason": f"atr_trail(mult={self.ATR_MULTIPLIER})",
                    }
                )

    def _queue_rebalance_to_targets(
        self,
        targets: Dict[str, float],
        date_idx: int,
        *,
        allow_exposure_resize: bool = False,
    ):
        current_date = self.close_m.index[date_idx]
        closes = self.close_m.loc[current_date]
        total_equity = self.cash + sum(
            qty * closes.get(s, np.nan)
            for s, qty in self.portfolio.items()
            if pd.notna(closes.get(s))
        )
        all_syms = set(self.portfolio.keys()) | set(targets.keys())
        for sym in all_syms:
            price = closes.get(sym, np.nan)
            if pd.isna(price) or price <= 0:
                continue
            in_targets = sym in targets and float(targets.get(sym, 0.0)) > 0
            target_value = float(targets.get(sym, 0.0)) * total_equity
            currently_held = sym in self.portfolio and int(self.portfolio.get(sym, 0)) > 0
            current_value = self.portfolio.get(sym, 0) * float(price)
            delta = target_value - current_value
            if abs(delta) < total_equity * 0.001:
                continue
            qty = int(abs(delta) / float(price))
            if qty <= 0:
                continue
            side = "BUY" if delta > 0 else "SELL"

            # --- DISABLE_TOPUP_CHASING (fill/order variant only) ---
            # Keep already-opened positions at their filled size until ATR exit
            # or the name drops out of this month's selection (full sell).
            # Exception: allow_exposure_resize=True on regime exposure *changes*
            # so 1.0→0.5/0.0 (or recovery) can resize once; no-chase still
            # applies when exposure is unchanged.
            if self.DISABLE_TOPUP_CHASING and currently_held and not allow_exposure_resize:
                if side == "BUY":
                    continue  # no catch-up top-up
                if side == "SELL" and in_targets:
                    continue  # no size-trim while still selected
                # else: SELL and not in_targets → full drop-out exit, allowed

            order_role = None
            if side == "BUY":
                order_role = "top_up" if currently_held else "new_entry"
            self.pending_orders.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "type": side,
                    "reason": (
                        "rebalance_entry"
                        if side == "BUY" and order_role == "new_entry"
                        else "rebalance_topup"
                        if side == "BUY"
                        else "rebalance_dropout"
                        if side == "SELL" and not in_targets
                        else "rebalance_trim"
                    ),
                    "order_role": order_role,
                    # Issue 3 diagnostics: intended equal-weight target vs fill
                    "target_qty": qty,
                    "target_notional": float(qty) * float(price),
                }
            )

    def _maybe_buy_qqq(self, date_idx):
        if self._qqq_bought:
            return
        bm = self.BENCHMARK_TICKER
        if bm not in self.open_m.columns:
            return
        px = self.open_m[bm].iloc[date_idx]
        if pd.isna(px) or float(px) <= 0:
            return
        # Buy once at first eligible open inside the evaluation window
        current_date = self.close_m.index[date_idx]
        if current_date < self._run_start:
            return
        shares = int(self.qqq_cash / float(px))
        if shares <= 0:
            return
        self.qqq_cash -= shares * float(px)
        self.qqq_shares = float(shares)
        self._qqq_bought = True
        print(f"[QQQ B&H] bought {shares} @ {float(px):.2f} on {current_date.date()}")

    def _mark_qqq(self, date_idx):
        bm = self.BENCHMARK_TICKER
        current_date = self.close_m.index[date_idx]
        if bm in self.close_m.columns and pd.notna(self.close_m[bm].iloc[date_idx]):
            eq = self.qqq_cash + self.qqq_shares * float(self.close_m[bm].iloc[date_idx])
        else:
            eq = self.qqq_cash
        self.qqq_equity_curve.append({"Date": current_date, "Total_Equity": eq})

    def _active_strategy_id(self) -> str:
        base = strategy_id_quality() if self.ENABLE_QUALITY_FACTOR else strategy_id_mom()
        if self.ENABLE_REGIME_EXPOSURE:
            return base + "_regime"
        return base

    def _active_strategy_knobs(self) -> Dict[str, object]:
        knobs = dict(
            strategy_knobs_quality() if self.ENABLE_QUALITY_FACTOR else strategy_knobs_mom()
        )
        knobs["strategy_id"] = self._active_strategy_id()
        knobs["enable_regime_exposure"] = bool(self.ENABLE_REGIME_EXPOSURE)
        knobs["regime_max_exposure"] = 1.0  # no >1.0x in this pass
        return knobs

    def _select_candidates(self, idx: int, live: List[str]) -> List[str]:
        if self.ENABLE_QUALITY_FACTOR:
            return select_monthly_candidates_mom_quality(
                date_idx=idx,
                close_m=self.close_m,
                dvol_m=self.dvol_m,
                mom_12_1_m=self.mom_12_1_m,
                eligible_symbols=live,
                fundamental_history=self.fundamental_history,
                fund_ts=self._fund_ts,
                top_liquid_pool=self.TOP_LIQUID_POOL,
                top_momentum_count=self.TOP_MOMENTUM_COUNT,
            )
        return select_monthly_candidates(
            date_idx=idx,
            close_m=self.close_m,
            dvol_m=self.dvol_m,
            mom_12_1_m=self.mom_12_1_m,
            eligible_symbols=live,
            top_liquid_pool=self.TOP_LIQUID_POOL,
            top_momentum_count=self.TOP_MOMENTUM_COUNT,
        )

    def _liquid_universe_for_breadth(self, live: List[str], current_date) -> List[str]:
        """Same liquidity filter as entry ranking (top dvol pool)."""
        syms = [s for s in live if s in self.dvol_m.columns]
        if not syms:
            return []
        dvol_row = pd.to_numeric(self.dvol_m.loc[current_date, syms], errors="coerce").dropna()
        if dvol_row.empty:
            return []
        return dvol_row.nlargest(min(int(self.TOP_LIQUID_POOL), len(dvol_row))).index.tolist()

    def _resolve_regime_exposure(self, idx: int, live: List[str], current_date) -> float:
        """Return target gross exposure for this rebalance (1.0 if regime off)."""
        if not self.ENABLE_REGIME_EXPOSURE:
            return float(self.GROSS_EXPOSURE)
        liquid = self._liquid_universe_for_breadth(live, current_date)
        state = determine_regime_exposure(
            close_m=self.close_m,
            sma50_m=self.sma50_m,
            sma200_m=self.sma200_m,
            date_idx=idx,
            benchmark=self.BENCHMARK_TICKER,
            liquid_symbols=liquid,
        )
        if state is None:
            exposure = float(self.GROSS_EXPOSURE)
            breadth = None
            above = None
        else:
            exposure = float(min(state.exposure, 1.0))  # never >1.0x
            breadth = float(state.breadth)
            above = bool(state.benchmark_above_ma200)
        self.diag_regime_months.append(
            {
                "date": current_date,
                "exposure": exposure,
                "tier": exposure_tier_label(exposure),
                "breadth": breadth,
                "benchmark_above_ma200": above,
                "n_liquid_for_breadth": len(liquid),
            }
        )
        return exposure

    def run(self) -> pd.DataFrame:
        print("=" * 50)
        print(f"BASELINE STRATEGY: {self._active_strategy_id()}")
        print("=" * 50)
        for k, v in self._active_strategy_knobs().items():
            print(f"  {k}: {v}")
        print(f"  cost_model: {self.COST_MODEL}")
        print(f"  winsorize_adv: {self.WINSORIZE_ADV}")
        print(f"  enable_topup_chasing: {self.ENABLE_TOPUP_CHASING}")
        print(f"  disable_topup_chasing: {self.DISABLE_TOPUP_CHASING}")
        print(f"  enable_quality_factor: {self.ENABLE_QUALITY_FACTOR}")
        print(f"  enable_regime_exposure: {self.ENABLE_REGIME_EXPOSURE}")
        if self.COST_MODEL == "flat":
            print(f"  flat_cost_one_way: {self.FLAT_COST_ONE_WAY}")
        if self.diag_cs_raw_stats:
            hit = self.diag_cs_raw_stats.get("pct_finite_ge_legacy_ceiling_5pct")
            if hit is not None:
                print(f"  cs_raw_%_ge_5pct_clip: {100 * float(hit):.2f}%")
        print("=" * 50)

        trading_days = self.close_m.index
        months = trading_days.to_period("M")
        month_starts = set(pd.to_datetime(trading_days[~months.duplicated(keep="first")]))

        # Need 252 bars of history for mom_12_1; panel is sliced with lookback before START_DATE.
        warmup = 252
        for idx, current_date in enumerate(trading_days):
            if idx < warmup:
                continue
            if current_date > self._run_end:
                break
            # Evaluation / trading actions only on/after START_DATE
            in_window = current_date >= self._run_start

            self._handle_delistings(idx)
            n_before = len(self.portfolio)
            self.execute_pending_orders(idx)
            slots_freed = n_before > len(self.portfolio)
            self._maybe_buy_qqq(idx)

            # ATR trail check (Low vs stop) → next-open exit
            self._check_atr_exits(idx)

            # Monthly: refresh momentum candidates; refill empty slots only.
            # Also run once on the first eligible session if the calendar month-start
            # fell inside the mom_12_1 warmup window.
            is_month_start = current_date in month_starts and in_window
            need_initial = in_window and not self.current_candidates and not self.portfolio
            if is_month_start or need_initial:
                live = [
                    s
                    for s in self._investable
                    if s in self.close_m.columns and pd.notna(self.close_m[s].iloc[idx])
                ]
                # Diagnostic: liquidity pool depth (read-only; same filter as strategy)
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
                    }
                )
                self.current_candidates = self._select_candidates(idx, live)
                exposure = self._resolve_regime_exposure(idx, live, current_date)
                exposure_changed = abs(float(exposure) - float(self._prev_regime_exposure)) > 1e-9
                self.current_regime_exposure = float(exposure)
                held = set(self.portfolio.keys())
                pending_sell = {o["symbol"] for o in self.pending_orders if o.get("type") == "SELL"}
                effective_held = held - pending_sell
                if float(exposure) <= 0.0:
                    # Confirmed downtrend: liquidate everything.
                    selected = []
                    targets = {}
                    allow_resize = True
                else:
                    selected = refill_from_candidates(
                        effective_held,
                        self.current_candidates,
                        max_positions=self.TOP_MOMENTUM_COUNT,
                    )
                    targets = equal_weight_targets(selected, exposure=float(exposure))
                    # Allow one-shot resize only when exposure tier changed.
                    allow_resize = bool(self.ENABLE_REGIME_EXPOSURE and exposure_changed)
                self._queue_rebalance_to_targets(
                    targets, idx, allow_exposure_resize=allow_resize
                )
                self._prev_regime_exposure = float(exposure)
                print(
                    f"[REBALANCE] {current_date.date()} candidates={len(self.current_candidates)} "
                    f"held={len(effective_held)} target_n={len(selected)} "
                    f"exposure={exposure:.2f}x tier={exposure_tier_label(exposure)}"
                    f"{' RESIZE' if allow_resize else ''}"
                )

            # After ATR (or delist) frees a slot: refill from current month candidates
            elif (
                slots_freed
                and current_date >= self._run_start
                and self.current_candidates
                and len(self.portfolio) < self.TOP_MOMENTUM_COUNT
                and float(self.current_regime_exposure) > 0.0
            ):
                held = set(self.portfolio.keys())
                pending_sell = {o["symbol"] for o in self.pending_orders if o.get("type") == "SELL"}
                effective = held - pending_sell
                selected = refill_from_candidates(
                    effective,
                    self.current_candidates,
                    max_positions=self.TOP_MOMENTUM_COUNT,
                )
                targets = equal_weight_targets(
                    selected, exposure=float(self.current_regime_exposure)
                )
                # Mid-month refill: no-chase still applies (exposure unchanged).
                self._queue_rebalance_to_targets(targets, idx, allow_exposure_resize=False)

            # Mark-to-market (evaluation window only for reported curve)
            closes = self.close_m.loc[current_date]
            if self.portfolio:
                held = list(self.portfolio.keys())
                px = closes.reindex(held)
                qty = pd.Series({s: self.portfolio[s] for s in held}, dtype=float)
                pv = float((qty * px).sum(skipna=True))
            else:
                pv = 0.0
            total_equity = self.cash + pv
            if current_date >= self._run_start:
                self.equity_curve.append({"Date": current_date, "Total_Equity": total_equity})
                # Gross ≈ net + cumulative execution costs paid so far (diagnostic only)
                self.diag_equity_curve.append(
                    {
                        "Date": current_date,
                        "Net_Equity": total_equity,
                        "Gross_Equity": total_equity + float(self.diag_total_cost_dollars),
                        "Cum_Cost_Dollars": float(self.diag_total_cost_dollars),
                    }
                )
                self._mark_qqq(idx)

        equity = pd.DataFrame(self.equity_curve).set_index("Date") if self.equity_curve else pd.DataFrame()
        qqq_eq = (
            pd.DataFrame(self.qqq_equity_curve).set_index("Date")
            if self.qqq_equity_curve
            else pd.DataFrame()
        )
        self.research_artifacts = self._emit_reports(equity, qqq_eq)
        return equity

    def _emit_reports(self, equity: pd.DataFrame, qqq_eq: pd.DataFrame) -> dict:
        out = Path("cache") / "baseline_v1"
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
            "enable_quality_factor": bool(self.ENABLE_QUALITY_FACTOR),
        }

        self._print_comparison(comparison)

        import json

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

    @staticmethod
    def _print_comparison(comparison: dict) -> None:
        s = comparison["strategy"]
        q = comparison["QQQ"]

        def _fmt(x, pct=False):
            if x is None:
                return "n/a"
            return f"{100 * float(x):.2f}%" if pct else f"{float(x):.3f}"

        print("\n" + "=" * 64)
        print("  KPI SUMMARY — Strategy vs QQQ Buy & Hold")
        print("=" * 64)
        print(f"{'Metric':<22} {'Strategy':>14} {'QQQ':>14}")
        print("-" * 64)
        print(f"{'CAGR':<22} {_fmt(s['CAGR'], True):>14} {_fmt(q['CAGR'], True):>14}")
        print(f"{'Sharpe':<22} {_fmt(s['Sharpe']):>14} {_fmt(q['Sharpe']):>14}")
        print(
            f"{'Maximum Drawdown':<22} {_fmt(s['Maximum_Drawdown'], True):>14} "
            f"{_fmt(q['Maximum_Drawdown'], True):>14}"
        )
        print("-" * 64)
        print(f"{'Alpha (CAGR−QQQ)':<22} {_fmt(comparison.get('Alpha_CAGR'), True):>14}")
        print("=" * 64 + "\n")
