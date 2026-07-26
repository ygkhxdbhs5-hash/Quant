"""130/30 long-short Baseline engine — separate from long-only BaselineEngineV1.

Structural variant (not a ranking tweak):
  - long sleeve 130% EW of top-K mom+quality
  - short sleeve 30% EW of bottom-K (HTB ADV filter)
  - mirrored ATR cover on shorts
  - daily borrow fee on short notional
  - Fixed CS v2 execution on SHORT open / COVER (symmetric)

Long-only BaselineEngineV1 is untouched and remains the comparable default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from engine.baseline_engine import BaselineEngineV1
from engine.research.kpi_report import build_hierarchical_kpi_report
from engine.strategy_baseline_v1 import atr_stop_triggered
from engine.strategy_baseline_v1_ls13030 import (
    BORROW_FEE_ANNUAL,
    HTB_ADV_PERCENTILE,
    LONG_EXPOSURE,
    SHORT_BASKET_COUNT,
    SHORT_EXPOSURE,
    equal_weight_sleeve_targets,
    refill_sleeve,
    select_monthly_long_short_mom_quality,
    strategy_id as strategy_id_ls,
    strategy_knobs as strategy_knobs_ls,
)


class BaselineEngineLS13030(BaselineEngineV1):
    """Mom+quality 130/30 long-short runner."""

    def __init__(self, config: Optional[dict] = None, config_path: str = "config/config.yaml"):
        cfg = dict(config or {})
        # Force long-only overlays OFF; quality ON for MQ score.
        cfg["enable_quality_factor"] = True
        cfg["enable_regime_exposure"] = False
        cfg["enable_regime_exposure_fast"] = False
        cfg["enable_industry_neutral_ranking"] = False
        cfg["enable_topup_chasing"] = False
        cfg["cost_model"] = cfg.get("cost_model", "corwin_schultz_v2")
        cfg["winsorize_adv"] = cfg.get("winsorize_adv", True)

        # Parent __init__ prints knobs via _active_strategy_knobs — set LS attrs first.
        bcfg = dict(cfg.get("baseline_v1") or {})
        self.LONG_EXPOSURE = float(cfg.get("ls_long_exposure", bcfg.get("LONG_EXPOSURE", LONG_EXPOSURE)))
        self.SHORT_EXPOSURE = float(
            cfg.get("ls_short_exposure", bcfg.get("SHORT_EXPOSURE", SHORT_EXPOSURE))
        )
        self.BORROW_FEE_ANNUAL = float(
            cfg.get("ls_borrow_fee_annual", bcfg.get("BORROW_FEE_ANNUAL", BORROW_FEE_ANNUAL))
        )
        self.HTB_ADV_PERCENTILE = float(
            cfg.get("ls_htb_adv_percentile", bcfg.get("HTB_ADV_PERCENTILE", HTB_ADV_PERCENTILE))
        )
        self.SHORT_BASKET_COUNT = int(
            cfg.get("ls_short_basket_count", bcfg.get("SHORT_BASKET_COUNT", SHORT_BASKET_COUNT))
        )
        self.lowest_closes: Dict[str, float] = {}
        self.current_long_candidates: List[str] = []
        self.current_short_candidates: List[str] = []
        self.position_side: Dict[str, str] = {}
        self.diag_borrow_cost_dollars = 0.0
        self.diag_long_exec_cost_dollars = 0.0
        self.diag_short_exec_cost_dollars = 0.0
        self.diag_ls_months: List[dict] = []
        self.diag_sleeve_daily: List[dict] = []
        self.diag_closed_lots_ls: List[dict] = []
        self._short_open: Dict[str, dict] = {}

        super().__init__(config=cfg, config_path=config_path)

    # ------------------------------------------------------------------ knobs
    def _active_strategy_id(self) -> str:
        return strategy_id_ls()

    def _active_strategy_knobs(self) -> Dict[str, object]:
        knobs = dict(strategy_knobs_ls())
        knobs["LONG_EXPOSURE"] = float(self.LONG_EXPOSURE)
        knobs["SHORT_EXPOSURE"] = float(self.SHORT_EXPOSURE)
        knobs["BORROW_FEE_ANNUAL"] = float(self.BORROW_FEE_ANNUAL)
        knobs["HTB_ADV_PERCENTILE"] = float(self.HTB_ADV_PERCENTILE)
        knobs["SHORT_BASKET_COUNT"] = int(self.SHORT_BASKET_COUNT)
        knobs["enable_quality_factor"] = True
        knobs["structure"] = "130/30_long_short"
        return knobs

    def _equity(self, date_idx: int) -> float:
        current_date = self.close_m.index[date_idx]
        closes = self.close_m.loc[current_date]
        pv = 0.0
        for sym, qty in self.portfolio.items():
            px = closes.get(sym, np.nan)
            if pd.notna(px):
                pv += float(qty) * float(px)
        return float(self.cash + pv)

    def _long_syms(self) -> List[str]:
        return [s for s, q in self.portfolio.items() if int(q) > 0]

    def _short_syms(self) -> List[str]:
        return [s for s, q in self.portfolio.items() if int(q) < 0]

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
        sleeve: Optional[str] = None,
    ) -> float:
        dollars = super()._record_cost(
            date=date,
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            cost_ratio=cost_ratio,
            breakdown=breakdown,
            order_role=order_role,
        )
        # Tag last event + sleeve buckets
        if self.diag_cost_events:
            self.diag_cost_events[-1]["sleeve"] = sleeve
        if sleeve == "LONG":
            self.diag_long_exec_cost_dollars += float(dollars)
        elif sleeve == "SHORT":
            self.diag_short_exec_cost_dollars += float(dollars)
        return dollars

    # ----------------------------------------------------------- short fills
    def _execute_short_open(self, sym, date_idx, qty, fill_price, reason) -> int:
        """Open/increase short: portfolio qty negative; cash credited."""
        if qty <= 0 or pd.isna(fill_price) or float(fill_price) <= 0:
            return 0
        if int(self.portfolio.get(sym, 0)) > 0:
            return 0  # never short a long name
        px = float(fill_price)
        current_date = self.close_m.index[date_idx]
        cap = self._max_shares_participation(sym, date_idx, px, self.PARTICIPATION_CAP_SELL)
        exec_qty = min(int(qty), max(int(cap), 0))
        if exec_qty <= 0:
            return 0
        bd = self._cost_breakdown(sym, date_idx, exec_qty, px)
        cost = float(bd["cost_ratio"])
        self._record_cost(
            date=current_date,
            symbol=sym,
            side="SHORT",
            qty=exec_qty,
            price=px,
            cost_ratio=cost,
            breakdown=bd,
            order_role="short_entry",
            sleeve="SHORT",
        )
        proceeds = exec_qty * px * (1.0 - cost)
        self.cash += proceeds
        self.portfolio[sym] = int(self.portfolio.get(sym, 0)) - exec_qty
        self.position_side[sym] = "SHORT"
        self.lowest_closes[sym] = min(self.lowest_closes.get(sym, px), px)
        if sym not in self._short_open:
            self._short_open[sym] = {
                "entry_date": current_date,
                "entry_price": px,
                "qty": exec_qty,
            }
        else:
            self._short_open[sym]["qty"] = int(self._short_open[sym]["qty"]) + exec_qty
        return exec_qty

    def _execute_cover_fill(self, sym, date_idx, qty, fill_price, reason) -> int:
        """Cover short: buy shares; cash debited; qty moves toward 0."""
        held = int(self.portfolio.get(sym, 0))
        if held >= 0 or qty <= 0:
            return 0
        if pd.isna(fill_price) or float(fill_price) <= 0:
            return 0
        px = float(fill_price)
        current_date = self.close_m.index[date_idx]
        short_shares = abs(held)
        cap = self._max_shares_participation(sym, date_idx, px, self.PARTICIPATION_CAP_BUY)
        exec_qty = min(short_shares, int(qty), max(int(cap), 0))
        if exec_qty <= 0:
            return 0
        bd = self._cost_breakdown(sym, date_idx, exec_qty, px)
        cost = float(bd["cost_ratio"])
        spend = exec_qty * px * (1.0 + cost)
        if spend > self.cash:
            exec_qty = int(self.cash / (px * (1.0 + cost)))
            if exec_qty <= 0:
                return 0
            bd = self._cost_breakdown(sym, date_idx, exec_qty, px)
            cost = float(bd["cost_ratio"])
            spend = exec_qty * px * (1.0 + cost)
        self._record_cost(
            date=current_date,
            symbol=sym,
            side="COVER",
            qty=exec_qty,
            price=px,
            cost_ratio=cost,
            breakdown=bd,
            order_role="short_cover",
            sleeve="SHORT",
        )
        self.cash -= spend
        self.portfolio[sym] = held + exec_qty
        lot = self._short_open.get(sym)
        if self.portfolio[sym] >= 0:
            del self.portfolio[sym]
            self.lowest_closes.pop(sym, None)
            self.position_side.pop(sym, None)
            if lot is not None:
                entry_px = float(lot["entry_price"])
                lot_qty = int(lot["qty"])
                # Short PnL: profit when price falls
                pct = (entry_px / px - 1.0) if px > 0 else float("nan")
                dollar_pnl = float(lot_qty) * (entry_px - px)
                self.diag_closed_lots_ls.append(
                    {
                        "symbol": sym,
                        "side": "SHORT",
                        "entry_date": lot["entry_date"],
                        "exit_date": current_date,
                        "entry_price": entry_px,
                        "exit_price": px,
                        "qty": lot_qty,
                        "pct_return": pct,
                        "dollar_pnl": dollar_pnl,
                        "exit_reason": reason,
                        "holding_days": int((pd.Timestamp(current_date) - pd.Timestamp(lot["entry_date"])).days),
                    }
                )
                self._short_open.pop(sym, None)
        return exec_qty

    def _execute_sell_fill(self, sym, date_idx, qty, fill_price, reason) -> int:
        """Long sells only (positive qty)."""
        held = int(self.portfolio.get(sym, 0))
        if held <= 0:
            return 0
        exec_qty = super()._execute_sell_fill(sym, date_idx, qty, fill_price, reason)
        if exec_qty > 0 and self.diag_cost_events:
            # Re-attribute last SELL cost to LONG sleeve if not already.
            ev = self.diag_cost_events[-1]
            if ev.get("side") == "SELL" and ev.get("sleeve") is None:
                ev["sleeve"] = "LONG"
                self.diag_long_exec_cost_dollars += float(ev.get("cost_dollars") or 0.0)
        if sym not in self.portfolio:
            self.position_side.pop(sym, None)
            self.highest_closes.pop(sym, None)
            # Capture long closed lot side tag
            if self.diag_closed_lots:
                last = self.diag_closed_lots[-1]
                if last.get("symbol") == sym and "side" not in last:
                    last["side"] = "LONG"
                    self.diag_closed_lots_ls.append(dict(last))
        return exec_qty

    def execute_pending_orders(self, date_idx):
        current_date = self.close_m.index[date_idx]
        opens = self.open_m.loc[current_date]
        still_pending = []
        # Raise cash from SHORT opens before BUY so 130% long sleeve can fund.
        # Then SELL/COVER. Preserve relative order within each type.
        type_rank = {"SHORT": 0, "SELL": 1, "COVER": 2, "BUY": 3}
        ordered = sorted(
            self.pending_orders,
            key=lambda o: type_rank.get(str(o.get("type")), 9),
        )
        for order in ordered:
            sym = order["symbol"]
            o_price = opens.get(sym, np.nan)
            if pd.isna(o_price) or o_price <= 0:
                still_pending.append(order)
                continue
            qty = int(order["qty"])
            typ = order["type"]
            if typ == "BUY":
                # Long buy — block if currently short
                if int(self.portfolio.get(sym, 0)) < 0:
                    continue
                order_role = order.get("order_role") or (
                    "top_up" if int(self.portfolio.get(sym, 0)) > 0 else "new_entry"
                )
                target_qty = int(order.get("target_qty", qty) or qty)
                target_notional = float(
                    order.get("target_notional", target_qty * float(o_price))
                )
                cap = self._max_shares_participation(
                    sym, date_idx, o_price, self.PARTICIPATION_CAP_BUY
                )
                exec_qty = min(qty, max(int(cap), 0))
                if exec_qty <= 0:
                    continue
                bd = self._cost_breakdown(sym, date_idx, exec_qty, float(o_price))
                cost = float(bd["cost_ratio"])
                spend = exec_qty * float(o_price) * (1.0 + cost)
                if spend > self.cash:
                    exec_qty = int(self.cash / (float(o_price) * (1.0 + cost)))
                    if exec_qty <= 0:
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
                    sleeve="LONG",
                )
                self.cash -= spend
                self.portfolio[sym] = int(self.portfolio.get(sym, 0)) + exec_qty
                self.position_side[sym] = "LONG"
                px = float(o_price)
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
                        if sym in self.atr14_m.columns
                        and pd.notna(self.atr14_m[sym].iloc[date_idx])
                        else None
                    ),
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
                        "filled_notional": float(exec_qty) * px,
                        "fill_pct_of_target": (
                            (float(exec_qty) * px) / target_notional
                            if target_notional > 0
                            else float("nan")
                        ),
                        "reason": "ok",
                    }
                )
            elif typ == "SELL":
                self._execute_sell_fill(
                    sym, date_idx, qty, float(o_price), order.get("reason", "atr_trail")
                )
            elif typ == "SHORT":
                self._execute_short_open(
                    sym, date_idx, qty, float(o_price), order.get("reason", "rebalance_short")
                )
            elif typ == "COVER":
                self._execute_cover_fill(
                    sym, date_idx, qty, float(o_price), order.get("reason", "atr_cover")
                )
            else:
                still_pending.append(order)
        self.pending_orders = still_pending

    def _handle_delistings(self, date_idx):
        current_date = self.close_m.index[date_idx]
        for sym in list(self.portfolio.keys()):
            meta = self.delisted_meta.get(sym)
            if not (meta and meta.get("delistingDate")):
                continue
            d_date = pd.to_datetime(meta["delistingDate"])
            if current_date != d_date:
                continue
            last_close = self.close_m[sym].iloc[date_idx]
            if pd.isna(last_close):
                del self.portfolio[sym]
                continue
            qty = int(self.portfolio[sym])
            px = float(last_close)
            if qty > 0:
                self.cash += qty * px
                self.trade_journal.on_exit(
                    symbol=sym,
                    date=current_date,
                    date_idx=date_idx,
                    price=px,
                    exit_reason="official_delist",
                    close_m=self.close_m,
                )
            elif qty < 0:
                # Forced buy-in at last close
                spend = abs(qty) * px
                self.cash -= spend
                lot = self._short_open.pop(sym, None)
                if lot is not None:
                    entry_px = float(lot["entry_price"])
                    self.diag_closed_lots_ls.append(
                        {
                            "symbol": sym,
                            "side": "SHORT",
                            "entry_date": lot["entry_date"],
                            "exit_date": current_date,
                            "entry_price": entry_px,
                            "exit_price": px,
                            "qty": abs(qty),
                            "pct_return": (entry_px / px - 1.0) if px > 0 else float("nan"),
                            "dollar_pnl": abs(qty) * (entry_px - px),
                            "exit_reason": "official_delist_cover",
                            "holding_days": int(
                                (pd.Timestamp(current_date) - pd.Timestamp(lot["entry_date"])).days
                            ),
                        }
                    )
            del self.portfolio[sym]
            self.highest_closes.pop(sym, None)
            self.lowest_closes.pop(sym, None)
            self.position_side.pop(sym, None)

        for sym, last_date in self.silent_delist_flags.items():
            if current_date != last_date + pd.Timedelta(days=1) or sym not in self.portfolio:
                continue
            last_valid = self.close_m[sym].iloc[self.close_m.index.get_loc(last_date)]
            if pd.isna(last_valid):
                del self.portfolio[sym]
                continue
            qty = int(self.portfolio[sym])
            px = float(last_valid) * self.SILENT_DELIST_RECOVERY
            if qty > 0:
                self.cash += qty * px
                self.trade_journal.on_exit(
                    symbol=sym,
                    date=current_date,
                    date_idx=date_idx,
                    price=px,
                    exit_reason="silent_delist",
                    close_m=self.close_m,
                )
            elif qty < 0:
                self.cash -= abs(qty) * px
                lot = self._short_open.pop(sym, None)
                if lot is not None:
                    entry_px = float(lot["entry_price"])
                    self.diag_closed_lots_ls.append(
                        {
                            "symbol": sym,
                            "side": "SHORT",
                            "entry_date": lot["entry_date"],
                            "exit_date": current_date,
                            "entry_price": entry_px,
                            "exit_price": px,
                            "qty": abs(qty),
                            "pct_return": (entry_px / px - 1.0) if px > 0 else float("nan"),
                            "dollar_pnl": abs(qty) * (entry_px - px),
                            "exit_reason": "silent_delist_cover",
                            "holding_days": int(
                                (pd.Timestamp(current_date) - pd.Timestamp(lot["entry_date"])).days
                            ),
                        }
                    )
            del self.portfolio[sym]
            self.highest_closes.pop(sym, None)
            self.lowest_closes.pop(sym, None)
            self.position_side.pop(sym, None)

    def _check_atr_exits(self, date_idx):
        current_date = self.close_m.index[date_idx]
        lows = self.low_m.loc[current_date]
        highs = self.high_m.loc[current_date]
        closes = self.close_m.loc[current_date]
        pending = {o["symbol"] for o in self.pending_orders}

        for sym in list(self.portfolio.keys()):
            if sym in pending:
                continue
            qty = int(self.portfolio[sym])
            if qty == 0:
                continue
            c = closes.get(sym, np.nan)
            atr = (
                self.atr14_m[sym].iloc[date_idx]
                if sym in self.atr14_m.columns
                else np.nan
            )
            if pd.isna(atr) or float(atr) <= 0 or pd.isna(c):
                continue

            if qty > 0:
                # Long trail
                self.highest_closes[sym] = max(
                    self.highest_closes.get(sym, float(c)), float(c)
                )
                day_low = lows.get(sym, np.nan)
                if pd.isna(day_low):
                    continue
                if atr_stop_triggered(
                    float(day_low),
                    self.highest_closes[sym],
                    float(atr),
                    self.ATR_MULTIPLIER,
                ):
                    self.pending_orders.append(
                        {
                            "symbol": sym,
                            "qty": qty,
                            "type": "SELL",
                            "reason": f"atr_trail(mult={self.ATR_MULTIPLIER})",
                        }
                    )
            else:
                # Short mirrored trail: cover if High > trough + mult*ATR
                self.lowest_closes[sym] = min(
                    self.lowest_closes.get(sym, float(c)), float(c)
                )
                day_high = highs.get(sym, np.nan)
                if pd.isna(day_high):
                    continue
                stop = float(self.lowest_closes[sym]) + float(self.ATR_MULTIPLIER) * float(atr)
                if float(day_high) > stop:
                    self.pending_orders.append(
                        {
                            "symbol": sym,
                            "qty": abs(qty),
                            "type": "COVER",
                            "reason": f"atr_cover(mult={self.ATR_MULTIPLIER})",
                        }
                    )

    def _accrue_borrow(self, date_idx) -> float:
        """Daily borrow ≈ annual_fee / 252 * short_notional."""
        current_date = self.close_m.index[date_idx]
        if current_date < self._run_start:
            return 0.0
        closes = self.close_m.loc[current_date]
        short_notional = 0.0
        for sym, qty in self.portfolio.items():
            if int(qty) >= 0:
                continue
            px = closes.get(sym, np.nan)
            if pd.notna(px) and float(px) > 0:
                short_notional += abs(int(qty)) * float(px)
        if short_notional <= 0:
            return 0.0
        fee = float(self.BORROW_FEE_ANNUAL) / 252.0 * short_notional
        self.cash -= fee
        self.diag_borrow_cost_dollars += fee
        self.diag_total_cost_dollars += fee
        self.diag_cost_events.append(
            {
                "date": current_date,
                "symbol": None,
                "side": "BORROW",
                "qty": 0,
                "price": 0.0,
                "cost_ratio": float(self.BORROW_FEE_ANNUAL) / 252.0,
                "cost_dollars": float(fee),
                "notional": float(short_notional),
                "order_role": "borrow_fee",
                "sleeve": "SHORT",
            }
        )
        return fee

    def _queue_ls_rebalance(self, long_targets: Dict[str, float], short_targets: Dict[str, float], date_idx: int):
        """Queue BUY/SELL for longs and SHORT/COVER for shorts (no-chase per sleeve)."""
        current_date = self.close_m.index[date_idx]
        closes = self.close_m.loc[current_date]
        total_equity = max(self._equity(date_idx), 1.0)

        # --- Long sleeve ---
        long_held = set(self._long_syms())
        long_all = long_held | set(long_targets.keys())
        for sym in long_all:
            price = closes.get(sym, np.nan)
            if pd.isna(price) or float(price) <= 0:
                continue
            if int(self.portfolio.get(sym, 0)) < 0:
                continue  # skip if short
            in_targets = sym in long_targets and float(long_targets.get(sym, 0.0)) > 0
            target_value = float(long_targets.get(sym, 0.0)) * total_equity
            currently_held = int(self.portfolio.get(sym, 0)) > 0
            current_value = max(int(self.portfolio.get(sym, 0)), 0) * float(price)
            delta = target_value - current_value
            if abs(delta) < total_equity * 0.001:
                continue
            qty = int(abs(delta) / float(price))
            if qty <= 0:
                continue
            side = "BUY" if delta > 0 else "SELL"
            if self.DISABLE_TOPUP_CHASING and currently_held:
                if side == "BUY":
                    continue
                if side == "SELL" and in_targets:
                    continue
            self.pending_orders.append(
                {
                    "symbol": sym,
                    "qty": qty if side == "BUY" else (
                        int(self.portfolio.get(sym, 0)) if not in_targets else qty
                    ),
                    "type": side,
                    "reason": (
                        "rebalance_entry"
                        if side == "BUY" and not currently_held
                        else "rebalance_topup"
                        if side == "BUY"
                        else "rebalance_dropout"
                        if side == "SELL" and not in_targets
                        else "rebalance_trim"
                    ),
                    "order_role": (
                        "new_entry"
                        if side == "BUY" and not currently_held
                        else "top_up"
                        if side == "BUY"
                        else None
                    ),
                    "target_qty": qty,
                    "target_notional": float(qty) * float(price),
                }
            )

        # --- Short sleeve (targets are negative weights) ---
        short_held = set(self._short_syms())
        short_tgt_syms = set(short_targets.keys())
        short_all = short_held | short_tgt_syms
        for sym in short_all:
            price = closes.get(sym, np.nan)
            if pd.isna(price) or float(price) <= 0:
                continue
            if int(self.portfolio.get(sym, 0)) > 0:
                continue  # never short a long
            in_targets = sym in short_targets and float(short_targets.get(sym, 0.0)) < 0
            # target_value is negative for shorts
            target_value = float(short_targets.get(sym, 0.0)) * total_equity
            currently_short = int(self.portfolio.get(sym, 0)) < 0
            current_value = min(int(self.portfolio.get(sym, 0)), 0) * float(price)  # ≤ 0
            delta = target_value - current_value
            # more short (more negative) → SHORT; less short → COVER
            if abs(delta) < total_equity * 0.001:
                continue
            qty = int(abs(delta) / float(price))
            if qty <= 0:
                continue
            if delta < 0:
                # Need more short exposure
                if self.DISABLE_TOPUP_CHASING and currently_short:
                    continue  # no short top-up chase
                self.pending_orders.append(
                    {
                        "symbol": sym,
                        "qty": qty,
                        "type": "SHORT",
                        "reason": "rebalance_short" if not currently_short else "rebalance_short_topup",
                        "order_role": "short_entry",
                        "target_qty": qty,
                        "target_notional": float(qty) * float(price),
                    }
                )
            else:
                # Need to reduce short (cover)
                if self.DISABLE_TOPUP_CHASING and currently_short and in_targets:
                    continue  # no partial cover while still selected
                cover_qty = abs(int(self.portfolio.get(sym, 0))) if not in_targets else qty
                self.pending_orders.append(
                    {
                        "symbol": sym,
                        "qty": cover_qty,
                        "type": "COVER",
                        "reason": "rebalance_cover" if not in_targets else "rebalance_cover_trim",
                        "order_role": "short_cover",
                        "target_qty": cover_qty,
                        "target_notional": float(cover_qty) * float(price),
                    }
                )

    def _log_sleeve_day(self, date_idx: int):
        current_date = self.close_m.index[date_idx]
        if current_date < self._run_start:
            return
        closes = self.close_m.loc[current_date]
        long_notional = 0.0
        short_notional = 0.0
        for sym, qty in self.portfolio.items():
            px = closes.get(sym, np.nan)
            if pd.isna(px):
                continue
            v = float(qty) * float(px)
            if qty > 0:
                long_notional += v
            elif qty < 0:
                short_notional += abs(v)
        eq = self._equity(date_idx)
        self.diag_sleeve_daily.append(
            {
                "date": current_date,
                "equity": eq,
                "long_notional": long_notional,
                "short_notional": short_notional,
                "net_exposure": (long_notional - short_notional) / eq if eq > 0 else None,
                "gross_exposure": (long_notional + short_notional) / eq if eq > 0 else None,
                "n_long": len(self._long_syms()),
                "n_short": len(self._short_syms()),
                "cash": float(self.cash),
            }
        )

    def run(self) -> pd.DataFrame:
        print("=" * 50)
        print(f"BASELINE STRATEGY: {self._active_strategy_id()}")
        print("=" * 50)
        for k, v in self._active_strategy_knobs().items():
            print(f"  {k}: {v}")
        print(f"  cost_model: {self.COST_MODEL}")
        print(f"  enable_topup_chasing: {self.ENABLE_TOPUP_CHASING}")
        print("=" * 50)

        trading_days = self.close_m.index
        months = trading_days.to_period("M")
        month_starts = set(pd.to_datetime(trading_days[~months.duplicated(keep="first")]))

        warmup = 252
        for idx, current_date in enumerate(trading_days):
            if idx < warmup:
                continue
            if current_date > self._run_end:
                break
            in_window = current_date >= self._run_start

            self._handle_delistings(idx)
            n_long_before = len(self._long_syms())
            n_short_before = len(self._short_syms())
            self.execute_pending_orders(idx)
            long_freed = n_long_before > len(self._long_syms())
            short_freed = n_short_before > len(self._short_syms())
            self._maybe_buy_qqq(idx)
            self._check_atr_exits(idx)
            self._accrue_borrow(idx)

            is_month_start = current_date in month_starts and in_window
            need_initial = (
                in_window
                and not self.current_long_candidates
                and not self.portfolio
            )
            if is_month_start or need_initial:
                live = [
                    s
                    for s in self._investable
                    if s in self.close_m.columns and pd.notna(self.close_m[s].iloc[idx])
                ]
                longs, shorts, sel_diag = select_monthly_long_short_mom_quality(
                    date_idx=idx,
                    close_m=self.close_m,
                    dvol_m=self.dvol_m,
                    mom_12_1_m=self.mom_12_1_m,
                    eligible_symbols=live,
                    fundamental_history=self.fundamental_history,
                    fund_ts=self._fund_ts,
                    top_liquid_pool=self.TOP_LIQUID_POOL,
                    long_count=self.TOP_MOMENTUM_COUNT,
                    short_count=self.SHORT_BASKET_COUNT,
                    htb_adv_percentile=self.HTB_ADV_PERCENTILE,
                )
                self.current_long_candidates = longs
                self.current_short_candidates = shorts
                self._log_industry_concentration(current_date, longs)

                pending_exit = {
                    o["symbol"]
                    for o in self.pending_orders
                    if o.get("type") in ("SELL", "COVER")
                }
                eff_long = [s for s in self._long_syms() if s not in pending_exit]
                eff_short = [s for s in self._short_syms() if s not in pending_exit]

                sel_long = refill_sleeve(
                    eff_long, longs, max_positions=self.TOP_MOMENTUM_COUNT
                )
                sel_short = refill_sleeve(
                    eff_short, shorts, max_positions=self.SHORT_BASKET_COUNT
                )
                # Avoid overlap if refill kept something that is now in other sleeve candidates
                overlap = set(sel_long) & set(sel_short)
                if overlap:
                    sel_short = [s for s in sel_short if s not in overlap]

                long_targets = equal_weight_sleeve_targets(sel_long, self.LONG_EXPOSURE)
                short_targets = equal_weight_sleeve_targets(sel_short, -self.SHORT_EXPOSURE)
                self._queue_ls_rebalance(long_targets, short_targets, idx)

                self.diag_ls_months.append(
                    {
                        "date": current_date,
                        **sel_diag,
                        "n_long_candidates": len(longs),
                        "n_short_candidates": len(shorts),
                        "n_long_selected": len(sel_long),
                        "n_short_selected": len(sel_short),
                    }
                )
                print(
                    f"[REBALANCE-LS] {current_date.date()} "
                    f"long_cand={len(longs)} short_cand={len(shorts)} "
                    f"held_L={len(eff_long)} held_S={len(eff_short)} "
                    f"tgt_L={len(sel_long)} tgt_S={len(sel_short)} "
                    f"exp={self.LONG_EXPOSURE:.2f}/{-self.SHORT_EXPOSURE:.2f}"
                )

            elif (
                (long_freed or short_freed)
                and current_date >= self._run_start
                and (self.current_long_candidates or self.current_short_candidates)
            ):
                pending_exit = {
                    o["symbol"]
                    for o in self.pending_orders
                    if o.get("type") in ("SELL", "COVER")
                }
                eff_long = [s for s in self._long_syms() if s not in pending_exit]
                eff_short = [s for s in self._short_syms() if s not in pending_exit]
                need_long = len(eff_long) < self.TOP_MOMENTUM_COUNT and self.current_long_candidates
                need_short = (
                    len(eff_short) < self.SHORT_BASKET_COUNT and self.current_short_candidates
                )
                if need_long or need_short:
                    sel_long = refill_sleeve(
                        eff_long,
                        self.current_long_candidates,
                        max_positions=self.TOP_MOMENTUM_COUNT,
                    )
                    sel_short = refill_sleeve(
                        eff_short,
                        self.current_short_candidates,
                        max_positions=self.SHORT_BASKET_COUNT,
                    )
                    overlap = set(sel_long) & set(sel_short)
                    if overlap:
                        sel_short = [s for s in sel_short if s not in overlap]
                    long_targets = equal_weight_sleeve_targets(sel_long, self.LONG_EXPOSURE)
                    short_targets = equal_weight_sleeve_targets(
                        sel_short, -self.SHORT_EXPOSURE
                    )
                    self._queue_ls_rebalance(long_targets, short_targets, idx)

            # Mark-to-market
            eq = self._equity(idx)
            if current_date >= self._run_start:
                self.equity_curve.append({"Date": current_date, "Total_Equity": eq})
                self.diag_equity_curve.append(
                    {
                        "Date": current_date,
                        "Net_Equity": eq,
                        "Gross_Equity": eq + float(self.diag_total_cost_dollars),
                        "Cum_Cost_Dollars": float(self.diag_total_cost_dollars),
                        "Borrow_Cost_Dollars": float(self.diag_borrow_cost_dollars),
                    }
                )
                self._log_sleeve_day(idx)
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
        self.research_artifacts = self._emit_reports_ls(equity, qqq_eq)
        return equity

    def _emit_reports_ls(self, equity: pd.DataFrame, qqq_eq: pd.DataFrame) -> dict:
        out = Path("cache") / "baseline_v1_ls13030"
        out.mkdir(parents=True, exist_ok=True)

        trades = self.trade_journal.to_frame()
        trades_path = out / "trade_journal_long.csv"
        if trades is not None and not trades.empty:
            trades.to_csv(trades_path, index=False)
        else:
            trades_path.write_text("", encoding="utf-8")

        lots = pd.DataFrame(self.diag_closed_lots_ls)
        lots_path = out / "closed_lots_ls.csv"
        if not lots.empty:
            lots.to_csv(lots_path, index=False)
        else:
            lots_path.write_text("", encoding="utf-8")

        bm_rets = None
        if not qqq_eq.empty and "Total_Equity" in qqq_eq.columns:
            bm_rets = qqq_eq["Total_Equity"].pct_change()

        kpi = build_hierarchical_kpi_report(
            equity, trades if trades is not None else pd.DataFrame(), benchmark_returns=bm_rets
        )
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
            "structure": "130/30",
            "costs": {
                "total": float(self.diag_total_cost_dollars),
                "long_exec": float(self.diag_long_exec_cost_dollars),
                "short_exec": float(self.diag_short_exec_cost_dollars),
                "borrow": float(self.diag_borrow_cost_dollars),
            },
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
        if self.diag_sleeve_daily:
            pd.DataFrame(self.diag_sleeve_daily).to_csv(out / "sleeve_daily.csv", index=False)

        return {
            "kpi": kpi,
            "qqq_kpi": qqq_kpi,
            "comparison": comparison,
            "trades_path": str(trades_path),
            "lots_path": str(lots_path),
            "out_dir": str(out),
        }
