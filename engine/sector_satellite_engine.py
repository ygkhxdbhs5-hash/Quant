"""Sector-restricted MQ satellite engine (wrapper; Strategy #1 core untouched)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from engine.baseline_engine import BaselineEngineV1
from engine.research.kpi_report import build_hierarchical_kpi_report
from engine.sector_taxonomy import SATELLITE_SECTORS, sector_of_meta
from engine.strategy import load_config
from engine.strategy_baseline_v1_sector import (
    K_MIN,
    select_monthly_candidates_mq_sector,
    strategy_id,
    strategy_knobs,
)


class SectorSatelliteEngine(BaselineEngineV1):
    """Momentum+quality with eligible universe restricted to one sector bucket."""

    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: str = "config/config.yaml",
        *,
        sector_bucket: str,
    ):
        if sector_bucket not in SATELLITE_SECTORS:
            raise ValueError(
                f"sector_bucket must be one of {SATELLITE_SECTORS}, got {sector_bucket!r}"
            )
        cfg = dict(config or load_config(config_path))
        cfg["enable_quality_factor"] = True
        cfg["enable_regime_exposure"] = False
        cfg["enable_regime_exposure_fast"] = False
        cfg["enable_industry_neutral_ranking"] = False
        cfg["enable_topup_chasing"] = False
        cfg.pop("disable_topup_chasing", None)
        cfg.setdefault("cost_model", "corwin_schultz_v2")
        cfg.setdefault("winsorize_adv", True)
        cfg["fixed_leverage"] = 1.0
        cfg["enable_cash_interest"] = False

        self.sector_bucket = sector_bucket
        self.diag_sector_months: List[dict] = []
        self.diag_holdings_sector: List[dict] = []

        super().__init__(config=cfg, config_path=config_path)

    def _active_strategy_id(self) -> str:
        return strategy_id(self.sector_bucket)

    def _active_strategy_knobs(self) -> Dict[str, object]:
        knobs = dict(strategy_knobs(self.sector_bucket))
        knobs["cost_model"] = self.COST_MODEL
        knobs["enable_topup_chasing"] = bool(self.ENABLE_TOPUP_CHASING)
        return knobs

    def _select_candidates(self, idx: int, live: List[str]) -> List[str]:
        cands, diag = select_monthly_candidates_mq_sector(
            date_idx=idx,
            close_m=self.close_m,
            dvol_m=self.dvol_m,
            mom_12_1_m=self.mom_12_1_m,
            eligible_symbols=live,
            fundamental_history=self.fundamental_history,
            fund_ts=self._fund_ts,
            profile_meta=self.profile_meta,
            sector_bucket=self.sector_bucket,
            top_liquid_pool=self.TOP_LIQUID_POOL,
        )
        diag = dict(diag)
        diag["date"] = self.close_m.index[idx]
        self.diag_sector_months.append(diag)
        # Drive refill max_positions for this month (parent run() reads TOP_MOMENTUM_COUNT).
        k = int(diag.get("adaptive_k") or 0)
        self.TOP_MOMENTUM_COUNT = int(k if k > 0 else K_MIN)
        return cands

    def _record_holdings_sector(self, current_date) -> None:
        if not self.portfolio:
            return
        closes = self.close_m.loc[current_date]
        rows = []
        total = 0.0
        for sym, qty in self.portfolio.items():
            px = closes.get(sym)
            if px is None or pd.isna(px):
                continue
            val = float(qty) * float(px)
            total += val
            rows.append((sym, val, sector_of_meta(self.profile_meta.get(sym))))
        if total <= 0:
            return
        by_sec: Dict[str, float] = {}
        n_by: Dict[str, int] = {}
        for _, val, sec in rows:
            by_sec[sec] = by_sec.get(sec, 0.0) + val
            n_by[sec] = n_by.get(sec, 0) + 1
        self.diag_holdings_sector.append(
            {
                "date": current_date,
                "n_holdings": len(rows),
                "pct_capital": {k: v / total for k, v in by_sec.items()},
                "pct_names": {k: n / len(rows) for k, n in n_by.items()},
            }
        )

    def _emit_reports(self, equity: pd.DataFrame, qqq_eq: pd.DataFrame) -> dict:
        out = Path("cache") / "satellite1_sector" / self.sector_bucket.replace(" ", "_")
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
        self._print_comparison(comparison)
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
        # Use parent run, but hook holdings snapshot on month starts via wrapping
        # mark-to-market by overriding execute path lightly: call parent.run after
        # patching _mark path. Simpler: copy-free — run parent then post-hoc can't
        # recover holdings. So instrument by subclassing the loop... 
        # Minimal change: wrap _queue_rebalance and snapshot after each day in a
        # custom run copied from parent is heavy. Instead snapshot in a monkeypatch
        # of equity append by overriding a small helper.
        return self._run_with_holdings_diag()

    def _run_with_holdings_diag(self) -> pd.DataFrame:
        """Parent run() with monthly holdings-sector snapshots."""
        # Call into BaselineEngineV1.run but we need hooks — duplicate the thin
        # wrapper: invoke super().run after injecting a callback via diag list
        # written from an overridden method used every day.
        # Override: after super constructs, we replace nothing — use composition:
        equity_before = len(self.equity_curve)

        # Patch: wrap _mark_qqq which is called once per day in-window to also
        # record holdings (side-effect ok).
        orig_mark = self._mark_qqq

        def _mark_and_snap(date_idx):
            orig_mark(date_idx)
            current_date = self.close_m.index[date_idx]
            if current_date >= self._run_start:
                # snapshot on month-start trading days only
                months = self.close_m.index.to_period("M")
                # if this date is first of its month in panel
                if date_idx == 0 or months[date_idx] != months[date_idx - 1]:
                    self._record_holdings_sector(current_date)

        self._mark_qqq = _mark_and_snap  # type: ignore
        try:
            return BaselineEngineV1.run(self)
        finally:
            self._mark_qqq = orig_mark  # type: ignore
            _ = equity_before
