"""Composite Institutional Entry Score — Z-score aggregation.

Composite =
  0.35 × Quality + 0.35 × Momentum + 0.15 × Trend + 0.10 × Liquidity − 0.05 × Risk
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from engine.institutional_entry.liquidity import liquidity_raw_row
from engine.institutional_entry.momentum import momentum_raw_row
from engine.institutional_entry.quality import quality_raw_row
from engine.institutional_entry.risk import risk_raw_row
from engine.institutional_entry.trend import (
    compute_adx_series,
    trend_raw_row,
)
from engine.institutional_entry.zscore import cross_sectional_zscore, nanmean_row

CATEGORY_WEIGHTS = {
    "quality": 0.35,
    "momentum": 0.35,
    "trend": 0.15,
    "liquidity": 0.10,
    "risk": -0.05,
}

QUALITY_COLS = ["gp_raw", "op_raw", "roe_raw", "roa_raw"]
MOMENTUM_COLS = ["mom_12_1_raw", "mom_6m_raw", "mom_3m_raw", "rs_bench_raw"]
TREND_COLS = ["px_gt_ema200_raw", "ema50_gt_ema200_raw", "adx_raw"]
LIQUIDITY_COLS = ["dvol_raw", "turnover_raw", "amihud_liq_raw"]
RISK_COLS = ["beta_raw", "idiovol_raw", "maxdd_raw"]


def _safe_get(panel: Optional[pd.DataFrame], sym: str, date_idx: int, default=np.nan) -> float:
    try:
        if panel is None or sym not in panel.columns:
            return float(default)
        val = panel[sym].iloc[date_idx]
        return float(val) if pd.notna(val) else float(default)
    except Exception:
        return float(default)


def _series_or_none(panel: Optional[pd.DataFrame], sym: str) -> Optional[pd.Series]:
    try:
        if panel is None or sym not in panel.columns:
            return None
        return panel[sym]
    except Exception:
        return None


def collect_raw_factors(
    engine: Any,
    date_idx: int,
    symbols: Sequence[str],
) -> pd.DataFrame:
    """Build one row per symbol with all raw academic factor values (PIT)."""
    current_date = engine.close_m.index[date_idx]
    close_m = engine.close_m
    high_m = getattr(engine, "high_m", None)
    low_m = getattr(engine, "low_m", None)
    dvol_m = getattr(engine, "dvol_m", None)
    ret_m = getattr(engine, "ret1_m", None)
    if ret_m is None:
        ret_m = close_m.pct_change()

    # Benchmark: prefer SPY, else configured BENCHMARK_TICKER
    bench = "SPY" if "SPY" in close_m.columns else getattr(engine, "BENCHMARK_TICKER", None)
    bench_close = close_m[bench] if bench and bench in close_m.columns else None
    bench_ret = ret_m[bench] if bench and bench in ret_m.columns else None

    # EMA200 / EMA50 (PIT ewm up to date_idx inclusive — use full series then iloc)
    ema50_m = getattr(engine, "ema50_m", None)
    if not hasattr(engine, "ema200_m") or engine.ema200_m is None:
        engine.ema200_m = close_m.ewm(span=200, adjust=False).mean()
    ema200_m = engine.ema200_m

    # ADX cache (compute once per backtest lazily)
    if not hasattr(engine, "_adx_cache"):
        engine._adx_cache = {}

    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            price = _safe_get(close_m, sym, date_idx)
            if not np.isfinite(price) or price <= 0:
                continue
            fund = engine.get_latest_available_fundamentals(sym, current_date)

            q = quality_raw_row(fund)
            stock_close = close_m[sym]
            m = momentum_raw_row(stock_close, bench_close, date_idx)

            ema50 = _safe_get(ema50_m, sym, date_idx)
            ema200 = _safe_get(ema200_m, sym, date_idx)
            if sym not in engine._adx_cache:
                h = _series_or_none(high_m, sym)
                l = _series_or_none(low_m, sym)
                if h is not None and l is not None:
                    engine._adx_cache[sym] = compute_adx_series(h, l, stock_close)
                else:
                    engine._adx_cache[sym] = pd.Series(np.nan, index=close_m.index)
            adx_s = engine._adx_cache[sym]
            adx = float(adx_s.iloc[date_idx]) if date_idx < len(adx_s) and pd.notna(adx_s.iloc[date_idx]) else float("nan")
            t = trend_raw_row(price, ema50, ema200, adx)

            adv = _safe_get(getattr(engine, "adv20_m", None), sym, date_idx)
            stock_ret = ret_m[sym] if sym in ret_m.columns else stock_close.pct_change()
            dvol_s = dvol_m[sym] if dvol_m is not None and sym in dvol_m.columns else pd.Series(np.nan, index=close_m.index)
            liq = liquidity_raw_row(adv, price, fund, stock_ret, dvol_s, date_idx)
            rsk = risk_raw_row(stock_ret, bench_ret, stock_close, date_idx)

            industry = engine.profile_meta.get(sym, {}).get("industry", "Unknown")
            row = {
                "symbol": sym,
                "price": price,
                "industry": industry,
                "bench": bench,
                **q,
                **m,
                **t,
                **liq,
                **rsk,
            }
            rows.append(row)
        except Exception:
            continue

    return pd.DataFrame(rows)


def _z_block(df: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            out[f"{prefix}_{c.replace('_raw', '')}_z"] = np.nan
            continue
        z = cross_sectional_zscore(df[c])
        out[f"{c.replace('_raw', '')}_z"] = z
    return out


def build_institutional_scores(
    engine: Any,
    date_idx: int,
    symbols: Sequence[str],
) -> pd.DataFrame:
    """Collect raw factors → Z-scores → category scores → composite final_score."""
    raw = collect_raw_factors(engine, date_idx, symbols)
    if raw.empty:
        return raw

    qz = pd.DataFrame({c: cross_sectional_zscore(raw[c]) for c in QUALITY_COLS if c in raw.columns})
    mz = pd.DataFrame({c: cross_sectional_zscore(raw[c]) for c in MOMENTUM_COLS if c in raw.columns})
    tz = pd.DataFrame({c: cross_sectional_zscore(raw[c]) for c in TREND_COLS if c in raw.columns})
    lz = pd.DataFrame({c: cross_sectional_zscore(raw[c]) for c in LIQUIDITY_COLS if c in raw.columns})
    rz = pd.DataFrame({c: cross_sectional_zscore(raw[c]) for c in RISK_COLS if c in raw.columns})

    # Rename z columns for logging clarity
    def _rename(block: pd.DataFrame) -> pd.DataFrame:
        return block.rename(columns={c: c.replace("_raw", "_z") for c in block.columns})

    qz, mz, tz, lz, rz = map(_rename, (qz, mz, tz, lz, rz))

    quality = nanmean_row(qz)
    momentum = nanmean_row(mz)
    trend = nanmean_row(tz)
    liquidity = nanmean_row(lz)
    risk = nanmean_row(rz)

    composite = (
        CATEGORY_WEIGHTS["quality"] * quality
        + CATEGORY_WEIGHTS["momentum"] * momentum
        + CATEGORY_WEIGHTS["trend"] * trend
        + CATEGORY_WEIGHTS["liquidity"] * liquidity
        + CATEGORY_WEIGHTS["risk"] * risk  # weight already negative
    )

    out = raw.copy()
    for block in (qz, mz, tz, lz, rz):
        for c in block.columns:
            out[c] = block[c].to_numpy()
    out["quality_score"] = quality.to_numpy()
    out["momentum_score"] = momentum.to_numpy()
    out["trend_score_inst"] = trend.to_numpy()
    out["liquidity_score"] = liquidity.to_numpy()
    out["risk_score"] = risk.to_numpy()
    out["final_score"] = composite.to_numpy()
    out["factor_mode"] = "institutional_v1"
    # Compatibility aliases used by exit diagnostics / journals
    out["cmvs_score"] = out["final_score"]
    out["eqs"] = 0.0
    return out.sort_values("final_score", ascending=False).reset_index(drop=True)


def format_factor_log(df: pd.DataFrame, *, max_rows: Optional[int] = None) -> str:
    """Human-readable log: raw factors, Z-scores, composite for each stock."""
    if df.empty:
        return "[INSTITUTIONAL ENTRY] (no names)"
    cols_pref = [
        "symbol",
        "final_score",
        "quality_score",
        "momentum_score",
        "trend_score_inst",
        "liquidity_score",
        "risk_score",
        "gp_raw",
        "op_raw",
        "roe_raw",
        "roa_raw",
        "mom_12_1_raw",
        "mom_6m_raw",
        "mom_3m_raw",
        "rs_bench_raw",
        "px_gt_ema200_raw",
        "ema50_gt_ema200_raw",
        "adx_raw",
        "dvol_raw",
        "turnover_raw",
        "amihud_raw",
        "beta_raw",
        "idiovol_raw",
        "maxdd_raw",
        "gp_z",
        "op_z",
        "roe_z",
        "roa_z",
        "mom_12_1_z",
        "mom_6m_z",
        "mom_3m_z",
        "rs_bench_z",
        "px_gt_ema200_z",
        "ema50_gt_ema200_z",
        "adx_z",
        "dvol_z",
        "turnover_z",
        "amihud_liq_z",
        "beta_z",
        "idiovol_z",
        "maxdd_z",
    ]
    cols = [c for c in cols_pref if c in df.columns]
    view = df[cols] if max_rows is None else df[cols].head(int(max_rows))
    lines = [
        "[INSTITUTIONAL ENTRY] Composite = "
        "0.35·Q + 0.35·M + 0.15·T + 0.10·L − 0.05·R (cross-sectional Z)",
        f"n={len(df)} logged={len(view)}",
    ]
    # Per-stock compact lines
    for _, r in (df if max_rows is None else df.head(int(max_rows))).iterrows():
        try:
            lines.append(
                f"  {r['symbol']}: composite={float(r['final_score']):+.3f} "
                f"Q={float(r['quality_score']):+.2f} M={float(r['momentum_score']):+.2f} "
                f"T={float(r['trend_score_inst']):+.2f} L={float(r['liquidity_score']):+.2f} "
                f"R={float(r['risk_score']):+.2f} | "
                f"raw GP={r.get('gp_raw', np.nan)} OP={r.get('op_raw', np.nan)} "
                f"ROE={r.get('roe_raw', np.nan)} ROA={r.get('roa_raw', np.nan)} "
                f"mom12-1={r.get('mom_12_1_raw', np.nan)} "
                f"ADX={r.get('adx_raw', np.nan)} beta={r.get('beta_raw', np.nan)} "
                f"maxDD={r.get('maxdd_raw', np.nan)}"
            )
        except Exception:
            continue
    # Also attach a truncated wide table for Z diagnostics
    try:
        lines.append(view.round(4).to_string(index=False))
    except Exception:
        pass
    return "\n".join(lines)
