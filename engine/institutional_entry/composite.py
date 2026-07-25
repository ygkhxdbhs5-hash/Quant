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
from engine.institutional_entry.risk import (
    beta_idiovol_panel_at,
    max_drawdown_panel_at,
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


def _panel_row(panel: Optional[pd.DataFrame], symbols: Sequence[str], date_idx: int) -> np.ndarray:
    """Float row for ``symbols`` at ``date_idx`` (NaN if missing)."""
    n = len(symbols)
    out = np.full(n, np.nan, dtype=float)
    if panel is None or n == 0:
        return out
    cols = [s for s in symbols if s in panel.columns]
    if not cols:
        return out
    try:
        row = panel[cols].iloc[date_idx]
    except Exception:
        return out
    col_to_i = {s: i for i, s in enumerate(symbols)}
    for c in cols:
        v = row[c]
        if pd.notna(v):
            out[col_to_i[c]] = float(v)
    return out


def collect_raw_factors(
    engine: Any,
    date_idx: int,
    symbols: Sequence[str],
) -> pd.DataFrame:
    """Build one row per symbol with all raw academic factor values (PIT).

    Uses precomputed panels when available (momentum / ADX / Amihud / EMAs)
    and vectorized beta / max-DD for the active symbol subset.
    """
    symbols = [str(s) for s in symbols]
    if not symbols:
        return pd.DataFrame()

    current_date = engine.close_m.index[date_idx]
    close_m = engine.close_m
    high_m = getattr(engine, "high_m", None)
    low_m = getattr(engine, "low_m", None)
    dvol_m = getattr(engine, "dvol_m", None)
    ret_m = getattr(engine, "ret1_m", None)
    if ret_m is None:
        ret_m = close_m.pct_change()

    bench = "SPY" if "SPY" in close_m.columns else getattr(engine, "BENCHMARK_TICKER", None)
    bench_ret = ret_m[bench] if bench and bench in ret_m.columns else None
    bench_mom6 = (
        _safe_get(getattr(engine, "mom_6m_m", None), bench, date_idx)
        if bench
        else float("nan")
    )
    if not np.isfinite(bench_mom6) and bench and bench in close_m.columns:
        # Fallback if mom_6m panel missing
        try:
            if date_idx >= 126:
                p0 = float(close_m[bench].iloc[date_idx])
                p1 = float(close_m[bench].iloc[date_idx - 126])
                if p1 != 0 and np.isfinite(p0) and np.isfinite(p1):
                    bench_mom6 = p0 / p1 - 1.0
        except Exception:
            bench_mom6 = float("nan")

    ema50_m = getattr(engine, "ema50_m", None)
    if not hasattr(engine, "ema200_m") or engine.ema200_m is None:
        engine.ema200_m = close_m.ewm(span=200, adjust=False).mean()
    ema200_m = engine.ema200_m

    # Ensure ADX panel exists (computed once in _precompute_matrices when possible)
    adx_m = getattr(engine, "adx14_m", None)
    if adx_m is None:
        if not hasattr(engine, "_adx_cache"):
            engine._adx_cache = {}
        # Lazy one-shot panel for requested symbols only
        if high_m is not None and low_m is not None:
            need = [s for s in symbols if s in close_m.columns and s not in engine._adx_cache]
            if need:
                from engine.institutional_entry.trend import compute_adx_panel

                sub = compute_adx_panel(high_m[need], low_m[need], close_m[need])
                for s in need:
                    engine._adx_cache[s] = sub[s]
            # Build a lightweight view from cache for this call
            adx_parts = {}
            for s in symbols:
                if s in engine._adx_cache:
                    adx_parts[s] = engine._adx_cache[s]
            adx_m = pd.DataFrame(adx_parts, index=close_m.index) if adx_parts else None

    # --- Vectorized market / technical cross-section ---
    present = [s for s in symbols if s in close_m.columns]
    if not present:
        return pd.DataFrame()

    price = _panel_row(close_m, present, date_idx)
    valid_mask = np.isfinite(price) & (price > 0)
    present = [s for s, ok in zip(present, valid_mask) if ok]
    if not present:
        return pd.DataFrame()
    price = price[valid_mask]

    mom_12_1 = _panel_row(getattr(engine, "mom_12_1_m", None), present, date_idx)
    mom_6m = _panel_row(getattr(engine, "mom_6m_m", None), present, date_idx)
    mom_3m = _panel_row(getattr(engine, "mom_3m_m", None), present, date_idx)
    # Fallback per-symbol only where panel missing (should be rare after precompute)
    if getattr(engine, "mom_6m_m", None) is None:
        for i, s in enumerate(present):
            m = momentum_raw_row(close_m[s], close_m[bench] if bench and bench in close_m.columns else None, date_idx)
            mom_12_1[i] = m["mom_12_1_raw"]
            mom_6m[i] = m["mom_6m_raw"]
            mom_3m[i] = m["mom_3m_raw"]

    rs_bench = np.full(len(present), np.nan, dtype=float)
    if np.isfinite(bench_mom6):
        rs_bench = mom_6m - bench_mom6

    ema50 = _panel_row(ema50_m, present, date_idx)
    ema200 = _panel_row(ema200_m, present, date_idx)
    adx = _panel_row(adx_m, present, date_idx)
    px_gt = np.where(
        np.isfinite(price) & np.isfinite(ema200) & (ema200 != 0),
        (price > ema200).astype(float),
        np.nan,
    )
    ema50_gt = np.where(
        np.isfinite(ema50) & np.isfinite(ema200),
        (ema50 > ema200).astype(float),
        np.nan,
    )

    adv = _panel_row(getattr(engine, "adv20_m", None), present, date_idx)
    amihud = _panel_row(getattr(engine, "amihud_m", None), present, date_idx)
    if getattr(engine, "amihud_m", None) is None and dvol_m is not None:
        # On-the-fly Amihud for subset
        amihud = np.full(len(present), np.nan, dtype=float)
        for i, s in enumerate(present):
            amihud[i] = liquidity_raw_row(
                adv[i],
                price[i],
                None,
                ret_m[s] if s in ret_m.columns else close_m[s].pct_change(),
                dvol_m[s] if s in dvol_m.columns else pd.Series(np.nan, index=close_m.index),
                date_idx,
            )["amihud_raw"]
    amihud_liq = np.where(np.isfinite(amihud), -amihud, np.nan)

    # Risk: vectorized over subset
    ret_sub = ret_m[present] if all(s in ret_m.columns for s in present) else ret_m.reindex(columns=present)
    close_sub = close_m[present]
    if bench_ret is not None:
        betas, idios = beta_idiovol_panel_at(ret_sub, bench_ret, date_idx)
    else:
        betas = np.full(len(present), np.nan)
        idios = np.full(len(present), np.nan)
    maxdd = max_drawdown_panel_at(close_sub, date_idx)

    # Quality + turnover still need PIT fundamentals (per name; typically cheap vs risk/ADX)
    rows: List[Dict[str, Any]] = []
    profile = getattr(engine, "profile_meta", {}) or {}
    for i, sym in enumerate(present):
        try:
            fund = engine.get_latest_available_fundamentals(sym, current_date)
            q = quality_raw_row(fund)
            # Turnover needs shares from fundamentals
            shares = float("nan")
            if fund is not None:
                for key in ("diluted_shares_outstanding", "basic_shares_outstanding"):
                    try:
                        val = fund[key]
                        if pd.notna(val) and float(val) > 0:
                            shares = float(val)
                            break
                    except Exception:
                        continue
            dv = float(adv[i]) if np.isfinite(adv[i]) and adv[i] > 0 else float("nan")
            if np.isfinite(dv) and np.isfinite(price[i]) and price[i] > 0 and np.isfinite(shares) and shares > 0:
                turnover = (dv / price[i]) / shares
            else:
                turnover = float("nan")

            industry = profile.get(sym, {}).get("industry", "Unknown")
            rows.append(
                {
                    "symbol": sym,
                    "price": float(price[i]),
                    "industry": industry,
                    "bench": bench,
                    **q,
                    "mom_12_1_raw": float(mom_12_1[i]) if np.isfinite(mom_12_1[i]) else float("nan"),
                    "mom_6m_raw": float(mom_6m[i]) if np.isfinite(mom_6m[i]) else float("nan"),
                    "mom_3m_raw": float(mom_3m[i]) if np.isfinite(mom_3m[i]) else float("nan"),
                    "rs_bench_raw": float(rs_bench[i]) if np.isfinite(rs_bench[i]) else float("nan"),
                    "px_gt_ema200_raw": float(px_gt[i]) if np.isfinite(px_gt[i]) else float("nan"),
                    "ema50_gt_ema200_raw": float(ema50_gt[i]) if np.isfinite(ema50_gt[i]) else float("nan"),
                    "adx_raw": float(adx[i]) if np.isfinite(adx[i]) else float("nan"),
                    "dvol_raw": dv,
                    "turnover_raw": turnover,
                    "amihud_liq_raw": float(amihud_liq[i]) if np.isfinite(amihud_liq[i]) else float("nan"),
                    "amihud_raw": float(amihud[i]) if np.isfinite(amihud[i]) else float("nan"),
                    "beta_raw": float(betas[i]) if np.isfinite(betas[i]) else float("nan"),
                    "idiovol_raw": float(idios[i]) if np.isfinite(idios[i]) else float("nan"),
                    "maxdd_raw": float(maxdd[i]) if np.isfinite(maxdd[i]) else float("nan"),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


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


def format_factor_log(df: pd.DataFrame, *, max_rows: Optional[int] = 20) -> str:
    """Human-readable log: raw factors, Z-scores, composite (default top 20)."""
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
    n_log = len(df) if max_rows is None else min(int(max_rows), len(df))
    view = df[cols].head(n_log)
    lines = [
        "[INSTITUTIONAL ENTRY] Composite = "
        "0.35·Q + 0.35·M + 0.15·T + 0.10·L − 0.05·R (cross-sectional Z)",
        f"n={len(df)} logged={n_log}",
    ]
    for _, r in view.iterrows():
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
    try:
        lines.append(view.round(4).to_string(index=False))
    except Exception:
        pass
    return "\n".join(lines)
