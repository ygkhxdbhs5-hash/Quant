"""Entry Quality Score (EQS) — live ranking component.

Modular setup-health score combined with CMVS at rebalance time.
Uses only information available as-of the rebalance date (no look-ahead).

FinalScore = CMVS_raw + EQS_WEIGHT * EQS

Each component is independently tunable via ``EQSWeights``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EQSWeights:
    """Per-component weights (should roughly sum to 1.0 before overextension subtracts)."""

    trend_structure: float = 0.28
    healthy_pullback: float = 0.14
    volume_contraction_expansion: float = 0.12
    volatility_contraction: float = 0.12
    rs_acceleration: float = 0.14
    sector_strength: float = 0.10
    # Overextension is a penalty multiplier applied after the weighted sum
    overextension_penalty_scale: float = 0.35

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


# How strongly EQS lifts/suppresses names vs CMVS (meaningful but not dominant).
DEFAULT_EQS_BLEND_WEIGHT = 0.55


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def score_trend_structure(df: pd.DataFrame) -> pd.Series:
    """Price > EMA20 > EMA50 with positive slopes — highest-weight setup health."""
    close = _col(df, "price")
    ema20 = _col(df, "ema20", np.nan)
    ema50 = _col(df, "ema50", np.nan)
    s20 = _col(df, "ema20_slope5", 0.0)
    s50 = _col(df, "ema50_slope5", 0.0)

    score = pd.Series(0.15, index=df.index, dtype=float)

    aligned = (close > ema20) & (ema20 > ema50)
    above20 = close > ema20
    above50 = close > ema50
    slope_ok = (s20 > 0.0) & (s50 > -0.002)

    score = np.where(aligned & slope_ok, 1.0, score)
    score = np.where(aligned & ~slope_ok, 0.80, score)
    score = np.where(~aligned & above20 & above50, 0.55, score)
    score = np.where(~aligned & above20 & ~above50, 0.35, score)
    score = np.where(~above20 & above50, 0.20, score)
    score = np.where(~above20 & ~above50, 0.05, score)

    # Strong penalty if EMA20 below EMA50 (broken structure)
    score = np.where(ema20 < ema50, np.minimum(score, 0.25), score)
    # Strong penalty if price below EMA20
    score = np.where(close < ema20, np.minimum(score, 0.30), score)
    return pd.Series(score, index=df.index, dtype=float).clip(0.0, 1.0)


def score_healthy_pullback(df: pd.DataFrame) -> pd.Series:
    """Prefer controlled 8–12% pullbacks on lighter volume, holding EMA20."""
    dd = _col(df, "pullback_pct", 0.0)  # 0 = at highs, 0.10 = 10% off
    vol_pb = _col(df, "pullback_vol_ratio", 1.0)  # <1 = contraction on down days
    close = _col(df, "price")
    ema20 = _col(df, "ema20", np.nan)
    hold20 = close >= ema20

    # Ideal band ~3%–12%
    dist_from_ideal = (dd - 0.08).abs()
    band = 1.0 - (dist_from_ideal / 0.12).clip(0.0, 1.0)
    # No pullback (extended) or violent selloff
    no_pb = dd < 0.02
    violent = dd > 0.18
    band = np.where(no_pb, 0.25, band)
    band = np.where(violent, 0.10, band)
    # Reward decreasing volume into the pullback
    vol_part = (1.2 - vol_pb).clip(0.0, 1.0)
    hold = np.where(hold20, 1.0, 0.25)
    # Breaking trend: deep pb below EMA20
    broken = (~hold20) & (dd > 0.08)
    out = 0.45 * band + 0.30 * vol_part + 0.25 * hold
    out = np.where(broken, out * 0.35, out)
    return pd.Series(out, index=df.index, dtype=float).clip(0.0, 1.0)


def score_volume_contraction_expansion(df: pd.DataFrame) -> pd.Series:
    """Accumulation: quiet consolidation then volume expansion on recovery."""
    quiet = _col(df, "vol_quiet_ratio", 1.0)  # recent consolid vol / prior — want < 1
    expand = _col(df, "vol_expand_ratio", 1.0)  # recovery vol / quiet — want > 1
    spike_persist = _col(df, "vol_spike_persist", 0.0)  # fraction of days with rv>2
    ret5 = _col(df, "ret5", 0.0)

    quiet_score = (1.15 - quiet).clip(0.0, 1.0)
    expand_score = ((expand - 1.0) / 1.0).clip(0.0, 1.0)
    # Expansion only counts with rising price (breakout/recovery)
    expand_score = expand_score * np.where(ret5 > 0.0, 1.0, 0.35)
    persist_pen = (1.0 - spike_persist * 1.5).clip(0.0, 1.0)
    out = 0.40 * quiet_score + 0.40 * expand_score + 0.20 * persist_pen
    return pd.Series(out, index=df.index, dtype=float).clip(0.0, 1.0)


def score_volatility_contraction(df: pd.DataFrame) -> pd.Series:
    """VCP-like: progressive shrinking of swing ranges / ATR."""
    # Prefer rng_short < rng_mid < rng_long and atr_now < atr_prior
    r5 = _col(df, "range_pct_5", np.nan)
    r10 = _col(df, "range_pct_10", np.nan)
    r20 = _col(df, "range_pct_20", np.nan)
    atr_shrink = _col(df, "atr_shrink_ratio", 1.0)  # atr14/atr14_15d_ago — <1 contracting

    progressive = (
        (r5 < r10).astype(float) * 0.35
        + (r10 < r20).astype(float) * 0.35
        + (r5 < r20).astype(float) * 0.30
    )
    # Handle NaNs as neutral 0.5 progressive
    progressive = progressive.where(r5.notna() & r10.notna() & r20.notna(), 0.50)
    shrink = (1.15 - atr_shrink).clip(0.0, 1.0)
    # Expanding vol penalized
    expanding = atr_shrink > 1.15
    out = 0.55 * progressive + 0.45 * shrink
    out = np.where(expanding, out * 0.45, out)
    return pd.Series(out, index=df.index, dtype=float).clip(0.0, 1.0)


def score_rs_acceleration(df: pd.DataFrame) -> pd.Series:
    """Improving relative strength — rising slope / near RS highs."""
    rs = _col(df, "rs_raw", 0.0)
    slope = _col(df, "rs_slope5", 0.0)
    near_high = _col(df, "rs_near_high60", 0.5)  # rs / rolling max — ~1 is leadership high

    # Cross-sectional rank of RS for leadership context (already in df as rss often)
    rss = _col(df, "rss", 0.5)
    slope_score = ((slope + 0.05) / 0.15).clip(0.0, 1.0)
    high_score = near_high.clip(0.0, 1.0)
    # Penalize flat/weakening RS
    weak = slope < 0.0
    out = 0.35 * rss + 0.40 * slope_score + 0.25 * high_score
    out = np.where(weak, out * 0.55, out)
    return pd.Series(out, index=df.index, dtype=float).clip(0.0, 1.0)


def score_sector_strength(df: pd.DataFrame) -> pd.Series:
    """Leaders in strong industries; penalize strong names in weak sectors."""
    if "industry" not in df.columns or df.empty:
        return pd.Series(0.5, index=df.index, dtype=float)

    rs = _col(df, "rs_raw", 0.0)
    above50 = _col(df, "above_ema50", 0.0)
    sector_rs = df.groupby("industry")["rs_raw"].transform("mean") if "rs_raw" in df.columns else rs * 0
    sector_breadth = (
        df.groupby("industry")["above_ema50"].transform("mean")
        if "above_ema50" in df.columns
        else pd.Series(0.5, index=df.index)
    )
    # Normalize sector RS cross-sectionally
    if float(sector_rs.std(ddof=0) or 0.0) > 1e-12:
        sector_rs_z = (sector_rs - sector_rs.mean()) / sector_rs.std(ddof=0)
        sector_rs_score = (sector_rs_z / 2.5 + 0.5).clip(0.0, 1.0)
    else:
        sector_rs_score = pd.Series(0.5, index=df.index)

    stock_rs_rank = rs.rank(pct=True)
    # Strong stock in weak sector → dampen
    weak_sector = sector_rs_score < 0.40
    out = 0.45 * sector_rs_score + 0.35 * sector_breadth.clip(0.0, 1.0) + 0.20 * stock_rs_rank
    out = np.where(weak_sector & (stock_rs_rank > 0.7), out * 0.55, out)
    # Reward leadership inside strong sectors
    strong_sector = sector_rs_score > 0.60
    out = np.where(strong_sector & (stock_rs_rank > 0.7), np.minimum(out + 0.10, 1.0), out)
    return pd.Series(out, index=df.index, dtype=float).clip(0.0, 1.0)


def score_overextension_penalty(df: pd.DataFrame) -> pd.Series:
    """0 = not extended, 1 = severely extended (FOMO). Used as a subtractive penalty."""
    rsi = _col(df, "rsi14", 50.0)
    dist20 = _col(df, "dist_ema20", 0.0)  # (close-ema20)/ema20
    dist50 = _col(df, "dist_ema50", 0.0)
    green_streak = _col(df, "green_streak", 0.0)
    ret10 = _col(df, "ret10", 0.0)

    pen = (
        np.maximum(0.0, (rsi - 72.0) / 28.0) * 0.30
        + np.maximum(0.0, (dist20 - 0.08) / 0.20) * 0.25
        + np.maximum(0.0, (dist50 - 0.15) / 0.25) * 0.20
        + np.maximum(0.0, (green_streak - 4.0) / 6.0) * 0.10
        + np.maximum(0.0, (ret10 - 0.25) / 0.25) * 0.15
    )
    return pd.Series(pen, index=df.index, dtype=float).clip(0.0, 1.0)


def compute_eqs_components(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of individual EQS component scores in [0, 1]."""
    return pd.DataFrame(
        {
            "eqs_trend": score_trend_structure(df),
            "eqs_pullback": score_healthy_pullback(df),
            "eqs_volume": score_volume_contraction_expansion(df),
            "eqs_vcp": score_volatility_contraction(df),
            "eqs_rs_accel": score_rs_acceleration(df),
            "eqs_sector": score_sector_strength(df),
            "eqs_overext_pen": score_overextension_penalty(df),
        },
        index=df.index,
    )


def combine_eqs(
    components: pd.DataFrame,
    weights: Optional[EQSWeights] = None,
) -> pd.Series:
    """Weighted EQS in roughly [0, 1] after overextension penalty."""
    w = weights or EQSWeights()
    base = (
        w.trend_structure * components["eqs_trend"]
        + w.healthy_pullback * components["eqs_pullback"]
        + w.volume_contraction_expansion * components["eqs_volume"]
        + w.volatility_contraction * components["eqs_vcp"]
        + w.rs_acceleration * components["eqs_rs_accel"]
        + w.sector_strength * components["eqs_sector"]
    )
    eqs = base - w.overextension_penalty_scale * components["eqs_overext_pen"]
    return eqs.clip(0.0, 1.0)


def blend_cmvs_eqs(
    cmvs: pd.Series,
    eqs: pd.Series,
    eqs_weight: float = DEFAULT_EQS_BLEND_WEIGHT,
) -> pd.Series:
    """Final live ranking score: CMVS + meaningful EQS contribution."""
    return (cmvs.astype(float) + float(eqs_weight) * eqs.astype(float)).astype(float)


def eqs_config_from_mapping(cfg: Optional[Mapping[str, Any]]) -> tuple[EQSWeights, float]:
    """Load EQSWeights + blend weight from config dict (optional ``eqs:`` block)."""
    cfg = cfg or {}
    block = dict(cfg.get("eqs") or {})
    blend = float(block.get("blend_weight", cfg.get("eqs_blend_weight", DEFAULT_EQS_BLEND_WEIGHT)))
    defaults = EQSWeights()
    weights = EQSWeights(
        trend_structure=float(block.get("trend_structure", defaults.trend_structure)),
        healthy_pullback=float(block.get("healthy_pullback", defaults.healthy_pullback)),
        volume_contraction_expansion=float(
            block.get("volume_contraction_expansion", defaults.volume_contraction_expansion)
        ),
        volatility_contraction=float(
            block.get("volatility_contraction", defaults.volatility_contraction)
        ),
        rs_acceleration=float(block.get("rs_acceleration", defaults.rs_acceleration)),
        sector_strength=float(block.get("sector_strength", defaults.sector_strength)),
        overextension_penalty_scale=float(
            block.get("overextension_penalty_scale", defaults.overextension_penalty_scale)
        ),
    )
    return weights, blend
