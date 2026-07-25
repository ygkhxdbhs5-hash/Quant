"""Execution-cost / data-quality layer (isolated from strategy logic).

Fixes identified by the cost-model audit:
  * Corwin–Schultz spread instability (negatives, 5% clip saturation)
  * Dollar-volume spike distortion of ADV20 used for participation/impact

Strategy entry/exit/sizing formulas live elsewhere and must not import
trading decisions from this module beyond the cost/ADV inputs the engine
already uses.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Legacy clip used by the original baseline cost model (kept for A/B).
LEGACY_CS_FLOOR = 0.0002
LEGACY_CS_CEILING = 0.05

# Robust CS: empirically justified from 2021–2026 raw distribution on this
# universe (see analyze_raw_corwin_schultz):
#   * 7.05% of finite raw estimates were >= 5% (legacy ceiling hit routinely)
#   * 39.5% of finite estimates were negative (overnight-gap / CS breakdown)
#   * nonnegative p50≈56bps, p90≈4.2%; positive-only p50≈1.7%
# A 1% full-spread ceiling (~50bps half-spread) is already large vs typical
# NASDAQ effective spreads; above that we trust liquidity-tiered flats more
# than the daily CS estimator on gap-heavy microcaps.
ROBUST_CS_FLOOR = 0.0002
ROBUST_CS_CEILING = 0.01

# Sqrt-impact uses daily return vol. Microcap σ often 10–50%+/day; left uncapped
# this alone produced 5–8% one-way impact in the cost audit (HUDI etc.). Cap σ
# and the impact ratio for the robust cost model only.
ROBUST_IMPACT_SIGMA_CAP = 0.05  # 5% daily vol ceiling for impact calc
ROBUST_IMPACT_RATIO_CAP = 0.005  # 50 bps one-way impact ceiling
ROBUST_IMPACT_COEFF = 0.6  # same structural form as legacy

# Winsorize daily dollar volume vs trailing median before ADV / participation.
DVOL_WINSOR_WINDOW = 20
DVOL_WINSOR_MAX_MULT = 5.0
DVOL_WINSOR_MIN_PERIODS = 5


def corwin_schultz_raw(high_m: pd.DataFrame, low_m: pd.DataFrame) -> pd.DataFrame:
    """Raw Corwin–Schultz (2012) spread estimator — no clip, no cleanup."""
    # Guard zero / negative prices to avoid log blow-ups
    high = high_m.where(high_m > 0)
    low = low_m.where(low_m > 0)
    # high < low (data error) → invalid
    high = high.where(high >= low)
    low = low.where(high >= low)

    hl = np.log(high / low) ** 2
    hl2_high = high.rolling(2, min_periods=2).max()
    hl2_low = low.rolling(2, min_periods=2).min()
    # near-zero two-day range → gamma ~ 0; ok. high==low → hl=0.
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.log(hl2_high / hl2_low) ** 2
        beta = hl + hl.shift(1)
        den = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread


def analyze_raw_corwin_schultz(spread_raw: pd.DataFrame) -> Dict[str, Any]:
    """Issue 1a/b — full distribution of pre-clip CS estimates."""
    vals = spread_raw.to_numpy(dtype=float).ravel()
    n = int(vals.size)
    n_nan = int(np.isnan(vals).sum())
    n_inf = int(np.isinf(vals).sum())
    finite = vals[np.isfinite(vals)]
    n_finite = int(finite.size)
    n_neg = int((finite < 0).sum())
    n_pos = int((finite > 0).sum())
    n_ge_5 = int((finite >= LEGACY_CS_CEILING).sum())
    n_ge_2 = int((finite >= 0.02).sum())
    n_ge_1 = int((finite >= 0.01).sum())

    def _pct(a: np.ndarray, p: float) -> Optional[float]:
        if a.size == 0:
            return None
        return float(np.percentile(a, p))

    return {
        "n_obs": n,
        "n_finite": n_finite,
        "pct_nan": n_nan / n if n else None,
        "pct_inf": n_inf / n if n else None,
        "pct_negative_of_finite": n_neg / n_finite if n_finite else None,
        "pct_positive_of_finite": n_pos / n_finite if n_finite else None,
        "min": _pct(finite, 0),
        "p50": _pct(finite, 50),
        "p90": _pct(finite, 90),
        "p99": _pct(finite, 99),
        "p99_9": _pct(finite, 99.9),
        "max": _pct(finite, 100),
        "pct_finite_ge_legacy_ceiling_5pct": n_ge_5 / n_finite if n_finite else None,
        "pct_finite_ge_2pct": n_ge_2 / n_finite if n_finite else None,
        "pct_finite_ge_1pct": n_ge_1 / n_finite if n_finite else None,
        "n_finite_ge_legacy_ceiling_5pct": n_ge_5,
        "legacy_ceiling_hit_rate_flag": bool(
            n_finite and (n_ge_5 / n_finite) > 0.02
        ),
    }


def diagnose_cs_breakdown(
    high_m: pd.DataFrame, low_m: pd.DataFrame, close_m: pd.DataFrame, spread_raw: pd.DataFrame
) -> Dict[str, Any]:
    """Issue 1c — why the estimator blows up on this universe."""
    rng = (high_m - low_m) / close_m.replace(0, np.nan)
    rng_vals = rng.to_numpy(dtype=float).ravel()
    rng_f = rng_vals[np.isfinite(rng_vals)]

    hl = np.log(high_m.where(high_m > 0) / low_m.where(low_m > 0)) ** 2
    hl2_high = high_m.rolling(2, min_periods=2).max()
    hl2_low = low_m.rolling(2, min_periods=2).min()
    gamma = np.log(hl2_high / hl2_low) ** 2
    beta = hl + hl.shift(1)
    gap_dom = (gamma > 2.0 * beta) & np.isfinite(gamma) & np.isfinite(beta) & (beta > 0)

    finite = spread_raw.to_numpy(dtype=float).ravel()
    finite_mask = np.isfinite(finite)
    high_cs = finite_mask & (finite >= LEGACY_CS_CEILING)

    return {
        "pct_zero_or_tiny_intraday_range": float((rng_f <= 1e-6).mean()) if rng_f.size else None,
        "intraday_range_over_close_p50": float(np.percentile(rng_f, 50)) if rng_f.size else None,
        "intraday_range_over_close_p99": float(np.percentile(rng_f, 99)) if rng_f.size else None,
        "pct_gap_dominated_gamma_gt_2beta": float(
            gap_dom.to_numpy().ravel()[np.isfinite(beta.to_numpy().ravel())].mean()
        )
        if np.isfinite(beta.to_numpy().ravel()).any()
        else None,
        "among_cs_ge_5pct": {
            "n": int(high_cs.sum()),
            "close_p50": float(
                np.nanpercentile(close_m.to_numpy(dtype=float).ravel()[high_cs], 50)
            )
            if high_cs.any()
            else None,
            "pct_close_lt_5": float(
                (close_m.to_numpy(dtype=float).ravel()[high_cs] < 5).mean()
            )
            if high_cs.any()
            else None,
        },
        "notes": [
            "Corwin–Schultz (2012) yields negative estimates when overnight variance "
            "dominates the two-day range; standard practice sets those to 0 / invalid.",
            "Legacy clip upper=5% was hit on >>2% of finite cells → ceiling is not a rare safety net.",
            "High CS cells concentrate in low-priced names with wide intraday ranges.",
        ],
    }


def liquidity_tiered_spread(adv: float) -> float:
    """Fallback full spread (not half) by ADV dollar-volume tier."""
    if not np.isfinite(adv) or adv <= 0:
        return 0.0100  # 100 bps
    if adv >= 20_000_000:
        return 0.0015  # 15 bps
    if adv >= 5_000_000:
        return 0.0030  # 30 bps
    if adv >= 1_000_000:
        return 0.0050  # 50 bps
    return 0.0100  # 100 bps


def apply_legacy_cs_clip(spread_raw: pd.DataFrame) -> pd.DataFrame:
    """Original baseline behaviour: blind clip to [2bps, 5%]."""
    return spread_raw.clip(lower=LEGACY_CS_FLOOR, upper=LEGACY_CS_CEILING)


def apply_robust_cs(
    spread_raw: pd.DataFrame,
    high_m: pd.DataFrame,
    low_m: pd.DataFrame,
    adv20_m: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Robust CS panel + per-cell source mask ('cs' | 'fallback' | 'invalid').

    Rules:
      1. Near-zero high/low range → invalid → liquidity-tiered fallback
      2. Non-finite or negative CS → fallback (CS breakdown / overnight gap)
      3. CS above ROBUST_CS_CEILING (1%) → fallback (estimator extreme)
      4. Else use CS floored at ROBUST_CS_FLOOR
    """
    rng_ok = (high_m - low_m) > (high_m * 1e-6)
    spread = spread_raw.copy()
    use_cs = rng_ok & np.isfinite(spread) & (spread >= 0) & (spread <= ROBUST_CS_CEILING)
    out = pd.DataFrame(np.nan, index=spread.index, columns=spread.columns)
    source = pd.DataFrame("invalid", index=spread.index, columns=spread.columns, dtype=object)

    # Vectorized fallback from ADV (may be NaN → tier handles it)
    # Build fallback panel column-wise to keep memory reasonable
    for col in spread.columns:
        adv_col = (
            adv20_m[col]
            if col in adv20_m.columns
            else pd.Series(np.nan, index=spread.index)
        )
        fb = adv_col.map(liquidity_tiered_spread)
        cs_col = spread[col]
        ok = use_cs[col].fillna(False)
        chosen = cs_col.where(ok, fb).clip(lower=ROBUST_CS_FLOOR)
        # still floor/ceil for safety
        chosen = chosen.clip(lower=ROBUST_CS_FLOOR, upper=max(ROBUST_CS_CEILING, 0.01))
        out[col] = chosen
        src = pd.Series("fallback", index=spread.index, dtype=object)
        src = src.mask(ok, "cs")
        source[col] = src
    return out, source


def winsorize_dollar_volume(
    dvol_m: pd.DataFrame,
    window: int = DVOL_WINSOR_WINDOW,
    max_mult: float = DVOL_WINSOR_MAX_MULT,
    min_periods: int = DVOL_WINSOR_MIN_PERIODS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cap daily dollar volume at max_mult × trailing median (exclusive of today).

    Used only for ADV / participation / impact — not for strategy universe ranking.
    Returns (winsorized_dvol, spike_flag_panel).
    """
    # Trailing median of prior days only (shift 1) so today's spike isn't in its own median.
    med = (
        dvol_m.shift(1)
        .rolling(window, min_periods=min_periods)
        .median()
        .replace(0, np.nan)
    )
    cap = med * float(max_mult)
    spike = dvol_m.gt(cap) & cap.notna()
    cleaned = dvol_m.where(~spike, cap)
    # If no median yet, keep raw
    cleaned = cleaned.fillna(dvol_m)
    return cleaned, spike


def format_cs_distribution_report(stats: Dict[str, Any], breakdown: Dict[str, Any]) -> str:
    def pct(x):
        if x is None:
            return "n/a"
        return f"{100 * float(x):.3f}%"

    def num(x):
        if x is None:
            return "n/a"
        return f"{float(x):.6f}"

    lines = [
        "### ISSUE 1a — Raw Corwin–Schultz spread distribution (pre-clip)",
        f"  n_obs              : {stats.get('n_obs', 0):,}",
        f"  n_finite           : {stats.get('n_finite', 0):,}",
        f"  % NaN              : {pct(stats.get('pct_nan'))}",
        f"  % Inf              : {pct(stats.get('pct_inf'))}",
        f"  % negative (finite): {pct(stats.get('pct_negative_of_finite'))}",
        f"  min / p50 / p90    : {num(stats.get('min'))} / {num(stats.get('p50'))} / {num(stats.get('p90'))}",
        f"  p99 / p99.9 / max  : {num(stats.get('p99'))} / {num(stats.get('p99_9'))} / {num(stats.get('max'))}",
        "",
        "### ISSUE 1b — Legacy 5% ceiling saturation",
        f"  % finite >= 5%     : {pct(stats.get('pct_finite_ge_legacy_ceiling_5pct'))}  "
        f"(n={stats.get('n_finite_ge_legacy_ceiling_5pct', 0):,})",
        f"  % finite >= 2%     : {pct(stats.get('pct_finite_ge_2pct'))}",
        f"  % finite >= 1%     : {pct(stats.get('pct_finite_ge_1pct'))}",
        f"  flag (>2% hit rate): {stats.get('legacy_ceiling_hit_rate_flag')}",
        "",
        "### ISSUE 1c — Breakdown diagnostics",
        f"  % tiny/zero range  : {pct(breakdown.get('pct_zero_or_tiny_intraday_range'))}",
        f"  range/close p50/p99: {num(breakdown.get('intraday_range_over_close_p50'))} / "
        f"{num(breakdown.get('intraday_range_over_close_p99'))}",
        f"  % gap-dominated    : {pct(breakdown.get('pct_gap_dominated_gamma_gt_2beta'))}",
        f"  among CS>=5%: n={breakdown.get('among_cs_ge_5pct', {}).get('n')} "
        f"close_p50={breakdown.get('among_cs_ge_5pct', {}).get('close_p50')} "
        f"%close<5={pct(breakdown.get('among_cs_ge_5pct', {}).get('pct_close_lt_5'))}",
        "",
        "### ISSUE 1d — Chosen robust fix",
        f"  negatives / NaN / near-zero-range → liquidity-tiered flat spread fallback",
        f"  CS ceiling {ROBUST_CS_CEILING:.2%} (was {LEGACY_CS_CEILING:.2%}); floor {ROBUST_CS_FLOOR:.2%}",
        f"  fallback tiers by ADV20: >=$20M→15bps, >=$5M→30bps, >=$1M→50bps, else→100bps (full spread)",
        f"  impact (v2 only): σ capped at {ROBUST_IMPACT_SIGMA_CAP:.0%} daily; "
        f"impact ratio capped at {ROBUST_IMPACT_RATIO_CAP:.2%} (legacy left 5–8% one-way impacts)",
        f"  Justification: legacy 5% ceiling hit {pct(stats.get('pct_finite_ge_legacy_ceiling_5pct'))} "
        f"of finite cells (>>1–2% rare-tail budget); daily CS on gap-heavy microcaps is upward-biased.",
    ]
    for n in breakdown.get("notes") or []:
        lines.append(f"  note: {n}")
    return "\n".join(lines)
