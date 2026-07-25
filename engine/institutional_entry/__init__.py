"""Institutional Entry Engine — academically validated multi-factor ranking.

Replaces empirical CMVS entry scoring. Factors follow Fama–French, Novy-Marx,
Jegadeesh–Titman, Asness/AQR-style definitions with cross-sectional Z-scores.

Composite =
  0.35 × Quality + 0.35 × Momentum + 0.15 × Trend + 0.10 × Liquidity − 0.05 × Risk
"""

from __future__ import annotations

from engine.institutional_entry.composite import (
    CATEGORY_WEIGHTS,
    build_institutional_scores,
    format_factor_log,
)
from engine.institutional_entry.zscore import cross_sectional_zscore

__all__ = [
    "CATEGORY_WEIGHTS",
    "build_institutional_scores",
    "cross_sectional_zscore",
    "format_factor_log",
]
