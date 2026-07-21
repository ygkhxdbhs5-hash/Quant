from __future__ import annotations

import numpy as np
import pandas as pd

from engine.research.cmvs_factor_audit import (
    _assign_deciles_by_rank,
    _decile_monotonicity_score,
    _safe_pearson,
)


def test_assign_deciles_by_rank_bounds_and_length():
    s = pd.Series([np.nan, 1, 2, 3, 4, 5])
    dec = _assign_deciles_by_rank(s, n_bins=10)
    assert len(dec) == 5  # NaN dropped
    assert dec.min() >= 0
    assert dec.max() <= 9


def test_safe_pearson_none_when_zero_variance():
    x = pd.Series([1, 1, 1, 1], dtype=float)
    y = pd.Series([1, 2, 3, 4], dtype=float)
    assert _safe_pearson(x, y) is None


def test_decile_monotonicity_score_perfect_monotone():
    # decile index 0..9, mean returns strictly increasing
    decile_mean_returns = np.arange(10, dtype=float)
    v = _decile_monotonicity_score(decile_mean_returns)
    assert v is not None
    assert v > 0.95

