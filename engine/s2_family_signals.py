"""Signal scorers for Strategy #2+ family search (families 10–26 viable/degraded).

Each scorer returns a Series of scores (higher = more desirable for long book)
for symbols in the liquid pool on ``as_of`` date. NaNs excluded by caller.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from engine.pit_fundamentals import get_latest_available_fundamentals

TOP_LIQUID = 250
TOP_K = 30

ScoreFn = Callable[..., pd.Series]


def liquid_pool(
    dvol_m: pd.DataFrame, as_of, eligible: Sequence[str], n: int = TOP_LIQUID
) -> List[str]:
    syms = [s for s in eligible if s in dvol_m.columns]
    if not syms:
        return []
    row = pd.to_numeric(dvol_m.loc[as_of, syms], errors="coerce").dropna()
    if row.empty:
        return []
    return row.nlargest(min(int(n), len(row))).index.tolist()


def _pit_row(hist, fund_ts, sym, as_of):
    return get_latest_available_fundamentals(hist, fund_ts, sym, as_of)


def _f(row, field) -> float:
    if row is None or field not in row.index:
        return float("nan")
    try:
        v = float(row[field])
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _pit_history_asof(hist, fund_ts, sym, as_of, n_rows: int = 8) -> pd.DataFrame:
    """Last n PIT rows with filing_date ≤ as_of."""
    h = hist.get(sym)
    if h is None or h.empty:
        return pd.DataFrame()
    ts = fund_ts.get(sym)
    if ts is not None and len(ts):
        asof = np.datetime64(pd.Timestamp(as_of).to_datetime64())
        i = int(np.searchsorted(ts, asof, side="right") - 1)
        if i < 0:
            return pd.DataFrame()
        lo = max(0, i - n_rows + 1)
        return h.iloc[lo : i + 1]
    past = h.loc[:as_of]
    return past.iloc[-n_rows:] if len(past) else pd.DataFrame()


# ----- family scorers: (close_m, dvol_m, hist, fund_ts, as_of, liquid, extras) -> Series -----


def score_accruals_low(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    """DEGRADED Sloan proxy: low (op_income - op_cf)/assets."""
    out = {}
    for sym in liquid:
        r = _pit_row(hist, fund_ts, sym, as_of)
        oi, cf, a = _f(r, "operating_income"), _f(r, "op_cf"), _f(r, "total_assets")
        if a and a > 0 and np.isfinite(oi) and np.isfinite(cf):
            out[sym] = -((oi - cf) / a)  # low accruals → high score
    return pd.Series(out, dtype=float)


def score_asset_growth_low(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    out = {}
    for sym in liquid:
        past = _pit_history_asof(hist, fund_ts, sym, as_of, n_rows=6)
        if len(past) < 5:
            continue
        a0 = _f(past.iloc[-1], "total_assets")
        a1 = _f(past.iloc[-5], "total_assets")  # ~4Q lag
        if a1 and a1 > 0 and np.isfinite(a0):
            out[sym] = -(a0 / a1 - 1.0)
    return pd.Series(out, dtype=float)


def score_net_issuance_low(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    out = {}
    for sym in liquid:
        past = _pit_history_asof(hist, fund_ts, sym, as_of, n_rows=6)
        if len(past) < 5:
            continue
        s0 = _f(past.iloc[-1], "diluted_shares_outstanding")
        s1 = _f(past.iloc[-5], "diluted_shares_outstanding")
        if s1 and s1 > 0 and np.isfinite(s0):
            out[sym] = -(s0 / s1 - 1.0)  # repurchasers / low issuance → high
    return pd.Series(out, dtype=float)


def score_52w_high_near(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    out = {}
    if as_of not in close_m.index:
        return pd.Series(dtype=float)
    loc = close_m.index.get_loc(as_of)
    if isinstance(loc, slice) or not np.isscalar(loc):
        return pd.Series(dtype=float)
    i = int(loc)
    if i < 252:
        return pd.Series(dtype=float)
    window = close_m.iloc[i - 251 : i + 1]
    for sym in liquid:
        if sym not in close_m.columns:
            continue
        s = pd.to_numeric(window[sym], errors="coerce")
        px = s.iloc[-1]
        mx = s.max()
        if pd.notna(px) and pd.notna(mx) and mx > 0:
            out[sym] = float(px / mx)  # closer to high → higher
    return pd.Series(out, dtype=float)


def score_52w_high_far(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    s = score_52w_high_near(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw)
    return -s


def score_piotroski_partial(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    """DEGRADED partial F-score (0–6ish) from available PIT fields."""
    out = {}
    for sym in liquid:
        past = _pit_history_asof(hist, fund_ts, sym, as_of, n_rows=6)
        if len(past) < 2:
            continue
        cur, prev = past.iloc[-1], past.iloc[-2]
        score = 0
        a = _f(cur, "total_assets")
        if not a or a <= 0:
            continue
        roa = _f(cur, "operating_income") / a
        if roa > 0:
            score += 1
        cf = _f(cur, "op_cf")
        if np.isfinite(cf) and cf > 0:
            score += 1
        # accruals proxy: CFO > OI
        oi = _f(cur, "operating_income")
        if np.isfinite(cf) and np.isfinite(oi) and cf > oi:
            score += 1
        lev = _f(cur, "total_debt") / a
        lev_p = _f(prev, "total_debt") / max(_f(prev, "total_assets"), 1e-12)
        if np.isfinite(lev) and np.isfinite(lev_p) and lev < lev_p:
            score += 1
        sh = _f(cur, "diluted_shares_outstanding")
        sh_p = _f(prev, "diluted_shares_outstanding")
        if np.isfinite(sh) and np.isfinite(sh_p) and sh <= sh_p:
            score += 1
        m = _f(cur, "op_margin")
        m_p = _f(prev, "op_margin")
        if np.isfinite(m) and np.isfinite(m_p) and m > m_p:
            score += 1
        out[sym] = float(score)
    return pd.Series(out, dtype=float)


def score_distress_safe(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    """DEGRADED: higher = safer (high op_income/assets, low debt/assets, high equity/assets)."""
    out = {}
    for sym in liquid:
        r = _pit_row(hist, fund_ts, sym, as_of)
        a = _f(r, "total_assets")
        if not a or a <= 0:
            continue
        oi = _f(r, "operating_income")
        d = _f(r, "total_debt")
        e = _f(r, "total_equity")
        if not all(np.isfinite(x) for x in (oi, d, e)):
            continue
        z = 6.72 * (oi / a) + 1.05 * (e / max(a - e, 1e-9)) - 1.0 * (d / a)
        out[sym] = z
    return pd.Series(out, dtype=float)


def score_distress_risky(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    return -score_distress_safe(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw)


def score_capex_low(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    out = {}
    for sym in liquid:
        r = _pit_row(hist, fund_ts, sym, as_of)
        a = _f(r, "total_assets")
        c = _f(r, "capex")
        if a and a > 0 and np.isfinite(c):
            out[sym] = -(c / a)
    return pd.Series(out, dtype=float)


def score_noa_low(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    """DEGRADED NOA/assets proxy: (equity + debt - cash) / assets."""
    out = {}
    for sym in liquid:
        r = _pit_row(hist, fund_ts, sym, as_of)
        a = _f(r, "total_assets")
        e = _f(r, "total_equity")
        d = _f(r, "total_debt")
        c = _f(r, "cash_eq")
        if a and a > 0 and all(np.isfinite(x) for x in (e, d, c)):
            noa = e + d - c
            out[sym] = -(noa / a)
    return pd.Series(out, dtype=float)


def score_margin_expand_op(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    out = {}
    for sym in liquid:
        past = _pit_history_asof(hist, fund_ts, sym, as_of, n_rows=6)
        if len(past) < 5:
            continue
        m0 = _f(past.iloc[-1], "op_margin")
        m1 = _f(past.iloc[-5], "op_margin")
        if np.isfinite(m0) and np.isfinite(m1):
            out[sym] = m0 - m1
    return pd.Series(out, dtype=float)


def score_margin_expand_gp(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    out = {}
    for sym in liquid:
        past = _pit_history_asof(hist, fund_ts, sym, as_of, n_rows=6)
        if len(past) < 5:
            continue
        m0 = _f(past.iloc[-1], "gross_profitability")
        m1 = _f(past.iloc[-5], "gross_profitability")
        if np.isfinite(m0) and np.isfinite(m1):
            out[sym] = m0 - m1
    return pd.Series(out, dtype=float)


def score_max_low(close_m, dvol_m, hist, fund_ts, as_of, liquid, **kw) -> pd.Series:
    out = {}
    if as_of not in close_m.index:
        return pd.Series(dtype=float)
    loc = close_m.index.get_loc(as_of)
    i = int(loc)
    if i < 22:
        return pd.Series(dtype=float)
    rets = close_m.iloc[i - 21 : i + 1].pct_change()
    for sym in liquid:
        if sym not in rets.columns:
            continue
        s = pd.to_numeric(rets[sym], errors="coerce").dropna()
        if len(s) < 10:
            continue
        out[sym] = -float(s.max())  # low MAX → high score
    return pd.Series(out, dtype=float)


def score_short_dtc_low(close_m, dvol_m, hist, fund_ts, as_of, liquid, si_panel=None, **kw) -> pd.Series:
    """DEGRADED: long low days-to-cover (SI panel already publication-lagged)."""
    if si_panel is None or si_panel.empty:
        return pd.Series(dtype=float)
    out = {}
    # asof row: last available ≤ as_of
    sub = si_panel.loc[:as_of]
    if sub.empty:
        return pd.Series(dtype=float)
    row = sub.iloc[-1]
    for sym in liquid:
        if sym in row.index and pd.notna(row[sym]):
            out[sym] = -float(row[sym])
    return pd.Series(out, dtype=float)


def score_short_dtc_high(close_m, dvol_m, hist, fund_ts, as_of, liquid, si_panel=None, **kw) -> pd.Series:
    s = score_short_dtc_low(close_m, dvol_m, hist, fund_ts, as_of, liquid, si_panel=si_panel, **kw)
    return -s


def score_short_dtc_decrease(close_m, dvol_m, hist, fund_ts, as_of, liquid, si_panel=None, **kw) -> pd.Series:
    if si_panel is None or si_panel.empty:
        return pd.Series(dtype=float)
    sub = si_panel.loc[:as_of]
    if len(sub) < 2:
        return pd.Series(dtype=float)
    cur, prev = sub.iloc[-1], sub.iloc[-2]
    out = {}
    for sym in liquid:
        if sym in cur.index and sym in prev.index and pd.notna(cur[sym]) and pd.notna(prev[sym]):
            out[sym] = float(prev[sym]) - float(cur[sym])  # decrease → high
    return pd.Series(out, dtype=float)


def top_k_from_scores(scores: pd.Series, k: int = TOP_K, extreme_frac: Optional[float] = None) -> List[str]:
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if s.empty:
        return []
    if extreme_frac is not None and 0 < extreme_frac < 1:
        n = max(int(np.ceil(len(s) * extreme_frac)), 1)
        s = s.nlargest(n)
    return s.sort_values(ascending=False).head(int(k)).index.tolist()


def with_quality_half(
    scores: pd.Series, hist, fund_ts, as_of, min_pctile: float = 0.50
) -> pd.Series:
    """Keep names with quality composite (GP, ROIC, op_margin) in top half of scored set."""
    from engine.strategy_baseline_v1_quality import _pct_rank  # type: ignore
    from engine.pit_fundamentals import extract_quality_raw

    if scores.empty:
        return scores
    gp, roic, opm = {}, {}, {}
    for sym in scores.index:
        row = _pit_row(hist, fund_ts, sym, as_of)
        q = extract_quality_raw(row)
        if q is None:
            continue
        gp[sym] = q["gross_profitability"]
        roic[sym] = q["roic"]
        opm[sym] = q["op_margin"]
    if not gp:
        return pd.Series(dtype=float)
    qpct = (
        pd.Series(gp).rank(pct=True)
        + pd.Series(roic).rank(pct=True)
        + pd.Series(opm).rank(pct=True)
    ) / 3.0
    keep = qpct[qpct >= min_pctile].index
    return scores.reindex(keep).dropna()


# Catalog: family_id -> list of (variant_id, score_fn, meta)
def build_catalog() -> List[Dict[str, Any]]:
    """Return variant specs for all VIABLE/DEGRADED families (2–3 each)."""
    cats: List[Dict[str, Any]] = []

    def add(fid, vid, fn, *, cadence="monthly", mode="rank", extra=None, note=""):
        cats.append(
            {
                "family_id": fid,
                "variant_id": vid,
                "score_fn_name": fn.__name__ if callable(fn) else str(fn),
                "score_fn": fn,
                "cadence": cadence,
                "mode": mode,  # rank | calendar_tom | event_div
                "extra": extra or {},
                "note": note,
            }
        )

    # 10 ex-div (event)
    add(10, "exdiv_m5p5", None, cadence="daily", mode="event_div", extra={"pre": 5, "post": 5}, note="VIABLE")
    add(10, "exdiv_m1p1", None, cadence="daily", mode="event_div", extra={"pre": 1, "post": 1}, note="VIABLE")
    add(10, "exdiv_p0p5", None, cadence="daily", mode="event_div", extra={"pre": 0, "post": 5}, note="VIABLE")

    # 11 TOM
    add(11, "tom_l1_f3", None, cadence="daily", mode="calendar_tom", extra={"last": 1, "first": 3}, note="VIABLE")
    add(11, "tom_l2_f4", None, cadence="daily", mode="calendar_tom", extra={"last": 2, "first": 4}, note="VIABLE")
    add(11, "tom_f3_only", None, cadence="daily", mode="calendar_tom", extra={"last": 0, "first": 3}, note="VIABLE")

    # 13 accruals
    add(13, "accruals_low", score_accruals_low, note="DEGRADED oi-cf proxy")
    add(13, "accruals_low_q20", score_accruals_low, extra={"extreme_frac": 0.20}, note="DEGRADED")
    add(13, "accruals_low_qual", score_accruals_low, extra={"quality_half": True}, note="DEGRADED")

    # 14 asset growth
    add(14, "asset_growth_low", score_asset_growth_low, note="VIABLE")
    add(14, "asset_growth_low_q20", score_asset_growth_low, extra={"extreme_frac": 0.20}, note="VIABLE")
    add(14, "asset_growth_low_qual", score_asset_growth_low, extra={"quality_half": True}, note="VIABLE")

    # 15 issuance
    add(15, "issuance_low", score_net_issuance_low, note="VIABLE")
    add(15, "issuance_low_q20", score_net_issuance_low, extra={"extreme_frac": 0.20}, note="VIABLE")
    add(15, "issuance_low_qual", score_net_issuance_low, extra={"quality_half": True}, note="VIABLE")

    # 16 short interest
    add(16, "si_dtc_low", score_short_dtc_low, note="DEGRADED +14d lag")
    add(16, "si_dtc_high", score_short_dtc_high, note="DEGRADED +14d lag")
    add(16, "si_dtc_decrease", score_short_dtc_decrease, note="DEGRADED +14d lag")

    # 17 52w
    add(17, "near_52w_high", score_52w_high_near, note="VIABLE")
    add(17, "far_52w_high", score_52w_high_far, note="VIABLE")
    add(17, "near_52w_qual", score_52w_high_near, extra={"quality_half": True}, note="VIABLE")

    # 18 Piotroski
    add(18, "fscore_high", score_piotroski_partial, note="DEGRADED partial; corr risk vs S1 quality")
    add(18, "fscore_high_q20", score_piotroski_partial, extra={"extreme_frac": 0.20}, note="DEGRADED")
    add(18, "fscore_high_qual", score_piotroski_partial, extra={"quality_half": True}, note="DEGRADED")

    # 19 distress
    add(19, "distress_safe", score_distress_safe, note="DEGRADED Z-proxy")
    add(19, "distress_risky", score_distress_risky, note="DEGRADED Z-proxy")
    add(19, "distress_safe_qual", score_distress_safe, extra={"quality_half": True}, note="DEGRADED")

    # 22 capex
    add(22, "capex_low", score_capex_low, note="VIABLE")
    add(22, "capex_low_q20", score_capex_low, extra={"extreme_frac": 0.20}, note="VIABLE")
    add(22, "capex_low_qual", score_capex_low, extra={"quality_half": True}, note="VIABLE")

    # 23 NOA
    add(23, "noa_low", score_noa_low, note="DEGRADED proxy")
    add(23, "noa_low_q20", score_noa_low, extra={"extreme_frac": 0.20}, note="DEGRADED")
    add(23, "noa_low_qual", score_noa_low, extra={"quality_half": True}, note="DEGRADED")

    # 25 margin trend
    add(25, "op_margin_expand", score_margin_expand_op, note="DEGRADED (op_margin not GM)")
    add(25, "gp_expand", score_margin_expand_gp, note="DEGRADED (GP/assets)")
    add(25, "op_margin_expand_qual", score_margin_expand_op, extra={"quality_half": True}, note="DEGRADED")

    # 26 MAX
    add(26, "max_low", score_max_low, note="VIABLE")
    add(26, "max_low_q20", score_max_low, extra={"extreme_frac": 0.20}, note="VIABLE")
    add(26, "max_low_qual", score_max_low, extra={"quality_half": True}, note="VIABLE")

    return cats
