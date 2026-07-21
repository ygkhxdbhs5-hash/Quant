"""CMVS factor predictive power audit (observation only).

Computes for each CMVS factor (BBS, VZS, CPS, RSIS, RSS):
  - IC (Pearson) and "Rank IC" (ranking by IC)
  - Spearman correlation
  - future 1M / 3M / 6M returns
  - monotonicity by decile + hit rate
Then computes marginal contribution to the composite score via:
  - leave-one-out composite (remove each weighted component)

Never modifies trading logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CMVS_FACTORS = ("bbs", "vzs", "cps", "rsis", "rss")


def _safe_pearson(x: pd.Series, y: pd.Series) -> Optional[float]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return None
    if float(x.std(ddof=0) or 0.0) == 0.0 or float(y.std(ddof=0) or 0.0) == 0.0:
        return None
    return float(x.corr(y, method="pearson"))


def _safe_spearman(x: pd.Series, y: pd.Series) -> Optional[float]:
    """Spearman rank correlation without SciPy.

    Computes Spearman as Pearson correlation of ranks.
    """
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return None
    if float(x.std(ddof=0) or 0.0) == 0.0 or float(y.std(ddof=0) or 0.0) == 0.0:
        return None
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    v = xr.corr(yr, method="pearson")
    if pd.isna(v):
        return None
    return float(v)


def _assign_deciles_by_rank(x: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Stable decile assignment using rank order.

    Returns int array in [0, n_bins-1], where 0 is lowest factor score.
    """
    x = pd.to_numeric(x, errors="coerce")
    x = x.dropna()
    if x.empty:
        return np.array([], dtype=int)
    ranks = x.rank(method="first")  # 1..n (unique)
    # Convert to [0, n_bins-1] with quantile-like behavior.
    pct = (ranks - 1) / max(len(x) - 1, 1)
    dec = np.floor(pct * (n_bins - 1e-12)).astype(int)
    dec = np.clip(dec, 0, n_bins - 1)
    return dec.to_numpy()


def _decile_monotonicity_score(decile_mean_returns: np.ndarray) -> Optional[float]:
    """Spearman correlation between decile index and mean future return."""
    if len(decile_mean_returns) != 10:
        return None
    y = pd.Series(decile_mean_returns, dtype=float)
    if y.isna().all():
        return None
    idx = pd.Series(np.arange(10), dtype=float)
    mask = idx.notna() & y.notna()
    if mask.sum() < 2:
        return None
    # Spearman without SciPy: Pearson on ranks
    xr = idx[mask].rank(method="average")
    yr = y[mask].rank(method="average")
    v = xr.corr(yr, method="pearson")
    if pd.isna(v):
        return None
    return float(v)


def _weighted_mean_sum(sum_: float, count: float) -> Optional[float]:
    if count <= 0:
        return None
    return float(sum_ / count)


@dataclass
class FactorAuditResult:
    horizons: Dict[str, int]
    factor_metrics: Dict[str, Any]
    marginal_contribution: Dict[str, Any]
    recommendations: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizons": self.horizons,
            "factor_metrics": self.factor_metrics,
            "marginal_contribution": self.marginal_contribution,
            "recommendations": self.recommendations,
        }


class CMVSAuditor:
    def __init__(
        self,
        engine: Any,
        *,
        horizons_days: Optional[Dict[str, int]] = None,
        max_rebalance_dates: Optional[int] = None,
        min_universe_size: int = 5,
    ) -> None:
        self.engine = engine
        self.horizons_days = horizons_days or {"1M": 21, "3M": 63, "6M": 126}
        self.max_rebalance_dates = max_rebalance_dates
        self.min_universe_size = min_universe_size

        # Composite weights used by the engine
        self.weights = {
            "bbs": float(getattr(engine, "w1", 0.2)),
            "vzs": float(getattr(engine, "w2", 0.2)),
            "cps": float(getattr(engine, "w3", 0.2)),
            "rsis": float(getattr(engine, "w4", 0.2)),
            "rss": float(getattr(engine, "w5", 0.2)),
        }

    def _rebalance_indices(self) -> List[int]:
        trading_days = self.engine.close_m.index
        rebalance_days = set(
            pd.to_datetime(
                self.engine.close_m.groupby(self.engine.close_m.index.to_period("M")).apply(
                    lambda x: x.index[0]
                ).values
            )
        )
        warmup = max(self.engine.MOM_WINDOW, self.engine.LOWVOL_WINDOW) + 5
        indices = [
            i
            for i, d in enumerate(trading_days)
            if i >= warmup and d in rebalance_days
        ]
        if self.max_rebalance_dates is not None:
            indices = indices[: int(self.max_rebalance_dates)]
        return indices

    def audit(self) -> FactorAuditResult:
        rebalance_indices = self._rebalance_indices()

        # Collect per-date correlations and decile aggregates.
        # factor → horizon → list[corr]
        pearson_ic: Dict[str, Dict[str, List[float]]] = {f: {} for f in CMVS_FACTORS}
        spearman_ic: Dict[str, Dict[str, List[float]]] = {f: {} for f in CMVS_FACTORS}
        # factor → horizon → decile sum/count for mean + hit
        decile_ret_sum: Dict[str, Dict[str, np.ndarray]] = {
            f: {h: np.zeros(10, dtype=float) for h in self.horizons_days} for f in CMVS_FACTORS
        }
        decile_ret_count: Dict[str, Dict[str, np.ndarray]] = {
            f: {h: np.zeros(10, dtype=float) for h in self.horizons_days} for f in CMVS_FACTORS
        }
        decile_hit_pos: Dict[str, Dict[str, np.ndarray]] = {
            f: {h: np.zeros(10, dtype=float) for h in self.horizons_days} for f in CMVS_FACTORS
        }
        decile_hit_count: Dict[str, Dict[str, np.ndarray]] = {
            f: {h: np.zeros(10, dtype=float) for h in self.horizons_days} for f in CMVS_FACTORS
        }

        # Composite marginal contribution
        full_ic_pearson: Dict[str, List[float]] = {h: [] for h in self.horizons_days}
        full_ic_spearman: Dict[str, List[float]] = {h: [] for h in self.horizons_days}
        wo_ic_pearson: Dict[str, Dict[str, List[float]]] = {f: {h: [] for h in self.horizons_days} for f in CMVS_FACTORS}
        wo_ic_spearman: Dict[str, Dict[str, List[float]]] = {f: {h: [] for h in self.horizons_days} for f in CMVS_FACTORS}

        for date_idx in rebalance_indices:
            current_date = self.engine.close_m.index[date_idx]
            current_closes = self.engine.close_m.loc[current_date]

            active_symbols = [
                s
                for s in self.engine.close_m.columns
                if s != self.engine.BENCHMARK_TICKER
                and pd.notna(current_closes.get(s))
                and not self.engine.profile_meta.get(s, {}).get("isEtf", False)
            ]
            if not active_symbols:
                continue

            regime = self.engine.determine_market_regime(date_idx, active_symbols)
            if regime is None:
                continue
            if float(regime.exposure) == 0.0:
                # Engine would skip factor/ranking for this month
                continue

            top_adv_symbols = (
                self.engine.dvol_m.loc[current_date, active_symbols]
                .nlargest(min(self.engine.TOP_ADV_POOL, len(active_symbols)))
                .index.tolist()
            )
            universe_symbols = self.engine.get_universe(top_adv_symbols, date_idx, quiet=True)
            if not universe_symbols:
                continue

            factor_df = self.engine.build_factors(date_idx, universe_symbols, quiet=True)
            if factor_df.empty:
                continue

            n = len(factor_df)
            if n < self.min_universe_size:
                continue

            # Precompute composite score for this date
            composite = (
                self.weights["bbs"] * factor_df["bbs"].astype(float)
                + self.weights["vzs"] * factor_df["vzs"].astype(float)
                + self.weights["cps"] * factor_df["cps"].astype(float)
                + self.weights["rsis"] * factor_df["rsis"].astype(float)
                + self.weights["rss"] * factor_df["rss"].astype(float)
            )

            # Factor arrays
            factor_series = {f: factor_df[f].astype(float) for f in CMVS_FACTORS}
            symbols = factor_df["symbol"].astype(str).tolist()

            for horizon_name, horizon_days in self.horizons_days.items():
                future_idx = date_idx + horizon_days
                if future_idx >= len(self.engine.close_m.index):
                    continue

                # Future returns aligned to factor_df row order
                now_px = self.engine.close_m.loc[current_date, symbols].astype(float).to_numpy()
                fut_px = self.engine.close_m.iloc[future_idx].reindex(symbols).astype(float).to_numpy()
                y = fut_px / now_px - 1.0
                mask = np.isfinite(y) & np.isfinite(composite.to_numpy())
                if mask.sum() < self.min_universe_size:
                    continue

                y_series = pd.Series(y[mask])
                for f in CMVS_FACTORS:
                    x_series = pd.Series(factor_series[f].to_numpy()[mask])
                    # ICs
                    pc = _safe_pearson(x_series, y_series)
                    sc = _safe_spearman(x_series, y_series)
                    if pc is not None:
                        pearson_ic[f].setdefault(horizon_name, []).append(pc)
                    if sc is not None:
                        spearman_ic[f].setdefault(horizon_name, []).append(sc)

                    # Decile monotonicity + hit rate
                    dec = _assign_deciles_by_rank(x_series)
                    # x_series is only masked subset; dec matches that subset length.
                    for d in range(10):
                        sel = dec == d
                        if not np.any(sel):
                            continue
                        yy = y_series.to_numpy()[sel]
                        decile_ret_sum[f][horizon_name][d] += float(np.sum(yy))
                        decile_ret_count[f][horizon_name][d] += float(len(yy))
                        decile_hit_pos[f][horizon_name][d] += float(np.sum(yy > 0))
                        decile_hit_count[f][horizon_name][d] += float(len(yy))

                # Composite ICs and leave-one-out ICs
                comp_masked = composite.to_numpy()[mask]
                full_p = _safe_pearson(pd.Series(comp_masked), y_series)
                full_s = _safe_spearman(pd.Series(comp_masked), y_series)
                if full_p is not None:
                    full_ic_pearson[horizon_name].append(full_p)
                if full_s is not None:
                    full_ic_spearman[horizon_name].append(full_s)

                for f in CMVS_FACTORS:
                    wo = comp_masked - self.weights[f] * factor_series[f].to_numpy()[mask].astype(float)
                    wo_p = _safe_pearson(pd.Series(wo), y_series)
                    wo_s = _safe_spearman(pd.Series(wo), y_series)
                    if wo_p is not None:
                        wo_ic_pearson[f][horizon_name].append(wo_p)
                    if wo_s is not None:
                        wo_ic_spearman[f][horizon_name].append(wo_s)

        # Summarize into reports
        factor_metrics: Dict[str, Any] = {}
        for f in CMVS_FACTORS:
            factor_metrics[f] = {}
            for horizon_name in self.horizons_days:
                p_list = pearson_ic[f].get(horizon_name) or []
                s_list = spearman_ic[f].get(horizon_name) or []

                def _mean_std(vals: List[float]) -> Dict[str, Optional[float]]:
                    if not vals:
                        return {"mean": None, "std": None, "n": 0}
                    arr = np.asarray(vals, dtype=float)
                    return {
                        "mean": float(arr.mean()),
                        "std": float(arr.std(ddof=0)),
                        "n": int(len(arr)),
                    }

                pearson_summary = _mean_std(p_list)
                spearman_summary = _mean_std(s_list)

                dec_mean = []
                hit_rate = []
                for d in range(10):
                    cnt = decile_ret_count[f][horizon_name][d]
                    if cnt <= 0:
                        dec_mean.append(None)
                        hit_rate.append(None)
                    else:
                        dec_mean.append(float(decile_ret_sum[f][horizon_name][d] / cnt))
                        hr_cnt = decile_hit_count[f][horizon_name][d]
                        hr_pos = decile_hit_pos[f][horizon_name][d]
                        hit_rate.append(float(hr_pos / hr_cnt) if hr_cnt > 0 else None)

                monot_score = _decile_monotonicity_score(
                    np.asarray([x if x is not None else np.nan for x in dec_mean], dtype=float)
                )

                factor_metrics[f][horizon_name] = {
                    "ic_pearson": pearson_summary,
                    "spearman": spearman_summary,
                    "monotonicity_by_decile": {
                        "decile_mean_future_return": dec_mean,
                        "decile_hit_rate": hit_rate,
                        "monotonicity_score_spearman(decile,mean_return)": monot_score,
                    },
                }

        # Rank IC (by Pearson IC mean) per horizon
        rank_ic: Dict[str, List[Tuple[str, Optional[float]]]] = {h: [] for h in self.horizons_days}
        for horizon_name in self.horizons_days:
            for f in CMVS_FACTORS:
                m = factor_metrics[f][horizon_name]["ic_pearson"]["mean"]
                rank_ic[horizon_name].append((f, m))
            # Sort: None last
            rank_ic[horizon_name].sort(key=lambda x: (-1e9 if x[1] is None else -x[1]))

        # Marginal contribution: composite IC change when removing component
        marginal_contribution: Dict[str, Any] = {}
        for horizon_name in self.horizons_days:
            marginal_contribution[horizon_name] = {}

            def _mean(vals: List[float]) -> Optional[float]:
                if not vals:
                    return None
                return float(np.mean(np.asarray(vals, dtype=float)))

            full_p = _mean(full_ic_pearson[horizon_name])
            full_s = _mean(full_ic_spearman[horizon_name])
            marginal_contribution[horizon_name]["full_composite_ic"] = {
                "ic_pearson_mean": full_p,
                "spearman_mean": full_s,
                "n_dates": len(full_ic_pearson[horizon_name]),
            }

            for f in CMVS_FACTORS:
                wo_p = _mean(wo_ic_pearson[f][horizon_name])
                wo_s = _mean(wo_ic_spearman[f][horizon_name])
                marginal_contribution[horizon_name][f] = {
                    "composite_without_factor_ic": {
                        "ic_pearson_mean": wo_p,
                        "spearman_mean": wo_s,
                    },
                    "marginal_contribution_to_composite": {
                        "ic_pearson_loss_when_removed": None
                        if full_p is None or wo_p is None
                        else float(full_p - wo_p),
                        "spearman_loss_when_removed": None
                        if full_s is None or wo_s is None
                        else float(full_s - wo_s),
                    },
                }

        # Recommendations (observation-only) — deterministic rules, no optimization.
        # Remove: factors with consistently weak / negative Pearson IC mean across horizons
        remove_candidates: List[str] = []
        reweight_candidates: List[str] = []

        for f in CMVS_FACTORS:
            ic_means = []
            for h in self.horizons_days:
                ic_means.append(factor_metrics[f][h]["ic_pearson"]["mean"])
            # Consistently None/weak/negative
            valid = [v for v in ic_means if v is not None]
            if not valid:
                continue
            if max(valid) <= 0.0:
                remove_candidates.append(f)
            else:
                reweight_candidates.append(f)

        # Suggested reweights proportional to positive Pearson IC mean (clipped).
        # This is a direct mapping from observed IC, not an optimization routine.
        positive_scores = {f: [] for f in CMVS_FACTORS}
        for f in CMVS_FACTORS:
            for h in self.horizons_days:
                v = factor_metrics[f][h]["ic_pearson"]["mean"]
                if v is not None and v > 0:
                    positive_scores[f].append(v)

        score = {}
        for f in CMVS_FACTORS:
            if positive_scores[f]:
                score[f] = float(np.mean(positive_scores[f]))
            else:
                score[f] = 0.0

        total_score = sum(score.values())
        suggested_weights = {}
        if total_score > 0:
            for f in CMVS_FACTORS:
                suggested_weights[f] = score[f] / total_score
        else:
            suggested_weights = dict(self.weights)

        # Add: no additional audited factors exist beyond the CMVS set.
        recommendations = {
            "remove_factors": remove_candidates,
            "reweight_factors_to_proportional_to_positive_ic": reweight_candidates,
            "suggested_relative_weights_from_positive_ic": suggested_weights,
            "add_factors": [],
            "notes": "Recommendations are derived only from observed IC / Spearman and marginal contribution. No trading logic is modified.",
            "rank_ic_by_horizon_pearson": rank_ic,
        }

        return FactorAuditResult(
            horizons=self.horizons_days,
            factor_metrics=factor_metrics,
            marginal_contribution=marginal_contribution,
            recommendations=recommendations,
        )

