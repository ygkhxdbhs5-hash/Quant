"""Evidence-based research recommendation engine with confidence labels."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _confidence(n: int) -> str:
    if n >= 80:
        return "High"
    if n >= 30:
        return "Medium"
    return "Low"


def build_research_recommendation_report(
    trades: pd.DataFrame,
    toggles: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze diagnostic trade logs → bottlenecks + next A/B experiment proposal."""
    findings: List[Dict[str, Any]] = []
    n = 0 if trades is None or trades.empty else len(trades)

    if n == 0:
        return {
            "sample_size": 0,
            "findings": [],
            "next_experiment": {
                "hypothesis": "Insufficient closed trades for inference.",
                "changed_variables": {},
                "expected_outcome": "n/a",
                "sample_size": 0,
                "statistical_confidence": "Low",
                "consistency": "n/a",
                "estimated_impact": "n/a",
                "risk_of_overfitting": "High — do not change strategy on empty sample.",
            },
            "protocol_note": (
                "Never recommend strategy modifications based on a single backtest. "
                "Require multi-period / multi-seed confirmation."
            ),
        }

    reasons = trades["exit_reason"].astype(str)
    ema_mask = reasons.str.contains("ema9", case=False, na=False)
    atr_mask = reasons.str.contains("atr_trail", case=False, na=False)
    exh_mask = reasons.str.contains("exhaustion", case=False, na=False)
    reb_mask = reasons.str.contains("rebalance", case=False, na=False)

    missed = pd.to_numeric(trades.get("missed_upside"), errors="coerce")
    saved = pd.to_numeric(trades.get("saved_drawdown"), errors="coerce")
    hold = pd.to_numeric(trades.get("holding_days"), errors="coerce")
    final_r = pd.to_numeric(trades.get("final_return"), errors="coerce")

    def _bucket(mask, label):
        sub_m = missed[mask]
        sub_s = saved[mask]
        sub_f = final_r[mask]
        return {
            "exit_family": label,
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "avg_missed_upside": float(sub_m.mean()) if sub_m.notna().any() else None,
            "avg_saved_drawdown": float(sub_s.mean()) if sub_s.notna().any() else None,
            "avg_final_return": float(sub_f.mean()) if sub_f.notna().any() else None,
            "confidence": _confidence(int(mask.sum())),
        }

    buckets = [
        _bucket(ema_mask, "EMA9"),
        _bucket(atr_mask, "ATR_trail"),
        _bucket(exh_mask, "Exhaustion"),
        _bucket(reb_mask, "Rebalance"),
    ]

    # Bottleneck: EMA9 sensitivity — high share + high missed upside
    ema = buckets[0]
    if ema["n"] >= 10 and (ema["avg_missed_upside"] or 0) > (missed.mean() or 0):
        findings.append(
            {
                "bottleneck": "EMA9 sensitivity",
                "evidence": (
                    f"{ema['share']*100:.1f}% of exits cite EMA9; "
                    f"avg missed upside={ema['avg_missed_upside']:.4f} vs "
                    f"global={missed.mean():.4f}"
                ),
                "suggestion": "A/B: USE_EMA9_EXIT=False (keep ATR/exhaustion).",
                "confidence": ema["confidence"],
                "sample_size": ema["n"],
                "consistency": "single-backtest — requires walk-forward confirmation",
                "estimated_impact": "Reduce premature exits; may increase MDD if ATR alone is loose",
                "risk_of_overfitting": "Medium",
            }
        )

    atr = buckets[1]
    atr_mult = float(toggles.get("ATR_MULTIPLIER", 2.0))
    if atr["n"] >= 10 and (atr["avg_missed_upside"] or 0) > 0.05 and atr_mult <= 2.0:
        findings.append(
            {
                "bottleneck": "ATR tightness",
                "evidence": (
                    f"ATR exits={atr['n']} ({atr['share']*100:.1f}%); "
                    f"avg missed upside={atr['avg_missed_upside']:.4f} at atr_multiplier={atr_mult}"
                ),
                "suggestion": "A/B: ATR_MULTIPLIER 2.0 → 2.5",
                "confidence": atr["confidence"],
                "sample_size": atr["n"],
                "consistency": "single-backtest — requires multi-period confirmation",
                "estimated_impact": "Fewer stop-outs; watch avg saved drawdown decline",
                "risk_of_overfitting": "Medium",
            }
        )

    avg_hold = float(hold.mean()) if hold.notna().any() else None
    if avg_hold is not None and avg_hold < 10 and n >= 30:
        findings.append(
            {
                "bottleneck": "Turnover / short holding period",
                "evidence": f"avg holding days={avg_hold:.1f}; closed trades={n}",
                "suggestion": "A/B: MIN_HOLD_DAYS=3 (block same-week churn exits)",
                "confidence": _confidence(n),
                "sample_size": n,
                "consistency": "single-backtest",
                "estimated_impact": "Lower turnover/costs; may delay valid stops",
                "risk_of_overfitting": "Medium-High",
            }
        )

    # Efficiency diagnostic
    if missed.notna().any() and saved.notna().any():
        eff = (saved / (missed.abs() + 1e-8)).mean()
        findings.append(
            {
                "bottleneck": "Exit efficiency snapshot",
                "evidence": (
                    f"avg missed upside={missed.mean():.4f}, "
                    f"avg saved drawdown={saved.mean():.4f}, "
                    f"efficiency_ratio≈{eff:.3f}"
                ),
                "suggestion": "Track efficiency_ratio across experiments; do not optimize on one run",
                "confidence": _confidence(n),
                "sample_size": n,
                "consistency": "descriptive only",
                "estimated_impact": "n/a (diagnostic)",
                "risk_of_overfitting": "Low if used only as monitor",
            }
        )

    # Pick next experiment: highest-confidence actionable finding
    actionable = [f for f in findings if f["bottleneck"] != "Exit efficiency snapshot"]
    if actionable:
        top = sorted(
            actionable,
            key=lambda x: {"High": 2, "Medium": 1, "Low": 0}[x["confidence"]],
            reverse=True,
        )[0]
        next_exp = {
            "hypothesis": top["suggestion"],
            "changed_variables": _parse_changed_vars(top["suggestion"], toggles),
            "expected_outcome": top["estimated_impact"],
            "sample_size": top["sample_size"],
            "statistical_confidence": top["confidence"],
            "consistency": top["consistency"],
            "estimated_impact": top["estimated_impact"],
            "risk_of_overfitting": top["risk_of_overfitting"],
            "parent_note": "Require ≥2 out-of-sample windows before adopting.",
        }
    else:
        next_exp = {
            "hypothesis": "Collect larger trade sample before changing toggles",
            "changed_variables": {},
            "expected_outcome": "n/a",
            "sample_size": n,
            "statistical_confidence": _confidence(n),
            "consistency": "n/a",
            "estimated_impact": "n/a",
            "risk_of_overfitting": "High if changing without signal",
        }

    return {
        "sample_size": n,
        "exit_family_stats": buckets,
        "findings": findings,
        "next_experiment": next_exp,
        "protocol_note": (
            "Never recommend strategy modifications based on a single backtest. "
            "Every recommendation must include sample size, confidence, consistency, "
            "estimated impact, and overfitting risk. Log results in Experiment History."
        ),
    }


def _parse_changed_vars(suggestion: str, toggles: Dict[str, Any]) -> Dict[str, Any]:
    s = suggestion.upper()
    out: Dict[str, Any] = {}
    if "USE_EMA9_EXIT=FALSE" in s.replace(" ", ""):
        out["USE_EMA9_EXIT"] = False
    if "ATR_MULTIPLIER" in s and "2.5" in suggestion:
        out["ATR_MULTIPLIER"] = 2.5
    if "MIN_HOLD_DAYS=3" in s.replace(" ", ""):
        out["MIN_HOLD_DAYS"] = 3
    return out


def format_recommendation_report(rep: Dict[str, Any]) -> str:
    lines = ["=" * 64, " RESEARCH RECOMMENDATION REPORT", "=" * 64]
    lines.append(f"Sample size (closed trades): {rep.get('sample_size')}")
    lines.append(f"Protocol: {rep.get('protocol_note')}")
    lines.append("")
    lines.append("Exit family stats:")
    for b in rep.get("exit_family_stats") or []:
        lines.append(
            f"  - {b['exit_family']}: n={b['n']} share={b['share']*100:.1f}% "
            f"missed={b['avg_missed_upside']} saved={b['avg_saved_drawdown']} "
            f"conf={b['confidence']}"
        )
    lines.append("")
    lines.append("Findings:")
    for f in rep.get("findings") or []:
        lines.append(f"  [{f['confidence']}] {f['bottleneck']}")
        lines.append(f"      evidence: {f['evidence']}")
        lines.append(f"      suggestion: {f['suggestion']}")
        lines.append(
            f"      n={f['sample_size']} | consistency={f['consistency']} | "
            f"impact={f['estimated_impact']} | overfit_risk={f['risk_of_overfitting']}"
        )
    lines.append("")
    ne = rep.get("next_experiment") or {}
    lines.append("Next A/B experiment proposal:")
    for k in (
        "hypothesis",
        "changed_variables",
        "expected_outcome",
        "sample_size",
        "statistical_confidence",
        "consistency",
        "estimated_impact",
        "risk_of_overfitting",
    ):
        lines.append(f"  - {k}: {ne.get(k)}")
    lines.append("=" * 64)
    return "\n".join(lines)
