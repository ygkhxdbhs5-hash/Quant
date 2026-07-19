"""Mandatory research reporting templates (Facts only; no unsupported advice)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def format_research_recommendation(
    *,
    fact: str,
    interpretation: str,
    confidence: str,
    evidence: str,
    next_experiment: str,
    why_information_gain: str,
) -> str:
    return "\n".join(
        [
            "=" * 64,
            " RESEARCH RECOMMENDATION TEMPLATE",
            "=" * 64,
            f"Fact:\n  {fact}",
            "",
            f"Interpretation:\n  {interpretation}",
            "",
            f"Confidence:\n  {confidence}",
            "",
            f"Evidence:\n  {evidence}",
            "",
            f"Next Recommended Experiment:\n  {next_experiment}",
            "",
            f"Why this experiment maximizes Information Gain:\n  {why_information_gain}",
            "=" * 64,
        ]
    )


def format_experiment_review(
    *,
    hypothesis: str,
    changed_variables: Dict[str, Any],
    baseline: Dict[str, Any],
    experiment: Dict[str, Any],
    delta: Dict[str, Any],
    new_fact: str,
    decision: str,
) -> str:
    lines = [
        "=" * 64,
        " EXPERIMENT REVIEW TEMPLATE",
        "=" * 64,
        f"Hypothesis:\n  {hypothesis}",
        "",
        f"Changed Variables:\n  {changed_variables}",
        "",
        "Baseline:",
    ]
    for k, v in baseline.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Experiment:")
    for k, v in experiment.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Δ(Delta):")
    metrics = (delta.get("metrics") or {}) if isinstance(delta, dict) else {}
    for name, row in metrics.items():
        lines.append(
            f"  - {name}: baseline={row.get('baseline')} exp={row.get('experiment')} Δ={row.get('delta')}"
        )
    lines += [
        "",
        f"새롭게 얻은 Fact:\n  {new_fact}",
        "",
        f"Decision (PASS / REPEAT / REJECT):\n  {decision}",
        "=" * 64,
    ]
    return "\n".join(lines)


def decide_from_delta(
    delta: Dict[str, Any],
    *,
    only_exit_rank_changed: bool,
    abort_reasons: Optional[List[str]] = None,
) -> str:
    """Decision rule: data-supported only. No optimization claims."""
    if abort_reasons:
        return "REJECT"
    if not only_exit_rank_changed:
        return "REJECT"
    metrics = delta.get("metrics") or {}
    # If no measurable change in turnover/holding/trades, experiment is uninformative on this sample
    keys = ("n_closed_trades", "turnover_trades_per_year", "avg_holding_period_days")
    deltas = [metrics.get(k, {}).get("delta") for k in keys]
    if all(d is None or abs(float(d)) < 1e-12 for d in deltas if d is not None) and all(
        d is None or abs(float(d)) < 1e-12 for d in deltas
    ):
        return "REPEAT"
    # Any non-zero behavioral delta → PASS as a completed controlled observation
    if any(d is not None and abs(float(d)) > 0 for d in deltas):
        return "PASS"
    return "REPEAT"
