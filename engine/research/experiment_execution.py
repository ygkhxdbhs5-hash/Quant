"""Experiment execution & delta reporting layer (no trading-logic changes).

Research Rules enforced here:
  #1 Only one behavioral variable may change per experiment.
  #2 No optimization; recommendations only when supported by observed results.
  #3 Baseline runs first; experiment second; both results preserved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# Behavioral knobs that must match between arms except the allowed single change.
BEHAVIORAL_COMPARE_KEYS = (
    "USE_EMA9_EXIT",
    "EMA_EXIT_LENGTH",
    "USE_ATR_EXIT",
    "ATR_MULTIPLIER",
    "USE_EXHAUSTION_EXIT",
    "ENTRY_RANK",
    "EXIT_RANK",
    "MIN_HOLD_DAYS",
    "USE_TIME_STOP",
    "TIME_STOP_DAYS",
    "USE_STOP_LOSS",
    "STOP_LOSS_PCT",
    "MONTHLY_REBALANCE",
    "MAX_PORTFOLIO_SIZE",
    "BENCHMARK_TICKER",
    "TOP_ADV_POOL",
    "CORR_THRESHOLD",
    "MAX_INDUSTRY_WEIGHT",
    "COMMISSION_RATE",
    "SLIPPAGE_RATE",
    "UNIVERSE_SAMPLE_SIZE",
    "UNIVERSE_SAMPLE_SEED",
    "START_DATE",
    "END_DATE",
    "N_PRICE_COLUMNS",
    "N_PRICE_ROWS",
    "PANELS_SHA256",
    "UNIVERSE_SHA256",
)


def extract_behavioral_identity(engine, cfg: Dict[str, Any], fingerprint: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot of behavioral + data identity for single-variable validation."""
    cache = (fingerprint.get("data_cache_version") or {}) if fingerprint else {}
    return {
        "USE_EMA9_EXIT": bool(engine.USE_EMA9_EXIT),
        "EMA_EXIT_LENGTH": int(getattr(engine, "EMA_EXIT_LENGTH", 9)),
        "USE_ATR_EXIT": bool(getattr(engine, "USE_ATR_EXIT", True)),
        "ATR_MULTIPLIER": float(engine.atr_multiplier),
        "USE_EXHAUSTION_EXIT": bool(getattr(engine, "USE_EXHAUSTION_EXIT", True)),
        "ENTRY_RANK": int(engine.ENTRY_RANK),
        "EXIT_RANK": int(engine.EXIT_RANK),
        "MIN_HOLD_DAYS": int(engine.MIN_HOLD_DAYS),
        "USE_TIME_STOP": bool(engine.USE_TIME_STOP),
        "TIME_STOP_DAYS": int(engine.TIME_STOP_DAYS),
        "USE_STOP_LOSS": bool(getattr(engine, "USE_STOP_LOSS", False)),
        "STOP_LOSS_PCT": float(getattr(engine, "STOP_LOSS_PCT", 0.0)),
        "MONTHLY_REBALANCE": bool(getattr(engine, "MONTHLY_REBALANCE", True)),
        "MAX_PORTFOLIO_SIZE": int(engine.MAX_PORTFOLIO_SIZE),
        "BENCHMARK_TICKER": str(engine.BENCHMARK_TICKER),
        "TOP_ADV_POOL": int(engine.TOP_ADV_POOL),
        "CORR_THRESHOLD": float(engine.CORR_THRESHOLD),
        "MAX_INDUSTRY_WEIGHT": float(engine.MAX_INDUSTRY_WEIGHT),
        "COMMISSION_RATE": float(engine.COMMISSION_RATE),
        "SLIPPAGE_RATE": float(engine.SLIPPAGE_RATE),
        "UNIVERSE_SAMPLE_SIZE": cfg.get("universe_sample_size"),
        "UNIVERSE_SAMPLE_SEED": cfg.get("universe_sample_seed"),
        "START_DATE": cfg.get("start_date"),
        "END_DATE": cfg.get("end_date"),
        "N_PRICE_COLUMNS": int(engine.close_m.shape[1]),
        "N_PRICE_ROWS": int(engine.close_m.shape[0]),
        "PANELS_SHA256": cache.get("panels_pkl_sha256"),
        "UNIVERSE_SHA256": cache.get("universe_pkl_sha256"),
    }


def compare_behavioral_identity(
    baseline: Dict[str, Any],
    experiment: Dict[str, Any],
    *,
    allowed_changes: Dict[str, Tuple[Any, Any]],
) -> Dict[str, Any]:
    """Return validation result. PASS only if diffs ⊆ allowed_changes with exact values.

    ``allowed_changes`` maps key → (baseline_value, experiment_value).
    """
    diffs: Dict[str, Tuple[Any, Any]] = {}
    for k in BEHAVIORAL_COMPARE_KEYS:
        bv = baseline.get(k)
        ev = experiment.get(k)
        if bv != ev:
            diffs[k] = (bv, ev)

    unexpected = {k: v for k, v in diffs.items() if k not in allowed_changes}
    missing_allowed = []
    wrong_allowed = []
    for k, (exp_b, exp_e) in allowed_changes.items():
        if k not in diffs:
            missing_allowed.append(k)
        else:
            got_b, got_e = diffs[k]
            if got_b != exp_b or got_e != exp_e:
                wrong_allowed.append({k: {"expected": (exp_b, exp_e), "got": (got_b, got_e)}})

    passed = not unexpected and not missing_allowed and not wrong_allowed
    differing = sorted(diffs.keys())
    return {
        "passed": passed,
        "diffs": {k: {"baseline": diffs[k][0], "experiment": diffs[k][1]} for k in differing},
        "unexpected_diffs": {
            k: {"baseline": unexpected[k][0], "experiment": unexpected[k][1]}
            for k in sorted(unexpected)
        },
        "missing_allowed_changes": missing_allowed,
        "wrong_allowed_changes": wrong_allowed,
        "allowed_changes": {
            k: {"baseline": v[0], "experiment": v[1]} for k, v in allowed_changes.items()
        },
    }


def format_validation_failed(validation: Dict[str, Any]) -> str:
    lines = [
        "VALIDATION FAILED",
        "The following behavioral parameter(s) differ beyond the single allowed change:",
    ]
    unexpected = validation.get("unexpected_diffs") or {}
    if unexpected:
        for k, row in unexpected.items():
            lines.append(f"  - {k}: baseline={row.get('baseline')} experiment={row.get('experiment')}")
    for item in validation.get("wrong_allowed_changes") or []:
        lines.append(f"  - wrong allowed change: {item}")
    for k in validation.get("missing_allowed_changes") or []:
        lines.append(f"  - expected change to {k} was not observed")
    if not unexpected and not validation.get("wrong_allowed_changes") and not validation.get(
        "missing_allowed_changes"
    ):
        lines.append("  - (see diffs) " + str(validation.get("diffs")))
    return "\n".join(lines)


def _m(delta: Dict[str, Any], name: str) -> Dict[str, Any]:
    return (delta.get("metrics") or {}).get(name) or {}


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    try:
        x = float(v)
    except Exception:
        return str(v)
    if pct:
        return f"{x:.4%} ({x:.6f})"
    return f"{x:.6f}"


def generate_facts(
    *,
    baseline_exit: int,
    experiment_exit: int,
    delta: Dict[str, Any],
    validation_passed: bool,
) -> List[str]:
    """Facts strictly supported by observed deltas. No speculation."""
    facts: List[str] = []
    if not validation_passed:
        facts.append(
            f"Fact: Single-variable validation failed; EXIT_RANK "
            f"{baseline_exit}→{experiment_exit} comparison is not valid."
        )
        return facts

    label = f"Increasing EXIT_RANK from {baseline_exit} to {experiment_exit}"

    def add_if_nonzero(metric: str, template: str, *, scale: float = 1.0, digits: int = 4) -> None:
        d = _m(delta, metric).get("delta")
        if d is None:
            return
        if abs(float(d)) < 1e-12:
            facts.append(
                f"Fact: {label} produced no change in {metric} "
                f"(Δ={float(d):.6f}) on this sample."
            )
            return
        val = float(d) * scale
        facts.append(template.format(label=label, value=val, delta=float(d)))

    # Turnover % change relative to baseline when baseline != 0
    turn = _m(delta, "turnover_trades_per_year")
    if turn.get("delta") is not None and turn.get("baseline") not in (None, 0):
        rel = float(turn["delta"]) / float(turn["baseline"]) * 100.0
        if abs(float(turn["delta"])) < 1e-12:
            facts.append(
                f"Fact: {label} produced no change in annual turnover "
                f"(Δ={float(turn['delta']):.6f}) on this sample."
            )
        else:
            facts.append(
                f"Fact: {label} changed annual turnover by {rel:.4f}% "
                f"(Δ={float(turn['delta']):.6f} trades/year)."
            )
    else:
        add_if_nonzero(
            "turnover_trades_per_year",
            "Fact: {label} changed annual turnover by Δ={delta:.6f} trades/year.",
        )

    add_if_nonzero(
        "avg_holding_period_days",
        "Fact: {label} changed average holding period by {delta:.6f} days.",
    )
    add_if_nonzero(
        "n_closed_trades",
        "Fact: {label} changed closed trades by {delta:.0f}.",
    )

    cagr_d = _m(delta, "cagr").get("delta")
    if cagr_d is not None:
        if abs(float(cagr_d)) < 1e-6:
            facts.append(
                f"Fact: EXIT_RANK={experiment_exit} produced no statistically meaningful "
                f"change in CAGR (Δ={float(cagr_d):.6f}) on this sample."
            )
        else:
            facts.append(
                f"Fact: {label} changed CAGR by {float(cagr_d):.6f} on this sample."
            )

    for metric, nice in (
        ("total_return", "total return"),
        ("sharpe", "Sharpe"),
        ("sortino", "Sortino"),
        ("calmar", "Calmar"),
        ("mdd", "max drawdown"),
        ("win_rate", "win rate"),
        ("profit_factor", "profit factor"),
        ("avg_missed_upside", "average missed upside"),
        ("avg_saved_drawdown", "average saved drawdown"),
        ("rank_exit_candidates", "rank_exit_candidates"),
        ("ema_preempted_rank_exit", "ema_preempted_rank_exit"),
    ):
        d = _m(delta, metric).get("delta")
        if d is None:
            continue
        if abs(float(d)) < 1e-12:
            continue  # avoid flooding; zero already covered for key metrics
        facts.append(f"Fact: {label} changed {nice} by Δ={float(d):.6f} on this sample.")

    if not facts:
        facts.append(
            f"Fact: {label} produced no measurable metric deltas on this sample."
        )
    return facts


def research_recommendation_from_results(
    *,
    facts: List[str],
    delta: Dict[str, Any],
    n_closed_trades_baseline: int,
    n_closed_trades_experiment: int,
    validation_passed: bool,
    decision: str,
) -> str:
    """Exactly one recommendation, or INSUFFICIENT EVIDENCE."""
    if not validation_passed:
        return "INSUFFICIENT EVIDENCE"
    n = min(int(n_closed_trades_baseline), int(n_closed_trades_experiment))
    if n < 30:
        return "INSUFFICIENT EVIDENCE"

    # All-zero core research deltas → evidence that EXIT_RANK is not identifiable here
    core = ("n_closed_trades", "turnover_trades_per_year", "avg_holding_period_days", "cagr")
    core_deltas = [_m(delta, k).get("delta") for k in core]
    if all(d is not None and abs(float(d)) < 1e-12 for d in core_deltas):
        return "INSUFFICIENT EVIDENCE"

    if decision == "REPEAT":
        return "INSUFFICIENT EVIDENCE"

    # One recommendation from observed facts only
    turn = _m(delta, "turnover_trades_per_year")
    hold = _m(delta, "avg_holding_period_days")
    if turn.get("delta") is not None and abs(float(turn["delta"])) >= 1e-12:
        return (
            "Retain EXIT_RANK as a research variable of interest: observed turnover "
            f"Δ={float(turn['delta']):.6f} under a single-variable design "
            "(no optimization; require out-of-sample repeat)."
        )
    if hold.get("delta") is not None and abs(float(hold["delta"])) >= 1e-12:
        return (
            "Retain EXIT_RANK as a research variable of interest: observed holding-period "
            f"Δ={float(hold['delta']):.6f} days under a single-variable design "
            "(no optimization; require out-of-sample repeat)."
        )
    if facts:
        return (
            "Record the observed single-variable EXIT_RANK facts and repeat the same "
            "experiment on an independent sample before any further change."
        )
    return "INSUFFICIENT EVIDENCE"


def decide_experiment(
    delta: Dict[str, Any],
    *,
    validation_passed: bool,
) -> str:
    if not validation_passed:
        return "REJECT"
    metrics = delta.get("metrics") or {}
    keys = ("n_closed_trades", "turnover_trades_per_year", "avg_holding_period_days")
    deltas = [metrics.get(k, {}).get("delta") for k in keys]
    if all(d is None or abs(float(d)) < 1e-12 for d in deltas):
        return "REPEAT"
    if any(d is not None and abs(float(d)) > 0 for d in deltas):
        return "PASS"
    return "REPEAT"


def format_delta_section(delta: Dict[str, Any]) -> str:
    """Human-readable Delta Report matching the Exp1 contract labels."""
    lines = ["Delta Report", ""]

    def line(label: str, metric: str, *, show_delta: bool = True) -> None:
        row = _m(delta, metric)
        b, e, d = row.get("baseline"), row.get("experiment"), row.get("delta")
        lines.append(f"{label}")
        lines.append(f"  Baseline:    {_fmt(b)}")
        lines.append(f"  Experiment:  {_fmt(e)}")
        if show_delta:
            lines.append(f"  Δ:           {_fmt(d)}")
        lines.append("")

    line("Closed Trades", "n_closed_trades")
    line("CAGR", "cagr")
    line("Total Return", "total_return")
    line("Sharpe", "sharpe")
    line("Sortino", "sortino")
    line("Calmar", "calmar")
    line("Max Drawdown", "mdd")
    line("Win Rate", "win_rate")
    line("Profit Factor", "profit_factor")
    line("Average Holding Days", "avg_holding_period_days")
    line("Average Missed Upside", "avg_missed_upside")
    line("Average Saved Drawdown", "avg_saved_drawdown")
    line("Turnover", "turnover_trades_per_year")
    line("Average Holding Rank", "avg_holding_rank", show_delta=True)
    line("Median Holding Rank", "median_holding_rank", show_delta=True)
    line("P75 Holding Rank", "p75_holding_rank", show_delta=True)
    line("P90 Holding Rank", "p90_holding_rank", show_delta=True)
    line("P95 Holding Rank", "p95_holding_rank", show_delta=True)
    line("EMA-preempted Rank Exits", "ema_preempted_rank_exit")
    line("Rank Exit Candidates", "rank_exit_candidates")
    return "\n".join(lines)


def format_mandatory_review(
    *,
    hypothesis: str,
    changed_variable: str,
    baseline: Dict[str, Any],
    experiment: Dict[str, Any],
    delta_summary: str,
    facts: List[str],
    recommendation: str,
    decision: str,
) -> str:
    lines = [
        "Hypothesis:",
        f"  {hypothesis}",
        "",
        "Changed Variable:",
        f"  {changed_variable}",
        "",
        "Baseline:",
    ]
    for k, v in baseline.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Experiment:")
    for k, v in experiment.items():
        lines.append(f"  {k}: {v}")
    lines += [
        "",
        "Delta:",
        f"  {delta_summary}",
        "",
        "New Fact(s):",
    ]
    for f in facts:
        lines.append(f"  {f}")
    lines += [
        "",
        "Research Recommendation:",
        f"  {recommendation}",
        "",
        "Decision:",
        f"  {decision}",
    ]
    return "\n".join(lines)


def format_experiment1_full_report(
    *,
    architecture_audit: str,
    baseline_verification: str,
    baseline_entry: int,
    baseline_exit: int,
    experiment_exit: int,
    baseline_identity: Dict[str, Any],
    experiment_identity: Dict[str, Any],
    validation: Dict[str, Any],
    delta: Dict[str, Any],
    facts: List[str],
    recommendation: str,
    decision: str,
    review_body: str,
) -> str:
    """Mandatory 8-section output contract."""
    val_status = "PASS" if validation.get("passed") else "FAIL"
    val_detail = (
        "EXIT_RANK is the only behavioral change"
        if validation.get("passed")
        else format_validation_failed(validation)
    )

    sections = [
        "=" * 50,
        "1. Architecture Audit",
        "=" * 50,
        architecture_audit.strip(),
        "",
        "=" * 50,
        "2. Baseline Verification",
        "=" * 50,
        baseline_verification.strip(),
        "",
        "=" * 50,
        "3. Experiment 1 Report",
        "=" * 50,
        "==================================================",
        "EXPERIMENT 1 REPORT",
        "==================================================",
        "",
        "Baseline",
        f"  ENTRY_RANK = {baseline_entry}",
        f"  EXIT_RANK  = {baseline_exit}",
        f"  identity   = {baseline_identity}",
        "",
        "Experiment",
        f"  ENTRY_RANK = {baseline_entry}",
        f"  EXIT_RANK  = {experiment_exit}",
        f"  identity   = {experiment_identity}",
        "",
        "Changed Variable",
        f"  EXIT_RANK: {baseline_exit} → {experiment_exit}",
        "",
        "=" * 50,
        "4. Validation",
        "=" * 50,
        f"Validation: {val_status}",
        val_detail,
        "",
        "=" * 50,
        "5. Delta Report",
        "=" * 50,
        format_delta_section(delta),
        "",
        "=" * 50,
        "6. Fact Generation",
        "=" * 50,
        *[f for f in facts],
        "",
        "=" * 50,
        "7. Research Recommendation",
        "=" * 50,
        recommendation,
        "",
        "=" * 50,
        "8. PASS / REPEAT / REJECT",
        "=" * 50,
        decision,
        "",
        "=" * 50,
        "Mandatory Review Template",
        "=" * 50,
        review_body,
    ]
    return "\n".join(sections)
