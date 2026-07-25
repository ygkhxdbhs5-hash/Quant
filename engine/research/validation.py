"""Research validation checklist (baseline identity + log completeness)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from engine.research.config_toggles import ResearchToggles


def run_research_validation_checklist(
    toggles: ResearchToggles,
    trades: pd.DataFrame,
    *,
    max_portfolio_size: int,
    selection_buffer_size: int,
    baseline_trade_count: Optional[int] = None,
    baseline_turnover: Optional[float] = None,
    observed_turnover: Optional[float] = None,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    baseline_defaults = toggles.is_baseline_defaults(max_portfolio_size, selection_buffer_size)
    checks.append(
        {
            "name": "toggles_reproduce_baseline_defaults",
            "pass": baseline_defaults,
            "detail": toggles.as_dict(),
            "note": (
                "Event-driven defaults: ENTRY_RANK=10 (buy), EXIT_RANK=selection_buffer "
                "(hold), MAX_PORTFOLIO_SIZE=max_portfolio_size (capacity)."
            ),
        }
    )

    n_trades = 0 if trades is None or trades.empty else len(trades)
    if baseline_trade_count is not None:
        checks.append(
            {
                "name": "baseline_trade_count_unchanged",
                "pass": n_trades == int(baseline_trade_count),
                "detail": {"observed": n_trades, "baseline": baseline_trade_count},
            }
        )
    else:
        checks.append(
            {
                "name": "baseline_trade_count_unchanged",
                "pass": True,
                "detail": {
                    "observed": n_trades,
                    "baseline": None,
                    "note": "No prior baseline fingerprint on disk; recorded for next run.",
                },
            }
        )

    if baseline_turnover is not None and observed_turnover is not None:
        ok = abs(float(observed_turnover) - float(baseline_turnover)) < 1e-9
        checks.append(
            {
                "name": "baseline_turnover_unchanged",
                "pass": ok,
                "detail": {"observed": observed_turnover, "baseline": baseline_turnover},
            }
        )
    else:
        checks.append(
            {
                "name": "baseline_turnover_unchanged",
                "pass": True,
                "detail": {
                    "observed": observed_turnover,
                    "baseline": baseline_turnover,
                    "note": "Fingerprint absent; skip strict compare.",
                },
            }
        )

    required_cols = [
        "entry_date",
        "exit_date",
        "holding_days",
        "exit_reason",
        "entry_rank",
        "exit_rank",
        "entry_cmvs",
        "exit_cmvs",
        "entry_rsi",
        "exit_rsi",
        "entry_atr",
        "exit_atr",
        "peak_return",
        "final_return",
        "forward_return_5d",
        "forward_return_10d",
        "forward_return_20d",
        "missed_upside",
        "saved_drawdown",
        "post_exit_max_return",
        "post_exit_max_drawdown",
        "days_to_peak",
        "days_to_trough",
    ]
    if trades is None or trades.empty:
        checks.append(
            {
                "name": "diagnostic_logs_populated",
                "pass": False,
                "detail": "No closed trades — journal empty (may be expected on tiny samples).",
            }
        )
    else:
        missing = [c for c in required_cols if c not in trades.columns]
        # Counterfactual columns should be non-all-null when horizon available
        cf_ok = trades["missed_upside"].notna().any() if "missed_upside" in trades.columns else False
        checks.append(
            {
                "name": "diagnostic_logs_populated",
                "pass": (not missing) and cf_ok,
                "detail": {"missing_columns": missing, "counterfactual_any": cf_ok, "n": len(trades)},
            }
        )

    all_pass = all(c["pass"] for c in checks)
    return {"all_pass": all_pass, "checks": checks}


def format_validation_checklist(result: Dict[str, Any]) -> str:
    lines = ["=" * 64, " RESEARCH VALIDATION CHECKLIST", "=" * 64]
    for c in result.get("checks") or []:
        flag = "PASS" if c.get("pass") else "FAIL"
        lines.append(f"[{flag}] {c.get('name')}")
        lines.append(f"       {c.get('detail')}")
    lines.append(f"Overall: {'PASS' if result.get('all_pass') else 'FAIL'}")
    lines.append("=" * 64)
    return "\n".join(lines)
