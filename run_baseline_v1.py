#!/usr/bin/env python3
"""Run Baseline v1 over required validation windows and record experiment_history.

Periods (change only START_DATE / END_DATE):
  * 2018–2020
  * 2021–2022
  * 2023–2025
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from engine.baseline_engine import BaselineEngineV1
from engine.research.experiment_history import ExperimentHistory
from engine.strategy import load_config
from engine.strategy_baseline_v1 import strategy_id, strategy_knobs

PERIODS = [
    ("2018-01-01", "2020-12-31", "2018-2020"),
    ("2021-01-01", "2022-12-31", "2021-2022"),
    ("2023-01-01", "2025-12-31", "2023-2025"),
]

FORBIDDEN = [
    "stop_loss",
    "cooldown",
    "ema",
    "exhaustion",
    "hysteresis",
    "industry_cap",
    "correlation",
]


def verify_strategy_clean(paths: list[Path]) -> None:
    """Ensure excluded feature keywords do not appear in baseline strategy logic."""
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        # Allow "Future experiment candidate" comments that mention excluded ideas
        stripped = "\n".join(
            ln for ln in text.splitlines() if "future experiment candidate" not in ln
        )
        for kw in FORBIDDEN:
            if re.search(rf"\b{re.escape(kw)}\b", stripped):
                raise SystemExit(f"Forbidden keyword {kw!r} found in {path}")
    print("[VALIDATION] strategy modules clean of excluded keywords:", ", ".join(FORBIDDEN))


def run_period(config_path: str, start: str, end: str, label: str) -> dict:
    cfg = load_config(config_path)
    cfg["start_date"] = start
    cfg["end_date"] = end
    print("\n" + "#" * 72)
    print(f"# PERIOD {label}: START_DATE={start}  END_DATE={end}")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path=config_path)
    equity = engine.run()
    arts = engine.research_artifacts or {}
    comparison = arts.get("comparison") or {}
    return {
        "label": label,
        "start": start,
        "end": end,
        "n_equity_days": 0 if equity is None or equity.empty else len(equity),
        "comparison": comparison,
        "kpi": arts.get("kpi"),
        "qqq_kpi": arts.get("qqq_kpi"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline v1 multi-period validation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--history",
        default="docs/experiments/BASELINE_V1/experiment_history.json",
    )
    parser.add_argument(
        "--skip-keyword-check",
        action="store_true",
        help="Skip forbidden-keyword scan (not recommended)",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    strat_paths = [
        root / "engine" / "strategy_baseline_v1.py",
        root / "engine" / "baseline_engine.py",
    ]
    if not args.skip_keyword_check:
        verify_strategy_clean(strat_paths)

    results = []
    for start, end, label in PERIODS:
        results.append(run_period(args.config, start, end, label))

    hist_path = Path(args.history)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    history = ExperimentHistory(hist_path)

    metrics = {
        "strategy_id": strategy_id(),
        "knobs": strategy_knobs(),
        "periods": {
            r["label"]: {
                "window": {"start": r["start"], "end": r["end"]},
                "n_equity_days": r["n_equity_days"],
                "comparison": r["comparison"],
            }
            for r in results
        },
    }
    exp_id = history.append(
        parent_exp=None,
        changed_variables={
            "strategy": strategy_id(),
            "entry": "mom_12_1_only",
            "exit": "atr_trail_2.5",
            "sizing": "equal_weight",
            "leverage": 1.0,
        },
        hypothesis="Minimal 12-1 momentum + ATR trail baseline for future factor experiments",
        expected_outcome="Runnable KPIs vs QQQ B&H on 2018-20 / 2021-22 / 2023-25",
        actual_outcome=json.dumps(
            {
                label: (metrics["periods"][label]["comparison"] or {}).get("Alpha_CAGR")
                for label in metrics["periods"]
            },
            default=str,
        ),
        decision="BASELINE",
        metrics=metrics,
        notes="Infrastructure preserved; strategy isolated in strategy_baseline_v1.py",
    )

    summary_path = hist_path.parent / "period_results.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote experiment_history -> {hist_path} (id={exp_id})")
    print(f"Wrote period_results   -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
