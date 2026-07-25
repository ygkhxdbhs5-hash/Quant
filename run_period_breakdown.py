#!/usr/bin/env python3
"""Multi-period Baseline v1 diagnostics (Hypothesis A vs B).

Runs strategy_baseline_v1 via BaselineEngineV1 independently over several
START_DATE/END_DATE windows. Does NOT change trading logic.

Usage:
  python run_period_breakdown.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.baseline_diagnostics import analyze_run, format_full_report
from engine.baseline_engine import BaselineEngineV1
from engine.strategy import load_config

PERIODS = [
    ("2018-01-01", "2020-12-31", "2018-2020"),
    ("2021-01-01", "2022-12-31", "2021-2022"),
    ("2023-01-01", "2025-12-31", "2023-2025"),
    ("2022-01-01", "2022-12-31", "2022 only"),
    ("2022-01-01", "2026-06-30", "2022-2026"),
]


def run_one(config_path: str, start: str, end: str, label: str):
    cfg = load_config(config_path)
    cfg["start_date"] = start
    cfg["end_date"] = end
    print("\n" + "#" * 72)
    print(f"# RUNNING {label}: START_DATE={start}  END_DATE={end}")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path=config_path)
    engine.run()
    return analyze_run(engine, label, start, end)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline v1 period breakdown diagnostics")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_DIAGNOSTICS",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for start, end, label in PERIODS:
        results.append(run_one(args.config, start, end, label))

    report = format_full_report(results, full_label="2022-2026")
    report_path = out_dir / "diagnostic_report.txt"
    report_path.write_text(report, encoding="utf-8")
    (out_dir / "diagnostic_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    print("\n")
    print(report)
    print(f"\nWrote report  -> {report_path}")
    print(f"Wrote JSON    -> {out_dir / 'diagnostic_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
