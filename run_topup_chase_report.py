#!/usr/bin/env python3
"""Top-up chasing diagnostics + no-chase fill variant comparison (2022-2026).

Runs Fixed CS v2 with chasing ON (Tasks 1–2), then Fixed CS v2 with
DISABLE_TOPUP_CHASING, plus flat 10/30bps benchmarks (Task 3).

Does NOT change momentum entry or ATR exit signal logic.

Usage:
  python run_topup_chase_report.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.baseline_engine import BaselineEngineV1
from engine.strategy import load_config
from engine.report_charts import render_topup_chase_dashboard
from engine.topup_chase_diagnostics import (
    analyze_cost_components,
    analyze_new_entry_vs_topup,
    format_chase_report,
    kpi_with_trades,
)

START = "2022-01-01"
END = "2026-06-30"

VARIANTS = [
    {
        "label": "Fixed CS v2 (chase ON)",
        "cost_model": "corwin_schultz_v2",
        "winsorize_adv": True,
        "disable_topup_chasing": False,
        "flat_cost_one_way": None,
    },
    {
        "label": "Fixed CS v2 + NO-CHASE",
        "cost_model": "corwin_schultz_v2",
        "winsorize_adv": True,
        "disable_topup_chasing": True,
        "flat_cost_one_way": None,
    },
    {
        "label": "Flat 10bps RT (5bps/side)",
        "cost_model": "flat",
        "winsorize_adv": False,
        "disable_topup_chasing": False,
        "flat_cost_one_way": 0.0005,
    },
    {
        "label": "Flat 30bps RT (15bps/side)",
        "cost_model": "flat",
        "winsorize_adv": False,
        "disable_topup_chasing": False,
        "flat_cost_one_way": 0.0015,
    },
]


def run_variant(config_path: str, variant: dict):
    cfg = load_config(config_path)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = variant["cost_model"]
    cfg["winsorize_adv"] = bool(variant["winsorize_adv"])
    cfg["disable_topup_chasing"] = bool(variant["disable_topup_chasing"])
    if variant["flat_cost_one_way"] is not None:
        cfg["flat_cost_one_way"] = variant["flat_cost_one_way"]
    print("\n" + "#" * 72)
    print(f"# {variant['label']}")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path=config_path)
    engine.run()
    metrics = kpi_with_trades(engine)
    metrics["label"] = variant["label"]
    return engine, metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out-dir", default="docs/experiments/BASELINE_V1_TOPUP_CHASE")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = []
    engine_chase = None
    for variant in VARIANTS:
        engine, metrics = run_variant(args.config, variant)
        comparison.append(metrics)
        if (
            variant["cost_model"] == "corwin_schultz_v2"
            and not variant["disable_topup_chasing"]
        ):
            engine_chase = engine

    assert engine_chase is not None
    task1 = analyze_new_entry_vs_topup(engine_chase)
    task2 = analyze_cost_components(engine_chase)
    report = format_chase_report(
        task1=task1, task2=task2, comparison_rows=comparison
    )

    payload = {
        "window": {"start": START, "end": END},
        "task1_new_entry_vs_topup": task1,
        "task2_cost_components": task2,
        "comparison": comparison,
    }
    (out_dir / "topup_chase_report.txt").write_text(report, encoding="utf-8")
    (out_dir / "topup_chase_results.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    # One composite PNG so all charts can be copied/downloaded at once
    chart_path = render_topup_chase_dashboard(
        payload, out_dir / "topup_chase_charts.png"
    )
    print("\n" + report)
    print(f"\nWrote -> {out_dir / 'topup_chase_report.txt'}")
    print(f"Wrote -> {chart_path}  (all charts in one image)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
