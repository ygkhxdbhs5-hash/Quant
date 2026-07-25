#!/usr/bin/env python3
"""Cost-model audit for Baseline v1 (2022-2026).

Runs:
  1) Corwin–Schultz + sqrt impact (baseline default) — Task 1/3/4 diagnostics
  2) Flat 10bps round-trip (5bps each way)
  3) Flat 30bps round-trip (15bps each way)

Does NOT change strategy entry/exit/sizing/leverage. Flat variants only override
cost_ratio via config cost_model=flat.

Usage:
  python run_cost_model_audit.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.baseline_engine import BaselineEngineV1
from engine.cost_model_audit import (
    entry_notional_over_adv_distribution,
    format_full_cost_audit_report,
    kpi_slice,
    worst_cost_fills,
)
from engine.strategy import load_config

START = "2022-01-01"
END = "2026-06-30"

VARIANTS = [
    {
        "label": "Corwin-Schultz+impact",
        "cost_model": "corwin_schultz",
        "flat_cost_one_way": None,
    },
    {
        "label": "Flat 10bps RT (5bps/side)",
        "cost_model": "flat",
        "flat_cost_one_way": 0.0005,
    },
    {
        "label": "Flat 30bps RT (15bps/side)",
        "cost_model": "flat",
        "flat_cost_one_way": 0.0015,
    },
]


def run_variant(config_path: str, variant: dict) -> tuple:
    cfg = load_config(config_path)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = variant["cost_model"]
    if variant["flat_cost_one_way"] is not None:
        cfg["flat_cost_one_way"] = variant["flat_cost_one_way"]
    else:
        cfg.pop("flat_cost_one_way", None)

    print("\n" + "#" * 72)
    print(f"# COST VARIANT: {variant['label']}")
    print(f"# WINDOW: {START} → {END}")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path=config_path)
    engine.run()
    metrics = kpi_slice(engine)
    metrics["label"] = variant["label"]
    return engine, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline v1 cost-model audit")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_COST_AUDIT",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    engine_cs = None
    for variant in VARIANTS:
        engine, metrics = run_variant(args.config, variant)
        comparison_rows.append(metrics)
        if variant["cost_model"] == "corwin_schultz":
            engine_cs = engine

    assert engine_cs is not None
    worst = worst_cost_fills(engine_cs, n=20)
    sizing = entry_notional_over_adv_distribution(engine_cs)

    report = format_full_cost_audit_report(
        worst=worst,
        comparison_rows=comparison_rows,
        engine_cs=engine_cs,
        sizing_dist=sizing,
    )

    report_path = out_dir / "cost_audit_report.txt"
    report_path.write_text(report, encoding="utf-8")
    (out_dir / "cost_audit_results.json").write_text(
        json.dumps(
            {
                "window": {"start": START, "end": END},
                "comparison": comparison_rows,
                "worst_cost_fills": worst.to_dict(orient="records") if worst is not None else [],
                "entry_notional_over_adv": sizing,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if worst is not None and not worst.empty:
        worst.to_csv(out_dir / "worst_cost_fills.csv", index=False)

    print("\n")
    print(report)
    print(f"\nWrote report -> {report_path}")
    print(f"Wrote JSON   -> {out_dir / 'cost_audit_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
