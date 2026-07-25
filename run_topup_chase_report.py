#!/usr/bin/env python3
"""Top-up chasing diagnostics + no-chase fill variant comparison (2022-2026).

ONE run produces text report + JSON + chart from the SAME in-memory payload
(with a shared repro fingerprint). Do not regenerate the chart from a stale
JSON without also regenerating the text report from that same payload.

Usage:
  python run_topup_chase_report.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from engine.baseline_engine import BaselineEngineV1
from engine.report_charts import render_topup_chase_dashboard
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
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
        "label": "Fixed CS v2 (chase ON, legacy)",
        "cost_model": "corwin_schultz_v2",
        "winsorize_adv": True,
        "enable_topup_chasing": True,
        "flat_cost_one_way": None,
    },
    {
        "label": "Fixed CS v2 + NO-CHASE (default)",
        "cost_model": "corwin_schultz_v2",
        "winsorize_adv": True,
        "enable_topup_chasing": False,
        "flat_cost_one_way": None,
    },
    {
        "label": "Flat 10bps RT (5bps/side)",
        "cost_model": "flat",
        "winsorize_adv": False,
        "enable_topup_chasing": True,
        "flat_cost_one_way": 0.0005,
    },
    {
        "label": "Flat 30bps RT (15bps/side)",
        "cost_model": "flat",
        "winsorize_adv": False,
        "enable_topup_chasing": True,
        "flat_cost_one_way": 0.0015,
    },
]


def run_variant(config_path: str, variant: dict):
    cfg = load_config(config_path)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = variant["cost_model"]
    cfg["winsorize_adv"] = bool(variant["winsorize_adv"])
    # Explicit enable flag (overrides engine default) so chase ON stays reproducible.
    cfg["enable_topup_chasing"] = bool(variant["enable_topup_chasing"])
    cfg.pop("disable_topup_chasing", None)
    if variant["flat_cost_one_way"] is not None:
        cfg["flat_cost_one_way"] = variant["flat_cost_one_way"]
    print("\n" + "#" * 72)
    print(f"# {variant['label']}")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path=config_path)
    engine.run()
    metrics = kpi_with_trades(engine)
    metrics["label"] = variant["label"]
    metrics["enable_topup_chasing"] = bool(variant["enable_topup_chasing"])
    return engine, metrics


def _payload_sha16(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out-dir", default="docs/experiments/BASELINE_V1_TOPUP_CHASE")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config)
    repro = build_repro_fingerprint(
        start=START,
        end=END,
        config=base_cfg,
        extra={"report": "topup_chase", "variants": [v["label"] for v in VARIANTS]},
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")

    comparison = []
    engine_chase = None
    for variant in VARIANTS:
        engine, metrics = run_variant(args.config, variant)
        comparison.append(metrics)
        if (
            variant["cost_model"] == "corwin_schultz_v2"
            and variant["enable_topup_chasing"]
        ):
            engine_chase = engine

    assert engine_chase is not None
    task1 = analyze_new_entry_vs_topup(engine_chase)
    task2 = analyze_cost_components(engine_chase)

    # Single in-memory payload → text + JSON + chart (guaranteed consistent)
    payload = {
        "window": {"start": START, "end": END},
        "repro": repro,
        "fingerprint_id": repro["fingerprint_id"],
        "task1_new_entry_vs_topup": task1,
        "task2_cost_components": task2,
        "comparison": comparison,
    }
    payload["payload_sha256_16"] = _payload_sha16(
        {k: v for k, v in payload.items() if k != "payload_sha256_16"}
    )

    report = format_chase_report(
        task1=task1, task2=task2, comparison_rows=comparison, repro=repro
    )
    # Append machine-checkable consistency line
    report += (
        f"\n### ARTIFACT CONSISTENCY\n"
        f"  payload_sha256_16     : {payload['payload_sha256_16']}\n"
        f"  chart+text+json share : same in-memory payload (this run)\n"
        f"  claimed_correct_nums  : n_buy={task1.get('n_buy_orders')}, "
        f"no_chase_CAGR={comparison[1].get('CAGR')}, "
        f"no_chase_cost$={comparison[1].get('total_cost_dollars')}\n"
    )

    fp_id = repro["fingerprint_id"]
    report_path = out_dir / "topup_chase_report.txt"
    json_path = out_dir / "topup_chase_results.json"
    chart_path = out_dir / "topup_chase_charts.png"
    chart_fp_path = out_dir / f"topup_chase_charts_{fp_id}.png"

    report_path.write_text(report, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    render_topup_chase_dashboard(payload, chart_path)
    render_topup_chase_dashboard(payload, chart_fp_path)

    # Write reconciliation note for humans
    recon = out_dir / "RECONCILIATION.md"
    recon.write_text(
        "\n".join(
            [
                "# Top-up chase artifact reconciliation",
                "",
                f"- **Authoritative numbers**: this directory's "
                f"`topup_chase_report.txt` / `topup_chase_results.json` / "
                f"`topup_chase_charts.png` from a **single** "
                f"`run_topup_chase_report.py` invocation.",
                f"- **repro_id**: `{fp_id}`",
                f"- **payload_sha256_16**: `{payload['payload_sha256_16']}`",
                f"- **git**: `{repro.get('git_head')}`"
                f"{' (dirty)' if repro.get('git_dirty') else ''}",
                f"- **window**: {START} → {END}",
                "",
                "## Investigation note (prior mismatch report)",
                "",
                "A claimed chart with BUY counts 738+3843 (~4,581), no-chase CAGR "
                "~10.6%, flat-10bps CAGR ~13% was **not found** in the repo "
                "artifacts. On-disk text/JSON/PNG already agreed on "
                "n_buy=6,830 / no-chase CAGR≈1.89% / flat-10bps≈1.12% before "
                "this regenerate. Likely causes of the claimed mismatch:",
                "1. Stale Streamlit-cached image from an older local render, or",
                "2. Dual-axis misread (green=CAGR%, brown=Cost $M), or",
                "3. Comparing chart from a different experiment/window.",
                "",
                "Going forward: refuse to compare text vs chart unless "
                "`repro_id` in the text banner matches the chart footer.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("\n" + report)
    print(f"\n[REPRO] {format_fingerprint_banner(repro)}")
    print(f"Wrote -> {report_path}")
    print(f"Wrote -> {json_path}")
    print(f"Wrote -> {chart_path}")
    print(f"Wrote -> {chart_fp_path}")
    print(f"Wrote -> {recon}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
