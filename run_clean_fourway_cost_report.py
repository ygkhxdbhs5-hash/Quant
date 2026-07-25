#!/usr/bin/env python3
"""Clean-tree four-way cost/fill comparison (single in-memory payload).

Variants (2022-01-01 → 2026-06-30), no entry/exit/sizing/leverage changes:
  1) Fixed CS v2 + chase ON (legacy)
  2) Fixed CS v2 + no-chase (default / baseline-v1-nochase)
  3) Flat 10bps RT (5bps/side) — chase ON (same mechanism as prior report)
  4) Flat 30bps RT (15bps/side) — chase ON

Must run on a clean git working tree.

Usage:
  python3 run_clean_fourway_cost_report.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from engine.baseline_engine import BaselineEngineV1
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
from engine.topup_chase_diagnostics import kpi_with_trades

START = "2022-01-01"
END = "2026-06-30"
ROOT = Path(__file__).resolve().parent

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

# Dirty-tree claims from unreproducible topup_chase_report_2 (DO NOT CITE).
INVALID_REPORT_2 = {
    "status": "INVALID_UNREPRODUCIBLE",
    "reason": "dirty-tree origin; code state lost; fingerprint 5146837eafd9 never recoverable",
    "claimed": {
        "chase_on_cagr": -0.0024,
        "nochase_cagr": 0.1055,
        "flat10_cagr": 0.1302,
        "flat30_cagr": 0.1034,
    },
}


def _git_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
    )
    return bool(out.strip())


def run_variant(config_path: str, variant: dict) -> dict:
    cfg = load_config(config_path)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = variant["cost_model"]
    cfg["winsorize_adv"] = bool(variant["winsorize_adv"])
    cfg["enable_topup_chasing"] = bool(variant["enable_topup_chasing"])
    cfg.pop("disable_topup_chasing", None)
    if variant["flat_cost_one_way"] is not None:
        cfg["flat_cost_one_way"] = variant["flat_cost_one_way"]
    print("\n" + "#" * 72)
    print(f"# {variant['label']}")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path=config_path)
    engine.run()
    m = kpi_with_trades(engine)
    m["label"] = variant["label"]
    m["enable_topup_chasing"] = bool(variant["enable_topup_chasing"])
    m["cost_model"] = variant["cost_model"]
    m["flat_cost_one_way"] = variant["flat_cost_one_way"]
    return m


def _fmt_row(r: dict) -> str:
    return (
        f"{r['label']:<42} "
        f"{100 * float(r['CAGR']):>+8.2f}% "
        f"{float(r['Sharpe']):>8.3f} "
        f"{100 * float(r['Maximum_Drawdown']):>8.2f}% "
        f"{float(r['total_cost_dollars']):>14,.0f} "
        f"{int(r['n_closed_trades']):>10,}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="docs/experiments/BASELINE_V1_NOCHASE_DEFAULT",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _git_dirty():
        print("ERROR: working tree dirty — commit first (clean-state requirement).")
        return 1

    base_cfg = load_config(args.config)
    repro = build_repro_fingerprint(
        start=START,
        end=END,
        config={
            **base_cfg,
            "cost_model": "corwin_schultz_v2",
            "winsorize_adv": True,
            "enable_topup_chasing": False,
        },
        extra={
            "report": "clean_fourway_cost",
            "variants": [v["label"] for v in VARIANTS],
            "invalidates": "topup_chase_report_2.txt dirty-tree claims",
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')}")

    comparison = [run_variant(args.config, v) for v in VARIANTS]

    payload = {
        "repro": repro,
        "comparison": comparison,
        "invalid_report_2": INVALID_REPORT_2,
        "notes": {
            "flat_variants_use_chase_on": True,
            "reason": "same isolated cost-model sensitivity mechanism as prior topup_chase_report",
            "baseline_tag": "baseline-v1-nochase",
            "prior_clean_nochase_repro_id": "97c21a974bf8",
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha16 = hashlib.sha256(blob).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha16

    lines = [
        "# Clean four-way cost/fill reference (VALID)",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha16}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        f"panels={(repro.get('data_cache') or {}).get('panels.pkl')}",
        "",
        "## INVALID — do not cite (dirty-tree topup_chase_report_2)",
        "  chase-on -0.24%, no-chase +10.55%, flat10 +13.02%, flat30 +10.34%",
        "  Status: INVALID / UNREPRODUCIBLE (code state lost). Superseded by this report.",
        "",
        "## Four-way comparison (clean committed state only)",
        f"{'Variant':<42} {'CAGR':>9} {'Sharpe':>8} {'MDD':>9} {'Cost $':>14} {'n_trades':>10}",
        "-" * 100,
    ]
    for r in comparison:
        lines.append(_fmt_row(r))
        lines.append(
            f"  fingerprint_note: enable_topup_chasing={r.get('enable_topup_chasing')} "
            f"cost_model={r.get('cost_model')} flat_one_way={r.get('flat_cost_one_way')}"
        )
    lines += [
        "-" * 100,
        "",
        "Flat variants: chase ON + flat cost model (cost sensitivity; not fill-policy change).",
        "CS v2 variants: Fixed CS v2 + ADV winsorize; chase ON vs no-chase default.",
        "Entry/exit/sizing/leverage unchanged.",
        "",
        "# END",
    ]

    txt = out_dir / "clean_fourway_cost_report.txt"
    js = out_dir / "clean_fourway_cost_report.json"
    txt.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("\n".join(lines))
    print(f"\nWrote -> {txt}")
    print(f"Wrote -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
