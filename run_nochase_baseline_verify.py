#!/usr/bin/env python3
"""Verify new default (no-chase) Baseline v1 against the confirmed reference KPIs.

Runs 2022-01-01 → 2026-06-30 with config defaults (ENABLE_TOPUP_CHASING=False)
and compares CAGR / cost / n_trades to the Streamlit-confirmed fingerprint.

Usage:
  python3 run_nochase_baseline_verify.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from engine.baseline_engine import BaselineEngineV1
from engine.repro_fingerprint import build_repro_fingerprint, format_fingerprint_banner
from engine.strategy import load_config
from engine.topup_chase_diagnostics import kpi_with_trades

START = "2022-01-01"
END = "2026-06-30"

# Confirmed reference from Streamlit fingerprint (user-provided).
REF = {
    "repro_id": "5146837eafd9",
    "payload_sha256_16": "75f6331be63401bb",
    "CAGR": 0.1055,
    "Sharpe": 0.602,
    "Maximum_Drawdown": -0.2120,
    "total_cost_dollars": 10_248_678.0,
    "n_closed_trades": 1052,
}


def _close(a: float, b: float, *, rel: float = 1e-3, abs_tol: float = 1.0) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= max(abs_tol, rel * max(abs(float(b)), 1e-12))


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

    cfg = load_config(args.config)
    cfg["start_date"] = START
    cfg["end_date"] = END
    # Force cost-layer knobs; do NOT set chase flags beyond config default.
    cfg["cost_model"] = "corwin_schultz_v2"
    cfg["winsorize_adv"] = True
    # Ensure we exercise the new default path if config omits the key.
    cfg.pop("disable_topup_chasing", None)
    if "enable_topup_chasing" not in cfg:
        cfg["enable_topup_chasing"] = False

    repro = build_repro_fingerprint(
        start=START,
        end=END,
        config=cfg,
        extra={"report": "nochase_baseline_verify", "enable_topup_chasing": False},
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")

    print("\n=== NEW DEFAULT: Fixed CS v2 + ADV winsorize + NO-CHASE ===")
    engine = BaselineEngineV1(config=cfg, config_path=args.config)
    engine.run()
    metrics = kpi_with_trades(engine)
    metrics["label"] = "default_nochase"

    payload = {
        "repro": repro,
        "metrics": metrics,
        "reference": REF,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha16 = hashlib.sha256(blob).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha16

    cagr = float(metrics.get("CAGR") or 0.0)
    sharpe = float(metrics.get("Sharpe") or 0.0)
    mdd = float(metrics.get("Maximum_Drawdown") or 0.0)
    cost = float(metrics.get("total_cost_dollars") or 0.0)
    n_trades = int(metrics.get("n_closed_trades") or 0)

    print(
        f"  CAGR={cagr:+.2%} Sharpe={sharpe:.3f} MDD={mdd:.2%} "
        f"cost=${cost:,.0f} n_trades={n_trades}"
    )
    print(
        f"  enable_topup_chasing={metrics.get('enable_topup_chasing')} "
        f"disable_topup_chasing={metrics.get('disable_topup_chasing')}"
    )
    print(f"  repro_id={repro.get('fingerprint_id')}  payload_sha256_16={payload_sha16}")

    checks = {
        "CAGR": _close(cagr, REF["CAGR"], rel=5e-3, abs_tol=5e-4),
        "Sharpe": _close(sharpe, REF["Sharpe"], rel=5e-3, abs_tol=5e-3),
        "Maximum_Drawdown": _close(mdd, REF["Maximum_Drawdown"], rel=5e-3, abs_tol=5e-4),
        "total_cost_dollars": _close(cost, REF["total_cost_dollars"], rel=1e-3, abs_tol=50.0),
        "n_closed_trades": n_trades == int(REF["n_closed_trades"]),
    }
    match = all(checks.values())

    report_lines = [
        "# Baseline v1 no-chase default — verification",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha16}",
        f"git_head={repro.get('git_head')} dirty={repro.get('git_dirty')}",
        "",
        "## Observed (new default)",
        f"CAGR={cagr:+.4%}  Sharpe={sharpe:.3f}  MDD={mdd:.2%}",
        f"cost=${cost:,.2f}  n_trades={n_trades}",
        f"enable_topup_chasing={metrics.get('enable_topup_chasing')}",
        "",
        "## Reference (user fingerprint 5146837eafd9)",
        f"CAGR={REF['CAGR']:+.2%}  Sharpe={REF['Sharpe']:.3f}  MDD={REF['Maximum_Drawdown']:.2%}",
        f"cost=${REF['total_cost_dollars']:,.0f}  n_trades={REF['n_closed_trades']}",
        f"payload_sha256_16={REF['payload_sha256_16']}",
        "",
        "## Match checks",
    ]
    for k, ok in checks.items():
        report_lines.append(f"- {k}: {'PASS' if ok else 'FAIL'}")
    report_lines += [
        "",
        f"OVERALL: {'PASS' if match else 'FAIL — STOP (do not promote checkpoint claiming reference KPIs)'}",
        "",
    ]

    json_path = out_dir / "nochase_default_verify.json"
    txt_path = out_dir / "nochase_default_verify.txt"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    txt_path.write_text("\n".join(report_lines) + "\n")
    print(f"Wrote -> {txt_path}")
    print(f"Wrote -> {json_path}")

    if not match:
        print("\n*** DISCREPANCY: observed KPIs do not match confirmed reference. ***")
        print("*** Stopping before Task 3 checkpoint that would claim reference numbers. ***")
        return 2
    print("\nVerification PASS — safe to create baseline-v1-nochase checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
