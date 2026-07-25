#!/usr/bin/env python3
"""Clean-tree no-chase baseline reference (single in-memory payload + fingerprint).

Runs chase-on (legacy flag) and no-chase (new default) under Fixed CS v2 for
2022-01-01 → 2026-06-30, then emits text+JSON from one payload.

Intended to be executed on a fully clean git working tree so the fingerprint
records git_dirty=False.

Usage:
  python3 run_nochase_clean_reference.py --config config/config.yaml
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

# Unreproducible dirty-tree claims (documented; not used as pass/fail gates).
OLD_DIRTY_NOCHASE = {
    "label": "old dirty-tree claim (UNREPRODUCIBLE)",
    "CAGR": 0.1055,
    "Sharpe": 0.602,
    "Maximum_Drawdown": -0.2120,
    "total_cost_dollars": 10_248_678.0,
    "n_closed_trades": 1052,
    "repro_id": "5146837eafd9",
}
OLD_DIRTY_CHASE = {
    "label": "old dirty-tree chase claim (UNREPRODUCIBLE)",
    "CAGR": -0.0024,
    "Sharpe": 0.115,
    "Maximum_Drawdown": -0.3402,
    "total_cost_dollars": 15_420_000.0,
    "n_closed_trades": 929,
}


def _git_dirty() -> bool:
    out = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(Path(__file__).resolve().parent), text=True
    )
    return bool(out.strip())


def _run(cfg_base: dict, *, enable_topup_chasing: bool, label: str) -> dict:
    cfg = dict(cfg_base)
    cfg["start_date"] = START
    cfg["end_date"] = END
    cfg["cost_model"] = "corwin_schultz_v2"
    cfg["winsorize_adv"] = True
    cfg["enable_topup_chasing"] = bool(enable_topup_chasing)
    cfg.pop("disable_topup_chasing", None)
    print("\n" + "#" * 72)
    print(f"# {label}")
    print("#" * 72)
    engine = BaselineEngineV1(config=cfg, config_path="config/config.yaml")
    engine.run()
    m = kpi_with_trades(engine)
    m["label"] = label
    m["enable_topup_chasing"] = bool(enable_topup_chasing)
    return m


def _fmt_row(label: str, r: dict) -> str:
    return (
        f"{label:<42} {100*float(r['CAGR']):>+8.2f}% "
        f"{float(r['Sharpe']):>8.3f} "
        f"{100*float(r['Maximum_Drawdown']):>8.2f}% "
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
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit run on a dirty tree (NOT for checkpoint tagging).",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirty = _git_dirty()
    if dirty and not args.allow_dirty:
        print("ERROR: working tree is dirty. Commit first, or pass --allow-dirty.")
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
            "report": "nochase_clean_reference",
            "variants": ["chase_on_legacy", "nochase_default"],
        },
    )
    print(f"[REPRO] {format_fingerprint_banner(repro)}")
    print(f"  git_dirty={repro.get('git_dirty')} (must be False for checkpoint)")

    chase = _run(base_cfg, enable_topup_chasing=True, label="Fixed CS v2 + chase ON (legacy)")
    nochase = _run(
        base_cfg, enable_topup_chasing=False, label="Fixed CS v2 + NO-CHASE (NEW DEFAULT)"
    )

    payload = {
        "repro": repro,
        "comparison": [chase, nochase],
        "old_dirty_claims": {
            "nochase": OLD_DIRTY_NOCHASE,
            "chase": OLD_DIRTY_CHASE,
        },
        "spec_notes": {
            "skip_buy_topups_when_held": True,
            "skip_size_trim_sell_when_still_selected": True,
            "allow_sell_on_momentum_dropout": True,
            "atr_exits_unaffected": True,
            "cost_model": "corwin_schultz_v2",
            "winsorize_adv": True,
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    payload_sha16 = hashlib.sha256(blob).hexdigest()[:16]
    payload["payload_sha256_16"] = payload_sha16

    better_than_chase = float(nochase["CAGR"]) > float(chase["CAGR"]) + 0.01
    # Directional vs old dirty claim: both positive and clearly > chase in this env.
    tag_ok = (not dirty) and better_than_chase and float(nochase["CAGR"]) > 0.0

    lines = [
        "# Baseline v1 NO-CHASE clean reference",
        "",
        f"Window: {START} → {END}",
        f"repro_id={repro.get('fingerprint_id')}",
        f"payload_sha256_16={payload_sha16}",
        f"git_head={repro.get('git_head')}",
        f"git_dirty={repro.get('git_dirty')}",
        f"panels={(repro.get('data_cache') or {}).get('panels.pkl')}",
        "",
        "## Spec (fill/order only)",
        "- Skip BUY top-ups for already-held symbols (no catch-up to equal-weight).",
        "- Skip size-trim SELLs while still in month selection.",
        "- Allow SELL on momentum drop-out; ATR trail exits unchanged.",
        "- Cost model Fixed CS v2 + ADV winsorize unchanged.",
        "",
        "## Three-way comparison",
        f"{'Variant':<42} {'CAGR':>9} {'Sharpe':>8} {'MDD':>9} {'Cost $':>14} {'n_trades':>10}",
        "-" * 100,
        _fmt_row("OLD dirty no-chase claim (UNREPRO)", OLD_DIRTY_NOCHASE),
        _fmt_row("OLD dirty chase claim (UNREPRO)", OLD_DIRTY_CHASE),
        _fmt_row(chase["label"], chase),
        _fmt_row(nochase["label"], nochase),
        "-" * 100,
        "",
        f"no-chase better than chase-on (this env): {better_than_chase}",
        f"tag_ok (clean tree + no-chase >0 and > chase+1pp): {tag_ok}",
        "",
        "# END",
    ]

    txt_path = out_dir / "nochase_clean_reference.txt"
    json_path = out_dir / "nochase_clean_reference.json"
    txt_path.write_text("\n".join(lines) + "\n")
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("\n".join(lines))
    print(f"\nWrote -> {txt_path}")
    print(f"Wrote -> {json_path}")
    return 0 if tag_ok else 2


if __name__ == "__main__":
    sys.exit(main())
