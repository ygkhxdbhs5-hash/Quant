#!/usr/bin/env python3
"""Audit predictive power of every CMVS factor (observation only).

Runs the CMVS Factor Audit using the exact BBS/VZS/CPS/RSIS/RSS formulas
from the engine, then prints:
  - IC (Pearson), Rank IC, Spearman correlation
  - future 1M/3M/6M returns
  - monotonicity by decile and hit rate
  - marginal contribution to composite score (leave-one-out)
  - observation-only recommendations (no trading changes)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.strategy import StandaloneEngine, load_config
from engine.research.cmvs_factor_audit import CMVS_FACTORS, CMVSAuditor


def _format_decile_list(x):
    return [None if v is None else float(v) for v in x]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CMVS factor predictive power")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out-dir", default="cache")
    parser.add_argument("--max-rebalance-dates", type=int, default=None)
    parser.add_argument("--min-universe-size", type=int, default=5)
    parser.add_argument("--h1", type=int, default=21)
    parser.add_argument("--h3", type=int, default=63)
    parser.add_argument("--h6", type=int, default=126)
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = StandaloneEngine(config=cfg, config_path=args.config)

    # Local panels may not contain the configured benchmark (e.g., QQQ).
    # CMVS factor building and regime logic expect BENCHMARK_TICKER to exist.
    if engine.BENCHMARK_TICKER not in engine.close_m.columns:
        if "SPY" in engine.close_m.columns:
            engine.BENCHMARK_TICKER = "SPY"
        else:
            raise RuntimeError(
                f"Benchmark ticker {engine.BENCHMARK_TICKER!r} missing from panel and SPY is not available."
            )

    auditor = CMVSAuditor(
        engine,
        horizons_days={"1M": args.h1, "3M": args.h3, "6M": args.h6},
        max_rebalance_dates=args.max_rebalance_dates,
        min_universe_size=args.min_universe_size,
    )

    res = auditor.audit()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "cmvs_factor_audit_report.txt"
    json_path = out_dir / "cmvs_factor_audit.json"

    d = res.to_dict()
    json_path.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")

    lines = []
    lines.append("=" * 72)
    lines.append("CMVS FACTOR AUDIT (observation only — no trading changes)")
    lines.append("=" * 72)
    lines.append(f"Horizons (days): {d['horizons']}")
    lines.append("")

    for horizon in d["horizons"]:
        lines.append("-" * 72)
        lines.append(f"Horizon: {horizon} ({d['horizons'][horizon]} days)")
        lines.append("-" * 72)

        # Rank by Pearson IC mean
        rank_list = d["recommendations"]["rank_ic_by_horizon_pearson"].get(horizon) or []
        lines.append("Rank IC (Pearson) — higher is better:")
        for i, (f, m) in enumerate(rank_list, start=1):
            lines.append(f"  {i}. {f}: IC_mean={('n/a' if m is None else f'{m:.6f}')}")
        lines.append("")

        for f in CMVS_FACTORS:
            fm = d["factor_metrics"][f][horizon]
            icp = fm["ic_pearson"]["mean"]
            sp = fm["spearman"]["mean"]
            lines.append(f"Factor: {f}")
            lines.append(
                f"  IC (Pearson) mean={('n/a' if icp is None else f'{icp:.6f}')} "
                f"(n_dates={fm['ic_pearson']['n']})"
            )
            lines.append(
                f"  Spearman mean={('n/a' if sp is None else f'{sp:.6f}')} "
                f"(n_dates={fm['spearman']['n']})"
            )
            mon = fm["monotonicity_by_decile"]
            mon_score = mon["monotonicity_score_spearman(decile,mean_return)"]
            mon_score_str = "n/a" if mon_score is None else f"{float(mon_score):.6f}"
            lines.append(
                f"  Monotonicity score (Spearman(decile, mean_return))={mon_score_str}"
            )

            lines.append("  Decile mean future return:")
            lines.append("    " + ", ".join("None" if v is None else f"{float(v):.4f}" for v in mon["decile_mean_future_return"]))
            lines.append("  Decile hit rate (future return > 0):")
            lines.append("    " + ", ".join("None" if v is None else f"{float(v):.2%}" for v in mon["decile_hit_rate"]))
            lines.append("")

    lines.append("=" * 72)
    lines.append("MARGINAL CONTRIBUTION TO COMPOSITE SCORE (leave-one-out)")
    lines.append("=" * 72)

    for horizon in d["horizons"]:
        lines.append("-" * 72)
        lines.append(f"Horizon: {horizon}")
        lines.append("-" * 72)
        full = d["marginal_contribution"][horizon]["full_composite_ic"]
        lines.append(
            f"Full composite IC: Pearson_mean={full['ic_pearson_mean']} Spearman_mean={full['spearman_mean']} "
            f"(n_dates={full['n_dates']})"
        )
        for f in CMVS_FACTORS:
            m = d["marginal_contribution"][horizon][f]["marginal_contribution_to_composite"]
            lines.append(
                f"  Remove {f}: Pearson_loss={m['ic_pearson_loss_when_removed']} "
                f"Spearman_loss={m['spearman_loss_when_removed']}"
            )
        lines.append("")

    lines.append("=" * 72)
    lines.append("RECOMMENDATIONS (derived only from observed audit metrics)")
    lines.append("=" * 72)
    rec = d["recommendations"]
    lines.append("Factors to remove (IC Pearson mean <= 0 across all horizons):")
    lines.append("  " + (", ".join(rec["remove_factors"]) if rec["remove_factors"] else "(none)"))
    lines.append("Factors to reweight (has any positive Pearson IC mean):")
    lines.append("  " + (", ".join(rec["reweight_factors_to_proportional_to_positive_ic"]) if rec["reweight_factors_to_proportional_to_positive_ic"] else "(none)"))
    lines.append("Suggested relative weights from positive IC (normalized to sum 1):")
    lines.append("  " + ", ".join(f"{k}={v:.3f}" for k, v in rec["suggested_relative_weights_from_positive_ic"].items()))
    lines.append("Factors to add:")
    lines.append("  (none — no additional audited factors beyond CMVS set)")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - Observation only. This script does not change trading behavior.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-60:]))
    print(f"\nWrote report -> {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

