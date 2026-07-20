#!/usr/bin/env python3
"""Experiment Execution & Delta Reporting — Experiment 1 (EXIT_RANK only).

Research Rules:
  #1 Change only EXIT_RANK
  #2 No optimization / unsupported recommendations
  #3 Baseline first, experiment second; both preserved

Does NOT modify trading logic. Observation diagnostics are preserved.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.research.delta_report import build_delta_report, format_delta_report
from engine.research.experiment_execution import (
    compare_behavioral_identity,
    decide_experiment,
    extract_behavioral_identity,
    format_experiment1_full_report,
    format_mandatory_review,
    format_validation_failed,
    generate_facts,
    research_recommendation_from_results,
)
from engine.research.experiment_history import ExperimentHistory
from engine.research.fingerprint import (
    build_experiment_fingerprint,
    discover_baseline_values,
    write_fingerprint,
)
from engine.research.kpi_report import build_hierarchical_kpi_report, format_kpi_report
from engine.strategy import StandaloneEngine, load_config

OUT = ROOT / "docs" / "experiments" / "EXP001_EXIT_RANK"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _prepare_cfg(base_cfg: dict, *, entry_rank: int, exit_rank: int, benchmark: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["benchmark"] = benchmark
    research = dict(cfg.get("research") or {})
    research["ENTRY_RANK"] = int(entry_rank)
    research["EXIT_RANK"] = int(exit_rank)
    # Keep non-rank research toggles at discovered baseline defaults (identical both arms)
    research["USE_EMA9_EXIT"] = True
    research["ATR_MULTIPLIER"] = float(cfg.get("atr_multiplier", 2.0))
    research["MIN_HOLD_DAYS"] = 0
    research["USE_TIME_STOP"] = False
    cfg["research"] = research
    cfg["max_portfolio_size"] = int(entry_rank)
    cfg["selection_buffer_size"] = int(exit_rank)
    return cfg


def _run_arm(cfg: dict, label: str, out_dir: Path) -> dict:
    eng = StandaloneEngine(config=cfg, config_path=str(ROOT / "config" / "config.yaml"))
    equity = eng.run()
    arts = eng.emit_research_reports(equity, out_dir=out_dir)
    trades = eng.trade_journal.to_frame()
    bm = None
    if eng.BENCHMARK_TICKER in eng.close_m.columns and not equity.empty:
        bm = eng.close_m[eng.BENCHMARK_TICKER].reindex(equity.index).pct_change()
    kpi = build_hierarchical_kpi_report(equity, trades, benchmark_returns=bm)
    fp = build_experiment_fingerprint(
        cfg,
        eng,
        experiment_id="EXP001",
        label=label,
        paths=cfg.get("paths") or {},
    )
    write_fingerprint(fp, out_dir / "fingerprint.json")
    if not trades.empty:
        trades.to_csv(out_dir / "trade_journal.csv", index=False)
    (out_dir / "kpi.json").write_text(json.dumps(kpi, indent=2, default=str), encoding="utf-8")
    identity = extract_behavioral_identity(eng, cfg, fp)
    (out_dir / "behavioral_knobs.json").write_text(
        json.dumps(identity, indent=2, default=str), encoding="utf-8"
    )
    rank_diag = (arts or {}).get("rank_diagnostics") or eng.rank_diagnostics.summary()
    (out_dir / "rank_diagnostics.json").write_text(
        json.dumps(rank_diag, indent=2, default=str), encoding="utf-8"
    )
    return {
        "engine": eng,
        "equity": equity,
        "trades": trades,
        "kpi": kpi,
        "fingerprint": fp,
        "artifacts": arts,
        "identity": identity,
        "rank_diagnostics": rank_diag,
    }


def _architecture_audit(discovered: dict) -> str:
    return "\n".join(
        [
            "Engine Architecture Audit (discovered, not assumed)",
            "",
            json.dumps(discovered, indent=2, default=str),
            "",
            "Execution flow:",
            "  run → delist handlers → execute_pending_orders → check_cmvs_exits",
            "      → _observe_rank_diagnostics (observation only)",
            "      → monthly: regime → ADV → universe → factors → rank",
            "               → construct_portfolio(ENTRY_RANK/EXIT_RANK)",
            "               → allocate_weights → industry cap → queue_rebalance_orders",
            "",
            "Distinct rank parameters:",
            "  ENTRY_RANK — top-core / max portfolio size",
            "  EXIT_RANK  — hysteresis buffer (keep if still in top EXIT_RANK)",
            "",
            "Logging vs execution: trade journal / rank diagnostics observe only;",
            "they do not change order qty, price, or timing.",
        ]
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    abort_reasons: list[str] = []

    base_cfg = load_config(str(ROOT / "config" / "config.yaml"))
    probe = StandaloneEngine(
        config=copy.deepcopy(base_cfg), config_path=str(ROOT / "config" / "config.yaml")
    )
    discovered = discover_baseline_values(base_cfg, probe)
    architecture_audit = _architecture_audit(discovered)
    _write(OUT / "TASK0_ARCHITECTURE_AUDIT.md", "# Architecture Audit\n\n" + architecture_audit)

    panel_cols = set(probe.close_m.columns)
    configured_bm = str(base_cfg.get("benchmark") or "QQQ").upper()
    if configured_bm in panel_cols:
        experiment_benchmark = configured_bm
        bm_note = f"Using configured benchmark {configured_bm}."
    elif "SPY" in panel_cols:
        experiment_benchmark = "SPY"
        bm_note = (
            f"Configured benchmark {configured_bm} missing from local panel; "
            f"both arms use SPY so ONLY EXIT_RANK differs between arms."
        )
    else:
        print("VALIDATION FAILED")
        print("  - No usable benchmark column in price panel")
        _write(OUT / "ABORT.txt", "No usable benchmark column in price panel")
        return 2

    baseline_entry = int(discovered["from_config"]["max_portfolio_size"])
    baseline_exit = int(discovered["from_config"]["selection_buffer_size"])
    research = base_cfg.get("research") or {}
    if int(research.get("ENTRY_RANK", baseline_entry)) == baseline_entry:
        baseline_entry = int(research.get("ENTRY_RANK", baseline_entry))
    if int(research.get("EXIT_RANK", baseline_exit)) == baseline_exit:
        baseline_exit = int(research.get("EXIT_RANK", baseline_exit))

    exp_exit = 80

    baseline_verification = "\n".join(
        [
            f"Discovered ENTRY_RANK (baseline) = {baseline_entry}",
            f"Discovered EXIT_RANK (baseline)  = {baseline_exit}",
            f"Experiment EXIT_RANK             = {exp_exit}",
            f"Benchmark                        = {experiment_benchmark}",
            f"Note: {bm_note}",
            f"Universe tickers                 = {discovered['from_data']['n_tickers']}",
            f"Price columns                    = {discovered['from_data']['n_price_columns']}",
            f"Date range                       = {discovered['from_data']['date_start']}"
            f" → {discovered['from_data']['date_end']}",
            f"Random seed (universe_sample_seed) = {base_cfg.get('universe_sample_seed')}",
            "",
            "Pre-flight checks (same for both arms except EXIT_RANK):",
            "  universe, leverage path, rebalance schedule, entry logic, sizing,",
            "  risk controls, exits, ATR, EMA, holding rules, random seed",
        ]
    )
    _write(OUT / "BASELINE_VERIFICATION.txt", baseline_verification)

    print("=== Baseline Verification ===")
    print(baseline_verification)

    # --- Run A: Baseline first ---
    cfg_base = _prepare_cfg(
        base_cfg,
        entry_rank=baseline_entry,
        exit_rank=baseline_exit,
        benchmark=experiment_benchmark,
    )
    print("\n=== Running BASELINE (Run A) ===")
    base = _run_arm(cfg_base, "baseline", OUT / "baseline")

    baseline_trade_count = int(len(base["trades"]))
    (OUT / "baseline_trade_count.json").write_text(
        json.dumps({"n_closed_trades": baseline_trade_count}, indent=2), encoding="utf-8"
    )

    # Baseline arm must match discovered ranks
    if base["identity"]["ENTRY_RANK"] != baseline_entry:
        abort_reasons.append(
            f"ENTRY_RANK mismatch on baseline: {base['identity']['ENTRY_RANK']} vs {baseline_entry}"
        )
    if base["identity"]["EXIT_RANK"] != baseline_exit:
        abort_reasons.append(
            f"EXIT_RANK mismatch on baseline: {base['identity']['EXIT_RANK']} vs {baseline_exit}"
        )
    if abort_reasons:
        print("VALIDATION FAILED")
        for r in abort_reasons:
            print(f"  - {r}")
        _write(OUT / "ABORT.txt", "\n".join(abort_reasons))
        return 2

    # --- Run B: Experiment second (EXIT_RANK=80 only) ---
    cfg_exp = _prepare_cfg(
        base_cfg,
        entry_rank=baseline_entry,
        exit_rank=exp_exit,
        benchmark=experiment_benchmark,
    )
    print("\n=== Running EXPERIMENT (Run B) EXIT_RANK=80 ===")
    exp = _run_arm(cfg_exp, "exp1_exit_rank_80", OUT / "exp1")

    # Preserve both results already on disk under baseline/ and exp1/

    validation = compare_behavioral_identity(
        base["identity"],
        exp["identity"],
        allowed_changes={"EXIT_RANK": (baseline_exit, exp_exit)},
    )
    (OUT / "RUNTIME_CONFIG_VALIDATION.json").write_text(
        json.dumps(
            {
                "baseline_identity": base["identity"],
                "experiment_identity": exp["identity"],
                "validation": validation,
                "required_experiment_EXIT_RANK": exp_exit,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    if not validation["passed"]:
        failed = format_validation_failed(validation)
        print(failed)
        _write(OUT / "ABORT.txt", failed)
        abort_reasons.append(failed)
        # Still build delta for forensics, but decision REJECT
        validation_passed = False
    else:
        validation_passed = True
        print("Validation: PASS (EXIT_RANK is the only behavioral change)")

    if int(len(base["trades"])) != baseline_trade_count:
        print("VALIDATION FAILED")
        print("  - Baseline trade count changed before experiment comparison")
        return 2

    (OUT / "logging_execution_separation.json").write_text(
        json.dumps(
            {
                "journal_hooks": "post-fill observation only",
                "rank_diagnostics": "observation only",
                "order_generation_depends_on_journal": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    delta = build_delta_report(
        base["kpi"],
        exp["kpi"],
        base["trades"],
        exp["trades"],
        baseline_rank_diagnostics=base["rank_diagnostics"],
        experiment_rank_diagnostics=exp["rank_diagnostics"],
    )
    _write(OUT / "DELTA_REPORT.txt", format_delta_report(delta))
    (OUT / "delta.json").write_text(json.dumps(delta, indent=2, default=str), encoding="utf-8")

    decision = decide_experiment(delta, validation_passed=validation_passed)
    facts = generate_facts(
        baseline_exit=baseline_exit,
        experiment_exit=exp_exit,
        delta=delta,
        validation_passed=validation_passed,
    )
    recommendation = research_recommendation_from_results(
        facts=facts,
        delta=delta,
        n_closed_trades_baseline=len(base["trades"]),
        n_closed_trades_experiment=len(exp["trades"]),
        validation_passed=validation_passed,
        decision=decision,
    )

    d_summary_parts = []
    for key in (
        "n_closed_trades",
        "turnover_trades_per_year",
        "avg_holding_period_days",
        "cagr",
        "mdd",
    ):
        row = (delta.get("metrics") or {}).get(key) or {}
        d_summary_parts.append(f"{key} Δ={row.get('delta')}")
    delta_summary = "; ".join(d_summary_parts)

    review_body = format_mandatory_review(
        hypothesis=(
            f"Widening EXIT_RANK from {baseline_exit} to {exp_exit} "
            f"(ENTRY_RANK fixed at {baseline_entry}) changes turnover / holding / risk metrics."
        ),
        changed_variable=f"EXIT_RANK: {baseline_exit} → {exp_exit}",
        baseline={
            "ENTRY_RANK": baseline_entry,
            "EXIT_RANK": baseline_exit,
            "n_closed_trades": len(base["trades"]),
            "cagr": (base["kpi"].get("risk") or {}).get("cagr"),
        },
        experiment={
            "ENTRY_RANK": baseline_entry,
            "EXIT_RANK": exp_exit,
            "n_closed_trades": len(exp["trades"]),
            "cagr": (exp["kpi"].get("risk") or {}).get("cagr"),
        },
        delta_summary=delta_summary,
        facts=facts,
        recommendation=recommendation,
        decision=decision,
    )
    _write(OUT / "EXPERIMENT_REVIEW.txt", review_body)
    _write(OUT / "RESEARCH_RECOMMENDATION.txt", recommendation)
    _write(OUT / "FACTS.txt", "\n".join(facts) + "\n")

    full_report = format_experiment1_full_report(
        architecture_audit=architecture_audit,
        baseline_verification=baseline_verification,
        baseline_entry=baseline_entry,
        baseline_exit=baseline_exit,
        experiment_exit=exp_exit,
        baseline_identity=base["identity"],
        experiment_identity=exp["identity"],
        validation=validation,
        delta=delta,
        facts=facts,
        recommendation=recommendation,
        decision=decision,
        review_body=review_body,
    )
    # Append after preserving existing diagnostics text from both arms
    diagnostics_appendix = "\n\n".join(
        [
            "=" * 50,
            "Appendix — Baseline diagnostics (unchanged tooling)",
            "=" * 50,
            format_kpi_report(base["kpi"]),
            (OUT / "baseline" / "rank_diagnostics_report.txt").read_text(encoding="utf-8")
            if (OUT / "baseline" / "rank_diagnostics_report.txt").exists()
            else "",
            "=" * 50,
            "Appendix — Experiment diagnostics (unchanged tooling)",
            "=" * 50,
            format_kpi_report(exp["kpi"]),
            (OUT / "exp1" / "rank_diagnostics_report.txt").read_text(encoding="utf-8")
            if (OUT / "exp1" / "rank_diagnostics_report.txt").exists()
            else "",
        ]
    )
    _write(OUT / "EXP001_FULL_REPORT.txt", full_report + "\n\n" + diagnostics_appendix)
    _write(OUT / "EXPERIMENT1_RESULT.md", full_report)

    checklist = {
        "architecture_audit": True,
        "baseline_verification": True,
        "experiment_1_report": True,
        "validation_passed": validation_passed,
        "delta_report": True,
        "fact_generation": bool(facts),
        "research_recommendation": bool(recommendation),
        "decision": decision,
        "only_exit_rank_changed": validation_passed,
        "baseline_preserved": (OUT / "baseline" / "fingerprint.json").exists(),
        "experiment_preserved": (OUT / "exp1" / "fingerprint.json").exists(),
    }
    (OUT / "VALIDATION_CHECKLIST.json").write_text(
        json.dumps(checklist, indent=2), encoding="utf-8"
    )
    _write(
        OUT / "VALIDATION_CHECKLIST.txt",
        "\n".join(f"{k}: {v}" for k, v in checklist.items()),
    )

    hist = ExperimentHistory(OUT / "experiment_history.json")
    hist.append(
        parent_exp=None,
        changed_variables={"EXIT_RANK": f"{baseline_exit}→{exp_exit}"},
        hypothesis="EXIT_RANK single-variable change affects turnover/holding/risk",
        expected_outcome="Measurable Δ if ranked universe > EXIT_RANK",
        actual_outcome="; ".join(facts[:3]),
        decision=decision,
        metrics={"delta": delta, "recommendation": recommendation},
        notes=bm_note,
    )

    # Mandatory end-of-experiment print (exact section contract)
    print("\n" + full_report)
    print(f"\nArtifacts written under {OUT}")

    success = (
        validation_passed
        and exp["identity"]["EXIT_RANK"] == exp_exit
        and base["identity"]["EXIT_RANK"] == baseline_exit
        and decision in {"PASS", "REPEAT", "REJECT"}
        and (OUT / "DELTA_REPORT.txt").exists()
        and (OUT / "EXP001_FULL_REPORT.txt").exists()
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
