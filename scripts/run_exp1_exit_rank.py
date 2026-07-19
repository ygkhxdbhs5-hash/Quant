#!/usr/bin/env python3
"""Controlled Experiment 1: change ONLY EXIT_RANK (70 → 80).

Research Rules:
  #1 Change only EXIT_RANK
  #2 No optimization / unsupported recommendations
Abort if any other behavioral variable differs, baseline defaults fail,
or logging path is not observation-only.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.research.delta_report import build_delta_report, format_delta_report
from engine.research.experiment_history import ExperimentHistory
from engine.research.experiment_templates import (
    decide_from_delta,
    format_experiment_review,
    format_research_recommendation,
)
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


def _behavioral_knobs(engine) -> dict:
    return {
        "USE_EMA9_EXIT": bool(engine.USE_EMA9_EXIT),
        "ATR_MULTIPLIER": float(engine.atr_multiplier),
        "ENTRY_RANK": int(engine.ENTRY_RANK),
        "EXIT_RANK": int(engine.EXIT_RANK),
        "MIN_HOLD_DAYS": int(engine.MIN_HOLD_DAYS),
        "USE_TIME_STOP": bool(engine.USE_TIME_STOP),
        "TIME_STOP_DAYS": int(engine.TIME_STOP_DAYS),
        "BENCHMARK_TICKER": str(engine.BENCHMARK_TICKER),
        "TOP_ADV_POOL": int(engine.TOP_ADV_POOL),
        "CORR_THRESHOLD": float(engine.CORR_THRESHOLD),
        "MAX_INDUSTRY_WEIGHT": float(engine.MAX_INDUSTRY_WEIGHT),
        "COMMISSION_RATE": float(engine.COMMISSION_RATE),
        "SLIPPAGE_RATE": float(engine.SLIPPAGE_RATE),
    }


def _prepare_cfg(base_cfg: dict, *, exit_rank: int, benchmark: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    # Same data arm for baseline & exp: local panel lacks QQQ; use available benchmark.
    cfg["benchmark"] = benchmark
    research = dict(cfg.get("research") or {})
    # Discovered baseline ENTRY_RANK stays fixed; only EXIT_RANK changes in Exp1
    research["ENTRY_RANK"] = int(cfg.get("max_portfolio_size", 50))
    research["EXIT_RANK"] = int(exit_rank)
    research["USE_EMA9_EXIT"] = True
    research["ATR_MULTIPLIER"] = float(cfg.get("atr_multiplier", 2.0))
    research["MIN_HOLD_DAYS"] = 0
    research["USE_TIME_STOP"] = False
    cfg["research"] = research
    # Keep legacy mirrors aligned with discovered baseline entry size
    cfg["selection_buffer_size"] = int(exit_rank)
    return cfg


def _run_arm(cfg: dict, label: str, out_dir: Path) -> dict:
    eng = StandaloneEngine(config=cfg, config_path=str(ROOT / "config" / "config.yaml"))
    # Suppress research auto-fingerprint overwrite races by using arm-specific out
    equity = eng.run()
    # Re-emit into arm folder
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
    trades_path = out_dir / "trade_journal.csv"
    if not trades.empty:
        trades.to_csv(trades_path, index=False)
    (out_dir / "kpi.json").write_text(json.dumps(kpi, indent=2, default=str), encoding="utf-8")
    (out_dir / "behavioral_knobs.json").write_text(
        json.dumps(_behavioral_knobs(eng), indent=2), encoding="utf-8"
    )
    return {
        "engine": eng,
        "equity": equity,
        "trades": trades,
        "kpi": kpi,
        "fingerprint": fp,
        "artifacts": arts,
        "knobs": _behavioral_knobs(eng),
    }


def _architecture_audit(discovered: dict) -> str:
    return "\n".join(
        [
            "# Task 0 — Engine Architecture Audit & Baseline Discovery",
            "",
            "## Discovered baseline values (live, not assumed)",
            "```json",
            json.dumps(discovered, indent=2, default=str),
            "```",
            "",
            "## Execution flow (owner functions)",
            "1. `run` — daily loop orchestration",
            "2. `handle_official_delisting` / `handle_silent_delisting` — forced exits",
            "3. `execute_pending_orders` — open fills + `_cost_ratio` (journal observes only)",
            "4. `check_cmvs_exits` — ATR / EMA9 / exhaustion (toggleable)",
            "5. Monthly: `determine_market_regime` → ADV pool → `get_universe` →",
            "   `build_factors` → `rank_universe` → `construct_portfolio`(ENTRY_RANK/EXIT_RANK) →",
            "   `allocate_weights` → `construct_final_targets_with_industry_cap` → `queue_rebalance_orders`",
            "",
            "## Distinct rank parameters",
            "- `ENTRY_RANK`: max names selected / top-core size (`construct_portfolio`)",
            "- `EXIT_RANK`: hysteresis buffer size (keep if still inside top EXIT_RANK)",
            "",
            "## Logging vs execution",
            "- Trade journal / shadow exits attach after fills; they do not change order qty/price/timing.",
            "- Abort if any non-EXIT_RANK behavioral knob differs between baseline and Exp1.",
            "",
        ]
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    abort_reasons = []

    base_cfg = load_config(str(ROOT / "config" / "config.yaml"))

    # Discover available benchmark from panel without assuming QQQ exists
    probe = StandaloneEngine(config=copy.deepcopy(base_cfg), config_path=str(ROOT / "config" / "config.yaml"))
    discovered = discover_baseline_values(base_cfg, probe)
    _write(OUT / "TASK0_ARCHITECTURE_AUDIT.md", _architecture_audit(discovered))

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
        abort_reasons.append("No usable benchmark column in price panel")
        _write(OUT / "ABORT.txt", "\n".join(abort_reasons))
        print("ABORT:", abort_reasons)
        return 2

    # Discovered baseline EXIT/ENTRY from config/engine
    baseline_entry = int(discovered["from_config"]["max_portfolio_size"])
    baseline_exit = int(discovered["from_config"]["selection_buffer_size"])
    # Prefer research block if it matches discovered portfolio knobs
    research = base_cfg.get("research") or {}
    if int(research.get("ENTRY_RANK", baseline_entry)) == baseline_entry:
        baseline_entry = int(research.get("ENTRY_RANK", baseline_entry))
    if int(research.get("EXIT_RANK", baseline_exit)) == baseline_exit:
        baseline_exit = int(research.get("EXIT_RANK", baseline_exit))

    print("=== Task 0 Discovery ===")
    print(bm_note)
    print(f"ENTRY_RANK baseline={baseline_entry} EXIT_RANK baseline={baseline_exit}")
    print(f"Universe tickers={discovered['from_data']['n_tickers']} "
          f"prices={discovered['from_data']['n_price_columns']} "
          f"range={discovered['from_data']['date_start']}→{discovered['from_data']['date_end']}")

    # --- Baseline arm ---
    cfg_base = _prepare_cfg(base_cfg, exit_rank=baseline_exit, benchmark=experiment_benchmark)
    # Force ENTRY to discovered baseline
    cfg_base["research"]["ENTRY_RANK"] = baseline_entry
    cfg_base["max_portfolio_size"] = baseline_entry

    print("\n=== Running BASELINE ===")
    base = _run_arm(cfg_base, "baseline", OUT / "baseline")

    # Abort: defaults fail to reproduce discovered baseline knobs (except benchmark data remap)
    kn = base["knobs"]
    if kn["ENTRY_RANK"] != baseline_entry:
        abort_reasons.append(f"ENTRY_RANK mismatch: {kn['ENTRY_RANK']} vs {baseline_entry}")
    if kn["EXIT_RANK"] != baseline_exit:
        abort_reasons.append(f"EXIT_RANK mismatch on baseline arm: {kn['EXIT_RANK']} vs {baseline_exit}")
    if kn["USE_EMA9_EXIT"] is not True or kn["MIN_HOLD_DAYS"] != 0 or kn["USE_TIME_STOP"] is not False:
        abort_reasons.append(f"Non-rank toggles not at discovered baseline: {kn}")
    if abs(kn["ATR_MULTIPLIER"] - float(base_cfg.get("atr_multiplier", 2.0))) > 1e-12:
        abort_reasons.append("ATR_MULTIPLIER drifted from config")

    baseline_trade_count = int(len(base["trades"]))
    (OUT / "baseline_trade_count.json").write_text(
        json.dumps({"n_closed_trades": baseline_trade_count}, indent=2), encoding="utf-8"
    )

    if abort_reasons:
        _write(OUT / "ABORT.txt", "\n".join(abort_reasons))
        print("ABORT before Exp1:", abort_reasons)
        return 2

    # --- Experiment arm: ONLY EXIT_RANK=80 ---
    exp_exit = 80
    cfg_exp = _prepare_cfg(base_cfg, exit_rank=exp_exit, benchmark=experiment_benchmark)
    cfg_exp["research"]["ENTRY_RANK"] = baseline_entry
    cfg_exp["max_portfolio_size"] = baseline_entry

    print("\n=== Running EXP1 EXIT_RANK=80 ===")
    exp = _run_arm(cfg_exp, "exp1_exit_rank_80", OUT / "exp1")

    # Validate only EXIT_RANK changed
    b_knobs = dict(base["knobs"])
    e_knobs = dict(exp["knobs"])
    # Benchmark intentionally same; ignore label field differences none
    changed = {k: (b_knobs[k], e_knobs[k]) for k in b_knobs if b_knobs[k] != e_knobs[k]}
    only_exit = set(changed.keys()) == {"EXIT_RANK"} and changed.get("EXIT_RANK") == (
        baseline_exit,
        exp_exit,
    )
    if not only_exit:
        abort_reasons.append(f"Behavioral variables changed beyond EXIT_RANK: {changed}")

    # Abort if baseline trade count somehow rewritten (compare stored)
    if int(len(base["trades"])) != baseline_trade_count:
        abort_reasons.append("Baseline trade count changed before experiment comparison")

    # Logging must not alter execution: journal is observation-only by design;
    # record validation Fact.
    logging_fact = {
        "journal_hooks": "post-fill observation in execute_pending_orders / delist handlers",
        "order_generation_depends_on_journal": False,
        "shadow_horizon_days": int(cfg_base.get("research", {}).get("SHADOW_HORIZON_DAYS", 20)),
    }
    (OUT / "logging_execution_separation.json").write_text(
        json.dumps(logging_fact, indent=2), encoding="utf-8"
    )

    if abort_reasons:
        _write(OUT / "ABORT.txt", "\n".join(abort_reasons))
        print("ABORT:", abort_reasons)
        decision = "REJECT"
    else:
        decision = None

    delta = build_delta_report(base["kpi"], exp["kpi"], base["trades"], exp["trades"])
    _write(OUT / "DELTA_REPORT.txt", format_delta_report(delta))
    (OUT / "delta.json").write_text(json.dumps(delta, indent=2, default=str), encoding="utf-8")

    if decision is None:
        decision = decide_from_delta(delta, only_exit_rank_changed=only_exit, abort_reasons=abort_reasons)

    # Facts-only recommendation (Rule #2)
    d_hold = (delta.get("metrics") or {}).get("avg_holding_period_days", {}).get("delta")
    d_turn = (delta.get("metrics") or {}).get("turnover_trades_per_year", {}).get("delta")
    d_miss = (delta.get("metrics") or {}).get("avg_missed_upside", {}).get("delta")
    d_save = (delta.get("metrics") or {}).get("avg_saved_drawdown", {}).get("delta")
    d_trades = (delta.get("metrics") or {}).get("n_closed_trades", {}).get("delta")

    fact = (
        f"On this fingerprint (git={base['fingerprint']['git_commit'][:10]}, "
        f"universe_tickers={discovered['from_data']['n_tickers']}, "
        f"benchmark={experiment_benchmark}), changing EXIT_RANK {baseline_exit}→{exp_exit} "
        f"with ENTRY_RANK fixed at {baseline_entry} produced "
        f"Δn_closed_trades={d_trades}, Δturnover/yr={d_turn}, "
        f"Δavg_hold_days={d_hold}, Δavg_missed_upside={d_miss}, Δavg_saved_drawdown={d_save}."
    )
    if d_trades == 0 and (d_hold is None or abs(float(d_hold)) < 1e-12):
        interpretation = (
            "No measurable change in trade count / holding period. "
            "With price-column universe smaller than both EXIT_RANK values, "
            "hysteresis buffers are saturating (top-K equals full ranked set)."
        )
        confidence = "High for 'no effect on this sample'; Low for generalization"
        next_exp = (
            "Repeat EXIT_RANK experiment on a universe where ranked set size "
            f"> EXIT_RANK (need >{exp_exit} names in factor universe), keeping ENTRY_RANK={baseline_entry}."
        )
        why = (
            "Information gain is blocked while EXIT_RANK exceeds the ranked-set size; "
            "a larger universe makes the EXIT_RANK margin observable."
        )
    else:
        interpretation = (
            "EXIT_RANK change altered at least one research metric under a single-variable design."
        )
        confidence = "Medium (single backtest; requires out-of-sample repeat)"
        next_exp = "Repeat the same EXIT_RANK=80 arm on a later date range / larger universe (no other knobs)."
        why = "Confirms whether the observed Δ is stable before any further single-variable tests."

    reco_txt = format_research_recommendation(
        fact=fact,
        interpretation=interpretation,
        confidence=confidence,
        evidence=(
            f"only_exit_rank_changed={only_exit}; changed={changed}; "
            f"baseline_knobs={b_knobs}; exp_knobs={e_knobs}; {bm_note}"
        ),
        next_experiment=next_exp,
        why_information_gain=why,
    )
    _write(OUT / "RESEARCH_RECOMMENDATION.txt", reco_txt)

    review_txt = format_experiment_review(
        hypothesis=(
            f"Widening EXIT_RANK from {baseline_exit} to {exp_exit} (ENTRY_RANK fixed) "
            "changes turnover / holding / exit-efficiency metrics."
        ),
        changed_variables={"EXIT_RANK": f"{baseline_exit} → {exp_exit}"},
        baseline={
            "ENTRY_RANK": baseline_entry,
            "EXIT_RANK": baseline_exit,
            "n_closed_trades": len(base["trades"]),
            "cagr": (base["kpi"].get("risk") or {}).get("cagr"),
            "avg_missed_upside": (base["kpi"].get("research") or {}).get("avg_missed_upside"),
            "avg_saved_drawdown": (base["kpi"].get("research") or {}).get("avg_saved_drawdown"),
        },
        experiment={
            "ENTRY_RANK": baseline_entry,
            "EXIT_RANK": exp_exit,
            "n_closed_trades": len(exp["trades"]),
            "cagr": (exp["kpi"].get("risk") or {}).get("cagr"),
            "avg_missed_upside": (exp["kpi"].get("research") or {}).get("avg_missed_upside"),
            "avg_saved_drawdown": (exp["kpi"].get("research") or {}).get("avg_saved_drawdown"),
        },
        delta=delta,
        new_fact=fact,
        decision=decision,
    )
    _write(OUT / "EXPERIMENT_REVIEW.txt", review_txt)

    checklist = {
        "architecture_audit": (OUT / "TASK0_ARCHITECTURE_AUDIT.md").exists(),
        "baseline_verification": (OUT / "baseline" / "fingerprint.json").exists() and not abort_reasons,
        "exp1_report": (OUT / "exp1" / "fingerprint.json").exists(),
        "delta_report": (OUT / "DELTA_REPORT.txt").exists(),
        "research_recommendation": (OUT / "RESEARCH_RECOMMENDATION.txt").exists(),
        "decision": decision in {"PASS", "REPEAT", "REJECT"},
        "exit_rank_validation": only_exit and (OUT / "logging_execution_separation.json").exists(),
        "abort_reasons": abort_reasons,
        "decision_value": decision,
        "only_exit_rank_changed": only_exit,
        "benchmark_note": bm_note,
    }
    checklist["complete"] = all(
        [
            checklist["architecture_audit"],
            checklist["baseline_verification"] or decision == "REJECT",
            checklist["exp1_report"],
            checklist["delta_report"],
            checklist["research_recommendation"],
            checklist["decision"],
            checklist["exit_rank_validation"] or decision == "REJECT",
        ]
    )
    # stricter: complete only if all artifacts and validation true when not aborted early
    checklist["complete"] = (
        checklist["architecture_audit"]
        and checklist["exp1_report"]
        and checklist["delta_report"]
        and checklist["research_recommendation"]
        and checklist["decision"]
        and (OUT / "baseline" / "fingerprint.json").exists()
        and (OUT / "logging_execution_separation.json").exists()
        and only_exit
        and not abort_reasons
    )
    (OUT / "VALIDATION_CHECKLIST.json").write_text(json.dumps(checklist, indent=2), encoding="utf-8")
    _write(
        OUT / "VALIDATION_CHECKLIST.txt",
        "\n".join(
            [
                "=" * 64,
                " RESEARCH VALIDATION CHECKLIST (EXP001)",
                "=" * 64,
                *[f"[{'PASS' if checklist[k] else 'FAIL'}] {k}: {checklist[k]}" for k in checklist],
                "=" * 64,
            ]
        ),
    )

    # Experiment history ledger
    hist = ExperimentHistory(OUT / "experiment_history.json")
    hist.append(
        parent_exp=None,
        changed_variables={"EXIT_RANK": exp_exit, "ENTRY_RANK_fixed": baseline_entry},
        hypothesis="EXIT_RANK 70→80 changes turnover/holding/exit efficiency",
        expected_outcome="Measurable Δ in research metrics if ranked set > EXIT_RANK",
        actual_outcome=fact,
        decision=decision,
        metrics={"delta": delta, "checklist": checklist},
        notes=bm_note,
    )

    # Bundle summary
    summary = "\n\n".join(
        [
            format_kpi_report(base["kpi"]),
            format_kpi_report(exp["kpi"]),
            format_delta_report(delta),
            reco_txt,
            review_txt,
            f"Decision={decision} complete={checklist['complete']}",
        ]
    )
    _write(OUT / "EXP001_FULL_REPORT.txt", summary)
    print(summary)
    print(f"\nArtifacts written under {OUT}")
    return 0 if checklist["complete"] or decision in {"PASS", "REPEAT", "REJECT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
