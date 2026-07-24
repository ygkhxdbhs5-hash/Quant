"""Unit tests for research infrastructure (no full backtest required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.research.config_toggles import ResearchToggles, load_research_toggles
from engine.research.kpi_report import build_hierarchical_kpi_report
from engine.research.rank_diagnostics import RankDiagnostics
from engine.research.recommendations import build_research_recommendation_report
from engine.research.trade_journal import TradeJournal
from engine.research.validation import run_research_validation_checklist


def test_toggles_baseline_defaults():
    t = load_research_toggles({"max_portfolio_size": 50, "selection_buffer_size": 70})
    assert t.USE_EMA9_EXIT is True
    assert t.EMA_EXIT_LENGTH == 9
    assert t.USE_ATR_EXIT is True
    assert t.ATR_MULTIPLIER == 2.0
    assert t.USE_EXHAUSTION_EXIT is True
    assert t.ENTRY_RANK == 50
    assert t.EXIT_RANK == 70
    assert t.MIN_HOLD_DAYS == 0
    assert t.USE_TIME_STOP is False
    assert t.TIME_STOP_DAYS == 20
    assert t.MAX_PORTFOLIO_SIZE == 50
    assert t.MONTHLY_REBALANCE is True
    assert t.MAX_INDUSTRY_WEIGHT == 0.40
    assert t.is_baseline_defaults(50, 70)


def test_research_config_panel_printout():
    t = ResearchToggles()
    panel = t.format_panel()
    assert "RESEARCH CONFIGURATION" in panel
    assert "ENTRY_RANK = 50" in panel
    assert "EXIT_RANK = 70" in panel
    assert "USE_EMA9_EXIT = True" in panel
    assert "EMA_EXIT_LENGTH = 9" in panel
    assert "USE_ATR_EXIT = True" in panel
    assert "ATR_MULTIPLIER = 2.0" in panel
    assert "USE_EXHAUSTION_EXIT = True" in panel
    assert "USE_TIME_STOP = False" in panel
    assert "TIME_STOP_DAYS = 20" in panel
    assert "MIN_HOLD_DAYS = 0" in panel
    assert "MAX_PORTFOLIO_SIZE = 50" in panel
    assert "MAX_INDUSTRY_WEIGHT = 0.40" in panel


def test_ema_exit_length_override():
    t = load_research_toggles(
        {
            "max_portfolio_size": 50,
            "selection_buffer_size": 70,
            "research": {"EMA_EXIT_LENGTH": 20, "USE_ATR_EXIT": False},
        }
    )
    assert t.EMA_EXIT_LENGTH == 20
    assert t.USE_ATR_EXIT is False
    assert t.USE_EMA9_EXIT is True  # unchanged
    assert not t.is_baseline_defaults(50, 70)


def test_shadow_exit_counterfactuals():
    idx = pd.bdate_range("2020-01-02", periods=40)
    close = pd.DataFrame({"AAA": np.linspace(100, 120, len(idx))}, index=idx)
    # Exit at day 10 price, then rises then dips
    close.loc[idx[11:16], "AAA"] = [105, 110, 115, 112, 108]
    j = TradeJournal(shadow_horizon_days=20)
    j.on_entry("AAA", idx[5], 5, 100.0, 10, rank=1, cmvs=0.8, rsi=55.0, atr=2.0)
    j.mark_peak("AAA", 104.0)
    rec = j.on_exit(
        "AAA",
        idx[10],
        10,
        float(close["AAA"].iloc[10]),
        "ema9_break",
        close,
        exit_rank=5,
        exit_cmvs=0.4,
        exit_rsi=45.0,
        exit_atr=2.1,
    )
    assert rec is not None
    assert rec.missed_upside is not None
    assert rec.saved_drawdown is not None
    assert rec.days_to_peak is not None
    assert rec.holding_days >= 0
    df = j.to_frame()
    assert "missed_upside" in df.columns
    assert "forward_return_20d" in df.columns


def test_recommendation_and_kpi_smoke():
    idx = pd.bdate_range("2020-01-02", periods=60)
    close = pd.DataFrame({"AAA": np.linspace(100, 130, len(idx))}, index=idx)
    j = TradeJournal(shadow_horizon_days=20)
    for k in range(40):
        # staggered synthetic round-trips with post-exit upside
        e = 1 + k
        x = e + 3
        j.on_entry("AAA", idx[e], e, float(close["AAA"].iloc[e]), 1, 1, 0.7, 55.0, 2.0)
        j.on_exit(
            "AAA",
            idx[x],
            x,
            float(close["AAA"].iloc[x]),
            "ema9_break",
            close,
            exit_rank=10,
            exit_cmvs=0.4,
            exit_rsi=48.0,
            exit_atr=2.1,
        )
    trades = j.to_frame()
    assert len(trades) == 40

    toggles = ResearchToggles().as_dict()
    reco = build_research_recommendation_report(trades, toggles)
    assert reco["sample_size"] == 40
    assert reco["next_experiment"]["statistical_confidence"] in {"High", "Medium", "Low"}

    eq = pd.DataFrame(
        {"Total_Equity": np.linspace(1e6, 1.2e6, 252)},
        index=pd.bdate_range("2020-01-01", periods=252),
    )
    kpi = build_hierarchical_kpi_report(eq, trades)
    assert "research" in kpi and "risk" in kpi and "return" in kpi

    val = run_research_validation_checklist(
        ResearchToggles(),
        trades,
        max_portfolio_size=50,
        selection_buffer_size=70,
    )
    assert val["all_pass"] is True, val


def test_rank_diagnostics_counts_and_distribution():
    d = RankDiagnostics(exit_rank=70)
    # Holding ranks below threshold — distribution only
    d.observe(date="2020-01-02", ticker="AAA", rank=5, ema9_break=False)
    d.observe(date="2020-01-03", ticker="AAA", rank=12, ema9_break=True)
    # Candidate without ema overlap
    d.observe(date="2020-01-06", ticker="BBB", rank=71, ema9_break=False)
    # Candidate with ema preempt (counts even if another exit also fired)
    d.observe(date="2020-01-07", ticker="CCC", rank=90, ema9_break=True)
    # Missing rank ignored
    d.observe(date="2020-01-08", ticker="DDD", rank=None, ema9_break=True)

    assert d.rank_exit_candidates == 2
    assert d.ema_preempted_rank_exit == 1
    assert len(d.holding_ranks) == 4

    summary = d.summary()
    dist = summary["holding_rank_distribution"]
    assert dist["n"] == 4
    assert dist["min"] == 5
    assert dist["mean"] is not None
    assert dist["median"] is not None
    assert dist["p75"] is not None
    assert dist["p90"] is not None
    assert dist["p95"] is not None
    assert dist["max"] == 90

    report = d.format_report()
    assert "rank_exit_candidates: 2" in report
    assert "ema_preempted_rank_exit: 1" in report
    assert "holding_rank_distribution:" in report
    assert "recommend" not in report.lower()


def test_experiment_execution_single_variable_validation_and_report():
    from engine.research.delta_report import build_delta_report
    from engine.research.experiment_execution import (
        compare_behavioral_identity,
        decide_experiment,
        format_experiment1_full_report,
        format_validation_failed,
        generate_facts,
        research_recommendation_from_results,
    )

    base_id = {
        "USE_EMA9_EXIT": True,
        "ATR_MULTIPLIER": 2.0,
        "ENTRY_RANK": 50,
        "EXIT_RANK": 70,
        "MIN_HOLD_DAYS": 0,
        "USE_TIME_STOP": False,
        "TIME_STOP_DAYS": 0,
        "BENCHMARK_TICKER": "SPY",
        "TOP_ADV_POOL": 500,
        "CORR_THRESHOLD": 0.85,
        "MAX_INDUSTRY_WEIGHT": 0.35,
        "COMMISSION_RATE": 0.001,
        "SLIPPAGE_RATE": 0.0005,
        "UNIVERSE_SAMPLE_SIZE": 500,
        "UNIVERSE_SAMPLE_SEED": 42,
        "START_DATE": "2018-01-01",
        "END_DATE": None,
        "N_PRICE_COLUMNS": 12,
        "N_PRICE_ROWS": 2000,
        "PANELS_SHA256": "abc",
        "UNIVERSE_SHA256": "def",
    }
    exp_id = dict(base_id)
    exp_id["EXIT_RANK"] = 80
    ok = compare_behavioral_identity(
        base_id, exp_id, allowed_changes={"EXIT_RANK": (70, 80)}
    )
    assert ok["passed"] is True

    bad = dict(exp_id)
    bad["ATR_MULTIPLIER"] = 3.0
    fail = compare_behavioral_identity(
        base_id, bad, allowed_changes={"EXIT_RANK": (70, 80)}
    )
    assert fail["passed"] is False
    assert "ATR_MULTIPLIER" in fail["unexpected_diffs"]
    assert "VALIDATION FAILED" in format_validation_failed(fail)

    # Synthetic identical KPIs → REPEAT + INSUFFICIENT EVIDENCE
    kpi = {
        "research": {
            "n_closed_trades": 100,
            "avg_holding_period_days": 7.0,
            "avg_missed_upside": 0.05,
            "avg_saved_drawdown": 0.04,
            "efficiency_ratio": 1.0,
            "turnover_trades_per_year": 50.0,
        },
        "risk": {"cagr": 0.1, "mdd": -0.2, "sharpe": 0.5, "sortino": 0.6, "calmar": 0.5},
        "return": {"total_return": 0.5, "win_rate": 0.55, "profit_factor": 1.2},
    }
    rd = {
        "rank_exit_candidates": 0,
        "ema_preempted_rank_exit": 0,
        "holding_rank_distribution": {
            "n": 10,
            "mean": 5.0,
            "median": 5.0,
            "p75": 7.0,
            "p90": 9.0,
            "p95": 9.5,
        },
    }
    empty_trades = pd.DataFrame({"exit_reason": [], "final_return": []})
    delta = build_delta_report(
        kpi,
        kpi,
        empty_trades,
        empty_trades,
        baseline_rank_diagnostics=rd,
        experiment_rank_diagnostics=rd,
    )
    assert "sharpe" in delta["metrics"]
    assert "rank_exit_candidates" in delta["metrics"]
    decision = decide_experiment(delta, validation_passed=True)
    assert decision == "REPEAT"
    facts = generate_facts(
        baseline_exit=70, experiment_exit=80, delta=delta, validation_passed=True
    )
    assert any("Fact:" in f for f in facts)
    reco = research_recommendation_from_results(
        facts=facts,
        delta=delta,
        n_closed_trades_baseline=100,
        n_closed_trades_experiment=100,
        validation_passed=True,
        decision=decision,
    )
    assert reco == "INSUFFICIENT EVIDENCE"

    report = format_experiment1_full_report(
        architecture_audit="audit",
        baseline_verification="verified",
        baseline_entry=50,
        baseline_exit=70,
        experiment_exit=80,
        baseline_identity=base_id,
        experiment_identity=exp_id,
        validation=ok,
        delta=delta,
        facts=facts,
        recommendation=reco,
        decision=decision,
        review_body="review",
    )
    for section in (
        "1. Architecture Audit",
        "2. Baseline Verification",
        "3. Experiment 1 Report",
        "4. Validation",
        "5. Delta Report",
        "6. Fact Generation",
        "7. Research Recommendation",
        "8. PASS / REPEAT / REJECT",
    ):
        assert section in report


if __name__ == "__main__":
    test_toggles_baseline_defaults()
    test_shadow_exit_counterfactuals()
    test_recommendation_and_kpi_smoke()
    test_rank_diagnostics_counts_and_distribution()
    test_experiment_execution_single_variable_validation_and_report()
    print("OK")
