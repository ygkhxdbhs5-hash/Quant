# Experiment 1 ONLY — EXIT_RANK 70 → 80

## Runtime configuration (verified)
- Baseline: ENTRY_RANK=50, EXIT_RANK=70
- Experiment: ENTRY_RANK=50, EXIT_RANK=80
- only_exit_rank_changed: True
- changed_knobs: {'EXIT_RANK': (70, 80)}
- abort_reasons: []

## 1) Baseline metrics
- Turnover (trades/yr): 98.46448087431695
- Avg Holding Period (days): 7.308108108108108
- CAGR: -0.05224010395993939
- MDD: -0.3818193415306011
- Win Rate: 0.43648648648648647
- Avg Missed Upside: 0.07070414309922131
- Avg Saved Drawdown: 0.051284119863297756
- Closed trades: 740

## 2) Experiment metrics (EXIT_RANK=80)
- Turnover (trades/yr): 98.46448087431695
- Avg Holding Period (days): 7.308108108108108
- CAGR: -0.05224010395993939
- MDD: -0.3818193415306011
- Win Rate: 0.43648648648648647
- Avg Missed Upside: 0.07070414309922131
- Avg Saved Drawdown: 0.051284119863297756
- Closed trades: 740

## 3) Delta Report (Experiment − Baseline)
- Δ Turnover: 0.0
- Δ Holding Period: 0.0
- Δ CAGR: 0.0
- Δ MDD: 0.0
- Δ Win Rate: 0.0
- Δ Missed Upside: 0.0
- Δ Saved Drawdown: 0.0

### Δ Exit Breakdown
- atr_trail: count Δ=0.0 (4.0 → 4.0); share Δ=0.0 (0.005405405405405406 → 0.005405405405405406)
- ema9_break: count Δ=0.0 (724.0 → 724.0); share Δ=0.0 (0.9783783783783784 → 0.9783783783783784)
- exhaustion: count Δ=0.0 (12.0 → 12.0); share Δ=0.0 (0.016216216216216217 → 0.016216216216216217)

## 4) Single-variable verification
- PASS: experiment EXIT_RANK == 80 → True
- PASS: baseline EXIT_RANK == 70 → True
- PASS: only EXIT_RANK changed → True

## Decision

**REPEAT**
