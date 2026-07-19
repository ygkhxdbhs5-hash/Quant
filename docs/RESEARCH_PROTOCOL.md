# Experiment Protocol (Task 7)

## Rules

1. **Never** recommend strategy modifications based on a single backtest.
2. Every recommendation must include:
   - Sample size  
   - Statistical confidence (High ≥80 trades / Medium ≥30 / Low &lt;30)  
   - Consistency requirement (walk-forward / multi-seed)  
   - Estimated impact  
   - Risk of overfitting  
3. Maintain **Experiment History** (`cache/experiment_history.json`) with:
   - ID, Parent Exp, Changed Variables, Hypothesis, Expected/Actual Outcome, Decision

## Baseline identity

Research toggles default to current CMVS behavior:

| Toggle | Baseline default | Notes |
|--------|------------------|-------|
| USE_EMA9_EXIT | True | Matches current CMVS exits |
| ATR_MULTIPLIER | 2.0 | Matches current ATR trail |
| ENTRY_RANK | `max_portfolio_size` (50) | Spec example 30 would change trades — not used as silent default |
| EXIT_RANK | `selection_buffer_size` (70) | Spec example 80 would change trades — not used as silent default |
| MIN_HOLD_DAYS | 0 | No min-hold gate |
| USE_TIME_STOP | False | Time stop disabled |

If any experiment changes these knobs, record Expected vs Actual and do **not** promote without out-of-sample confirmation.

## Artifacts (Facts produced each run)

- `cache/trade_journal.csv` — closed trades + shadow counterfactuals  
- `cache/research_report.txt` — KPI + recommendations + validation  
- `cache/research_artifacts.json` — machine-readable bundle  
- `cache/baseline_fingerprint.json` — trade count / turnover at baseline defaults  
- `cache/experiment_history.json` — append-only ledger  
