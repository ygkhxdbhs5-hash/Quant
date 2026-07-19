# EXP001 — EXIT_RANK controlled experiment

## Research rules applied
1. Changed **only** `EXIT_RANK` (70 → 80); `ENTRY_RANK` fixed at discovered baseline **50**.
2. No performance optimization; recommendations are Facts-only.
3. Abort criteria checked (single-variable, baseline defaults, logging separation).

## How to reproduce
```bash
PYTHONPATH=. python3 scripts/run_exp1_exit_rank.py
```

## Decision
See `EXPERIMENT_REVIEW.txt` — on the local 12-name panel, Δ=0 → **REPEAT** on a universe with ranked-set size > 80.

## Artifacts
- `TASK0_ARCHITECTURE_AUDIT.md`
- `baseline/` + `exp1/` fingerprints, journals, KPIs
- `DELTA_REPORT.txt`
- `RESEARCH_RECOMMENDATION.txt`
- `EXPERIMENT_REVIEW.txt`
- `VALIDATION_CHECKLIST.json`
