# Top-up chase artifact reconciliation

- **Authoritative numbers**: this directory's `topup_chase_report.txt` / `topup_chase_results.json` / `topup_chase_charts.png` from a **single** `run_topup_chase_report.py` invocation.
- **repro_id**: `34d18034f13c`
- **payload_sha256_16**: `289c9f35070fd690`
- **git**: `f1191274f92a72e592937519a2a7e520340fdc6c` (dirty)
- **window**: 2022-01-01 → 2026-06-30

## Investigation note (prior mismatch report)

A claimed chart with BUY counts 738+3843 (~4,581), no-chase CAGR ~10.6%, flat-10bps CAGR ~13% was **not found** in the repo artifacts. On-disk text/JSON/PNG already agreed on n_buy=6,830 / no-chase CAGR≈1.89% / flat-10bps≈1.12% before this regenerate. Likely causes of the claimed mismatch:
1. Stale Streamlit-cached image from an older local render, or
2. Dual-axis misread (green=CAGR%, brown=Cost $M), or
3. Comparing chart from a different experiment/window.

Going forward: refuse to compare text vs chart unless `repro_id` in the text banner matches the chart footer.
