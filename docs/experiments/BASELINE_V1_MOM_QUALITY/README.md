# Baseline v1 — Momentum + Quality A/B

**Status: ADOPTED as new default** (`enable_quality_factor: true`)

## Fingerprint (clean)

| Field | Value |
|---|---|
| repro_id | `42799a276531` |
| payload_sha256_16 | `2e80f15acc894c9f` |
| git_dirty | **False** |
| git_head | `6af4c423790f51e16c2f2f1272c163253f72e865` |
| window | 2022-01-01 → 2026-06-30 |

## Decision rule

Sharpe improved (0.194 → 0.268) and MDD improved (−31.97% → −25.60%), so quality is kept as the new default base for the next increment (industry-neutral ranking).

## 2023 alpha callout

Mom-only Alpha −62.77% → Mom+Quality −51.00% (Δ +11.77pp): **improved** (less negative) but the reversal-lag pattern remains.

## Artifacts

- `mom_quality_ab_report.txt` / `.json` — single-payload A/B
- Prior momentum-only checkpoint: tag `baseline-v1-nochase` / repro `50978c877d89`
