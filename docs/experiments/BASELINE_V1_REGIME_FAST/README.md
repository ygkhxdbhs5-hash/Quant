# Baseline v1 — Fast Regime (SMA50 + breadth RoC) A/B

**Status: NOT ADOPTED — stop regime tuning this round**

## Fingerprint (clean)

| Field | Value |
|---|---|
| repro_id | `7303e0d94dbc` |
| payload_sha256_16 | `80f48095842e48bd` |
| git_dirty | **False** |

## Decision

2023 Alpha did **not** improve vs MQ (−51.00%) or vs sma200-regime (−53.77%); FAST landed at **−54.07%**. Sharpe collapsed (0.268 → 0.035). January 2023 stayed at **0.5x** (same lag as sma200).

Per decision rule: reject, **stop further regime-threshold tuning**, keep momentum+quality as default, next increment = **industry-neutral ranking**.
