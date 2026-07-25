# Baseline v1 — Mom+Quality + Regime Exposure A/B

**Status: NOT ADOPTED** (`enable_regime_exposure` remains false)

## Fingerprint (clean)

| Field | Value |
|---|---|
| repro_id | `a2f5d870e697` |
| payload_sha256_16 | `e394f38c12771d81` |
| git_dirty | **False** |

## Decision

Sharpe did not improve (0.268 → 0.255) and **2023 alpha worsened** (−51.00% → −53.77%). Do not adopt.

Q1 2023: Jan still at 0.5x (below MA200); Feb–Mar already 1.0x — regime lagged the early recovery and cut exposure into the rally start.

Default remains momentum+quality (`enable_quality_factor: true`, regime off).
