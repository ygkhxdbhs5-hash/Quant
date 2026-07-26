# Baseline v1 — Industry-neutral mom+quality A/B

**Status: NOT ADOPTED** (`enable_industry_neutral_ranking` remains false)

## Fingerprint (clean)

| Field | Value |
|---|---|
| repro_id | `94a59514bcf8` |
| payload_sha256_16 | `566234f030cf2943` |
| git_dirty | **False** |

## Decision

Sharpe fell (0.268 → 0.153). Concentration did **not** improve (mean max industry weight 10.19% → 11.30%; mean # industries 25.17 → 23.20). Keep global-ranking mom+quality as default.

Likely cause in this 500-name sample: many industries are small → frequent global fallback; within-industry ranks among thin groups can also concentrate on a few large sectors that dominate the liquid+quality set.
