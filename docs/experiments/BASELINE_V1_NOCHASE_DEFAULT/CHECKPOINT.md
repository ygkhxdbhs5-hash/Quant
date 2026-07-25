# Checkpoint: baseline-v1-nochase

**Tag:** `baseline-v1-nochase`  
**Purpose:** Clean, fully committed reference for all future factor-addition experiments.

## Fingerprint (clean tree)

| Field | Value |
|---|---|
| repro_id | `97c21a974bf8` |
| payload_sha256_16 | `caf0cb3f75ae98bb` |
| git_head | `0c7dbced0f3349dfa3386d249e145ca423241471` |
| git_dirty | **False** |
| window | 2022-01-01 → 2026-06-30 |
| panels.pkl | `5ce58a430ac1beec` |
| cost model | Fixed CS v2 + ADV winsorize |
| fill policy | `enable_topup_chasing=false` (default) |

## KPI summary (this checkpoint)

| Metric | No-chase (default) | Chase-on (legacy flag) |
|---|---:|---:|
| CAGR | **+1.89%** | −8.00% |
| Sharpe | 0.194 | −0.223 |
| MDD | −31.97% | −55.47% |
| Cost $ | 12,536,785 | 17,743,691 |
| n_trades | 1,589 | 1,420 |

## Unreproducible prior claim (not this checkpoint)

`repro_id=5146837eafd9` (+10.55% / $10.25M / 1052) could not be recovered (dirty-tree data loss). See `ROOT_CAUSE.md`.

## Artifacts

- `nochase_clean_reference.txt` / `.json` — single in-memory payload
- `ROOT_CAUSE.md` — Task 1/2 findings
