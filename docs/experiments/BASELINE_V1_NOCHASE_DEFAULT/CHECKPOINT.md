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

## Clean four-way cost reference (corrected)

| Field | Value |
|---|---|
| repro_id | `c8ce2d33f4c2` |
| payload_sha256_16 | `a24320c8485a1276` |
| git_dirty | **False** |
| artifact | `clean_fourway_cost_report.txt` / `.json` |

| Variant | CAGR | Sharpe | MDD | Cost $ | n_trades |
|---|---:|---:|---:|---:|---:|
| Fixed CS v2 chase ON | −8.00% | −0.223 | −55.47% | 17,743,691 | 1,420 |
| Fixed CS v2 no-chase (default) | **+1.89%** | 0.194 | −31.97% | 12,536,785 | 1,589 |
| Flat 10bps RT (5bps/side) | **+1.12%** | 0.166 | −39.85% | 1,805,089 | 1,415 |
| Flat 30bps RT (15bps/side) | −0.30% | 0.108 | −41.63% | 5,280,676 | 1,420 |

`topup_chase_report_2.txt` flat10=+13.02% / flat30=+10.34% are **INVALID** (dirty-tree). Flat legs above match the original clean `topup_chase_report.txt`.

## Artifacts

- `nochase_clean_reference.txt` / `.json` — tagged no-chase checkpoint payload
- `clean_fourway_cost_report.txt` / `.json` — four-way clean reference
- `ROOT_CAUSE.md` — dirty-tree recovery findings
