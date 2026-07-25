# Root cause: why +10.55% (repro_id=5146837eafd9) did not reproduce

## Task 1 — Dirty-tree recovery

**Outcome: NOT RECOVERABLE (data loss).**

| Source checked | Result |
|---|---|
| `git stash list` | empty |
| Editor autosave / `*~` / `*.bak` / Local History | none found |
| `topup_chase_report_2.txt` | **does not exist** in repo or workspace |
| Fingerprint `5146837eafd9` / `75f6331be63401bb` | never emitted by any agent run in this environment; only appears as a user-supplied REF |
| Commit `4d2f2a4` message / artifacts | records **no-chase CAGR = +1.89%**, not +10.55% |

The KEY CLUE (`git_head=4d2f2a4` dirty) cannot be verified here: a verify run on that same HEAD with a dirty tree in *this* environment still produced **+1.8891%**, not +10.55%. Dirty@4d2f2a4 alone does not encode the +10.55% logic.

`docs/experiments/BASELINE_V1_TOPUP_CHASE/RECONCILIATION.md` already noted that a claimed ~10.6% chart was **not present** in repo artifacts (stale Streamlit cache / dual-axis misread / different machine).

**Per Task 1c:** the +10.55% number cannot be exactly reproduced. The new reference is a **clean, fully committed** no-chase implementation from spec, with *its* KPIs as the checkpoint.

## Task 2 — What actually changed (narrow diff)

### No-chase skip rule (spec check)

In `_queue_rebalance_to_targets` (`engine/baseline_engine.py`), when no-chase is active and the symbol is already held:

- **BUY** → skip (no catch-up top-up to equal-weight)
- **SELL while still in this month's targets** → skip (no size-trim)
- **SELL when not in targets** → allowed (momentum drop-out)
- **ATR trail SELLs** → separate path (`_check_atr_exits`); **not** gated by the top-up skip

This skip block is **byte-identical** from the original introduce commit `fb82d1a` through `4d2f2a4` to current HEAD (`IDENTICAL_SKIP_BLOCK=True`).

### Accidental side effects?

| Layer | Diff vs introduce (`fb82d1a`) / pre-promotion (`4d2f2a4`) |
|---|---|
| Skip / order-generation body | **unchanged** |
| Cost model / `execution_costs.py` | **0 lines** changed since `fb82d1a` |
| ADV winsorize / participation caps | **unchanged** |
| Momentum entry / ATR exit / sizing / leverage | **unchanged** (strategy formulas untouched; only docstring/`strategy_id` text) |
| Default flag wiring only | default flipped to `ENABLE_TOPUP_CHASING=False`; chase-on still available via explicit flag |

**Conclusion:** the promotion refactor did **not** alter fill behavior relative to the original `DISABLE_TOPUP_CHASING=True` path. The KPI gap vs the claimed Streamlit fingerprint is **not** explained by a logic regression in this repo.

### KPI gap context (this environment's authoritative report)

From `topup_chase_report.txt` (Fixed CS v2, same panels):

| Variant | CAGR | Sharpe | MDD | Cost | n_trades |
|---|---:|---:|---:|---:|---:|
| Chase ON | −8.00% | −0.223 | −55.47% | $17.74M | 1,420 |
| No-chase | **+1.89%** | 0.194 | −31.97% | $12.54M | 1,589 |

Claimed Streamlit pair (−0.24% chase / +10.55% no-chase) was never produced here; both legs disagree, which points to a **different runtime** (data cache / machine), not a one-sided no-chase bug.
