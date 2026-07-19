# Task 0 — Engine Architecture Snapshot

**Scope:** Read-only inventory of `engine/strategy.py` (`StandaloneEngine`) as of CMVS v3 / webapp tip.  
**No code was modified for this report.**

## 1. Backtest entry

| Step | Owner |
|------|--------|
| Load config | `load_config` / `run_backtest.main` |
| Construct engine | `StandaloneEngine.__init__` |
| Run loop | `StandaloneEngine.run` |
| Persist equity / chart | `run_backtest.main` |

## 2. Daily execution order (`run`)

For each trading day after warmup (`MOM_WINDOW` / `LOWVOL_WINDOW`):

1. `handle_official_delisting` — force-liquidate at close on official delist date  
2. `handle_silent_delisting` — haircut liquidation after silent gap  
3. `execute_pending_orders` — fill prior-day orders at **open** with `_cost_ratio`  
4. `check_cmvs_exits` — daily CMVS risk exits → queue **SELL** (next open)  
5. Mark equity (`equity_curve`) from cash + close marks  
6. **If month-start rebalance day:**
   - Build active symbols (non-benchmark, non-ETF, priced)  
   - `determine_market_regime` (+ optional 1.2x leverage append)  
   - Bear (`exposure==0`): queue full SELL, skip selection  
   - Else: ADV pool → universe → factors → rank → construct → size → industry cap → `queue_rebalance_orders`  
   - `log_rebalance_summary`

## 3. Universe selection

| Responsibility | Owner |
|----------------|--------|
| Active listed names (price/ETF filter) | `run` (inline) |
| Top ADV pool (`TOP_ADV_POOL`) | `run` via `dvol_m.nlargest` |
| Soft screens (currently pass-through; fundamentals commented) | `get_universe` |

## 4. Factor calculation

| Responsibility | Owner |
|----------------|--------|
| Precompute BB / vol-z / CP / RSI / EMA9 / ATR / ret20 / spreads | `_precompute_matrices` |
| Cross-sectional RSS + per-name BBS/VZS/CPS/RSIS/RSS row build | `build_factors` |
| Optional PIT lookup (unused by CMVS path) | `get_latest_available_fundamentals` |

## 5. Ranking flow

| Responsibility | Owner |
|----------------|--------|
| `final_score = Σ w_i * factor_i` | `rank_universe` |
| Sort descending by `final_score` | `rank_universe` |

## 6. Portfolio construction & sizing

| Responsibility | Owner |
|----------------|--------|
| Top-N + buffer hysteresis + Spearman corr filter | `construct_portfolio` |
| Equal-weight × regime exposure (ATR sizing commented) | `allocate_weights` |
| Industry weight cap sequential fill | `construct_final_targets_with_industry_cap` |
| Assertions / warnings | `validate_portfolio` |

## 7. Entry / exit generation

| Signal | Owner | Notes |
|--------|--------|--------|
| Rebalance entry/exit deltas | `queue_rebalance_orders` | Target weight vs current → next-open orders |
| ATR trail | `check_cmvs_exits` | `close < peak_close − atr_multiplier × atr_14` |
| EMA9 breakdown | `check_cmvs_exits` | `close < ema_9` |
| RSI exhaustion | `check_cmvs_exits` | `rsi>80` and `daily_cp<0.3` |
| Forced cash / bear flatten | `run` | exposure 0 |

## 8. Order execution & costs

| Responsibility | Owner |
|----------------|--------|
| Participation caps | `_max_shares_participation` |
| Spread + impact + commission + slippage | `_cost_ratio` |
| Fill at open, update cash/qty/peaks | `execute_pending_orders` |

## 9. State carried across days

- `portfolio`, `cash`, `pending_orders`, `previous_target_symbols`  
- `highest_prices` (peak close since entry)  
- `_cmvs_forced_exits` (same-day rebuy block)  
- `equity_curve`

## 10. Baseline-identity note (for upcoming research toggles)

Current live knobs that define trade identity:

- `atr_multiplier = 2.0`  
- EMA9 exit **enabled**  
- Top portfolio size = `max_portfolio_size` (**50** in config)  
- Buffer = `selection_buffer_size` (**70** in config)  
- No min-hold gate, no time stop  

Research toggles introduced later must default to these values. Spec examples `ENTRY_RANK=30` / `EXIT_RANK=80` would **change** baseline vs current 50/70 and therefore must **not** be applied as silent defaults (Core Research Principle).
