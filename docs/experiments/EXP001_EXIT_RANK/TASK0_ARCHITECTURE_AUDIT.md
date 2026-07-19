# Task 0 — Engine Architecture Audit & Baseline Discovery

## Discovered baseline values (live, not assumed)
```json
{
  "from_config": {
    "max_portfolio_size": 50,
    "selection_buffer_size": 70,
    "atr_multiplier": 2.0,
    "benchmark": "QQQ",
    "universe_sample_size": 500,
    "universe_sample_seed": 42,
    "start_date": "2010-01-01",
    "end_date": "2026-07-19",
    "research_block": {
      "USE_EMA9_EXIT": true,
      "ATR_MULTIPLIER": 2.0,
      "ENTRY_RANK": 50,
      "EXIT_RANK": 70,
      "MIN_HOLD_DAYS": 0,
      "USE_TIME_STOP": false,
      "TIME_STOP_DAYS": 20,
      "SHADOW_HORIZON_DAYS": 20
    }
  },
  "from_engine": {
    "USE_EMA9_EXIT": true,
    "ATR_MULTIPLIER": 2.0,
    "ENTRY_RANK": 50,
    "EXIT_RANK": 70,
    "MIN_HOLD_DAYS": 0,
    "USE_TIME_STOP": false,
    "TIME_STOP_DAYS": 20,
    "BENCHMARK_TICKER": "QQQ",
    "TOP_ADV_POOL": 250,
    "CORR_THRESHOLD": 0.95
  },
  "from_data": {
    "n_tickers": 12,
    "n_all_tickers": 30,
    "n_price_columns": 12,
    "n_price_rows": 2146,
    "date_start": "2018-01-02",
    "date_end": "2026-07-17",
    "benchmark_in_panel": false
  }
}
```

## Execution flow (owner functions)
1. `run` — daily loop orchestration
2. `handle_official_delisting` / `handle_silent_delisting` — forced exits
3. `execute_pending_orders` — open fills + `_cost_ratio` (journal observes only)
4. `check_cmvs_exits` — ATR / EMA9 / exhaustion (toggleable)
5. Monthly: `determine_market_regime` → ADV pool → `get_universe` →
   `build_factors` → `rank_universe` → `construct_portfolio`(ENTRY_RANK/EXIT_RANK) →
   `allocate_weights` → `construct_final_targets_with_industry_cap` → `queue_rebalance_orders`

## Distinct rank parameters
- `ENTRY_RANK`: max names selected / top-core size (`construct_portfolio`)
- `EXIT_RANK`: hysteresis buffer size (keep if still inside top EXIT_RANK)

## Logging vs execution
- Trade journal / shadow exits attach after fills; they do not change order qty/price/timing.
- Abort if any non-EXIT_RANK behavioral knob differs between baseline and Exp1.
