# Architecture Audit

Engine Architecture Audit (discovered, not assumed)

{
  "from_config": {
    "max_portfolio_size": 50,
    "selection_buffer_size": 70,
    "atr_multiplier": 2.0,
    "benchmark": "QQQ",
    "universe_sample_size": 500,
    "universe_sample_seed": 42,
    "start_date": "2010-01-01",
    "end_date": "2026-07-20",
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

Execution flow:
  run → delist handlers → execute_pending_orders → check_cmvs_exits
      → _observe_rank_diagnostics (observation only)
      → monthly: regime → ADV → universe → factors → rank
               → construct_portfolio(ENTRY_RANK/EXIT_RANK)
               → allocate_weights → industry cap → queue_rebalance_orders

Distinct rank parameters:
  ENTRY_RANK — top-core / max portfolio size
  EXIT_RANK  — hysteresis buffer (keep if still in top EXIT_RANK)

Logging vs execution: trade journal / rank diagnostics observe only;
they do not change order qty, price, or timing.