"""Event-driven buffered rebalancing: buy Top ENTRY_RANK, hold to EXIT_RANK."""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.research.config_toggles import DEFAULT_ENTRY_RANK, load_research_toggles
from engine.strategy import RebalanceContext, StandaloneEngine


class _TinyEngine(StandaloneEngine):
    """Bypass data loading; exercise construct_portfolio only."""

    def __init__(self):
        self.MAX_PORTFOLIO_SIZE = 5
        self.SELECTION_BUFFER_SIZE = 8
        self.ENTRY_RANK = 3  # buy band Top 3
        self.EXIT_RANK = 8  # hold buffer Top 8
        self.MAX_INDUSTRY_WEIGHT = 0.40
        self.CORR_WINDOW = 5
        self.CORR_THRESHOLD = 0.99  # effectively off for synthetic uncorrelated data
        self.previous_target_symbols = set()
        self._previous_rank_map = {}
        self.reentry_cooldown = None
        # Uncorrelated synthetic closes so correlation filter does not reject
        dates = pd.bdate_range("2024-01-01", periods=30)
        rng = np.random.default_rng(0)
        cols = {f"S{i}": 100 + rng.normal(0, 1, size=len(dates)).cumsum() for i in range(1, 16)}
        self.close_m = pd.DataFrame(cols, index=dates)


def _ranked(n: int = 15) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append(
            {
                "symbol": f"S{i}",
                "final_score": float(n - i + 1),
                "industry": "Tech" if i % 2 == 0 else "Health",
                "vol_raw": 0.02,
            }
        )
    return pd.DataFrame(rows)


def test_defaults_buy_band_decoupled_from_capacity():
    t = load_research_toggles({"max_portfolio_size": 20, "selection_buffer_size": 30})
    assert t.ENTRY_RANK == DEFAULT_ENTRY_RANK == 10
    assert t.EXIT_RANK == 30
    assert t.MAX_PORTFOLIO_SIZE == 20
    assert t.is_baseline_defaults(20, 30)


def test_hold_within_buffer_no_replacement_by_higher_rank():
    """Existing holding at rank 5 stays; rank-1 outsider does not force a swap when full."""
    eng = _TinyEngine()
    eng.MAX_PORTFOLIO_SIZE = 3
    eng.ENTRY_RANK = 2
    eng.EXIT_RANK = 6
    # Full book of buffer-zone names (ranks 3,4,5 in current ranking)
    eng.previous_target_symbols = {"S3", "S4", "S5"}
    eng._previous_rank_map = {"S3": 2, "S4": 3, "S5": 4}

    ranked = _ranked(12)
    ctx = RebalanceContext()
    out = eng.construct_portfolio(ranked, date_idx=20, ctx=ctx)

    held = set(out["symbol"].tolist())
    # All three remain (within Top 6); S1/S2 are Top-2 buy band but no capacity
    assert held == {"S3", "S4", "S5"}
    assert ctx.n_forced_rank_exits == 0
    assert ctx.n_new_entries == 0
    actions = {d["symbol"]: d["action"] for d in ctx.rebalance_decisions if d["action"] == "Hold"}
    assert set(actions) == {"S3", "S4", "S5"}


def test_forced_rank_exit_when_below_buffer():
    eng = _TinyEngine()
    eng.MAX_PORTFOLIO_SIZE = 5
    eng.ENTRY_RANK = 3
    eng.EXIT_RANK = 5
    eng.previous_target_symbols = {"S1", "S6"}  # S6 rank=6 > EXIT_RANK=5
    eng._previous_rank_map = {"S1": 1, "S6": 4}

    ranked = _ranked(10)
    ctx = RebalanceContext()
    out = eng.construct_portfolio(ranked, date_idx=20, ctx=ctx)

    held = set(out["symbol"].tolist())
    assert "S6" not in held
    assert "S1" in held
    assert ctx.n_forced_rank_exits == 1
    sell = [d for d in ctx.rebalance_decisions if d["symbol"] == "S6"][0]
    assert sell["action"] == "Sell"
    assert "dropped below" in sell["reason"].lower()


def test_new_entries_only_from_buy_band_and_capacity():
    eng = _TinyEngine()
    eng.MAX_PORTFOLIO_SIZE = 4
    eng.ENTRY_RANK = 3  # Top 3 only
    eng.EXIT_RANK = 8
    eng.previous_target_symbols = {"S2"}
    eng._previous_rank_map = {"S2": 1}

    ranked = _ranked(12)
    ctx = RebalanceContext()
    out = eng.construct_portfolio(ranked, date_idx=20, ctx=ctx)

    held = set(out["symbol"].tolist())
    # Keep S2; refill from Top 3 (S1, S3) — cannot reach size 4 from buy band alone
    assert "S2" in held
    assert held <= {"S1", "S2", "S3"}
    assert len(held) == 3
    assert ctx.n_new_entries == 2
    buys = [d for d in ctx.rebalance_decisions if d["action"] == "Buy"]
    assert {b["symbol"] for b in buys} == {"S1", "S3"}
    assert all("Top" in b["reason"] for b in buys)


def test_does_not_buy_rank_outside_entry_band_to_fill_capacity():
    """Vacant slots must NOT be filled with ranks 11–20 when ENTRY_RANK=10."""
    eng = _TinyEngine()
    eng.MAX_PORTFOLIO_SIZE = 5
    eng.ENTRY_RANK = 2
    eng.EXIT_RANK = 8
    eng.previous_target_symbols = set()

    ranked = _ranked(12)
    ctx = RebalanceContext()
    out = eng.construct_portfolio(ranked, date_idx=20, ctx=ctx)

    held = set(out["symbol"].tolist())
    assert held == {"S1", "S2"}
    assert len(held) < eng.MAX_PORTFOLIO_SIZE
