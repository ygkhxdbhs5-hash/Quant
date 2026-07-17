"""Verify investment-logic methods match the original lean_engine_v3 source text."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.strategy import NasdaqInstitutionalProductionEngine, RebalanceContext


ORIGINAL = Path(__file__).resolve().parents[1] / "engine" / "_reference_lean_engine_v3.py"
CURRENT = Path(__file__).resolve().parents[1] / "engine" / "strategy.py"

LOGIC_METHODS = [
    "GetLatestAvailableFundamentals",
    "CalculateMarketBreadth",
    "_build_correlation_matrix",
    "DetermineMarketRegime",
    "BuildFactors",
    "RankUniverse",
    "ConstructPortfolio",
    "ApplyRiskAdjustments",
    "ValidatePortfolio",
    "ExecuteTrades",
    "LogRebalanceSummary",
    "RebalanceStrategy",
]


def _method_source(path: Path, method_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "NasdaqInstitutionalProductionEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    # Use unparsed body statements to ignore decorators/signatures whitespace
                    body = "\n".join(ast.unparse(stmt) for stmt in item.body)
                    return body
    raise AssertionError(f"{method_name} not found in {path}")


@pytest.mark.skipif(not ORIGINAL.exists(), reason="original upload not present")
@pytest.mark.parametrize("method_name", LOGIC_METHODS)
def test_logic_method_body_matches_original(method_name: str):
    """Structural plumbing may differ in Initialize; pipeline methods must stay identical."""
    # Methods that reference .Price vs .price were normalized back to .Price.
    # Coarse/Fine keep original selection predicates.
    original_body = _method_source(ORIGINAL, method_name)
    current_body = _method_source(CURRENT, method_name)

    # Allow Securities.Price attribute access (identical to original).
    # Normalize trivial whitespace via hashing of unparsed AST bodies.
    assert hashlib.sha256(current_body.encode()).hexdigest() == hashlib.sha256(original_body.encode()).hexdigest(), (
        f"{method_name} body diverged from original investment logic"
    )


def test_rank_and_risk_numeric_smoke():
    engine = NasdaqInstitutionalProductionEngine()
    engine.Initialize(
        {
            "start_date": [2020, 1, 1],
            "end_date": [2020, 12, 31],
            "initial_cash": 1_000_000,
            "max_portfolio_size": 5,
            "selection_buffer_size": 8,
        }
    )
    # Silence logs
    engine.Log = lambda msg: None

    df = pd.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(12)],
            "mom_raw": np.linspace(0.1, 0.5, 12),
            "vol_raw": np.linspace(0.01, 0.05, 12),
            "op_margin_raw": np.linspace(0.1, 0.3, 12),
            "roic_raw": np.linspace(0.1, 0.3, 12),
            "gross_prof_raw": np.linspace(0.1, 0.3, 12),
            "fcf_yield_raw": np.linspace(0.01, 0.08, 12),
            "sales_yield_raw": np.linspace(0.05, 0.2, 12),
            "industry": ["A"] * 6 + ["B"] * 6,
        }
    )
    ranked = engine.RankUniverse(df)
    assert list(ranked.columns)
    assert ranked["final_score"].is_monotonic_decreasing or ranked["final_score"].iloc[0] >= ranked["final_score"].iloc[-1]

    ctx = RebalanceContext()
    # Provide empty correlation path by stubbing indicators for selected symbols
    engine.indicators = {
        sym: {"price_window": type("W", (), {"Count": 0})()} for sym in ranked["symbol"]
    }
    targets = engine.ConstructPortfolio(ranked, ctx)
    sized = engine.ApplyRiskAdjustments(targets if not targets.empty else ranked.head(5), exposure=1.0)
    assert pytest.approx(sized["final_weight"].sum(), abs=1e-8) == 1.0
    assert (sized["final_weight"] > 0).all()
