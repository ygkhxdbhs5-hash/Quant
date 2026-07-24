"""Research Configuration Panel.

Single place to edit research parameters for experiments. Defaults reproduce
the current CMVS engine trade identity (Core Research Principle).

Edit values here or under ``research:`` in ``config/config.yaml``.
The engine reads these before each run — no code edits required for experiments.

Spec examples ENTRY_RANK=30 / EXIT_RANK=80 are NOT silent defaults; they would
change baseline vs max_portfolio_size=50 / selection_buffer_size=70.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ResearchToggles:
    """Research Configuration Panel — all experiment knobs in one object."""

    # --- Entry / Exit ranks ---
    ENTRY_RANK: int = 50  # baseline: max_portfolio_size
    EXIT_RANK: int = 70  # baseline: selection_buffer_size

    # --- EMA / trend exit (USE_EMA9_EXIT enables confirmation-based trend exit) ---
    # EMA9 pierce alone never sells; needs multi-signal confirmation.
    USE_EMA9_EXIT: bool = True
    EMA_EXIT_LENGTH: int = 9  # short EMA used in confirmation logic

    # --- ATR trailing exit ---
    USE_ATR_EXIT: bool = True
    ATR_MULTIPLIER: float = 2.0

    # --- RSI exhaustion exit ---
    USE_EXHAUSTION_EXIT: bool = True

    # --- Time stop ---
    USE_TIME_STOP: bool = False
    TIME_STOP_DAYS: int = 20  # only used when USE_TIME_STOP=True

    # --- Holding ---
    MIN_HOLD_DAYS: int = 0

    # --- Portfolio ---
    MAX_PORTFOLIO_SIZE: int = 50  # mirrored from ENTRY_RANK at load when unset
    MAX_INDUSTRY_WEIGHT: float = 0.40

    # --- Rebalance ---
    MONTHLY_REBALANCE: bool = True  # False → daily rebalance (research mode)

    # --- Observation / research tooling (does not affect trades) ---
    SHADOW_HORIZON_DAYS: int = 20

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def panel_lines(self) -> List[str]:
        """Lines for the start-of-backtest RESEARCH CONFIGURATION printout."""
        return [
            "ENTRY_RANK = " + str(self.ENTRY_RANK),
            "EXIT_RANK = " + str(self.EXIT_RANK),
            "",
            "USE_EMA9_EXIT = " + str(self.USE_EMA9_EXIT),
            "EMA_EXIT_LENGTH = " + str(self.EMA_EXIT_LENGTH),
            "",
            "USE_ATR_EXIT = " + str(self.USE_ATR_EXIT),
            "ATR_MULTIPLIER = " + str(float(self.ATR_MULTIPLIER)),
            "",
            "USE_EXHAUSTION_EXIT = " + str(self.USE_EXHAUSTION_EXIT),
            "",
            "USE_TIME_STOP = " + str(self.USE_TIME_STOP),
            "TIME_STOP_DAYS = " + str(self.TIME_STOP_DAYS),
            "",
            "MIN_HOLD_DAYS = " + str(self.MIN_HOLD_DAYS),
            "",
            "MAX_PORTFOLIO_SIZE = " + str(self.MAX_PORTFOLIO_SIZE),
            "",
            "MONTHLY_REBALANCE = " + str(self.MONTHLY_REBALANCE),
            "",
            "MAX_INDUSTRY_WEIGHT = " + f"{float(self.MAX_INDUSTRY_WEIGHT):.2f}",
        ]

    def format_panel(self) -> str:
        lines = [
            "=" * 50,
            "RESEARCH CONFIGURATION",
            "=" * 50,
            *self.panel_lines(),
            "=" * 50,
        ]
        return "\n".join(lines)

    def is_baseline_defaults(
        self, max_portfolio_size: int = 50, selection_buffer_size: int = 70
    ) -> bool:
        return (
            self.USE_EMA9_EXIT is True
            and int(self.EMA_EXIT_LENGTH) == 9
            and self.USE_ATR_EXIT is True
            and float(self.ATR_MULTIPLIER) == 2.0
            and self.USE_EXHAUSTION_EXIT is True
            and int(self.ENTRY_RANK) == int(max_portfolio_size)
            and int(self.EXIT_RANK) == int(selection_buffer_size)
            and int(self.MIN_HOLD_DAYS) == 0
            and self.USE_TIME_STOP is False
            and self.MONTHLY_REBALANCE is True
        )


def load_research_toggles(cfg: Dict[str, Any]) -> ResearchToggles:
    """Load toggles from config; missing keys fall back to baseline-identical defaults."""
    max_n = int(cfg.get("max_portfolio_size", 50))
    buf_n = int(cfg.get("selection_buffer_size", 70))
    industry_w = float(cfg.get("max_industry_weight", 0.40))
    research = cfg.get("research") or {}

    # Prefer nested research.*; also accept flat keys for convenience.
    def _get(key: str, default):
        if key in research:
            return research[key]
        flat_map = {
            "USE_EMA9_EXIT": "use_ema9_exit",
            "EMA_EXIT_LENGTH": "ema_exit_length",
            "USE_ATR_EXIT": "use_atr_exit",
            "ATR_MULTIPLIER": "atr_multiplier",
            "USE_EXHAUSTION_EXIT": "use_exhaustion_exit",
            "ENTRY_RANK": "entry_rank",
            "EXIT_RANK": "exit_rank",
            "MIN_HOLD_DAYS": "min_hold_days",
            "USE_TIME_STOP": "use_time_stop",
            "TIME_STOP_DAYS": "time_stop_days",
            "MAX_PORTFOLIO_SIZE": "max_portfolio_size",
            "MAX_INDUSTRY_WEIGHT": "max_industry_weight",
            "MONTHLY_REBALANCE": "monthly_rebalance",
            "SHADOW_HORIZON_DAYS": "shadow_horizon_days",
        }
        flat = flat_map.get(key, key.lower())
        if flat in cfg:
            return cfg[flat]
        return default

    entry_rank = int(_get("ENTRY_RANK", max_n))
    return ResearchToggles(
        ENTRY_RANK=entry_rank,
        EXIT_RANK=int(_get("EXIT_RANK", buf_n)),
        USE_EMA9_EXIT=bool(_get("USE_EMA9_EXIT", True)),
        EMA_EXIT_LENGTH=int(_get("EMA_EXIT_LENGTH", 9)),
        USE_ATR_EXIT=bool(_get("USE_ATR_EXIT", True)),
        ATR_MULTIPLIER=float(_get("ATR_MULTIPLIER", cfg.get("atr_multiplier", 2.0))),
        USE_EXHAUSTION_EXIT=bool(_get("USE_EXHAUSTION_EXIT", True)),
        USE_TIME_STOP=bool(_get("USE_TIME_STOP", False)),
        TIME_STOP_DAYS=int(_get("TIME_STOP_DAYS", 20)),
        MIN_HOLD_DAYS=int(_get("MIN_HOLD_DAYS", 0)),
        MAX_PORTFOLIO_SIZE=int(_get("MAX_PORTFOLIO_SIZE", entry_rank)),
        MAX_INDUSTRY_WEIGHT=float(_get("MAX_INDUSTRY_WEIGHT", industry_w)),
        MONTHLY_REBALANCE=bool(_get("MONTHLY_REBALANCE", True)),
        SHADOW_HORIZON_DAYS=int(_get("SHADOW_HORIZON_DAYS", 20)),
    )
