"""Research configuration toggles.

Defaults are chosen to reproduce the current CMVS engine trade identity
(Core Research Principle). Spec examples ENTRY_RANK=30 / EXIT_RANK=80 are
NOT used as silent defaults because they would change baseline vs config
max_portfolio_size=50 / selection_buffer_size=70 — see docs/ENGINE_ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ResearchToggles:
    USE_EMA9_EXIT: bool = True
    ATR_MULTIPLIER: float = 2.0
    ENTRY_RANK: int = 50  # baseline: max_portfolio_size (spec example 30 would alter trades)
    EXIT_RANK: int = 70  # baseline: selection_buffer_size (spec example 80 would alter trades)
    MIN_HOLD_DAYS: int = 0
    USE_TIME_STOP: bool = False
    TIME_STOP_DAYS: int = 20  # only used when USE_TIME_STOP=True
    SHADOW_HORIZON_DAYS: int = 20

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_baseline_defaults(self, max_portfolio_size: int = 50, selection_buffer_size: int = 70) -> bool:
        return (
            self.USE_EMA9_EXIT is True
            and float(self.ATR_MULTIPLIER) == 2.0
            and int(self.ENTRY_RANK) == int(max_portfolio_size)
            and int(self.EXIT_RANK) == int(selection_buffer_size)
            and int(self.MIN_HOLD_DAYS) == 0
            and self.USE_TIME_STOP is False
        )


def load_research_toggles(cfg: Dict[str, Any]) -> ResearchToggles:
    """Load toggles from config; missing keys fall back to baseline-identical defaults."""
    max_n = int(cfg.get("max_portfolio_size", 50))
    buf_n = int(cfg.get("selection_buffer_size", 70))
    research = cfg.get("research") or {}

    # Prefer nested research.*; also accept flat keys for convenience.
    def _get(key: str, default):
        if key in research:
            return research[key]
        # flat aliases
        flat_map = {
            "USE_EMA9_EXIT": "use_ema9_exit",
            "ATR_MULTIPLIER": "atr_multiplier",
            "ENTRY_RANK": "entry_rank",
            "EXIT_RANK": "exit_rank",
            "MIN_HOLD_DAYS": "min_hold_days",
            "USE_TIME_STOP": "use_time_stop",
            "TIME_STOP_DAYS": "time_stop_days",
            "SHADOW_HORIZON_DAYS": "shadow_horizon_days",
        }
        flat = flat_map.get(key, key.lower())
        if flat in cfg:
            return cfg[flat]
        return default

    return ResearchToggles(
        USE_EMA9_EXIT=bool(_get("USE_EMA9_EXIT", True)),
        ATR_MULTIPLIER=float(_get("ATR_MULTIPLIER", cfg.get("atr_multiplier", 2.0))),
        ENTRY_RANK=int(_get("ENTRY_RANK", max_n)),
        EXIT_RANK=int(_get("EXIT_RANK", buf_n)),
        MIN_HOLD_DAYS=int(_get("MIN_HOLD_DAYS", 0)),
        USE_TIME_STOP=bool(_get("USE_TIME_STOP", False)),
        TIME_STOP_DAYS=int(_get("TIME_STOP_DAYS", 20)),
        SHADOW_HORIZON_DAYS=int(_get("SHADOW_HORIZON_DAYS", 20)),
    )
