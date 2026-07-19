"""Research & experimentation layer (counterfactuals, KPIs, recommendations).

Kept separate from the core execution engine so analytical tools can grow
without changing trade generation at baseline defaults.
"""

from engine.research.config_toggles import ResearchToggles, load_research_toggles
from engine.research.trade_journal import TradeJournal
from engine.research.kpi_report import build_hierarchical_kpi_report
from engine.research.recommendations import build_research_recommendation_report
from engine.research.experiment_history import ExperimentHistory
from engine.research.validation import run_research_validation_checklist

__all__ = [
    "ResearchToggles",
    "load_research_toggles",
    "TradeJournal",
    "build_hierarchical_kpi_report",
    "build_research_recommendation_report",
    "ExperimentHistory",
    "run_research_validation_checklist",
]
