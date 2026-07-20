"""Observation-only rank-exit diagnostics (does not affect trading)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RankDiagnostics:
    """Collects three diagnostic series during a backtest."""

    exit_rank: float = 70.0
    rank_exit_candidates: int = 0
    ema_preempted_rank_exit: int = 0
    holding_ranks: List[float] = field(default_factory=list)

    def observe(
        self,
        *,
        date: Any,
        ticker: str,
        rank: Optional[float],
        ema9_break: bool,
    ) -> None:
        """Record one holding observation for one decision date.

        Counts ``rank > EXIT_RANK`` even if another exit rule also fired.
        ``date`` / ``ticker`` are accepted for call-site clarity; not stored.
        """
        del date, ticker  # observation aggregates only
        if rank is None:
            return
        r = float(rank)
        self.holding_ranks.append(r)
        if r > self.exit_rank:
            self.rank_exit_candidates += 1
            if ema9_break:
                self.ema_preempted_rank_exit += 1

    def holding_rank_distribution(self) -> Dict[str, Any]:
        """min, median, 75th / 90th / 95th percentiles, max."""
        ranks = sorted(self.holding_ranks)
        n = len(ranks)
        if n == 0:
            return {
                "n": 0,
                "min": None,
                "mean": None,
                "median": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "max": None,
            }

        def pct(p: float) -> float:
            if n == 1:
                return ranks[0]
            idx = (p / 100.0) * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            w = idx - lo
            return ranks[lo] * (1 - w) + ranks[hi] * w

        return {
            "n": n,
            "min": ranks[0],
            "mean": sum(ranks) / n,
            "median": pct(50),
            "p75": pct(75),
            "p90": pct(90),
            "p95": pct(95),
            "max": ranks[-1],
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "exit_rank": self.exit_rank,
            "rank_exit_candidates": self.rank_exit_candidates,
            "ema_preempted_rank_exit": self.ema_preempted_rank_exit,
            "holding_rank_distribution": self.holding_rank_distribution(),
        }

    def format_report(self) -> str:
        """End-of-backtest print block (facts only; no strategy advice)."""
        dist = self.holding_rank_distribution()
        lines = [
            "=" * 72,
            "RANK DIAGNOSTICS (observation only — trading unchanged)",
            "=" * 72,
            f"rank_exit_candidates: {self.rank_exit_candidates}",
            f"ema_preempted_rank_exit: {self.ema_preempted_rank_exit}",
            "holding_rank_distribution:",
        ]
        if dist["n"] == 0:
            lines.append("  (no holding-rank observations)")
        else:
            lines.extend(
                [
                    f"  min: {dist['min']:.4f}",
                    f"  median: {dist['median']:.4f}",
                    f"  p75: {dist['p75']:.4f}",
                    f"  p90: {dist['p90']:.4f}",
                    f"  p95: {dist['p95']:.4f}",
                    f"  max: {dist['max']:.4f}",
                    f"  n: {dist['n']}",
                ]
            )
        lines.append("=" * 72)
        return "\n".join(lines)

    def write_report(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_report(), encoding="utf-8")
        return path
