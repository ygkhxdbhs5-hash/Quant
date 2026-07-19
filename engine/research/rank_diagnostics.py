"""Observation-only rank-exit diagnostics (does not affect trading)."""

from __future__ import annotations

from collections import Counter
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
    # Optional detail rows for debugging (not required for summary).
    candidate_events: List[Dict[str, Any]] = field(default_factory=list)

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
        """
        if rank is None:
            return
        r = float(rank)
        self.holding_ranks.append(r)
        if r > self.exit_rank:
            self.rank_exit_candidates += 1
            if ema9_break:
                self.ema_preempted_rank_exit += 1
            self.candidate_events.append(
                {
                    "date": str(date)[:10],
                    "ticker": ticker,
                    "rank": r,
                    "ema9_break": bool(ema9_break),
                }
            )

    def percentile_summary(self) -> Dict[str, Any]:
        ranks = sorted(self.holding_ranks)
        n = len(ranks)
        if n == 0:
            return {"n": 0}

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
            "p10": pct(10),
            "p25": pct(25),
            "p50": pct(50),
            "p75": pct(75),
            "p90": pct(90),
            "p95": pct(95),
            "p99": pct(99),
            "max": ranks[-1],
            "mean": sum(ranks) / n,
            "gt_exit_rank_share": sum(1 for r in ranks if r > self.exit_rank) / n,
        }

    def histogram(self, bins: Optional[List[float]] = None) -> Dict[str, int]:
        """Histogram of holding ranks. Default bins suited to rank integers."""
        if not self.holding_ranks:
            return {}
        if bins is None:
            # Adaptive bins from observed min/max (closed on right).
            mx = max(self.holding_ranks)
            if mx <= 20:
                edges = list(range(0, int(mx) + 2))
            elif mx <= 100:
                edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, mx + 1]
            else:
                step = 25
                edges = list(range(0, int(mx) + step, step))
                if edges[-1] <= mx:
                    edges.append(mx + 1)
        else:
            edges = bins

        counts: Counter = Counter()
        for r in self.holding_ranks:
            placed = False
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                if lo < r <= hi or (i == 0 and r == lo):
                    key = f"({lo},{hi}]" if r != lo or i > 0 else f"[{lo},{hi}]"
                    # Simpler label:
                    key = f"{lo}<r<={hi}"
                    counts[key] += 1
                    placed = True
                    break
            if not placed:
                counts["other"] += 1
        # Stable order by bin start
        ordered = {}
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            key = f"{lo}<r<={hi}"
            ordered[key] = int(counts.get(key, 0))
        if counts.get("other"):
            ordered["other"] = int(counts["other"])
        return ordered

    def interpretation(self) -> List[str]:
        """Facts only — no strategy recommendations."""
        lines: List[str] = []
        pct = self.percentile_summary()
        n = int(pct.get("n", 0))
        lines.append(
            f"EXIT_RANK threshold used for diagnostics: {self.exit_rank:g}."
        )
        lines.append(
            f"Holding-day rank observations recorded: {n}."
        )
        lines.append(
            f"rank_exit_candidates (rank > EXIT_RANK): {self.rank_exit_candidates}."
        )
        lines.append(
            f"ema_preempted_rank_exit (rank > EXIT_RANK and ema9_break): "
            f"{self.ema_preempted_rank_exit}."
        )
        if n == 0:
            lines.append("No holding-rank observations were recorded.")
            return lines

        lines.append(
            "Holding-rank percentile summary: "
            f"min={pct['min']:.2f}, p25={pct['p25']:.2f}, p50={pct['p50']:.2f}, "
            f"p75={pct['p75']:.2f}, p90={pct['p90']:.2f}, max={pct['max']:.2f}, "
            f"mean={pct['mean']:.2f}."
        )
        share = float(pct["gt_exit_rank_share"])
        lines.append(
            f"Share of holding-day ranks with rank > EXIT_RANK: {share:.2%}."
        )

        if self.rank_exit_candidates == 0:
            lines.append(
                "Observed zero rank_exit_candidates: no held position had "
                f"rank > {self.exit_rank:g} on any decision date in this run."
            )
        else:
            preempt_share = (
                self.ema_preempted_rank_exit / self.rank_exit_candidates
            )
            lines.append(
                "Among rank_exit_candidates, share that also had ema9_break "
                f"on the same decision date: {preempt_share:.2%} "
                f"({self.ema_preempted_rank_exit}/{self.rank_exit_candidates})."
            )
            lines.append(
                "Among rank_exit_candidates, share without ema9_break on that "
                f"date: {1.0 - preempt_share:.2%} "
                f"({self.rank_exit_candidates - self.ema_preempted_rank_exit}/"
                f"{self.rank_exit_candidates})."
            )

        if self.ema_preempted_rank_exit == 0 and self.rank_exit_candidates > 0:
            lines.append(
                "Observed zero ema_preempted_rank_exit: no day had both "
                "rank > EXIT_RANK and ema9_break for the same holding."
            )
        elif self.ema_preempted_rank_exit == 0 and self.rank_exit_candidates == 0:
            lines.append(
                "Observed zero ema_preempted_rank_exit (no rank_exit_candidates "
                "to overlap with ema9_break)."
            )

        return lines

    def summary(self) -> Dict[str, Any]:
        return {
            "exit_rank": self.exit_rank,
            "rank_exit_candidates": self.rank_exit_candidates,
            "ema_preempted_rank_exit": self.ema_preempted_rank_exit,
            "rank_distribution_while_holding": {
                "percentile_summary": self.percentile_summary(),
                "histogram": self.histogram(),
            },
            "interpretation": self.interpretation(),
        }

    def format_report(self) -> str:
        s = self.summary()
        hist = s["rank_distribution_while_holding"]["histogram"]
        pct = s["rank_distribution_while_holding"]["percentile_summary"]
        lines = [
            "=" * 72,
            "RANK DIAGNOSTICS (observation only — trading unchanged)",
            "=" * 72,
            f"EXIT_RANK: {s['exit_rank']:g}",
            f"Total rank_exit_candidates: {s['rank_exit_candidates']}",
            f"Total ema_preempted_rank_exit: {s['ema_preempted_rank_exit']}",
            "",
            "rank_distribution_while_holding — percentile summary:",
        ]
        if pct.get("n", 0) == 0:
            lines.append("  (no observations)")
        else:
            for k in (
                "n",
                "min",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
                "mean",
                "gt_exit_rank_share",
            ):
                v = pct[k]
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("rank_distribution_while_holding — histogram:")
        if not hist:
            lines.append("  (empty)")
        else:
            for bucket, count in hist.items():
                lines.append(f"  {bucket}: {count}")
        lines.append("")
        lines.append("Interpretation (from observed data only):")
        for fact in s["interpretation"]:
            lines.append(f"  - {fact}")
        lines.append("=" * 72)
        return "\n".join(lines)

    def write_report(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.format_report()
        path.write_text(text, encoding="utf-8")
        return path
