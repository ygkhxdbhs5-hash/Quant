"""Experiment History ledger (append-only Facts)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentHistory:
    def __init__(self, path: str | Path = "cache/experiment_history.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        if self.path.exists():
            try:
                self.records = json.loads(self.path.read_text(encoding="utf-8")) or []
            except Exception:
                self.records = []

    def append(
        self,
        *,
        parent_exp: Optional[str],
        changed_variables: Dict[str, Any],
        hypothesis: str,
        expected_outcome: str,
        actual_outcome: Optional[str] = None,
        decision: str = "pending",
        metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> str:
        exp_id = f"EXP-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
        rec = {
            "id": exp_id,
            "parent_exp": parent_exp,
            "changed_variables": changed_variables,
            "hypothesis": hypothesis,
            "expected_outcome": expected_outcome,
            "actual_outcome": actual_outcome,
            "decision": decision,
            "metrics": metrics or {},
            "notes": notes,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.records.append(rec)
        self.save()
        return exp_id

    def save(self) -> None:
        self.path.write_text(json.dumps(self.records, indent=2, default=str), encoding="utf-8")

    def latest(self) -> Optional[Dict[str, Any]]:
        return self.records[-1] if self.records else None
