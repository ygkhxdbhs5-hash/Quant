"""Reproducibility fingerprint for diagnostic reports / charts.

Embeds a short hash of code + config + window + data cache identity so text
and chart artifacts can be checked for consistency at a glance.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]

# Files that affect report numerics (strategy entry/exit NOT required for
# fingerprint identity of cost/chase reports, but included for safety).
FINGERPRINT_PATHS = [
    "engine/baseline_engine.py",
    "engine/strategy_baseline_v1.py",
    "engine/strategy_baseline_v1_quality.py",
    "engine/pit_fundamentals.py",
    "engine/regime_exposure.py",
    "engine/regime_exposure_fast.py",
    "engine/strategy_baseline_v1_indneutral.py",
    "engine/execution_costs.py",
    "engine/topup_chase_diagnostics.py",
    "engine/report_charts.py",
    "engine/cost_model_audit.py",
    "engine/kelly_leverage.py",
    "engine/strategy_s2_meanrev.py",
    "engine/strategy2_engine.py",
    "engine/s2_family_signals.py",
    "engine/s2_family_engine.py",
    "engine/s2_family_data.py",
    "run_topup_chase_report.py",
    "run_kelly_leverage_report.py",
    "run_strategy2_meanrev_report.py",
    "run_s2_family_search_20.py",
    "config/config.yaml",
]


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return False


def build_repro_fingerprint(
    *,
    start: str,
    end: str,
    config: Optional[dict] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return fingerprint dict + short id for footers / filenames."""
    cfg = dict(config or {})
    # Only keys that affect numerics of these reports
    cfg_slice = {
        "start_date": start,
        "end_date": end,
        "initial_cash": cfg.get("initial_cash"),
        "cost_model": cfg.get("cost_model"),
        "winsorize_adv": cfg.get("winsorize_adv"),
        "enable_topup_chasing": cfg.get("enable_topup_chasing"),
        "disable_topup_chasing": cfg.get("disable_topup_chasing"),
        "flat_cost_one_way": cfg.get("flat_cost_one_way"),
        "participation_cap_buy": cfg.get("participation_cap_buy"),
        "participation_cap_sell": cfg.get("participation_cap_sell"),
        "commission_rate": cfg.get("commission_rate"),
        "slippage_rate": cfg.get("slippage_rate"),
        "universe_sample_size": cfg.get("universe_sample_size"),
        "universe_sample_seed": cfg.get("universe_sample_seed"),
        "baseline_v1": cfg.get("baseline_v1"),
    }

    file_hashes = {}
    for rel in FINGERPRINT_PATHS:
        p = ROOT / rel
        if p.exists():
            file_hashes[rel] = _file_sha256(p)[:16]

    panels = ROOT / "data" / "prices" / "panels.pkl"
    universe = ROOT / "data" / "metadata" / "universe.pkl"
    data_hashes = {
        "panels.pkl": _file_sha256(panels)[:16] if panels.exists() else None,
        "universe.pkl": _file_sha256(universe)[:16] if universe.exists() else None,
        "panels_bytes": panels.stat().st_size if panels.exists() else None,
        "universe_bytes": universe.stat().st_size if universe.exists() else None,
    }

    payload = {
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "window": {"start": str(start), "end": str(end)},
        "config_slice": cfg_slice,
        "code_file_sha256_16": file_hashes,
        "data_cache": data_hashes,
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    full = hashlib.sha256(blob).hexdigest()
    short = full[:12]
    payload["fingerprint_sha256"] = full
    payload["fingerprint_id"] = short
    return payload


def format_fingerprint_banner(fp: Dict[str, Any]) -> str:
    return (
        f"repro_id={fp.get('fingerprint_id')}  "
        f"git={str(fp.get('git_head', ''))[:10]}"
        f"{'*' if fp.get('git_dirty') else ''}  "
        f"window={fp.get('window', {}).get('start')}→{fp.get('window', {}).get('end')}  "
        f"panels={((fp.get('data_cache') or {}).get('panels.pkl'))}"
    )
