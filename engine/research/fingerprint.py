"""Experiment fingerprint Fact — environment identity for a research run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd="/workspace")
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_baseline_values(cfg: Dict[str, Any], engine) -> Dict[str, Any]:
    """Discover actual baseline knobs from config + live engine (do not assume)."""
    research = cfg.get("research") or {}
    return {
        "from_config": {
            "max_portfolio_size": cfg.get("max_portfolio_size"),
            "selection_buffer_size": cfg.get("selection_buffer_size"),
            "atr_multiplier": cfg.get("atr_multiplier"),
            "benchmark": cfg.get("benchmark"),
            "universe_sample_size": cfg.get("universe_sample_size"),
            "universe_sample_seed": cfg.get("universe_sample_seed"),
            "start_date": cfg.get("start_date"),
            "end_date": cfg.get("end_date"),
            "research_block": research,
        },
        "from_engine": {
            "USE_EMA9_EXIT": bool(getattr(engine, "USE_EMA9_EXIT", None)),
            "EMA_EXIT_LENGTH": int(getattr(engine, "EMA_EXIT_LENGTH", 9)),
            "USE_ATR_EXIT": bool(getattr(engine, "USE_ATR_EXIT", True)),
            "ATR_MULTIPLIER": float(getattr(engine, "atr_multiplier", float("nan"))),
            "USE_EXHAUSTION_EXIT": bool(getattr(engine, "USE_EXHAUSTION_EXIT", True)),
            "ENTRY_RANK": int(getattr(engine, "ENTRY_RANK", -1)),
            "EXIT_RANK": int(getattr(engine, "EXIT_RANK", -1)),
            "MIN_HOLD_DAYS": int(getattr(engine, "MIN_HOLD_DAYS", -1)),
            "USE_TIME_STOP": bool(getattr(engine, "USE_TIME_STOP", None)),
            "TIME_STOP_DAYS": int(getattr(engine, "TIME_STOP_DAYS", -1)),
            "USE_STOP_LOSS": bool(getattr(engine, "USE_STOP_LOSS", False)),
            "STOP_LOSS_PCT": float(getattr(engine, "STOP_LOSS_PCT", 0.0)),
            "MONTHLY_REBALANCE": bool(getattr(engine, "MONTHLY_REBALANCE", True)),
            "MAX_PORTFOLIO_SIZE": int(getattr(engine, "MAX_PORTFOLIO_SIZE", -1)),
            "MAX_INDUSTRY_WEIGHT": float(getattr(engine, "MAX_INDUSTRY_WEIGHT", float("nan"))),
            "BENCHMARK_TICKER": getattr(engine, "BENCHMARK_TICKER", None),
            "TOP_ADV_POOL": int(getattr(engine, "TOP_ADV_POOL", -1)),
            "CORR_THRESHOLD": float(getattr(engine, "CORR_THRESHOLD", float("nan"))),
        },
        "from_data": {
            "n_tickers": len(getattr(engine, "tickers", []) or []),
            "n_all_tickers": len(getattr(engine, "all_tickers", []) or []),
            "n_price_columns": int(engine.close_m.shape[1]) if hasattr(engine, "close_m") else None,
            "n_price_rows": int(engine.close_m.shape[0]) if hasattr(engine, "close_m") else None,
            "date_start": str(engine.close_m.index.min().date()) if hasattr(engine, "close_m") else None,
            "date_end": str(engine.close_m.index.max().date()) if hasattr(engine, "close_m") else None,
            "benchmark_in_panel": (
                getattr(engine, "BENCHMARK_TICKER", None) in engine.close_m.columns
                if hasattr(engine, "close_m")
                else None
            ),
        },
    }


def build_experiment_fingerprint(
    cfg: Dict[str, Any],
    engine,
    *,
    experiment_id: str,
    label: str,
    paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    paths = paths or {}
    meta_path = Path(paths.get("metadata", "data/metadata")) / "universe.pkl"
    prices_path = Path(paths.get("prices", "data/prices")) / "panels.pkl"
    cache_dir = Path(paths.get("cache", "cache"))
    massive_cache = Path(paths.get("massive_cache", "cache/massive_cache"))

    discovered = discover_baseline_values(cfg, engine)
    fp = {
        "experiment_id": experiment_id,
        "label": label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_values": {
            "research": (cfg.get("research") or {}),
            "max_portfolio_size": cfg.get("max_portfolio_size"),
            "selection_buffer_size": cfg.get("selection_buffer_size"),
            "atr_multiplier": cfg.get("atr_multiplier"),
            "benchmark": cfg.get("benchmark"),
            "start_date": cfg.get("start_date"),
            "end_date": cfg.get("end_date") or discovered["from_data"]["date_end"],
            "universe_sample_size": cfg.get("universe_sample_size"),
            "universe_sample_seed": cfg.get("universe_sample_seed"),
        },
        "engine_toggles": discovered["from_engine"],
        "universe_size": {
            "tickers": discovered["from_data"]["n_tickers"],
            "all_tickers": discovered["from_data"]["n_all_tickers"],
            "price_columns": discovered["from_data"]["n_price_columns"],
        },
        "date_range": {
            "start": discovered["from_data"]["date_start"],
            "end": discovered["from_data"]["date_end"],
            "n_rows": discovered["from_data"]["n_price_rows"],
        },
        "random_seed": cfg.get("universe_sample_seed"),
        "data_cache_version": {
            "universe_pkl_sha256": _file_sha256(meta_path),
            "panels_pkl_sha256": _file_sha256(prices_path),
            "universe_pkl_mtime": meta_path.stat().st_mtime if meta_path.exists() else None,
            "panels_pkl_mtime": prices_path.stat().st_mtime if prices_path.exists() else None,
            "massive_cache_dir": str(massive_cache),
            "massive_cache_file_count": (
                sum(1 for _ in massive_cache.rglob("*") if _.is_file()) if massive_cache.exists() else 0
            ),
            "cache_dir": str(cache_dir),
        },
        "discovered_baseline": discovered,
    }
    return fp


def write_fingerprint(fp: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp, indent=2, default=str), encoding="utf-8")
    return path
