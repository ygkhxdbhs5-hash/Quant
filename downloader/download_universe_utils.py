"""Shared config helpers for downloaders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
import yaml

from downloader.fmp_client import FmpClient


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    env_key = os.environ.get("FMP_API_KEY")
    if env_key:
        cfg["fmp_api_key"] = env_key
    if not cfg.get("end_date"):
        cfg["end_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    return cfg


def make_client(config: dict) -> FmpClient:
    paths = config.get("paths", {})
    return FmpClient(
        api_key=config["fmp_api_key"],
        cache_dir=paths.get("fmp_cache", "cache/fmp_cache_v5"),
        request_interval_sec=float(config.get("request_interval_sec", 0.25)),
        max_retries=int(config.get("max_retries", 3)),
    )


def read_symbol_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
