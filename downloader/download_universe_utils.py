"""Shared config helpers for Massive downloaders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
import yaml

from downloader.massive_client import MassiveClient


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    env_key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    if env_key:
        cfg["massive_api_key"] = env_key
    if not cfg.get("end_date"):
        cfg["end_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    return cfg


def make_client(config: dict) -> MassiveClient:
    paths = config.get("paths", {})
    workers = max(1, int(config.get("download_workers", 8)))
    return MassiveClient(
        api_key=str(config.get("massive_api_key") or ""),
        cache_dir=paths.get("massive_cache", paths.get("fmp_cache", "cache/massive_cache")),
        base_url=str(config.get("massive_base_url", "https://api.massive.com")),
        request_interval_sec=float(config.get("request_interval_sec", 0.10)),
        max_retries=int(config.get("max_retries", 3)),
        pool_size=max(16, workers * 4),
    )


def read_symbol_file(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
