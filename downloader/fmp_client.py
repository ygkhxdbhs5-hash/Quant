"""Shared FMP HTTP helpers (cache + throttle). Data plumbing only."""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def make_session(max_retries: int = 3) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class FmpClient:
    def __init__(
        self,
        api_key: str,
        cache_dir: str | Path,
        request_interval_sec: float = 0.25,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_interval_sec = request_interval_sec
        self.session = make_session(max_retries)
        self._last_call_ts = 0.0

    def throttled_get(self, url: str) -> Any:
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.request_interval_sec:
            time.sleep(self.request_interval_sec - elapsed)
        try:
            res = self.session.get(url, timeout=15)
            self._last_call_ts = time.time()
            if res.status_code != 200:
                return None
            try:
                return res.json()
            except Exception:
                return None
        except Exception:
            self._last_call_ts = time.time()
            return None

    def cached_get(self, url: str, cache_key: str, ttl_days: int = 7) -> Any:
        fname = hashlib.md5(cache_key.encode()).hexdigest() + ".pkl"
        fpath = self.cache_dir / fname
        if fpath.exists():
            if (time.time() - fpath.stat().st_mtime) / 86400 < ttl_days:
                with open(fpath, "rb") as f:
                    return pickle.load(f)
        data = self.throttled_get(url)
        if data is not None:
            with open(fpath, "wb") as f:
                pickle.dump(data, f)
        return data
