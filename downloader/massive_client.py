"""Massive.com (ex-Polygon) HTTP client with disk cache + throttle."""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class MassiveClient:
    """Thin REST client for https://api.massive.com."""

    def __init__(
        self,
        api_key: str,
        cache_dir: str | Path,
        base_url: str = "https://api.massive.com",
        request_interval_sec: float = 0.20,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Massive API key missing. Set MASSIVE_API_KEY or config.massive_api_key."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_interval_sec = request_interval_sec
        self.session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
        )
        self._last_call_ts = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.request_interval_sec:
            time.sleep(self.request_interval_sec - elapsed)

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET a relative path or absolute next_url; returns parsed JSON or None."""
        self._throttle()
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        try:
            res = self.session.get(url, params=params, timeout=30)
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

    def cached_get(
        self,
        path: str,
        cache_key: str,
        params: Optional[Dict[str, Any]] = None,
        ttl_days: int = 7,
    ) -> Any:
        fname = hashlib.md5(cache_key.encode()).hexdigest() + ".pkl"
        fpath = self.cache_dir / fname
        if fpath.exists():
            if (time.time() - fpath.stat().st_mtime) / 86400 < ttl_days:
                with open(fpath, "rb") as f:
                    return pickle.load(f)
        data = self.get_json(path, params=params)
        if data is not None:
            with open(fpath, "wb") as f:
                pickle.dump(data, f)
        return data

    def paginate(
        self,
        path: str,
        cache_key_prefix: str,
        params: Optional[Dict[str, Any]] = None,
        ttl_days: int = 7,
        max_pages: int = 100,
    ) -> list:
        """Follow next_url pagination; accumulates ``results`` arrays."""
        params = dict(params or {})
        page = 0
        out: list = []
        next_path: Optional[str] = path
        next_params: Optional[Dict[str, Any]] = params
        while next_path and page < max_pages:
            key = f"{cache_key_prefix}_p{page}_{urlencode(sorted((next_params or {}).items()))}"
            payload = self.cached_get(next_path, key, params=next_params, ttl_days=ttl_days)
            if not isinstance(payload, dict):
                break
            results = payload.get("results") or []
            if isinstance(results, list):
                out.extend(results)
            next_url = payload.get("next_url")
            if not next_url:
                break
            # next_url is absolute and already includes query params / cursor
            next_path = next_url
            next_params = None
            page += 1
        return out
