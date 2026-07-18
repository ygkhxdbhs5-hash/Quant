"""Massive.com (ex-Polygon) HTTP client with disk cache + throttle."""

from __future__ import annotations

import hashlib
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class MassiveClient:
    """Thin REST client for https://api.massive.com.

    Thread-safe enough for parallel downloaders: request slots are reserved
    under a lock so multiple workers can have HTTP calls in flight while still
    respecting ``request_interval_sec``.
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: str | Path,
        base_url: str = "https://api.massive.com",
        request_interval_sec: float = 0.10,
        max_retries: int = 3,
        pool_size: int = 32,
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
        # urllib3 pool is thread-safe; size it for parallel workers so HTTP can overlap.
        pool = max(8, int(pool_size))
        adapter = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            }
        )
        self._next_slot_ts = 0.0
        self._throttle_lock = threading.Lock()
        self._cache_lock = threading.Lock()

    def _throttle(self) -> None:
        """Reserve the next request slot, then sleep outside the lock."""
        with self._throttle_lock:
            now = time.time()
            slot = max(now, self._next_slot_ts)
            self._next_slot_ts = slot + max(0.0, float(self.request_interval_sec))
            wait = slot - now
        if wait > 0:
            time.sleep(wait)

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """GET a relative path or absolute next_url; returns parsed JSON or None."""
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        last_status = None
        for attempt in range(5):
            self._throttle()
            try:
                # No session lock: connection pool handles concurrency; throttle
                # staggers starts so workers overlap on network wait, not bursts.
                res = self.session.get(url, params=params, timeout=60)
                last_status = res.status_code
                if res.status_code == 200:
                    try:
                        return res.json()
                    except Exception:
                        return None
                if res.status_code in {429, 500, 502, 503, 504}:
                    sleep_s = min(2.0 ** attempt, 20.0)
                    print(
                        f"[massive] HTTP {res.status_code} attempt={attempt+1}/5 "
                        f"sleep={sleep_s:.1f}s url={url[:120]}",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                    continue
                print(f"[massive] HTTP {res.status_code} giving up url={url[:120]}", flush=True)
                return None
            except Exception as exc:
                sleep_s = min(2.0 ** attempt, 20.0)
                print(
                    f"[massive] request error {type(exc).__name__}: {exc}; sleep={sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
        print(
            f"[massive] failed after retries last_status={last_status} url={url[:120]}",
            flush=True,
        )
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
        with self._cache_lock:
            if fpath.exists():
                if (time.time() - fpath.stat().st_mtime) / 86400 < ttl_days:
                    with open(fpath, "rb") as f:
                        cached = pickle.load(f)
                    if isinstance(cached, dict):
                        results = cached.get("results")
                        if results is None or (isinstance(results, list) and len(results) == 0):
                            fpath.unlink(missing_ok=True)
                        else:
                            return cached
                    else:
                        return cached

        data = self.get_json(path, params=params)

        with self._cache_lock:
            if isinstance(data, dict) and data.get("results"):
                with open(fpath, "wb") as f:
                    pickle.dump(data, f)
            elif data is not None and not (isinstance(data, dict) and "results" in data):
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
        verbose: bool = True,
    ) -> list:
        """Follow next_url pagination; accumulates ``results`` arrays."""
        params = dict(params or {})
        page = 0
        out: list = []
        next_path: Optional[str] = path
        next_params: Optional[Dict[str, Any]] = params
        if verbose:
            print(
                f"[paginate] start path={path!r} max_pages={max_pages} "
                f"params={params!r} cache_prefix={cache_key_prefix!r}",
                flush=True,
            )
        while next_path and page < max_pages:
            key = f"{cache_key_prefix}_p{page}_{urlencode(sorted((next_params or {}).items()))}"
            payload = self.cached_get(next_path, key, params=next_params, ttl_days=ttl_days)
            if not isinstance(payload, dict):
                if verbose:
                    print(
                        f"[paginate] Page {page} returned non-dict payload "
                        f"({type(payload).__name__}); stopping. accumulated={len(out)}",
                        flush=True,
                    )
                break
            results = payload.get("results") or []
            n_results = len(results) if isinstance(results, list) else 0
            if isinstance(results, list):
                out.extend(results)
            next_url = payload.get("next_url")
            if verbose:
                print(
                    f"Page {page} returned {n_results} items "
                    f"(accumulated={len(out)}, next_url={'yes' if next_url else 'no'})",
                    flush=True,
                )
            page += 1
            if not next_url:
                next_path = None
                break
            next_path = next_url
            next_params = None
        if next_path and page >= max_pages and verbose:
            print(
                f"[paginate] WARNING: hit max_pages={max_pages}; "
                f"there may be more results. accumulated={len(out)}",
                flush=True,
            )
        if verbose:
            print(f"[paginate] done pages_fetched={page} total_rows={len(out)}", flush=True)
        return out
