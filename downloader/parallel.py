"""Small helpers for parallel Massive downloads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_parallel(
    items: Iterable[T],
    fn: Callable[[T], R],
    workers: int = 8,
    progress_every: int = 10,
    label: str = "items",
) -> List[R]:
    """Run ``fn`` over items with a thread pool; preserve input order in results."""
    seq = list(items)
    n = len(seq)
    if n == 0:
        return []
    workers = max(1, int(workers))
    if workers == 1 or n == 1:
        out: List[R] = []
        for i, item in enumerate(seq, 1):
            out.append(fn(item))
            if progress_every and (i % progress_every == 0 or i == n):
                print(f"    ... {label} {i}/{n}", flush=True)
        return out

    print(f"    [parallel] {label}: {n} items, workers={workers}", flush=True)
    results: List[Optional[R]] = [None] * n
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): idx for idx, item in enumerate(seq)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            done += 1
            if progress_every and (done % progress_every == 0 or done == n):
                print(f"    ... {label} {done}/{n}", flush=True)
    return results  # type: ignore[return-value]
