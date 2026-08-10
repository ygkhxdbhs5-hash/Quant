"""Pull Massive stock snapshots (live/delayed) and merge into local panels.

Uses the same Massive client + universe/panels layout as the historical
downloader. Does **not** place trades.

Writes:
  data/prices/realtime_snapshot.pkl   — raw/normalized snapshot table
  data/prices/panels.pkl              — optional merge of today's live day bar
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from downloader.download_universe_utils import load_config, make_client
from downloader.massive_client import MassiveClient


SNAPSHOT_PATH = "realtime_snapshot.pkl"


def _ny_session_date(ts_ns: Optional[int] = None) -> pd.Timestamp:
    """Normalize a timestamp (ns) or 'now' to a NY session calendar date."""
    if ts_ns is not None and ts_ns > 0:
        # Massive may return ns or ms; treat large values as ns.
        unit = "ns" if ts_ns > 10_000_000_000_000 else "ms"
        ts = pd.to_datetime(ts_ns, unit=unit, utc=True)
    else:
        ts = pd.Timestamp.now(tz="UTC")
    return ts.tz_convert("America/New_York").normalize().tz_localize(None)


def _chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _parse_snapshot_row(item: dict, as_of: pd.Timestamp) -> Optional[dict]:
    if not isinstance(item, dict):
        return None
    symbol = str(item.get("ticker") or "").upper().strip()
    if not symbol:
        return None

    day = item.get("day") or {}
    prev = item.get("prevDay") or {}
    last_trade = item.get("lastTrade") or {}
    last_quote = item.get("lastQuote") or {}
    minute = item.get("min") or {}

    open_ = day.get("o")
    high = day.get("h")
    low = day.get("l")
    close = day.get("c")
    volume = day.get("v")

    # Prefer last trade as the live mark when available
    last_price = last_trade.get("p")
    if last_price is None:
        # bid/ask midpoint fallback
        bid = last_quote.get("p")
        ask = last_quote.get("P")
        if bid is not None and ask is not None:
            last_price = (float(bid) + float(ask)) / 2.0
    if close is None:
        close = last_price
    if last_price is None:
        last_price = close

    if close is None and prev.get("c") is not None:
        # Pre-market / empty day — fall back to previous close for continuity
        close = prev.get("c")
        last_price = last_price if last_price is not None else close
        open_ = open_ if open_ is not None else close
        high = high if high is not None else close
        low = low if low is not None else close
        volume = volume if volume is not None else 0

    if close is None:
        return None

    open_ = float(open_ if open_ is not None else close)
    high = float(high if high is not None else max(open_, float(close)))
    low = float(low if low is not None else min(open_, float(close)))
    close = float(close)
    last_price = float(last_price if last_price is not None else close)
    volume = float(volume or 0.0)
    # Keep high/low consistent with live mark
    high = max(high, last_price, open_, close)
    low = min(low, last_price, open_, close)

    updated = item.get("updated") or last_trade.get("t") or minute.get("t")
    session = as_of
    if updated:
        try:
            session = _ny_session_date(int(updated))
        except Exception:
            session = as_of

    return {
        "symbol": symbol,
        "session_date": session,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "last_price": last_price,
        "volume": volume,
        "dvol": last_price * volume,
        "prev_close": float(prev["c"]) if prev.get("c") is not None else float("nan"),
        "todays_change": item.get("todaysChange"),
        "todays_change_pct": item.get("todaysChangePerc"),
        "updated": updated,
        "fetched_at": pd.Timestamp.now(tz="UTC"),
    }


def fetch_snapshots(
    client: MassiveClient,
    tickers: Sequence[str],
    *,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Fetch Massive snapshots for ``tickers`` (batched). No disk cache."""
    as_of = _ny_session_date()
    wanted = [str(t).upper().strip() for t in tickers if str(t).strip()]
    wanted = list(dict.fromkeys(wanted))
    rows: List[dict] = []

    print(f">> Realtime snapshots for {len(wanted)} tickers (batch={batch_size})...")
    for bi, batch in enumerate(_chunked(wanted, batch_size), 1):
        params = {"tickers": ",".join(batch), "include_otc": "false"}
        payload = client.get_json("/v2/snapshot/locale/us/markets/stocks/tickers", params=params)
        if not isinstance(payload, dict):
            print(f"    [realtime] batch {bi}: empty/non-dict response", flush=True)
            continue
        items = payload.get("tickers") or []
        if not isinstance(items, list):
            print(f"    [realtime] batch {bi}: unexpected payload keys={list(payload)}", flush=True)
            continue
        for item in items:
            parsed = _parse_snapshot_row(item, as_of)
            if parsed:
                rows.append(parsed)
        print(f"    ... realtime batch {bi}: got {len(items)} raw / {len(rows)} total parsed", flush=True)

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "session_date",
                "open",
                "high",
                "low",
                "close",
                "last_price",
                "volume",
                "dvol",
                "prev_close",
                "todays_change",
                "todays_change_pct",
                "updated",
                "fetched_at",
            ]
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol"], keep="last")
    return df.sort_values("symbol").reset_index(drop=True)


def save_snapshot(prices_dir: Path, snapshot: pd.DataFrame) -> Path:
    prices_dir.mkdir(parents=True, exist_ok=True)
    path = prices_dir / SNAPSHOT_PATH
    payload = {
        "fetched_at": pd.Timestamp.now(tz="UTC"),
        "rows": snapshot,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    csv_path = prices_dir / "realtime_snapshot.csv"
    snapshot.to_csv(csv_path, index=False)
    print(f"[realtime] wrote {path} ({len(snapshot)} symbols) + {csv_path.name}")
    return path


def merge_snapshot_into_panels(panels: dict, snapshot: pd.DataFrame) -> dict:
    """Upsert each snapshot's session day bar into OHLCV matrices."""
    if snapshot is None or snapshot.empty:
        return panels

    open_m = panels["open_m"].copy()
    close_m = panels["close_m"].copy()
    high_m = panels["high_m"].copy()
    low_m = panels["low_m"].copy()
    dvol_m = panels["dvol_m"].copy()

    for _, row in snapshot.iterrows():
        sym = row["symbol"]
        session = pd.Timestamp(row["session_date"]).normalize()
        close_px = float(row["last_price"] if pd.notna(row.get("last_price")) else row["close"])
        open_px = float(row["open"])
        high_px = float(row["high"])
        low_px = float(row["low"])
        dvol = float(row["dvol"])

        for frame in (open_m, close_m, high_m, low_m, dvol_m):
            if sym not in frame.columns:
                frame[sym] = pd.NA
            if session not in frame.index:
                # append chronologically
                frame.loc[session] = pd.NA

        open_m.loc[session, sym] = open_px
        close_m.loc[session, sym] = close_px
        high_m.loc[session, sym] = high_px
        low_m.loc[session, sym] = low_px
        dvol_m.loc[session, sym] = dvol

    # Keep indexes sorted after appends
    open_m = open_m.sort_index()
    close_m = close_m.sort_index()
    high_m = high_m.sort_index()
    low_m = low_m.sort_index()
    dvol_m = dvol_m.sort_index()

    out = dict(panels)
    out.update(
        {
            "open_m": open_m,
            "close_m": close_m,
            "high_m": high_m,
            "low_m": low_m,
            "dvol_m": dvol_m,
            "realtime_merged_at": pd.Timestamp.now(tz="UTC"),
        }
    )
    return out


def refresh_realtime(
    config: dict,
    *,
    merge_into_panels: bool = True,
    tickers: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Fetch snapshots for the local universe and optionally merge into panels."""
    paths = config.get("paths", {})
    prices_dir = Path(paths.get("prices", "data/prices"))
    universe_path = Path(paths.get("metadata", "data/metadata")) / "universe.pkl"
    panels_path = prices_dir / "panels.pkl"

    if not universe_path.exists():
        raise FileNotFoundError(f"Missing {universe_path}. Run download_universe first.")

    with open(universe_path, "rb") as f:
        universe = pickle.load(f)

    if tickers is None:
        tickers = list(universe.get("tickers") or universe.get("all_tickers") or [])
        benchmark = config.get("benchmark", "QQQ")
        if benchmark and benchmark not in tickers:
            tickers = list(tickers) + [benchmark]

    rt_cfg = config.get("realtime") or {}
    batch_size = int(rt_cfg.get("snapshot_batch_size", 50))
    if "merge_into_panels" in rt_cfg:
        merge_into_panels = bool(rt_cfg.get("merge_into_panels"))

    client = make_client(config)
    snapshot = fetch_snapshots(client, tickers, batch_size=batch_size)
    save_snapshot(prices_dir, snapshot)

    merged = False
    if merge_into_panels:
        if not panels_path.exists():
            raise FileNotFoundError(
                f"Missing {panels_path}. Run download_prices (historical) before merging realtime."
            )
        with open(panels_path, "rb") as f:
            panels = pickle.load(f)
        panels = merge_snapshot_into_panels(panels, snapshot)
        with open(panels_path, "wb") as f:
            pickle.dump(panels, f)
        merged = True
        print(f"[realtime] merged live day bars into {panels_path}")

    return {
        "n_symbols": int(len(snapshot)),
        "merged_into_panels": merged,
        "snapshot_path": str(prices_dir / SNAPSHOT_PATH),
        "session_dates": sorted({str(pd.Timestamp(d).date()) for d in snapshot["session_date"]})
        if not snapshot.empty
        else [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Massive realtime/delayed stock snapshots into local data/"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Only write realtime_snapshot.pkl; do not update panels.pkl",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    summary = refresh_realtime(config, merge_into_panels=not args.no_merge)
    print(
        f"[realtime] done symbols={summary['n_symbols']} "
        f"merged={summary['merged_into_panels']} sessions={summary['session_dates']}"
    )
    return 0 if summary["n_symbols"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
