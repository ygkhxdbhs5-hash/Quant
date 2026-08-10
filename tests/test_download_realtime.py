"""Unit tests for realtime snapshot parsing / panel merge (no network)."""

from __future__ import annotations

import pandas as pd

from downloader.download_realtime import (
    _parse_snapshot_row,
    merge_snapshot_into_panels,
    _ny_session_date,
)


def test_parse_snapshot_row_uses_last_trade_mark():
    as_of = pd.Timestamp("2024-06-17")
    item = {
        "ticker": "AAA",
        "day": {"o": 10.0, "h": 11.0, "l": 9.5, "c": 10.5, "v": 1000},
        "prevDay": {"c": 10.0},
        "lastTrade": {"p": 10.8, "t": 1_718_640_000_000_000_000},
        "todaysChange": 0.8,
        "todaysChangePerc": 8.0,
        "updated": 1_718_640_000_000_000_000,
    }
    row = _parse_snapshot_row(item, as_of)
    assert row is not None
    assert row["symbol"] == "AAA"
    assert row["last_price"] == 10.8
    assert row["close"] == 10.5
    assert row["volume"] == 1000
    assert row["high"] >= 10.8


def test_merge_snapshot_into_panels_appends_session_row():
    idx = pd.bdate_range("2024-06-10", periods=5)
    panels = {
        "open_m": pd.DataFrame({"AAA": 10.0}, index=idx),
        "close_m": pd.DataFrame({"AAA": 10.0}, index=idx),
        "high_m": pd.DataFrame({"AAA": 10.5}, index=idx),
        "low_m": pd.DataFrame({"AAA": 9.5}, index=idx),
        "dvol_m": pd.DataFrame({"AAA": 1e5}, index=idx),
        "silent_delist_flags": {},
    }
    snap = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "session_date": pd.Timestamp("2024-06-17"),
                "open": 10.0,
                "high": 11.0,
                "low": 9.8,
                "close": 10.7,
                "last_price": 10.9,
                "volume": 2000,
                "dvol": 10.9 * 2000,
            }
        ]
    )
    out = merge_snapshot_into_panels(panels, snap)
    assert pd.Timestamp("2024-06-17") in out["close_m"].index
    assert float(out["close_m"].loc[pd.Timestamp("2024-06-17"), "AAA"]) == 10.9
    assert float(out["open_m"].loc[pd.Timestamp("2024-06-17"), "AAA"]) == 10.0


def test_ny_session_date_from_ms():
    # 2024-06-17 16:00 ET roughly
    ts_ms = 1_718_654_400_000
    d = _ny_session_date(ts_ms)
    assert str(d.date()) == "2024-06-17"
