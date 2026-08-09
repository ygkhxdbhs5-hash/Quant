"""Streamlit UI: download Massive data, breakout screener, backtest.

Local:
    streamlit run app.py
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from engine.breakout_screener import (
    BreakoutScreenerConfig,
    breakout_config_from_dict,
    run_screener,
)
from engine.strategy import load_config

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
UNI_PATH = ROOT / "data" / "metadata" / "universe.pkl"
PX_PATH = ROOT / "data" / "prices" / "panels.pkl"
FUND_PATH = ROOT / "data" / "fundamentals" / "pit_history.pkl"
EQUITY_CSV = ROOT / "cache" / "equity_curve.csv"
EQUITY_PNG = ROOT / "cache" / "equity_curve_v5.png"
BREAKOUT_CSV = ROOT / "cache" / "breakout_candidates.csv"

st.set_page_config(page_title="Quant — Breakout Screener", page_icon="📈", layout="wide")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)


def resolve_api_key(sidebar_value: str) -> str:
    key = (sidebar_value or "").strip()
    if key:
        return key
    key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY") or ""
    if key:
        return key
    try:
        key = st.secrets.get("MASSIVE_API_KEY", "")
    except Exception:
        key = ""
    return key or ""


def data_status() -> dict:
    status = {
        "universe_tickers": None,
        "all_tickers": None,
        "prices_shape": None,
        "fundamentals": None,
        "price_range": None,
    }
    if UNI_PATH.exists():
        uni = pickle.load(open(UNI_PATH, "rb"))
        status["universe_tickers"] = len(uni.get("tickers") or [])
        status["all_tickers"] = len(uni.get("all_tickers") or [])
    if PX_PATH.exists():
        panels = pickle.load(open(PX_PATH, "rb"))
        close_m = panels["close_m"]
        status["prices_shape"] = tuple(close_m.shape)
        status["price_range"] = f"{close_m.index.min().date()} → {close_m.index.max().date()}"
    if FUND_PATH.exists():
        funds = pickle.load(open(FUND_PATH, "rb"))
        status["fundamentals"] = len(funds)
    return status


def stream_command(cmd: list[str], env: dict, log_box) -> int:
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
        log_box.code("\n".join(lines[-200:]), language="text")
    return proc.wait()


def apply_download_config(
    sample_size: int | None,
    request_interval: float,
    api_key: str,
) -> None:
    cfg = load_yaml(CONFIG_PATH)
    cfg["massive_api_key"] = api_key
    cfg["universe_limit"] = sample_size
    cfg["request_interval_sec"] = float(request_interval)
    save_yaml(CONFIG_PATH, cfg)


def persist_breakout_config(screener_cfg: BreakoutScreenerConfig) -> None:
    cfg = load_yaml(CONFIG_PATH)
    cfg["breakout_screener"] = asdict(screener_cfg)
    save_yaml(CONFIG_PATH, cfg)


st.title("Quant — Download, Breakout Screener & Backtest")
st.caption("Uses local Massive.com panels already under `data/`. Screener does not re-download.")

_cfg0 = load_yaml(CONFIG_PATH)
_break0 = breakout_config_from_dict(_cfg0)

with st.sidebar:
    st.header("Settings")
    api_key_in = st.text_input(
        "Massive API key",
        type="password",
        help="Or set Streamlit secret / env MASSIVE_API_KEY",
    )
    sample_mode = st.selectbox(
        "Universe mode",
        options=["50 sample (fast)", "500 sample", "Full (slow)"],
        index=0,
    )
    sample_map = {
        "50 sample (fast)": 50,
        "500 sample": 500,
        "Full (slow)": None,
    }
    sample_size = sample_map[sample_mode]
    request_interval = st.slider("API interval (sec)", 0.05, 0.50, 0.20, 0.01)
    download_fundamentals = st.checkbox(
        "Also download fundamentals",
        value=True,
        help="Required for the backtest engine; not required for the breakout screener.",
    )
    clear_cache = st.checkbox("Clear HTTP cache before download", value=False)

    st.divider()
    st.header("Breakout filter")
    st.caption("Applied when you run the Breakout Screener tab.")
    ema_period = st.number_input("EMA period", min_value=5, max_value=200, value=int(_break0.ema_period), step=1)
    resistance_lookback = st.number_input(
        "Resistance lookback", min_value=10, max_value=252, value=int(_break0.resistance_lookback), step=1
    )
    volume_avg_window = st.number_input(
        "Volume average window", min_value=5, max_value=120, value=int(_break0.volume_avg_window), step=1
    )
    volume_threshold = st.number_input(
        "Volume threshold (× avg)",
        min_value=1.0,
        max_value=5.0,
        value=float(_break0.volume_threshold),
        step=0.1,
        format="%.1f",
    )
    max_breakout_age = st.number_input(
        "Max breakout age (bars)", min_value=1, max_value=30, value=int(_break0.max_breakout_age), step=1
    )
    max_breakout_separation = st.number_input(
        "Max EMA/resistance separation",
        min_value=0,
        max_value=20,
        value=int(_break0.max_breakout_separation),
        step=1,
    )
    max_extension_pct = st.slider(
        "Max extension above resistance",
        min_value=0.0,
        max_value=0.25,
        value=float(_break0.max_extension_pct),
        step=0.01,
        format="%.2f",
        help="Exclude names already this far above the resistance level (fraction, e.g. 0.05 = 5%).",
    )
    min_history_bars = st.number_input(
        "Min history bars", min_value=40, max_value=400, value=int(_break0.min_history_bars), step=5
    )
    as_of_date = st.text_input("As-of date (optional YYYY-MM-DD)", value="", help="Blank = latest session in panels")

api_key = resolve_api_key(api_key_in)
status = data_status()
prices_ready = PX_PATH.exists() and UNI_PATH.exists()
backtest_ready = prices_ready and FUND_PATH.exists()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers", status["universe_tickers"] if status["universe_tickers"] is not None else "—")
c2.metric("All pulled", status["all_tickers"] if status["all_tickers"] is not None else "—")
c3.metric("Fundamentals", status["fundamentals"] if status["fundamentals"] is not None else "—")
c4.metric("Price panel", str(status["prices_shape"]) if status["prices_shape"] else "—")
if status["price_range"]:
    st.write(f"Price history: `{status['price_range']}`")

tab_screen, tab_dl, tab_bt, tab_help = st.tabs(
    ["Breakout Screener", "Download data", "Run backtest", "Help"]
)

with tab_screen:
    st.subheader("Breakout candidate screener")
    st.write(
        "Scans the **already-downloaded** price panels for a fresh EMA upward breakout "
        "and a fresh horizontal resistance breakout, occurring close together, with "
        "elevated volume and limited extension above the breakout level."
    )
    if not prices_ready:
        st.warning("Missing `data/metadata/universe.pkl` or `data/prices/panels.pkl`. Use the Download tab first.")
    else:
        st.success("Local price panels found — screener will not re-download.")

    if st.button("Run breakout screener", type="primary", disabled=not prices_ready):
        screener_cfg = BreakoutScreenerConfig(
            ema_period=int(ema_period),
            resistance_lookback=int(resistance_lookback),
            volume_avg_window=int(volume_avg_window),
            volume_threshold=float(volume_threshold),
            max_breakout_age=int(max_breakout_age),
            max_breakout_separation=int(max_breakout_separation),
            max_extension_pct=float(max_extension_pct),
            min_history_bars=int(min_history_bars),
        )
        persist_breakout_config(screener_cfg)
        cfg = load_config(str(CONFIG_PATH))
        cfg["breakout_screener"] = asdict(screener_cfg)
        as_of = as_of_date.strip() or None
        with st.spinner("Scanning local panels…"):
            try:
                candidates = run_screener(config=cfg, as_of=as_of)
            except FileNotFoundError as exc:
                st.error(str(exc))
                candidates = pd.DataFrame()
            except Exception as exc:  # noqa: BLE001 — surface in UI
                st.exception(exc)
                candidates = pd.DataFrame()

        st.session_state["breakout_candidates"] = candidates
        BREAKOUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(BREAKOUT_CSV, index=False)

    candidates = st.session_state.get("breakout_candidates")
    if candidates is None and BREAKOUT_CSV.exists():
        candidates = pd.read_csv(BREAKOUT_CSV)
        st.session_state["breakout_candidates"] = candidates

    if candidates is not None:
        st.metric("Candidates", len(candidates))
        if candidates.empty:
            st.info("No breakout candidates matched the filter with the current parameters.")
        else:
            view = candidates.copy()
            if "as_of" in view.columns:
                view["as_of"] = pd.to_datetime(view["as_of"]).dt.strftime("%Y-%m-%d")
            if "extension_pct" in view.columns:
                view["extension_pct"] = (view["extension_pct"] * 100).map(lambda x: f"{x:.2f}%")
            if "volume_ratio" in view.columns:
                view["volume_ratio"] = view["volume_ratio"].map(lambda x: f"{float(x):.2f}x")
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.download_button(
                "Download candidates CSV",
                data=candidates.to_csv(index=False),
                file_name="breakout_candidates.csv",
                mime="text/csv",
            )

with tab_dl:
    st.subheader("Download from Massive.com")
    fund_note = "universe → prices" + (" → fundamentals" if download_fundamentals else "")
    st.write(f"Runs {fund_note}. Mode: **{sample_mode}**.")
    if not api_key:
        st.warning("Enter a Massive API key in the sidebar (or configure secrets).")

    if st.button("Start download", type="primary", disabled=not bool(api_key)):
        apply_download_config(sample_size, request_interval, api_key)
        env = dict(os.environ)
        env["MASSIVE_API_KEY"] = api_key
        env["PYTHONUNBUFFERED"] = "1"

        if clear_cache:
            cache_dir = ROOT / "cache"
            for p in cache_dir.rglob("*"):
                if p.is_file() and p.name != ".gitkeep":
                    p.unlink(missing_ok=True)
            (cache_dir / "massive_cache").mkdir(parents=True, exist_ok=True)
            st.info("Cache cleared.")

        log = st.empty()
        steps = [
            ("Universe", "downloader.download_universe"),
            ("Prices", "downloader.download_prices"),
        ]
        if download_fundamentals:
            steps.append(("Fundamentals", "downloader.download_fundamentals"))
        failed = False
        for title, module in steps:
            st.write(f"### {title}")
            code = stream_command(
                [sys.executable, "-u", "-m", module, "--config", str(CONFIG_PATH)],
                env,
                log,
            )
            if code != 0:
                st.error(f"{title} failed (exit {code})")
                failed = True
                break
            st.success(f"{title} done")
        if not failed:
            st.success("Downloads finished. Open the Breakout Screener tab.")
            st.rerun()

with tab_bt:
    st.subheader("Run standalone backtest")
    if not backtest_ready:
        st.warning("Backtest needs universe, prices, and fundamentals. Run Download with fundamentals enabled.")
    if st.button("Run backtest", type="primary", disabled=not backtest_ready):
        env = dict(os.environ)
        if api_key:
            env["MASSIVE_API_KEY"] = api_key
        env["PYTHONUNBUFFERED"] = "1"
        log = st.empty()
        code = stream_command(
            [sys.executable, "-u", str(ROOT / "run_backtest.py"), "--config", str(CONFIG_PATH)],
            env,
            log,
        )
        if code != 0:
            st.error(f"Backtest failed (exit {code})")
        else:
            st.success("Backtest finished")
            st.rerun()

    if EQUITY_PNG.exists():
        st.image(str(EQUITY_PNG), caption="Equity curve", use_container_width=True)
    if EQUITY_CSV.exists():
        eq = pd.read_csv(EQUITY_CSV)
        st.dataframe(eq.tail(30), use_container_width=True, hide_index=True)
        st.download_button(
            "Download equity_curve.csv",
            data=EQUITY_CSV.read_bytes(),
            file_name="equity_curve.csv",
            mime="text/csv",
        )

with tab_help:
    st.markdown(
        """
### Workflow
1. **Download data** — pull Massive universe/prices (once).
2. **Breakout Screener** — filter local panels for fresh EMA + resistance breakouts.
3. **Run backtest** — optional; uses the existing Q_Alpha v5 engine.

### Breakout filter (event-based)
- Fresh EMA cross: previous close ≤ EMA and current close > EMA
- Fresh horizontal resistance break (prior rolling high)
- Both breakouts recent and close together
- Volume above recent average
- Not already extended too far above resistance

Console alternative:
```bash
python run_breakout_screener.py --config config/config.yaml
streamlit run app.py
```
"""
    )
