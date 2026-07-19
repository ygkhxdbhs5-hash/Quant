"""Streamlit web UI: download Massive data + run Q_Alpha backtest.

Local:
    streamlit run app.py

Streamlit Community Cloud:
    Deploy this repo, set Main file path = app.py,
    add secret MASSIVE_API_KEY in app settings.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
UNI_PATH = ROOT / "data" / "metadata" / "universe.pkl"
PX_PATH = ROOT / "data" / "prices" / "panels.pkl"
FUND_PATH = ROOT / "data" / "fundamentals" / "pit_history.pkl"
EQUITY_CSV = ROOT / "cache" / "equity_curve.csv"
EQUITY_PNG = ROOT / "cache" / "equity_curve_v5.png"
RESEARCH_REPORT = ROOT / "cache" / "research_report.txt"
TRADE_JOURNAL = ROOT / "cache" / "trade_journal.csv"
RANK_DIAG_JSON = ROOT / "cache" / "rank_diagnostics.json"
RANK_DIAG_TXT = ROOT / "cache" / "rank_diagnostics_report.txt"


st.set_page_config(page_title="Quant Backtest", page_icon="📈", layout="wide")


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
    """Run a subprocess and stream stdout/stderr into the Streamlit log box."""
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
        # keep last ~200 lines visible
        log_box.code("\n".join(lines[-200:]), language="text")
    return proc.wait()


def apply_ui_config(
    sample_size: int | None,
    request_interval: float,
    max_portfolio: int,
    api_key: str,
    download_workers: int = 8,
    atr_multiplier: float = 2.0,
    download_fundamentals: bool = False,
) -> None:
    cfg = load_yaml(CONFIG_PATH)
    cfg["universe_sample_size"] = sample_size
    cfg["request_interval_sec"] = float(request_interval)
    cfg["download_workers"] = int(download_workers)
    cfg["download_fundamentals"] = bool(download_fundamentals)
    # CMVS v3 exit parameter (old trailing_stop_pct UI removed — unused by engine)
    cfg["atr_multiplier"] = float(atr_multiplier)
    # cfg["trailing_stop_pct"] = ...  # legacy % trail; replaced by CMVS exits
    cfg["max_portfolio_size"] = int(max_portfolio)
    cfg["selection_buffer_size"] = max(int(max_portfolio) + 20, int(cfg.get("selection_buffer_size", 70)))
    if api_key and api_key != "unused":
        cfg["massive_api_key"] = api_key
    save_yaml(CONFIG_PATH, cfg)


st.title("Quant — Download & Backtest")
st.caption("Massive.com data download + Q_Alpha v5 standalone engine")

with st.sidebar:
    st.header("Settings")
    api_key_in = st.text_input(
        "Massive API key",
        type="password",
        help="Or set Streamlit secret / env MASSIVE_API_KEY",
    )
    sample_mode = st.selectbox(
        "Universe mode",
        options=["500 sample (recommended)", "1000 sample", "Full NASDAQ (slow)"],
        index=0,
    )
    sample_map = {
        "500 sample (recommended)": 500,
        "1000 sample": 1000,
        "Full NASDAQ (slow)": None,
    }
    sample_size = sample_map[sample_mode]
    download_workers = st.slider("Download workers", 1, 16, 8, 1)
    request_interval = st.slider("API interval (sec)", 0.05, 0.50, 0.08, 0.01)
    # Old % trailing-stop slider removed — CMVS uses ATR trail / EMA9 / RSI exits
    atr_multiplier = st.slider(
        "ATR trail multiplier",
        1.0,
        4.0,
        2.0,
        0.1,
        help="CMVS exit: sell if close < peak_close − multiplier × ATR(14)",
    )
    max_portfolio = st.number_input("Max portfolio size", min_value=10, max_value=100, value=50, step=5)
    download_fundamentals = st.checkbox(
        "Also download fundamentals (optional)",
        value=False,
        help="Not required for CMVS v3 (price/volume only). Enable only for legacy fundamental screens.",
    )
    clear_cache = st.checkbox("Clear HTTP cache before download", value=False)
    st.divider()
    st.markdown(
        "Deploy tip: on **Streamlit Cloud**, long downloads may time out. "
        "Prefer a VPS / local run for full refreshes."
    )

api_key = resolve_api_key(api_key_in)
status = data_status()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers (sample)", status["universe_tickers"] if status["universe_tickers"] is not None else "—")
c2.metric("All tickers pulled", status["all_tickers"] if status["all_tickers"] is not None else "—")
c3.metric(
    "Fundamentals (optional)",
    status["fundamentals"] if status["fundamentals"] is not None else "skip",
)
c4.metric("Price panel", str(status["prices_shape"]) if status["prices_shape"] else "—")
if status["price_range"]:
    st.write(f"Price history: `{status['price_range']}`")

tab_dl, tab_bt, tab_help = st.tabs(["1) Download data", "2) Run backtest", "Help"])

with tab_dl:
    st.subheader("Download from Massive.com")
    fund_note = "universe → prices" + (" → fundamentals" if download_fundamentals else " (fundamentals skipped)")
    st.write(f"Runs {fund_note}. Mode: **{sample_mode}**.")
    if not api_key:
        st.warning("Enter a Massive API key in the sidebar (or configure secrets).")

    if st.button("Start download", type="primary", disabled=not bool(api_key)):
        apply_ui_config(
            sample_size,
            request_interval,
            max_portfolio,
            api_key,
            download_workers,
            atr_multiplier,
            download_fundamentals,
        )
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
            st.success("All downloads finished. Open the Backtest tab.")
            st.rerun()

with tab_bt:
    st.subheader("Run backtest")
    # CMVS needs universe + prices only
    ready = UNI_PATH.exists() and PX_PATH.exists()
    if not ready:
        st.warning("Missing universe/prices data. Run Download first.")
    else:
        st.write("Uses `data/` artifacts already on disk — no re-download needed.")
        if not FUND_PATH.exists():
            st.caption("Fundamentals not present (optional for CMVS v3).")

    if st.button("Run backtest", type="primary", disabled=not ready):
        apply_ui_config(
            sample_size,
            request_interval,
            max_portfolio,
            api_key or "unused",
            download_workers,
            atr_multiplier,
            download_fundamentals,
        )
        env = dict(os.environ)
        if api_key:
            env["MASSIVE_API_KEY"] = api_key
        env["PYTHONUNBUFFERED"] = "1"
        log = st.empty()
        code = stream_command(
            [
                sys.executable,
                "-u",
                str(ROOT / "run_backtest.py"),
                "--config",
                str(CONFIG_PATH),
            ],
            env,
            log,
        )
        if code != 0:
            st.error(f"Backtest failed (exit {code})")
        else:
            st.success("Backtest finished")

    if EQUITY_CSV.exists():
        eq = pd.read_csv(EQUITY_CSV, parse_dates=["Date"]).set_index("Date")
        if "Total_Equity" in eq.columns and len(eq) > 1:
            ret = (eq["Total_Equity"].iloc[-1] / eq["Total_Equity"].iloc[0] - 1) * 100
            peak = eq["Total_Equity"].cummax()
            dd = ((eq["Total_Equity"] - peak) / peak).min() * 100
            m1, m2, m3 = st.columns(3)
            m1.metric("Cumulative return", f"{ret:.2f}%")
            m2.metric("Max drawdown", f"{dd:.2f}%")
            m3.metric("Final equity", f"{eq['Total_Equity'].iloc[-1]:,.0f}")
            st.line_chart(eq["Total_Equity"], height=360)
            st.download_button(
                "Download equity_curve.csv",
                data=EQUITY_CSV.read_bytes(),
                file_name="equity_curve.csv",
                mime="text/csv",
            )
        if EQUITY_PNG.exists():
            st.image(str(EQUITY_PNG))

    # Rank diagnostics (observation-only; written at end of StandaloneEngine.run)
    rank_diag = None
    if RANK_DIAG_JSON.exists():
        try:
            rank_diag = json.loads(RANK_DIAG_JSON.read_text(encoding="utf-8"))
        except Exception:
            rank_diag = None
    if rank_diag is not None or RANK_DIAG_TXT.exists():
        st.subheader("Rank diagnostics")
        st.caption("Observation only — does not change trading behavior.")
        if rank_diag is not None:
            c1, c2 = st.columns(2)
            c1.metric(
                "rank_exit_candidates",
                f"{int(rank_diag.get('rank_exit_candidates', 0))}",
            )
            c2.metric(
                "ema_preempted_rank_exit",
                f"{int(rank_diag.get('ema_preempted_rank_exit', 0))}",
            )
            dist = rank_diag.get("holding_rank_distribution") or {}
            st.markdown("**holding_rank_distribution**")
            if not dist or dist.get("n", 0) == 0:
                st.info("No holding-rank observations recorded.")
            else:
                d1, d2, d3, d4, d5, d6 = st.columns(6)
                d1.metric("min", f"{float(dist['min']):.2f}")
                d2.metric("median", f"{float(dist['median']):.2f}")
                d3.metric("p75", f"{float(dist['p75']):.2f}")
                d4.metric("p90", f"{float(dist['p90']):.2f}")
                d5.metric("p95", f"{float(dist['p95']):.2f}")
                d6.metric("max", f"{float(dist['max']):.2f}")
                st.caption(f"n holding-day rank observations: {int(dist.get('n', 0))}")
        if RANK_DIAG_TXT.exists():
            st.code(RANK_DIAG_TXT.read_text(encoding="utf-8"), language="text")
            st.download_button(
                "Download rank_diagnostics_report.txt",
                data=RANK_DIAG_TXT.read_bytes(),
                file_name="rank_diagnostics_report.txt",
                mime="text/plain",
            )

    if RESEARCH_REPORT.exists():
        st.subheader("Research report")
        # Full report (rank diagnostics also appear at the end when present)
        st.code(RESEARCH_REPORT.read_text(encoding="utf-8"), language="text")
        st.download_button(
            "Download research_report.txt",
            data=RESEARCH_REPORT.read_bytes(),
            file_name="research_report.txt",
            mime="text/plain",
        )
    if TRADE_JOURNAL.exists() and TRADE_JOURNAL.stat().st_size > 0:
        st.download_button(
            "Download trade_journal.csv",
            data=TRADE_JOURNAL.read_bytes(),
            file_name="trade_journal.csv",
            mime="text/csv",
        )

with tab_help:
    st.markdown(
        """
### Local run
```bash
pip install -r requirements.txt
export MASSIVE_API_KEY=your_key
streamlit run app.py
```

### Streamlit Community Cloud
1. Push this repo to GitHub  
2. [share.streamlit.io](https://share.streamlit.io) → New app  
3. Main file: `app.py`  
4. Secrets:
```toml
MASSIVE_API_KEY = "your_key"
```

### Notes
- Free Streamlit Cloud often **times out** on long Massive downloads. Use **500 sample** locally, or download on a VPS/Colab then upload `data/` into the app environment.
- CMVS v3 needs **universe + prices** only. Fundamentals download is optional / off by default.
- Backtest-only is fast once `data/` exists.
- After a backtest, **Rank diagnostics** shows `rank_exit_candidates`, `ema_preempted_rank_exit`, and `holding_rank_distribution` from `cache/rank_diagnostics.json` (observation only).
"""
    )
