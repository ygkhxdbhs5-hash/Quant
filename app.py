"""Streamlit web UI: download Massive data + run Baseline v1 backtest.

Baseline v1 = 12-1 momentum entry + ATR trail exit only.
Does NOT expose EMA / stop-loss / institutional / rank-buffer controls.

Local:
    streamlit run app.py

Streamlit Community Cloud:
    Main file path = app.py
    Branch = cursor/baseline-v1-cb1c
    Secret MASSIVE_API_KEY
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
UNI_PATH = ROOT / "data" / "metadata" / "universe.pkl"
PX_PATH = ROOT / "data" / "prices" / "panels.pkl"
FUND_PATH = ROOT / "data" / "fundamentals" / "pit_history.pkl"
BASELINE_DIR = ROOT / "cache" / "baseline_v1"
EQUITY_CSV = BASELINE_DIR / "equity_curve.csv"
EQUITY_PNG = BASELINE_DIR / "equity_curve.png"
QQQ_CSV = BASELINE_DIR / "qqq_equity_curve.csv"
COMPARISON_JSON = BASELINE_DIR / "benchmark_comparison.json"
TRADE_JOURNAL = BASELINE_DIR / "trade_journal.csv"
BASELINE_HISTORY = ROOT / "docs" / "experiments" / "BASELINE_V1" / "experiment_history.json"
BASELINE_PERIODS = ROOT / "docs" / "experiments" / "BASELINE_V1" / "period_results.json"
CHASE_DIR = ROOT / "docs" / "experiments" / "BASELINE_V1_TOPUP_CHASE"
CHASE_CHART = CHASE_DIR / "topup_chase_charts.png"
CHASE_JSON = CHASE_DIR / "topup_chase_results.json"
CHASE_REPORT = CHASE_DIR / "topup_chase_report.txt"
PERIOD_DIR = ROOT / "docs" / "experiments" / "BASELINE_V1_DIAGNOSTICS"
PERIOD_REPORT = PERIOD_DIR / "diagnostic_report.txt"
COST_FIX_DIR = ROOT / "docs" / "experiments" / "BASELINE_V1_COST_DATA_FIX"
COST_FIX_REPORT = COST_FIX_DIR / "cost_data_fix_report.txt"
COST_AUDIT_DIR = ROOT / "docs" / "experiments" / "BASELINE_V1_COST_AUDIT"
COST_AUDIT_REPORT = COST_AUDIT_DIR / "cost_audit_report.txt"

st.set_page_config(page_title="Quant Baseline v1", page_icon="📈", layout="wide")


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


def apply_ui_config(
    *,
    sample_size: int | None,
    request_interval: float,
    download_workers: int,
    api_key: str,
    start_date: str,
    end_date: str | None,
    top_liquid_pool: int,
    top_momentum_count: int,
    atr_multiplier: float,
    download_fundamentals: bool = False,
) -> None:
    cfg = load_yaml(CONFIG_PATH)
    cfg["universe_sample_size"] = sample_size
    cfg["request_interval_sec"] = float(request_interval)
    cfg["download_workers"] = int(download_workers)
    cfg["download_fundamentals"] = bool(download_fundamentals)
    cfg["start_date"] = str(start_date)
    cfg["end_date"] = end_date
    cfg["benchmark"] = "QQQ"
    cfg["atr_multiplier"] = float(atr_multiplier)
    cfg["max_portfolio_size"] = int(top_momentum_count)
    # Baseline knobs (read by BaselineEngineV1)
    baseline = dict(cfg.get("baseline_v1") or {})
    baseline.update(
        {
            "TOP_LIQUID_POOL": int(top_liquid_pool),
            "TOP_MOMENTUM_COUNT": int(top_momentum_count),
            "ATR_MULTIPLIER": float(atr_multiplier),
            "GROSS_EXPOSURE": 1.0,
        }
    )
    cfg["baseline_v1"] = baseline
    if api_key and api_key != "unused":
        cfg["massive_api_key"] = api_key
    save_yaml(CONFIG_PATH, cfg)


def _fmt_pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def _fmt_num(x) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.3f}"


st.title("Quant — Baseline v1")
st.caption(
    "12-1 Momentum entry · ATR(14)×2.5 trail exit · Equal weight · 1.0x · QQQ B&H benchmark"
)

_cfg0 = load_yaml(CONFIG_PATH)
_b0 = dict(_cfg0.get("baseline_v1") or {})

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
    clear_cache = st.checkbox("Clear HTTP cache before download", value=False)

    st.divider()
    st.header("Baseline v1 Config")
    st.caption(
        "Only knobs used by Baseline v1. "
        "No EMA exit, hard stop, institutional entry, rank buffer, or cooldown."
    )

    start_date = st.text_input(
        "START_DATE",
        value=str(_cfg0.get("start_date", "2018-01-01")),
        help="Backtest window start (YYYY-MM-DD)",
    )
    end_date_raw = st.text_input(
        "END_DATE",
        value=str(_cfg0.get("end_date") or ""),
        help="Empty = today",
    )
    end_date = end_date_raw.strip() or None

    top_liquid_pool = st.number_input(
        "TOP_LIQUID_POOL",
        min_value=50,
        max_value=1000,
        value=int(_b0.get("TOP_LIQUID_POOL", 250)),
        step=50,
        help="Rank by dollar volume; keep this many most liquid names",
    )
    top_momentum_count = st.number_input(
        "TOP_MOMENTUM_COUNT",
        min_value=5,
        max_value=100,
        value=int(_b0.get("TOP_MOMENTUM_COUNT", 30)),
        step=5,
        help="Within liquid pool, buy top N by mom_12_1",
    )
    atr_multiplier = st.slider(
        "ATR_MULTIPLIER",
        1.0,
        5.0,
        float(_b0.get("ATR_MULTIPLIER", _cfg0.get("atr_multiplier", 2.5))),
        0.1,
        help="stop = highest_close_since_entry − mult × ATR(14). Exit next Open if Low < stop.",
    )

    st.markdown("**Fixed (not editable)**")
    st.code(
        "Entry: mom_12_1 = price[t-21] / price[t-252] - 1\n"
        "Exit: ATR trail only (Low < stop → next Open)\n"
        "Sizing: equal weight 1/N\n"
        "Leverage: always 1.0x\n"
        "Rebalance: monthly candidates; hold until ATR; refill empties",
        language="text",
    )

    st.divider()
    st.markdown(
        "Deploy tip: Streamlit Cloud may time out on long downloads. "
        "Prefer **500 sample** or download elsewhere then upload `data/`."
    )

api_key = resolve_api_key(api_key_in)
status = data_status()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers (sample)", status["universe_tickers"] if status["universe_tickers"] is not None else "—")
c2.metric("All tickers pulled", status["all_tickers"] if status["all_tickers"] is not None else "—")
c3.metric("Price panel", str(status["prices_shape"]) if status["prices_shape"] else "—")
c4.metric("Benchmark", "QQQ B&H")
if status["price_range"]:
    st.write(f"Price history: `{status['price_range']}`")

tab_dl, tab_bt, tab_val, tab_reports, tab_help = st.tabs(
    [
        "1) Download data",
        "2) Run backtest",
        "3) 3-period validation",
        "4) Reports",
        "Help",
    ]
)

with tab_dl:
    st.subheader("Download from Massive.com")
    st.write(f"Runs universe → prices. Mode: **{sample_mode}**.")
    st.caption("Fundamentals are not used by Baseline v1 and are not downloaded.")
    if not api_key:
        st.warning("Enter a Massive API key in the sidebar (or configure secrets).")

    if st.button("Start download", type="primary", disabled=not bool(api_key)):
        apply_ui_config(
            sample_size=sample_size,
            request_interval=request_interval,
            download_workers=download_workers,
            api_key=api_key,
            start_date=start_date,
            end_date=end_date,
            top_liquid_pool=int(top_liquid_pool),
            top_momentum_count=int(top_momentum_count),
            atr_multiplier=float(atr_multiplier),
            download_fundamentals=False,
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
        failed = False
        for title, module in (
            ("Universe", "downloader.download_universe"),
            ("Prices", "downloader.download_prices"),
        ):
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
            st.success("Download finished. Open the Backtest tab.")
            st.rerun()

with tab_bt:
    st.subheader("Run Baseline v1 backtest")
    ready = UNI_PATH.exists() and PX_PATH.exists()
    if not ready:
        st.warning("Missing universe/prices data. Run Download first.")
    else:
        st.write("Uses `data/` artifacts already on disk — no re-download needed.")

    from engine.strategy_baseline_v1 import strategy_knobs

    knobs = strategy_knobs()
    knobs["TOP_LIQUID_POOL"] = int(top_liquid_pool)
    knobs["TOP_MOMENTUM_COUNT"] = int(top_momentum_count)
    knobs["ATR_MULTIPLIER"] = float(atr_multiplier)
    knobs["START_DATE"] = start_date
    knobs["END_DATE"] = end_date or "today"

    st.markdown("#### Active Baseline Configuration")
    st.code(
        "\n".join(f"{k}: {v}" for k, v in knobs.items()),
        language="text",
    )

    if st.button("Run backtest", type="primary", disabled=not ready):
        apply_ui_config(
            sample_size=sample_size,
            request_interval=request_interval,
            download_workers=download_workers,
            api_key=api_key or "unused",
            start_date=start_date,
            end_date=end_date,
            top_liquid_pool=int(top_liquid_pool),
            top_momentum_count=int(top_momentum_count),
            atr_multiplier=float(atr_multiplier),
        )
        env = dict(os.environ)
        if api_key:
            env["MASSIVE_API_KEY"] = api_key
        env["PYTHONUNBUFFERED"] = "1"
        log = st.empty()
        cmd = [
            sys.executable,
            "-u",
            str(ROOT / "run_backtest.py"),
            "--config",
            str(CONFIG_PATH),
            "--start",
            start_date,
        ]
        if end_date:
            cmd.extend(["--end", end_date])
        code = stream_command(cmd, env, log)
        if code != 0:
            st.error(f"Backtest failed (exit {code})")
        else:
            st.success("Backtest finished")
            st.rerun()

    if COMPARISON_JSON.exists():
        try:
            comparison = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))
            s = comparison.get("strategy") or {}
            q = comparison.get("QQQ") or {}
            st.subheader("Strategy vs QQQ Buy & Hold")
            m1, m2, m3 = st.columns(3)
            m1.metric("Strategy CAGR", _fmt_pct(s.get("CAGR")))
            m2.metric("Strategy Sharpe", _fmt_num(s.get("Sharpe")))
            m3.metric("Strategy MDD", _fmt_pct(s.get("Maximum_Drawdown")))
            n1, n2, n3 = st.columns(3)
            n1.metric("QQQ CAGR", _fmt_pct(q.get("CAGR")))
            n2.metric("QQQ Sharpe", _fmt_num(q.get("Sharpe")))
            n3.metric("QQQ MDD", _fmt_pct(q.get("Maximum_Drawdown")))
            st.metric("Alpha (CAGR − QQQ)", _fmt_pct(comparison.get("Alpha_CAGR")))
        except Exception as exc:
            st.warning(f"Could not read benchmark comparison: {exc}")

    if EQUITY_CSV.exists():
        eq = pd.read_csv(EQUITY_CSV, parse_dates=["Date"]).set_index("Date")
        if "Total_Equity" in eq.columns and len(eq) > 1:
            chart = eq[["Total_Equity"]].rename(columns={"Total_Equity": "Strategy"})
            if QQQ_CSV.exists():
                qqq = pd.read_csv(QQQ_CSV, parse_dates=["Date"]).set_index("Date")
                if "Total_Equity" in qqq.columns:
                    chart["QQQ B&H"] = qqq["Total_Equity"]
            st.line_chart(chart, height=360)
            st.download_button(
                "Download equity_curve.csv",
                data=EQUITY_CSV.read_bytes(),
                file_name="equity_curve.csv",
                mime="text/csv",
            )
        if EQUITY_PNG.exists():
            st.image(str(EQUITY_PNG))

    if TRADE_JOURNAL.exists() and TRADE_JOURNAL.stat().st_size > 0:
        st.download_button(
            "Download trade_journal.csv",
            data=TRADE_JOURNAL.read_bytes(),
            file_name="trade_journal.csv",
            mime="text/csv",
        )

with tab_val:
    st.subheader("3-period validation")
    st.caption("Runs 2018–2020, 2021–2022, 2023–2025 and appends experiment_history.json")
    ready = UNI_PATH.exists() and PX_PATH.exists()
    if not ready:
        st.warning("Missing universe/prices data. Run Download first.")

    if st.button("Run 3-period validation", type="primary", disabled=not ready):
        apply_ui_config(
            sample_size=sample_size,
            request_interval=request_interval,
            download_workers=download_workers,
            api_key=api_key or "unused",
            start_date=start_date,
            end_date=end_date,
            top_liquid_pool=int(top_liquid_pool),
            top_momentum_count=int(top_momentum_count),
            atr_multiplier=float(atr_multiplier),
        )
        env = dict(os.environ)
        if api_key:
            env["MASSIVE_API_KEY"] = api_key
        env["PYTHONUNBUFFERED"] = "1"
        log = st.empty()
        code = stream_command(
            [sys.executable, "-u", str(ROOT / "run_baseline_v1.py"), "--config", str(CONFIG_PATH)],
            env,
            log,
        )
        if code != 0:
            st.error(f"Validation failed (exit {code})")
        else:
            st.success("Validation finished")
            st.rerun()

    if BASELINE_PERIODS.exists():
        try:
            periods = json.loads(BASELINE_PERIODS.read_text(encoding="utf-8"))
            for row in periods:
                comp = row.get("comparison") or {}
                s = comp.get("strategy") or {}
                q = comp.get("QQQ") or {}
                st.markdown(f"### {row.get('label')}")
                a, b, c, d = st.columns(4)
                a.metric("Strategy CAGR", _fmt_pct(s.get("CAGR")))
                b.metric("QQQ CAGR", _fmt_pct(q.get("CAGR")))
                c.metric("Alpha", _fmt_pct(comp.get("Alpha_CAGR")))
                d.metric("Strategy MDD", _fmt_pct(s.get("Maximum_Drawdown")))
        except Exception as exc:
            st.warning(f"Could not parse period results: {exc}")

    if BASELINE_HISTORY.exists():
        st.download_button(
            "Download experiment_history.json",
            data=BASELINE_HISTORY.read_bytes(),
            file_name="experiment_history.json",
            mime="application/json",
        )

with tab_reports:
    st.subheader("Generate diagnostic reports")
    st.caption(
        "Click a button below — no terminal needed. Reports write under "
        "`docs/experiments/`. The top-up chase report also builds a single PNG "
        "with all charts for one-click download."
    )
    ready = UNI_PATH.exists() and PX_PATH.exists()
    if not ready:
        st.warning("Missing universe/prices data. Run Download first.")

    REPORT_JOBS = [
        {
            "key": "topup",
            "title": "Top-up chase + charts (recommended)",
            "desc": (
                "Fixed CS v2 chase vs no-chase vs flat 10/30bps (2022–2026). "
                "Writes text report + one combined PNG."
            ),
            "script": "run_topup_chase_report.py",
            "out_dir": str(CHASE_DIR),
            "report": CHASE_REPORT,
            "extra_args": ["--out-dir", str(CHASE_DIR)],
            "primary": True,
        },
        {
            "key": "period",
            "title": "Multi-period breakdown (Hypothesis A vs B)",
            "desc": "Independent runs: 2018–20, 2021–22, 2023–25, 2022-only, 2022–2026.",
            "script": "run_period_breakdown.py",
            "out_dir": str(PERIOD_DIR),
            "report": PERIOD_REPORT,
            "extra_args": ["--out-dir", str(PERIOD_DIR)],
            "primary": False,
        },
        {
            "key": "cost_fix",
            "title": "Cost/data fix comparison (CS v2 vs flats)",
            "desc": "Legacy CS vs Fixed CS v2 + ADV winsorize vs flat 10/30bps.",
            "script": "run_cost_data_fix_report.py",
            "out_dir": str(COST_FIX_DIR),
            "report": COST_FIX_REPORT,
            "extra_args": ["--out-dir", str(COST_FIX_DIR)],
            "primary": False,
        },
        {
            "key": "cost_audit",
            "title": "Cost-model audit (worst fills + flat benchmarks)",
            "desc": "Worst-cost fills, ADV windows, sizing vs ADV, flat 10/30bps.",
            "script": "run_cost_model_audit.py",
            "out_dir": str(COST_AUDIT_DIR),
            "report": COST_AUDIT_REPORT,
            "extra_args": ["--out-dir", str(COST_AUDIT_DIR)],
            "primary": False,
        },
    ]

    for job in REPORT_JOBS:
        with st.expander(job["title"], expanded=bool(job.get("primary"))):
            st.write(job["desc"])
            st.code(f"Output → {job['out_dir']}", language="text")
            btn_label = f"Generate: {job['title']}"
            if st.button(
                btn_label,
                key=f"gen_{job['key']}",
                type="primary" if job.get("primary") else "secondary",
                disabled=not ready,
            ):
                apply_ui_config(
                    sample_size=sample_size,
                    request_interval=request_interval,
                    download_workers=download_workers,
                    api_key=api_key or "unused",
                    start_date=start_date,
                    end_date=end_date,
                    top_liquid_pool=int(top_liquid_pool),
                    top_momentum_count=int(top_momentum_count),
                    atr_multiplier=float(atr_multiplier),
                    download_fundamentals=False,
                )
                env = dict(os.environ)
                if api_key:
                    env["MASSIVE_API_KEY"] = api_key
                env["PYTHONUNBUFFERED"] = "1"
                Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
                log = st.empty()
                st.info("Running… this can take a few minutes (multiple backtests). Keep this tab open.")
                code = stream_command(
                    [
                        sys.executable,
                        "-u",
                        str(ROOT / job["script"]),
                        "--config",
                        str(CONFIG_PATH),
                        *job["extra_args"],
                    ],
                    env,
                    log,
                )
                if code != 0:
                    st.error(f"Report failed (exit {code})")
                else:
                    st.success(f"Report finished → {job['out_dir']}")
                    st.rerun()

            # Downloads / preview for this job if artifacts exist
            rep: Path = job["report"]
            if rep.exists():
                st.download_button(
                    f"Download {rep.name}",
                    data=rep.read_bytes(),
                    file_name=rep.name,
                    mime="text/plain",
                    key=f"dl_txt_{job['key']}",
                )
                with st.expander(f"Preview {rep.name}", expanded=False):
                    text = rep.read_text(encoding="utf-8", errors="replace")
                    st.code(text[-12000:] if len(text) > 12000 else text, language="text")

            if job["key"] == "topup" and CHASE_CHART.exists():
                st.markdown("#### Combined charts (copy / download once)")
                st.image(str(CHASE_CHART), use_container_width=True)
                st.download_button(
                    "Download all charts (one PNG)",
                    data=CHASE_CHART.read_bytes(),
                    file_name="topup_chase_charts.png",
                    mime="image/png",
                    type="primary",
                    key="dl_chase_png",
                )
                if CHASE_JSON.exists():
                    st.download_button(
                        "Download results JSON",
                        data=CHASE_JSON.read_bytes(),
                        file_name="topup_chase_results.json",
                        mime="application/json",
                        key="dl_chase_json",
                    )
            elif job["key"] == "topup" and CHASE_JSON.exists() and not CHASE_CHART.exists():
                if st.button("Build chart PNG from saved JSON", key="build_chase_png"):
                    from engine.report_charts import render_from_json_file

                    path = render_from_json_file(CHASE_JSON, CHASE_CHART)
                    st.success(f"Wrote {path}")
                    st.rerun()

with tab_help:
    st.markdown(
        """
### What this app runs
**Baseline v1 only**
- Entry: `mom_12_1 = price[t-21] / price[t-252] - 1` (Top liquid 250 → Top 30)
- Exit: ATR(14) Wilder trail × multiplier (default 2.5); Low < stop → next Open
- Sizing: equal weight `1/N`
- Leverage: always `1.0x`
- Benchmark: QQQ buy & hold

### Not in this UI / strategy
EMA exits · hard stop-loss · institutional multi-factor · rank hysteresis ·
exhaustion · cooldown · industry caps · correlation filters

### Reports (no terminal needed)
Open tab **4) Reports** and click **Generate**. Recommended first:
**Top-up chase + charts** — produces the text report and one PNG with all charts.

### Streamlit Cloud
1. Deploy branch that includes this Reports tab (e.g. `cursor/no-chase-topup-cb1c`)
2. Main file: `app.py`
3. Secret:
```toml
MASSIVE_API_KEY = "your_key"
```
"""
    )
