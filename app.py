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
EXP001_DIR = ROOT / "docs" / "experiments" / "EXP001_EXIT_RANK"
EXP001_FULL_REPORT = EXP001_DIR / "EXP001_FULL_REPORT.txt"
EXP001_RESULT = EXP001_DIR / "EXPERIMENT1_RESULT.md"
EXP001_DELTA = EXP001_DIR / "DELTA_REPORT.txt"
EXP001_FACTS = EXP001_DIR / "FACTS.txt"
EXP001_RECO = EXP001_DIR / "RESEARCH_RECOMMENDATION.txt"
EXP001_VALIDATION = EXP001_DIR / "RUNTIME_CONFIG_VALIDATION.json"
EXP001_CHECKLIST = EXP001_DIR / "VALIDATION_CHECKLIST.json"


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
    research: dict | None = None,
    exit_rank: int | None = None,
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
    buf = int(exit_rank) if exit_rank is not None else int(cfg.get("selection_buffer_size", 70))
    cfg["selection_buffer_size"] = max(int(max_portfolio) + 20, buf) if exit_rank is None else buf
    if research is not None:
        # Research Configuration Panel → config research: block (engine reads before run)
        block = dict(cfg.get("research") or {})
        block.update(research)
        # Keep portfolio/ATR top-level aliases in sync with panel
        block["ENTRY_RANK"] = int(max_portfolio)
        block["EXIT_RANK"] = int(cfg["selection_buffer_size"])
        block["MAX_PORTFOLIO_SIZE"] = int(max_portfolio)
        block["ATR_MULTIPLIER"] = float(atr_multiplier)
        cfg["research"] = block
        if "MAX_INDUSTRY_WEIGHT" in block:
            cfg["max_industry_weight"] = float(block["MAX_INDUSTRY_WEIGHT"])
    if api_key and api_key != "unused":
        cfg["massive_api_key"] = api_key
    save_yaml(CONFIG_PATH, cfg)


st.title("Quant — Download & Backtest")
st.caption("Massive.com data download + Q_Alpha v5 standalone engine")

_cfg0 = load_yaml(CONFIG_PATH)
_r0 = dict(_cfg0.get("research") or {})


def _r_get(key: str, default):
    return _r0[key] if key in _r0 else default


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
    download_fundamentals = st.checkbox(
        "Also download fundamentals (optional)",
        value=False,
        help="Not required for CMVS v3 (price/volume only). Enable only for legacy fundamental screens.",
    )
    clear_cache = st.checkbox("Clear HTTP cache before download", value=False)

    st.divider()
    st.header("Research Config")
    st.caption(
        "Experiment knobs only — written to config research: before each run. "
        "Defaults match current CMVS trade identity."
    )

    st.markdown("**Entry / Exit**")
    entry_rank = st.number_input(
        "ENTRY_RANK",
        min_value=5,
        max_value=200,
        value=int(_r_get("ENTRY_RANK", _cfg0.get("max_portfolio_size", 50))),
        step=5,
        help="Top-N core / max portfolio size",
    )
    exit_rank = st.number_input(
        "EXIT_RANK",
        min_value=5,
        max_value=300,
        value=int(_r_get("EXIT_RANK", _cfg0.get("selection_buffer_size", 70))),
        step=5,
        help="Hysteresis buffer — keep if still in top EXIT_RANK",
    )

    st.markdown("**EMA Exit**")
    use_ema9_exit = st.checkbox(
        "USE_EMA9_EXIT (confirmed trend exit)",
        value=bool(_r_get("USE_EMA9_EXIT", True)),
        help=(
            "Enables confirmation-based trend exit. "
            "A single close below EMA9 never sells — needs ≥2 confirmations "
            "(2 closes below EMA, EMA20 break, RSI<45, vol on decline, neg 5d mom). "
            "High ATR% names use EMA20 and need ≥3 confirms. ATR trail stays primary."
        ),
    )
    ema_exit_length = st.selectbox(
        "EMA_EXIT_LENGTH",
        options=[9, 10, 15, 20, 30],
        index=[9, 10, 15, 20, 30].index(int(_r_get("EMA_EXIT_LENGTH", 9)))
        if int(_r_get("EMA_EXIT_LENGTH", 9)) in (9, 10, 15, 20, 30)
        else 0,
        help="EMA period for trend-breakdown exit (no code change needed)",
    )

    st.markdown("**ATR**")
    use_atr_exit = st.checkbox(
        "USE_ATR_EXIT",
        value=bool(_r_get("USE_ATR_EXIT", True)),
        help="Dynamic ATR trailing stop",
    )
    atr_multiplier = st.slider(
        "ATR_MULTIPLIER",
        1.0,
        4.0,
        float(_r_get("ATR_MULTIPLIER", _cfg0.get("atr_multiplier", 2.0))),
        0.1,
        help="Sell if close < peak_close − multiplier × ATR(14)",
    )

    st.markdown("**Exhaustion**")
    use_exhaustion_exit = st.checkbox(
        "USE_EXHAUSTION_EXIT",
        value=bool(_r_get("USE_EXHAUSTION_EXIT", True)),
        help="RSI > 80 and daily close position < 0.3",
    )

    st.markdown("**Time Stop**")
    use_time_stop = st.checkbox(
        "USE_TIME_STOP",
        value=bool(_r_get("USE_TIME_STOP", False)),
        help="Optional — off by default (no baseline impact)",
    )
    time_stop_days = st.number_input(
        "TIME_STOP_DAYS",
        min_value=1,
        max_value=120,
        value=int(_r_get("TIME_STOP_DAYS", 20)),
        step=1,
        disabled=not use_time_stop,
    )

    st.markdown("**Hard Stop Loss**")
    use_stop_loss = st.checkbox(
        "USE_STOP_LOSS",
        value=bool(_r_get("USE_STOP_LOSS", False)),
        help=(
            "Hard filter: sell if close ≤ entry × (1 − STOP_LOSS_PCT). "
            "Measured from entry price (not peak). Off by default — no baseline impact. "
            "When on, bypasses MIN_HOLD_DAYS."
        ),
    )
    stop_loss_pct_ui = st.slider(
        "STOP_LOSS_PCT",
        min_value=0.0,
        max_value=50.0,
        value=float(_r_get("STOP_LOSS_PCT", 0.20)) * 100.0
        if float(_r_get("STOP_LOSS_PCT", 0.20)) <= 1.0
        else float(_r_get("STOP_LOSS_PCT", 20.0)),
        step=1.0,
        format="%.0f%%",
        disabled=not use_stop_loss,
        help="0% = break-even stop (sell at/below entry). 50% = allow 50% drawdown from entry.",
    )

    st.markdown("**Holding / Portfolio / Risk**")
    min_hold_days = st.number_input(
        "MIN_HOLD_DAYS",
        min_value=0,
        max_value=60,
        value=int(_r_get("MIN_HOLD_DAYS", 0)),
        step=1,
    )
    max_portfolio = int(entry_rank)  # MAX_PORTFOLIO_SIZE mirrors ENTRY_RANK
    monthly_rebalance = st.checkbox(
        "MONTHLY_REBALANCE",
        value=bool(_r_get("MONTHLY_REBALANCE", True)),
        help="On = monthly first-day rebalance (baseline). Off = daily rebalance.",
    )
    max_industry_weight = st.slider(
        "MAX_INDUSTRY_WEIGHT",
        0.10,
        1.00,
        float(_r_get("MAX_INDUSTRY_WEIGHT", _cfg0.get("max_industry_weight", 0.40))),
        0.05,
    )

    research_ui = {
        "ENTRY_RANK": int(entry_rank),
        "EXIT_RANK": int(exit_rank),
        "USE_EMA9_EXIT": bool(use_ema9_exit),
        "EMA_EXIT_LENGTH": int(ema_exit_length),
        "USE_ATR_EXIT": bool(use_atr_exit),
        "ATR_MULTIPLIER": float(atr_multiplier),
        "USE_EXHAUSTION_EXIT": bool(use_exhaustion_exit),
        "USE_TIME_STOP": bool(use_time_stop),
        "TIME_STOP_DAYS": int(time_stop_days),
        "USE_STOP_LOSS": bool(use_stop_loss),
        "STOP_LOSS_PCT": float(stop_loss_pct_ui) / 100.0,
        "MIN_HOLD_DAYS": int(min_hold_days),
        "MAX_PORTFOLIO_SIZE": int(max_portfolio),
        "MONTHLY_REBALANCE": bool(monthly_rebalance),
        "MAX_INDUSTRY_WEIGHT": float(max_industry_weight),
        "SHADOW_HORIZON_DAYS": int(_r_get("SHADOW_HORIZON_DAYS", 20)),
    }

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

tab_dl, tab_bt, tab_exp, tab_help = st.tabs(
    ["1) Download data", "2) Run backtest", "3) Experiment 1", "Help"]
)

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
            research=research_ui,
            exit_rank=int(exit_rank),
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

    from engine.research.config_toggles import ResearchToggles

    st.markdown("#### Research Configuration (will be applied)")
    st.caption("Edit values in the sidebar Research Config section.")
    st.code(
        ResearchToggles(
            ENTRY_RANK=int(research_ui["ENTRY_RANK"]),
            EXIT_RANK=int(research_ui["EXIT_RANK"]),
            USE_EMA9_EXIT=bool(research_ui["USE_EMA9_EXIT"]),
            EMA_EXIT_LENGTH=int(research_ui["EMA_EXIT_LENGTH"]),
            USE_ATR_EXIT=bool(research_ui["USE_ATR_EXIT"]),
            ATR_MULTIPLIER=float(research_ui["ATR_MULTIPLIER"]),
            USE_EXHAUSTION_EXIT=bool(research_ui["USE_EXHAUSTION_EXIT"]),
            USE_TIME_STOP=bool(research_ui["USE_TIME_STOP"]),
            TIME_STOP_DAYS=int(research_ui["TIME_STOP_DAYS"]),
            USE_STOP_LOSS=bool(research_ui["USE_STOP_LOSS"]),
            STOP_LOSS_PCT=float(research_ui["STOP_LOSS_PCT"]),
            MIN_HOLD_DAYS=int(research_ui["MIN_HOLD_DAYS"]),
            MAX_PORTFOLIO_SIZE=int(research_ui["MAX_PORTFOLIO_SIZE"]),
            MAX_INDUSTRY_WEIGHT=float(research_ui["MAX_INDUSTRY_WEIGHT"]),
            MONTHLY_REBALANCE=bool(research_ui["MONTHLY_REBALANCE"]),
            SHADOW_HORIZON_DAYS=int(research_ui["SHADOW_HORIZON_DAYS"]),
        ).format_panel(),
        language="text",
    )

    if st.button("Run backtest", type="primary", disabled=not ready):
        apply_ui_config(
            sample_size,
            request_interval,
            max_portfolio,
            api_key or "unused",
            download_workers,
            atr_multiplier,
            download_fundamentals,
            research=research_ui,
            exit_rank=int(exit_rank),
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

with tab_exp:
    st.subheader("Experiment 1 — EXIT_RANK only")
    st.caption(
        "Research Rule #1: only EXIT_RANK changes (discovered baseline → 80). "
        "Baseline runs first, experiment second. No trading-logic edits."
    )
    ready_exp = UNI_PATH.exists() and PX_PATH.exists()
    if not ready_exp:
        st.warning("Missing universe/prices data. Run Download first.")
    else:
        st.write(
            "Runs `scripts/run_exp1_exit_rank.py`: "
            "baseline EXIT_RANK → experiment EXIT_RANK=80, then prints the "
            "8-section report (Architecture Audit → Decision)."
        )

    if st.button("Run Experiment 1", type="primary", disabled=not ready_exp):
        env = dict(os.environ)
        if api_key:
            env["MASSIVE_API_KEY"] = api_key
        env["PYTHONUNBUFFERED"] = "1"
        log = st.empty()
        code = stream_command(
            [sys.executable, "-u", str(ROOT / "scripts" / "run_exp1_exit_rank.py")],
            env,
            log,
        )
        if code != 0:
            st.error(f"Experiment 1 failed (exit {code})")
            abort = EXP001_DIR / "ABORT.txt"
            if abort.exists():
                st.code(abort.read_text(encoding="utf-8"), language="text")
        else:
            st.success("Experiment 1 finished")
            st.rerun()

    # --- Show preserved Exp1 artifacts ---
    if EXP001_VALIDATION.exists():
        try:
            val = json.loads(EXP001_VALIDATION.read_text(encoding="utf-8"))
            v = val.get("validation") or {}
            passed = bool(v.get("passed"))
            st.markdown("### Validation")
            if passed:
                st.success("PASS — EXIT_RANK is the only behavioral change")
            else:
                st.error("FAIL — unexpected behavioral differences")
                unexpected = v.get("unexpected_diffs") or {}
                if unexpected:
                    st.json(unexpected)
            b_id = val.get("baseline_identity") or {}
            e_id = val.get("experiment_identity") or {}
            c1, c2 = st.columns(2)
            c1.metric("Baseline EXIT_RANK", f"{b_id.get('EXIT_RANK', 'n/a')}")
            c2.metric("Experiment EXIT_RANK", f"{e_id.get('EXIT_RANK', 'n/a')}")
            c1.metric("Baseline ENTRY_RANK", f"{b_id.get('ENTRY_RANK', 'n/a')}")
            c2.metric("Experiment ENTRY_RANK", f"{e_id.get('ENTRY_RANK', 'n/a')}")
        except Exception as exc:
            st.warning(f"Could not parse validation JSON: {exc}")

    if EXP001_CHECKLIST.exists():
        try:
            checklist = json.loads(EXP001_CHECKLIST.read_text(encoding="utf-8"))
            decision = checklist.get("decision")
            if decision:
                st.markdown("### Decision")
                st.info(str(decision))
            reco = None
            if EXP001_RECO.exists():
                reco = EXP001_RECO.read_text(encoding="utf-8").strip()
            if reco:
                st.markdown("### Research Recommendation")
                st.code(reco, language="text")
        except Exception:
            pass

    if EXP001_DELTA.exists():
        st.markdown("### Delta Report")
        st.code(EXP001_DELTA.read_text(encoding="utf-8"), language="text")
        st.download_button(
            "Download DELTA_REPORT.txt",
            data=EXP001_DELTA.read_bytes(),
            file_name="DELTA_REPORT.txt",
            mime="text/plain",
            key="dl_exp1_delta",
        )

    if EXP001_FACTS.exists():
        st.markdown("### Fact Generation")
        st.code(EXP001_FACTS.read_text(encoding="utf-8"), language="text")

    report_path = EXP001_RESULT if EXP001_RESULT.exists() else EXP001_FULL_REPORT
    if report_path.exists():
        st.markdown("### Full Experiment 1 Report (8 sections)")
        st.code(report_path.read_text(encoding="utf-8"), language="text")
        st.download_button(
            "Download Experiment 1 report",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="text/plain",
            key="dl_exp1_full",
        )
    elif ready_exp:
        st.info("No Experiment 1 artifacts yet. Click **Run Experiment 1**.")

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
4. Branch: `cursor/webapp-cb1c`  
5. Secrets:
```toml
MASSIVE_API_KEY = "your_key"
```

### Notes
- Free Streamlit Cloud often **times out** on long Massive downloads. Use **500 sample** locally, or download on a VPS/Colab then upload `data/` into the app environment.
- CMVS v3 needs **universe + prices** only. Fundamentals download is optional / off by default.
- Backtest-only is fast once `data/` exists.
- After a backtest, **Rank diagnostics** shows `rank_exit_candidates`, `ema_preempted_rank_exit`, and `holding_rank_distribution` from `cache/rank_diagnostics.json` (observation only).
- **Experiment 1** tab runs baseline vs `EXIT_RANK=80` (single variable) and shows Validation, Delta Report, Facts, Recommendation, and Decision.
"""
    )
