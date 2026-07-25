"""
Q_Alpha - Pure Python Standalone Backtest Engine (v5)
=====================================================
QuantConnect/LEAN 의존성을 완전히 제거한 순수 파이썬 버전.
lean_engine_v3.py의 파이프라인 설계(PIT 추상화, industry-relative 랭킹 + 소규모 산업군 폴백,
turnover 히스테리시스, 상관관계 필터, 역변동성 사이징+floor, industry cap, 검증 assertion,
구조화 로깅)를 그대로 유지하고, 데이터 소스/체결 모델만 Massive.com API 기반으로 교체.

Data download lives in ``downloader/`` (Massive.com). This module only reads
local artifacts under ``data/``.

LEAN -> 순수 파이썬 대응관계:
  AddUniverse(Coarse/Fine)     -> Massive /v3/reference/tickers + ticker overview
  Fine 재무 필드                -> Massive financials statements (PIT by filing_date)
  SetHoldings(symbol, weight)  -> 목표비중 대비 델타를 다음날 시가 주문으로 큐잉
  VolumeShareSlippageModel     -> Corwin-Schultz 스프레드 + sqrt 시장충격 근사
  InteractiveBrokersFeeModel   -> 위 비용모델에 포함(별도 수수료 미분리, 근사)
  data.Delistings              -> 공식 delisted 리스트 + "조용한 상장폐지" 탐지
  RollingWindow[float]         -> 사전 계산된 벡터화 매트릭스(가격/모멘텀/변동성)

[근사/한계 - 명시]
  - ROIC은 Massive financials 항목으로 직접 근사 계산한다.
    invested_capital = total_debt + total_equity - cash (가능하면), 아니면 total_assets로 폴백.
    세율은 21% 고정 가정 (실제 유효세율과 다를 수 있음) -> 근사치임을 명시.
  - LEAN의 FeeModel/SlippageModel만큼 검증된 모델이 아니라 학술 근사식(Corwin-Schultz, sqrt-impact)이다.
  - PIT의 완전성 한계는 이전 리뷰(CODE_REVIEW_v3_to_v4.md) 참고 -> 여전히 유효.
  - Massive filing_date를 acceptedDate 대용으로 사용한다.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from engine.research.config_toggles import load_research_toggles
from engine.research.experiment_history import ExperimentHistory
from engine.research.kpi_report import build_hierarchical_kpi_report, format_kpi_report
from engine.research.rank_diagnostics import RankDiagnostics
from engine.research.recommendations import (
    build_research_recommendation_report,
    format_recommendation_report,
)
from engine.research.trade_journal import TradeJournal
from engine.research.validation import (
    format_validation_checklist,
    run_research_validation_checklist,
)
from engine.reentry_cooldown import AdaptiveReentryCooldown
from engine.entry_quality import (
    EQSWeights,
    blend_cmvs_eqs,
    combine_eqs,
    compute_eqs_components,
    eqs_config_from_mapping,
)
from engine.fundamental_quality import (
    DEFAULT_FUND_POINT_SCALE,
    compute_fundamental_bonus_points,
    extract_fundamental_metrics,
    format_quality_diagnostics,
    quality_bonus,
)


def _short_reason(exit_reason: str) -> str:
    r = (exit_reason or "").lower()
    if "atr_trail" in r:
        return "atr_trail"
    if "ema" in r:
        return "ema_break"
    if "exhaustion" in r:
        return "exhaustion"
    if "bear" in r:
        return "bear_flatten"
    if "rebalance" in r:
        return "rebalance_exit"
    return (exit_reason or "other")[:24]


# =====================================================================
# 컨텍스트 / 상태
# =====================================================================
@dataclass
class RegimeState:
    exposure: float
    breadth: float
    benchmark_above_ma200: bool


@dataclass
class RebalanceContext:
    as_of_date: object = None
    regime: Optional[RegimeState] = None
    factor_df: Optional[pd.DataFrame] = None
    ranked_df: Optional[pd.DataFrame] = None
    targets_df: Optional[pd.DataFrame] = None
    correlation_rejections: list = field(default_factory=list)
    industry_cap_rejections: list = field(default_factory=list)
    actually_invested: dict = field(default_factory=dict)  # symbol -> target weight


def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    env_key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    if env_key:
        cfg["massive_api_key"] = env_key
    if not cfg.get("end_date"):
        cfg["end_date"] = pd.Timestamp.today().strftime("%Y-%m-%d")
    return cfg


class StandaloneEngine:
    """Investment logic matches the provided Q_Alpha v5 engine."""

    def __init__(self, config: Optional[dict] = None, config_path: str = "config/config.yaml"):
        self.config = config or load_config(config_path)
        cfg = self.config
        paths = cfg.get("paths", {})

        # LEAN 클래스 속성과 1:1 대응되는 설정값
        self.START_DATE = cfg.get("start_date", "2010-01-01")
        self.END_DATE = cfg.get("end_date")
        self.INITIAL_CASH = float(cfg.get("initial_cash", 50_000_000))
        self.BENCHMARK_TICKER = cfg.get("benchmark", "QQQ")
        self.MAX_PORTFOLIO_SIZE = int(cfg.get("max_portfolio_size", 50))
        self.SELECTION_BUFFER_SIZE = int(cfg.get("selection_buffer_size", 70))
        # Fingerprint of config portfolio knobs before research aliases bind
        self._baseline_max_n = self.MAX_PORTFOLIO_SIZE
        self._baseline_buf_n = self.SELECTION_BUFFER_SIZE
        # Aggressive test default 0.40 (was 0.20); override via config
        self.MAX_INDUSTRY_WEIGHT = float(cfg.get("max_industry_weight", 0.40))
        self.MIN_INDUSTRY_SIZE = int(cfg.get("min_industry_size", 8))
        self.VOL_FLOOR = float(cfg.get("vol_floor", 1e-4))
        self.WEIGHT_SUM_TOLERANCE = float(cfg.get("weight_sum_tolerance", 0.02))
        self.LOWVOL_WINDOW = int(cfg.get("lowvol_window", 60))
        self.MOM_WINDOW = int(cfg.get("mom_window", 252))
        self.CORR_WINDOW = int(cfg.get("corr_window", 60))
        # Aggressive test default 0.95 (was 0.80); override via config
        self.CORR_THRESHOLD = float(cfg.get("corr_threshold", 0.95))
        # Volatility breakout filter: (close - open) > mult * 20-day ATR
        self.VOL_BREAKOUT_ATR_MULT = float(cfg.get("vol_breakout_atr_mult", 1.5))
        self.FLAT_TAX_RATE = float(cfg.get("flat_tax_rate", 0.21))
        self.SILENT_DELIST_RECOVERY = float(cfg.get("silent_delist_recovery", 0.30))
        self.SILENT_DELIST_GAP_DAYS = int(cfg.get("silent_delist_gap_days", 10))
        self.TOP_ADV_POOL = int(cfg.get("top_adv_pool", 250))
        self.PARTICIPATION_CAP_BUY = float(cfg.get("participation_cap_buy", 0.08))
        self.PARTICIPATION_CAP_SELL = float(cfg.get("participation_cap_sell", 0.15))
        # Trading costs applied on every buy/sell at rebalance fill
        self.COMMISSION_RATE = float(cfg.get("commission_rate", 0.0005))  # 0.05%
        self.SLIPPAGE_RATE = float(cfg.get("slippage_rate", 0.0002))  # 0.02%
        # Universe filters (loosened for more candidates; missing metrics do not hard-reject)
        self.MIN_ROIC = float(cfg.get("min_roic", 0.03))  # Quality: ROIC > 3% (was 10%)
        self.MIN_FCF_SALES_YIELD = float(cfg.get("min_fcf_sales_yield", -0.05))  # allow mild neg FCF
        self.MAX_DEBT_TO_EQUITY = float(cfg.get("max_debt_to_equity", 3.0))  # D/E < 300% (was 150%)
        self.MIN_REVENUE_GROWTH_YOY = float(cfg.get("min_revenue_growth_yoy", -0.15))  # allow mild contraction

        # --- Research Configuration Panel (must bind BEFORE _precompute_matrices) ---
        # Spec examples ENTRY_RANK=30 / EXIT_RANK=80 are NOT silent defaults;
        # they would change baseline vs max_portfolio_size / selection_buffer_size.
        self.research_toggles = load_research_toggles(cfg)
        self.ENTRY_RANK = int(self.research_toggles.ENTRY_RANK)
        self.EXIT_RANK = int(self.research_toggles.EXIT_RANK)
        self.USE_EMA9_EXIT = bool(self.research_toggles.USE_EMA9_EXIT)
        self.EMA_EXIT_LENGTH = int(self.research_toggles.EMA_EXIT_LENGTH)
        self.USE_ATR_EXIT = bool(self.research_toggles.USE_ATR_EXIT)
        self.atr_multiplier = float(self.research_toggles.ATR_MULTIPLIER)
        self.ATR_MULTIPLIER = self.atr_multiplier  # alias for research panel naming
        self.USE_EXHAUSTION_EXIT = bool(self.research_toggles.USE_EXHAUSTION_EXIT)
        self.USE_TIME_STOP = bool(self.research_toggles.USE_TIME_STOP)
        self.TIME_STOP_DAYS = int(self.research_toggles.TIME_STOP_DAYS)
        self.MIN_HOLD_DAYS = int(self.research_toggles.MIN_HOLD_DAYS)
        self.MONTHLY_REBALANCE = bool(self.research_toggles.MONTHLY_REBALANCE)
        self.MAX_INDUSTRY_WEIGHT = float(self.research_toggles.MAX_INDUSTRY_WEIGHT)
        # Keep legacy names synchronized with research aliases (no behavior change at defaults).
        # Portfolio construction uses ENTRY_RANK; MAX_PORTFOLIO_SIZE mirrors it.
        self.MAX_PORTFOLIO_SIZE = self.ENTRY_RANK
        self.SELECTION_BUFFER_SIZE = self.EXIT_RANK

        print(">> 로컬 데이터 로드...")
        universe_path = Path(paths.get("metadata", "data/metadata")) / "universe.pkl"
        panels_path = Path(paths.get("prices", "data/prices")) / "panels.pkl"
        funds_path = Path(paths.get("fundamentals", "data/fundamentals")) / "pit_history.pkl"
        # CMVS v3 needs universe + prices only; fundamentals are optional.
        for required in (universe_path, panels_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"Missing {required}. Run: python -m downloader.update_data --config {config_path}"
                )

        with open(universe_path, "rb") as f:
            universe = pickle.load(f)
        with open(panels_path, "rb") as f:
            panels = pickle.load(f)
        if funds_path.exists():
            with open(funds_path, "rb") as f:
                self.fundamental_history = pickle.load(f) or {}
            print(f"    fundamentals loaded: {len(self.fundamental_history)} symbols (optional)")
        else:
            self.fundamental_history = {}
            print("    fundamentals skipped (no pit_history.pkl) — OK for CMVS v3")

        self.all_tickers = universe["all_tickers"]
        self.tickers = universe["tickers"]
        self.delisted_meta = universe["delisted_meta"]
        self.profile_meta = universe["profile_meta"]
        self.open_m = panels["open_m"]
        self.close_m = panels["close_m"]
        self.high_m = panels["high_m"]
        self.low_m = panels["low_m"]
        self.dvol_m = panels["dvol_m"]
        self.silent_delist_flags = panels["silent_delist_flags"]

        self._precompute_matrices()

        self.cash = self.INITIAL_CASH
        self.portfolio = {}  # symbol -> qty
        self.previous_target_symbols = set()
        self.pending_orders = []
        self.equity_curve = []
        self._prior_invested_for_log = set()
        # Peak close since entry (used by CMVS dynamic ATR trailing stop)
        self.highest_prices = {}
        self._cmvs_forced_exits = set()
        # Adaptive re-entry ban after losing exits (stops CURX/HKIT/JEM-style loops)
        self.reentry_cooldown = AdaptiveReentryCooldown(enabled=True)
        # Prior % trailing stop (replaced by CMVS exits; kept for revert):
        # self.TRAILING_STOP_PCT = float(cfg.get("trailing_stop_pct", 0.20))

        # CMVS v3 component weights — tilted toward sustainable RS / structure.
        # Prior equal 0.2 weights over-rewarded BB expansion, volume spikes, and late RSI.
        self.w1 = float(cfg.get("cmvs_w1", 0.10))  # BBS (capped / dampened in rank)
        self.w2 = float(cfg.get("cmvs_w2", 0.10))  # VZS (only with rising price)
        self.w3 = float(cfg.get("cmvs_w3", 0.15))  # CPS
        self.w4 = float(cfg.get("cmvs_w4", 0.10))  # RSIS (mid-zone preferred)
        self.w5 = float(cfg.get("cmvs_w5", 0.25))  # RSS relative strength
        # atr_multiplier already bound from research panel above
        # Entry quality thresholds (hard filters + score penalties)
        self.MIN_ENTRY_PRICE = float(cfg.get("min_entry_price", 3.0))
        self.MAX_ATR_PCT_ENTRY = float(cfg.get("max_atr_pct_entry", 0.15))
        self.MAX_SPIKE_5D = float(cfg.get("max_spike_5d", 0.35))
        self.MIN_CLOSE_VS_HIGH20 = float(cfg.get("min_close_vs_high20", 0.60))

        # High-vol threshold for stricter trend-exit confirmations / EMA20 primary
        self.HIGH_ATR_PCT_EXIT = float(cfg.get("high_atr_pct_exit", 0.08))
        self.TREND_EXIT_MIN_CONFIRM = int(cfg.get("trend_exit_min_confirm", 2))
        self.TREND_EXIT_MIN_CONFIRM_HIGH_VOL = int(cfg.get("trend_exit_min_confirm_high_vol", 3))
        # Pullback protection: skip trend exits while above rising SMA50 with shallow DD
        self.PROTECT_HEALTHY_TREND_PULLBACK = bool(cfg.get("protect_healthy_trend_pullback", True))

        # Entry Quality Score (EQS) — live ranking blend with CMVS
        self.eqs_weights, self.EQS_BLEND_WEIGHT = eqs_config_from_mapping(cfg)
        # Fundamental bonus (recovered screens → signed points, small score tilt)
        self.USE_QUALITY_BONUS = bool(self.research_toggles.USE_QUALITY_BONUS)
        # EQS_WEIGHT == FUND_POINT_SCALE: points * scale added to CMVS(+tech EQS)
        self.EQS_WEIGHT = float(self.research_toggles.EQS_WEIGHT)
        self._last_quality_diag: str = ""
        self._last_quality_map: dict = {}
        self._last_quality_bonus_map: dict = {}
        self._last_fund_points_map: dict = {}

        self.trade_journal = TradeJournal(
            shadow_horizon_days=int(self.research_toggles.SHADOW_HORIZON_DAYS)
        )
        # Observation-only counters; never used to queue orders or alter targets.
        self.rank_diagnostics = RankDiagnostics(exit_rank=float(self.EXIT_RANK))
        self._last_rank_map: dict = {}
        self._last_cmvs_map: dict = {}
        self.research_artifacts: dict = {}

    # -------------------------------------------------------------
    def _precompute_matrices(self):
        print(">> 벡터화 매트릭스(ATR/모멘텀/변동성/스프레드/CMVS) 사전 계산...")
        close_m, high_m, low_m = self.close_m, self.high_m, self.low_m
        volume_m = self.dvol_m  # dollar-volume panel used as volume proxy

        self.adv20_m = volume_m.rolling(20, min_periods=5).mean()
        ret_m = close_m.pct_change()
        self.vol20_m = ret_m.rolling(20, min_periods=10).std()
        self.vol60_m = ret_m.rolling(self.LOWVOL_WINDOW, min_periods=20).std()
        self.sma50_m = close_m.rolling(50, min_periods=50).mean()
        self.sma200_m = close_m.rolling(200, min_periods=200).mean()

        # ATR (Wilder-style) for exits (14) and optional sizing (20)
        prev_close = close_m.shift(1)
        tr = (high_m - low_m).combine((high_m - prev_close).abs(), np.maximum).combine(
            (low_m - prev_close).abs(), np.maximum
        )
        self.atr20_m = tr.rolling(20, min_periods=10).mean()
        self.atr14_m = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()

        # [버그 수정 계승] 12-1 모멘텀: t-252~t-21 구간의 누적수익률
        self.mom_12_1_m = close_m.shift(21) / close_m.shift(self.MOM_WINDOW) - 1

        # --- Prior swing technicals (replaced by CMVS; kept for revert) ---
        # self.short_term_mom_m = close_m / close_m.shift(20) - 1
        # self.rel_vol_m = self.dvol_m / self.adv20_m.replace(0, np.nan)

        # --- CMVS v3 matrices ---
        # Bollinger (20, 2σ)
        bb_middle = close_m.rolling(20, min_periods=20).mean()
        bb_std = close_m.rolling(20, min_periods=20).std()
        bb_upper = bb_middle + 2.0 * bb_std
        bb_lower = bb_middle - 2.0 * bb_std
        bb_range = (bb_upper - bb_lower).replace(0, np.nan)
        percent_b = (close_m - bb_lower) / bb_range
        bbw = bb_range / bb_middle.replace(0, np.nan)
        bbw_ma20 = bbw.rolling(20, min_periods=20).mean().replace(0, np.nan)
        bbw_expansion = bbw / bbw_ma20
        bbs_raw = percent_b * np.log1p(bbw_expansion.clip(lower=0.0))
        self.bbs_m = bbs_raw.clip(lower=0.0, upper=1.0)

        # Volume Z-Score (normalized)
        vol_mean20 = volume_m.rolling(20, min_periods=20).mean()
        vol_std20 = volume_m.rolling(20, min_periods=20).std().replace(0, np.nan)
        vol_z = (volume_m - vol_mean20) / vol_std20
        self.vzs_m = (vol_z / 3.0).clip(lower=0.0, upper=1.0)

        # Close Position Score
        self.daily_cp_m = (close_m - low_m) / (high_m - low_m + 1e-8)
        self.cps_m = self.daily_cp_m.rolling(3, min_periods=1).mean().clip(lower=0.0, upper=1.0)

        # RSI(14) + RSIS
        delta = close_m.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.rsi14_m = 100.0 - (100.0 / (1.0 + rs))
        self.rsis_m = ((self.rsi14_m - 50.0) / 30.0).clip(lower=0.0, upper=1.0)

        # Configurable short EMA for trend diagnostics (default 9).
        # Standalone EMA break no longer sells — see confirmed trend exit.
        _ema_len = max(1, int(getattr(self, "EMA_EXIT_LENGTH", 9)))
        self.ema_exit_m = close_m.ewm(span=_ema_len, adjust=False).mean()
        self.ema9_m = self.ema_exit_m
        self.ema20_m = close_m.ewm(span=20, adjust=False).mean()
        self.ema50_m = close_m.ewm(span=50, adjust=False).mean()
        self.ema20_slope5_m = self.ema20_m / self.ema20_m.shift(5) - 1.0
        self.ema50_slope5_m = self.ema50_m / self.ema50_m.shift(5) - 1.0
        # Relative volume for decline confirmation (dvol vs 20d average)
        self.rel_vol_m = volume_m / vol_mean20.replace(0, np.nan)

        # Returns / structure matrices for entry quality + adaptive cooldown
        self.ret1_m = ret_m
        self.ret5_m = close_m / close_m.shift(5) - 1.0
        self.ret10_m = close_m / close_m.shift(10) - 1.0
        self.ret20_m = close_m / close_m.shift(20) - 1.0
        self.ret60_m = close_m / close_m.shift(60) - 1.0
        self.sma20_m = close_m.rolling(20, min_periods=20).mean()
        self.high20_m = high_m.rolling(20, min_periods=10).max()
        self.close_vs_high20_m = close_m / self.high20_m.replace(0, np.nan)
        self.pullback_pct_m = 1.0 - self.close_vs_high20_m
        self.atr_pct_m = self.atr14_m / close_m.replace(0, np.nan)
        # Share of up-days in last 20 — sustainable momentum proxy
        up = (ret_m > 0).astype(float)
        self.up_frac20_m = up.rolling(20, min_periods=10).mean()
        self.green_streak5_m = up.rolling(5, min_periods=1).sum()
        # Rising SMA50 used to protect healthy long-term trends from noise exits
        self.sma50_slope5_m = self.sma50_m / self.sma50_m.shift(5) - 1.0

        # EQS feature matrices (PIT; computed once)
        range_pct = (high_m - low_m) / close_m.replace(0, np.nan)
        self.range_pct_5_m = range_pct.rolling(5, min_periods=3).mean()
        self.range_pct_10_m = range_pct.rolling(10, min_periods=5).mean()
        self.range_pct_20_m = range_pct.rolling(20, min_periods=10).mean()
        atr_lag15 = self.atr14_m.shift(15)
        self.atr_shrink_ratio_m = self.atr14_m / atr_lag15.replace(0, np.nan)
        # Volume: quiet consolidation (days -15..-5) vs recovery (last 5)
        vol_mid = volume_m.shift(5).rolling(10, min_periods=5).mean()
        vol_recent = volume_m.rolling(5, min_periods=3).mean()
        vol_prior = volume_m.shift(15).rolling(15, min_periods=8).mean()
        self.vol_quiet_ratio_m = vol_mid / vol_prior.replace(0, np.nan)
        self.vol_expand_ratio_m = vol_recent / vol_mid.replace(0, np.nan)
        self.vol_spike_persist_m = (self.rel_vol_m > 2.0).astype(float).rolling(
            20, min_periods=10
        ).mean()
        # Down-day relative volume over last 10 sessions (pullback volume)
        down_vol = volume_m.where(ret_m < 0.0)
        self.pullback_vol_ratio_m = (
            down_vol.rolling(10, min_periods=3).mean() / vol_mean20.replace(0, np.nan)
        )
        # Relative strength vs benchmark (panel) for RS acceleration / sector
        bm = self.BENCHMARK_TICKER
        if bm in self.ret20_m.columns:
            self.rs_vs_bm_m = self.ret20_m.sub(self.ret20_m[bm], axis=0)
        else:
            self.rs_vs_bm_m = self.ret20_m.copy()
        self.rs_slope5_m = self.rs_vs_bm_m - self.rs_vs_bm_m.shift(5)
        rs_high60 = self.rs_vs_bm_m.rolling(60, min_periods=20).max().replace(0, np.nan)
        self.rs_near_high60_m = self.rs_vs_bm_m / rs_high60
        self.dist_ema20_m = close_m / self.ema20_m.replace(0, np.nan) - 1.0
        self.dist_ema50_m = close_m / self.ema50_m.replace(0, np.nan) - 1.0
        self.above_ema50_m = (close_m > self.ema50_m).astype(float)

        hl = np.log(high_m / low_m) ** 2
        hl2_high = high_m.rolling(2, min_periods=2).max()
        hl2_low = low_m.rolling(2, min_periods=2).min()
        gamma = np.log(hl2_high / hl2_low) ** 2
        beta = hl + hl.shift(1)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        self.cs_spread_m = spread.clip(lower=0.0002, upper=0.05)

    def get_latest_available_fundamentals(self, symbol, as_of_date):
        """[PIT] as_of_date 이전에 실제로 accepted된 가장 최근 재무 스냅샷만 반환."""
        hist = self.fundamental_history.get(symbol)
        if hist is None or hist.empty:
            return None
        past = hist.loc[:as_of_date]
        if past.empty:
            return None
        return past.iloc[-1]

    def _revenue_growth_yoy(self, symbol, as_of_date):
        """YoY revenue growth from PIT history (latest vs ~1y prior filing)."""
        hist = self.fundamental_history.get(symbol)
        if hist is None or hist.empty or "revenue" not in hist.columns:
            return np.nan
        past = hist.loc[:as_of_date]
        if past.empty:
            return np.nan
        latest_rev = past["revenue"].iloc[-1]
        if pd.isna(latest_rev):
            return np.nan
        latest_dt = past.index[-1]
        target = latest_dt - pd.DateOffset(years=1)
        prior = past.loc[:target]
        if not prior.empty and pd.notna(prior["revenue"].iloc[-1]):
            prior_rev = float(prior["revenue"].iloc[-1])
        elif len(past) >= 5 and pd.notna(past["revenue"].iloc[-5]):
            # Quarterly fallback: ~4 periods earlier
            prior_rev = float(past["revenue"].iloc[-5])
        else:
            return np.nan
        if prior_rev == 0:
            return np.nan
        return (float(latest_rev) - prior_rev) / abs(prior_rev)

    def get_universe(self, candidate_symbols, date_idx, quiet: bool = False):
        """Price/volume universe only — fundamental screens disabled (commented).

        Prior (soft fundamental screens; missing metrics did not reject):
        Quality: ROIC > min_roic when ROIC is available.
        Value: FCF/Sales (or Sales/MCap approx) > min when available.
        Debt: D/E < max when available.
        Growth: Revenue YoY > min when available.
        No PIT fundamentals -> keep name for price/volume ranking fallback.
        """
        current_date = self.close_m.index[date_idx]
        # --- Aggressive swing mode: pass all price-active candidates ---
        filtered = list(candidate_symbols)
        if not quiet:
            print(
                f"    [UNIVERSE] {current_date.date()} in={len(candidate_symbols)} "
                f"out={len(filtered)} mode=price_volume_only (fundamentals ignored)"
            )
        return filtered

        # --- Prior fundamental soft screens (kept for revert) ---
        # current_closes = self.close_m.loc[current_date]
        # filtered = []
        # n_fail_quality = 0
        # n_fail_value = 0
        # n_fail_debt = 0
        # n_fail_growth = 0
        # n_value_approx = 0
        # n_pass_no_fundamentals = 0
        #
        # for sym in candidate_symbols:
        #     fundamentals = self.get_latest_available_fundamentals(sym, current_date)
        #     if fundamentals is None:
        #         # Loosened: keep price-only names instead of hard-rejecting
        #         filtered.append(sym)
        #         n_pass_no_fundamentals += 1
        #         continue
        #
        #     def _fget(key, default=np.nan):
        #         try:
        #             val = fundamentals[key]
        #             return default if pd.isna(val) else float(val)
        #         except Exception:
        #             return default
        #
        #     # --- Quality: only reject when ROIC is present and too low ---
        #     roic = _fget("roic")
        #     if pd.notna(roic) and roic <= self.MIN_ROIC:
        #         n_fail_quality += 1
        #         continue
        #
        #     # --- Value: FCF/Sales (fallback Sales/MCap); missing -> pass ---
        #     revenue = _fget("revenue")
        #     op_cf = _fget("op_cf")
        #     capex = _fget("capex")
        #     value_yield = np.nan
        #     if pd.notna(revenue) and revenue != 0 and pd.notna(op_cf) and pd.notna(capex):
        #         value_yield = (op_cf - capex) / revenue
        #     else:
        #         shares = _fget("diluted_shares_outstanding")
        #         if pd.isna(shares) or shares <= 0:
        #             shares = _fget("basic_shares_outstanding")
        #         price = current_closes.get(sym, np.nan)
        #         if (
        #             pd.notna(revenue)
        #             and revenue > 0
        #             and pd.notna(shares)
        #             and shares > 0
        #             and pd.notna(price)
        #             and price > 0
        #         ):
        #             market_cap = float(price) * float(shares)
        #             if market_cap > 0:
        #                 value_yield = float(revenue) / market_cap
        #                 n_value_approx += 1
        #
        #     if pd.notna(value_yield) and value_yield <= self.MIN_FCF_SALES_YIELD:
        #         n_fail_value += 1
        #         continue
        #
        #     # --- Debt: only reject when D/E is present and too high ---
        #     debt_to_equity = _fget("debt_to_equity")
        #     if pd.isna(debt_to_equity):
        #         total_debt = _fget("total_debt")
        #         total_equity = _fget("total_equity")
        #         if pd.notna(total_debt) and pd.notna(total_equity) and total_equity != 0:
        #             debt_to_equity = float(total_debt) / float(total_equity)
        #     if pd.notna(debt_to_equity) and debt_to_equity >= self.MAX_DEBT_TO_EQUITY:
        #         n_fail_debt += 1
        #         continue
        #
        #     # --- Growth: only reject when YoY is present and too weak ---
        #     rev_growth = _fget("revenue_growth_yoy")
        #     if pd.isna(rev_growth):
        #         rev_growth = self._revenue_growth_yoy(sym, current_date)
        #     if pd.notna(rev_growth) and rev_growth <= self.MIN_REVENUE_GROWTH_YOY:
        #         n_fail_growth += 1
        #         continue
        #
        #     filtered.append(sym)
        #
        # print(
        #     f"    [UNIVERSE] {current_date.date()} in={len(candidate_symbols)} "
        #     f"out={len(filtered)} fail_quality={n_fail_quality} fail_value={n_fail_value} "
        #     f"fail_debt={n_fail_debt} fail_growth={n_fail_growth} "
        #     f"pass_no_fundamentals={n_pass_no_fundamentals} "
        #     f"value_approx_mcap_sales={n_value_approx}"
        # )
        # return filtered

    # -------------------------------------------------------------
    def determine_market_regime(self, date_idx, active_symbols) -> Optional[RegimeState]:
        current_date = self.close_m.index[date_idx]
        bm_price = self.close_m[self.BENCHMARK_TICKER].iloc[date_idx]
        bm_ma200 = self.sma200_m[self.BENCHMARK_TICKER].iloc[date_idx]
        if pd.isna(bm_ma200):
            return None

        above_ma = bm_price >= bm_ma200

        above_50 = 0
        total = 0
        for sym in active_symbols:
            sma50 = self.sma50_m[sym].iloc[date_idx]
            price = self.close_m[sym].iloc[date_idx]
            if pd.notna(sma50) and pd.notna(price):
                total += 1
                if price > sma50:
                    above_50 += 1
        breadth = (above_50 / total) if total > 0 else 1.0

        if not above_ma and breadth < 0.20:
            exposure = 0.0
        elif (not above_ma) or breadth < 0.40:
            exposure = 0.5
        else:
            exposure = 1.0

        return RegimeState(exposure=exposure, breadth=breadth, benchmark_above_ma200=above_ma)

    def build_factors(self, date_idx, active_symbols, quiet: bool = False) -> pd.DataFrame:
        """CMVS v3 factors: BBS, VZS, CPS, RSIS, RSS (all clipped to [0, 1])."""
        current_date = self.close_m.index[date_idx]
        bm = self.BENCHMARK_TICKER
        qqq_ret_20 = (
            self.ret20_m[bm].iloc[date_idx]
            if bm in self.ret20_m.columns
            else np.nan
        )

        # Relative strength vs benchmark, then cross-sectional Z across today's universe
        rs_map = {}
        for sym in active_symbols:
            price = self.close_m[sym].iloc[date_idx] if sym in self.close_m.columns else np.nan
            if pd.isna(price) or price <= 0:
                continue
            stock_ret_20 = (
                self.ret20_m[sym].iloc[date_idx] if sym in self.ret20_m.columns else np.nan
            )
            if pd.isna(stock_ret_20) or pd.isna(qqq_ret_20):
                continue
            rs_map[sym] = float(stock_ret_20) - float(qqq_ret_20)

        rs_series = pd.Series(rs_map, dtype=float)
        if len(rs_series) >= 2 and float(rs_series.std(ddof=0) or 0.0) > 0:
            rs_z = (rs_series - rs_series.mean()) / rs_series.std(ddof=0)
        else:
            rs_z = pd.Series(0.0, index=rs_series.index, dtype=float)
        rss_map = (rs_z / 3.0).clip(lower=0.0, upper=1.0).to_dict()

        rows = []
        rejected = {"penny": 0, "extreme_vol": 0, "collapsed": 0, "failed_spike": 0}
        for sym in active_symbols:
            price = self.close_m[sym].iloc[date_idx] if sym in self.close_m.columns else np.nan
            if pd.isna(price) or price <= 0:
                continue
            if sym not in rss_map:
                continue

            # --- Hard entry-quality filters (CURX/HKIT/JEM-class traps) ---
            if float(price) < self.MIN_ENTRY_PRICE:
                rejected["penny"] += 1
                continue
            atr_pct = (
                self.atr_pct_m[sym].iloc[date_idx]
                if sym in self.atr_pct_m.columns
                else np.nan
            )
            if pd.notna(atr_pct) and float(atr_pct) > self.MAX_ATR_PCT_ENTRY:
                rejected["extreme_vol"] += 1
                continue
            cvh = (
                self.close_vs_high20_m[sym].iloc[date_idx]
                if sym in self.close_vs_high20_m.columns
                else np.nan
            )
            if pd.notna(cvh) and float(cvh) < self.MIN_CLOSE_VS_HIGH20:
                rejected["collapsed"] += 1
                continue
            ret5 = (
                self.ret5_m[sym].iloc[date_idx] if sym in self.ret5_m.columns else np.nan
            )
            sma20 = (
                self.sma20_m[sym].iloc[date_idx] if sym in self.sma20_m.columns else np.nan
            )
            # Spike already failing: huge 5d gain but already below SMA20
            if (
                pd.notna(ret5)
                and pd.notna(sma20)
                and float(ret5) > self.MAX_SPIKE_5D
                and float(price) < float(sma20)
            ):
                rejected["failed_spike"] += 1
                continue

            bbs = self.bbs_m[sym].iloc[date_idx] if sym in self.bbs_m.columns else np.nan
            vzs = self.vzs_m[sym].iloc[date_idx] if sym in self.vzs_m.columns else np.nan
            cps = self.cps_m[sym].iloc[date_idx] if sym in self.cps_m.columns else np.nan
            rsis = self.rsis_m[sym].iloc[date_idx] if sym in self.rsis_m.columns else np.nan
            rss = rss_map[sym]
            rsi14 = (
                self.rsi14_m[sym].iloc[date_idx] if sym in self.rsi14_m.columns else np.nan
            )
            sma50 = (
                self.sma50_m[sym].iloc[date_idx] if sym in self.sma50_m.columns else np.nan
            )
            up_frac = (
                self.up_frac20_m[sym].iloc[date_idx]
                if sym in self.up_frac20_m.columns
                else np.nan
            )
            ret20 = (
                self.ret20_m[sym].iloc[date_idx] if sym in self.ret20_m.columns else np.nan
            )
            ret10 = (
                self.ret10_m[sym].iloc[date_idx]
                if hasattr(self, "ret10_m") and sym in self.ret10_m.columns
                else np.nan
            )

            def _f(matrix_name, default=np.nan):
                mat = getattr(self, matrix_name, None)
                if mat is None or sym not in mat.columns:
                    return default
                v = mat[sym].iloc[date_idx]
                return float(v) if pd.notna(v) else default

            ema20 = _f("ema20_m")
            ema50 = _f("ema50_m")
            ema20_slope5 = _f("ema20_slope5_m", 0.0)
            ema50_slope5 = _f("ema50_slope5_m", 0.0)
            pullback_pct = _f("pullback_pct_m", 0.0)
            pullback_vol_ratio = _f("pullback_vol_ratio_m", 1.0)
            vol_quiet = _f("vol_quiet_ratio_m", 1.0)
            vol_expand = _f("vol_expand_ratio_m", 1.0)
            vol_spike_persist = _f("vol_spike_persist_m", 0.0)
            range5 = _f("range_pct_5_m")
            range10 = _f("range_pct_10_m")
            range20 = _f("range_pct_20_m")
            atr_shrink = _f("atr_shrink_ratio_m", 1.0)
            rs_raw = _f("rs_vs_bm_m")
            if pd.isna(rs_raw):
                # Fallback: same-day RS used for RSS map construction
                stock_ret_20 = (
                    self.ret20_m[sym].iloc[date_idx] if sym in self.ret20_m.columns else np.nan
                )
                rs_raw = (
                    float(stock_ret_20) - float(qqq_ret_20)
                    if pd.notna(stock_ret_20) and pd.notna(qqq_ret_20)
                    else 0.0
                )
            rs_slope5 = _f("rs_slope5_m", 0.0)
            rs_near_high60 = _f("rs_near_high60_m", 0.5)
            dist_ema20 = _f("dist_ema20_m", 0.0)
            dist_ema50 = _f("dist_ema50_m", 0.0)
            above_ema50 = _f("above_ema50_m", 0.0)
            green_streak = _f("green_streak5_m", 0.0)

            if any(pd.isna(x) for x in (bbs, vzs, cps, rsis, rss)):
                continue

            # vol_raw retained for optional inverse-vol sizing revert path
            vol60 = self.vol60_m[sym].iloc[date_idx] if sym in self.vol60_m.columns else np.nan

            # Trend structure: sustainable uptrend preferred over spike
            if pd.notna(sma20) and pd.notna(sma50) and float(price) > float(sma20) > float(sma50):
                trend_score = 1.0
            elif pd.notna(sma20) and float(price) > float(sma20):
                trend_score = 0.65
            elif pd.notna(sma50) and float(price) > float(sma50):
                trend_score = 0.35
            else:
                trend_score = 0.05

            rows.append({
                "symbol": sym,
                "bbs": float(np.clip(bbs, 0.0, 1.0)),
                "vzs": float(np.clip(vzs, 0.0, 1.0)),
                "cps": float(np.clip(cps, 0.0, 1.0)),
                "rsis": float(np.clip(rsis, 0.0, 1.0)),
                "rss": float(np.clip(rss, 0.0, 1.0)),
                "vol_raw": vol60,
                "price": float(price),
                "ret5": float(ret5) if pd.notna(ret5) else 0.0,
                "ret10": float(ret10) if pd.notna(ret10) else 0.0,
                "ret20": float(ret20) if pd.notna(ret20) else 0.0,
                "atr_pct": float(atr_pct) if pd.notna(atr_pct) else 0.0,
                "close_vs_high20": float(cvh) if pd.notna(cvh) else 1.0,
                "rsi14": float(rsi14) if pd.notna(rsi14) else 50.0,
                "trend_score": float(trend_score),
                "up_frac20": float(up_frac) if pd.notna(up_frac) else 0.5,
                "industry": self.profile_meta.get(sym, {}).get("industry", "Unknown"),
                # --- EQS feature columns (PIT as-of date_idx) ---
                "ema20": float(ema20) if pd.notna(ema20) else float(price),
                "ema50": float(ema50) if pd.notna(ema50) else float(price),
                "ema20_slope5": float(ema20_slope5),
                "ema50_slope5": float(ema50_slope5),
                "pullback_pct": float(pullback_pct) if pd.notna(pullback_pct) else 0.0,
                "pullback_vol_ratio": float(pullback_vol_ratio) if pd.notna(pullback_vol_ratio) else 1.0,
                "vol_quiet_ratio": float(vol_quiet) if pd.notna(vol_quiet) else 1.0,
                "vol_expand_ratio": float(vol_expand) if pd.notna(vol_expand) else 1.0,
                "vol_spike_persist": float(vol_spike_persist) if pd.notna(vol_spike_persist) else 0.0,
                "range_pct_5": float(range5) if pd.notna(range5) else np.nan,
                "range_pct_10": float(range10) if pd.notna(range10) else np.nan,
                "range_pct_20": float(range20) if pd.notna(range20) else np.nan,
                "atr_shrink_ratio": float(atr_shrink) if pd.notna(atr_shrink) else 1.0,
                "rs_raw": float(rs_raw),
                "rs_slope5": float(rs_slope5),
                "rs_near_high60": float(rs_near_high60) if pd.notna(rs_near_high60) else 0.5,
                "dist_ema20": float(dist_ema20) if pd.notna(dist_ema20) else 0.0,
                "dist_ema50": float(dist_ema50) if pd.notna(dist_ema50) else 0.0,
                "above_ema50": float(above_ema50) if pd.notna(above_ema50) else 0.0,
                "green_streak": float(green_streak),
            })
            # Recovered fundamental fields (never used to drop the name)
            fund = extract_fundamental_metrics(self, sym, current_date, float(price))
            rows[-1].update(
                {
                    "roic_raw": fund["roic_raw"],
                    "op_margin_raw": fund["op_margin_raw"],
                    "gross_prof_raw": fund["gross_prof_raw"],
                    "rev_growth_raw": fund["rev_growth_raw"],
                    "fcf_sales_yield": fund["fcf_sales_yield"],
                    "debt_to_equity": fund["debt_to_equity"],
                    "has_fundamentals": bool(fund["has_fundamentals"]),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.dropna(subset=["bbs", "vzs", "cps", "rsis", "rss"])
        if not quiet:
            print(
                f"    [FACTORS] {current_date.date()} cmvs_v3={len(df)} "
                f"(BBS/VZS/CPS/RSIS/RSS + EQS features; "
                f"rejected penny={rejected['penny']} vol={rejected['extreme_vol']} "
                f"collapse={rejected['collapsed']} spike={rejected['failed_spike']})"
            )
        return df

    def rank_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Live ranking: CMVS + technical EQS + optional fundamental bonus points.

        CMVSScore (baseline) = CMVS + EQS_BLEND_WEIGHT * technical_EQS
        FinalScore           = CMVSScore + FundamentalBonusPoints * EQS_WEIGHT

        Fundamental bonus reuses commented get_universe thresholds / quality
        ranks as signed points (~[-10, +20]); never excludes names.
        Missing metrics → 0. ``EQS_WEIGHT`` is the point→score scale (default
        0.01 so +20 ≈ +0.20 ≈ 15–20% influence). Scale 0 → identical baseline.
        EXIT_RANK buffer unchanged — small bonus tilts should not force churn.
        Hard rejects remain in ``build_factors`` (price/volume traps only).
        """
        if df.empty:
            return df
        df = df.copy()

        # ----- CMVS quality core (unchanged) -----
        ret5 = df["ret5"] if "ret5" in df.columns else 0.0
        vzs_confirmed = df["vzs"] * np.where(ret5 > 0.0, 1.0, 0.30)
        rsi = df["rsi14"] if "rsi14" in df.columns else (df["rsis"] * 30.0 + 50.0)
        rsis_quality = (1.0 - ((rsi - 58.0).abs() / 30.0)).clip(lower=0.0, upper=1.0)
        bbs_damped = df["bbs"] * np.where(ret5 > 0.20, 0.45, 1.0)
        trend = df["trend_score"] if "trend_score" in df.columns else 0.5
        persist = df["up_frac20"] if "up_frac20" in df.columns else 0.5

        cmvs_raw = (
            (self.w5 * df["rss"])
            + (0.20 * trend)
            + (0.15 * persist)
            + (self.w3 * df["cps"])
            + (self.w2 * vzs_confirmed)
            + (self.w1 * bbs_damped)
            + (self.w4 * rsis_quality)
        )
        atr_pct = df["atr_pct"] if "atr_pct" in df.columns else 0.0
        cvh = df["close_vs_high20"] if "close_vs_high20" in df.columns else 1.0
        cmvs_penalty = (
            np.maximum(0.0, ret5 - 0.20) * 1.25
            + np.maximum(0.0, 0.85 - cvh) * 1.50
            + np.maximum(0.0, atr_pct - 0.08) * 2.00
            + np.maximum(0.0, (rsi - 72.0) / 28.0) * 0.40
        )
        cmvs = (cmvs_raw - cmvs_penalty).astype(float)
        df["cmvs_score"] = cmvs

        # ----- Entry Quality Score (modular; live ranking) -----
        eqs_w = getattr(self, "eqs_weights", None) or EQSWeights()
        blend_w = float(getattr(self, "EQS_BLEND_WEIGHT", 0.55))
        comps = compute_eqs_components(df)
        for c in comps.columns:
            df[c] = comps[c]
        eqs = combine_eqs(comps, eqs_w)
        df["eqs"] = eqs
        baseline_final = blend_cmvs_eqs(cmvs, eqs, blend_w)

        # ----- Fundamental bonus points (recovered screens; never excludes) -----
        fund_pts, fund_parts = compute_fundamental_bonus_points(
            df,
            min_roic=float(getattr(self, "MIN_ROIC", 0.03)),
            min_fcf_sales_yield=float(getattr(self, "MIN_FCF_SALES_YIELD", -0.05)),
            max_debt_to_equity=float(getattr(self, "MAX_DEBT_TO_EQUITY", 3.0)),
            min_revenue_growth_yoy=float(getattr(self, "MIN_REVENUE_GROWTH_YOY", -0.15)),
            min_industry_size=int(getattr(self, "MIN_INDUSTRY_SIZE", 8)),
        )
        df["fundamental_bonus_points"] = fund_pts
        for k, s in fund_parts.items():
            df[k] = s
        # Normalized view for rank diagnostics (0..1 from [-10,+20])
        df["quality_score"] = ((fund_pts - (-10.0)) / 30.0).clip(0.0, 1.0)
        use_q = bool(getattr(self, "USE_QUALITY_BONUS", True))
        point_scale = float(getattr(self, "EQS_WEIGHT", DEFAULT_FUND_POINT_SCALE))
        q_bonus = quality_bonus(
            fund_pts, use_quality_bonus=use_q, eqs_weight=point_scale, input_is_points=True
        )
        df["quality_bonus"] = q_bonus
        df["fundamental_bonus"] = q_bonus
        # point_scale == 0 (or bonus off) → bit-identical to baseline_final
        df["final_score"] = (baseline_final + q_bonus).astype(float)
        df["factor_mode"] = "cmvs_v3_eqs"
        self._last_fund_points_map = {
            str(sym): float(p)
            for sym, p in zip(df["symbol"].tolist(), df["fundamental_bonus_points"].tolist())
        }
        self._last_quality_map = {
            str(sym): float(q)
            for sym, q in zip(df["symbol"].tolist(), df["quality_score"].tolist())
        }
        self._last_quality_bonus_map = {
            str(sym): float(b)
            for sym, b in zip(df["symbol"].tolist(), df["quality_bonus"].tolist())
        }
        self._last_quality_diag = format_quality_diagnostics(
            df["fundamental_bonus_points"], eqs_weight=point_scale, use_quality_bonus=use_q
        )
        return df.sort_values("final_score", ascending=False).reset_index(drop=True)

    def construct_portfolio(self, ranked_df, date_idx, ctx: RebalanceContext) -> pd.DataFrame:
        if ranked_df.empty:
            return ranked_df
        # ENTRY_RANK / EXIT_RANK are research aliases; at defaults == max_portfolio / buffer
        top_core = set(ranked_df.head(self.ENTRY_RANK)["symbol"].tolist())
        top_buffer = set(ranked_df.head(self.EXIT_RANK)["symbol"].tolist())

        keep = self.previous_target_symbols & top_buffer
        new_candidates = ranked_df[
            ranked_df["symbol"].isin(top_core) & ~ranked_df["symbol"].isin(keep)
        ]

        # Adaptive re-entry cooldown: never re-buy losing stop-outs until strength returns
        blocked = self._cooldown_blocked_set(date_idx)
        if blocked:
            before = len(new_candidates)
            new_candidates = new_candidates[~new_candidates["symbol"].isin(blocked)]
            n_blocked = before - len(new_candidates)
            if n_blocked > 0:
                print(
                    f"    [COOLDOWN] blocked {n_blocked} re-entries "
                    f"(active={len(self.reentry_cooldown._records)})"
                )

        selected = list(keep)
        for _, row in new_candidates.iterrows():
            if len(selected) >= self.ENTRY_RANK:
                break
            selected.append(row["symbol"])

        targets = ranked_df[ranked_df["symbol"].isin(selected)].copy()

        candidate_syms = targets["symbol"].tolist()
        window = self.close_m[candidate_syms].iloc[date_idx - self.CORR_WINDOW: date_idx].pct_change().dropna(how="all")
        corr_matrix = window.corr(method="spearman") if not window.empty else pd.DataFrame()

        final_selected = []
        for _, row in targets.sort_values("final_score", ascending=False).iterrows():
            sym = row["symbol"]
            if sym in keep:
                final_selected.append(sym)
                continue
            correlated = False
            if not corr_matrix.empty and sym in corr_matrix.columns:
                for held in final_selected:
                    if held in corr_matrix.columns and pd.notna(corr_matrix.loc[sym, held]):
                        if corr_matrix.loc[sym, held] >= self.CORR_THRESHOLD:
                            correlated = True
                            break
            if correlated:
                ctx.correlation_rejections.append(sym)
            else:
                final_selected.append(sym)

        return ranked_df[ranked_df["symbol"].isin(final_selected)].copy()

    def allocate_weights(self, targets: pd.DataFrame, date_idx: int, exposure: float) -> pd.DataFrame:
        """Position sizing for selected names.

        Aggressive test: equal weighting (inverse-vol / ATR sizing commented out
        below so it can be restored if MDD becomes too high).
        """
        if targets.empty:
            return targets
        targets = targets.copy()

        # --- Aggressive test: equal weight all selected names ---
        n = max(len(targets), 1)
        targets["raw_weight"] = 1.0 / n
        targets["final_weight"] = targets["raw_weight"] * exposure
        return targets

        # --- Original inverse-volatility / 20-day ATR weighting (kept for easy revert) ---
        # atr_pcts = []
        # for sym in targets["symbol"]:
        #     atr = (
        #         self.atr20_m[sym].iloc[date_idx]
        #         if hasattr(self, "atr20_m") and sym in self.atr20_m.columns
        #         else np.nan
        #     )
        #     px = (
        #         self.close_m[sym].iloc[date_idx]
        #         if sym in self.close_m.columns
        #         else np.nan
        #     )
        #     if pd.notna(atr) and pd.notna(px) and float(px) > 0:
        #         atr_pcts.append(float(atr) / float(px))
        #     else:
        #         # Fallback to existing vol_raw if ATR unavailable for a name
        #         row = targets.loc[targets["symbol"] == sym, "vol_raw"]
        #         atr_pcts.append(float(row.iloc[0]) if len(row) and pd.notna(row.iloc[0]) else np.nan)
        #
        # targets["atr20_pct"] = atr_pcts
        # safe_vol = targets["atr20_pct"].astype(float)
        # if safe_vol.isna().any():
        #     med = safe_vol.median()
        #     safe_vol = safe_vol.fillna(med if pd.notna(med) else self.VOL_FLOOR)
        # safe_vol = safe_vol.clip(lower=self.VOL_FLOOR)
        #
        # targets["inv_vol"] = 1.0 / safe_vol
        # targets["raw_weight"] = targets["inv_vol"] / targets["inv_vol"].sum()
        # max_single = (1.0 / self.MAX_PORTFOLIO_SIZE) * 2.0
        # targets["raw_weight"] = targets["raw_weight"].clip(upper=max_single)
        # targets["final_weight"] = (targets["raw_weight"] / targets["raw_weight"].sum()) * exposure
        # return targets

    def apply_risk_adjustments(self, targets: pd.DataFrame, exposure: float, date_idx: int = None) -> pd.DataFrame:
        # Prefer allocate_weights when a date index is available.
        if date_idx is not None:
            return self.allocate_weights(targets, date_idx, exposure)
        if targets.empty:
            return targets
        targets = targets.copy()

        # --- Aggressive test: equal weight (inverse-vol commented out for easy revert) ---
        n = max(len(targets), 1)
        targets["raw_weight"] = 1.0 / n
        targets["final_weight"] = targets["raw_weight"] * exposure
        return targets

        # --- Original inverse-volatility weighting (kept for easy revert) ---
        # safe_vol = targets["vol_raw"].clip(lower=self.VOL_FLOOR)
        # targets["inv_vol"] = 1.0 / safe_vol
        # targets["raw_weight"] = targets["inv_vol"] / targets["inv_vol"].sum()
        # max_single = (1.0 / self.MAX_PORTFOLIO_SIZE) * 2.0
        # targets["raw_weight"] = targets["raw_weight"].clip(upper=max_single)
        # targets["final_weight"] = (targets["raw_weight"] / targets["raw_weight"].sum()) * exposure
        # return targets

    def validate_portfolio(self, final_targets: pd.DataFrame, exposure: float):
        issues = []
        if final_targets["symbol"].duplicated().any():
            issues.append("중복 심볼")
        if len(final_targets) > self.ENTRY_RANK:
            issues.append(f"포트폴리오 크기 초과 {len(final_targets)}")
        total_w = final_targets["final_weight"].sum()
        if abs(total_w - exposure) > self.WEIGHT_SUM_TOLERANCE:
            issues.append(f"총비중 불일치 {total_w:.3f} vs {exposure:.3f}")
        max_single = (1.0 / max(self.ENTRY_RANK, 1)) * 2.0
        if (final_targets["final_weight"] > max_single + 1e-6).any():
            issues.append("개별 상한 초과")
        ind_exp = final_targets.groupby("industry")["final_weight"].sum()
        if (ind_exp > self.MAX_INDUSTRY_WEIGHT * exposure + 1e-6).any():
            issues.append(f"산업 상한 초과 {ind_exp[ind_exp > self.MAX_INDUSTRY_WEIGHT * exposure].to_dict()}")
        if issues:
            print(f"    [VALIDATION 경고] {'; '.join(issues)}")
        return issues

    def construct_final_targets_with_industry_cap(self, targets, exposure, ctx):
        """industry cap 순차 배분 + 초과분은 명시적으로 제외(=이후 청산 대상)."""
        industry_tracker = {}
        rows = []
        for _, row in targets.sort_values("final_score", ascending=False).iterrows():
            ind = row["industry"]; w = row["final_weight"]
            cur = industry_tracker.get(ind, 0.0)
            if cur + w > self.MAX_INDUSTRY_WEIGHT * exposure:
                ctx.industry_cap_rejections.append(row["symbol"])
                continue
            industry_tracker[ind] = cur + w
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=targets.columns)

    # -------------------------------------------------------------
    def queue_rebalance_orders(self, final_targets: pd.DataFrame, date_idx: int):
        """목표비중 대비 델타를 다음날 시가 체결 주문으로 큐잉 (LEAN SetHoldings 대응)."""
        current_date = self.close_m.index[date_idx]
        current_closes = self.close_m.loc[current_date]
        total_equity = self.cash + sum(
            qty * current_closes.get(s, np.nan) for s, qty in self.portfolio.items() if pd.notna(current_closes.get(s))
        )

        target_weight_map = dict(zip(final_targets["symbol"], final_targets["final_weight"])) if not final_targets.empty else {}

        all_symbols = set(self.portfolio.keys()) | set(target_weight_map.keys())
        for sym in all_symbols:
            price = current_closes.get(sym, np.nan)
            if pd.isna(price) or price <= 0:
                continue
            target_value = target_weight_map.get(sym, 0.0) * total_equity
            current_value = self.portfolio.get(sym, 0) * price
            delta_value = target_value - current_value
            if abs(delta_value) < total_equity * 0.001:  # 미미한 리밸런싱 무시 (거래비용 절감)
                continue
            qty = int(abs(delta_value) / price)
            if qty <= 0:
                continue
            order_type = "BUY" if delta_value > 0 else "SELL"
            self.pending_orders.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "type": order_type,
                    "reason": "rebalance_entry" if order_type == "BUY" else "rebalance_exit",
                }
            )

    def _symbol_snapshot(self, symbol: str, date_idx: int) -> dict:
        rsi = (
            float(self.rsi14_m[symbol].iloc[date_idx])
            if symbol in self.rsi14_m.columns and pd.notna(self.rsi14_m[symbol].iloc[date_idx])
            else None
        )
        atr = (
            float(self.atr14_m[symbol].iloc[date_idx])
            if symbol in self.atr14_m.columns and pd.notna(self.atr14_m[symbol].iloc[date_idx])
            else None
        )
        return {
            "rank": self._last_rank_map.get(symbol),
            "cmvs": self._last_cmvs_map.get(symbol),
            "rsi": rsi,
            "atr": atr,
        }

    def _strength_snap(self, symbol: str, date_idx: int) -> dict:
        """Market state used to decide adaptive cooldown clearance."""
        def _get(matrix, default=None):
            if symbol not in matrix.columns:
                return default
            v = matrix[symbol].iloc[date_idx]
            if pd.isna(v):
                return default
            return float(v)

        return {
            "close": _get(self.close_m),
            "sma20": _get(self.sma20_m),
            "sma50": _get(self.sma50_m),
            "rsi14": _get(self.rsi14_m),
            "ret5": _get(self.ret5_m),
            "close_vs_high20": _get(self.close_vs_high20_m),
            "atr_pct": _get(self.atr_pct_m),
        }

    def _cooldown_blocked_set(self, date_idx: int) -> set:
        if not getattr(self, "reentry_cooldown", None):
            return set()
        return self.reentry_cooldown.blocked_symbols(
            date_idx, lambda sym: self._strength_snap(sym, date_idx)
        )

    def _cost_ratio(self, symbol, date_idx, qty, price):
        adv = self.adv20_m[symbol].iloc[date_idx]
        sigma = self.vol20_m[symbol].iloc[date_idx]
        cs = self.cs_spread_m[symbol].iloc[date_idx] if symbol in self.cs_spread_m.columns else np.nan
        adv = adv if pd.notna(adv) and adv > 0 else 1e6
        sigma = sigma if pd.notna(sigma) and sigma > 0 else 0.02
        half_spread = float(cs) / 2 if pd.notna(cs) and cs > 0 else 0.0015
        participation = (qty * price) / max(adv, 1.0)
        impact = 0.6 * sigma * np.sqrt(max(participation, 0.0))
        # Append fixed commission (0.05%) and slippage (0.02%) on every trade
        return float(half_spread + impact + self.COMMISSION_RATE + self.SLIPPAGE_RATE)

    def _max_shares_participation(self, symbol, date_idx, price, cap_ratio):
        adv = self.adv20_m[symbol].iloc[date_idx]
        if pd.isna(adv) or adv <= 0 or price <= 0:
            return 0
        return int((adv * cap_ratio) / price)

    def execute_pending_orders(self, date_idx):
        if not self.pending_orders:
            return
        current_date = self.close_m.index[date_idx]
        current_opens = self.open_m.loc[current_date]

        for order in self.pending_orders:
            sym, qty = order["symbol"], order["qty"]
            o_price = current_opens.get(sym, np.nan)
            if pd.isna(o_price) or o_price <= 0:
                continue
            if order["type"] == "BUY":
                # Belt-and-suspenders: never fill a buy while on adaptive cooldown
                if self.reentry_cooldown.is_blocked(
                    sym, date_idx, self._strength_snap(sym, date_idx)
                ):
                    continue
                cap = self._max_shares_participation(sym, date_idx, o_price, self.PARTICIPATION_CAP_BUY)
                exec_qty = min(qty, cap)
                if exec_qty <= 0:
                    continue
                cost_ratio = self._cost_ratio(sym, date_idx, exec_qty, o_price)
                total_cost = exec_qty * o_price * (1 + cost_ratio)
                if self.cash >= total_cost:
                    was_held = self.portfolio.get(sym, 0) > 0
                    self.portfolio[sym] = self.portfolio.get(sym, 0) + exec_qty
                    self.cash -= total_cost
                    # New position: seed peak with entry open (updated to closes in exit check)
                    if not was_held:
                        self.highest_prices[sym] = float(o_price)
                    # Research Fact: journal entry (observation only)
                    snap = self._symbol_snapshot(sym, date_idx)
                    self.trade_journal.on_entry(
                        symbol=sym,
                        date=current_date,
                        date_idx=date_idx,
                        price=float(o_price),
                        qty=int(exec_qty),
                        rank=snap["rank"],
                        cmvs=snap["cmvs"],
                        rsi=snap["rsi"],
                        atr=snap["atr"],
                    )
            else:
                if sym in self.portfolio:
                    cap = self._max_shares_participation(sym, date_idx, o_price, self.PARTICIPATION_CAP_SELL)
                    exec_qty = min(self.portfolio[sym], qty, max(cap, 0))
                    if exec_qty <= 0:
                        continue
                    cost_ratio = self._cost_ratio(sym, date_idx, exec_qty, o_price)
                    self.cash += exec_qty * o_price * (1 - cost_ratio)
                    self.portfolio[sym] -= exec_qty
                    if self.portfolio[sym] <= 0:
                        del self.portfolio[sym]
                        self.highest_prices.pop(sym, None)
                        # Research Fact: full exit + shadow counterfactual
                        snap = self._symbol_snapshot(sym, date_idx)
                        closed = self.trade_journal.on_exit(
                            symbol=sym,
                            date=current_date,
                            date_idx=date_idx,
                            price=float(o_price),
                            exit_reason=str(order.get("reason") or "sell"),
                            close_m=self.close_m,
                            exit_rank=snap["rank"],
                            exit_cmvs=snap["cmvs"],
                            exit_rsi=snap["rsi"],
                            exit_atr=snap["atr"],
                        )
                        # Priority 1: arm adaptive cooldown after losing exits
                        if closed is not None and float(closed.final_return) < 0.0:
                            atr_pct = (
                                float(self.atr_pct_m[sym].iloc[date_idx])
                                if sym in self.atr_pct_m.columns
                                and pd.notna(self.atr_pct_m[sym].iloc[date_idx])
                                else 0.0
                            )
                            vol20 = (
                                float(self.vol20_m[sym].iloc[date_idx])
                                if sym in self.vol20_m.columns
                                and pd.notna(self.vol20_m[sym].iloc[date_idx])
                                else 0.0
                            )
                            rec = self.reentry_cooldown.record_exit(
                                symbol=sym,
                                date_idx=date_idx,
                                exit_price=float(o_price),
                                exit_reason=str(closed.exit_reason),
                                final_return=float(closed.final_return),
                                atr_pct=atr_pct,
                                vol20=vol20,
                                peak_return=float(closed.peak_return),
                            )
                            if rec is not None:
                                print(
                                    f"    [COOLDOWN ARM] {sym} loss={closed.final_return:.1%} "
                                    f"reason={_short_reason(closed.exit_reason)} "
                                    f"holdout={rec.hard_expiry_idx - date_idx}d "
                                    f"streak={rec.consecutive_losses}"
                                )
        self.pending_orders = []

    def check_cmvs_exits(self, date_idx):
        """Daily exits: ATR trail (primary) + confirmed trend exit + exhaustion.

        EMA9/short-EMA break alone never sells. Trend exits require multiple
        confirmations; high-ATR% names use EMA20 and a higher confirmation bar.
        Healthy long-term uptrends skip trend exits on shallow pullbacks.
        """
        current_date = self.close_m.index[date_idx]
        current_closes = self.close_m.loc[current_date]
        pending_sell_syms = {
            o["symbol"] for o in self.pending_orders if o.get("type") == "SELL"
        }

        for sym in list(self.portfolio.keys()):
            if sym in pending_sell_syms:
                continue
            price = current_closes.get(sym, np.nan)
            if pd.isna(price) or price <= 0:
                continue
            close_px = float(price)

            # Track highest close since entry
            if sym not in self.highest_prices:
                self.highest_prices[sym] = close_px
            else:
                self.highest_prices[sym] = max(float(self.highest_prices[sym]), close_px)
            peak = float(self.highest_prices[sym])
            self.trade_journal.mark_peak(sym, close_px)

            # Min-hold gate (default 0 → no change)
            if self.MIN_HOLD_DAYS > 0:
                held = self.trade_journal.holding_days(sym, current_date)
                if held < self.MIN_HOLD_DAYS:
                    continue

            atr14 = (
                self.atr14_m[sym].iloc[date_idx]
                if sym in self.atr14_m.columns
                else np.nan
            )
            rsi14 = (
                self.rsi14_m[sym].iloc[date_idx]
                if sym in self.rsi14_m.columns
                else np.nan
            )
            daily_cp = (
                self.daily_cp_m[sym].iloc[date_idx]
                if sym in self.daily_cp_m.columns
                else np.nan
            )

            reasons = []
            # 1) ATR trailing stop — primary catastrophic exit (standalone OK)
            if self.USE_ATR_EXIT and pd.notna(atr14) and atr14 > 0:
                stop_level = peak - (self.atr_multiplier * float(atr14))
                if close_px < stop_level:
                    reasons.append(f"atr_trail(stop={stop_level:.2f})")

            # 2) Confirmation-based trend exit (EMA9 alone never sells)
            if self.USE_EMA9_EXIT:
                conf = self._trend_exit_confirmations(sym, date_idx, close_px, peak, atr14)
                if conf:
                    reasons.append("trend_confirm(" + "+".join(conf) + ")")

            # 3) Exhaustion: rsi_14 > 80 AND daily_cp < 0.3
            if (
                self.USE_EXHAUSTION_EXIT
                and pd.notna(rsi14)
                and pd.notna(daily_cp)
                and float(rsi14) > 80.0
                and float(daily_cp) < 0.3
            ):
                reasons.append(f"exhaustion(rsi={float(rsi14):.1f},cp={float(daily_cp):.2f})")
            # 4) Optional time stop (default OFF — no baseline impact)
            if self.USE_TIME_STOP and self.TIME_STOP_DAYS > 0:
                held = self.trade_journal.holding_days(sym, current_date)
                if held >= self.TIME_STOP_DAYS:
                    reasons.append(f"time_stop(days={held})")

            if not reasons:
                continue

            qty = int(self.portfolio[sym])
            if qty <= 0:
                continue
            reason = ",".join(reasons)
            self.pending_orders.append(
                {"symbol": sym, "qty": qty, "type": "SELL", "reason": reason}
            )
            self.previous_target_symbols.discard(sym)
            self._cmvs_forced_exits.add(sym)
            print(
                f"    🛑 [CMVS EXIT] {sym} close={close_px:.2f} peak={peak:.2f} "
                f"reasons={reason} -> SELL queued"
            )

        # --- Prior hardcoded % trailing stop (commented; kept for revert) ---
        # def check_trailing_stops(self, date_idx):
        #     ... sell if close <= peak * (1 - TRAILING_STOP_PCT) ...

    def _in_healthy_long_trend_pullback(
        self, sym: str, date_idx: int, close_px: float, peak: float, atr14
    ) -> bool:
        """True when weakness looks like a normal pullback inside a healthy uptrend.

        Used to suppress confirmation trend exits (not ATR catastrophic stops).
        """
        if not getattr(self, "PROTECT_HEALTHY_TREND_PULLBACK", True):
            return False
        sma50 = (
            self.sma50_m[sym].iloc[date_idx]
            if sym in self.sma50_m.columns
            else np.nan
        )
        slope = (
            self.sma50_slope5_m[sym].iloc[date_idx]
            if hasattr(self, "sma50_slope5_m") and sym in self.sma50_slope5_m.columns
            else np.nan
        )
        rsi = (
            self.rsi14_m[sym].iloc[date_idx]
            if sym in self.rsi14_m.columns
            else np.nan
        )
        if pd.isna(sma50) or float(close_px) < float(sma50):
            return False
        # SMA50 still rising (or flat) — long-term trend intact
        if pd.isna(slope) or float(slope) < -0.005:
            return False
        # Shallow pullback vs peak (within ~2 ATR or 8%)
        if peak > 0:
            dd = 1.0 - float(close_px) / float(peak)
            atr_band = (
                (2.0 * float(atr14) / float(peak))
                if pd.notna(atr14) and atr14 > 0
                else 0.08
            )
            if dd > max(0.08, atr_band):
                return False
        # RSI not structurally broken
        if pd.notna(rsi) and float(rsi) < 40.0:
            return False
        return True

    def _trend_exit_confirmations(
        self, sym: str, date_idx: int, close_px: float, peak: float, atr14
    ) -> list:
        """Return confirmation tags if a multi-signal trend exit should fire.

        EMA9 (or configured short EMA) break alone is never enough.
        High ATR% names use EMA20 as the primary structure line and need more votes.
        """
        if self._in_healthy_long_trend_pullback(sym, date_idx, close_px, peak, atr14):
            return []

        atr_pct = (
            self.atr_pct_m[sym].iloc[date_idx]
            if hasattr(self, "atr_pct_m") and sym in self.atr_pct_m.columns
            else np.nan
        )
        high_vol = pd.notna(atr_pct) and float(atr_pct) >= float(
            getattr(self, "HIGH_ATR_PCT_EXIT", 0.08)
        )
        min_need = int(
            getattr(self, "TREND_EXIT_MIN_CONFIRM_HIGH_VOL", 3)
            if high_vol
            else getattr(self, "TREND_EXIT_MIN_CONFIRM", 2)
        )

        # Primary structure EMA: short EMA for normal names, EMA20 for high-vol
        ema_short = (
            self.ema_exit_m[sym].iloc[date_idx]
            if sym in self.ema_exit_m.columns
            else np.nan
        )
        ema20 = (
            self.ema20_m[sym].iloc[date_idx]
            if hasattr(self, "ema20_m") and sym in self.ema20_m.columns
            else np.nan
        )
        primary = ema20 if high_vol else ema_short
        primary_label = "ema20" if high_vol else f"ema{int(getattr(self, 'EMA_EXIT_LENGTH', 9))}"

        def _close_at(i):
            if i < 0 or sym not in self.close_m.columns:
                return np.nan
            v = self.close_m[sym].iloc[i]
            return float(v) if pd.notna(v) else np.nan

        def _ema_primary_at(i):
            mat = self.ema20_m if high_vol else self.ema_exit_m
            if i < 0 or sym not in mat.columns:
                return np.nan
            v = mat[sym].iloc[i]
            return float(v) if pd.notna(v) else np.nan

        confirms = []

        # A) Two consecutive closes below primary EMA (not a single touch)
        c0, c1 = _close_at(date_idx), _close_at(date_idx - 1)
        e0, e1 = _ema_primary_at(date_idx), _ema_primary_at(date_idx - 1)
        if (
            pd.notna(c0)
            and pd.notna(c1)
            and pd.notna(e0)
            and pd.notna(e1)
            and c0 < e0
            and c1 < e1
        ):
            confirms.append(f"two_closes_below_{primary_label}")

        # B) Close below EMA20 (structure break) — always a confirmation vote
        if pd.notna(ema20) and close_px < float(ema20):
            confirms.append("ema20_break")

        # C) RSI soft breakdown
        rsi = (
            self.rsi14_m[sym].iloc[date_idx]
            if sym in self.rsi14_m.columns
            else np.nan
        )
        if pd.notna(rsi) and float(rsi) < 45.0:
            confirms.append(f"rsi_lt45({float(rsi):.0f})")

        # D) Expanding volume on a down day
        ret1 = (
            self.ret1_m[sym].iloc[date_idx]
            if hasattr(self, "ret1_m") and sym in self.ret1_m.columns
            else np.nan
        )
        rel_vol = (
            self.rel_vol_m[sym].iloc[date_idx]
            if hasattr(self, "rel_vol_m") and sym in self.rel_vol_m.columns
            else np.nan
        )
        if pd.notna(ret1) and float(ret1) < 0.0 and pd.notna(rel_vol) and float(rel_vol) >= 1.25:
            confirms.append(f"vol_decline(rv={float(rel_vol):.2f})")

        # E) Negative short-term momentum
        ret5 = (
            self.ret5_m[sym].iloc[date_idx]
            if sym in self.ret5_m.columns
            else np.nan
        )
        if pd.notna(ret5) and float(ret5) < 0.0:
            confirms.append(f"neg_mom5({float(ret5):.1%})")

        # Guard: never sell on a lone short-EMA pierce — require enough independent votes
        if len(confirms) < min_need:
            return []
        # High-vol: must include EMA20 structure break among the votes
        if high_vol and "ema20_break" not in confirms and not any(
            t.startswith("two_closes_below_ema20") for t in confirms
        ):
            return []
        return confirms

    def _ema9_break_signal(self, sym: str, date_idx: int) -> bool:
        """Observation Fact: raw close < short EMA (does not imply an exit order)."""
        current_date = self.close_m.index[date_idx]
        price = self.close_m.loc[current_date].get(sym, np.nan)
        if pd.isna(price) or price <= 0:
            return False
        ema9 = (
            self.ema9_m[sym].iloc[date_idx]
            if sym in self.ema9_m.columns
            else np.nan
        )
        return bool(pd.notna(ema9) and float(price) < float(ema9))

    def _confirmed_trend_exit_signal(self, sym: str, date_idx: int) -> bool:
        """True when confirmation-based trend exit would queue a sell."""
        price = self.close_m[sym].iloc[date_idx] if sym in self.close_m.columns else np.nan
        if pd.isna(price) or price <= 0:
            return False
        peak = float(self.highest_prices.get(sym, price))
        atr14 = (
            self.atr14_m[sym].iloc[date_idx]
            if sym in self.atr14_m.columns
            else np.nan
        )
        return bool(self._trend_exit_confirmations(sym, date_idx, float(price), peak, atr14))

    def _observe_rank_diagnostics(self, date_idx: int) -> None:
        """Daily observation-only rank diagnostics for current holdings.

        Does not queue orders, mutate targets, or change exit decisions.
        Counts ``rank > EXIT_RANK`` even when another exit rule also fired.
        """
        if not self.portfolio:
            return
        current_date = self.close_m.index[date_idx]
        current_closes = self.close_m.loc[current_date]
        holdings = [s for s, qty in self.portfolio.items() if int(qty) > 0]
        if not holdings:
            return

        active_symbols = [
            s
            for s in self.close_m.columns
            if s != self.BENCHMARK_TICKER
            and pd.notna(current_closes.get(s))
            and not self.profile_meta.get(s, {}).get("isEtf", False)
        ]
        if not active_symbols:
            return

        top_adv_symbols = (
            self.dvol_m.loc[current_date, active_symbols]
            .nlargest(min(self.TOP_ADV_POOL, len(active_symbols)))
            .index.tolist()
        )
        # Union holdings so held names remain rankable on non-rebalance days.
        pool = list(dict.fromkeys(list(top_adv_symbols) + holdings))
        universe_symbols = self.get_universe(pool, date_idx, quiet=True)
        for h in holdings:
            if h not in universe_symbols:
                universe_symbols.append(h)

        factor_df = self.build_factors(date_idx, universe_symbols, quiet=True)
        if factor_df.empty:
            return
        ranked_df = self.rank_universe(factor_df)
        rank_map = {
            str(sym): int(i + 1)
            for i, sym in enumerate(ranked_df["symbol"].tolist())
        }

        for sym in holdings:
            self.rank_diagnostics.observe(
                date=current_date,
                ticker=str(sym),
                rank=rank_map.get(str(sym)),
                ema9_break=self._ema9_break_signal(sym, date_idx),
                quality_score=self._last_quality_map.get(str(sym)),
                quality_bonus=self._last_quality_bonus_map.get(str(sym)),
            )

    def handle_official_delisting(self, date_idx):
        """공식 상장폐지일 도래 시 즉시(당일 종가) 강제 청산 - LEAN의 Delisting.Warning 대응."""
        current_date = self.close_m.index[date_idx]
        for sym in list(self.portfolio.keys()):
            meta = self.delisted_meta.get(sym)
            if meta and meta.get("delistingDate"):
                d_date = pd.to_datetime(meta["delistingDate"])
                if current_date == d_date:
                    last_close = self.close_m[sym].iloc[date_idx]
                    if pd.notna(last_close):
                        self.cash += self.portfolio[sym] * last_close
                        snap = self._symbol_snapshot(sym, date_idx)
                        self.trade_journal.on_exit(
                            symbol=sym,
                            date=current_date,
                            date_idx=date_idx,
                            price=float(last_close),
                            exit_reason="official_delist",
                            close_m=self.close_m,
                            exit_rank=snap["rank"],
                            exit_cmvs=snap["cmvs"],
                            exit_rsi=snap["rsi"],
                            exit_atr=snap["atr"],
                        )
                    del self.portfolio[sym]
                    self.highest_prices.pop(sym, None)
                    print(f"    ⚠️ [DELISTING] {sym} 공식 상장폐지일 강제청산")

    def handle_silent_delisting(self, date_idx):
        current_date = self.close_m.index[date_idx]
        for sym, last_date in self.silent_delist_flags.items():
            if current_date == last_date + pd.Timedelta(days=1) and sym in self.portfolio:
                last_valid = self.close_m[sym].iloc[self.close_m.index.get_loc(last_date)]
                if pd.notna(last_valid):
                    self.cash += self.portfolio[sym] * last_valid * self.SILENT_DELIST_RECOVERY
                    snap = self._symbol_snapshot(sym, date_idx)
                    self.trade_journal.on_exit(
                        symbol=sym,
                        date=current_date,
                        date_idx=date_idx,
                        price=float(last_valid) * self.SILENT_DELIST_RECOVERY,
                        exit_reason="silent_delist",
                        close_m=self.close_m,
                        exit_rank=snap["rank"],
                        exit_cmvs=snap["cmvs"],
                        exit_rsi=snap["rsi"],
                        exit_atr=snap["atr"],
                    )
                del self.portfolio[sym]
                self.highest_prices.pop(sym, None)
                print(f"    ⚠️ [조용한 상장폐지] {sym} haircut({self.SILENT_DELIST_RECOVERY:.0%}) 청산")

    def log_rebalance_summary(self, ctx: RebalanceContext):
        entries = set(ctx.actually_invested.keys()) - self._prior_invested_for_log
        exits = self._prior_invested_for_log - set(ctx.actually_invested.keys())
        prior_n = len(self._prior_invested_for_log)
        turnover = (len(entries) + len(exits)) / max(prior_n, 1)
        self.trade_journal.record_rebalance_turnover(
            ctx.as_of_date, len(entries), len(exits), prior_n
        )
        self._prior_invested_for_log = set(ctx.actually_invested.keys())
        print(
            f"[REBALANCE] date={ctx.as_of_date.date()} exposure={ctx.regime.exposure:.2f} "
            f"breadth={ctx.regime.breadth:.2f} above_ma200={ctx.regime.benchmark_above_ma200} "
            f"n_selected={len(ctx.actually_invested)} turnover={turnover:.2%} "
            f"entries={len(entries)} exits={len(exits)} "
            f"corr_rejections={len(ctx.correlation_rejections)} "
            f"industry_cap_rejections={len(ctx.industry_cap_rejections)}"
        )

    # -------------------------------------------------------------
    def run(self):
        print(self.research_toggles.format_panel())
        trading_days = self.close_m.index
        warmup = max(self.MOM_WINDOW, self.LOWVOL_WINDOW) + 5
        if self.MONTHLY_REBALANCE:
            rebalance_days = set(pd.to_datetime(
                self.close_m.groupby(self.close_m.index.to_period("M")).apply(lambda x: x.index[0]).values
            ))
        else:
            # Research mode: rebalance every trading day (same construct_portfolio path)
            rebalance_days = set(pd.to_datetime(trading_days[warmup:]))

        for idx, current_date in enumerate(trading_days):
            if idx < warmup:
                continue

            self.handle_official_delisting(idx)
            self.handle_silent_delisting(idx)
            self.execute_pending_orders(idx)
            # CMVS v3 daily exits (ATR trail / EMA / RSI exhaustion) -> SELL orders
            self._cmvs_forced_exits = set()
            self.check_cmvs_exits(idx)
            # Prior: self.check_trailing_stops(idx)
            # Observation only — after exits are decided so same-day overlap is visible.
            self._observe_rank_diagnostics(idx)

            current_closes = self.close_m.loc[current_date]
            portfolio_value = sum(
                qty * current_closes.get(s, np.nan) for s, qty in self.portfolio.items() if pd.notna(current_closes.get(s))
            )
            total_equity = self.cash + portfolio_value
            self.equity_curve.append({"Date": current_date, "Total_Equity": total_equity})

            if current_date in rebalance_days:
                ctx = RebalanceContext(as_of_date=current_date)

                active_symbols = [
                    s for s in self.close_m.columns
                    if s != self.BENCHMARK_TICKER and pd.notna(current_closes.get(s))
                    and not self.profile_meta.get(s, {}).get("isEtf", False)
                ]

                ctx.regime = self.determine_market_regime(idx, active_symbols)
                if ctx.regime is None:
                    continue

                # Dynamic leverage (appended): strong uptrend -> 1.2x target exposure.
                # Strong uptrend = benchmark above its 200-day MA with full regime exposure.
                # On trend reversal, determine_market_regime returns 0.0 or 0.5 (never leaves 1.2 sticky).
                if ctx.regime.benchmark_above_ma200 and ctx.regime.exposure >= 1.0:
                    ctx.regime.exposure = 1.2
                    print(
                        f"📈 [LEVERAGE] {current_date.date()} strong uptrend "
                        f"(above MA200) target exposure={ctx.regime.exposure:.2f}x"
                    )

                if ctx.regime.exposure == 0.0:
                    print(f"🚨 [BEAR MARKET] {current_date.date()} 노출도 0% (breadth={ctx.regime.breadth:.2f})")
                    for s in list(self.portfolio.keys()):
                        self.pending_orders.append(
                            {
                                "symbol": s,
                                "qty": self.portfolio[s],
                                "type": "SELL",
                                "reason": "bear_flatten",
                            }
                        )
                    self.previous_target_symbols = set()
                    self.log_rebalance_summary(ctx)
                    continue
                elif ctx.regime.exposure == 0.5:
                    print(f"⚠️ [WARNING] {current_date.date()} 노출도 50% (breadth={ctx.regime.breadth:.2f})")

                top_adv_symbols = self.dvol_m.loc[current_date, active_symbols].nlargest(
                    min(self.TOP_ADV_POOL, len(active_symbols))
                ).index.tolist()

                # Append Quality (ROIC>10%) + Value (FCF/Sales) screens on PIT data
                universe_symbols = self.get_universe(top_adv_symbols, idx)

                ctx.factor_df = self.build_factors(idx, universe_symbols)
                if ctx.factor_df.empty:
                    self.log_rebalance_summary(ctx)
                    continue

                ctx.ranked_df = self.rank_universe(ctx.factor_df)
                # Fundamental quality bonus diagnostics (monthly rebalance)
                if getattr(self, "_last_quality_diag", ""):
                    print(self._last_quality_diag)
                # Research Fact: persist ranks/CMVS for entry/exit journals
                self._last_rank_map = {
                    str(sym): int(i + 1)
                    for i, sym in enumerate(ctx.ranked_df["symbol"].tolist())
                }
                if "final_score" in ctx.ranked_df.columns:
                    self._last_cmvs_map = {
                        str(sym): float(score)
                        for sym, score in zip(
                            ctx.ranked_df["symbol"].tolist(),
                            ctx.ranked_df["final_score"].tolist(),
                        )
                    }
                if "fundamental_bonus_points" in ctx.ranked_df.columns:
                    top_n = ctx.ranked_df.head(int(self.ENTRY_RANK))
                    if not top_n.empty:
                        print(
                            f"    [FUND RANK] top{int(self.ENTRY_RANK)} "
                            f"avg_points={float(top_n['fundamental_bonus_points'].mean()):.2f} "
                            f"avg_score_bonus={float(top_n['quality_bonus'].mean()):.4f} "
                            f"point_scale={float(getattr(self, 'EQS_WEIGHT', 0.0)):.4f}"
                        )
                # Top-N by CMVS final_score (construct_portfolio still applies buffer/corr caps)
                ctx.targets_df = self.construct_portfolio(ctx.ranked_df, idx, ctx)
                # Do not re-buy names that triggered a CMVS exit today OR are on cooldown
                block = set(getattr(self, "_cmvs_forced_exits", None) or set())
                block |= self._cooldown_blocked_set(idx)
                if block and not ctx.targets_df.empty:
                    ctx.targets_df = ctx.targets_df[
                        ~ctx.targets_df["symbol"].isin(block)
                    ].copy()
                # Portfolio sizing infrastructure preserved (equal-weight / optional ATR)
                ctx.targets_df = self.allocate_weights(ctx.targets_df, idx, ctx.regime.exposure)
                final_targets = self.construct_final_targets_with_industry_cap(ctx.targets_df, ctx.regime.exposure, ctx)

                if not final_targets.empty:
                    self.validate_portfolio(final_targets, ctx.regime.exposure)

                self.queue_rebalance_orders(final_targets, idx)
                ctx.actually_invested = dict(zip(final_targets["symbol"], final_targets["final_weight"])) if not final_targets.empty else {}
                self.previous_target_symbols = set(ctx.actually_invested.keys())
                self.log_rebalance_summary(ctx)

        equity = pd.DataFrame(self.equity_curve).set_index("Date")
        self.research_artifacts = self.emit_research_reports(equity)
        cd = self.reentry_cooldown.summary()
        print(
            f"[COOLDOWN SUMMARY] arms={cd['arms']} blocks={cd['blocks']} "
            f"early_clears={cd['early_clears']} still_active={cd['active']}"
        )
        return equity

    def emit_research_reports(self, equity: pd.DataFrame, out_dir: str | Path = "cache") -> dict:
        """Build research Facts: trade journal, KPIs, recommendations, validation, experiment log."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        trades = self.trade_journal.to_frame()
        trades_path = out / "trade_journal.csv"
        if not trades.empty:
            trades.to_csv(trades_path, index=False)
        else:
            trades_path.write_text("", encoding="utf-8")

        bm_rets = None
        if self.BENCHMARK_TICKER in self.close_m.columns and not equity.empty:
            bm = self.close_m[self.BENCHMARK_TICKER].reindex(equity.index)
            bm_rets = bm.pct_change()

        kpi = build_hierarchical_kpi_report(equity, trades, benchmark_returns=bm_rets)
        reco = build_research_recommendation_report(trades, self.research_toggles.as_dict())

        observed_turnover = None
        if self.trade_journal.rebalance_events:
            observed_turnover = float(
                np.mean([e["turnover"] for e in self.trade_journal.rebalance_events])
            )

        fingerprint_path = out / "baseline_fingerprint.json"
        baseline_trade_count = None
        baseline_turnover = None
        if fingerprint_path.exists():
            try:
                fp = json.loads(fingerprint_path.read_text(encoding="utf-8"))
                baseline_trade_count = fp.get("n_closed_trades")
                baseline_turnover = fp.get("avg_rebalance_turnover")
            except Exception:
                pass

        validation = run_research_validation_checklist(
            self.research_toggles,
            trades,
            max_portfolio_size=self._baseline_max_n,
            selection_buffer_size=self._baseline_buf_n,
            baseline_trade_count=baseline_trade_count,
            baseline_turnover=baseline_turnover,
            observed_turnover=observed_turnover,
        )

        # Record fingerprint when running at baseline defaults (for future Task-6 compares)
        if self.research_toggles.is_baseline_defaults(self._baseline_max_n, self._baseline_buf_n):
            fingerprint_path.write_text(
                json.dumps(
                    {
                        "n_closed_trades": int(len(trades)) if trades is not None else 0,
                        "avg_rebalance_turnover": observed_turnover,
                        "toggles": self.research_toggles.as_dict(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        hist = ExperimentHistory(out / "experiment_history.json")
        next_exp = reco.get("next_experiment") or {}
        exp_id = hist.append(
            parent_exp=None,
            changed_variables=next_exp.get("changed_variables") or {},
            hypothesis=str(next_exp.get("hypothesis") or ""),
            expected_outcome=str(next_exp.get("expected_outcome") or ""),
            actual_outcome=None,
            decision="proposed",
            metrics={
                "kpi": kpi,
                "sample_size": reco.get("sample_size"),
                "statistical_confidence": next_exp.get("statistical_confidence"),
                "consistency": next_exp.get("consistency"),
                "estimated_impact": next_exp.get("estimated_impact"),
                "risk_of_overfitting": next_exp.get("risk_of_overfitting"),
            },
            notes="Auto-logged proposal from single backtest — do not adopt without confirmation.",
        )

        rank_diag_summary = self.rank_diagnostics.summary()
        rank_diag_path = self.rank_diagnostics.write_report(
            out / "rank_diagnostics_report.txt"
        )
        (out / "rank_diagnostics.json").write_text(
            json.dumps(rank_diag_summary, indent=2, default=str),
            encoding="utf-8",
        )
        rank_diag_txt = self.rank_diagnostics.format_report()

        report_txt = "\n\n".join(
            [
                format_kpi_report(kpi),
                format_recommendation_report(reco),
                format_validation_checklist(validation),
                f"Experiment History entry: {exp_id} -> {hist.path}",
                f"Trade journal: {trades_path} (n={len(trades)})",
                rank_diag_txt,
            ]
        )
        report_path = out / "research_report.txt"
        report_path.write_text(report_txt, encoding="utf-8")
        print("\n" + report_txt)

        artifacts = {
            "kpi": kpi,
            "recommendations": reco,
            "validation": validation,
            "experiment_id": exp_id,
            "trades_path": str(trades_path),
            "report_path": str(report_path),
            "rank_diagnostics": rank_diag_summary,
            "rank_diagnostics_path": str(rank_diag_path),
        }
        (out / "research_artifacts.json").write_text(
            json.dumps(artifacts, indent=2, default=str), encoding="utf-8"
        )
        return artifacts
