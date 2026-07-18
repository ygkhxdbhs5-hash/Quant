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

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


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
        self.MAX_PORTFOLIO_SIZE = int(cfg.get("max_portfolio_size", 30))
        self.SELECTION_BUFFER_SIZE = int(cfg.get("selection_buffer_size", 40))
        self.MAX_INDUSTRY_WEIGHT = float(cfg.get("max_industry_weight", 0.20))
        self.MIN_INDUSTRY_SIZE = int(cfg.get("min_industry_size", 8))
        self.VOL_FLOOR = float(cfg.get("vol_floor", 1e-4))
        self.WEIGHT_SUM_TOLERANCE = float(cfg.get("weight_sum_tolerance", 0.02))
        self.LOWVOL_WINDOW = int(cfg.get("lowvol_window", 60))
        self.MOM_WINDOW = int(cfg.get("mom_window", 252))
        self.CORR_WINDOW = int(cfg.get("corr_window", 60))
        self.CORR_THRESHOLD = float(cfg.get("corr_threshold", 0.80))
        self.FLAT_TAX_RATE = float(cfg.get("flat_tax_rate", 0.21))
        self.SILENT_DELIST_RECOVERY = float(cfg.get("silent_delist_recovery", 0.30))
        self.SILENT_DELIST_GAP_DAYS = int(cfg.get("silent_delist_gap_days", 10))
        self.TOP_ADV_POOL = int(cfg.get("top_adv_pool", 250))
        self.PARTICIPATION_CAP_BUY = float(cfg.get("participation_cap_buy", 0.08))
        self.PARTICIPATION_CAP_SELL = float(cfg.get("participation_cap_sell", 0.15))
        # Trading costs applied on every buy/sell at rebalance fill
        self.COMMISSION_RATE = float(cfg.get("commission_rate", 0.0005))  # 0.05%
        self.SLIPPAGE_RATE = float(cfg.get("slippage_rate", 0.0002))  # 0.02%
        # Universe Quality / Value filter thresholds (appended selection screens)
        self.MIN_ROIC = float(cfg.get("min_roic", 0.10))  # Quality: ROIC > 10%
        self.MIN_FCF_SALES_YIELD = float(cfg.get("min_fcf_sales_yield", 0.0))  # Value: FCF/Sales

        print(">> 로컬 데이터 로드...")
        universe_path = Path(paths.get("metadata", "data/metadata")) / "universe.pkl"
        panels_path = Path(paths.get("prices", "data/prices")) / "panels.pkl"
        funds_path = Path(paths.get("fundamentals", "data/fundamentals")) / "pit_history.pkl"
        for required in (universe_path, panels_path, funds_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"Missing {required}. Run: python -m downloader.update_data --config {config_path}"
                )

        with open(universe_path, "rb") as f:
            universe = pickle.load(f)
        with open(panels_path, "rb") as f:
            panels = pickle.load(f)
        with open(funds_path, "rb") as f:
            self.fundamental_history = pickle.load(f)

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

    # -------------------------------------------------------------
    def _precompute_matrices(self):
        print(">> 벡터화 매트릭스(ATR/모멘텀/변동성/스프레드) 사전 계산...")
        close_m, high_m, low_m = self.close_m, self.high_m, self.low_m

        self.adv20_m = self.dvol_m.rolling(20, min_periods=5).mean()
        ret_m = close_m.pct_change()
        self.vol20_m = ret_m.rolling(20, min_periods=10).std()
        self.vol60_m = ret_m.rolling(self.LOWVOL_WINDOW, min_periods=20).std()
        self.sma50_m = close_m.rolling(50, min_periods=50).mean()
        self.sma200_m = close_m.rolling(200, min_periods=200).mean()

        # [버그 수정 계승] 12-1 모멘텀: t-252~t-21 구간의 누적수익률
        self.mom_12_1_m = close_m.shift(21) / close_m.shift(self.MOM_WINDOW) - 1

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

    def get_universe(self, candidate_symbols, date_idx):
        """Append Quality (ROIC > 10%) and Value (FCF/Sales) filters using PIT data.

        Value metric: FCF/Sales = (op_cf - capex) / revenue.
        If FCF inputs are limited, approximate with Sales / Market Cap where
        Market Cap ≈ price * diluted_shares_outstanding (PIT shares when present).
        Existing candidate construction / ranking logic is left unchanged.
        """
        current_date = self.close_m.index[date_idx]
        current_closes = self.close_m.loc[current_date]
        filtered = []
        n_fail_quality = 0
        n_fail_value = 0
        n_value_approx = 0

        for sym in candidate_symbols:
            fundamentals = self.get_latest_available_fundamentals(sym, current_date)
            if fundamentals is None:
                n_fail_quality += 1
                continue

            def _fget(key, default=np.nan):
                try:
                    val = fundamentals[key]
                    return default if pd.isna(val) else float(val)
                except Exception:
                    return default

            # --- Quality filter: ROIC > 10% ---
            roic = _fget("roic")
            if pd.isna(roic) or roic <= self.MIN_ROIC:
                n_fail_quality += 1
                continue

            # --- Value filter: FCF/Sales Yield (fallback: Sales / Market Cap) ---
            revenue = _fget("revenue")
            op_cf = _fget("op_cf")
            capex = _fget("capex")
            value_yield = np.nan
            if pd.notna(revenue) and revenue != 0 and pd.notna(op_cf) and pd.notna(capex):
                value_yield = (op_cf - capex) / revenue
            else:
                # Limited FCF data: approximate value with Sales / Market Cap
                shares = _fget("diluted_shares_outstanding")
                if pd.isna(shares) or shares <= 0:
                    shares = _fget("basic_shares_outstanding")
                price = current_closes.get(sym, np.nan)
                if (
                    pd.notna(revenue)
                    and revenue > 0
                    and pd.notna(shares)
                    and shares > 0
                    and pd.notna(price)
                    and price > 0
                ):
                    market_cap = float(price) * float(shares)
                    if market_cap > 0:
                        value_yield = float(revenue) / market_cap
                        n_value_approx += 1

            if pd.isna(value_yield) or value_yield <= self.MIN_FCF_SALES_YIELD:
                n_fail_value += 1
                continue

            filtered.append(sym)

        print(
            f"    [UNIVERSE] {current_date.date()} in={len(candidate_symbols)} "
            f"out={len(filtered)} fail_quality={n_fail_quality} fail_value={n_fail_value} "
            f"value_approx_mcap_sales={n_value_approx}"
        )
        return filtered

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

    def build_factors(self, date_idx, active_symbols) -> pd.DataFrame:
        current_date = self.close_m.index[date_idx]
        rows = []
        n_price_only = 0
        n_full = 0
        for sym in active_symbols:
            price = self.close_m[sym].iloc[date_idx]
            if pd.isna(price) or price <= 0:
                continue

            mom = self.mom_12_1_m[sym].iloc[date_idx]
            vol60 = self.vol60_m[sym].iloc[date_idx]
            if pd.isna(mom) or pd.isna(vol60) or vol60 <= 0:
                continue

            fundamentals = self.get_latest_available_fundamentals(sym, current_date)
            if fundamentals is None:
                # Fallback: keep ticker with price/volume factors only.
                rows.append({
                    "symbol": sym,
                    "mom_raw": mom,
                    "vol_raw": vol60,
                    "op_margin_raw": np.nan,
                    "roic_raw": np.nan,
                    "gross_prof_raw": np.nan,
                    "op_cf": np.nan,
                    "capex": np.nan,
                    "revenue": np.nan,
                    "industry": self.profile_meta.get(sym, {}).get("industry", "Unknown"),
                    "has_fundamentals": False,
                })
                n_price_only += 1
                continue

            def _fget(key, default=np.nan):
                try:
                    val = fundamentals[key]
                    return default if pd.isna(val) else val
                except Exception:
                    return default

            rows.append({
                "symbol": sym,
                "mom_raw": mom,
                "vol_raw": vol60,
                "op_margin_raw": _fget("op_margin"),
                "roic_raw": _fget("roic"),
                "gross_prof_raw": _fget("gross_profitability"),
                "op_cf": _fget("op_cf"),
                "capex": _fget("capex"),
                "revenue": _fget("revenue"),
                "industry": self.profile_meta.get(sym, {}).get("industry", "Unknown"),
                "has_fundamentals": True,
            })
            n_full += 1

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # Require only price-based factors; quality may be missing (fallback mode).
        df = df.dropna(subset=["mom_raw", "vol_raw"])
        if n_price_only > 0:
            print(
                f"    [FACTORS] {current_date.date()} full={n_full} "
                f"price_volume_fallback={n_price_only}"
            )
        return df

    def rank_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        industry_counts = df.groupby("industry")["symbol"].transform("count")
        df["_group_key"] = np.where(industry_counts >= self.MIN_INDUSTRY_SIZE, df["industry"], "__GLOBAL_FALLBACK__")

        def _rank(col):
            return df.groupby("_group_key")[col].rank(pct=True)

        df["rank_mom"] = _rank("mom_raw")
        df["neg_vol"] = -df["vol_raw"]
        df["rank_lowvol"] = _rank("neg_vol")

        has_quality = (
            df["roic_raw"].notna() & df["gross_prof_raw"].notna() & df["op_margin_raw"].notna()
        )
        # Quality ranks: NaN inputs stay NaN (pandas rank skips them within group).
        df["rank_roic"] = _rank("roic_raw")
        df["rank_gp"] = _rank("gross_prof_raw")
        df["rank_op"] = _rank("op_margin_raw")
        df["rank_quality"] = (df["rank_roic"] * 0.50) + (df["rank_gp"] * 0.30) + (df["rank_op"] * 0.20)

        # Full model when fundamentals exist; otherwise momentum + low-vol only.
        full_score = (df["rank_mom"] * 0.45) + (df["rank_quality"] * 0.45) + (df["rank_lowvol"] * 0.10)
        fallback_score = (df["rank_mom"] * 0.80) + (df["rank_lowvol"] * 0.20)
        df["final_score"] = np.where(has_quality, full_score, fallback_score)
        df["factor_mode"] = np.where(has_quality, "full", "price_volume_fallback")
        return df.sort_values("final_score", ascending=False).reset_index(drop=True)

    def construct_portfolio(self, ranked_df, date_idx, ctx: RebalanceContext) -> pd.DataFrame:
        if ranked_df.empty:
            return ranked_df
        top_core = set(ranked_df.head(self.MAX_PORTFOLIO_SIZE)["symbol"].tolist())
        top_buffer = set(ranked_df.head(self.SELECTION_BUFFER_SIZE)["symbol"].tolist())

        keep = self.previous_target_symbols & top_buffer
        new_candidates = ranked_df[ranked_df["symbol"].isin(top_core) & ~ranked_df["symbol"].isin(keep)]

        selected = list(keep)
        for _, row in new_candidates.iterrows():
            if len(selected) >= self.MAX_PORTFOLIO_SIZE:
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

    def apply_risk_adjustments(self, targets: pd.DataFrame, exposure: float) -> pd.DataFrame:
        if targets.empty:
            return targets
        targets = targets.copy()
        safe_vol = targets["vol_raw"].clip(lower=self.VOL_FLOOR)
        targets["inv_vol"] = 1.0 / safe_vol
        targets["raw_weight"] = targets["inv_vol"] / targets["inv_vol"].sum()
        max_single = (1.0 / self.MAX_PORTFOLIO_SIZE) * 2.0
        targets["raw_weight"] = targets["raw_weight"].clip(upper=max_single)
        targets["final_weight"] = (targets["raw_weight"] / targets["raw_weight"].sum()) * exposure
        return targets

    def validate_portfolio(self, final_targets: pd.DataFrame, exposure: float):
        issues = []
        if final_targets["symbol"].duplicated().any():
            issues.append("중복 심볼")
        if len(final_targets) > self.MAX_PORTFOLIO_SIZE:
            issues.append(f"포트폴리오 크기 초과 {len(final_targets)}")
        total_w = final_targets["final_weight"].sum()
        if abs(total_w - exposure) > self.WEIGHT_SUM_TOLERANCE:
            issues.append(f"총비중 불일치 {total_w:.3f} vs {exposure:.3f}")
        max_single = (1.0 / self.MAX_PORTFOLIO_SIZE) * 2.0
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
            self.pending_orders.append({"symbol": sym, "qty": qty, "type": order_type})

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
                cap = self._max_shares_participation(sym, date_idx, o_price, self.PARTICIPATION_CAP_BUY)
                exec_qty = min(qty, cap)
                if exec_qty <= 0:
                    continue
                cost_ratio = self._cost_ratio(sym, date_idx, exec_qty, o_price)
                total_cost = exec_qty * o_price * (1 + cost_ratio)
                if self.cash >= total_cost:
                    self.portfolio[sym] = self.portfolio.get(sym, 0) + exec_qty
                    self.cash -= total_cost
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
        self.pending_orders = []

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
                    del self.portfolio[sym]
                    print(f"    ⚠️ [DELISTING] {sym} 공식 상장폐지일 강제청산")

    def handle_silent_delisting(self, date_idx):
        current_date = self.close_m.index[date_idx]
        for sym, last_date in self.silent_delist_flags.items():
            if current_date == last_date + pd.Timedelta(days=1) and sym in self.portfolio:
                last_valid = self.close_m[sym].iloc[self.close_m.index.get_loc(last_date)]
                if pd.notna(last_valid):
                    self.cash += self.portfolio[sym] * last_valid * self.SILENT_DELIST_RECOVERY
                del self.portfolio[sym]
                print(f"    ⚠️ [조용한 상장폐지] {sym} haircut({self.SILENT_DELIST_RECOVERY:.0%}) 청산")

    def log_rebalance_summary(self, ctx: RebalanceContext):
        entries = set(ctx.actually_invested.keys()) - self._prior_invested_for_log
        exits = self._prior_invested_for_log - set(ctx.actually_invested.keys())
        turnover = (len(entries) + len(exits)) / max(len(self._prior_invested_for_log), 1)
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
        trading_days = self.close_m.index
        warmup = max(self.MOM_WINDOW, self.LOWVOL_WINDOW) + 5
        rebalance_days = set(pd.to_datetime(
            self.close_m.groupby(self.close_m.index.to_period("M")).apply(lambda x: x.index[0]).values
        ))

        for idx, current_date in enumerate(trading_days):
            if idx < warmup:
                continue

            self.handle_official_delisting(idx)
            self.handle_silent_delisting(idx)
            self.execute_pending_orders(idx)

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

                if ctx.regime.exposure == 0.0:
                    print(f"🚨 [BEAR MARKET] {current_date.date()} 노출도 0% (breadth={ctx.regime.breadth:.2f})")
                    for s in list(self.portfolio.keys()):
                        self.pending_orders.append({"symbol": s, "qty": self.portfolio[s], "type": "SELL"})
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
                ctx.targets_df = self.construct_portfolio(ctx.ranked_df, idx, ctx)
                ctx.targets_df = self.apply_risk_adjustments(ctx.targets_df, ctx.regime.exposure)
                final_targets = self.construct_final_targets_with_industry_cap(ctx.targets_df, ctx.regime.exposure, ctx)

                if not final_targets.empty:
                    self.validate_portfolio(final_targets, ctx.regime.exposure)

                self.queue_rebalance_orders(final_targets, idx)
                ctx.actually_invested = dict(zip(final_targets["symbol"], final_targets["final_weight"])) if not final_targets.empty else {}
                self.previous_target_symbols = set(ctx.actually_invested.keys())
                self.log_rebalance_summary(ctx)

        return pd.DataFrame(self.equity_curve).set_index("Date")
