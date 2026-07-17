"""
Nasdaq Institutional Production Engine (v3).

Investment logic is copied from the proven lean_engine_v3 implementation.
Do not modify factor calculations, ranking, portfolio construction, or risk management.

Only the host runtime differs: this module reads local data via the project
data/ directory instead of QuantConnect cloud feeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from engine.runtime import (
    DelistingType,
    InteractiveBrokersFeeModel,
    LocalAlgorithm,
    Resolution,
    RollingWindow,
    Universe,
    UniverseSettings,
    VolumeShareSlippageModel,
)


@dataclass
class RegimeState:
    exposure: float
    breadth: float
    benchmark_above_ma200: bool


@dataclass
class RebalanceContext:
    """한 번의 리밸런싱에서 파이프라인 단계 간 상태를 넘기는 컨테이너."""
    as_of_date: object = None
    regime: Optional[RegimeState] = None
    factor_df: Optional[pd.DataFrame] = None
    ranked_df: Optional[pd.DataFrame] = None
    target_symbols: set = field(default_factory=set)
    targets_df: Optional[pd.DataFrame] = None
    correlation_rejections: list = field(default_factory=list)
    industry_cap_rejections: list = field(default_factory=list)
    actually_invested: set = field(default_factory=set)


class NasdaqInstitutionalProductionEngine(LocalAlgorithm):
    """
    v3 수정 사항 (리뷰 우선순위 1~6):
      [1] Look-ahead bias 제거: Fine 데이터를 그 자리에서 바로 읽지 않고
          GetLatestAvailableFundamentals(symbol, asOfDate)를 통해 "그 시점에 실제로
          쓸 수 있었던 최신 리포트"만 조회하는 PIT 추상화 계층 도입.
      [2] RebalanceStrategy()를 단일 책임 함수들의 파이프라인으로 분리:
          DetermineMarketRegime -> BuildFactors -> RankUniverse ->
          ConstructPortfolio -> ApplyRiskAdjustments -> ExecuteTrades
      [3] 검증 assertion 추가: 총 비중, 개별 상한, 중복 심볼, 포트폴리오 크기,
          산업 노출 상한을 주문 제출 전에 점검하고 위반 시 로그.
      [4] 소규모 산업군(<MIN_INDUSTRY_SIZE) 팩터는 산업 내 랭킹 대신 전체 유니버스 랭킹으로 폴백.
      [5] 역변동성 사이징에 변동성 하한(epsilon)을 적용해 극단적으로 낮은 변동성에서
          가중치가 폭주하는 것을 방지.
      [6] 구조화 로깅: 리밸런싱마다 날짜/레짐/노출도/breadth/turnover/상관관계 및
          산업cap 거부 종목/최종 비중을 한 줄로 요약해 로그.
      [향후 개선, 이번엔 미구현]
        - 상관관계 행렬을 매번 전체 재계산하는 대신 rolling covariance를 incremental 업데이트
          (유니버스가 500~1000종목으로 커질 때 고려).
        - industry-cap greedy 선택을 constrained optimization/iterative replacement로 교체
          (현재는 스코어 순으로 처리해 path dependence가 있음, 방향성은 합리적이지만 최적은 아님).
    """

    MIN_INDUSTRY_SIZE = 8       # 이보다 작은 산업군은 전체 유니버스 기준으로 랭킹
    VOL_FLOOR = 1e-4            # 역변동성 사이징 시 변동성 하한 (0에 가까운 값으로 인한 가중치 폭주 방지
    WEIGHT_SUM_TOLERANCE = 0.02  # 총 비중 검증 허용 오차

    def Initialize(self, config: Optional[dict] = None):
        cfg = config or {}

        start = cfg.get("start_date", [2007, 1, 1])
        end = cfg.get("end_date", [2026, 6, 30])
        self.SetStartDate(start[0], start[1], start[2])
        self.SetEndDate(end[0], end[1], end[2])
        self.SetCash(cfg.get("initial_cash", 50_000_000))

        benchmark = cfg.get("benchmark", "QQQ")
        self.benchmark = self.AddEquity(benchmark, Resolution.Daily).symbol
        self.SetBenchmark(self.benchmark)

        self.qqq_ma200 = self.SMA(self.benchmark, 200, Resolution.Daily)
        self.qqq_ma_window = RollingWindow(2)
        self.qqq_ma200.Updated = lambda sender, updated: self.qqq_ma_window.Add(updated)

        self.UniverseSettings = UniverseSettings()
        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.SlippageModel = VolumeShareSlippageModel(
            cfg.get("slippage", {}).get("volume_limit", 0.10)
        )
        self.UniverseSettings.FeeModel = InteractiveBrokersFeeModel()

        # Local runner invokes Coarse/Fine on month-start; schedule retained for parity.
        self._rebalance_callback = self.RebalanceStrategy

        self.active_universe = set()
        self.indicators = {}

        # [1] PIT 저장소: symbol -> [(as_of_date, fundamentals_dict), ...] (날짜 오름차순)
        # FineSelectionFunction이 호출될 때마다 "그 시점의 스냅샷"을 append만 하고 덮어쓰지 않는다.
        self.fundamental_history = {}

        self.last_selection_month = None
        self.cached_fine_symbols = []

        self.max_portfolio_size = cfg.get("max_portfolio_size", 30)
        self.selection_buffer_size = cfg.get("selection_buffer_size", 40)
        self.max_industry_weight = cfg.get("max_industry_weight", 0.20)
        self.base_stock_weight = 1.0 / self.max_portfolio_size
        self.current_exposure = 1.0

        self.LOWVOL_WINDOW = cfg.get("lowvol_window", 60)
        self.MOM_WINDOW = cfg.get("mom_window", 252)
        self.CORR_WINDOW = cfg.get("corr_window", 60)
        self.CORR_THRESHOLD = cfg.get("corr_threshold", 0.80)

        # Optional overrides for class-level constants (defaults preserve original behavior)
        if "min_industry_size" in cfg:
            self.MIN_INDUSTRY_SIZE = cfg["min_industry_size"]
        if "vol_floor" in cfg:
            self.VOL_FLOOR = cfg["vol_floor"]
        if "weight_sum_tolerance" in cfg:
            self.WEIGHT_SUM_TOLERANCE = cfg["weight_sum_tolerance"]

        self.previous_target_symbols = set()

    # =====================================================================
    # 유니버스 선정 (월 1회만 실제 재계산)
    # =====================================================================
    def CoarseSelectionFunction(self, coarse):
        current_month = self.Time.month, self.Time.year
        if self.last_selection_month == current_month:
            return Universe.Unchanged
        return [x.Symbol for x in coarse if x.HasFundamentalData and x.Price > 2.0 and x.Volume > 50000]

    def FineSelectionFunction(self, fine):
        current_month = self.Time.month, self.Time.year
        if self.last_selection_month == current_month:
            return Universe.Unchanged
        self.last_selection_month = current_month

        sorted_by_adv = sorted(fine, key=lambda x: x.DollarVolume, reverse=True)
        top_250 = sorted_by_adv[:250]

        snapshot_symbols = []
        for x in top_250:
            has_data = (
                x.OperationRatios.OperatingMargin and
                x.OperationRatios.ROIC and
                x.FinancialStatements.IncomeStatement.GrossProfit and
                x.FinancialStatements.IncomeStatement.TotalRevenue and
                x.FinancialStatements.BalanceSheet.TotalAssets and
                x.FinancialStatements.CashFlowStatement.CashFlowFromOperatingActivities and
                x.FinancialStatements.CashFlowStatement.CapEx
            )
            if not has_data:
                continue

            market_cap = x.Summary.MarketCap.Value if x.Summary.MarketCap else 0
            if market_cap <= 0:
                continue

            total_assets = x.FinancialStatements.BalanceSheet.TotalAssets.Value
            if not total_assets or total_assets <= 0:
                continue

            ocf = x.FinancialStatements.CashFlowStatement.CashFlowFromOperatingActivities.Value
            capex = abs(x.FinancialStatements.CashFlowStatement.CapEx.Value)
            sales = x.FinancialStatements.IncomeStatement.TotalRevenue.Value

            total_debt = getattr(x.FinancialStatements.BalanceSheet, "TotalDebt", None)
            cash_eq = getattr(x.FinancialStatements.BalanceSheet, "CashAndCashEquivalents", None)
            ev = market_cap
            if total_debt and cash_eq and total_debt.Value is not None and cash_eq.Value is not None:
                candidate_ev = market_cap + total_debt.Value - cash_eq.Value
                if candidate_ev > 0:
                    ev = candidate_ev

            snapshot = {
                'op_margin': x.OperationRatios.OperatingMargin.Value,
                'roic': x.OperationRatios.ROIC.Value,
                'gross_profitability': x.FinancialStatements.IncomeStatement.GrossProfit.Value / total_assets,
                'fcf_yield': (ocf - capex) / ev,
                'sales_yield': sales / ev,
                'industry': x.CompanyReference.IndustryGroup if x.CompanyReference.IndustryGroup else "Unknown",
                'dollar_volume': x.DollarVolume,
            }

            # [1] PIT 저장소에 "덮어쓰기"가 아니라 "추가"만 함. 이후 as_of_date 조회 시
            # 그 날짜 이전에 기록된 가장 최근 스냅샷만 사용하도록 강제한다.
            self.fundamental_history.setdefault(x.Symbol, []).append((self.Time, snapshot))
            snapshot_symbols.append(x.Symbol)

        self.cached_fine_symbols = snapshot_symbols
        return self.cached_fine_symbols

    def IngestLocalFundamentalSnapshot(self, symbol, as_of_date, snapshot: dict):
        """Load a precomputed PIT snapshot from local fundamentals files into history.

        This is data plumbing only — snapshot fields must already match the engine schema.
        """
        self.fundamental_history.setdefault(symbol, []).append((as_of_date, snapshot))

    def GetLatestAvailableFundamentals(self, symbol, as_of_date):
        """
        [1] Look-ahead bias 제거의 핵심.
        symbol의 히스토리에서 as_of_date '이전 또는 그 시점'에 기록된 가장 최근 스냅샷만 반환.
        미래 시점 데이터는 절대 반환하지 않는다. 데이터가 없으면 None.
        """
        history = self.fundamental_history.get(symbol)
        if not history:
            return None
        # history는 append 순서(=시간 순)로 쌓이므로 뒤에서부터 탐색하면 O(k)로 충분히 빠름
        for record_date, snapshot in reversed(history):
            if record_date <= as_of_date:
                return snapshot
        return None

    # =====================================================================
    def OnSecuritiesChanged(self, changes):
        added_symbols = [sec.Symbol for sec in changes.AddedSecurities if sec.Symbol != self.benchmark]

        for symbol in added_symbols:
            self.active_universe.add(symbol)
            if symbol not in self.indicators:
                self.indicators[symbol] = {
                    'price_window': RollingWindow(self.MOM_WINDOW + 5),
                    'sma_50': self.SMA(symbol, 50, Resolution.Daily),
                }

        if added_symbols:
            hist_df = self.History(added_symbols, self.MOM_WINDOW + 5, Resolution.Daily)
            if not hist_df.empty:
                valid_symbols = hist_df.index.get_level_values('symbol').unique()
                for symbol in added_symbols:
                    if symbol in valid_symbols:
                        sym_data = hist_df.xs(symbol, level='symbol').sort_index()
                        for time, row in sym_data.iterrows():
                            self.indicators[symbol]['price_window'].Add(row['close'])

        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)
            if symbol in self.active_universe:
                self.active_universe.remove(symbol)
            if symbol in self.indicators:
                del self.indicators[symbol]

    def OnData(self, data):
        if data.Delistings:
            for symbol, delisting in data.Delistings.items():
                if delisting.Type == DelistingType.Warning:
                    if self.Portfolio[symbol].Invested:
                        self.Liquidate(symbol, "Delisting warning - preemptive liquidation")
                        self.Log(f"⚠️ [DELISTING] {symbol} 선제 청산")

        for symbol in self.active_universe:
            if data.Bars.ContainsKey(symbol) and symbol in self.indicators:
                self.indicators[symbol]['price_window'].Add(data.Bars[symbol].Close)

    def CalculateMarketBreadth(self):
        total_active = 0
        above_50ma = 0
        for sym in list(self.active_universe):
            if sym in self.indicators and self.indicators[sym]['sma_50'].IsReady:
                total_active += 1
                if self.Securities[sym].Price > self.indicators[sym]['sma_50'].Current.Value:
                    above_50ma += 1
        return (above_50ma / total_active) if total_active > 0 else 1.0

    def _build_correlation_matrix(self, symbols):
        returns_dict = {}
        for sym in symbols:
            p_win = self.indicators[sym]['price_window']
            if p_win.Count < self.CORR_WINDOW + 1:
                continue
            prices = np.array([p_win[i] for i in range(self.CORR_WINDOW + 1)])[::-1]
            rets = prices[1:] / prices[:-1] - 1.0
            returns_dict[sym] = rets
        if len(returns_dict) < 2:
            return pd.DataFrame()
        ret_df = pd.DataFrame(returns_dict)
        return ret_df.corr(method="spearman")

    # =====================================================================
    # [2] 파이프라인 단계별 함수들
    # =====================================================================
    def DetermineMarketRegime(self) -> Optional[RegimeState]:
        if not (self.qqq_ma200.IsReady and self.qqq_ma_window.IsReady):
            return None

        benchmark_price = self.Securities[self.benchmark].Price
        ma_200_price = self.qqq_ma_window[0].Value
        breadth_ratio = self.CalculateMarketBreadth()
        above_ma = benchmark_price >= ma_200_price

        if benchmark_price < ma_200_price and breadth_ratio < 0.20:
            exposure = 0.0
        elif benchmark_price < ma_200_price or breadth_ratio < 0.40:
            exposure = 0.5
        else:
            exposure = 1.0

        return RegimeState(exposure=exposure, breadth=breadth_ratio, benchmark_above_ma200=above_ma)

    def BuildFactors(self, as_of_date) -> pd.DataFrame:
        """[1] 모든 팩터를 GetLatestAvailableFundamentals를 통해서만 조회 (Fine 객체 직접 참조 금지)."""
        raw_factors = []
        for symbol in list(self.active_universe):
            if symbol not in self.indicators:
                continue

            fundamentals = self.GetLatestAvailableFundamentals(symbol, as_of_date)
            if fundamentals is None:
                continue

            p_win = self.indicators[symbol]['price_window']
            if p_win.Count < self.MOM_WINDOW + 1:
                continue

            price_t_21 = p_win[21]
            price_t_252 = p_win[251]
            if price_t_252 <= 0:
                continue
            mom_12_1 = (price_t_21 / price_t_252) - 1.0

            if p_win.Count < self.LOWVOL_WINDOW + 1:
                continue
            prices = np.array([p_win[i] for i in range(self.LOWVOL_WINDOW + 1)])
            returns = (prices[:-1] / prices[1:]) - 1.0
            volatility = np.std(returns)
            if volatility <= 0:
                continue

            raw_factors.append({
                'symbol': symbol,
                'mom_raw': mom_12_1,
                'vol_raw': volatility,
                'op_margin_raw': fundamentals['op_margin'],
                'roic_raw': fundamentals['roic'],
                'gross_prof_raw': fundamentals['gross_profitability'],
                'fcf_yield_raw': fundamentals['fcf_yield'],
                'sales_yield_raw': fundamentals['sales_yield'],
                'industry': fundamentals['industry'],
            })

        return pd.DataFrame(raw_factors)

    def RankUniverse(self, df: pd.DataFrame) -> pd.DataFrame:
        """[4] 소규모 산업군은 산업 내 랭킹 대신 전체 유니버스 랭킹으로 폴백."""
        if df.empty:
            return df

        industry_counts = df.groupby('industry')['symbol'].transform('count')
        df = df.copy()
        df['_industry_group_key'] = np.where(
            industry_counts >= self.MIN_INDUSTRY_SIZE, df['industry'], '__GLOBAL_FALLBACK__'
        )
        small_industry_count = (industry_counts < self.MIN_INDUSTRY_SIZE).sum()
        if small_industry_count > 0:
            self.Log(f"    [소규모 산업군 폴백] {small_industry_count}개 종목이 전체 유니버스 랭킹으로 처리됨")

        def _rank(col):
            return df.groupby('_industry_group_key')[col].rank(pct=True)

        df['rank_mom'] = _rank('mom_raw')
        df['rank_fcf'] = _rank('fcf_yield_raw')
        df['rank_sales'] = _rank('sales_yield_raw')
        df['rank_value'] = (df['rank_fcf'] * 0.5) + (df['rank_sales'] * 0.5)

        df['neg_vol'] = -df['vol_raw']
        df['rank_lowvol'] = _rank('neg_vol')

        df['rank_roic'] = _rank('roic_raw')
        df['rank_gp'] = _rank('gross_prof_raw')
        df['rank_op'] = _rank('op_margin_raw')
        df['rank_quality'] = (df['rank_roic'] * 0.50) + (df['rank_gp'] * 0.30) + (df['rank_op'] * 0.20)

        df['final_score'] = (df['rank_mom'] * 0.40) + (df['rank_quality'] * 0.40) + (df['rank_value'] * 0.10) + (df['rank_lowvol'] * 0.10)
        return df.sort_values(by='final_score', ascending=False).reset_index(drop=True)

    def ConstructPortfolio(self, ranked_df: pd.DataFrame, ctx: RebalanceContext) -> pd.DataFrame:
        """Turnover 히스테리시스 + 상관관계 필터링 (구조는 v2와 동일, 거부 사유를 ctx에 기록)."""
        if ranked_df.empty:
            return ranked_df

        top_core = set(ranked_df.head(self.max_portfolio_size)['symbol'].tolist())
        top_buffer = set(ranked_df.head(self.selection_buffer_size)['symbol'].tolist())

        keep_symbols = self.previous_target_symbols & top_buffer
        new_entry_candidates = ranked_df[ranked_df['symbol'].isin(top_core) & ~ranked_df['symbol'].isin(keep_symbols)]

        selected = list(keep_symbols)
        for _, row in new_entry_candidates.iterrows():
            if len(selected) >= self.max_portfolio_size:
                break
            selected.append(row['symbol'])

        targets = ranked_df[ranked_df['symbol'].isin(selected)].copy()

        corr_matrix = self._build_correlation_matrix(targets['symbol'].tolist())
        final_selected = []
        for _, row in targets.sort_values('final_score', ascending=False).iterrows():
            sym = row['symbol']
            if sym in keep_symbols:
                final_selected.append(sym)
                continue
            is_correlated = False
            if not corr_matrix.empty and sym in corr_matrix.columns:
                for held in final_selected:
                    if held in corr_matrix.columns and pd.notna(corr_matrix.loc[sym, held]):
                        if corr_matrix.loc[sym, held] >= self.CORR_THRESHOLD:
                            is_correlated = True
                            break
            if is_correlated:
                ctx.correlation_rejections.append(sym)
            else:
                final_selected.append(sym)

        return ranked_df[ranked_df['symbol'].isin(final_selected)].copy()

    def ApplyRiskAdjustments(self, targets: pd.DataFrame, exposure: float) -> pd.DataFrame:
        """[5] 역변동성 사이징 + 변동성 하한(epsilon) 적용."""
        if targets.empty:
            return targets
        targets = targets.copy()
        safe_vol = targets['vol_raw'].clip(lower=self.VOL_FLOOR)
        targets['inv_vol'] = 1.0 / safe_vol
        targets['raw_weight'] = targets['inv_vol'] / targets['inv_vol'].sum()
        max_single_weight = self.base_stock_weight * 2.0
        targets['raw_weight'] = targets['raw_weight'].clip(upper=max_single_weight)
        targets['final_weight'] = (targets['raw_weight'] / targets['raw_weight'].sum()) * exposure
        return targets

    def ValidatePortfolio(self, targets: pd.DataFrame, actually_invested: set, exposure: float):
        """[3] 주문 제출 전 검증. 위반 시 self.Error로 로그(백테스트를 중단시키진 않음)."""
        issues = []

        if targets['symbol'].duplicated().any():
            issues.append("중복 심볼 발견")

        if len(actually_invested) > self.max_portfolio_size:
            issues.append(f"포트폴리오 크기 초과: {len(actually_invested)} > {self.max_portfolio_size}")

        invested_weights = targets[targets['symbol'].isin(actually_invested)]
        total_weight = invested_weights['final_weight'].sum()
        if abs(total_weight - exposure) > self.WEIGHT_SUM_TOLERANCE:
            issues.append(f"총 비중 불일치: {total_weight:.3f} (목표 {exposure:.3f})")

        max_single_weight = self.base_stock_weight * 2.0
        over_weight = invested_weights[invested_weights['final_weight'] > max_single_weight + 1e-6]
        if not over_weight.empty:
            issues.append(f"개별 상한 초과 종목: {over_weight['symbol'].tolist()}")

        industry_exposure = invested_weights.groupby('industry')['final_weight'].sum()
        breached = industry_exposure[industry_exposure > self.max_industry_weight * exposure + 1e-6]
        if not breached.empty:
            issues.append(f"산업 상한 초과: {breached.to_dict()}")

        if issues:
            self.Error(f"[VALIDATION] {self.Time.date()} 포트폴리오 검증 실패: {'; '.join(issues)}")
        return issues

    def ExecuteTrades(self, targets: pd.DataFrame, ctx: RebalanceContext):
        target_symbols = set(targets['symbol'].tolist()) if not targets.empty else set()

        for symbol in list(self.Portfolio.Keys):
            if symbol == self.benchmark:
                continue
            if symbol not in target_symbols and self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)

        if targets.empty:
            self.previous_target_symbols = set()
            return

        industry_tracker = {}
        actually_invested = set()

        for _, row in targets.sort_values('final_score', ascending=False).iterrows():
            symbol = row['symbol']
            industry = row['industry']
            weight = row['final_weight']

            current_ind_weight = industry_tracker.get(industry, 0.0)
            if current_ind_weight + weight > (self.max_industry_weight * ctx.regime.exposure):
                ctx.industry_cap_rejections.append(symbol)
                continue

            self.SetHoldings(symbol, weight)
            industry_tracker[industry] = current_ind_weight + weight
            actually_invested.add(symbol)

        for symbol in target_symbols - actually_invested:
            if self.Portfolio[symbol].Invested:
                self.Liquidate(symbol, "Industry cap exceeded - explicit liquidation")

        self.ValidatePortfolio(targets, actually_invested, ctx.regime.exposure)

        ctx.actually_invested = actually_invested
        self.previous_target_symbols = actually_invested

    def LogRebalanceSummary(self, ctx: RebalanceContext):
        """[6] 구조화 로깅: 리밸런싱 1회당 한 줄 요약."""
        prior = getattr(self, "_prior_invested_for_log", set())
        entries = ctx.actually_invested - prior
        exits = prior - ctx.actually_invested
        turnover_pct = (len(entries) + len(exits)) / max(len(prior), 1)
        self._prior_invested_for_log = set(ctx.actually_invested)

        summary = (
            f"[REBALANCE] date={self.Time.date()} "
            f"regime_exposure={ctx.regime.exposure:.2f} breadth={ctx.regime.breadth:.2f} "
            f"above_ma200={ctx.regime.benchmark_above_ma200} "
            f"n_selected={len(ctx.actually_invested)} "
            f"turnover={turnover_pct:.2%} "
            f"entries={len(entries)} exits={len(exits)} "
            f"corr_rejections={len(ctx.correlation_rejections)} "
            f"industry_cap_rejections={len(ctx.industry_cap_rejections)}"
        )
        self.Log(summary)

    # =====================================================================
    # [2] 오케스트레이터: 파이프라인 순서로만 호출, 계산 로직은 각 단계에 위임
    # =====================================================================
    def RebalanceStrategy(self):
        ctx = RebalanceContext(as_of_date=self.Time)

        ctx.regime = self.DetermineMarketRegime()
        if ctx.regime is None:
            return
        self.current_exposure = ctx.regime.exposure

        if ctx.regime.exposure == 0.0:
            self.Log(f"🚨 [BEAR MARKET] 노출도 0%. (Breadth: {ctx.regime.breadth:.2f})")
            self.Liquidate()
            self.previous_target_symbols = set()
            ctx.actually_invested = set()
            self.LogRebalanceSummary(ctx)
            return
        elif ctx.regime.exposure == 0.5:
            self.Log(f"⚠️ [MARKET WARNING] 노출도 50%. (Breadth: {ctx.regime.breadth:.2f})")

        ctx.factor_df = self.BuildFactors(ctx.as_of_date)
        if ctx.factor_df.empty:
            return

        ctx.ranked_df = self.RankUniverse(ctx.factor_df)
        ctx.targets_df = self.ConstructPortfolio(ctx.ranked_df, ctx)
        ctx.targets_df = self.ApplyRiskAdjustments(ctx.targets_df, ctx.regime.exposure)
        self.ExecuteTrades(ctx.targets_df, ctx)
        self.LogRebalanceSummary(ctx)
