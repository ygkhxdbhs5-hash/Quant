"""Adapters that present local fundamental rows as LEAN Fine-like objects.

Used only so FineSelectionFunction can run unchanged against on-disk data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class _Value:
    Value: Any


@dataclass
class _OperationRatios:
    OperatingMargin: Optional[_Value] = None
    ROIC: Optional[_Value] = None


@dataclass
class _IncomeStatement:
    GrossProfit: Optional[_Value] = None
    TotalRevenue: Optional[_Value] = None


@dataclass
class _BalanceSheet:
    TotalAssets: Optional[_Value] = None
    TotalDebt: Optional[_Value] = None
    CashAndCashEquivalents: Optional[_Value] = None


@dataclass
class _CashFlowStatement:
    CashFlowFromOperatingActivities: Optional[_Value] = None
    CapEx: Optional[_Value] = None


@dataclass
class _FinancialStatements:
    IncomeStatement: _IncomeStatement = field(default_factory=_IncomeStatement)
    BalanceSheet: _BalanceSheet = field(default_factory=_BalanceSheet)
    CashFlowStatement: _CashFlowStatement = field(default_factory=_CashFlowStatement)


@dataclass
class _Summary:
    MarketCap: Optional[_Value] = None


@dataclass
class _CompanyReference:
    IndustryGroup: str = "Unknown"


@dataclass
class CoarseFundamental:
    Symbol: str
    Price: float
    Volume: float
    HasFundamentalData: bool = True
    DollarVolume: float = 0.0


@dataclass
class FineFundamental:
    Symbol: str
    DollarVolume: float
    OperationRatios: _OperationRatios
    FinancialStatements: _FinancialStatements
    Summary: _Summary
    CompanyReference: _CompanyReference


def fine_from_row(symbol: str, row: dict) -> FineFundamental:
    """Build a Fine-like object from a local fundamentals row.

    Expected columns (engine snapshot inputs):
      op_margin, roic, gross_profit, total_revenue, total_assets,
      ocf, capex, market_cap, total_debt, cash, industry, dollar_volume
    """
    total_assets = float(row.get("total_assets") or 0.0)
    gross_profit = float(row.get("gross_profit") or 0.0)
    # Allow precomputed gross_profitability if raw gross_profit absent
    if gross_profit == 0.0 and total_assets > 0 and row.get("gross_profitability") is not None:
        gross_profit = float(row["gross_profitability"]) * total_assets

    total_debt = row.get("total_debt")
    cash = row.get("cash")

    return FineFundamental(
        Symbol=symbol,
        DollarVolume=float(row.get("dollar_volume") or 0.0),
        OperationRatios=_OperationRatios(
            OperatingMargin=_Value(float(row["op_margin"])) if row.get("op_margin") is not None else None,
            ROIC=_Value(float(row["roic"])) if row.get("roic") is not None else None,
        ),
        FinancialStatements=_FinancialStatements(
            IncomeStatement=_IncomeStatement(
                GrossProfit=_Value(gross_profit) if gross_profit else None,
                TotalRevenue=_Value(float(row["total_revenue"])) if row.get("total_revenue") is not None else None,
            ),
            BalanceSheet=_BalanceSheet(
                TotalAssets=_Value(total_assets) if total_assets else None,
                TotalDebt=_Value(float(total_debt)) if total_debt is not None else None,
                CashAndCashEquivalents=_Value(float(cash)) if cash is not None else None,
            ),
            CashFlowStatement=_CashFlowStatement(
                CashFlowFromOperatingActivities=_Value(float(row["ocf"])) if row.get("ocf") is not None else None,
                CapEx=_Value(float(row["capex"])) if row.get("capex") is not None else None,
            ),
        ),
        Summary=_Summary(
            MarketCap=_Value(float(row["market_cap"])) if row.get("market_cap") is not None else None,
        ),
        CompanyReference=_CompanyReference(
            IndustryGroup=str(row.get("industry") or "Unknown"),
        ),
    )


@dataclass
class SecurityChanges:
    AddedSecurities: list = field(default_factory=list)
    RemovedSecurities: list = field(default_factory=list)
