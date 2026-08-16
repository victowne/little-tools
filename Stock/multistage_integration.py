"""Thin real-company adapter for the pure multi-stage DCF engine.

This module deliberately does not fetch data, inspect market prices, derive
forecast assumptions from history, or duplicate valuation formulas.
"""

from dataclasses import dataclass
import math
from typing import Protocol

import pandas as pd

from Stock.fundamentals import (
    REVENUE,
    ROIC,
    FundamentalHistory,
)
from Stock.share_normalization import NormalizedShareCount, normalize_share_count
from Stock.valuation_support import assess_per_security_valuation_support
from Stock.valuation import (
    EnterpriseValueResult,
    EquityValueResult,
    MultiStageDCFAssumptions,
    MultiStageDiscountedForecast,
    MultiStageForecastPath,
    MultiStageOperatingForecast,
    PerShareValueResult,
    TerminalValueResult,
    aggregate_enterprise_value,
    bridge_enterprise_to_equity_value,
    build_operating_forecast,
    calculate_intrinsic_value_per_share,
    calculate_terminal_value,
    discount_operating_forecast,
    generate_forecast_path,
)


class CompanySnapshotLike(Protocol):
    """The existing snapshot fields required by this diagnostic adapter."""

    ticker: str
    shares_outstanding: float | None
    net_debt: float | None
    net_debt_source: str | None
    net_debt_period: pd.Timestamp | None
    financial_currency: str | None
    price_currency: str | None


@dataclass(frozen=True)
class RealCompanyDCFInputs:
    """Validated real-company inputs and non-binding historical context."""

    ticker: str
    starting_revenue: float
    starting_revenue_source: str
    starting_revenue_periods: tuple[pd.Timestamp, ...]
    net_debt: float
    net_debt_source: str
    net_debt_period: pd.Timestamp | None
    shares_outstanding: float | None
    normalized_share_count: NormalizedShareCount
    historical_sales_to_capital_3y: float | None
    current_accounting_roic: float | None
    per_security_valuation_supported: bool = True
    per_security_valuation_unsupported_reason: str | None = None
    statement_currency: str | None = None
    security_currency: str | None = None

    def __post_init__(self) -> None:
        if self.per_security_valuation_supported:
            if self.per_security_valuation_unsupported_reason is not None:
                raise ValueError(
                    "supported per-security valuation cannot have an "
                    "unsupported reason"
                )
        elif not self.per_security_valuation_unsupported_reason:
            raise ValueError(
                "unsupported per-security valuation requires a reason"
            )


@dataclass(frozen=True)
class MultiStageDCFRunResult:
    """All existing pure-engine results from one explicitly assumed run."""

    inputs: RealCompanyDCFInputs
    assumptions: MultiStageDCFAssumptions
    forecast_path: MultiStageForecastPath
    operating_forecast: MultiStageOperatingForecast
    discounted_forecast: MultiStageDiscountedForecast
    terminal_value: TerminalValueResult
    enterprise_value: EnterpriseValueResult
    equity_value: EquityValueResult
    per_share_value: PerShareValueResult | None
    per_share_unavailable_reason: str | None

    @property
    def per_security_valuation_supported(self) -> bool:
        return self.inputs.per_security_valuation_supported

    @property
    def warnings(self) -> tuple[str, ...]:
        combined = list(self.equity_value.warnings)
        combined.extend(self.inputs.normalized_share_count.warnings)
        if self.per_share_value is not None:
            combined.extend(self.per_share_value.warnings)
        if self.per_share_unavailable_reason is not None:
            combined.append(self.per_share_unavailable_reason)
        return tuple(dict.fromkeys(combined))


def _finite_optional(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _latest_annual_value(
    history: FundamentalHistory,
    metric: str,
) -> tuple[float | None, pd.Timestamp | None]:
    if history.annual is None or history.annual.empty or metric not in history.annual:
        return None, None
    frame = history.annual[[metric]].copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame[~frame.index.isna()].sort_index()
    if frame.empty:
        return None, None
    period = pd.Timestamp(frame.index[-1])
    return _finite_optional(frame.iloc[-1][metric]), period


def extract_real_company_dcf_inputs(
    snapshot: CompanySnapshotLike,
    fundamentals: FundamentalHistory,
) -> RealCompanyDCFInputs:
    """Extract required inputs without silently manufacturing fallbacks.

    Validated TTM Revenue is preferred. If it is unavailable, only Revenue at
    the latest annual period is accepted, and the result explicitly identifies
    ``annual_fallback``. A missing latest annual value is not replaced by an
    older observation. Net debt and current common shares are consumed exactly
    as stored in ``CompanySnapshot``; neither is recomputed here.
    """
    if not isinstance(fundamentals, FundamentalHistory):
        raise TypeError("fundamentals must be FundamentalHistory")

    ttm_revenue = fundamentals.ttm.get(REVENUE)
    if (
        ttm_revenue is not None
        and ttm_revenue.available
        and _finite_optional(ttm_revenue.value) is not None
        and float(ttm_revenue.value) > 0
    ):
        starting_revenue = float(ttm_revenue.value)
        revenue_source = "ttm"
        revenue_periods = tuple(pd.Timestamp(p) for p in ttm_revenue.periods_used)
    else:
        annual_revenue, annual_period = _latest_annual_value(
            fundamentals, REVENUE
        )
        if annual_revenue is None or annual_revenue <= 0 or annual_period is None:
            raise ValueError("starting_revenue_unavailable")
        starting_revenue = annual_revenue
        revenue_source = "annual_fallback"
        revenue_periods = (annual_period,)

    net_debt = _finite_optional(getattr(snapshot, "net_debt", None))
    if net_debt is None:
        raise ValueError("net_debt_unavailable")
    normalized_shares = normalize_share_count(snapshot)
    shares = normalized_shares.shares_outstanding
    support = assess_per_security_valuation_support(
        ticker=str(snapshot.ticker),
        statement_currency=getattr(snapshot, "financial_currency", None),
        security_currency=getattr(snapshot, "price_currency", None),
    )

    historical_anchor = fundamentals.dcf_anchors.normalized_sales_to_capital.get(3)
    historical_sales_to_capital = (
        _finite_optional(historical_anchor.value)
        if historical_anchor is not None and historical_anchor.available
        else None
    )
    current_roic, _ = _latest_annual_value(fundamentals, ROIC)

    source = getattr(snapshot, "net_debt_source", None) or "company_snapshot"
    source_period = getattr(snapshot, "net_debt_period", None)
    if source_period is not None:
        source_period = pd.Timestamp(source_period)

    return RealCompanyDCFInputs(
        ticker=str(snapshot.ticker).strip().upper(),
        starting_revenue=starting_revenue,
        starting_revenue_source=revenue_source,
        starting_revenue_periods=revenue_periods,
        net_debt=net_debt,
        net_debt_source=source,
        net_debt_period=source_period,
        shares_outstanding=shares,
        normalized_share_count=normalized_shares,
        historical_sales_to_capital_3y=historical_sales_to_capital,
        current_accounting_roic=current_roic,
        per_security_valuation_supported=support.supported,
        per_security_valuation_unsupported_reason=support.reason,
        statement_currency=support.statement_currency,
        security_currency=support.security_currency,
    )


def run_multistage_dcf(
    inputs: RealCompanyDCFInputs,
    assumptions: MultiStageDCFAssumptions,
) -> MultiStageDCFRunResult:
    """Compose the seven existing pure valuation layers without new formulas."""
    if not isinstance(inputs, RealCompanyDCFInputs):
        raise TypeError("inputs must be RealCompanyDCFInputs")
    if not isinstance(assumptions, MultiStageDCFAssumptions):
        raise TypeError("assumptions must be MultiStageDCFAssumptions")

    path = generate_forecast_path(assumptions)
    operating = build_operating_forecast(inputs.starting_revenue, assumptions, path)
    discounted = discount_operating_forecast(operating, assumptions)
    terminal = calculate_terminal_value(operating, discounted, assumptions)
    enterprise = aggregate_enterprise_value(discounted, terminal, assumptions)
    equity = bridge_enterprise_to_equity_value(enterprise, inputs.net_debt)
    if not inputs.per_security_valuation_supported:
        per_share = None
        per_share_reason = inputs.per_security_valuation_unsupported_reason
    elif inputs.shares_outstanding is None:
        per_share = None
        per_share_reason = (
            inputs.normalized_share_count.reason
            or "consolidated_share_count_unavailable"
        )
    else:
        per_share = calculate_intrinsic_value_per_share(
            equity, inputs.shares_outstanding
        )
        per_share_reason = None
    return MultiStageDCFRunResult(
        inputs=inputs,
        assumptions=assumptions,
        forecast_path=path,
        operating_forecast=operating,
        discounted_forecast=discounted,
        terminal_value=terminal,
        enterprise_value=enterprise,
        equity_value=equity,
        per_share_value=per_share,
        per_share_unavailable_reason=per_share_reason,
    )


def run_real_company_multistage_dcf(
    snapshot: CompanySnapshotLike,
    fundamentals: FundamentalHistory,
    assumptions: MultiStageDCFAssumptions,
) -> MultiStageDCFRunResult:
    """Extract real-company inputs, then run explicit supplied assumptions."""
    return run_multistage_dcf(
        extract_real_company_dcf_inputs(snapshot, fundamentals), assumptions
    )
