"""Pure evidence-versus-assumption diagnostics for the multi-stage DCF.

This module does not fetch data, change assumptions, or calculate valuation.
It only exposes comparisons and ratios from already-built engine results.
"""

from dataclasses import dataclass
import math
from typing import Literal

import pandas as pd

from Stock.fundamentals import (
    FCF_MARGIN,
    OPERATING_MARGIN,
    REVENUE,
    ROIC,
    SALES_TO_CAPITAL,
    FundamentalHistory,
)
from Stock.multistage_integration import RealCompanyDCFInputs
from Stock.valuation import (
    EnterpriseValueResult,
    MultiStageDCFAssumptions,
    MultiStageForecastPath,
    MultiStageOperatingForecast,
    TerminalValueResult,
)


MARGIN_DISTANCE_FLAG_THRESHOLD = 0.05
SALES_TO_CAPITAL_RELATIVE_DISTANCE_FLAG_THRESHOLD = 0.25
DIAGNOSTIC_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RevenueDiagnostics:
    historical_cagr_3y: float | None
    latest_ttm_revenue: float | None
    latest_annual_revenue: float | None
    year_1_growth: float
    year_2_growth: float | None
    year_3_growth: float | None
    final_explicit_growth: float
    terminal_growth: float
    year_5_revenue: float | None
    final_forecast_revenue: float
    final_to_starting_revenue_multiple: float
    cumulative_forecast_revenue_growth: float
    year_1_growth_minus_historical_cagr_3y: float | None


@dataclass(frozen=True)
class OperatingMarginDiagnostics:
    latest_annual_margin: float | None
    latest_ttm_margin: float | None
    starting_forecast_margin: float
    year_5_margin: float | None
    final_forecast_margin: float
    mature_margin: float
    total_margin_change: float
    direction: Literal["expansion", "contraction", "unchanged"]
    start_minus_current_ttm: float | None
    mature_minus_current_ttm: float | None


@dataclass(frozen=True)
class SalesToCapitalDiagnostics:
    latest_annual: float | None
    historical_normalized_3y: float | None
    starting_forecast: float
    year_5: float | None
    final_forecast: float
    mature: float
    start_minus_historical_3y: float | None
    mature_minus_historical_3y: float | None
    start_to_historical_3y_ratio: float | None


@dataclass(frozen=True)
class ROICDiagnostics:
    current_accounting_roic: float | None
    implied_mature_after_tax_operating_margin: float
    year_1_implied_operating_roic: float
    year_5_implied_operating_roic: float | None
    final_year_implied_operating_roic: float
    terminal_derived_roic: float
    terminal_minus_current_accounting_roic: float | None


@dataclass(frozen=True)
class ForecastCashFlowPoint:
    year_index: int
    reinvestment: float
    fcff: float
    fcff_to_nopat: float | None
    fcff_margin: float | None


@dataclass(frozen=True)
class CashFlowEconomicsDiagnostics:
    historical_fundamental_ttm_fcf_margin: float | None
    year_1: ForecastCashFlowPoint
    year_5: ForecastCashFlowPoint | None
    final_year: ForecastCashFlowPoint
    terminal_reinvestment_rate: float
    terminal_fcff: float
    terminal_fcff_to_nopat: float | None
    terminal_fcff_margin: float | None


@dataclass(frozen=True)
class TerminalDependencyDiagnostics:
    explicit_forecast_pv: float
    terminal_value_pv: float
    enterprise_value: float
    terminal_value_share: float | None


@dataclass(frozen=True)
class AssumptionDiagnostics:
    revenue: RevenueDiagnostics
    operating_margin: OperatingMarginDiagnostics
    sales_to_capital: SalesToCapitalDiagnostics
    roic: ROICDiagnostics
    cash_flow_economics: CashFlowEconomicsDiagnostics
    terminal_dependency: TerminalDependencyDiagnostics
    flags: tuple[str, ...]
    warnings: tuple[str, ...]


def _finite_optional(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _latest_annual(history: FundamentalHistory, metric: str) -> float | None:
    if history.annual is None or history.annual.empty or metric not in history.annual:
        return None
    series = history.annual[metric].copy()
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[~series.index.isna()].sort_index()
    return _finite_optional(series.iloc[-1]) if not series.empty else None


def _available_ttm(history: FundamentalHistory, metric: str) -> float | None:
    result = history.ttm.get(metric)
    if result is None or not result.available:
        return None
    return _finite_optional(result.value)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) <= DIAGNOSTIC_TOLERANCE:
        return None
    return numerator / denominator


def _year_or_none(operating: MultiStageOperatingForecast, year_index: int):
    return operating.years[year_index - 1] if len(operating.years) >= year_index else None


def _cash_flow_point(year) -> ForecastCashFlowPoint:
    return ForecastCashFlowPoint(
        year_index=year.year_index,
        reinvestment=year.reinvestment,
        fcff=year.fcff,
        fcff_to_nopat=_safe_ratio(year.fcff, year.nopat),
        fcff_margin=_safe_ratio(year.fcff, year.revenue),
    )


def _margin_direction(delta: float) -> Literal["expansion", "contraction", "unchanged"]:
    if delta > DIAGNOSTIC_TOLERANCE:
        return "expansion"
    if delta < -DIAGNOSTIC_TOLERANCE:
        return "contraction"
    return "unchanged"


def build_assumption_diagnostics(
    fundamentals: FundamentalHistory,
    inputs: RealCompanyDCFInputs,
    assumptions: MultiStageDCFAssumptions,
    forecast_path: MultiStageForecastPath,
    operating_forecast: MultiStageOperatingForecast,
    terminal_result: TerminalValueResult,
    enterprise_result: EnterpriseValueResult,
) -> AssumptionDiagnostics:
    """Build descriptive diagnostics from existing evidence and engine output."""
    if not isinstance(fundamentals, FundamentalHistory):
        raise TypeError("fundamentals must be FundamentalHistory")
    if not isinstance(inputs, RealCompanyDCFInputs):
        raise TypeError("inputs must be RealCompanyDCFInputs")
    if not isinstance(assumptions, MultiStageDCFAssumptions):
        raise TypeError("assumptions must be MultiStageDCFAssumptions")
    if not isinstance(forecast_path, MultiStageForecastPath):
        raise TypeError("forecast_path must be MultiStageForecastPath")
    if not isinstance(operating_forecast, MultiStageOperatingForecast):
        raise TypeError("operating_forecast must be MultiStageOperatingForecast")
    if not isinstance(terminal_result, TerminalValueResult):
        raise TypeError("terminal_result must be TerminalValueResult")
    if not isinstance(enterprise_result, EnterpriseValueResult):
        raise TypeError("enterprise_result must be EnterpriseValueResult")
    if len(forecast_path.years) != len(operating_forecast.years) or not forecast_path.years:
        raise ValueError("forecast path and operating forecast must be non-empty and aligned")

    year_1 = operating_forecast.years[0]
    year_5 = _year_or_none(operating_forecast, 5)
    final_year = operating_forecast.years[-1]
    historical_cagr_result = fundamentals.dcf_anchors.revenue_cagr.get(3)
    historical_cagr = (
        _finite_optional(historical_cagr_result.value)
        if historical_cagr_result is not None and historical_cagr_result.available
        else None
    )
    latest_ttm_revenue = _available_ttm(fundamentals, REVENUE)
    latest_annual_revenue = _latest_annual(fundamentals, REVENUE)
    revenue = RevenueDiagnostics(
        historical_cagr_3y=historical_cagr,
        latest_ttm_revenue=latest_ttm_revenue,
        latest_annual_revenue=latest_annual_revenue,
        year_1_growth=forecast_path.years[0].revenue_growth,
        year_2_growth=(forecast_path.years[1].revenue_growth if len(forecast_path.years) >= 2 else None),
        year_3_growth=(forecast_path.years[2].revenue_growth if len(forecast_path.years) >= 3 else None),
        final_explicit_growth=forecast_path.years[-1].revenue_growth,
        terminal_growth=assumptions.terminal_growth,
        year_5_revenue=year_5.revenue if year_5 is not None else None,
        final_forecast_revenue=operating_forecast.ending_revenue,
        final_to_starting_revenue_multiple=(
            operating_forecast.ending_revenue / operating_forecast.starting_revenue
        ),
        cumulative_forecast_revenue_growth=operating_forecast.cumulative_revenue_growth,
        year_1_growth_minus_historical_cagr_3y=(
            year_1.revenue_growth - historical_cagr if historical_cagr is not None else None
        ),
    )

    latest_annual_margin = _latest_annual(fundamentals, OPERATING_MARGIN)
    latest_ttm_margin = _available_ttm(fundamentals, OPERATING_MARGIN)
    margin_delta = assumptions.mature_operating_margin - assumptions.starting_operating_margin
    operating_margin = OperatingMarginDiagnostics(
        latest_annual_margin=latest_annual_margin,
        latest_ttm_margin=latest_ttm_margin,
        starting_forecast_margin=assumptions.starting_operating_margin,
        year_5_margin=year_5.operating_margin if year_5 is not None else None,
        final_forecast_margin=final_year.operating_margin,
        mature_margin=assumptions.mature_operating_margin,
        total_margin_change=margin_delta,
        direction=_margin_direction(margin_delta),
        start_minus_current_ttm=(
            assumptions.starting_operating_margin - latest_ttm_margin
            if latest_ttm_margin is not None else None
        ),
        mature_minus_current_ttm=(
            assumptions.mature_operating_margin - latest_ttm_margin
            if latest_ttm_margin is not None else None
        ),
    )

    latest_annual_stc = _latest_annual(fundamentals, SALES_TO_CAPITAL)
    historical_stc = inputs.historical_sales_to_capital_3y
    sales_to_capital = SalesToCapitalDiagnostics(
        latest_annual=latest_annual_stc,
        historical_normalized_3y=historical_stc,
        starting_forecast=assumptions.starting_sales_to_capital,
        year_5=year_5.sales_to_capital if year_5 is not None else None,
        final_forecast=final_year.sales_to_capital,
        mature=assumptions.mature_sales_to_capital,
        start_minus_historical_3y=(
            assumptions.starting_sales_to_capital - historical_stc
            if historical_stc is not None else None
        ),
        mature_minus_historical_3y=(
            assumptions.mature_sales_to_capital - historical_stc
            if historical_stc is not None else None
        ),
        start_to_historical_3y_ratio=(
            _safe_ratio(assumptions.starting_sales_to_capital, historical_stc)
            if historical_stc is not None else None
        ),
    )

    def implied_roic(year) -> float:
        return year.operating_margin * (1 - assumptions.operating_tax_rate) * year.sales_to_capital

    current_roic = inputs.current_accounting_roic
    roic = ROICDiagnostics(
        current_accounting_roic=current_roic,
        implied_mature_after_tax_operating_margin=assumptions.after_tax_mature_operating_margin,
        year_1_implied_operating_roic=implied_roic(year_1),
        year_5_implied_operating_roic=(implied_roic(year_5) if year_5 is not None else None),
        final_year_implied_operating_roic=implied_roic(final_year),
        terminal_derived_roic=terminal_result.derived_terminal_roic,
        terminal_minus_current_accounting_roic=(
            terminal_result.derived_terminal_roic - current_roic
            if current_roic is not None else None
        ),
    )

    cash_flow = CashFlowEconomicsDiagnostics(
        historical_fundamental_ttm_fcf_margin=_available_ttm(fundamentals, FCF_MARGIN),
        year_1=_cash_flow_point(year_1),
        year_5=_cash_flow_point(year_5) if year_5 is not None else None,
        final_year=_cash_flow_point(final_year),
        terminal_reinvestment_rate=terminal_result.terminal_reinvestment_rate,
        terminal_fcff=terminal_result.terminal_fcff,
        terminal_fcff_to_nopat=_safe_ratio(terminal_result.terminal_fcff, terminal_result.terminal_nopat),
        terminal_fcff_margin=_safe_ratio(terminal_result.terminal_fcff, terminal_result.terminal_year_revenue),
    )
    terminal_dependency = TerminalDependencyDiagnostics(
        explicit_forecast_pv=enterprise_result.explicit_forecast_pv,
        terminal_value_pv=enterprise_result.terminal_value_pv,
        enterprise_value=enterprise_result.enterprise_value,
        terminal_value_share=enterprise_result.terminal_value_share,
    )

    flags: list[str] = []
    if (
        operating_margin.start_minus_current_ttm is not None
        and abs(operating_margin.start_minus_current_ttm) >= MARGIN_DISTANCE_FLAG_THRESHOLD
    ):
        flags.append("forecast_start_margin_far_from_current")
    if historical_stc is not None and abs(historical_stc) > DIAGNOSTIC_TOLERANCE:
        relative_distance = abs(
            assumptions.starting_sales_to_capital - historical_stc
        ) / abs(historical_stc)
        if relative_distance >= SALES_TO_CAPITAL_RELATIVE_DISTANCE_FLAG_THRESHOLD:
            flags.append("forecast_start_sales_to_capital_far_from_historical")
    if current_roic is not None and terminal_result.derived_terminal_roic > current_roic + DIAGNOSTIC_TOLERANCE:
        flags.append("terminal_roic_above_current_accounting_roic")
    if not math.isclose(final_year.revenue_growth, assumptions.terminal_growth, abs_tol=DIAGNOSTIC_TOLERANCE):
        flags.append("revenue_never_reaches_terminal_growth")
    if not (
        math.isclose(final_year.operating_margin, assumptions.mature_operating_margin, abs_tol=DIAGNOSTIC_TOLERANCE)
        and math.isclose(final_year.sales_to_capital, assumptions.mature_sales_to_capital, abs_tol=DIAGNOSTIC_TOLERANCE)
    ):
        flags.append("final_state_not_mature")
    warnings = tuple(dict.fromkeys(terminal_result.warnings + enterprise_result.warnings))
    if "terminal_value_dominates_enterprise_value" in warnings:
        flags.append("terminal_value_dominates_enterprise_value")

    return AssumptionDiagnostics(
        revenue=revenue,
        operating_margin=operating_margin,
        sales_to_capital=sales_to_capital,
        roic=roic,
        cash_flow_economics=cash_flow,
        terminal_dependency=terminal_dependency,
        flags=tuple(dict.fromkeys(flags)),
        warnings=warnings,
    )
