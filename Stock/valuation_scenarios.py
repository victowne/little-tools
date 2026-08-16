"""Pure coherent Bear/Base/Bull orchestration for the existing DCF engine.

Every available scenario contains a complete ``MultiStageDCFAssumptions``
object and is run independently through ``run_multistage_dcf``.  This module
does not generate scenario assumptions, assign probabilities, access market
data, or compare intrinsic values with market prices.
"""

from dataclasses import dataclass, replace
import math
from typing import Literal

from Stock.assumption_diagnostics import (
    AssumptionDiagnostics,
    build_assumption_diagnostics,
)
from Stock.fundamentals import FundamentalHistory
from Stock.multistage_integration import (
    MultiStageDCFRunResult,
    RealCompanyDCFInputs,
    run_multistage_dcf,
)
from Stock.valuation import MultiStageDCFAssumptions


ScenarioName = Literal["bear", "base", "bull"]


@dataclass(frozen=True)
class ScenarioCase:
    name: ScenarioName
    assumptions: MultiStageDCFAssumptions | None
    rationale: str
    warnings: tuple[str, ...]
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.name not in {"bear", "base", "bull"}:
            raise ValueError("scenario name must be bear, base, or bull")
        if self.assumptions is None and not self.unavailable_reason:
            raise ValueError("unavailable scenario requires a reason")
        if self.assumptions is not None and self.unavailable_reason is not None:
            raise ValueError("available scenario must not have an unavailable reason")


@dataclass(frozen=True)
class ScenarioMetrics:
    intrinsic_value_per_share: float | None
    enterprise_value: float
    equity_value: float
    explicit_forecast_pv: float
    terminal_value_pv: float
    terminal_value_share: float | None
    year_1_revenue_growth: float
    year_2_revenue_growth: float | None
    year_3_revenue_growth: float | None
    revenue_fade_years: int
    year_5_revenue: float | None
    final_forecast_revenue: float
    final_revenue_to_starting_revenue: float
    starting_operating_margin: float
    year_5_operating_margin: float | None
    mature_operating_margin: float
    starting_sales_to_capital: float
    year_5_sales_to_capital: float | None
    mature_sales_to_capital: float
    terminal_roic: float
    terminal_reinvestment_rate: float
    year_1_fcff_margin: float | None
    year_5_fcff_margin: float | None
    final_year_fcff_margin: float | None
    terminal_fcff_to_nopat: float | None
    research_wacc: float
    terminal_growth: float
    wacc_terminal_growth_spread: float


@dataclass(frozen=True)
class ScenarioDeltaVsBase:
    intrinsic_value_difference: float | None
    intrinsic_value_percentage_difference: float | None
    final_revenue_difference: float | None
    mature_operating_margin_difference: float | None
    mature_sales_to_capital_difference: float | None
    research_wacc_difference: float | None


@dataclass(frozen=True)
class ScenarioRunResult:
    name: ScenarioName
    assumptions: MultiStageDCFAssumptions | None
    dcf_result: MultiStageDCFRunResult | None
    diagnostics: AssumptionDiagnostics | None
    metrics: ScenarioMetrics | None
    delta_vs_base: ScenarioDeltaVsBase | None
    available: bool
    reason: str | None
    rationale: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MultiScenarioDCFResult:
    bear: ScenarioRunResult
    base: ScenarioRunResult
    bull: ScenarioRunResult
    warnings: tuple[str, ...]

    @property
    def scenarios(self) -> tuple[ScenarioRunResult, ...]:
        return self.bear, self.base, self.bull


def create_scenario_from_base(
    name: ScenarioName,
    base: MultiStageDCFAssumptions,
    *,
    rationale: str = "",
    warnings: tuple[str, ...] = (),
    **explicit_overrides,
) -> ScenarioCase:
    """Resolve explicit overrides into a complete immutable assumption object.

    ``research_wacc`` is accepted as a terminology alias for the existing
    assumptions field ``wacc``.  Invalid overrides produce one unavailable
    case so another scenario can still run.
    """
    if not isinstance(base, MultiStageDCFAssumptions):
        raise TypeError("base must be MultiStageDCFAssumptions")
    overrides = dict(explicit_overrides)
    if "research_wacc" in overrides:
        if "wacc" in overrides:
            return ScenarioCase(
                name, None, str(rationale).strip(), tuple(warnings),
                "both_research_wacc_and_wacc_supplied",
            )
        overrides["wacc"] = overrides.pop("research_wacc")
    if not overrides:
        return ScenarioCase(
            name=name,
            assumptions=base,
            rationale=str(rationale).strip(),
            warnings=tuple(warnings),
            unavailable_reason=None,
        )
    try:
        assumptions = replace(base, **overrides)
    except (TypeError, ValueError) as exc:
        return ScenarioCase(
            name=name, assumptions=None, rationale=str(rationale).strip(),
            warnings=tuple(warnings), unavailable_reason=str(exc),
        )
    return ScenarioCase(
        name=name, assumptions=assumptions, rationale=str(rationale).strip(),
        warnings=tuple(warnings), unavailable_reason=None,
    )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def _metrics(
    result: MultiStageDCFRunResult,
    diagnostics: AssumptionDiagnostics,
) -> ScenarioMetrics:
    assumptions = result.assumptions
    revenue = diagnostics.revenue
    margin = diagnostics.operating_margin
    capital = diagnostics.sales_to_capital
    cash = diagnostics.cash_flow_economics
    terminal = diagnostics.terminal_dependency
    return ScenarioMetrics(
        intrinsic_value_per_share=(
            result.per_share_value.intrinsic_value_per_share
            if result.per_share_value is not None else None
        ),
        enterprise_value=result.enterprise_value.enterprise_value,
        equity_value=result.equity_value.equity_value,
        explicit_forecast_pv=terminal.explicit_forecast_pv,
        terminal_value_pv=terminal.terminal_value_pv,
        terminal_value_share=terminal.terminal_value_share,
        year_1_revenue_growth=revenue.year_1_growth,
        year_2_revenue_growth=revenue.year_2_growth,
        year_3_revenue_growth=revenue.year_3_growth,
        revenue_fade_years=assumptions.revenue_fade_years,
        year_5_revenue=revenue.year_5_revenue,
        final_forecast_revenue=revenue.final_forecast_revenue,
        final_revenue_to_starting_revenue=revenue.final_to_starting_revenue_multiple,
        starting_operating_margin=margin.starting_forecast_margin,
        year_5_operating_margin=margin.year_5_margin,
        mature_operating_margin=margin.mature_margin,
        starting_sales_to_capital=capital.starting_forecast,
        year_5_sales_to_capital=capital.year_5,
        mature_sales_to_capital=capital.mature,
        terminal_roic=diagnostics.roic.terminal_derived_roic,
        terminal_reinvestment_rate=cash.terminal_reinvestment_rate,
        year_1_fcff_margin=cash.year_1.fcff_margin,
        year_5_fcff_margin=(cash.year_5.fcff_margin if cash.year_5 else None),
        final_year_fcff_margin=cash.final_year.fcff_margin,
        terminal_fcff_to_nopat=cash.terminal_fcff_to_nopat,
        research_wacc=assumptions.wacc,
        terminal_growth=assumptions.terminal_growth,
        wacc_terminal_growth_spread=assumptions.wacc - assumptions.terminal_growth,
    )


def _run_case(
    case: ScenarioCase,
    inputs: RealCompanyDCFInputs,
    fundamentals: FundamentalHistory,
) -> ScenarioRunResult:
    if case.assumptions is None:
        return ScenarioRunResult(
            name=case.name, assumptions=None, dcf_result=None,
            diagnostics=None, metrics=None, delta_vs_base=None,
            available=False, reason=case.unavailable_reason,
            rationale=case.rationale, warnings=case.warnings,
        )
    try:
        result = run_multistage_dcf(inputs, case.assumptions)
        diagnostics = build_assumption_diagnostics(
            fundamentals, inputs, case.assumptions, result.forecast_path,
            result.operating_forecast, result.terminal_value,
            result.enterprise_value,
        )
        metrics = _metrics(result, diagnostics)
    except (TypeError, ValueError) as exc:
        return ScenarioRunResult(
            name=case.name, assumptions=case.assumptions, dcf_result=None,
            diagnostics=None, metrics=None, delta_vs_base=None,
            available=False, reason=str(exc), rationale=case.rationale,
            warnings=case.warnings,
        )
    warnings = tuple(dict.fromkeys(
        case.warnings
        + case.assumptions.validation_warnings
        + result.warnings
        + diagnostics.warnings
    ))
    return ScenarioRunResult(
        name=case.name, assumptions=case.assumptions, dcf_result=result,
        diagnostics=diagnostics, metrics=metrics, delta_vs_base=None,
        available=True, reason=result.per_share_unavailable_reason,
        rationale=case.rationale, warnings=warnings,
    )


def _delta(
    scenario: ScenarioRunResult,
    base: ScenarioRunResult,
) -> ScenarioDeltaVsBase | None:
    if scenario.metrics is None or base.metrics is None:
        return None
    scenario_value = scenario.metrics.intrinsic_value_per_share
    base_value = base.metrics.intrinsic_value_per_share
    value_difference = (
        scenario_value - base_value
        if scenario_value is not None and base_value is not None else None
    )
    percentage_difference = (
        scenario_value / base_value - 1
        if scenario_value is not None and base_value is not None
        and abs(base_value) > 1e-12 else None
    )
    return ScenarioDeltaVsBase(
        intrinsic_value_difference=value_difference,
        intrinsic_value_percentage_difference=percentage_difference,
        final_revenue_difference=(
            scenario.metrics.final_forecast_revenue
            - base.metrics.final_forecast_revenue
        ),
        mature_operating_margin_difference=(
            scenario.metrics.mature_operating_margin
            - base.metrics.mature_operating_margin
        ),
        mature_sales_to_capital_difference=(
            scenario.metrics.mature_sales_to_capital
            - base.metrics.mature_sales_to_capital
        ),
        research_wacc_difference=(
            scenario.metrics.research_wacc - base.metrics.research_wacc
        ),
    )


def run_multi_scenario_dcf(
    *,
    inputs: RealCompanyDCFInputs,
    fundamentals: FundamentalHistory,
    bear: ScenarioCase,
    base: ScenarioCase,
    bull: ScenarioCase,
) -> MultiScenarioDCFResult:
    """Run three complete independent DCF cases with per-case fault isolation."""
    if not isinstance(inputs, RealCompanyDCFInputs):
        raise TypeError("inputs must be RealCompanyDCFInputs")
    if not isinstance(fundamentals, FundamentalHistory):
        raise TypeError("fundamentals must be FundamentalHistory")
    if (bear.name, base.name, bull.name) != ("bear", "base", "bull"):
        raise ValueError("scenario arguments must be ordered bear, base, bull")

    bear_run = _run_case(bear, inputs, fundamentals)
    base_run = _run_case(base, inputs, fundamentals)
    bull_run = _run_case(bull, inputs, fundamentals)
    bear_run = replace(bear_run, delta_vs_base=_delta(bear_run, base_run))
    base_run = replace(base_run, delta_vs_base=_delta(base_run, base_run))
    bull_run = replace(bull_run, delta_vs_base=_delta(bull_run, base_run))

    warnings = []
    if all(
        item.metrics is not None
        and item.metrics.intrinsic_value_per_share is not None
        for item in (bear_run, base_run, bull_run)
    ):
        bear_value = bear_run.metrics.intrinsic_value_per_share
        base_value = base_run.metrics.intrinsic_value_per_share
        bull_value = bull_run.metrics.intrinsic_value_per_share
        if bear_value > base_value or bull_value < base_value:
            warnings.append("scenario_value_order_unexpected")
    if not bear_run.available:
        warnings.append("bear_scenario_unavailable")
    if not base_run.available:
        warnings.append("base_scenario_unavailable")
    if not bull_run.available:
        warnings.append("bull_scenario_unavailable")
    return MultiScenarioDCFResult(
        bear=bear_run, base=base_run, bull=bull_run,
        warnings=tuple(warnings),
    )
