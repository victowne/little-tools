"""Pure five-year hybrid explicit-reinvestment methodology prototype.

CapEx and D&A are positive amounts.  Positive change in working capital is a
cash investment/outflow; a negative value is a cash release.  The prototype
replaces only explicit-period reinvestment and FCFF, then reuses the existing
discounting, terminal, enterprise, equity, and per-share valuation functions.
"""

from dataclasses import dataclass, replace
import math
from typing import Literal

from Stock.multistage_integration import MultiStageDCFRunResult
from Stock.valuation import (
    MultiStageOperatingForecast,
    aggregate_enterprise_value,
    bridge_enterprise_to_equity_value,
    calculate_intrinsic_value_per_share,
    calculate_terminal_value,
    discount_operating_forecast,
)


Confidence = Literal["High", "Medium", "Low"]
MethodologyClassification = Literal[
    "CLEARLY BETTER REPRESENTATION",
    "SOMEWHAT BETTER",
    "NO MATERIAL IMPROVEMENT",
    "WORSE / MORE MISLEADING",
    "INCONCLUSIVE",
]
EXPLICIT_PROTOTYPE_YEARS = 5
MATCH_TOLERANCE = 1e-9


@dataclass(frozen=True)
class HybridReinvestmentYearInput:
    year: int
    revenue: float
    operating_margin: float
    operating_tax_rate: float
    capex: float
    depreciation_amortization: float
    change_in_working_capital: float = 0.0
    other_reinvestment: float = 0.0
    source_confidence: Confidence = "Low"
    rationale: str = ""


@dataclass(frozen=True)
class HybridReinvestmentYearResult:
    year: int
    revenue: float
    operating_margin: float
    operating_tax_rate: float
    nopat: float
    gross_capex: float
    depreciation_amortization: float
    net_capex: float
    change_in_working_capital: float
    other_reinvestment: float
    total_reinvestment: float
    fcff: float
    fcff_margin: float
    source_confidence: Confidence
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TransitionToSalesToCapitalDiagnostic:
    final_explicit_net_capex: float
    final_explicit_total_reinvestment: float
    final_year_sales_to_capital_reinvestment: float
    first_normalized_year_sales_to_capital_reinvestment: float | None
    final_explicit_gap: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class HybridReinvestmentComparison:
    ticker: str
    existing_run: MultiStageDCFRunResult
    hybrid_run: MultiStageDCFRunResult
    hybrid_years: tuple[HybridReinvestmentYearResult, ...]
    cumulative_sales_to_capital_reinvestment: float
    cumulative_hybrid_reinvestment: float
    cumulative_sales_to_capital_fcff: float
    cumulative_hybrid_fcff: float
    five_year_sales_to_capital_fcff_pv: float
    five_year_hybrid_fcff_pv: float
    transition: TransitionToSalesToCapitalDiagnostic
    classification: MethodologyClassification
    warnings: tuple[str, ...]


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def calculate_hybrid_reinvestment_year(
    item: HybridReinvestmentYearInput,
) -> HybridReinvestmentYearResult:
    if item.year < 1:
        raise ValueError("year must be positive")
    revenue = _finite("revenue", item.revenue)
    margin = _finite("operating_margin", item.operating_margin)
    tax = _finite("operating_tax_rate", item.operating_tax_rate)
    capex = _finite("capex", item.capex)
    depreciation = _finite(
        "depreciation_amortization", item.depreciation_amortization
    )
    working_capital = _finite(
        "change_in_working_capital", item.change_in_working_capital
    )
    other = _finite("other_reinvestment", item.other_reinvestment)
    if revenue <= 0:
        raise ValueError("revenue must be positive")
    if capex < 0:
        raise ValueError("capex must use positive-outflow convention")
    if depreciation < 0:
        raise ValueError("depreciation must use positive-expense convention")
    if not 0 <= tax <= 1:
        raise ValueError("operating_tax_rate must be between zero and one")

    nopat = revenue * margin * (1 - tax)
    net_capex = capex - depreciation
    reinvestment = net_capex + working_capital + other
    fcff = nopat - reinvestment
    warnings = []
    if net_capex < 0:
        warnings.append("negative_net_capex")
    if depreciation > capex:
        warnings.append("depreciation_exceeds_capex")
    if working_capital < -0.10 * revenue:
        warnings.append("large_working_capital_release")
    if abs(reinvestment) > revenue:
        warnings.append("reinvestment_exceeds_revenue")
    return HybridReinvestmentYearResult(
        item.year, revenue, margin, tax, nopat, capex, depreciation,
        net_capex, working_capital, other, reinvestment, fcff,
        fcff / revenue, item.source_confidence, tuple(warnings),
    )


def calculate_hybrid_reinvestment_path(
    inputs: tuple[HybridReinvestmentYearInput, ...],
) -> tuple[HybridReinvestmentYearResult, ...]:
    if len(inputs) != EXPLICIT_PROTOTYPE_YEARS:
        raise ValueError("hybrid prototype requires exactly five years")
    if tuple(item.year for item in inputs) != tuple(
        range(1, EXPLICIT_PROTOTYPE_YEARS + 1)
    ):
        raise ValueError("hybrid years must be consecutive from one")
    return tuple(calculate_hybrid_reinvestment_year(item) for item in inputs)


def build_hybrid_shadow_dcf(
    existing_run: MultiStageDCFRunResult,
    hybrid_inputs: tuple[HybridReinvestmentYearInput, ...],
) -> tuple[MultiStageDCFRunResult, tuple[HybridReinvestmentYearResult, ...]]:
    """Replace five FCFF/reinvestment rows while reusing all valuation layers."""
    hybrid_results = calculate_hybrid_reinvestment_path(hybrid_inputs)
    existing_years = existing_run.operating_forecast.years
    if len(existing_years) < EXPLICIT_PROTOTYPE_YEARS:
        raise ValueError("existing forecast has fewer than five years")
    replaced_years = list(existing_years)
    for index, (existing, hybrid) in enumerate(
        zip(existing_years[:EXPLICIT_PROTOTYPE_YEARS], hybrid_results)
    ):
        for name, left, right in (
            ("revenue", existing.revenue, hybrid.revenue),
            ("operating_margin", existing.operating_margin, hybrid.operating_margin),
            ("operating_tax_rate", existing.operating_tax_rate, hybrid.operating_tax_rate),
            ("nopat", existing.nopat, hybrid.nopat),
        ):
            if not math.isclose(left, right, rel_tol=MATCH_TOLERANCE, abs_tol=1e-6):
                raise ValueError(f"hybrid input changes fixed {name} in year {index + 1}")
        replaced_years[index] = replace(
            existing, reinvestment=hybrid.total_reinvestment, fcff=hybrid.fcff
        )

    operating = MultiStageOperatingForecast(
        existing_run.operating_forecast.starting_revenue,
        tuple(replaced_years),
    )
    assumptions = existing_run.assumptions
    discounted = discount_operating_forecast(operating, assumptions)
    terminal = calculate_terminal_value(operating, discounted, assumptions)
    enterprise = aggregate_enterprise_value(discounted, terminal, assumptions)
    equity = bridge_enterprise_to_equity_value(
        enterprise, existing_run.inputs.net_debt
    )
    if existing_run.inputs.shares_outstanding is None:
        per_share = None
        per_share_reason = existing_run.per_share_unavailable_reason
    else:
        per_share = calculate_intrinsic_value_per_share(
            equity, existing_run.inputs.shares_outstanding
        )
        per_share_reason = None
    return MultiStageDCFRunResult(
        existing_run.inputs, assumptions, existing_run.forecast_path,
        operating, discounted, terminal, enterprise, equity, per_share,
        per_share_reason,
    ), hybrid_results


def compare_hybrid_reinvestment(
    existing_run: MultiStageDCFRunResult,
    hybrid_inputs: tuple[HybridReinvestmentYearInput, ...],
    *,
    classification: MethodologyClassification = "INCONCLUSIVE",
) -> HybridReinvestmentComparison:
    hybrid_run, hybrid_years = build_hybrid_shadow_dcf(
        existing_run, hybrid_inputs
    )
    existing_operating = existing_run.operating_forecast.years[:5]
    existing_discounted = existing_run.discounted_forecast.years[:5]
    hybrid_discounted = hybrid_run.discounted_forecast.years[:5]
    final = hybrid_years[-1]
    first_normalized = (
        existing_run.operating_forecast.years[5].reinvestment
        if len(existing_run.operating_forecast.years) > 5 else None
    )
    gap = final.total_reinvestment - existing_operating[-1].reinvestment
    transition_warnings = []
    denominator = max(abs(final.total_reinvestment), 1.0)
    if abs(gap) / denominator > 0.25:
        transition_warnings.append("material_final_explicit_to_sales_to_capital_gap")
    if first_normalized is not None:
        next_gap = first_normalized - final.total_reinvestment
        if abs(next_gap) / denominator > 0.50:
            transition_warnings.append("abrupt_handoff_to_normalized_sales_to_capital")
    warnings = list(transition_warnings)
    for year in hybrid_years:
        warnings.extend(year.warnings)
    transition = TransitionToSalesToCapitalDiagnostic(
        final.net_capex, final.total_reinvestment,
        existing_operating[-1].reinvestment, first_normalized, gap,
        tuple(transition_warnings),
    )
    return HybridReinvestmentComparison(
        existing_run.inputs.ticker,
        existing_run,
        hybrid_run,
        hybrid_years,
        sum(year.reinvestment for year in existing_operating),
        sum(year.total_reinvestment for year in hybrid_years),
        sum(year.fcff for year in existing_operating),
        sum(year.fcff for year in hybrid_years),
        sum(year.present_value_fcff for year in existing_discounted),
        sum(year.present_value_fcff for year in hybrid_discounted),
        transition,
        classification,
        tuple(dict.fromkeys(warnings)),
    )


def scale_capex_path(
    inputs: tuple[HybridReinvestmentYearInput, ...],
    multiplier: float,
) -> tuple[HybridReinvestmentYearInput, ...]:
    factor = _finite("multiplier", multiplier)
    if factor <= 0:
        raise ValueError("multiplier must be positive")
    return tuple(replace(item, capex=item.capex * factor) for item in inputs)


def scale_depreciation_path(
    inputs: tuple[HybridReinvestmentYearInput, ...],
    multiplier: float,
) -> tuple[HybridReinvestmentYearInput, ...]:
    factor = _finite("multiplier", multiplier)
    if factor <= 0:
        raise ValueError("multiplier must be positive")
    return tuple(replace(
        item, depreciation_amortization=item.depreciation_amortization * factor
    ) for item in inputs)


def build_intensity_based_inputs(
    existing_run: MultiStageDCFRunResult,
    *,
    capex_to_revenue: tuple[float, float, float, float, float],
    depreciation_to_revenue: tuple[float, float, float, float, float],
    working_capital_to_revenue: tuple[float, float, float, float, float] = (
        0.0, 0.0, 0.0, 0.0, 0.0
    ),
    confidence: tuple[Confidence, Confidence, Confidence, Confidence, Confidence] = (
        "High", "Medium", "Medium", "Low", "Low"
    ),
    rationale: str,
) -> tuple[HybridReinvestmentYearInput, ...]:
    """Normalize externally researched positive intensities into explicit inputs."""
    years = existing_run.operating_forecast.years[:5]
    if len(years) != 5:
        raise ValueError("existing forecast has fewer than five years")
    inputs = []
    for index, year in enumerate(years):
        capex_ratio = _finite("capex_to_revenue", capex_to_revenue[index])
        depreciation_ratio = _finite(
            "depreciation_to_revenue", depreciation_to_revenue[index]
        )
        wc_ratio = _finite(
            "working_capital_to_revenue", working_capital_to_revenue[index]
        )
        if capex_ratio < 0 or depreciation_ratio < 0:
            raise ValueError("CapEx and D&A intensities cannot be negative")
        inputs.append(HybridReinvestmentYearInput(
            year.year_index, year.revenue, year.operating_margin,
            year.operating_tax_rate, year.revenue * capex_ratio,
            year.revenue * depreciation_ratio, year.revenue * wc_ratio,
            0.0, confidence[index], rationale,
        ))
    return tuple(inputs)
