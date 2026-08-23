"""Pure Amazon consolidated-DCF structural attribution research.

The audit preserves the existing issuer-level DCF and isolates Revenue base,
margin, explicit reinvestment, segment aggregation, and terminal economics.
It performs no network access, imports no Streamlit code, creates no Company
Research Profile, and never consumes market price when constructing a model.
Advertising is an economic overlay embedded in the geographic retail segments;
Amazon does not disclose a standalone advertising operating margin, so none is
invented here.
"""

from dataclasses import dataclass, replace
import math
from typing import Literal

from Stock.forecast_methodology_audit import build_audit_candidate, spec_for_ticker
from Stock.hybrid_reinvestment_prototype import (
    HybridReinvestmentComparison,
    build_hybrid_shadow_dcf,
    compare_hybrid_reinvestment,
)
from Stock.hyperscaler_hybrid_audit import (
    build_hyperscaler_hybrid_inputs,
    hyperscaler_hybrid_research_specs,
)
from Stock.multistage_integration import (
    MultiStageDCFRunResult,
    RealCompanyDCFInputs,
    run_multistage_dcf,
)
from Stock.valuation import (
    MultiStageDCFAssumptions,
    MultiStageOperatingForecast,
    OperatingForecastYear,
    aggregate_enterprise_value,
    bridge_enterprise_to_equity_value,
    calculate_intrinsic_value_per_share,
    calculate_terminal_value,
    discount_operating_forecast,
    generate_forecast_path,
)


Severity = Literal["minor", "meaningful", "major"]


@dataclass(frozen=True)
class AmazonSegmentEvidence:
    segment: str
    period: str
    revenue: float
    operating_income: float | None
    source: str
    retrieved_at: str
    notes: str = ""

    @property
    def operating_margin(self) -> float | None:
        if self.operating_income is None:
            return None
        return self.operating_income / self.revenue


@dataclass(frozen=True)
class AmazonSegmentForecastSpec:
    segment: str
    starting_revenue: float
    growth_path: tuple[float, ...]
    margin_path: tuple[float, ...]
    rationale: str


@dataclass(frozen=True)
class AmazonSegmentForecastYear:
    year: int
    segment: str
    revenue_growth: float
    revenue: float
    operating_margin: float
    operating_income: float


@dataclass(frozen=True)
class AmazonConsolidatedSegmentYear:
    year: int
    revenue: float
    operating_income: float
    operating_margin: float
    segments: tuple[AmazonSegmentForecastYear, ...]


@dataclass(frozen=True)
class ReinvestmentPathologyYear:
    year: int
    delta_revenue: float
    sales_to_capital: float
    implied_reinvestment: float
    nopat: float
    fcff: float
    reinvestment_to_nopat: float | None
    reinvestment_to_delta_revenue: float | None
    break_even_sales_to_capital: float | None
    pathological: bool


@dataclass(frozen=True)
class AttributionModelSummary:
    model: str
    run: MultiStageDCFRunResult
    intrinsic_value_per_share: float | None
    enterprise_value: float
    equity_value: float
    explicit_fcff_pv: float
    terminal_value_pv: float
    terminal_value_share: float


@dataclass(frozen=True)
class GrowthMonotonicityPoint:
    year3_growth: float
    intrinsic_value_per_share: float | None
    explicit_fcff_pv: float
    terminal_value_pv: float
    total_reinvestment: float


@dataclass(frozen=True)
class MarginSalesToCapitalPoint:
    mature_operating_margin: float
    starting_sales_to_capital: float
    explicit_fcff_positive: bool
    intrinsic_value_per_share: float | None
    terminal_value_share: float


@dataclass(frozen=True)
class AmazonStructuralAuditResult:
    baseline: AttributionModelSummary
    revenue_base_fix: AttributionModelSummary | None
    margin_only: AttributionModelSummary
    hybrid_only: AttributionModelSummary
    margin_hybrid: AttributionModelSummary
    higher_explicit_sc: AttributionModelSummary
    segment_shadow: AttributionModelSummary
    segment_forecast: tuple[AmazonConsolidatedSegmentYear, ...]
    baseline_pathology: tuple[ReinvestmentPathologyYear, ...]
    growth_monotonicity: tuple[GrowthMonotonicityPoint, ...]
    margin_sc_grid: tuple[MarginSalesToCapitalPoint, ...]
    hybrid_comparison: HybridReinvestmentComparison
    margin_hybrid_comparison: HybridReinvestmentComparison
    severity: tuple[tuple[str, Severity], ...]
    warnings: tuple[str, ...]


AMAZON_2025_10K = "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm"
AMAZON_Q2_2026_10Q = "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000026/amzn-20260630.htm"


def amazon_segment_evidence(
    retrieved_at: str = "2026-08-23",
) -> tuple[AmazonSegmentEvidence, ...]:
    """Return fixed SEC-reported segment evidence; amounts are USD."""
    annual = (
        ("North America", "2023", 352.828e9, 14.877e9),
        ("North America", "2024", 387.497e9, 24.967e9),
        ("North America", "2025", 426.305e9, 29.619e9),
        ("International", "2023", 131.200e9, -2.656e9),
        ("International", "2024", 142.906e9, 3.792e9),
        ("International", "2025", 161.894e9, 4.750e9),
        ("AWS", "2023", 90.757e9, 24.631e9),
        ("AWS", "2024", 107.556e9, 39.834e9),
        ("AWS", "2025", 128.725e9, 45.606e9),
        ("Advertising overlay", "2023", 46.906e9, None),
        ("Advertising overlay", "2024", 56.214e9, None),
        ("Advertising overlay", "2025", 68.635e9, None),
    )
    items = [AmazonSegmentEvidence(
        segment, period, revenue, operating_income, AMAZON_2025_10K,
        retrieved_at,
        "Advertising is embedded in North America/International and is not additive."
        if segment == "Advertising overlay" else "SEC reportable segment.",
    ) for segment, period, revenue, operating_income in annual]
    items.extend((
        AmazonSegmentEvidence("North America", "2026 Q2", 116.177e9, 9.123e9, AMAZON_Q2_2026_10Q, retrieved_at),
        AmazonSegmentEvidence("International", "2026 Q2", 42.197e9, 1.717e9, AMAZON_Q2_2026_10Q, retrieved_at),
        AmazonSegmentEvidence("AWS", "2026 Q2", 42.232e9, 16.621e9, AMAZON_Q2_2026_10Q, retrieved_at),
        AmazonSegmentEvidence("Advertising overlay", "2026 Q2", 19.809e9, None, AMAZON_Q2_2026_10Q, retrieved_at, "No standalone operating margin is disclosed; Revenue is embedded in geographic segments."),
    ))
    return tuple(items)


def aggregate_segment_year(
    year: int,
    segments: tuple[AmazonSegmentForecastYear, ...],
) -> AmazonConsolidatedSegmentYear:
    if not segments or any(item.year != year for item in segments):
        raise ValueError("segment rows must be non-empty and match the year")
    revenue = sum(item.revenue for item in segments)
    operating_income = sum(item.operating_income for item in segments)
    if revenue <= 0:
        raise ValueError("consolidated segment Revenue must be positive")
    return AmazonConsolidatedSegmentYear(
        year, revenue, operating_income, operating_income / revenue, segments
    )


def build_segment_forecast(
    specs: tuple[AmazonSegmentForecastSpec, ...],
) -> tuple[AmazonConsolidatedSegmentYear, ...]:
    if not specs:
        raise ValueError("segment forecast specs are required")
    years = len(specs[0].growth_path)
    if years == 0:
        raise ValueError("segment forecast path cannot be empty")
    if any(
        len(spec.growth_path) != years or len(spec.margin_path) != years
        for spec in specs
    ):
        raise ValueError("segment paths must have equal lengths")
    prior = {spec.segment: float(spec.starting_revenue) for spec in specs}
    consolidated = []
    for year in range(1, years + 1):
        rows = []
        for spec in specs:
            revenue = prior[spec.segment] * (1 + spec.growth_path[year - 1])
            margin = spec.margin_path[year - 1]
            rows.append(AmazonSegmentForecastYear(
                year, spec.segment, spec.growth_path[year - 1], revenue,
                margin, revenue * margin,
            ))
            prior[spec.segment] = revenue
        consolidated.append(aggregate_segment_year(year, tuple(rows)))
    return tuple(consolidated)


def amazon_segment_forecast_specs(
    *,
    north_america_revenue: float = 453.670e9,
    international_revenue: float = 173.606e9,
    aws_revenue: float = 148.404e9,
) -> tuple[AmazonSegmentForecastSpec, ...]:
    """Research-only paths anchored to validated June-2026 segment TTM."""
    return (
        AmazonSegmentForecastSpec(
            "North America", north_america_revenue,
            (.14, .12, .10, .09, .08, .07, .06, .05, .04, .035, .03),
            (.075, .080, .085, .090, .095, .095, .095, .095, .090, .090, .090),
            "Retail, marketplace, subscriptions and embedded ads normalize from current operating leverage.",
        ),
        AmazonSegmentForecastSpec(
            "International", international_revenue,
            (.15, .12, .10, .09, .08, .07, .06, .05, .04, .035, .03),
            (.035, .040, .045, .050, .055, .055, .055, .055, .050, .050, .050),
            "Regional scale and fulfillment efficiency improve without assuming North America margins.",
        ),
        AmazonSegmentForecastSpec(
            "AWS", aws_revenue,
            (.30, .25, .20, .18, .16, .14, .12, .10, .08, .06, .03),
            (.365, .355, .345, .340, .335, .330, .325, .320, .320, .320, .320),
            "AI/cloud demand remains strong while infrastructure and depreciation normalize AWS margin below the current peak.",
        ),
    )


def break_even_sales_to_capital(
    delta_revenue: float,
    nopat: float,
) -> float | None:
    delta = float(delta_revenue)
    profit = float(nopat)
    if not math.isfinite(delta) or not math.isfinite(profit):
        raise ValueError("break-even inputs must be finite")
    if delta <= 0 or profit <= 0:
        return None
    return delta / profit


def reinvestment_pathology(
    run: MultiStageDCFRunResult,
) -> tuple[ReinvestmentPathologyYear, ...]:
    rows = []
    for year in run.operating_forecast.years:
        reinvestment_to_nopat = (
            None if abs(year.nopat) <= 1e-12 else year.reinvestment / year.nopat
        )
        reinvestment_to_delta = (
            None if abs(year.delta_revenue) <= 1e-12
            else year.reinvestment / year.delta_revenue
        )
        rows.append(ReinvestmentPathologyYear(
            year.year_index, year.delta_revenue, year.sales_to_capital,
            year.reinvestment, year.nopat, year.fcff,
            reinvestment_to_nopat, reinvestment_to_delta,
            break_even_sales_to_capital(year.delta_revenue, year.nopat),
            year.fcff < 0 or (
                reinvestment_to_nopat is not None and reinvestment_to_nopat > 1
            ),
        ))
    return tuple(rows)


def _revalue_operating_forecast(
    template: MultiStageDCFRunResult,
    operating: MultiStageOperatingForecast,
    assumptions: MultiStageDCFAssumptions | None = None,
) -> MultiStageDCFRunResult:
    model = assumptions or template.assumptions
    discounted = discount_operating_forecast(operating, model)
    terminal = calculate_terminal_value(operating, discounted, model)
    enterprise = aggregate_enterprise_value(discounted, terminal, model)
    equity = bridge_enterprise_to_equity_value(
        enterprise, template.inputs.net_debt
    )
    shares = template.inputs.shares_outstanding
    per_share = (
        None if shares is None
        else calculate_intrinsic_value_per_share(equity, shares)
    )
    return MultiStageDCFRunResult(
        template.inputs, model, template.forecast_path, operating, discounted,
        terminal, enterprise, equity, per_share,
        template.per_share_unavailable_reason if per_share is None else None,
    )


def build_segment_operating_shadow(
    template: MultiStageDCFRunResult,
    segment_forecast: tuple[AmazonConsolidatedSegmentYear, ...],
    assumptions: MultiStageDCFAssumptions,
) -> MultiStageDCFRunResult:
    if len(segment_forecast) != assumptions.forecast_years:
        raise ValueError("segment forecast length must equal forecast horizon")
    path = generate_forecast_path(assumptions)
    prior_revenue = template.inputs.starting_revenue
    rows = []
    for aggregate, path_year in zip(segment_forecast, path.years):
        delta = aggregate.revenue - prior_revenue
        nopat = aggregate.operating_income * (1 - assumptions.operating_tax_rate)
        reinvestment = delta / path_year.sales_to_capital
        rows.append(OperatingForecastYear(
            aggregate.year, path_year.stage, aggregate.revenue / prior_revenue - 1,
            aggregate.revenue, aggregate.operating_margin,
            aggregate.operating_income, assumptions.operating_tax_rate, nopat,
            path_year.sales_to_capital, delta, reinvestment,
            nopat - reinvestment,
        ))
        prior_revenue = aggregate.revenue
    operating = MultiStageOperatingForecast(
        template.inputs.starting_revenue, tuple(rows)
    )
    shadow = _revalue_operating_forecast(template, operating, assumptions)
    return replace(shadow, forecast_path=path)


def build_margin_only_shadow(
    template: MultiStageDCFRunResult,
    segment_forecast: tuple[AmazonConsolidatedSegmentYear, ...],
) -> MultiStageDCFRunResult:
    """Replace only annual operating margins; preserve Revenue and S/C paths."""
    if len(segment_forecast) != len(template.operating_forecast.years):
        raise ValueError("segment margin path must equal forecast horizon")
    rows = []
    for original, aggregate in zip(
        template.operating_forecast.years, segment_forecast
    ):
        operating_income = original.revenue * aggregate.operating_margin
        nopat = operating_income * (1 - original.operating_tax_rate)
        rows.append(replace(
            original,
            operating_margin=aggregate.operating_margin,
            operating_income=operating_income,
            nopat=nopat,
            fcff=nopat - original.reinvestment,
        ))
    operating = MultiStageOperatingForecast(
        template.operating_forecast.starting_revenue, tuple(rows)
    )
    assumptions = replace(
        template.assumptions,
        mature_operating_margin=segment_forecast[-1].operating_margin,
    )
    return _revalue_operating_forecast(template, operating, assumptions)


def summarize_model(
    model: str,
    run: MultiStageDCFRunResult,
) -> AttributionModelSummary:
    value = (
        None if run.per_share_value is None
        else run.per_share_value.intrinsic_value_per_share
    )
    return AttributionModelSummary(
        model, run, value, run.enterprise_value.enterprise_value,
        run.equity_value.equity_value,
        run.enterprise_value.explicit_forecast_pv,
        run.enterprise_value.terminal_value_pv,
        run.enterprise_value.terminal_value_share,
    )


def run_amazon_structural_audit(
    annual_inputs: RealCompanyDCFInputs,
    starting_operating_margin: float,
    *,
    validated_ttm_inputs: RealCompanyDCFInputs | None,
    starting_depreciation_to_revenue: float,
    segment_specs: tuple[AmazonSegmentForecastSpec, ...] | None = None,
) -> AmazonStructuralAuditResult:
    """Run isolated attribution models without market-price input."""
    if annual_inputs.ticker.upper() != "AMZN":
        raise ValueError("Amazon audit requires AMZN inputs")
    spec = spec_for_ticker("AMZN")
    baseline_assumptions = build_audit_candidate(spec, starting_operating_margin)
    baseline_run = run_multistage_dcf(annual_inputs, baseline_assumptions)
    revenue_fix_run = (
        None if validated_ttm_inputs is None
        else run_multistage_dcf(validated_ttm_inputs, baseline_assumptions)
    )

    segments = build_segment_forecast(
        segment_specs or amazon_segment_forecast_specs()
    )
    margin_run = build_margin_only_shadow(baseline_run, segments)

    hybrid_spec = next(
        item for item in hyperscaler_hybrid_research_specs()
        if item.ticker == "AMZN"
    )
    hybrid_inputs = build_hyperscaler_hybrid_inputs(
        baseline_run, hybrid_spec,
        starting_depreciation_to_revenue=starting_depreciation_to_revenue,
    )
    hybrid_comparison = compare_hybrid_reinvestment(
        baseline_run, hybrid_inputs,
        classification=hybrid_spec.classification,
    )
    margin_hybrid_inputs = build_hyperscaler_hybrid_inputs(
        margin_run, hybrid_spec,
        starting_depreciation_to_revenue=starting_depreciation_to_revenue,
    )
    margin_hybrid_comparison = compare_hybrid_reinvestment(
        margin_run, margin_hybrid_inputs,
        classification=hybrid_spec.classification,
    )

    higher_sc_assumptions = replace(
        baseline_assumptions, starting_sales_to_capital=1.50
    )
    higher_sc_run = run_multistage_dcf(annual_inputs, higher_sc_assumptions)

    segment_assumptions = replace(
        baseline_assumptions,
        starting_sales_to_capital=0.65,
        mature_sales_to_capital=0.85,
        mature_operating_margin=segments[-1].operating_margin,
    )
    segment_base_inputs = validated_ttm_inputs or annual_inputs
    segment_template = run_multistage_dcf(
        segment_base_inputs, segment_assumptions
    )
    segment_sc_run = build_segment_operating_shadow(
        segment_template, segments, segment_assumptions
    )
    segment_hybrid_inputs = build_hyperscaler_hybrid_inputs(
        segment_sc_run, hybrid_spec,
        starting_depreciation_to_revenue=starting_depreciation_to_revenue,
    )
    segment_hybrid_run, _ = build_hybrid_shadow_dcf(
        segment_sc_run, segment_hybrid_inputs
    )

    monotonicity = []
    for y3 in (.10, .12, .14):
        assumptions = replace(
            baseline_assumptions,
            near_term_revenue_growth=(.15, .14, y3),
        )
        run = run_multistage_dcf(annual_inputs, assumptions)
        monotonicity.append(GrowthMonotonicityPoint(
            y3,
            None if run.per_share_value is None else run.per_share_value.intrinsic_value_per_share,
            run.enterprise_value.explicit_forecast_pv,
            run.enterprise_value.terminal_value_pv,
            run.operating_forecast.total_reinvestment,
        ))

    grid = []
    for margin in (.10, .12, .14):
        for starting_sc in (.60, .85, 1.10):
            assumptions = replace(
                baseline_assumptions,
                starting_operating_margin=margin,
                mature_operating_margin=margin,
                starting_sales_to_capital=starting_sc,
            )
            run = run_multistage_dcf(annual_inputs, assumptions)
            grid.append(MarginSalesToCapitalPoint(
                margin, starting_sc,
                run.enterprise_value.explicit_forecast_pv >= 0,
                None if run.per_share_value is None else run.per_share_value.intrinsic_value_per_share,
                run.enterprise_value.terminal_value_share,
            ))

    severity = (
        ("revenue_base", "meaningful" if validated_ttm_inputs else "major"),
        ("operating_margin", "meaningful"),
        ("explicit_reinvestment", "major"),
        ("segment_aggregation", "major"),
        ("terminal_economics", "meaningful"),
    )
    warnings = (
        "research_only_not_a_company_profile",
        "advertising_margin_not_disclosed_or_assumed",
        "hybrid_reinvestment_remains_research_only",
        "segment_shadow_is_not_a_sum_of_segment_values",
        "market_price_excluded_from_model_construction",
    )
    return AmazonStructuralAuditResult(
        summarize_model("baseline_annual_fallback", baseline_run),
        None if revenue_fix_run is None else summarize_model("validated_ttm_revenue_base", revenue_fix_run),
        summarize_model("segment_informed_margin_only", margin_run),
        summarize_model("hybrid_reinvestment_only", hybrid_comparison.hybrid_run),
        summarize_model("margin_plus_hybrid", margin_hybrid_comparison.hybrid_run),
        summarize_model("higher_explicit_sales_to_capital", higher_sc_run),
        summarize_model("segment_informed_consolidated_shadow", segment_hybrid_run),
        segments,
        reinvestment_pathology(baseline_run),
        tuple(monotonicity), tuple(grid), hybrid_comparison,
        margin_hybrid_comparison, severity, warnings,
    )
