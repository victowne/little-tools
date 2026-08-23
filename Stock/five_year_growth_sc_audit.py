"""Pure Phase 3D.2 five-year growth and mature S/C methodology audit.

The module is deterministic and deliberately has no market-price, network,
Streamlit, Company Profile, or session-state dependency.  It reuses the
existing multi-stage DCF orchestration and changes only the two dimensions
under audit: explicit Revenue growth duration and mature Sales-to-Capital.
"""

from dataclasses import dataclass, replace
import math
from typing import Literal

from Stock.multistage_integration import (
    MultiStageDCFRunResult,
    RealCompanyDCFInputs,
    run_multistage_dcf,
)
from Stock.valuation import MultiStageDCFAssumptions, generate_forecast_path


Confidence = Literal["High", "Medium", "Low"]


@dataclass(frozen=True)
class ExplicitGrowthResearchPoint:
    year: int
    growth: float
    confidence: Confidence
    evidence: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class HyperscalerGrowthSCAuditSpec:
    ticker: str
    issuer: str
    explicit_growth: tuple[ExplicitGrowthResearchPoint, ...]
    mature_sales_to_capital_values: tuple[float, ...]
    research_mature_sales_to_capital: float | None
    mature_sales_to_capital_rationale: str
    explicit_period_classification: str
    mature_sales_to_capital_classification: str


@dataclass(frozen=True)
class ValuationModelSummary:
    model: str
    assumptions: MultiStageDCFAssumptions
    run: MultiStageDCFRunResult
    intrinsic_value_per_share: float | None
    enterprise_value: float
    equity_value: float
    explicit_fcff_pv: float
    terminal_value_pv: float
    terminal_value_share: float | None


@dataclass(frozen=True)
class MatureSCSensitivityPoint:
    mature_sales_to_capital: float
    terminal_roic: float
    terminal_reinvestment_rate: float | None
    terminal_fcff_to_nopat: float | None
    intrinsic_value_per_share: float | None
    terminal_value_share: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FiveYearGrowthSCAuditResult:
    ticker: str
    research_growth: tuple[ExplicitGrowthResearchPoint, ...]
    baseline_growth_path: tuple[float, ...]
    baseline: ValuationModelSummary
    growth_only: ValuationModelSummary
    mature_sc_only: ValuationModelSummary | None
    combined: ValuationModelSummary | None
    mature_sc_sensitivity: tuple[MatureSCSensitivityPoint, ...]


def hyperscaler_growth_sc_specs() -> tuple[HyperscalerGrowthSCAuditSpec, ...]:
    """Return four pre-valuation research specifications for the audit."""
    return (
        HyperscalerGrowthSCAuditSpec(
            "GOOGL", "Alphabet",
            (
                ExplicitGrowthResearchPoint(1, .23, "High", ("current_ttm", "fy1_consensus", "q2_growth"), "Current consolidated and forward Revenue evidence."),
                ExplicitGrowthResearchPoint(2, .20, "High", ("fy2_consensus", "cloud_backlog"), "Cloud backlog and capacity-limited monetization support durability."),
                ExplicitGrowthResearchPoint(3, .20, "Medium", ("user_requested_growth_duration_test", "cloud_growth", "ai_monetization"), "User-specified research test above currently available aligned consensus."),
                ExplicitGrowthResearchPoint(4, .18, "Low", ("user_requested_growth_duration_test", "cloud_backlog"), "User-specified research assumption; no analyst-consensus claim."),
                ExplicitGrowthResearchPoint(5, .16, "Low", ("user_requested_growth_duration_test", "mixed_business_normalization"), "User-specified research assumption; no analyst-consensus claim."),
            ),
            (.60, .70, .75, .80, .90), .75,
            "Mixed Search/YouTube platform economics and normalized Cloud utilization support a mature S/C above the buildout trough, but below a pure software platform.",
            "mildly inadequate", "possibly too low",
        ),
        HyperscalerGrowthSCAuditSpec(
            "META", "Meta",
            (
                ExplicitGrowthResearchPoint(1, .24, "High", ("current_ttm", "fy1_consensus", "ad_volume_and_price"), "Current advertising and consolidated evidence."),
                ExplicitGrowthResearchPoint(2, .20, "High", ("fy2_consensus", "ai_recommendations"), "Recommendation quality and monetization remain the operating anchors."),
                ExplicitGrowthResearchPoint(3, .20, "Medium", ("user_requested_growth_duration_test", "messaging_monetization", "ai_buildout"), "User-specified research test with infrastructure-cost uncertainty."),
                ExplicitGrowthResearchPoint(4, .18, "Low", ("user_requested_growth_duration_test", "ad_platform_normalization"), "User-specified research assumption, not consensus."),
                ExplicitGrowthResearchPoint(5, .16, "Low", ("user_requested_growth_duration_test", "mature_ad_growth"), "User-specified research assumption, not consensus."),
            ),
            (.55, .66, .70, .80, .90), .80,
            "Advertising-platform economics can regain capital efficiency after the AI buildout, while Reality Labs and continuing infrastructure prevent pure-platform S/C.",
            "mildly inadequate", "likely too low",
        ),
        HyperscalerGrowthSCAuditSpec(
            "MSFT", "Microsoft",
            (
                ExplicitGrowthResearchPoint(1, .18, "High", ("current_ttm", "fy1_consensus", "azure_growth"), "Consolidated growth is anchored below Azure growth."),
                ExplicitGrowthResearchPoint(2, .19, "High", ("user_requested_growth_duration_test", "fy2_consensus", "commercial_rpo"), "User-specified research test supported by RPO and supply constraints."),
                ExplicitGrowthResearchPoint(3, .20, "Medium", ("user_requested_growth_duration_test", "m365", "copilot", "security"), "User-specified acceleration test across cloud and recurring software."),
                ExplicitGrowthResearchPoint(4, .19, "Low", ("user_requested_growth_duration_test", "capacity_monetization"), "User-specified research assumption after currently visible consensus."),
                ExplicitGrowthResearchPoint(5, .18, "Low", ("user_requested_growth_duration_test", "mixed_cloud_software_normalization"), "User-specified research assumption, not Azure growth extrapolation."),
            ),
            (.50, .60, .75, .90, 1.05), .75,
            "A mature mix of Azure infrastructure and high-margin recurring software should not automatically retain buildout-era consolidated S/C near 0.5x.",
            "mildly inadequate", "likely too low",
        ),
        HyperscalerGrowthSCAuditSpec(
            "AMZN", "Amazon",
            (
                ExplicitGrowthResearchPoint(1, .15, "High", ("current_ttm", "fy1_consensus", "retail_aws_ads"), "Consolidated anchor, not AWS growth copied to the group."),
                ExplicitGrowthResearchPoint(2, .18, "Medium", ("user_requested_growth_duration_test", "fy2_consensus", "aws_demand"), "User-specified consolidated growth-duration test."),
                ExplicitGrowthResearchPoint(3, .20, "Medium", ("user_requested_growth_duration_test", "ai_infrastructure_monetization"), "User-specified acceleration test with segment-mix uncertainty."),
                ExplicitGrowthResearchPoint(4, .18, "Low", ("user_requested_growth_duration_test", "retail_and_cloud_normalization"), "User-specified research assumption, not consensus."),
                ExplicitGrowthResearchPoint(5, .16, "Low", ("user_requested_growth_duration_test", "mature_mixed_business_growth"), "User-specified research assumption for the consolidated issuer."),
            ),
            (.60, .70, .85, 1.00, 1.15), .85,
            "AWS and advertising lift mature efficiency, while fulfillment and logistics justify a lower range than platform-heavy peers.",
            "mildly inadequate", "uncertain",
        ),
    )


def build_five_year_growth_assumptions(
    baseline: MultiStageDCFAssumptions,
    research_growth: tuple[ExplicitGrowthResearchPoint, ...],
) -> MultiStageDCFAssumptions:
    """Replace Y1-Y5 and consume two fade years while preserving the horizon.

    The baseline mature-state year count is preserved where possible.  A
    baseline with three explicit years therefore converts from ``3 + F`` to
    ``5 + max(F-2, 0)`` without lengthening the forecast horizon.
    """
    if len(research_growth) != 5:
        raise ValueError("research growth path must contain exactly five years")
    if tuple(point.year for point in research_growth) != (1, 2, 3, 4, 5):
        raise ValueError("research growth years must be consecutive from one")
    values = tuple(float(point.growth) for point in research_growth)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("research growth values must be finite")
    fade_years = max(0, baseline.revenue_fade_years - 2)
    if baseline.forecast_years < 5 + fade_years:
        raise ValueError("forecast horizon cannot contain five explicit years")
    return replace(
        baseline,
        near_term_revenue_growth=values,
        revenue_fade_years=fade_years,
    )


def build_mature_sc_assumptions(
    baseline: MultiStageDCFAssumptions,
    mature_sales_to_capital: float,
) -> MultiStageDCFAssumptions:
    value = float(mature_sales_to_capital)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("mature Sales-to-Capital must be finite and positive")
    return replace(baseline, mature_sales_to_capital=value)


def _summarize(model: str, run: MultiStageDCFRunResult) -> ValuationModelSummary:
    per_share = (
        None if run.per_share_value is None
        else run.per_share_value.intrinsic_value_per_share
    )
    return ValuationModelSummary(
        model, run.assumptions, run, per_share,
        run.enterprise_value.enterprise_value,
        run.equity_value.equity_value,
        run.enterprise_value.explicit_forecast_pv,
        run.enterprise_value.terminal_value_pv,
        run.enterprise_value.terminal_value_share,
    )


def build_mature_sc_sensitivity(
    inputs: RealCompanyDCFInputs,
    baseline: MultiStageDCFAssumptions,
    values: tuple[float, ...],
) -> tuple[MatureSCSensitivityPoint, ...]:
    if len(values) < 5:
        raise ValueError("mature S/C grid requires at least five points")
    if len(set(values)) != len(values):
        raise ValueError("mature S/C grid values must be distinct")
    points = []
    for value in values:
        assumptions = build_mature_sc_assumptions(baseline, value)
        run = run_multistage_dcf(inputs, assumptions)
        reinvestment = assumptions.terminal_reinvestment_rate
        warnings = list(assumptions.validation_warnings)
        if reinvestment is not None and not 0 <= reinvestment <= 1:
            warnings.append("implausible_terminal_reinvestment_rate")
        points.append(MatureSCSensitivityPoint(
            value,
            assumptions.derived_terminal_roic,
            reinvestment,
            None if reinvestment is None else 1 - reinvestment,
            None if run.per_share_value is None else run.per_share_value.intrinsic_value_per_share,
            run.enterprise_value.terminal_value_share,
            tuple(dict.fromkeys(warnings)),
        ))
    return tuple(points)


def run_five_year_growth_sc_audit(
    inputs: RealCompanyDCFInputs,
    baseline: MultiStageDCFAssumptions,
    spec: HyperscalerGrowthSCAuditSpec,
) -> FiveYearGrowthSCAuditResult:
    if inputs.ticker.upper() != spec.ticker:
        raise ValueError("audit spec does not match valuation ticker")
    growth_assumptions = build_five_year_growth_assumptions(
        baseline, spec.explicit_growth
    )
    baseline_run = run_multistage_dcf(inputs, baseline)
    growth_run = run_multistage_dcf(inputs, growth_assumptions)
    baseline_path = generate_forecast_path(baseline).revenue_growth_path
    sensitivity = build_mature_sc_sensitivity(
        inputs, baseline, spec.mature_sales_to_capital_values
    )
    if spec.research_mature_sales_to_capital is None:
        sc_only = combined = None
    else:
        sc_assumptions = build_mature_sc_assumptions(
            baseline, spec.research_mature_sales_to_capital
        )
        combined_assumptions = build_mature_sc_assumptions(
            growth_assumptions, spec.research_mature_sales_to_capital
        )
        sc_only = _summarize(
            "mature_sc_only", run_multistage_dcf(inputs, sc_assumptions)
        )
        combined = _summarize(
            "combined", run_multistage_dcf(inputs, combined_assumptions)
        )
    return FiveYearGrowthSCAuditResult(
        inputs.ticker,
        spec.explicit_growth,
        baseline_path,
        _summarize("baseline", baseline_run),
        _summarize("five_year_growth_only", growth_run),
        sc_only,
        combined,
        sensitivity,
    )
