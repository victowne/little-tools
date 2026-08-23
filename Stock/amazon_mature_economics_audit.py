"""Pure Amazon mature-economics bridge; research-only and price-independent."""

from dataclasses import dataclass, replace
import math
from typing import Literal

from Stock.forecast_methodology_audit import build_audit_candidate, spec_for_ticker
from Stock.amazon_structural_dcf_audit import run_amazon_structural_audit
from Stock.hyperscaler_hybrid_audit import (
    build_hyperscaler_hybrid_inputs,
    hyperscaler_hybrid_research_specs,
)
from Stock.hybrid_reinvestment_prototype import compare_hybrid_reinvestment
from Stock.multistage_integration import (
    MultiStageDCFRunResult,
    RealCompanyDCFInputs,
    run_multistage_dcf,
)


Confidence = Literal["High", "Medium", "Low"]
CapitalIntensity = Literal["Very Low", "Low", "Medium", "High", "Very High"]


@dataclass(frozen=True)
class EconomicBucket:
    bucket: str
    revenue_share: float
    operating_margin: float
    sales_to_capital: float
    capital_intensity: CapitalIntensity
    confidence: Confidence
    rationale: str

    @property
    def operating_income_share_of_revenue(self) -> float:
        return self.revenue_share * self.operating_margin


@dataclass(frozen=True)
class EconomicMixEvidence:
    period: str
    total_revenue: float
    bucket_revenue: tuple[tuple[str, float], ...]
    source: str
    retrieved_at: str
    notes: str

    @property
    def shares(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (name, value / self.total_revenue)
            for name, value in self.bucket_revenue
        )


@dataclass(frozen=True)
class MatureEconomicsScenario:
    name: str
    buckets: tuple[EconomicBucket, ...]
    shared_cost_adjustment: float
    consolidated_margin: float
    consolidated_sales_to_capital: float
    terminal_roic: float
    terminal_reinvestment_rate: float
    terminal_fcff_to_nopat: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class EconomicsMatrixPoint:
    operating_margin: float
    sales_to_capital: float
    terminal_roic: float
    terminal_reinvestment_rate: float
    terminal_fcff_to_nopat: float
    roic_vs_wacc: str


@dataclass(frozen=True)
class ReverseMarginBridge:
    target_margin: float
    required_aws_revenue_share: float
    resulting_first_party_share: float
    aws_margin: float
    other_platform_share: float
    shared_cost_adjustment: float
    assessment: str


@dataclass(frozen=True)
class ProfitPoolPoint:
    bucket: str
    revenue_share: float
    operating_income_contribution: float
    share_of_consolidated_operating_income: float


@dataclass(frozen=True)
class MatureEconomicsValuation:
    case: str
    scenario: MatureEconomicsScenario
    run: MultiStageDCFRunResult


@dataclass(frozen=True)
class EconomicGrowthYear:
    year: int
    consolidated_growth: float
    total_revenue: float
    bucket_mix: tuple[tuple[str, float], ...]


AMAZON_2025_10K = "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm"
AMAZON_Q2_2026_10Q = "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000026/amzn-20260630.htm"


def amazon_economic_mix_evidence(
    retrieved_at: str = "2026-08-23",
) -> tuple[EconomicMixEvidence, ...]:
    """Return mutually exclusive SEC product/service categories in USD."""
    annual = (
        ("2023", (251.902, 140.053, 46.906, 40.209, 90.757, 4.958)),
        ("2024", (268.244, 156.146, 56.214, 44.374, 107.556, 5.425)),
        ("2025", (291.848, 172.162, 68.635, 49.619, 128.725, 5.935)),
    )
    names = (
        "first_party_retail", "marketplace", "advertising",
        "subscriptions", "aws", "other",
    )
    rows = [EconomicMixEvidence(
        period, sum(values) * 1e9,
        tuple((name, value * 1e9) for name, value in zip(names, values)),
        AMAZON_2025_10K, retrieved_at,
        "First-party combines Online and Physical Stores; all buckets are exclusive.",
    ) for period, values in annual]
    ttm = (308.093, 183.660, 76.072, 52.853, 148.404, 6.598)
    q2 = (76.226, 46.780, 19.809, 13.730, 42.232, 1.829)
    rows.extend((
        EconomicMixEvidence(
            "TTM 2026-06-30", sum(ttm) * 1e9,
            tuple((name, value * 1e9) for name, value in zip(names, ttm)),
            AMAZON_2025_10K + " + " + AMAZON_Q2_2026_10Q,
            retrieved_at,
            "FY2025 + H1 2026 - H1 2025; validated total is 775.680B.",
        ),
        EconomicMixEvidence(
            "2026 Q2", sum(q2) * 1e9,
            tuple((name, value * 1e9) for name, value in zip(names, q2)),
            AMAZON_Q2_2026_10Q, retrieved_at,
            "First-party combines Online and Physical Stores; no category overlaps.",
        ),
    ))
    return tuple(rows)


def _validate_unit_interval(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    return numeric


def build_mature_scenario(
    name: str,
    buckets: tuple[EconomicBucket, ...],
    *,
    shared_cost_adjustment: float,
    operating_tax_rate: float = 0.21,
    terminal_growth: float = 0.03,
) -> MatureEconomicsScenario:
    """Aggregate margin arithmetically and S/C as a weighted harmonic mean.

    For one dollar of incremental issuer Revenue, bucket ``i`` requires
    ``share_i / S-C_i`` incremental capital. Therefore consolidated S/C is
    ``1 / sum(share_i / S-C_i)``, not the arithmetic average of bucket S/C.
    """
    if not buckets:
        raise ValueError("economic buckets are required")
    total_share = sum(item.revenue_share for item in buckets)
    if not math.isclose(total_share, 1.0, abs_tol=1e-9):
        raise ValueError("economic bucket shares must sum to 100%")
    if any(item.sales_to_capital <= 0 for item in buckets):
        raise ValueError("bucket Sales-to-Capital must be positive")
    shared = _validate_unit_interval(shared_cost_adjustment, "shared cost")
    tax = _validate_unit_interval(operating_tax_rate, "operating tax rate")
    margin = sum(item.operating_income_share_of_revenue for item in buckets) - shared
    sales_to_capital = 1 / sum(
        item.revenue_share / item.sales_to_capital for item in buckets
    )
    roic = margin * (1 - tax) * sales_to_capital
    reinvestment = math.inf if abs(roic) <= 1e-12 else terminal_growth / roic
    warnings = []
    if roic <= 0:
        warnings.append("non_positive_terminal_roic")
    if reinvestment > 1:
        warnings.append("terminal_reinvestment_exceeds_nopat")
    if margin >= .25 and sales_to_capital < .70:
        warnings.append("high_margin_with_low_capital_efficiency")
    return MatureEconomicsScenario(
        name, buckets, shared, margin, sales_to_capital, roic, reinvestment,
        1 - reinvestment, tuple(warnings),
    )


def amazon_mature_scenarios() -> tuple[MatureEconomicsScenario, ...]:
    """Evidence-constrained low/central/high economic-bucket bridges."""
    definitions = (
        ("conservative", .020, (
            ("first_party_retail", .34, .03, .60, "High", "Medium"),
            ("marketplace", .24, .12, 1.00, "Medium", "Low"),
            ("advertising", .11, .35, 3.00, "Low", "Low"),
            ("subscriptions", .07, .08, 1.50, "Low", "Low"),
            ("aws", .22, .28, .35, "Very High", "Medium"),
            ("other", .02, .00, .50, "Medium", "Low"),
        )),
        ("central", .015, (
            ("first_party_retail", .28, .05, .80, "High", "Medium"),
            ("marketplace", .25, .16, 1.40, "Medium", "Low"),
            ("advertising", .13, .45, 5.00, "Low", "Low"),
            ("subscriptions", .07, .12, 2.00, "Low", "Low"),
            ("aws", .25, .33, .50, "Very High", "Medium"),
            ("other", .02, .03, .70, "Medium", "Low"),
        )),
        ("high_platform_cloud", .015, (
            ("first_party_retail", .22, .07, 1.00, "High", "Low"),
            ("marketplace", .25, .20, 1.80, "Medium", "Low"),
            ("advertising", .15, .55, 7.00, "Very Low", "Low"),
            ("subscriptions", .07, .16, 3.00, "Low", "Low"),
            ("aws", .29, .38, .70, "High", "Low"),
            ("other", .02, .05, 1.00, "Medium", "Low"),
        )),
    )
    return tuple(build_mature_scenario(
        name,
        tuple(EconomicBucket(*row, "Research range; not a disclosed bucket margin or S/C.") for row in rows),
        shared_cost_adjustment=shared,
    ) for name, shared, rows in definitions)


def economics_matrix(
    margins: tuple[float, ...],
    sales_to_capital_values: tuple[float, ...],
    *,
    tax_rate: float = .21,
    terminal_growth: float = .03,
    wacc: float = .105,
) -> tuple[EconomicsMatrixPoint, ...]:
    points = []
    for margin in margins:
        for sales_to_capital in sales_to_capital_values:
            roic = margin * (1 - tax_rate) * sales_to_capital
            reinvestment = terminal_growth / roic
            spread = roic - wacc
            classification = (
                "below" if spread < -.01 else
                "near" if abs(spread) <= .01 else
                "modestly_above" if spread <= .05 else "materially_above"
            )
            points.append(EconomicsMatrixPoint(
                margin, sales_to_capital, roic, reinvestment,
                1 - reinvestment, classification,
            ))
    return tuple(points)


def reverse_bridge_for_margin(target_margin: float = .30) -> ReverseMarginBridge:
    """Solve AWS share needed for target margin under already aggressive margins.

    Marketplace/Ads/Subscription/Other shares remain at the central mix; AWS
    takes share only from first-party retail. This is intentionally a reverse
    economics diagnostic, never an assumption generator.
    """
    target = float(target_margin)
    marketplace, ads, subscriptions, other = .25, .13, .07, .02
    available = 1 - marketplace - ads - subscriptions - other
    aws_margin, retail_margin, shared = .38, .07, .015
    fixed = marketplace * .20 + ads * .55 + subscriptions * .16 + other * .05
    required_aws = (target + shared - fixed - available * retail_margin) / (
        aws_margin - retail_margin
    )
    return ReverseMarginBridge(
        target, required_aws, available - required_aws, aws_margin,
        marketplace + ads + subscriptions + other, shared,
        "aggressive" if 0 <= required_aws <= available else "implausible",
    )


def profit_pool(scenario: MatureEconomicsScenario) -> tuple[ProfitPoolPoint, ...]:
    if abs(scenario.consolidated_margin) <= 1e-12:
        raise ValueError("profit-pool denominator is zero")
    return tuple(ProfitPoolPoint(
        item.bucket, item.revenue_share,
        item.operating_income_share_of_revenue,
        item.operating_income_share_of_revenue / scenario.consolidated_margin,
    ) for item in scenario.buckets)


def bucket_margin_sensitivity(
    scenario: MatureEconomicsScenario,
    bucket: str,
    margins: tuple[float, ...],
) -> tuple[MatureEconomicsScenario, ...]:
    if bucket not in {item.bucket for item in scenario.buckets}:
        raise ValueError("unknown economic bucket")
    output = []
    for margin in margins:
        rows = tuple(
            replace(item, operating_margin=float(margin))
            if item.bucket == bucket else item
            for item in scenario.buckets
        )
        output.append(build_mature_scenario(
            f"{scenario.name}_{bucket}_{margin:.4f}", rows,
            shared_cost_adjustment=scenario.shared_cost_adjustment,
        ))
    return tuple(output)


def aws_mix_sensitivity(
    scenario: MatureEconomicsScenario,
    aws_shares: tuple[float, ...],
) -> tuple[MatureEconomicsScenario, ...]:
    """Move Revenue share between AWS and first-party retail only."""
    current_aws = next(x for x in scenario.buckets if x.bucket == "aws")
    current_retail = next(
        x for x in scenario.buckets if x.bucket == "first_party_retail"
    )
    available = current_aws.revenue_share + current_retail.revenue_share
    output = []
    for share in aws_shares:
        if not 0 <= share <= available:
            raise ValueError("AWS share leaves an invalid first-party share")
        rows = tuple(
            replace(item, revenue_share=share)
            if item.bucket == "aws" else
            replace(item, revenue_share=available - share)
            if item.bucket == "first_party_retail" else item
            for item in scenario.buckets
        )
        output.append(build_mature_scenario(
            f"{scenario.name}_aws_mix_{share:.4f}", rows,
            shared_cost_adjustment=scenario.shared_cost_adjustment,
        ))
    return tuple(output)


def segment_summed_growth_diagnostic() -> tuple[EconomicGrowthYear, ...]:
    """Five-year research path from mutually exclusive June-2026 TTM buckets."""
    ttm = dict(amazon_economic_mix_evidence()[-2].bucket_revenue)
    growth = {
        "first_party_retail": (.10, .09, .08, .07, .06),
        "marketplace": (.16, .14, .12, .10, .08),
        "advertising": (.22, .19, .17, .15, .13),
        "subscriptions": (.12, .10, .09, .08, .07),
        "aws": (.30, .25, .20, .18, .16),
        "other": (.10, .09, .08, .07, .06),
    }
    prior = dict(ttm)
    prior_total = sum(prior.values())
    rows = []
    for year in range(1, 6):
        current = {
            name: value * (1 + growth[name][year - 1])
            for name, value in prior.items()
        }
        total = sum(current.values())
        rows.append(EconomicGrowthYear(
            year, total / prior_total - 1, total,
            tuple((name, value / total) for name, value in current.items()),
        ))
        prior, prior_total = current, total
    return tuple(rows)


def required_aws_share_for_profit_pool(
    target_profit_share: float,
    *,
    aws_margin: float = .33,
) -> float:
    """AWS Revenue share required for a target consolidated OI contribution."""
    target = _validate_unit_interval(target_profit_share, "profit share")
    central = amazon_mature_scenarios()[1]
    non_aws = tuple(item for item in central.buckets if item.bucket != "aws")
    original_non_aws_share = sum(item.revenue_share for item in non_aws)
    normalized_non_aws_margin = sum(
        item.revenue_share / original_non_aws_share * item.operating_margin
        for item in non_aws
    )
    numerator = target * (
        normalized_non_aws_margin - central.shared_cost_adjustment
    )
    denominator = aws_margin * (1 - target) + target * normalized_non_aws_margin
    return numerator / denominator


def run_mature_economics_valuations(
    inputs: RealCompanyDCFInputs,
    *,
    starting_operating_margin: float,
    starting_depreciation_to_revenue: float,
) -> tuple[MatureEconomicsValuation, ...]:
    """Value fixed economics only after pure scenarios have been constructed."""
    base = build_audit_candidate(spec_for_ticker("AMZN"), starting_operating_margin)
    hybrid_spec = next(
        item for item in hyperscaler_hybrid_research_specs() if item.ticker == "AMZN"
    )
    phase3f = build_mature_scenario(
        "phase3f_bridge",
        amazon_mature_scenarios()[1].buckets,
        shared_cost_adjustment=amazon_mature_scenarios()[1].shared_cost_adjustment,
    )
    phase3f = replace(
        phase3f, consolidated_margin=.15683546885666055,
        consolidated_sales_to_capital=.85,
        terminal_roic=.15683546885666055 * .79 * .85,
        terminal_reinvestment_rate=.03 / (.15683546885666055 * .79 * .85),
        terminal_fcff_to_nopat=1 - .03 / (.15683546885666055 * .79 * .85),
    )
    scenarios = (phase3f,) + amazon_mature_scenarios()
    reverse = reverse_bridge_for_margin()
    central = amazon_mature_scenarios()[1]
    thirty = replace(
        central, name="thirty_percent_margin_diagnostic",
        consolidated_margin=.30,
        terminal_roic=.30 * .79 * central.consolidated_sales_to_capital,
        terminal_reinvestment_rate=.03 / (.30 * .79 * central.consolidated_sales_to_capital),
        terminal_fcff_to_nopat=1 - .03 / (.30 * .79 * central.consolidated_sales_to_capital),
        warnings=central.warnings + (f"reverse_required_aws_share_{reverse.required_aws_revenue_share:.4f}",),
    )
    structural = run_amazon_structural_audit(
        inputs, starting_operating_margin,
        validated_ttm_inputs=None,
        starting_depreciation_to_revenue=starting_depreciation_to_revenue,
    )
    output = [MatureEconomicsValuation(
        phase3f.name, phase3f, structural.margin_hybrid.run
    )]
    for scenario in scenarios[1:] + (thirty,):
        assumptions = replace(
            base,
            mature_operating_margin=scenario.consolidated_margin,
            mature_sales_to_capital=scenario.consolidated_sales_to_capital,
        )
        standard = run_multistage_dcf(inputs, assumptions)
        hybrid_inputs = build_hyperscaler_hybrid_inputs(
            standard, hybrid_spec,
            starting_depreciation_to_revenue=starting_depreciation_to_revenue,
        )
        comparison = compare_hybrid_reinvestment(
            standard, hybrid_inputs, classification=hybrid_spec.classification
        )
        output.append(MatureEconomicsValuation(
            scenario.name, scenario, comparison.hybrid_run
        ))
    return tuple(output)
