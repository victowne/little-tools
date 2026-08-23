"""Pure Phase 3F.2 evidence validation for Amazon economic buckets."""

from dataclasses import dataclass, replace
import math
from typing import Literal

from Stock.amazon_mature_economics_audit import (
    EconomicBucket,
    MatureEconomicsScenario,
    MatureEconomicsValuation,
    ReverseMarginBridge,
    amazon_mature_scenarios,
    build_mature_scenario,
)
from Stock.forecast_methodology_audit import build_audit_candidate, spec_for_ticker
from Stock.hybrid_reinvestment_prototype import compare_hybrid_reinvestment
from Stock.hyperscaler_hybrid_audit import (
    build_hyperscaler_hybrid_inputs,
    hyperscaler_hybrid_research_specs,
)
from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf


EvidenceTier = Literal[1, 2, 3]
EvidenceQuality = Literal[
    "Direct disclosure", "Derived from Amazon disclosure",
    "Comparable-supported", "Weak inference",
]
ValidationDecision = Literal["retained", "revised_modestly", "revised_materially"]


@dataclass(frozen=True)
class BucketEvidenceItem:
    evidence_id: str
    bucket: str
    metric: str
    value: float | None
    unit: str
    period: str
    source: str
    retrieved_at: str
    tier: EvidenceTier
    quality: EvidenceQuality
    notes: str


@dataclass(frozen=True)
class ValidatedBucketRange:
    bucket: str
    margin_low: float
    margin_central: float
    margin_high: float
    sales_to_capital_low: float
    sales_to_capital_central: float
    sales_to_capital_high: float
    confidence: str
    margin_decision: ValidationDecision
    sales_to_capital_decision: ValidationDecision
    rationale: str


@dataclass(frozen=True)
class CapitalAllocation:
    bucket: str
    revenue_share: float
    operating_income_share: float
    incremental_capital_share: float


@dataclass(frozen=True)
class AssumptionChange:
    assumption: str
    previous: float
    validated: float
    direction: str
    reason: str
    confidence: str


AMZN_10K = "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm"
AMZN_10Q = "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000026/amzn-20260630.htm"
WMT_10K = "https://www.sec.gov/Archives/edgar/data/104169/000010416926000055/wmt-20260131.htm"
EBAY_10K = "https://www.sec.gov/Archives/edgar/data/1065088/000106508826000027/ebay-20251231.htm"
META_10K = "https://www.sec.gov/Archives/edgar/data/1326801/000162828026003942/meta-20251231.htm"
GOOG_10K = "https://www.sec.gov/Archives/edgar/data/1652044/000165204426000018/goog-20251231.htm"


def validation_evidence(
    retrieved_at: str = "2026-08-23",
) -> tuple[BucketEvidenceItem, ...]:
    """Fixed evidence registry. Comparable facts remain contextual only."""
    return (
        BucketEvidenceItem("amzn_q2_na_margin", "first_party_retail", "reported_NA_margin", .0785267, "ratio", "2026 Q2", AMZN_10Q, retrieved_at, 1, "Direct disclosure", "NA also contains marketplace, ads, and subscriptions; not a first-party margin."),
        BucketEvidenceItem("amzn_q2_intl_margin", "first_party_retail", "reported_International_margin", .0406901, "ratio", "2026 Q2", AMZN_10Q, retrieved_at, 1, "Direct disclosure", "International also contains marketplace, ads, and subscriptions."),
        BucketEvidenceItem("wmt_margin", "first_party_retail", "comparable_operating_margin", .042, "ratio", "FY2026", WMT_10K, retrieved_at, 2, "Comparable-supported", "Scaled omnichannel retail anchor; Amazon business mix differs."),
        BucketEvidenceItem("ebay_margin", "marketplace", "comparable_operating_margin", .205, "ratio", "FY2025", EBAY_10K, retrieved_at, 2, "Comparable-supported", "Asset-light marketplace with much less owned fulfillment infrastructure."),
        BucketEvidenceItem("amzn_3p_revenue", "marketplace", "revenue", 183.660e9, "USD", "TTM 2026-06-30", AMZN_10K + " + " + AMZN_10Q, retrieved_at, 1, "Derived from Amazon disclosure", "FY2025 + H1 2026 - H1 2025; includes commissions, fulfillment, and shipping fees."),
        BucketEvidenceItem("meta_foa_margin", "advertising", "comparable_operating_margin", 102.469 / 198.759, "ratio", "FY2025", META_10K, retrieved_at, 2, "Comparable-supported", "FoA is predominantly advertising but is not a pure advertising segment."),
        BucketEvidenceItem("goog_services_margin", "advertising", "comparable_operating_margin", 139.404 / (139.404 + 203.317), "ratio", "FY2025", GOOG_10K, retrieved_at, 2, "Comparable-supported", "Services includes ads, subscriptions, platforms, and devices; shared AI R&D is outside the segment."),
        BucketEvidenceItem("amzn_ads_revenue", "advertising", "revenue", 76.072e9, "USD", "TTM 2026-06-30", AMZN_10K + " + " + AMZN_10Q, retrieved_at, 1, "Derived from Amazon disclosure", "No standalone operating income is disclosed."),
        BucketEvidenceItem("amzn_sub_revenue", "subscriptions", "revenue", 52.853e9, "USD", "TTM 2026-06-30", AMZN_10K + " + " + AMZN_10Q, retrieved_at, 1, "Derived from Amazon disclosure", "Includes Prime and content services with bundled shipping/content costs."),
        BucketEvidenceItem("aws_2023_margin", "aws", "operating_margin", .271395, "ratio", "FY2023", AMZN_10K, retrieved_at, 1, "Direct disclosure", "Reported AWS segment."),
        BucketEvidenceItem("aws_2024_margin", "aws", "operating_margin", .370356, "ratio", "FY2024", AMZN_10K, retrieved_at, 1, "Direct disclosure", "Reported AWS segment."),
        BucketEvidenceItem("aws_2025_margin", "aws", "operating_margin", .354290, "ratio", "FY2025", AMZN_10K, retrieved_at, 1, "Direct disclosure", "Reported AWS segment."),
        BucketEvidenceItem("aws_q2_margin", "aws", "operating_margin", .393564, "ratio", "2026 Q2", AMZN_10Q, retrieved_at, 1, "Direct disclosure", "Quarterly peak is not assumed mature."),
        BucketEvidenceItem("aws_ppe", "aws", "net_PPE", 263.750e9, "USD", "2026-06-30", AMZN_10Q, retrieved_at, 1, "Direct disclosure", "AWS segment PP&E."),
        BucketEvidenceItem("aws_h1_additions", "aws", "net_PPE_additions", 90.120e9, "USD", "H1 2026", AMZN_10Q, retrieved_at, 1, "Direct disclosure", "Includes finance leases and non-cash additions; current buildout creates lead-lag."),
        BucketEvidenceItem("aws_h1_da", "aws", "depreciation_amortization", 15.353e9, "USD", "H1 2026", AMZN_10Q, retrieved_at, 1, "Direct disclosure", "Usage-allocated D&A."),
        BucketEvidenceItem("goog_shared_ai", "shared", "comparable_shared_AI_cost", 16.760e9, "USD", "FY2025", GOOG_10K, retrieved_at, 2, "Comparable-supported", "Shows centralized AI R&D can be material; not an Amazon amount."),
    )


def validated_bucket_ranges() -> tuple[ValidatedBucketRange, ...]:
    """Ranges revised only where evidence narrows Phase 3F.1 uncertainty."""
    return (
        ValidatedBucketRange("first_party_retail", .03, .045, .06, .60, .75, .90, "Medium", "revised_modestly", "revised_modestly", "Walmart and Amazon geographic margins support mid-single digits; 7% remains possible but not central/high validated."),
        ValidatedBucketRange("marketplace", .12, .16, .19, .90, 1.30, 1.70, "Low", "revised_modestly", "revised_modestly", "eBay supports platform profitability, while FBA/logistics makes Amazon materially more capital intensive."),
        ValidatedBucketRange("advertising", .35, .43, .52, 3.0, 4.5, 6.0, "Low", "revised_modestly", "revised_modestly", "Meta/Google support high margins, but Amazon shares traffic, compute, sales, and R&D with commerce."),
        ValidatedBucketRange("subscriptions", .06, .10, .14, 1.2, 2.0, 2.8, "Low", "revised_modestly", "revised_modestly", "Prime bundles shipping and content, preventing software-like economics."),
        ValidatedBucketRange("aws", .28, .33, .37, .30, .45, .65, "Medium", "retained", "revised_modestly", "33% is supported by multi-year margins; massive current PP&E additions argue against high mature capital turnover."),
        ValidatedBucketRange("other", .00, .02, .04, .50, .70, 1.00, "Low", "retained", "retained", "Residual bucket remains intentionally conservative."),
    )


def validated_mature_scenarios() -> tuple[MatureEconomicsScenario, ...]:
    ranges = {item.bucket: item for item in validated_bucket_ranges()}
    definitions = (
        ("validated_conservative", .020, {"first_party_retail": .34, "marketplace": .24, "advertising": .11, "subscriptions": .07, "aws": .22, "other": .02}, "low"),
        ("validated_central", .015, {"first_party_retail": .28, "marketplace": .25, "advertising": .13, "subscriptions": .07, "aws": .25, "other": .02}, "central"),
        ("validated_high", .010, {"first_party_retail": .24, "marketplace": .25, "advertising": .14, "subscriptions": .07, "aws": .28, "other": .02}, "high"),
    )
    output = []
    for name, shared, shares, level in definitions:
        rows = []
        for bucket, share in shares.items():
            evidence = ranges[bucket]
            margin = getattr(evidence, f"margin_{level}")
            sc = getattr(evidence, f"sales_to_capital_{level}")
            intensity = "Very High" if bucket == "aws" else "High" if bucket == "first_party_retail" else "Medium" if bucket in {"marketplace", "other"} else "Low"
            rows.append(EconomicBucket(bucket, share, margin, sc, intensity, evidence.confidence, evidence.rationale))
        output.append(build_mature_scenario(name, tuple(rows), shared_cost_adjustment=shared))
    return tuple(output)


def capital_and_profit_allocation(
    scenario: MatureEconomicsScenario,
) -> tuple[CapitalAllocation, ...]:
    capital = tuple(item.revenue_share / item.sales_to_capital for item in scenario.buckets)
    capital_total = sum(capital)
    margin = scenario.consolidated_margin
    return tuple(CapitalAllocation(
        item.bucket, item.revenue_share,
        item.operating_income_share_of_revenue / margin,
        required / capital_total,
    ) for item, required in zip(scenario.buckets, capital))


def shared_cost_sensitivity(
    scenario: MatureEconomicsScenario,
    deductions: tuple[float, ...] = (.01, .015, .02, .025),
) -> tuple[MatureEconomicsScenario, ...]:
    return tuple(build_mature_scenario(
        f"shared_cost_{deduction:.3f}", scenario.buckets,
        shared_cost_adjustment=deduction,
    ) for deduction in deductions)


def phase3f1_change_attribution() -> tuple[AssumptionChange, ...]:
    old = amazon_mature_scenarios()[1]
    new = validated_mature_scenarios()[1]
    return (
        AssumptionChange("mature_margin", old.consolidated_margin, new.consolidated_margin, "down", "Comparable and shared-cost evidence narrows high hidden-bucket margins.", "Low"),
        AssumptionChange("mature_sales_to_capital", old.consolidated_sales_to_capital, new.consolidated_sales_to_capital, "down", "AWS PP&E build and FBA logistics lower validated capital efficiency.", "Low"),
        AssumptionChange("terminal_roic", old.terminal_roic, new.terminal_roic, "down", "Mechanical consequence of validated margin and S/C.", "Low"),
    )


def aws_capital_diagnostics() -> tuple[tuple[str, float], ...]:
    revenue_ttm = 148.404e9
    ppe = 263.750e9
    return (
        ("TTM_Revenue_to_net_PPE", revenue_ttm / ppe),
        ("H1_net_additions_to_H1_Revenue", 90.120 / 79.819),
        ("H1_DA_to_H1_Revenue", 15.353 / 79.819),
    )


def validated_reverse_thirty_percent() -> ReverseMarginBridge:
    """Re-test 30% using validated high bucket margins and a 1% shared cost."""
    marketplace, ads, subscriptions, other = .25, .13, .07, .02
    available = 1 - marketplace - ads - subscriptions - other
    aws_margin, retail_margin, shared = .37, .06, .01
    fixed = marketplace * .19 + ads * .52 + subscriptions * .14 + other * .04
    required_aws = (.30 + shared - fixed - available * retail_margin) / (
        aws_margin - retail_margin
    )
    return ReverseMarginBridge(
        .30, required_aws, available - required_aws, aws_margin,
        marketplace + ads + subscriptions + other, shared, "aggressive",
    )


def run_validated_mature_valuations(
    inputs: RealCompanyDCFInputs,
    *,
    starting_operating_margin: float,
    starting_depreciation_to_revenue: float,
) -> tuple[MatureEconomicsValuation, ...]:
    """Run fixed old/validated economics without price or formula duplication."""
    base = build_audit_candidate(spec_for_ticker("AMZN"), starting_operating_margin)
    hybrid_spec = next(x for x in hyperscaler_hybrid_research_specs() if x.ticker == "AMZN")
    central = validated_mature_scenarios()[1]
    reverse = validated_reverse_thirty_percent()
    thirty = replace(
        central, name="validated_thirty_percent_diagnostic",
        consolidated_margin=.30,
        terminal_roic=.30 * .79 * central.consolidated_sales_to_capital,
        terminal_reinvestment_rate=.03 / (.30 * .79 * central.consolidated_sales_to_capital),
        terminal_fcff_to_nopat=1 - .03 / (.30 * .79 * central.consolidated_sales_to_capital),
        warnings=central.warnings + (f"required_aws_share_{reverse.required_aws_revenue_share:.4f}",),
    )
    scenarios = (amazon_mature_scenarios()[1],) + validated_mature_scenarios() + (thirty,)
    output = []
    for scenario in scenarios:
        assumptions = replace(base, mature_operating_margin=scenario.consolidated_margin, mature_sales_to_capital=scenario.consolidated_sales_to_capital)
        standard = run_multistage_dcf(inputs, assumptions)
        hybrid_inputs = build_hyperscaler_hybrid_inputs(standard, hybrid_spec, starting_depreciation_to_revenue=starting_depreciation_to_revenue)
        run = compare_hybrid_reinvestment(standard, hybrid_inputs, classification=hybrid_spec.classification).hybrid_run
        output.append(MatureEconomicsValuation(scenario.name, scenario, run))
    return tuple(output)
