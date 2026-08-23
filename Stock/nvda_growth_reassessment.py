"""Pure NVDA growth-duration and product-cycle reassessment helpers.

The module is deliberately network-free and Streamlit-free.  It constructs a
read-only five-year research shadow without mutating the NVDA Company Profile,
its review state, any reviewed snapshot, or the applied Base DCF.
"""

from dataclasses import dataclass, replace
import math
from typing import Literal

import pandas as pd

from Stock.multistage_integration import (
    MultiStageDCFRunResult,
    RealCompanyDCFInputs,
    run_multistage_dcf,
)
from Stock.valuation import MultiStageDCFAssumptions, generate_forecast_path


Confidence = Literal["High", "Medium", "Low"]
Alignment = Literal["exact", "near_aligned", "partial_overlap", "mismatched"]
GrowthDurationDecision = Literal[
    "KEEP CURRENT",
    "MODESTLY EXTEND",
    "MATERIALLY EXTEND",
    "INSUFFICIENT EVIDENCE",
]


NVIDIA_Q1_FY27_URL = (
    "https://investor.nvidia.com/news/press-release-details/2026/"
    "NVIDIA-Announces-Financial-Results-for-First-Quarter-Fiscal-2027/"
    "default.aspx"
)
NVIDIA_RUBIN_URL = (
    "https://investor.nvidia.com/news/press-release-details/2026/"
    "NVIDIA-Kicks-Off-the-Next-Generation-of-AI-With-Rubin--Six-New-Chips-"
    "One-Incredible-AI-Supercomputer/default.aspx"
)
NVIDIA_FY26_10K_URL = (
    "https://www.sec.gov/Archives/edgar/data/1045810/"
    "000104581026000021/nvda-20260125.htm"
)


@dataclass(frozen=True)
class QuarterlyRevenuePoint:
    fiscal_quarter: str
    period_end: str
    revenue: float
    yoy_growth: float | None
    sequential_growth: float | None
    source: str


@dataclass(frozen=True)
class DataCenterRevenuePoint:
    fiscal_quarter: str
    period_end: str
    revenue: float
    yoy_growth: float | None
    sequential_growth: float | None
    share_of_total_revenue: float
    source: str


@dataclass(frozen=True)
class ConsensusRevenuePoint:
    fiscal_year: str
    period_end: str
    revenue: float
    implied_growth: float | None
    analyst_count: int | None
    source: str
    retrieved_at: str


@dataclass(frozen=True)
class ProductCyclePoint:
    platform: str
    ramp_window: str
    current_evidence: str
    revenue_relevance: str
    confidence: Confidence
    source: str


@dataclass(frozen=True)
class GrowthResearchYear:
    year_index: int
    growth: float
    confidence: Confidence
    evidence: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class RevenueRunRateDiagnostics:
    validated_ttm_revenue: float
    latest_quarter_annualized: float
    guidance_midpoint_annualized: float
    fy2027_consensus_revenue: float | None
    warning: str = "annualized_quarters_are_run_rate_diagnostics_not_forecasts"


@dataclass(frozen=True)
class DCFYearAlignment:
    dcf_year: int
    dcf_period_end: str
    fiscal_consensus_period_end: str | None
    alignment: Alignment
    note: str


@dataclass(frozen=True)
class GrowthDurationReassessment:
    current_assumptions: MultiStageDCFAssumptions
    shadow_assumptions: MultiStageDCFAssumptions
    research_path: tuple[GrowthResearchYear, ...]
    current_implied_first_five_growth: tuple[float, ...]
    quarterly_revenue: tuple[QuarterlyRevenuePoint, ...]
    data_center_revenue: tuple[DataCenterRevenuePoint, ...]
    product_cycles: tuple[ProductCyclePoint, ...]
    run_rates: RevenueRunRateDiagnostics
    alignments: tuple[DCFYearAlignment, ...]
    decision: GrowthDurationDecision
    supporting_evidence: tuple[str, ...]
    opposing_evidence: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GrowthDurationDCFComparison:
    existing: MultiStageDCFRunResult
    shadow: MultiStageDCFRunResult


def _finite_positive(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def calculate_quarterly_revenue_points(
    observations: tuple[tuple[str, str, float], ...],
    *,
    source: str,
) -> tuple[QuarterlyRevenuePoint, ...]:
    """Calculate sequential and four-quarter YoY growth in chronological order."""
    normalized = []
    for label, raw_period, raw_revenue in observations:
        period = pd.to_datetime(raw_period, errors="coerce")
        if pd.isna(period):
            raise ValueError("invalid_quarterly_period")
        normalized.append((pd.Timestamp(period), str(label), _finite_positive("revenue", raw_revenue)))
    normalized.sort(key=lambda item: item[0])
    if len({item[0] for item in normalized}) != len(normalized):
        raise ValueError("duplicate_quarterly_period")
    points = []
    for index, (period, label, revenue) in enumerate(normalized):
        sequential = None if index == 0 else revenue / normalized[index - 1][2] - 1
        yoy = None if index < 4 else revenue / normalized[index - 4][2] - 1
        points.append(QuarterlyRevenuePoint(
            label, period.date().isoformat(), revenue, yoy, sequential, source
        ))
    return tuple(points)


def _official_quarterly_revenue() -> tuple[QuarterlyRevenuePoint, ...]:
    observations = (
        ("Q2 FY2025", "2024-07-28", 30.040e9),
        ("Q3 FY2025", "2024-10-27", 35.082e9),
        ("Q4 FY2025", "2025-01-26", 39.331e9),
        ("Q1 FY2026", "2025-04-27", 44.062e9),
        ("Q2 FY2026", "2025-07-27", 46.743e9),
        ("Q3 FY2026", "2025-10-26", 57.006e9),
        ("Q4 FY2026", "2026-01-25", 68.127e9),
        ("Q1 FY2027", "2026-04-26", 81.615e9),
    )
    return calculate_quarterly_revenue_points(
        observations, source="NVIDIA quarterly earnings releases"
    )


def _official_data_center_revenue(
    total_points: tuple[QuarterlyRevenuePoint, ...],
) -> tuple[DataCenterRevenuePoint, ...]:
    values = (
        ("Q2 FY2025", "2024-07-28", 26.272e9, 1.54, 0.16),
        ("Q3 FY2025", "2024-10-27", 30.771e9, 1.12, 0.17),
        ("Q4 FY2025", "2025-01-26", 35.6e9, 0.93, 0.16),
        ("Q1 FY2026", "2025-04-27", 39.1e9, 0.73, 0.10),
        ("Q2 FY2026", "2025-07-27", 41.1e9, 0.56, 0.05),
        ("Q3 FY2026", "2025-10-26", 51.2e9, 0.66, 0.25),
        ("Q4 FY2026", "2026-01-25", 62.3e9, 0.75, 0.22),
        ("Q1 FY2027", "2026-04-26", 75.2e9, 0.92, 0.21),
    )
    total_by_label = {point.fiscal_quarter: point.revenue for point in total_points}
    return tuple(DataCenterRevenuePoint(
        label, period, revenue, yoy, sequential,
        revenue / total_by_label[label], "NVIDIA quarterly earnings releases",
    ) for label, period, revenue, yoy, sequential in values)


def nvda_product_cycle_timeline() -> tuple[ProductCyclePoint, ...]:
    return (
        ProductCyclePoint(
            "Blackwell", "FY2025–FY2027 continuing ramp",
            "Blackwell demand remained strong and Q1 FY2027 Data Center Revenue reached $75.2B.",
            "Current primary compute and rack-scale Revenue platform.", "High",
            NVIDIA_Q1_FY27_URL,
        ),
        ProductCyclePoint(
            "Blackwell Ultra", "Production shipments began in Q2 FY2026",
            "GB300 production units began shipping while Blackwell Revenue continued growing.",
            "Extends Blackwell systems and inference deployment before Rubin.", "High",
            NVIDIA_FY26_10K_URL,
        ),
        ProductCyclePoint(
            "Vera Rubin", "Production ramp; partner availability from 2H 2026",
            "Six-chip platform in production with AWS, Google Cloud, Microsoft and OCI among initial deployers.",
            "Supports a new compute, CPU, networking and storage cycle across DCF Y1–Y2.", "High",
            NVIDIA_RUBIN_URL,
        ),
        ProductCyclePoint(
            "Rubin follow-on / later architecture", "No sufficiently precise official ramp window",
            "Annual platform cadence is stated, but a finance-grade launch and Revenue ramp was not verified.",
            "Potential Y3+ support, retained as low-confidence evidence only.", "Low",
            NVIDIA_RUBIN_URL,
        ),
    )


def build_run_rate_diagnostics(
    *,
    ttm_revenue: float,
    latest_quarter_revenue: float = 81.615e9,
    guidance_midpoint: float = 91.0e9,
    fy2027_consensus_revenue: float | None,
) -> RevenueRunRateDiagnostics:
    consensus = None
    if fy2027_consensus_revenue is not None:
        consensus = _finite_positive("fy2027_consensus_revenue", fy2027_consensus_revenue)
    return RevenueRunRateDiagnostics(
        _finite_positive("ttm_revenue", ttm_revenue),
        _finite_positive("latest_quarter_revenue", latest_quarter_revenue) * 4,
        _finite_positive("guidance_midpoint", guidance_midpoint) * 4,
        consensus,
    )


def _alignments(
    ttm_period_end: str,
    consensus: tuple[ConsensusRevenuePoint, ...],
) -> tuple[DCFYearAlignment, ...]:
    start = pd.to_datetime(ttm_period_end, errors="coerce")
    if pd.isna(start):
        raise ValueError("invalid_ttm_period_end")
    by_year = {item.fiscal_year: item for item in consensus}
    results = []
    for year_index, fiscal_year in ((1, "FY2027"), (2, "FY2028"), (3, "FY2029")):
        dcf_end = pd.Timestamp(start) + pd.DateOffset(years=year_index)
        point = by_year.get(fiscal_year)
        if point is None:
            results.append(DCFYearAlignment(
                year_index, dcf_end.date().isoformat(), None, "mismatched",
                f"{fiscal_year} consensus unavailable",
            ))
            continue
        fiscal_end = pd.to_datetime(point.period_end, errors="coerce")
        if pd.isna(fiscal_end):
            alignment: Alignment = "mismatched"
            note = "consensus fiscal period is invalid"
        else:
            day_gap = abs((dcf_end - pd.Timestamp(fiscal_end)).days)
            alignment = "exact" if day_gap <= 7 else "near_aligned" if day_gap <= 120 else "partial_overlap" if day_gap <= 275 else "mismatched"
            note = f"DCF year ends {day_gap} days from fiscal consensus period"
        results.append(DCFYearAlignment(
            year_index, dcf_end.date().isoformat(), point.period_end,
            alignment, note,
        ))
    return tuple(results)


def build_nvda_growth_reassessment(
    current_assumptions: MultiStageDCFAssumptions,
    *,
    ttm_revenue: float,
    ttm_period_end: str,
    consensus: tuple[ConsensusRevenuePoint, ...] = (),
) -> GrowthDurationReassessment:
    """Build a slower-normalization shadow while leaving the candidate intact."""
    if current_assumptions.near_term_revenue_growth != (0.55, 0.40, 0.25):
        raise ValueError("unexpected_nvda_candidate_growth_baseline")
    if current_assumptions.forecast_years != 12 or current_assumptions.revenue_fade_years != 9:
        raise ValueError("unexpected_nvda_candidate_horizon_baseline")

    # This is an evidence stress path, not a stored profile update.  The total
    # 12-year transition horizon stays fixed: five explicit years plus seven
    # fade years replace three explicit years plus nine fade years.
    research_years = (
        GrowthResearchYear(1, 0.55, "High", ("FY2027 consensus", "Q2 FY2027 guidance"), "TTM-to-Y1 Revenue is already close to live FY2027 consensus despite the three-month period offset."),
        GrowthResearchYear(2, 0.40, "High", ("FY2028 consensus", "Rubin deployment"), "FY2028 consensus and Rubin deployment support another high-growth year, with a larger-base step-down."),
        GrowthResearchYear(3, 0.30, "Medium", ("annual product cadence", "hyperscaler infrastructure commitments"), "Tests whether overlapping platform ramps defer normalization; no FY2029 consensus validates this rate."),
        GrowthResearchYear(4, 0.25, "Low", ("inference", "networking", "sovereign AI"), "Research-only duration extension supported qualitatively, not by a precise fiscal consensus."),
        GrowthResearchYear(5, 0.20, "Low", ("larger installed base", "competition and digestion risk"), "Maintains elevated growth but recognizes scale, custom silicon, competition and CapEx digestion."),
    )
    shadow = replace(
        current_assumptions,
        near_term_revenue_growth=tuple(item.growth for item in research_years),
        revenue_fade_years=7,
    )
    current_path = generate_forecast_path(current_assumptions)
    run_rates = build_run_rate_diagnostics(
        ttm_revenue=ttm_revenue,
        fy2027_consensus_revenue=next(
            (item.revenue for item in consensus if item.fiscal_year == "FY2027"),
            None,
        ),
    )
    quarterly = _official_quarterly_revenue()
    return GrowthDurationReassessment(
        current_assumptions=current_assumptions,
        shadow_assumptions=shadow,
        research_path=research_years,
        current_implied_first_five_growth=current_path.revenue_growth_path[:5],
        quarterly_revenue=quarterly,
        data_center_revenue=_official_data_center_revenue(quarterly),
        product_cycles=nvda_product_cycle_timeline(),
        run_rates=run_rates,
        alignments=_alignments(ttm_period_end, consensus),
        decision="INSUFFICIENT EVIDENCE",
        supporting_evidence=(
            "Blackwell, Blackwell Ultra and Rubin create overlapping platform ramps.",
            "Q1 FY2027 Data Center Revenue grew 92% YoY and networking grew 199% YoY.",
            "Rubin deployments and hyperscaler infrastructure plans extend into 2H 2026 and beyond.",
            "Inference, networking and sovereign infrastructure broaden demand beyond training accelerators.",
        ),
        opposing_evidence=(
            "FY2029 consensus is unavailable, leaving Y3–Y5 weakly anchored.",
            "The existing nine-year fade already implies approximately 22.6% and 20.2% growth in Y4 and Y5.",
            "Customer concentration, custom silicon, AMD, export controls and CapEx digestion remain material.",
            "Both paths imply Revenue near or above $1T by Year 5, requiring unusually large end-market expansion.",
        ),
        warnings=(
            "research_shadow_not_a_company_profile_candidate",
            "no_review_or_apply_action",
            "market_price_not_used",
            "fy2029_consensus_unavailable",
        ),
    )


def compare_growth_duration_dcf(
    inputs: RealCompanyDCFInputs,
    reassessment: GrowthDurationReassessment,
) -> GrowthDurationDCFComparison:
    """Run identical economics with Revenue growth duration as the only factor."""
    current = reassessment.current_assumptions
    shadow = reassessment.shadow_assumptions
    fixed_fields = (
        "forecast_years", "terminal_growth", "starting_operating_margin",
        "mature_operating_margin", "starting_sales_to_capital",
        "mature_sales_to_capital", "operating_tax_rate", "wacc",
    )
    if any(getattr(current, field) != getattr(shadow, field) for field in fixed_fields):
        raise ValueError("shadow_changes_non_growth_economics")
    return GrowthDurationDCFComparison(
        run_multistage_dcf(inputs, current),
        run_multistage_dcf(inputs, shadow),
    )


def forecast_revenue_levels(
    starting_revenue: float,
    growth_rates: tuple[float, ...],
) -> tuple[float, ...]:
    revenue = _finite_positive("starting_revenue", starting_revenue)
    levels = []
    for growth in growth_rates:
        numeric_growth = float(growth)
        if not math.isfinite(numeric_growth) or numeric_growth <= -1:
            raise ValueError("growth must be finite and greater than -100%")
        revenue *= 1 + numeric_growth
        levels.append(revenue)
    return tuple(levels)
