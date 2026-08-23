"""Dated, reproducible research specifications for the hybrid prototype.

This module is deliberately pure.  Management guidance and normalization
judgments are explicit research inputs; live accounting observations are
supplied by a caller and never fetched here.
"""

from dataclasses import dataclass

from Stock.hybrid_reinvestment_prototype import (
    HybridReinvestmentYearInput,
    MethodologyClassification,
    build_intensity_based_inputs,
)
from Stock.multistage_integration import MultiStageDCFRunResult


@dataclass(frozen=True)
class HyperscalerHybridResearchSpec:
    ticker: str
    issuer: str
    evidence_as_of: str
    year_one_capex_guidance: float
    normalized_capex_to_revenue: float
    normalized_depreciation_as_capex_share: float
    working_capital_to_revenue: tuple[float, float, float, float, float]
    confidence: tuple[str, str, str, str, str]
    classification: MethodologyClassification
    rationale: str


def hyperscaler_hybrid_research_specs(
) -> tuple[HyperscalerHybridResearchSpec, ...]:
    """Return the fixed four-company Phase 3D.1 audit universe and judgments."""
    standard_confidence = ("High", "Medium", "Medium", "Low", "Low")
    return (
        HyperscalerHybridResearchSpec(
            "GOOGL", "Alphabet", "2026-07-23", 180e9, 0.17, 0.75,
            (0.010, 0.010, 0.008, 0.006, 0.005), standard_confidence,
            "SOMEWHAT BETTER",
            "2026 CapEx guidance midpoint fades toward a still-elevated AI and "
            "cloud infrastructure intensity; working capital is kept modest.",
        ),
        HyperscalerHybridResearchSpec(
            "META", "Meta", "2026-07-29", 137.5e9, 0.20, 0.75,
            (0.010, 0.010, 0.008, 0.006, 0.005), standard_confidence,
            "SOMEWHAT BETTER",
            "2026 CapEx guidance midpoint captures the near-term AI buildout; "
            "normalization remains conservative relative to pre-AI intensity.",
        ),
        HyperscalerHybridResearchSpec(
            "MSFT", "Microsoft", "2026-04-29", 190e9, 0.20, 0.75,
            (0.005, 0.005, 0.004, 0.003, 0.003), standard_confidence,
            "SOMEWHAT BETTER",
            "Calendar-2026 CapEx commentary anchors Year 1 and then normalizes; "
            "D&A catches up with a lag because the asset mix has mixed lives.",
        ),
        HyperscalerHybridResearchSpec(
            "AMZN", "Amazon", "2026-07-30", 200e9, 0.14, 0.75,
            (0.0, 0.0, 0.0, 0.0, 0.0), standard_confidence,
            "CLEARLY BETTER REPRESENTATION",
            "Near-term infrastructure spending is modeled directly because the "
            "Sales-to-Capital method otherwise produces economically unstable "
            "reinvestment for the current Amazon forecast path.",
        ),
    )


def build_hyperscaler_hybrid_inputs(
    existing_run: MultiStageDCFRunResult,
    spec: HyperscalerHybridResearchSpec,
    *,
    starting_depreciation_to_revenue: float,
) -> tuple[HybridReinvestmentYearInput, ...]:
    """Translate one dated research spec into a five-year intensity path."""
    years = existing_run.operating_forecast.years[:5]
    if len(years) != 5:
        raise ValueError("existing forecast has fewer than five years")
    if existing_run.inputs.ticker.upper() != spec.ticker:
        raise ValueError("research spec does not match valuation ticker")
    if starting_depreciation_to_revenue < 0:
        raise ValueError("starting depreciation intensity cannot be negative")

    year_one_capex_intensity = spec.year_one_capex_guidance / years[0].revenue
    capex_step = (
        spec.normalized_capex_to_revenue - year_one_capex_intensity
    ) / 4
    capex_path = tuple(
        year_one_capex_intensity + capex_step * index for index in range(5)
    )
    target_depreciation = (
        spec.normalized_capex_to_revenue
        * spec.normalized_depreciation_as_capex_share
    )
    depreciation_step = (
        target_depreciation - starting_depreciation_to_revenue
    ) / 4
    depreciation_path = tuple(
        starting_depreciation_to_revenue + depreciation_step * index
        for index in range(5)
    )
    return build_intensity_based_inputs(
        existing_run,
        capex_to_revenue=capex_path,
        depreciation_to_revenue=depreciation_path,
        working_capital_to_revenue=spec.working_capital_to_revenue,
        confidence=spec.confidence,
        rationale=spec.rationale,
    )
