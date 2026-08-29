"""Pure, read-only Amazon unified-production Research Profile Candidate.

The profile packages the conclusions of Phase 3F.2/3F.3.  It does not fetch
data, inspect market price, review/apply state, or duplicate valuation formulas.
Prior Hybrid work remains evidence about model limitations, not production logic.
"""

from dataclasses import dataclass, replace
import math

import pandas as pd

from Stock.alphabet_research import ResearchRange, RevenueEvidenceRow, _annual_items, _latest_annual, _ttm
from Stock.company_profiles import (
    BusinessContext,
    CapitalEfficiencyResearchFramework,
    CompanyProfileLookupResult,
    CompanyResearchProfile,
    MarginResearchFramework,
    ResearchAssumption,
    ResearchEvidenceItem,
    RevenueResearchFramework,
    TerminalResearchFramework,
    WACCResearchFramework,
    build_multistage_assumptions_from_profile,
)
from Stock.fundamentals import GROSS_MARGIN, OPERATING_MARGIN, REVENUE, REVENUE_GROWTH, ROIC, FundamentalHistory
from Stock.hyperscaler_research import ConfidenceAssessment
from Stock.multistage_integration import MultiStageDCFRunResult, RealCompanyDCFInputs, run_multistage_dcf
from Stock.valuation import MultiStageDCFAssumptions
from Stock.wacc_audit import WACCAuditResult


VALIDATED_TTM_REVENUE = 775.680e9
VALIDATED_TTM_OPERATING_MARGIN = 0.11155296795755199
VALIDATED_TTM_PERIODS = tuple(pd.Timestamp(value) for value in (
    "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30",
))
STARTING_PPE_DEPRECIATION = 49.741e9
MATURE_MARGIN_RANGE = (0.1233, 0.2389)
MATURE_SALES_TO_CAPITAL_RANGE = (0.588, 1.095)
FROZEN_MATURE_MARGIN = 0.1834
FROZEN_MATURE_SALES_TO_CAPITAL = 0.824
FROZEN_TAX_RATE = 0.21
FROZEN_WACC = 0.105
FROZEN_TERMINAL_GROWTH = 0.03
AMAZON_2025_10K = "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm"
AMAZON_2026_Q2_10Q = "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000026/amzn-20260630.htm"


@dataclass(frozen=True)
class AmazonResearchProfileResult:
    lookup: CompanyProfileLookupResult
    revenue_evidence: tuple[RevenueEvidenceRow, ...]
    growth_ranges: tuple[ResearchRange, ...]
    current_assumptions: MultiStageDCFAssumptions
    period_reconciliation: tuple[str, ...]
    confidence_assessments: tuple[ConfidenceAssessment, ...]
    candidate_preview: MultiStageDCFRunResult | None = None
    reviewed: bool = False
    applied: bool = False
    warnings: tuple[str, ...] = ()


def _assumption(name, value, rationale, references):
    return ResearchAssumption(name, value, "research_in_progress", rationale, references)


def _evidence(evidence_id, label, value, unit, period, source, source_date, retrieved_at, *, category="company_specific_research", notes=""):
    return ResearchEvidenceItem(
        evidence_id, category, label, value, unit, period, source,
        source_date, retrieved_at, notes=notes,
    )


def _validated_inputs(inputs: RealCompanyDCFInputs) -> RealCompanyDCFInputs:
    if inputs.ticker.strip().upper() != "AMZN":
        raise ValueError("amazon_candidate_requires_amzn_inputs")
    return replace(
        inputs,
        starting_revenue=VALIDATED_TTM_REVENUE,
        starting_revenue_source="SEC validated TTM: FY2025 + H1 2026 - H1 2025",
        starting_revenue_periods=VALIDATED_TTM_PERIODS,
    )


def run_amazon_candidate_preview(
    inputs: RealCompanyDCFInputs,
    profile: CompanyResearchProfile,
) -> MultiStageDCFRunResult:
    """Run Amazon through the same standard S/C engine as every company."""
    if profile.issuer_id != "AMZN":
        raise ValueError("amazon_profile_required")
    translation = build_multistage_assumptions_from_profile(profile)
    if not translation.available or translation.assumptions is None:
        raise ValueError(translation.reason or "amazon_candidate_translation_unavailable")
    return run_multistage_dcf(_validated_inputs(inputs), translation.assumptions)


def build_amazon_research_profile(
    current_assumptions: MultiStageDCFAssumptions,
    history: FundamentalHistory,
    *,
    wacc_audit: WACCAuditResult | None = None,
    retrieved_at: str = "2026-08-23",
) -> AmazonResearchProfileResult:
    """Build an unreviewed, unapplied candidate from frozen research evidence."""
    annual_revenue = _latest_annual(history, REVENUE, "latest_annual_revenue", "Latest annual Revenue", "currency_amount")
    annual_growth = _latest_annual(history, REVENUE_GROWTH, "latest_annual_growth", "Latest annual Revenue growth", "ratio")
    annual_margin = _latest_annual(history, OPERATING_MARGIN, "latest_annual_operating_margin", "Latest annual Operating Margin", "ratio")
    accounting_roic = _latest_annual(history, ROIC, "accounting_roic", "Accounting ROIC", "ratio")
    sec_source = AMAZON_2025_10K + " + " + AMAZON_2026_Q2_10Q
    starting_margin = VALIDATED_TTM_OPERATING_MARGIN
    ttm_margin = _evidence(
        "ttm_operating_margin", "Validated SEC TTM consolidated Operating Margin",
        starting_margin, "ratio", "TTM ended 2026-06-30", sec_source,
        "2026-07-31", retrieved_at, category="historical_financial",
        notes="Four-quarter SEC bridge; Yahoo annual fallback is not used.",
    )
    ttm_revenue = _evidence(
        "validated_sec_ttm_revenue", "Validated SEC TTM Revenue",
        VALIDATED_TTM_REVENUE, "currency_amount", "TTM ended 2026-06-30",
        sec_source, "2026-07-31", retrieved_at,
        category="historical_financial",
        notes="FY2025 + H1 2026 - H1 2025; annual fallback is prohibited.",
    )
    external = (
        _evidence("near_term_growth", "Top-down Revenue growth research", "15% / 14% / 12%", None, "DCF Y1-Y3", "Phase 3F.2 segment-summed diagnostic", "2026-08-23", retrieved_at, notes="Modestly conservative versus the 16.56% / 14.68% / 12.75% segment-summed diagnostic."),
        _evidence("business_mix", "Amazon economic-bucket business mix", "Retail, Marketplace, Advertising, Subscription and AWS", None, "mature framework", "Phase 3F.2 economic-bucket validation", "2026-08-23", retrieved_at),
        _evidence("starting_margin_evidence", "Validated consolidated Operating Margin", starting_margin, "ratio", ttm_margin.period, ttm_margin.source, "2026-07-31", retrieved_at, category="historical_financial", notes="SEC TTM, not an assumption selected to improve valuation."),
        _evidence("mature_margin_bridge", "Economic-bucket Mature Operating Margin", FROZEN_MATURE_MARGIN, "ratio", "mature period", "Phase 3F.2 economic-bucket validation", "2026-08-23", retrieved_at, notes="Research range 12.33%-23.89%; not directly disclosed by Amazon."),
        _evidence("capital_intensity", "Current consolidated capital-intensity distortion", "AWS/AI infrastructure buildout makes current consolidated S/C unsuitable for explicit-period reinvestment.", None, "current buildout", sec_source, "2026-07-31", retrieved_at),
        _evidence("mature_sc_bridge", "Capital-demand-weighted Mature Sales-to-Capital", FROZEN_MATURE_SALES_TO_CAPITAL, "multiple", "mature period", "Phase 3F.2 economic-bucket validation", "2026-08-23", retrieved_at, notes="Range 0.588x-1.095x; 1 / Sum(RevenueShare_i / S-C_i), not a simple average; lower than prior 0.894x."),
        _evidence("explicit_transition", "H1 Central explicit CapEx / PP&E D&A transition", "24.5% / 22.0% / 19.5% / 17.0% / 15.0%", None, "DCF Y1-Y5", "Phase 3F.3 explicit-transition validation", "2026-08-23", retrieved_at, notes="Economic CapEx is total net PP&E additions; cash purchases are reconciliation only and financed equipment is not double counted."),
        _evidence("hybrid_handoff", "Hybrid to Sales-to-Capital handoff", "3 years", None, "DCF Y6-Y8", "Phase 3F.3 handoff validation", "2026-08-23", retrieved_at, notes="H1 reinvestment is replaced, not added, then blended to standard S/C."),
        _evidence("research_wacc", "Amazon Research WACC", FROZEN_WACC, "ratio", "long horizon", "Phase 2 research WACC decision", "2026-08-23", retrieved_at, category="market_risk", notes="Separate from Formula WACC and beta evidence."),
        _evidence("terminal_growth", "Mature nominal growth", FROZEN_TERMINAL_GROWTH, "ratio", "terminal period", "Research framework", "2026-08-23", retrieved_at, category="industry_reference"),
        _evidence("confidence_summary", "Research evidence confidence", "Revenue Base High; Y1 High; Y2 Medium; Y3 Low; Mature Margin Low; Mature S/C Low; Explicit CapEx/D&A/Handoff Medium; Research WACC Medium; Terminal Economics Low", None, "evidence as of 2026-08-23", "Phase 3F.2/3F.3 validation", "2026-08-23", retrieved_at),
    )
    terminal_roic = FROZEN_MATURE_MARGIN * (1 - FROZEN_TAX_RATE) * FROZEN_MATURE_SALES_TO_CAPITAL
    terminal_reinvestment = FROZEN_TERMINAL_GROWTH / terminal_roic
    terminal_economics = _evidence(
        "terminal_economics", "Mechanically derived terminal economics",
        f"ROIC {terminal_roic:.6%}; reinvestment {terminal_reinvestment:.6%}; FCFF/NOPAT {1-terminal_reinvestment:.6%}",
        None, "terminal period", "Candidate assumptions (mechanical derivation)", "2026-08-23", retrieved_at,
    )
    evidence = tuple(item for item in (annual_revenue, annual_growth, annual_margin, accounting_roic) if item is not None) + (ttm_revenue,) + external + (terminal_economics,)

    y1 = _assumption("year1_growth", 0.15, "Top-down path is modestly conservative versus segment-summed evidence; market price is excluded.", ("validated_sec_ttm_revenue", "near_term_growth", "business_mix"))
    y2 = _assumption("year2_growth", 0.14, "Growth normalizes while AWS, advertising and marketplace mix continue to support expansion.", ("near_term_growth", "business_mix"))
    y3 = _assumption("year3_growth", 0.12, "Third-year research carries lower confidence and precedes deterministic fade.", ("near_term_growth", "business_mix"))
    fade = _assumption("revenue_fade_years", 8, "Eight deterministic fade years follow the three researched years; Y4/Y5 are not production inputs.", ("near_term_growth", "terminal_growth"))
    terminal_g = _assumption("terminal_growth", FROZEN_TERMINAL_GROWTH, "Long-run nominal-growth anchor; not adjusted to market price.", ("terminal_growth",))
    start_margin = _assumption("starting_operating_margin", starting_margin, "Exact validated TTM consolidated margin with source period preserved.", ("starting_margin_evidence",))
    mature_margin = _assumption("mature_operating_margin", FROZEN_MATURE_MARGIN, "Capital- and profit-pool-weighted bucket bridge; not a direct disclosure.", ("mature_margin_bridge", "business_mix"))
    start_sc = _assumption("starting_sales_to_capital", 0.57, "Research-normalized starting S/C for the unified production path; current accounting efficiency is distorted by the infrastructure buildout.", ("capital_intensity", "explicit_transition"))
    mature_sc = _assumption("mature_sales_to_capital", FROZEN_MATURE_SALES_TO_CAPITAL, "Capital-demand-weighted bucket anchor, not a simple average.", ("mature_sc_bridge", "capital_intensity"))
    tax = _assumption("operating_tax_rate", FROZEN_TAX_RATE, "Frozen normalized operating tax rate.", ("terminal_economics",))
    wacc = _assumption("research_wacc", FROZEN_WACC, "Research WACC remains separate from formula WACC/beta evidence.", ("research_wacc",))
    horizon = _assumption("forecast_years", 11, "Three explicit research years plus eight fade years.", ("near_term_growth", "terminal_growth"))
    context = BusinessContext(
        business_model_summary="Amazon combines capital-intensive retail/logistics with higher-margin Marketplace, Advertising, Subscription and AWS economics.",
        primary_growth_drivers=("AWS", "Advertising", "Marketplace", "Prime/Subscription", "Retail volume"),
        primary_margin_drivers=("business mix", "AWS utilization", "fulfillment efficiency", "advertising scale", "PP&E depreciation"),
        capital_intensity_notes=("AWS/AI capacity buildout distorts consolidated explicit-period Sales-to-Capital.", "Unified production methodology intentionally uses standard ΔRevenue/S-C despite this timing limitation."),
        major_profile_risks=("Advertising standalone margin is not reported", "Marketplace standalone margin is not reported", "economic-bucket capital allocation is inferred", "utilization is proxied", "Hybrid reinvestment remains research methodology", "market-value gap remains large"),
    )
    profile = CompanyResearchProfile(
        ticker="AMZN", issuer_id="AMZN", company_name="Amazon.com, Inc.", profile_status="research_in_progress",
        business_summary=context.business_model_summary, business_context=context,
        revenue_framework=RevenueResearchFramework(
            starting_revenue=ttm_revenue, latest_annual_revenue=annual_revenue,
            ttm_revenue=ttm_revenue, latest_annual_growth=annual_growth,
            year1_growth=y1, year2_growth=y2, year3_growth=y3,
            revenue_fade_years=fade, terminal_growth=terminal_g,
            near_term_growth_rationale="Only Y1-Y3 are researched explicitly; later years use deterministic fade.",
            fade_rationale=fade.rationale, terminal_growth_rationale=terminal_g.rationale,
            warnings=("validated_sec_ttm_required",),
        ),
        margin_framework=MarginResearchFramework(
            annual_margin, ttm_margin,
            _annual_items(history, OPERATING_MARGIN, prefix="operating_margin", label="Operating Margin", unit="ratio"),
            _annual_items(history, GROSS_MARGIN, prefix="gross_margin", label="Gross Margin", unit="ratio"),
            start_margin, mature_margin, start_margin.rationale, mature_margin.rationale,
        ),
        capital_efficiency_framework=CapitalEfficiencyResearchFramework(
            None, None, accounting_roic, None, None, start_sc, mature_sc,
            starting_margin * (1 - FROZEN_TAX_RATE) * 0.57, terminal_roic,
            start_sc.rationale, mature_sc.rationale,
            ("unified_standard_sc_may_be_overly_punitive_during_buildout", "hybrid_research_is_audit_evidence_only"),
        ),
        wacc_framework=WACCResearchFramework(wacc_audit=wacc_audit, research_wacc=wacc, rationale=wacc.rationale, warnings=("formula_wacc_separate_from_research_wacc",)),
        terminal_framework=TerminalResearchFramework(
            terminal_g, mature_margin, mature_sc, terminal_roic,
            terminal_reinvestment, 1 - terminal_reinvestment,
            terminal_g.rationale, mature_margin.rationale, mature_sc.rationale,
            ("mature_margin_bridge", "mature_sc_bridge", "terminal_economics"),
            ("terminal_economics_low_confidence",),
        ),
        operating_tax_rate=tax, forecast_years=horizon,
        rationale="Read-only Amazon candidate using the unified standard S/C production architecture; prior Hybrid work is limitation evidence and market price is excluded.",
        warnings=("research_candidate_not_reviewed", "candidate_not_applied_to_live_dcf", "unified_standard_sc_model_risk_high"),
        last_reviewed_at=None, evidence_items=evidence,
        uncertainty_notes=context.major_profile_risks + (
            "Unified production methodology selected for cross-company consistency; prior Hybrid research indicates explicit-period capital timing may not be fully captured.",
            "Candidate remains conditional on all research assumptions.",
        ),
        future_scenario_drivers=("Revenue growth duration", "mature margin", "mature Sales-to-Capital", "Research WACC"),
        reinvestment_strategy=None,
        model_risk="High",
    )
    growth_ranges = (
        ResearchRange("year1_growth", 0.13, 0.15, 0.17, y1.rationale, y1.evidence_references),
        ResearchRange("year2_growth", 0.12, 0.14, 0.16, y2.rationale, y2.evidence_references),
        ResearchRange("year3_growth", 0.10, 0.12, 0.14, y3.rationale, y3.evidence_references),
    )
    revenue_evidence = (
        RevenueEvidenceRow("FY2025 annual (context only)", "FY ended 2025-12-31", 716.924e9, None, AMAZON_2025_10K, "2026-02-06", retrieved_at, notes="Not used as the DCF starting base."),
        RevenueEvidenceRow("Validated SEC TTM (DCF base)", "TTM ended 2026-06-30", VALIDATED_TTM_REVENUE, None, sec_source, "2026-07-31", retrieved_at, notes="FY2025 + H1 2026 - H1 2025; stale annual fallback prohibited."),
    )
    confidence = (
        ConfidenceAssessment("Revenue Base", "High", "Direct SEC-period arithmetic with validated TTM semantics."),
        ConfidenceAssessment("Y1 Growth", "High", "Near-term top-down path is supported by segment-summed evidence."),
        ConfidenceAssessment("Y2 Growth", "Medium", "Mix and demand support remain visible but timing is less certain."),
        ConfidenceAssessment("Y3 Growth", "Low", "Longer-duration segment outcomes are less observable."),
        ConfidenceAssessment("Mature Margin", "Low", "Medium-Low central judgment, but existing vocabulary records Low because standalone bucket margins are inferred."),
        ConfidenceAssessment("Mature S/C", "Low", "Capital-demand weights and bucket efficiencies are inferred."),
        ConfidenceAssessment("AWS Margin Evidence", "Medium", "AWS segment operating evidence is disclosed."),
        ConfidenceAssessment("AWS S/C", "Medium", "Capital intensity is bounded by disclosed infrastructure evidence but allocated."),
        ConfidenceAssessment("Advertising Margin", "Low", "Standalone margin is not reported."),
        ConfidenceAssessment("Marketplace Margin", "Low", "Standalone margin is not reported."),
        ConfidenceAssessment("Explicit CapEx Transition", "Medium", "SEC additions and guidance anchor the path; normalization timing remains judgmental."),
        ConfidenceAssessment("D&A Transition", "Medium", "PP&E depreciation and useful lives are disclosed; cohort mix is modeled."),
        ConfidenceAssessment("Handoff Methodology", "Medium", "Three-year handoff passed deterministic continuity diagnostics."),
        ConfidenceAssessment("Research WACC", "Medium", "Research judgment remains separate from formula WACC."),
        ConfidenceAssessment("Terminal Economics", "Low", "Long-run margin and capital efficiency remain inferred."),
    )
    return AmazonResearchProfileResult(
        CompanyProfileLookupResult(profile, True, None), revenue_evidence,
        growth_ranges, current_assumptions,
        (
            "Validated SEC TTM ended 2026-06-30 is the only Candidate starting Revenue base.",
            "FY2025 Revenue is shown only as context and is never an automatic fallback.",
            "Y4/Y5 are deterministic fade outputs, not production research fields.",
        ),
        confidence, None, False, False,
        ("market_price_excluded_from_candidate_generation", "unified_standard_sc_production"),
    )
