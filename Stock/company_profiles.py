"""Pure Company Research Profile domain models and DCF translation.

Evidence, research judgment, research assumptions, and valuation output are
deliberately separate.  This module does not fetch data, import Streamlit,
mutate UI state, or run a valuation.  A profile can describe evidence without
turning that evidence into an assumption, and a provisional profile remains
provisional even when it is complete enough to translate into DCF inputs.
"""

from dataclasses import dataclass
import math
from typing import Literal

from Stock.forecast_anchors import RevenueForecastAnchors
from Stock.fundamentals import (
    FUNDAMENTAL_GROWTH_CAPACITY,
    GROSS_MARGIN,
    OPERATING_MARGIN,
    REINVESTMENT_RATE,
    REVENUE,
    REVENUE_GROWTH,
    ROIC,
    FundamentalHistory,
)
from Stock.valuation import MultiStageDCFAssumptions
from Stock.research_wacc import ResearchWACCDecision
from Stock.wacc_audit import WACCAuditResult, issuer_normalization_metadata


ProfileStatus = Literal["provisional", "research_in_progress", "reviewed"]
AssumptionStatus = Literal["provisional", "research_in_progress", "reviewed"]
EvidenceCategory = Literal[
    "historical_financial",
    "forward_consensus",
    "management_guidance",
    "market_risk",
    "industry_reference",
    "company_specific_research",
]

PROFILE_STATUSES = {"provisional", "research_in_progress", "reviewed"}
ASSUMPTION_STATUSES = PROFILE_STATUSES


def _finite_optional(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _period_text(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class ResearchEvidenceItem:
    """One auditable numeric or narrative item; never an assumption by itself."""

    evidence_id: str
    category: EvidenceCategory
    label: str
    value: float | str | None
    unit: str | None
    period: str | None
    source: str
    source_date: str | None = None
    retrieved_at: str | None = None
    analyst_count: int | None = None
    notes: str = ""
    available: bool = True


@dataclass(frozen=True)
class ResearchAssumption:
    """A researcher-controlled numeric assumption with explicit provenance."""

    assumption_id: str
    value: float | int | None
    status: AssumptionStatus
    rationale: str = ""
    evidence_references: tuple[str, ...] = ()
    last_reviewed_at: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ASSUMPTION_STATUSES:
            raise ValueError("invalid_research_assumption_status")


@dataclass(frozen=True)
class BusinessContext:
    business_model_summary: str = ""
    primary_growth_drivers: tuple[str, ...] = ()
    primary_margin_drivers: tuple[str, ...] = ()
    capital_intensity_notes: tuple[str, ...] = ()
    cyclicality_notes: tuple[str, ...] = ()
    competitive_structure_notes: tuple[str, ...] = ()
    major_profile_risks: tuple[str, ...] = ()


@dataclass(frozen=True)
class RevenueResearchFramework:
    starting_revenue: ResearchEvidenceItem | None = None
    latest_annual_revenue: ResearchEvidenceItem | None = None
    ttm_revenue: ResearchEvidenceItem | None = None
    latest_annual_growth: ResearchEvidenceItem | None = None
    historical_3y_cagr: ResearchEvidenceItem | None = None
    forward_revenue_anchors: RevenueForecastAnchors | None = None
    year1_growth: ResearchAssumption | None = None
    year2_growth: ResearchAssumption | None = None
    year3_growth: ResearchAssumption | None = None
    revenue_fade_years: ResearchAssumption | None = None
    terminal_growth: ResearchAssumption | None = None
    near_term_growth_rationale: str = ""
    fade_rationale: str = ""
    terminal_growth_rationale: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarginResearchFramework:
    latest_annual_operating_margin: ResearchEvidenceItem | None = None
    ttm_operating_margin: ResearchEvidenceItem | None = None
    historical_operating_margin: tuple[ResearchEvidenceItem, ...] = ()
    historical_gross_margin: tuple[ResearchEvidenceItem, ...] = ()
    starting_operating_margin: ResearchAssumption | None = None
    mature_operating_margin: ResearchAssumption | None = None
    current_margin_rationale: str = ""
    mature_margin_rationale: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapitalEfficiencyResearchFramework:
    latest_sales_to_capital: ResearchEvidenceItem | None = None
    normalized_3y_sales_to_capital: ResearchEvidenceItem | None = None
    accounting_roic: ResearchEvidenceItem | None = None
    reinvestment_rate: ResearchEvidenceItem | None = None
    fundamental_growth_capacity: ResearchEvidenceItem | None = None
    starting_sales_to_capital: ResearchAssumption | None = None
    mature_sales_to_capital: ResearchAssumption | None = None
    implied_starting_roic: float | None = None
    implied_terminal_roic: float | None = None
    starting_s2c_rationale: str = ""
    mature_s2c_rationale: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class WACCResearchFramework:
    """Profile reference to Phase 2 WACC evidence; contains no WACC formula."""

    wacc_audit: WACCAuditResult | None = None
    wacc_decision: ResearchWACCDecision | None = None
    research_wacc: ResearchAssumption | None = None
    rationale: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TerminalResearchFramework:
    terminal_growth: ResearchAssumption | None = None
    mature_operating_margin: ResearchAssumption | None = None
    mature_sales_to_capital: ResearchAssumption | None = None
    terminal_roic: float | None = None
    terminal_reinvestment_rate: float | None = None
    terminal_fcff_conversion: float | None = None
    terminal_growth_rationale: str = ""
    mature_margin_rationale: str = ""
    mature_capital_efficiency_rationale: str = ""
    evidence_references: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompanyResearchProfile:
    ticker: str
    issuer_id: str
    company_name: str
    profile_status: ProfileStatus
    business_summary: str
    business_context: BusinessContext
    revenue_framework: RevenueResearchFramework
    margin_framework: MarginResearchFramework
    capital_efficiency_framework: CapitalEfficiencyResearchFramework
    wacc_framework: WACCResearchFramework
    terminal_framework: TerminalResearchFramework
    operating_tax_rate: ResearchAssumption | None
    forecast_years: ResearchAssumption | None
    rationale: str = ""
    warnings: tuple[str, ...] = ()
    last_reviewed_at: str | None = None
    evidence_items: tuple[ResearchEvidenceItem, ...] = ()
    uncertainty_notes: tuple[str, ...] = ()
    future_scenario_drivers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.profile_status not in PROFILE_STATUSES:
            raise ValueError("invalid_company_profile_status")


@dataclass(frozen=True)
class CompanyProfileLookupResult:
    profile: CompanyResearchProfile | None
    available: bool
    reason: str | None


@dataclass(frozen=True)
class ProfileDCFTranslationResult:
    assumptions: MultiStageDCFAssumptions | None
    available: bool
    missing_fields: tuple[str, ...]
    profile_status: ProfileStatus
    reason: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssumptionEvidenceComparison:
    metric: str
    research_value: float | None
    evidence_value: float | None
    delta: float | None
    unit: str
    available: bool
    reason: str | None


@dataclass(frozen=True)
class CompanyProfileDefinition:
    issuer_id: str
    ticker: str
    company_name: str
    business_summary: str
    warnings: tuple[str, ...]


COMPANY_PROFILE_DEFINITIONS: dict[str, CompanyProfileDefinition] = {
    "NVDA": CompanyProfileDefinition(
        issuer_id="NVDA",
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        business_summary="Provisional profile skeleton; company research is pending.",
        warnings=("profile_contains_provisional_assumptions",),
    ),
    "ALPHABET_INC": CompanyProfileDefinition(
        issuer_id="ALPHABET_INC",
        ticker="GOOGL",
        company_name="Alphabet Inc.",
        business_summary="Provisional issuer-level profile skeleton; company research is pending.",
        warnings=(
            "profile_contains_provisional_assumptions",
            "multi_class_security_structure",
        ),
    ),
}


def normalize_profile_issuer(ticker: str) -> str:
    """Reuse Phase 2 issuer normalization, including GOOG/GOOGL."""
    return issuer_normalization_metadata(ticker)[0]


def _assumption(
    assumption_id: str,
    value: float | int,
    *,
    status: AssumptionStatus = "provisional",
) -> ResearchAssumption:
    return ResearchAssumption(
        assumption_id=assumption_id,
        value=value,
        status=status,
        rationale="Inherited from the current provisional DCF state.",
    )


def _annual_evidence(
    history: FundamentalHistory | None,
    metric: str,
    evidence_id: str,
    label: str,
    unit: str,
) -> ResearchEvidenceItem | None:
    if history is None or history.annual.empty or metric not in history.annual:
        return None
    value = _finite_optional(history.annual.iloc[-1].get(metric))
    if value is None:
        return None
    return ResearchEvidenceItem(
        evidence_id=evidence_id,
        category="historical_financial",
        label=label,
        value=value,
        unit=unit,
        period=_period_text(history.annual.index[-1]),
        source="FundamentalHistory annual",
    )


def _ttm_evidence(
    history: FundamentalHistory | None,
    metric: str,
    evidence_id: str,
    label: str,
    unit: str,
) -> ResearchEvidenceItem | None:
    if history is None:
        return None
    result = history.ttm.get(metric)
    if result is None or not result.available:
        return None
    value = _finite_optional(result.value)
    if value is None:
        return None
    period = (
        f"{result.periods_used[0].date()} to {result.periods_used[-1].date()}"
        if result.periods_used else None
    )
    return ResearchEvidenceItem(
        evidence_id=evidence_id,
        category="historical_financial",
        label=label,
        value=value,
        unit=unit,
        period=period,
        source="FundamentalHistory validated TTM",
    )


def _historical_series(
    history: FundamentalHistory | None,
    metric: str,
    evidence_prefix: str,
    label: str,
) -> tuple[ResearchEvidenceItem, ...]:
    if history is None or history.annual.empty or metric not in history.annual:
        return ()
    items = []
    for period, raw_value in history.annual[metric].items():
        value = _finite_optional(raw_value)
        if value is None:
            continue
        period_text = _period_text(period)
        items.append(ResearchEvidenceItem(
            evidence_id=f"{evidence_prefix}_{period_text}",
            category="historical_financial",
            label=label,
            value=value,
            unit="ratio",
            period=period_text,
            source="FundamentalHistory annual",
        ))
    return tuple(items)


def _historical_anchor_evidence(
    history: FundamentalHistory | None,
    *,
    anchor: str,
) -> ResearchEvidenceItem | None:
    if history is None:
        return None
    if anchor == "revenue_cagr_3y":
        result = history.dcf_anchors.revenue_cagr.get(3)
        label, unit = "Historical Revenue CAGR 3Y", "ratio"
    elif anchor == "sales_to_capital_3y":
        result = history.dcf_anchors.normalized_sales_to_capital.get(3)
        label, unit = "Normalized Sales-to-Capital 3Y", "multiple"
    else:
        values = history.dcf_anchors.annual_sales_to_capital
        result = values[max(values)] if values else None
        label, unit = "Latest annual Sales-to-Capital", "multiple"
    if result is None or not result.available:
        return None
    value = _finite_optional(result.value)
    if value is None:
        return None
    return ResearchEvidenceItem(
        evidence_id=anchor,
        category="historical_financial",
        label=label,
        value=value,
        unit=unit,
        period=(
            f"{_period_text(result.start_period)} to {_period_text(result.end_period)}"
            if result.start_period is not None else _period_text(result.end_period)
        ),
        source="FundamentalHistory historical DCF anchors",
    )


def build_provisional_company_profile(
    ticker: str,
    assumptions: MultiStageDCFAssumptions | None = None,
    *,
    history: FundamentalHistory | None = None,
    revenue_anchors: RevenueForecastAnchors | None = None,
    wacc_audit: WACCAuditResult | None = None,
) -> CompanyProfileLookupResult:
    """Build a read-only provisional skeleton from explicitly supplied inputs.

    Existing DCF assumptions may be carried into the skeleton for visibility;
    no historical or forward evidence is used to populate them.
    """
    issuer_id = normalize_profile_issuer(ticker)
    definition = COMPANY_PROFILE_DEFINITIONS.get(issuer_id)
    if definition is None:
        return CompanyProfileLookupResult(None, False, "profile_unavailable")

    if assumptions is None:
        year1 = year2 = year3 = fade = terminal_growth = None
        starting_margin = mature_margin = None
        starting_s2c = mature_s2c = None
        research_wacc = tax = horizon = None
    else:
        growth = assumptions.near_term_revenue_growth
        year1 = _assumption("year1_growth", growth[0]) if len(growth) > 0 else None
        year2 = _assumption("year2_growth", growth[1]) if len(growth) > 1 else None
        year3 = _assumption("year3_growth", growth[2]) if len(growth) > 2 else None
        fade = _assumption("revenue_fade_years", assumptions.revenue_fade_years)
        terminal_growth = _assumption("terminal_growth", assumptions.terminal_growth)
        starting_margin = _assumption(
            "starting_operating_margin", assumptions.starting_operating_margin
        )
        mature_margin = _assumption(
            "mature_operating_margin", assumptions.mature_operating_margin
        )
        starting_s2c = _assumption(
            "starting_sales_to_capital", assumptions.starting_sales_to_capital
        )
        mature_s2c = _assumption(
            "mature_sales_to_capital", assumptions.mature_sales_to_capital
        )
        research_wacc = _assumption("research_wacc", assumptions.wacc)
        tax = _assumption("operating_tax_rate", assumptions.operating_tax_rate)
        horizon = _assumption("forecast_years", assumptions.forecast_years)

    starting_revenue = None
    if revenue_anchors is not None:
        starting_revenue = ResearchEvidenceItem(
            evidence_id="starting_revenue",
            category="historical_financial",
            label="DCF starting Revenue",
            value=revenue_anchors.current_revenue_base,
            unit="currency_amount",
            period=_period_text(revenue_anchors.base_period),
            source=revenue_anchors.base_kind,
        )

    warnings = list(definition.warnings)
    if revenue_anchors is None:
        warnings.append("forward_evidence_missing")
    if wacc_audit is None or not wacc_audit.available:
        warnings.append("research_wacc_evidence_unavailable")
    warnings.append("mature_economics_require_review")

    implied_starting_roic = (
        assumptions.starting_operating_margin
        * (1 - assumptions.operating_tax_rate)
        * assumptions.starting_sales_to_capital
        if assumptions is not None else None
    )
    profile = CompanyResearchProfile(
        ticker=definition.ticker,
        issuer_id=definition.issuer_id,
        company_name=definition.company_name,
        profile_status="provisional",
        business_summary=definition.business_summary,
        business_context=BusinessContext(),
        revenue_framework=RevenueResearchFramework(
            starting_revenue=starting_revenue,
            latest_annual_revenue=_annual_evidence(
                history, REVENUE, "latest_annual_revenue", "Latest annual Revenue",
                "currency_amount",
            ),
            ttm_revenue=_ttm_evidence(
                history, REVENUE, "ttm_revenue", "TTM Revenue", "currency_amount"
            ),
            latest_annual_growth=_annual_evidence(
                history, REVENUE_GROWTH, "latest_annual_growth",
                "Latest annual Revenue growth", "ratio",
            ),
            historical_3y_cagr=_historical_anchor_evidence(
                history, anchor="revenue_cagr_3y"
            ),
            forward_revenue_anchors=revenue_anchors,
            year1_growth=year1, year2_growth=year2, year3_growth=year3,
            revenue_fade_years=fade, terminal_growth=terminal_growth,
            warnings=("evidence_period_mismatch",) if revenue_anchors else (),
        ),
        margin_framework=MarginResearchFramework(
            latest_annual_operating_margin=_annual_evidence(
                history, OPERATING_MARGIN, "latest_annual_operating_margin",
                "Latest annual Operating Margin", "ratio",
            ),
            ttm_operating_margin=_ttm_evidence(
                history, OPERATING_MARGIN, "ttm_operating_margin",
                "TTM Operating Margin", "ratio",
            ),
            historical_operating_margin=_historical_series(
                history, OPERATING_MARGIN, "operating_margin", "Operating Margin"
            ),
            historical_gross_margin=_historical_series(
                history, GROSS_MARGIN, "gross_margin", "Gross Margin"
            ),
            starting_operating_margin=starting_margin,
            mature_operating_margin=mature_margin,
        ),
        capital_efficiency_framework=CapitalEfficiencyResearchFramework(
            latest_sales_to_capital=_historical_anchor_evidence(
                history, anchor="latest_sales_to_capital"
            ),
            normalized_3y_sales_to_capital=_historical_anchor_evidence(
                history, anchor="sales_to_capital_3y"
            ),
            accounting_roic=_annual_evidence(
                history, ROIC, "accounting_roic", "Accounting ROIC", "ratio"
            ),
            reinvestment_rate=_annual_evidence(
                history, REINVESTMENT_RATE, "reinvestment_rate",
                "Simplified Reinvestment Rate", "ratio",
            ),
            fundamental_growth_capacity=_annual_evidence(
                history, FUNDAMENTAL_GROWTH_CAPACITY,
                "fundamental_growth_capacity", "Fundamental Growth Capacity",
                "ratio",
            ),
            starting_sales_to_capital=starting_s2c,
            mature_sales_to_capital=mature_s2c,
            implied_starting_roic=implied_starting_roic,
            implied_terminal_roic=(
                assumptions.derived_terminal_roic if assumptions else None
            ),
        ),
        wacc_framework=WACCResearchFramework(
            wacc_audit=wacc_audit,
            research_wacc=research_wacc,
            warnings=("research_wacc_not_reviewed",),
        ),
        terminal_framework=TerminalResearchFramework(
            terminal_growth=terminal_growth,
            mature_operating_margin=mature_margin,
            mature_sales_to_capital=mature_s2c,
            terminal_roic=(assumptions.derived_terminal_roic if assumptions else None),
            terminal_reinvestment_rate=(
                assumptions.terminal_reinvestment_rate if assumptions else None
            ),
            terminal_fcff_conversion=(
                1 - assumptions.terminal_reinvestment_rate
                if assumptions is not None
                and assumptions.terminal_reinvestment_rate is not None else None
            ),
        ),
        operating_tax_rate=tax,
        forecast_years=horizon,
        rationale="Current Phase 2 values are carried only as provisional assumptions.",
        warnings=tuple(dict.fromkeys(warnings)),
    )
    return CompanyProfileLookupResult(profile, True, None)


def get_company_profile(ticker: str) -> CompanyProfileLookupResult:
    """Resolve an explicit provisional skeleton; unknown issuers fail clearly."""
    return build_provisional_company_profile(ticker)


def build_multistage_assumptions_from_profile(
    profile: CompanyResearchProfile,
) -> ProfileDCFTranslationResult:
    """Translate explicit research assumptions without using evidence fallbacks."""
    fields = {
        "revenue_framework.year1_growth": profile.revenue_framework.year1_growth,
        "revenue_framework.year2_growth": profile.revenue_framework.year2_growth,
        "revenue_framework.year3_growth": profile.revenue_framework.year3_growth,
        "revenue_framework.revenue_fade_years": (
            profile.revenue_framework.revenue_fade_years
        ),
        "revenue_framework.terminal_growth": profile.revenue_framework.terminal_growth,
        "margin_framework.starting_operating_margin": (
            profile.margin_framework.starting_operating_margin
        ),
        "margin_framework.mature_operating_margin": (
            profile.margin_framework.mature_operating_margin
        ),
        "capital_efficiency_framework.starting_sales_to_capital": (
            profile.capital_efficiency_framework.starting_sales_to_capital
        ),
        "capital_efficiency_framework.mature_sales_to_capital": (
            profile.capital_efficiency_framework.mature_sales_to_capital
        ),
        "wacc_framework.research_wacc": profile.wacc_framework.research_wacc,
        "operating_tax_rate": profile.operating_tax_rate,
        "forecast_years": profile.forecast_years,
    }
    missing = tuple(
        name for name, assumption in fields.items()
        if assumption is None or assumption.value is None
    )
    if missing:
        return ProfileDCFTranslationResult(
            None, False, missing, profile.profile_status,
            "research_profile_incomplete", profile.warnings,
        )
    try:
        values = {name: assumption.value for name, assumption in fields.items()}
        assumptions = MultiStageDCFAssumptions(
            forecast_years=values["forecast_years"],
            near_term_revenue_growth=(
                values["revenue_framework.year1_growth"],
                values["revenue_framework.year2_growth"],
                values["revenue_framework.year3_growth"],
            ),
            revenue_fade_years=values[
                "revenue_framework.revenue_fade_years"
            ],
            terminal_growth=values["revenue_framework.terminal_growth"],
            starting_operating_margin=values[
                "margin_framework.starting_operating_margin"
            ],
            mature_operating_margin=values[
                "margin_framework.mature_operating_margin"
            ],
            starting_sales_to_capital=values[
                "capital_efficiency_framework.starting_sales_to_capital"
            ],
            mature_sales_to_capital=values[
                "capital_efficiency_framework.mature_sales_to_capital"
            ],
            operating_tax_rate=values["operating_tax_rate"],
            wacc=values["wacc_framework.research_wacc"],
        )
    except (TypeError, ValueError) as exc:
        return ProfileDCFTranslationResult(
            None, False, (), profile.profile_status,
            "invalid_research_assumptions",
            tuple(profile.warnings) + (str(exc),),
        )
    return ProfileDCFTranslationResult(
        assumptions, True, (), profile.profile_status, None,
        tuple(profile.warnings) + assumptions.validation_warnings,
    )


def compare_research_assumption_to_evidence(
    metric: str,
    research_assumption: ResearchAssumption | None,
    evidence: ResearchEvidenceItem | None,
    *,
    unit: str,
) -> AssumptionEvidenceComparison:
    """Return a signed numeric delta without judging either input."""
    research_value = (
        _finite_optional(research_assumption.value)
        if research_assumption is not None else None
    )
    evidence_value = (
        _finite_optional(evidence.value) if evidence is not None and evidence.available
        else None
    )
    if research_value is None or evidence_value is None:
        return AssumptionEvidenceComparison(
            metric, research_value, evidence_value, None, unit, False,
            "comparison_inputs_unavailable",
        )
    return AssumptionEvidenceComparison(
        metric, research_value, evidence_value,
        research_value - evidence_value, unit, True, None,
    )


def profile_assumption_comparisons(
    profile: CompanyResearchProfile,
) -> tuple[AssumptionEvidenceComparison, ...]:
    """Descriptive profile comparisons; absence/alignment remains explicit."""
    revenue = profile.revenue_framework
    aligned_y1 = None
    if revenue.forward_revenue_anchors is not None:
        point = revenue.forward_revenue_anchors.points[0]
        # Existing anchor growth is FY-to-FY. It is only directly comparable
        # when the anchor layer does not flag a TTM-period mismatch.
        if (
            point.available
            and point.implied_revenue_growth is not None
            and "ttm_base_not_directly_comparable_to_fiscal_consensus"
            not in revenue.forward_revenue_anchors.warnings
        ):
            aligned_y1 = ResearchEvidenceItem(
                "aligned_consensus_y1_growth", "forward_consensus",
                "Aligned consensus Y1 growth", point.implied_revenue_growth,
                "ratio", _period_text(point.fiscal_period), point.source,
                source_date=_period_text(point.source_as_of),
                analyst_count=point.analyst_count,
            )
    capital = profile.capital_efficiency_framework
    formula_wacc = None
    audit = profile.wacc_framework.wacc_audit
    if audit is not None and audit.available and audit.calculated_wacc is not None:
        formula_wacc = ResearchEvidenceItem(
            "formula_based_wacc", "market_risk", "Formula-Based WACC",
            audit.calculated_wacc, "ratio", audit.risk_free_period,
            "Phase 2 WACC audit",
        )
    return (
        compare_research_assumption_to_evidence(
            "year1_growth", revenue.year1_growth, aligned_y1, unit="percentage_points"
        ),
        compare_research_assumption_to_evidence(
            "starting_operating_margin",
            profile.margin_framework.starting_operating_margin,
            profile.margin_framework.ttm_operating_margin,
            unit="percentage_points",
        ),
        compare_research_assumption_to_evidence(
            "mature_operating_margin",
            profile.margin_framework.mature_operating_margin,
            profile.margin_framework.ttm_operating_margin,
            unit="percentage_points",
        ),
        compare_research_assumption_to_evidence(
            "starting_sales_to_capital",
            capital.starting_sales_to_capital,
            capital.normalized_3y_sales_to_capital,
            unit="multiple",
        ),
        compare_research_assumption_to_evidence(
            "research_wacc", profile.wacc_framework.research_wacc,
            formula_wacc, unit="percentage_points",
        ),
    )
