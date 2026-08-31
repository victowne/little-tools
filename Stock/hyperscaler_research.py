"""Pure production research profiles for Microsoft and Meta.

The dated evidence and researcher-selected assumptions are explicit.  This
module performs no network access, receives no market price, and never reviews
or applies a candidate.  Production remains Y1-Y3 followed by the existing
deterministic fade.
"""

from dataclasses import dataclass
import math

from Stock.alphabet_research import (
    _anchor,
    _annual_items,
    _forward_evidence,
    _latest_annual,
    _ttm,
    _wacc_evidence,
)
from Stock.company_research_types import (
    ConfidenceAssessment,
    ResearchRange,
    RevenueEvidenceRow,
)
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
)
from Stock.forecast_anchors import RevenueForecastAnchors
from Stock.fundamentals import (
    GROSS_MARGIN, OPERATING_MARGIN, OPERATING_TAX_RATE, REVENUE,
    REVENUE_GROWTH, ROIC, FundamentalHistory,
)
from Stock.valuation import MultiStageDCFAssumptions
from Stock.wacc_audit import WACCAuditResult


MICROSOFT_FY26_Q4 = "https://www.microsoft.com/en-us/Investor/earnings/FY-2026-Q4/press-release-webcast"
MICROSOFT_FY26_Q3_METRICS = "https://www.microsoft.com/en-us/investor/earnings/fy-2026-q3/metrics"
MICROSOFT_FY26_Q3_CALL = "https://www.microsoft.com/en-us/investor/events/fy-2026/earnings-fy-2026-q3"
META_Q2_2026 = "https://investor.atmeta.com/investor-news/press-release-details/2026/Meta-Reports-Second-Quarter-2026-Results/default.aspx"


@dataclass(frozen=True)
class HyperscalerResearchProfileResult:
    lookup: CompanyProfileLookupResult
    revenue_evidence: tuple[RevenueEvidenceRow, ...]
    growth_ranges: tuple[ResearchRange, ...]
    current_assumptions: MultiStageDCFAssumptions
    period_reconciliation: tuple[str, ...]
    confidence_assessments: tuple[ConfidenceAssessment, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ProfileSpec:
    ticker: str
    company_name: str
    growth: tuple[float, float, float]
    mature_margin: float
    starting_sales_to_capital: float
    mature_sales_to_capital: float
    tax: float
    wacc: float
    terminal_growth: float
    evidence: tuple[ResearchEvidenceItem, ...]
    context: BusinessContext
    growth_rationales: tuple[str, str, str]
    margin_rationale: str
    starting_sc_rationale: str
    mature_sc_rationale: str
    confidence: tuple[ConfidenceAssessment, ...]


def _assumption(name, value, rationale, references):
    return ResearchAssumption(name, value, "research_in_progress", rationale, references)


def _evidence_item(evidence_id, label, value, unit, period, source, source_date, retrieved_at, *, category="company_specific_research", notes=""):
    return ResearchEvidenceItem(evidence_id, category, label, value, unit, period, source, source_date, retrieved_at, notes=notes)


def _microsoft_spec(retrieved_at: str) -> _ProfileSpec:
    evidence = (
        _evidence_item("latest_quarter_growth", "FY2026 Q4 consolidated Revenue growth", 0.18, "ratio", "quarter ended 2026-06-30", MICROSOFT_FY26_Q4, "2026-07-29", retrieved_at, category="historical_financial"),
        _evidence_item("cloud_growth", "Microsoft Cloud Revenue growth", 0.27, "ratio", "quarter ended 2026-06-30", MICROSOFT_FY26_Q4, "2026-07-29", retrieved_at),
        _evidence_item("azure_growth", "Azure and other cloud services growth", 0.43, "ratio", "quarter ended 2026-06-30", MICROSOFT_FY26_Q4, "2026-07-29", retrieved_at),
        _evidence_item("commercial_rpo", "Commercial remaining performance obligations", 678e9, "currency_amount", "2026-06-30", MICROSOFT_FY26_Q4, "2026-07-29", retrieved_at, category="management_guidance"),
        _evidence_item("m365_growth", "Microsoft 365 Commercial cloud growth", 0.19, "ratio", "quarter ended 2026-03-31", MICROSOFT_FY26_Q3_METRICS, "2026-04-29", retrieved_at),
        _evidence_item("dynamics_growth", "Dynamics 365 growth", 0.22, "ratio", "quarter ended 2026-03-31", MICROSOFT_FY26_Q3_METRICS, "2026-04-29", retrieved_at),
        _evidence_item("capacity_constraint", "AI capacity and investment context", "Demand exceeded supply and management expected constraints through 2026; calendar-2026 CapEx was guided near $190B.", None, "2026", MICROSOFT_FY26_Q3_CALL, "2026-04-29", retrieved_at, category="management_guidance"),
        _evidence_item("cloud_margin_pressure", "Microsoft Cloud gross margin", 0.66, "ratio", "quarter ended 2026-03-31", MICROSOFT_FY26_Q3_METRICS, "2026-04-29", retrieved_at, notes="AI infrastructure investment and usage pressured gross margin, partly offset by Azure/M365 efficiency."),
        _evidence_item("terminal_macro", "Mature nominal growth framework", "Long-run nominal global growth anchor", None, "terminal period", "Research framework", None, retrieved_at, category="industry_reference"),
    )
    return _ProfileSpec(
        "MSFT", "Microsoft Corporation", (0.18, 0.19, 0.17), 0.42, 0.48, 0.70, 0.19, 0.0925, 0.0325, evidence,
        BusinessContext(
            business_model_summary="Azure and Microsoft 365 anchor a diversified cloud/software franchise, with Security, Dynamics, GitHub, LinkedIn and Windows contributing mixed recurring and transactional economics.",
            primary_growth_drivers=("Azure and AI services", "Microsoft 365 and Copilot", "Security", "Dynamics", "GitHub developer tools"),
            primary_margin_drivers=("software mix", "Azure utilization", "AI inference cost", "data-center depreciation", "Copilot monetization"),
            capital_intensity_notes=("Current Azure/AI capacity buildout depresses accounting Sales-to-Capital.", "Mature efficiency assumes utilization recovery but permanent infrastructure replacement."),
            major_profile_risks=("AI capacity execution", "cloud competition", "inference economics", "depreciation intensity", "regulation"),
        ),
        (
            "18% reflects current consolidated growth and the near-term fiscal evidence without treating Azure growth as company growth.",
            "19% allows a modest capacity-led plateau/acceleration as constrained Azure supply and commercial RPO convert to Revenue.",
            "17% begins normalization while preserving durable cloud, productivity, security and developer-tool growth; no reliable FY3 consensus is treated as fact.",
        ),
        "42% is below current consolidated operating margin: recurring software mix and utilization support it, while AI inference, depreciation and product R&D prevent extrapolating today's margin.",
        "0.48x retains the current buildout economics as the transition starting point.",
        "0.70x is above the 0.49x normalized accounting anchor but below asset-light software economics; it assumes better utilization across a mixed Azure/software portfolio, not a reversion to legacy capital intensity.",
        (
            ConfidenceAssessment("Y1 Growth", "High", "Latest consolidated results and near-term consensus overlap."),
            ConfidenceAssessment("Y2 Growth", "Medium", "RPO and capacity support durability, but timing is uncertain."),
            ConfidenceAssessment("Y3 Growth", "Low", "No dependable third-year consensus endpoint."),
            ConfidenceAssessment("Mature Margin", "Medium", "Strong mix evidence, offset by uncertain AI costs."),
            ConfidenceAssessment("Mature S/C", "Low", "Current buildout obscures normalized incremental efficiency."),
            ConfidenceAssessment("WACC", "Medium", "Existing research framework retained without balancing."),
            ConfidenceAssessment("Terminal Economics", "Low", "Long-horizon utilization and mix remain uncertain."),
        ),
    )


def _meta_spec(retrieved_at: str) -> _ProfileSpec:
    evidence = (
        _evidence_item("latest_quarter_growth", "Q2 2026 consolidated Revenue growth", 0.28, "ratio", "quarter ended 2026-06-30", META_Q2_2026, "2026-07-29", retrieved_at, category="historical_financial"),
        _evidence_item("ad_impressions", "Family of Apps ad impressions growth", 0.14, "ratio", "quarter ended 2026-06-30", META_Q2_2026, "2026-07-29", retrieved_at),
        _evidence_item("ad_price", "Average price per ad growth", 0.12, "ratio", "quarter ended 2026-06-30", META_Q2_2026, "2026-07-29", retrieved_at),
        _evidence_item("daily_people", "Family daily active people growth", 0.03, "ratio", "June 2026", META_Q2_2026, "2026-07-29", retrieved_at),
        _evidence_item("q2_capex", "Q2 2026 capital expenditures", 31.08e9, "currency_amount", "quarter ended 2026-06-30", META_Q2_2026, "2026-07-29", retrieved_at, category="historical_financial"),
        _evidence_item("q2_fcf", "Q2 2026 free cash flow", 0.784e9, "currency_amount", "quarter ended 2026-06-30", META_Q2_2026, "2026-07-29", retrieved_at, category="historical_financial"),
        _evidence_item("one_off_costs", "Q2 legal and severance charges", 3.58e9, "currency_amount", "quarter ended 2026-06-30", META_Q2_2026, "2026-07-29", retrieved_at, notes="Reported margin includes $2.40B legal and $1.18B severance charges."),
        _evidence_item("ai_monetization", "AI recommendation and monetization context", "Management attributed engagement and advertising gains to recommendation improvements while messaging monetization remains an additional option.", None, "2026", META_Q2_2026, "2026-07-29", retrieved_at, category="management_guidance"),
        _evidence_item("terminal_macro", "Mature nominal growth framework", "Long-run nominal global growth anchor", None, "terminal period", "Research framework", None, retrieved_at, category="industry_reference"),
    )
    return _ProfileSpec(
        "META", "Meta Platforms, Inc.", (0.24, 0.20, 0.17), 0.36, 0.47, 0.75, 0.16, 0.0975, 0.0325, evidence,
        BusinessContext(
            business_model_summary="Family of Apps advertising funds AI recommendation, messaging monetization and Reality Labs while a large infrastructure cycle supports engagement and ad efficiency.",
            primary_growth_drivers=("ad impressions", "ad pricing", "AI recommendations", "Reels monetization", "WhatsApp and messaging monetization"),
            primary_margin_drivers=("Family of Apps monetization", "AI R&D", "infrastructure depreciation", "Reality Labs losses"),
            capital_intensity_notes=("The AI infrastructure cycle sharply lowers current free cash flow and accounting capital efficiency.",),
            major_profile_risks=("advertising cyclicality", "Reality Labs losses", "AI infrastructure returns", "regulation", "engagement competition"),
        ),
        (
            "24% is below the latest 28% quarter and reflects both volume and pricing support without annualizing a single quarter.",
            "20% recognizes durable recommendation and monetization gains while allowing comparison and scale normalization.",
            "17% preserves medium-term AI, Reels and messaging opportunity but discounts the absence of reliable FY3 consensus and ongoing infrastructure intensity.",
        ),
        "36% is a normalized consolidated midpoint below current TTM economics: Family of Apps supports it, but permanent AI R&D, depreciation and Reality Labs losses remain.",
        "0.47x preserves current buildout conditions at the beginning of the transition.",
        "0.75x sits above the 0.66x normalized anchor as utilization and monetization mature, but remains constrained by infrastructure replacement and Reality Labs rather than assuming pure ad-platform economics.",
        (
            ConfidenceAssessment("Y1 Growth", "High", "Latest ad volume/price and consensus provide direct support."),
            ConfidenceAssessment("Y2 Growth", "Medium", "AI monetization evidence is strong but comparisons normalize."),
            ConfidenceAssessment("Y3 Growth", "Low", "No dependable third-year consensus endpoint."),
            ConfidenceAssessment("Mature Margin", "Medium", "Family of Apps economics are observable; long-run AI/RL costs are not."),
            ConfidenceAssessment("Mature S/C", "Low", "The current infrastructure cycle obscures normalized efficiency."),
            ConfidenceAssessment("WACC", "Medium", "Existing research framework retained independently."),
            ConfidenceAssessment("Terminal Economics", "Low", "Long-horizon mix and capital replacement are uncertain."),
        ),
    )


def _build(ticker: str, current: MultiStageDCFAssumptions, history: FundamentalHistory, *, revenue_anchors=None, wacc_audit=None, retrieved_at: str) -> HyperscalerResearchProfileResult:
    normalized = ticker.strip().upper()
    if normalized not in {"MSFT", "META"}:
        raise ValueError("unsupported_hyperscaler_profile")
    spec = _microsoft_spec(retrieved_at) if normalized == "MSFT" else _meta_spec(retrieved_at)
    annual_revenue = _latest_annual(history, REVENUE, "latest_annual_revenue", "Latest annual Revenue", "currency_amount")
    ttm_revenue = _ttm(history, REVENUE, "ttm_revenue", "TTM Revenue", "currency_amount")
    annual_growth = _latest_annual(history, REVENUE_GROWTH, "latest_annual_growth", "Latest annual Revenue growth", "ratio")
    cagr = _anchor(history, "revenue_cagr_3y")
    annual_margin = _latest_annual(history, OPERATING_MARGIN, "latest_annual_operating_margin", "Latest annual Operating Margin", "ratio")
    ttm_margin = _ttm(history, OPERATING_MARGIN, "ttm_operating_margin", "TTM Operating Margin", "ratio")
    latest_sc = _anchor(history, "latest_sales_to_capital")
    normalized_sc = _anchor(history, "sales_to_capital_3y")
    accounting_roic = _latest_annual(history, ROIC, "accounting_roic", "Accounting ROIC", "ratio")
    annual_tax = _latest_annual(history, OPERATING_TAX_RATE, "latest_operating_tax_rate", "Latest annual Operating Tax Rate", "ratio")
    forward = _forward_evidence(revenue_anchors)
    risk = _wacc_evidence(wacc_audit, None, None, retrieved_at)
    base_evidence = tuple(x for x in (annual_revenue, ttm_revenue, annual_growth, cagr, annual_margin, ttm_margin, latest_sc, normalized_sc, accounting_roic, annual_tax) if x is not None)
    evidence = base_evidence + spec.evidence + forward + risk
    refs = {item.evidence_id for item in evidence}
    def a(name, value, rationale, wanted):
        selected = tuple(x for x in wanted if x in refs)
        return _assumption(name, value, rationale, selected or tuple(sorted(refs)[:1]))
    y1 = a("year1_growth", spec.growth[0], spec.growth_rationales[0], ("ttm_revenue", "fy1_consensus_revenue", "latest_quarter_growth"))
    y2 = a("year2_growth", spec.growth[1], spec.growth_rationales[1], ("fy2_consensus_revenue", "commercial_rpo", "ai_monetization"))
    y3 = a("year3_growth", spec.growth[2], spec.growth_rationales[2], ("revenue_cagr_3y", "cloud_growth", "ad_impressions", "ad_price"))
    fade = a("revenue_fade_years", 8, "Eight years extend the medium-term view to mature growth without pretending that Y4/Y5 are separately researched endpoints.", ("capacity_constraint", "q2_capex"))
    terminal_g = a("terminal_growth", spec.terminal_growth, "3.25% is a conservative mature nominal-growth assumption and is not used to offset explicit-period value.", ("terminal_macro",))
    start_margin_value = float(ttm_margin.value) if ttm_margin is not None else current.starting_operating_margin
    start_margin = a("starting_operating_margin", start_margin_value, "Exact validated TTM consolidated operating margin; not a discretionary override.", ("ttm_operating_margin", "latest_quarter_growth"))
    mature_margin = a("mature_operating_margin", spec.mature_margin, spec.margin_rationale, ("ttm_operating_margin", "cloud_margin_pressure", "one_off_costs"))
    start_sc = a("starting_sales_to_capital", spec.starting_sales_to_capital, spec.starting_sc_rationale, ("latest_sales_to_capital", "q2_capex", "capacity_constraint"))
    mature_sc = a("mature_sales_to_capital", spec.mature_sales_to_capital, spec.mature_sc_rationale, ("sales_to_capital_3y", "q2_capex", "capacity_constraint"))
    tax = a("operating_tax_rate", spec.tax, "Normalized operating tax assumption; it is not fitted to a single unusual year.", ("latest_operating_tax_rate",))
    wacc = a("research_wacc", spec.wacc, "Existing long-horizon Research WACC is retained independently of operating assumptions and market price.", ("formula_based_wacc", "historical_raw_beta"))
    horizon = a("forecast_years", 11, "Eleven years exactly equals three explicit years plus eight fade years.", ("capacity_constraint", "terminal_macro"))
    terminal_roic = spec.mature_margin * (1 - spec.tax) * spec.mature_sales_to_capital
    terminal_reinvestment = spec.terminal_growth / terminal_roic
    profile = CompanyResearchProfile(
        ticker=spec.ticker, issuer_id=spec.ticker, company_name=spec.company_name,
        profile_status="research_in_progress", business_summary=spec.context.business_model_summary,
        business_context=spec.context,
        revenue_framework=RevenueResearchFramework(ttm_revenue, annual_revenue, ttm_revenue, annual_growth, cagr, revenue_anchors, y1, y2, y3, fade, terminal_g, "Fiscal consensus is supporting level evidence; DCF years remain TTM-based.", fade.rationale, terminal_g.rationale, ("ttm_and_fiscal_consensus_periods_differ",)),
        margin_framework=MarginResearchFramework(annual_margin, ttm_margin, _annual_items(history, OPERATING_MARGIN, prefix="operating_margin", label="Operating Margin", unit="ratio"), _annual_items(history, GROSS_MARGIN, prefix="gross_margin", label="Gross Margin", unit="ratio"), start_margin, mature_margin, start_margin.rationale, mature_margin.rationale),
        capital_efficiency_framework=CapitalEfficiencyResearchFramework(latest_sc, normalized_sc, accounting_roic, None, None, start_sc, mature_sc, start_margin_value * (1-spec.tax) * spec.starting_sales_to_capital, terminal_roic, start_sc.rationale, mature_sc.rationale, ("current_ai_buildout_depresses_accounting_efficiency",)),
        wacc_framework=WACCResearchFramework(wacc_audit=wacc_audit, research_wacc=wacc, rationale=wacc.rationale, warnings=("research_wacc_candidate_not_reviewed",)),
        terminal_framework=TerminalResearchFramework(terminal_g, mature_margin, mature_sc, terminal_roic, terminal_reinvestment, 1-terminal_reinvestment, terminal_g.rationale, mature_margin.rationale, mature_sc.rationale, ("terminal_macro",), ()),
        operating_tax_rate=tax, forecast_years=horizon,
        rationale=f"Unreviewed {spec.company_name} research candidate; evidence precedes assumptions and market price is excluded from construction.",
        warnings=("research_candidate_not_reviewed", "candidate_not_applied_to_live_dcf"), last_reviewed_at=None,
        evidence_items=evidence, uncertainty_notes=spec.context.major_profile_risks,
        future_scenario_drivers=("Y3 growth duration", "mature operating margin", "mature Sales-to-Capital", "Research WACC"),
    )
    revenues = []
    for item in _annual_items(history, REVENUE, prefix="annual_revenue", label="Annual Revenue", unit="currency_amount")[-4:]:
        revenues.append(RevenueEvidenceRow(f"FY ended {item.period}", item.period, float(item.value), None, item.source, item.period, retrieved_at))
    if ttm_revenue is not None:
        revenues.append(RevenueEvidenceRow("Current validated TTM", ttm_revenue.period, float(ttm_revenue.value), None, ttm_revenue.source, ttm_revenue.period, retrieved_at, notes="DCF starting base."))
    if revenue_anchors is not None:
        for point in revenue_anchors.points[:2]:
            revenues.append(RevenueEvidenceRow(f"FY{point.forecast_year_index} consensus", str(point.fiscal_period.date()) if point.fiscal_period is not None else None, point.revenue_estimate, point.implied_revenue_growth, point.source, str(point.source_as_of.date()) if point.source_as_of is not None else None, str(point.source_as_of.date()) if point.source_as_of is not None else None, point.analyst_count, "Fiscal endpoint; not identical to the TTM-based DCF year."))
    growth_ranges = tuple(ResearchRange(f"year{i}_growth", v-0.02, v, v+0.02, r, getattr(profile.revenue_framework, f"year{i}_growth").evidence_references) for i,(v,r) in enumerate(zip(spec.growth, spec.growth_rationales),1))
    reconciliation = ("Validated TTM is the DCF starting Revenue.", "Fiscal consensus endpoints are evidence, not mechanically copied into TTM-based DCF years.", "Y4/Y5 are implied by the deterministic fade and are not production assumptions.")
    return HyperscalerResearchProfileResult(CompanyProfileLookupResult(profile, True, None), tuple(revenues), growth_ranges, current, reconciliation, spec.confidence, ("market_price_excluded_from_candidate_generation",))


def build_microsoft_research_profile(current_assumptions, history, *, revenue_anchors=None, wacc_audit=None, retrieved_at="2026-08-23"):
    return _build("MSFT", current_assumptions, history, revenue_anchors=revenue_anchors, wacc_audit=wacc_audit, retrieved_at=retrieved_at)


def build_meta_research_profile(current_assumptions, history, *, revenue_anchors=None, wacc_audit=None, retrieved_at="2026-08-23"):
    return _build("META", current_assumptions, history, revenue_anchors=revenue_anchors, wacc_audit=wacc_audit, retrieved_at=retrieved_at)
