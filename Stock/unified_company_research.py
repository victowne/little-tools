"""Pure unified-production Research Candidates for MU, AAPL, and AVGO.

Company differences are expressed only through researched assumptions,
evidence, confidence, and model-risk disclosure.  Every translated candidate
uses the existing standard Sales-to-Capital production DCF.
"""

from dataclasses import dataclass

from Stock.alphabet_research import (
    ResearchRange,
    RevenueEvidenceRow,
    _anchor,
    _annual_items,
    _forward_evidence,
    _latest_annual,
    _ttm,
    _wacc_evidence,
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
from Stock.fundamentals import GROSS_MARGIN, OPERATING_MARGIN, OPERATING_TAX_RATE, REVENUE, REVENUE_GROWTH, ROIC, FundamentalHistory
from Stock.hyperscaler_research import ConfidenceAssessment
from Stock.micron_recalibration import (
    MicronPeriodAlignment,
    build_micron_period_alignment,
)
from Stock.valuation import MultiStageDCFAssumptions
from Stock.wacc_audit import WACCAuditResult


MU_Q3_2026 = "https://www.sec.gov/Archives/edgar/data/723125/000072312526000013/a2026q3ex991-pressrelease.htm"
MU_Q3_2026_10Q = "https://www.sec.gov/Archives/edgar/data/723125/000072312526000015/mu-20260528.htm"
AAPL_Q3_2026 = "https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/"
AAPL_Q3_2026_10Q = "https://www.sec.gov/Archives/edgar/data/320193/000032019326000020/aapl-20260627.htm"
AVGO_Q2_2026 = "https://investors.broadcom.com/news-releases/news-release-details/broadcom-inc-announces-second-quarter-fiscal-year-2026-financial"
AVGO_Q2_2026_10Q = "https://www.sec.gov/Archives/edgar/data/1730168/000173016826000054/avgo-20260503.htm"
MU_Q3_2026_REMARKS = "https://investors.micron.com/static-files/631b1a32-5537-46ae-8f40-82e42fc79dfe"
MU_FACTSET_CONSENSUS = "https://www.finanzen.net/schaetzungen/micron_technology"


@dataclass(frozen=True)
class UnifiedCompanyResearchResult:
    lookup: CompanyProfileLookupResult
    revenue_evidence: tuple[RevenueEvidenceRow, ...]
    growth_ranges: tuple[ResearchRange, ...]
    current_assumptions: MultiStageDCFAssumptions
    period_reconciliation: tuple[str, ...]
    confidence_assessments: tuple[ConfidenceAssessment, ...]
    warnings: tuple[str, ...] = ()
    micron_period_alignment: MicronPeriodAlignment | None = None


@dataclass(frozen=True)
class _Spec:
    ticker: str
    company_name: str
    growth: tuple[float, float, float]
    mature_margin: float
    margin_range: tuple[float, float]
    starting_sc: float
    mature_sc: float
    sc_range: tuple[float, float]
    tax: float
    wacc: float
    terminal_growth: float
    model_risk: str
    context: BusinessContext
    evidence: tuple[ResearchEvidenceItem, ...]
    growth_rationale: tuple[str, str, str]
    margin_rationale: str
    starting_sc_rationale: str
    mature_sc_rationale: str
    limitation: str
    confidence: tuple[ConfidenceAssessment, ...]


def _e(evidence_id, label, value, unit, period, source, source_date, retrieved_at, *, category="company_specific_research", notes=""):
    return ResearchEvidenceItem(
        evidence_id, category, label, value, unit, period, source,
        source_date, retrieved_at, notes=notes,
    )


def _a(name, value, rationale, references):
    return ResearchAssumption(name, value, "research_in_progress", rationale, references)


def _mu_spec(retrieved_at: str) -> _Spec:
    alignment = build_micron_period_alignment(retrieved_at=retrieved_at)
    evidence = (
        _e("latest_quarter", "FQ3 2026 Revenue and growth", 41.456e9, "currency_amount", "quarter ended 2026-05-28", MU_Q3_2026, "2026-06-24", retrieved_at, category="historical_financial", notes="Revenue increased 346% YoY and 74% sequentially."),
        _e("q3_operating_margin", "FQ3 2026 GAAP Operating Margin", .804, "ratio", "quarter ended 2026-05-28", MU_Q3_2026, "2026-06-24", retrieved_at, category="historical_financial", notes="Peak-cycle evidence, not the mature assumption."),
        _e("q4_guidance", "FQ4 2026 Revenue guidance midpoint", 50e9, "currency_amount", "quarter ending 2026-08-27", MU_Q3_2026, "2026-06-24", retrieved_at, category="management_guidance", notes="Guidance range $49B-$51B; approximately 86% gross margin."),
        _e("fy2026_consensus", "FY2026 Revenue consensus", 129.39528e9, "currency_amount", "FY ending 2026-08-31", MU_FACTSET_CONSENSUS, "2026-08-23", retrieved_at, category="forward_consensus", notes="FactSet; 45 analysts."),
        _e("fy2027_consensus", "FY2027 Revenue consensus", 249.49006e9, "currency_amount", "FY ending 2027-08-31", MU_FACTSET_CONSENSUS, "2026-08-23", retrieved_at, category="forward_consensus", notes="FactSet; 45 analysts; 92.81% YoY."),
        _e("fy2028_consensus", "FY2028 Revenue consensus", 281.30581e9, "currency_amount", "FY ending 2028-08-31", MU_FACTSET_CONSENSUS, "2026-08-23", retrieved_at, category="forward_consensus", notes="FactSet; analyst count not displayed; 12.75% YoY."),
        _e("aligned_y1", "DCF-aligned Forward Year 1 Revenue", alignment.rolling_years[0].revenue, "currency_amount", alignment.rolling_years[0].period, "Phase 4.1 rolling-period alignment", "2026-08-23", retrieved_at, notes=alignment.interpolation_method),
        _e("aligned_y2", "DCF-aligned Forward Year 2 Revenue", alignment.rolling_years[1].revenue, "currency_amount", alignment.rolling_years[1].period, "Phase 4.1 rolling-period alignment", "2026-08-23", retrieved_at),
        _e("aligned_y3", "DCF-aligned Forward Year 3 Revenue", alignment.rolling_years[2].revenue, "currency_amount", alignment.rolling_years[2].period, "Phase 4.1 rolling-period alignment", "2026-08-23", retrieved_at, notes="Uses low-confidence FY2029 consensus only to align the rolling period."),
        _e("hbm4", "HBM4 production and qualification", "High-volume shipments for lead platform; samples shipped to multiple customers.", None, "2026", MU_Q3_2026, "2026-06-24", retrieved_at, category="management_guidance"),
        _e("hbm4e", "HBM4E roadmap", "Development underway on 1-gamma DRAM; volume production expected in calendar 2027.", None, "calendar 2027", MU_Q3_2026, "2026-06-24", retrieved_at, category="management_guidance"),
        _e("supply_visibility", "Memory supply visibility", "Demand exceeds supply; tight conditions expected beyond calendar 2027, with gradual supply improvement in 2028.", None, "calendar 2026-2028", MU_Q3_2026_REMARKS, "2026-06-24", retrieved_at, category="management_guidance"),
        _e("strategic_customer_agreements", "Strategic Customer Agreements", "16 multi-year take-or-pay agreements; binding volume commitments; most include fixed prices or price bands.", None, "multi-year through approximately 2030", MU_Q3_2026_10Q, "2026-06-24", retrieved_at, category="management_guidance", notes="14 agreements represent about $100B cumulative Revenue at contractual minimum prices; projected deposits and related commitments total $22B."),
        _e("sca_coverage", "SCA product coverage", "Committed DRAM, including HBM where appropriate, and NAND supply across data center, consumer and automotive customers.", None, "multi-year", MU_Q3_2026_REMARKS, "2026-06-24", retrieved_at, category="management_guidance"),
        _e("capex", "Nine-month FY2026 net PP&E investment", 16.613e9, "currency_amount", "nine months ended 2026-05-28", MU_Q3_2026_10Q, "2026-06-24", retrieved_at, category="historical_financial", notes="$19.602B PP&E expenditure less $2.989B government incentives."),
        _e("capacity_expansion", "Capacity expansion timing", "Tongluo existing fab expected to support meaningful shipments from mid-2027; Singapore HBM packaging and new nodes add supply while greenfield fabs remain long lead-time.", None, "2027-2028", MU_Q3_2026_10Q, "2026-06-24", retrieved_at, category="management_guidance"),
        _e("cycle_normalization", "Through-cycle memory economics", "HBM mix and supply discipline improve structure, but pricing, utilization and new capacity remain cyclical.", None, "mature period", "Research normalization from issuer cycle disclosures", "2026-08-23", retrieved_at),
        _e("terminal_macro", "Mature memory producer nominal growth", "Below broad nominal growth because terminal economics must include cycle normalization.", None, "terminal period", "Research framework", "2026-08-23", retrieved_at, category="industry_reference"),
    )
    context = BusinessContext(
        business_model_summary="Micron supplies DRAM, NAND and HBM memory across cloud, data-center, client, mobile, automotive and embedded markets.",
        primary_growth_drivers=("HBM", "AI data-center memory content", "DRAM pricing", "NAND pricing", "bit shipments"),
        primary_margin_drivers=("pricing", "utilization", "HBM mix", "node transitions", "supply discipline"),
        capital_intensity_notes=("Memory fabrication requires sustained PP&E investment and depreciation through the cycle.",),
        cyclicality_notes=("Micron remains cyclical, but AI/HBM structural demand and binding multi-year SCAs may delay and dampen the traditional cycle.", "Current FQ3/FQ4 economics are exceptional and are not extrapolated into Mature Margin."),
        major_profile_risks=("conventional DRAM/NAND pricing", "2027-2028 capacity additions", "HBM customer/platform concentration", "technology transitions", "AI infrastructure demand durability"),
    )
    confidence = (
        ConfidenceAssessment("Revenue Base", "High", "Validated company financial history supplies the base."),
        ConfidenceAssessment("Y1 Growth", "High", "Known Q4 guidance, Q1 consensus and fiscal totals support rolling-period alignment."),
        ConfidenceAssessment("Y2 Growth", "Medium", "FY2028 consensus, SCAs and supply constraints support positive growth, but quarterly interpolation remains derived."),
        ConfidenceAssessment("Y3 Growth", "Low", "Positive normalization is supported structurally, but FY2029 timing and pricing are highly uncertain."),
        ConfidenceAssessment("Mature Margin", "Low", "Through-cycle margin cannot be inferred from the current peak."),
        ConfidenceAssessment("Mature S/C", "Low", "Fab intensity and cycle utilization vary materially."),
        ConfidenceAssessment("WACC", "Medium", "Research risk premium reflects cycle volatility."),
        ConfidenceAssessment("Terminal Economics", "Low", "Long-run memory profitability remains cyclical."),
    )
    return _Spec(
        "MU", "Micron Technology, Inc.", (1.55, .20, .15), .28, (.18, .38),
        .45, .55, (.40, .70), .15, .105, .025, "High", context, evidence,
        (
            "155% conservatively rounds the 156.4% DCF-aligned forward-twelve-month growth; it does not reuse FY2026 as DCF Y1.",
            "20% rounds the 20.2% aligned rolling-year growth supported by FY2028 consensus and contracted demand visibility.",
            "15% is a lower positive normalization rate near the 15.4% aligned result; no current evidence supports forcing a negative year.",
        ),
        "28% is a normalized through-cycle operating margin, far below the current peak but above legacy cycles because HBM mix and supply discipline may improve structural economics.",
        "0.45x represents current capital intensity without treating peak Revenue as normalized efficiency.",
        "0.55x is a through-cycle economic efficiency anchor consistent with continued fab investment.",
        "Remaining memory cyclicality is moderated—not eliminated—by AI/HBM structural demand, multi-year take-or-pay customer agreements, constrained supply and higher-value mix; pricing, capacity and capital intensity remain material uncertainties.",
        confidence,
    )


def _aapl_spec(retrieved_at: str) -> _Spec:
    evidence = (
        _e("latest_quarter", "Fiscal Q3 2026 Revenue", 109.4e9, "currency_amount", "quarter ended 2026-06-27", AAPL_Q3_2026, "2026-07-30", retrieved_at, category="historical_financial", notes="Revenue increased 16% YoY; iPhone, Mac and Services reached June-quarter records."),
        _e("gross_margin", "Fiscal Q3 2026 Gross Margin", .501, "ratio", "quarter ended 2026-06-27", AAPL_Q3_2026, "2026-07-30", retrieved_at, category="historical_financial", notes="Included approximately 2pp favorable tariff-refund impact."),
        _e("installed_base", "Installed base", "All-time high across all major product categories and geographies.", None, "2026-06-27", AAPL_Q3_2026, "2026-07-30", retrieved_at, category="management_guidance"),
        _e("nine_month_revenue", "Nine-month FY2026 Revenue", 364.357e9, "currency_amount", "nine months ended 2026-06-27", AAPL_Q3_2026_10Q, "2026-07-31", retrieved_at, category="historical_financial"),
        _e("capital_efficiency", "Economic capital-efficiency framework", "Outsourced manufacturing and ecosystem monetization support high economic S/C; buybacks and cash management distort accounting invested capital.", None, "mature period", AAPL_Q3_2026_10Q, "2026-07-31", retrieved_at),
        _e("terminal_macro", "Mature nominal growth", "Large installed base and Services support growth modestly above mature device units, but scale limits the terminal rate.", None, "terminal period", "Research framework", "2026-08-23", retrieved_at, category="industry_reference"),
    )
    context = BusinessContext(
        business_model_summary="Apple combines premium devices with Services monetization across a large integrated installed base.",
        primary_growth_drivers=("iPhone replacement and mix", "Services", "installed base", "Mac/iPad", "wearables"),
        primary_margin_drivers=("Services mix", "hardware mix", "pricing", "component costs", "tariffs"),
        capital_intensity_notes=("Outsourced manufacturing produces asset-light economics; buybacks and cash management distort accounting invested capital.",),
        major_profile_risks=("device replacement cycles", "Services regulation", "China exposure", "AI execution", "mature scale"),
    )
    confidence = (
        ConfidenceAssessment("Revenue Base", "High", "Validated company statements provide the TTM base."),
        ConfidenceAssessment("Y1 Growth", "High", "Latest broad-based double-digit growth supports the near-term assumption."),
        ConfidenceAssessment("Y2 Growth", "Medium", "Services and installed-base monetization support growth, but device cycles matter."),
        ConfidenceAssessment("Y3 Growth", "Medium", "Scale implies normalization toward mature growth."),
        ConfidenceAssessment("Mature Margin", "Medium", "Services mix supports margin while hardware remains the majority of Revenue."),
        ConfidenceAssessment("Mature S/C", "Low", "Economic efficiency is clear but accounting invested capital is distorted."),
        ConfidenceAssessment("WACC", "Medium", "Large, diversified cash generation supports a lower risk assumption."),
        ConfidenceAssessment("Terminal Economics", "Low", "High economic S/C makes terminal ROIC sensitive to interpretation."),
    )
    return _Spec(
        "AAPL", "Apple Inc.", (.12, .08, .06), .32, (.29, .35),
        2.50, 1.80, (1.40, 2.20), .16, .085, .03, "Medium", context, evidence,
        (
            "12% is below the latest 16% quarter and avoids extrapolating tariff-refund and product-cycle effects.",
            "8% reflects continued Services and installed-base monetization with normalizing hardware comparisons.",
            "6% recognizes the scale of the Revenue base while retaining ecosystem growth.",
        ),
        "32% allows modest Services-mix support without turning a high gross-margin quarter into permanent operating leverage.",
        "2.50x is an explicit economic assumption reflecting outsourced manufacturing rather than book-equity arithmetic.",
        "1.80x remains asset-light but allows mature reinvestment needs and slower growth.",
        "Capital efficiency is an economic research assumption because accounting invested capital is distorted by capital returns.",
        confidence,
    )


def _avgo_spec(retrieved_at: str) -> _Spec:
    evidence = (
        _e("latest_quarter", "Fiscal Q2 2026 Revenue", 22.187e9, "currency_amount", "quarter ended 2026-05-03", AVGO_Q2_2026, "2026-06-03", retrieved_at, category="historical_financial", notes="Revenue increased 48% YoY."),
        _e("ai_revenue", "Q2 AI semiconductor Revenue", 10.8e9, "currency_amount", "quarter ended 2026-05-03", AVGO_Q2_2026, "2026-06-03", retrieved_at, category="management_guidance", notes="Increased 143% YoY; Q3 AI revenue expected around $16B."),
        _e("q3_guidance", "Fiscal Q3 2026 Revenue guidance", 29.4e9, "currency_amount", "quarter ending 2026-08-02", AVGO_Q2_2026, "2026-06-03", retrieved_at, category="management_guidance", notes="Approximately 84% YoY growth; non-GAAP operating income guidance approximately 67%."),
        _e("business_mix", "Q2 semiconductor / infrastructure software Revenue", "15.009B / 7.178B", None, "quarter ended 2026-05-03", AVGO_Q2_2026, "2026-06-03", retrieved_at, notes="Semiconductor grew 79%; infrastructure software grew 9%."),
        _e("debt_cash", "Q2 cash and gross debt context", "Cash 19.628B; debt remains material following VMware financing.", None, "2026-05-03", AVGO_Q2_2026_10Q, "2026-06-09", retrieved_at, category="historical_financial", notes="Equity bridge must use live net debt separately from operating assumptions."),
        _e("capital_efficiency", "Economic S/C normalization", "Software is asset-light, semiconductors use outsourced manufacturing, while goodwill and acquired intangibles distort accounting capital.", None, "mature period", AVGO_Q2_2026_10Q, "2026-06-09", retrieved_at),
        _e("terminal_macro", "Mature mixed-business growth", "AI semiconductors and infrastructure software normalize toward long-run nominal growth.", None, "terminal period", "Research framework", "2026-08-23", retrieved_at, category="industry_reference"),
    )
    context = BusinessContext(
        business_model_summary="Broadcom combines AI/custom semiconductor and networking franchises with VMware-led infrastructure software.",
        primary_growth_drivers=("custom AI accelerators", "AI networking", "VMware", "infrastructure software", "non-AI semiconductor cycle"),
        primary_margin_drivers=("software mix", "AI semiconductor mix", "VMware integration", "customer concentration", "R&D"),
        capital_intensity_notes=("Outsourced semiconductor manufacturing and software are asset-light, while acquisitions create large goodwill/intangible balances.",),
        major_profile_risks=("AI customer concentration", "VMware integration", "semiconductor cycle", "acquisition accounting", "gross debt"),
    )
    confidence = (
        ConfidenceAssessment("Revenue Base", "High", "Validated statements provide the starting base."),
        ConfidenceAssessment("Y1 Growth", "High", "Q3 guidance and AI revenue are issuer-provided."),
        ConfidenceAssessment("Y2 Growth", "Medium", "AI backlog supports growth but customer timing is concentrated."),
        ConfidenceAssessment("Y3 Growth", "Low", "AI and acquisition comparisons make duration uncertain."),
        ConfidenceAssessment("Mature Margin", "Low", "Long-run semiconductor/software mix and GAAP acquisition charges remain uncertain."),
        ConfidenceAssessment("Mature S/C", "Low", "Goodwill and software mix obscure organic economic capital."),
        ConfidenceAssessment("WACC", "Medium", "Strong cash generation is offset by leverage and concentration."),
        ConfidenceAssessment("Terminal Economics", "Low", "Mixed-business economics require a consolidated research bridge."),
    )
    return _Spec(
        "AVGO", "Broadcom Inc.", (.35, .22, .15), .46, (.40, .52),
        .65, .75, (.55, .95), .15, .095, .03, "High", context, evidence,
        (
            "35% recognizes Q3 guidance and AI growth while avoiding annualization of an 84% comparison.",
            "22% retains custom accelerator, networking and VMware momentum as acquisition comparisons normalize.",
            "15% begins normalization across AI and non-AI semiconductors without adding a separate acquisition architecture.",
        ),
        "46% is a consolidated GAAP-oriented mature margin reflecting high-margin software and semiconductors without copying non-GAAP guidance.",
        "0.65x is a normalized transition anchor because acquisition accounting obscures current invested capital.",
        "0.75x blends asset-light software and outsourced semiconductor economics while retaining acquisition and R&D capital needs.",
        "Acquisition and segment complexity are represented through researched consolidated assumptions.",
        confidence,
    )


def _build(
    ticker: str,
    current: MultiStageDCFAssumptions,
    history: FundamentalHistory,
    *,
    revenue_anchors: RevenueForecastAnchors | None = None,
    wacc_audit: WACCAuditResult | None = None,
    retrieved_at: str,
) -> UnifiedCompanyResearchResult:
    normalized = ticker.strip().upper()
    if normalized == "MU":
        spec = _mu_spec(retrieved_at)
        micron_alignment = build_micron_period_alignment(
            retrieved_at=retrieved_at
        )
    elif normalized == "AAPL":
        spec = _aapl_spec(retrieved_at)
    elif normalized == "AVGO":
        spec = _avgo_spec(retrieved_at)
    else:
        raise ValueError("unsupported_unified_company_profile")
    if normalized != "MU":
        micron_alignment = None

    annual_revenue = _latest_annual(history, REVENUE, "latest_annual_revenue", "Latest annual Revenue", "currency_amount")
    ttm_revenue = _ttm(history, REVENUE, "ttm_revenue", "Validated TTM Revenue", "currency_amount")
    annual_growth = _latest_annual(history, REVENUE_GROWTH, "latest_annual_growth", "Latest annual Revenue growth", "ratio")
    cagr = _anchor(history, "revenue_cagr_3y")
    annual_margin = _latest_annual(history, OPERATING_MARGIN, "latest_annual_operating_margin", "Latest annual Operating Margin", "ratio")
    ttm_margin = _ttm(history, OPERATING_MARGIN, "ttm_operating_margin", "Validated TTM Operating Margin", "ratio")
    latest_sc = _anchor(history, "latest_sales_to_capital")
    normalized_sc = _anchor(history, "sales_to_capital_3y")
    accounting_roic = _latest_annual(history, ROIC, "accounting_roic", "Accounting ROIC", "ratio")
    annual_tax = _latest_annual(history, OPERATING_TAX_RATE, "latest_operating_tax_rate", "Latest annual Operating Tax Rate", "ratio")
    forward = _forward_evidence(revenue_anchors)
    risk = _wacc_evidence(wacc_audit, None, None, retrieved_at)
    evidence = tuple(item for item in (
        annual_revenue, ttm_revenue, annual_growth, cagr, annual_margin,
        ttm_margin, latest_sc, normalized_sc, accounting_roic, annual_tax,
    ) if item is not None) + spec.evidence + forward + risk
    evidence_ids = {item.evidence_id for item in evidence}

    def refs(*candidates: str) -> tuple[str, ...]:
        return tuple(item for item in candidates if item in evidence_ids)

    start_margin_value = float(ttm_margin.value) if ttm_margin is not None else current.starting_operating_margin
    y1 = _a("year1_growth", spec.growth[0], spec.growth_rationale[0], refs("aligned_y1", "latest_quarter", "q3_guidance", "q4_guidance", "fy2027_consensus"))
    y2 = _a("year2_growth", spec.growth[1], spec.growth_rationale[1], refs("aligned_y2", "fy2028_consensus", "hbm4", "hbm4e", "strategic_customer_agreements", "installed_base", "ai_revenue", "business_mix"))
    y3 = _a("year3_growth", spec.growth[2], spec.growth_rationale[2], refs("aligned_y3", "cycle_normalization", "supply_visibility", "capacity_expansion", "installed_base", "business_mix"))
    fade = _a("revenue_fade_years", 8, "Eight deterministic fade years follow Y1-Y3; no production Y4/Y5 assumptions exist.", refs("terminal_macro"))
    terminal_g = _a("terminal_growth", spec.terminal_growth, "Mature nominal-growth anchor independent of market price.", refs("terminal_macro"))
    start_margin = _a("starting_operating_margin", start_margin_value, "Validated TTM margin where available; it is an accounting starting point, not Mature Margin.", refs("ttm_operating_margin", "q3_operating_margin", "gross_margin"))
    mature_margin = _a("mature_operating_margin", spec.mature_margin, spec.margin_rationale, refs("q3_operating_margin", "gross_margin", "business_mix", "cycle_normalization"))
    start_sc = _a("starting_sales_to_capital", spec.starting_sc, spec.starting_sc_rationale, refs("capital_efficiency", "capex"))
    mature_sc = _a("mature_sales_to_capital", spec.mature_sc, spec.mature_sc_rationale, refs("capital_efficiency", "capex"))
    tax = _a("operating_tax_rate", spec.tax, "Normalized operating-tax assumption, not a single-quarter effective rate.", refs("latest_operating_tax_rate"))
    wacc = _a("research_wacc", spec.wacc, "Research WACC remains separate from formula WACC and is not fitted to market price.", refs("formula_based_wacc", "historical_raw_beta"))
    horizon = _a("forecast_years", 11, "Three researched years plus eight deterministic fade years.", refs("terminal_macro"))
    terminal_roic = spec.mature_margin * (1 - spec.tax) * spec.mature_sc
    terminal_reinvestment = spec.terminal_growth / terminal_roic
    margin_range = _e("mature_margin_range", "Mature Margin research range", f"{spec.margin_range[0]:.2%}–{spec.margin_range[1]:.2%}", None, "mature period", "Research synthesis", "2026-08-23", retrieved_at)
    sc_range = _e("mature_sc_range", "Mature S/C research range", f"{spec.sc_range[0]:.2f}x–{spec.sc_range[1]:.2f}x", None, "mature period", "Research synthesis", "2026-08-23", retrieved_at)
    evidence += (margin_range, sc_range)
    profile = CompanyResearchProfile(
        ticker=spec.ticker, issuer_id=spec.ticker, company_name=spec.company_name,
        profile_status="research_in_progress", business_summary=spec.context.business_model_summary,
        business_context=spec.context,
        revenue_framework=RevenueResearchFramework(
            starting_revenue=ttm_revenue or annual_revenue,
            latest_annual_revenue=annual_revenue, ttm_revenue=ttm_revenue,
            latest_annual_growth=annual_growth, historical_3y_cagr=cagr,
            forward_revenue_anchors=revenue_anchors,
            year1_growth=y1, year2_growth=y2, year3_growth=y3,
            revenue_fade_years=fade, terminal_growth=terminal_g,
            near_term_growth_rationale="Y1-Y3 use issuer evidence and explicit research judgment; later years fade mechanically.",
            fade_rationale=fade.rationale, terminal_growth_rationale=terminal_g.rationale,
        ),
        margin_framework=MarginResearchFramework(
            annual_margin, ttm_margin,
            _annual_items(history, OPERATING_MARGIN, prefix="operating_margin", label="Operating Margin", unit="ratio"),
            _annual_items(history, GROSS_MARGIN, prefix="gross_margin", label="Gross Margin", unit="ratio"),
            start_margin, mature_margin, start_margin.rationale, mature_margin.rationale,
        ),
        capital_efficiency_framework=CapitalEfficiencyResearchFramework(
            latest_sc, normalized_sc, accounting_roic, None, None,
            start_sc, mature_sc, start_margin_value * (1-spec.tax) * spec.starting_sc,
            terminal_roic, start_sc.rationale, mature_sc.rationale,
            ("economic_sales_to_capital_is_research_assumption",),
        ),
        wacc_framework=WACCResearchFramework(
            wacc_audit=wacc_audit, research_wacc=wacc, rationale=wacc.rationale,
            warnings=("formula_wacc_separate_from_research_wacc",),
        ),
        terminal_framework=TerminalResearchFramework(
            terminal_g, mature_margin, mature_sc, terminal_roic,
            terminal_reinvestment, 1-terminal_reinvestment,
            terminal_g.rationale, mature_margin.rationale, mature_sc.rationale,
            ("mature_margin_range", "mature_sc_range", "terminal_macro"),
            ("terminal_economics_are_research_assumptions",),
        ),
        operating_tax_rate=tax, forecast_years=horizon,
        rationale=f"Unreviewed {spec.company_name} unified-production Research Candidate; market price is excluded.",
        warnings=("research_candidate_not_reviewed", "candidate_not_applied_to_live_dcf", "standard_sales_to_capital_production"),
        evidence_items=evidence,
        uncertainty_notes=spec.context.major_profile_risks + (spec.limitation,),
        future_scenario_drivers=("Y3 growth", "mature margin", "mature Sales-to-Capital", "Research WACC"),
        model_risk=spec.model_risk,
    )
    revenue_rows = []
    for item in _annual_items(history, REVENUE, prefix="annual_revenue", label="Annual Revenue", unit="currency_amount")[-4:]:
        revenue_rows.append(RevenueEvidenceRow(f"FY ended {item.period}", item.period, float(item.value), None, item.source, item.period, retrieved_at))
    if ttm_revenue is not None:
        revenue_rows.append(RevenueEvidenceRow("Current validated TTM", ttm_revenue.period, float(ttm_revenue.value), None, ttm_revenue.source, ttm_revenue.period, retrieved_at, notes="DCF starting base."))
    if micron_alignment is not None:
        conservative, central, high = micron_alignment.growth_cases
        ranges = tuple(
            ResearchRange(
                f"year{index}_growth",
                conservative.growth[index - 1], central.growth[index - 1],
                high.growth[index - 1], assumption.rationale,
                assumption.evidence_references,
            )
            for index, assumption in enumerate((y1, y2, y3), 1)
        )
    else:
        ranges = tuple(
            ResearchRange(f"year{index}_growth", value-.03, value, value+.03, rationale, assumption.evidence_references)
            for index, (value, rationale, assumption) in enumerate(zip(spec.growth, spec.growth_rationale, (y1, y2, y3)), 1)
        )
    return UnifiedCompanyResearchResult(
        CompanyProfileLookupResult(profile, True, None), tuple(revenue_rows),
        ranges, current,
        (
            "Validated TTM is preferred as the DCF Revenue base; latest annual is evidence context only.",
            "Y4/Y5 are deterministic fade outputs and are not production research fields.",
            spec.limitation,
        ),
        spec.confidence,
        ("market_price_excluded_from_candidate_generation", "unified_standard_sc_production"),
        micron_alignment,
    )


def build_micron_research_profile(current_assumptions, history, *, revenue_anchors=None, wacc_audit=None, retrieved_at="2026-08-23"):
    return _build("MU", current_assumptions, history, revenue_anchors=revenue_anchors, wacc_audit=wacc_audit, retrieved_at=retrieved_at)


def build_apple_research_profile(current_assumptions, history, *, revenue_anchors=None, wacc_audit=None, retrieved_at="2026-08-23"):
    return _build("AAPL", current_assumptions, history, revenue_anchors=revenue_anchors, wacc_audit=wacc_audit, retrieved_at=retrieved_at)


def build_broadcom_research_profile(current_assumptions, history, *, revenue_anchors=None, wacc_audit=None, retrieved_at="2026-08-23"):
    return _build("AVGO", current_assumptions, history, revenue_anchors=revenue_anchors, wacc_audit=wacc_audit, retrieved_at=retrieved_at)
